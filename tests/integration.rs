//! Integration tests for Triglav client-server communication.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use triglav::crypto::{KeyPair, NoiseSession};
use triglav::error::Result;
use triglav::multipath::{MultipathManager, MultipathConfig, UplinkConfig};
use triglav::protocol::{Packet, PacketType, PacketFlags, HEADER_SIZE};
use triglav::proxy::{Socks5Server, Socks5Config, HttpProxyServer, HttpProxyConfig};
use triglav::transport::TransportProtocol;
use triglav::types::{SessionId, SequenceNumber, UplinkId};

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::net::UdpSocket;
use tokio::sync::broadcast;

/// Simple test server that echoes data back.
struct TestServer {
    keypair: KeyPair,
    socket: Arc<UdpSocket>,
    sessions: DashMap<SessionId, TestSession>,
    shutdown_tx: broadcast::Sender<()>,
    next_seq: AtomicU64,
}

struct TestSession {
    noise: RwLock<Option<NoiseSession>>,
}

impl TestServer {
    async fn new(addr: SocketAddr) -> Result<Self> {
        let socket = UdpSocket::bind(addr).await?;
        let (shutdown_tx, _) = broadcast::channel(1);
        Ok(Self {
            keypair: KeyPair::generate(),
            socket: Arc::new(socket),
            sessions: DashMap::new(),
            shutdown_tx,
            next_seq: AtomicU64::new(1),
        })
    }

    fn next_sequence(&self) -> SequenceNumber {
        SequenceNumber(self.next_seq.fetch_add(1, Ordering::SeqCst))
    }

    fn public_key(&self) -> &triglav::crypto::PublicKey {
        &self.keypair.public
    }

    fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    async fn run(&self) -> Result<()> {
        let mut buf = vec![0u8; 65536];
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                result = self.socket.recv_from(&mut buf) => {
                    let (len, addr) = result?;
                    if let Err(e) = self.handle_packet(&buf[..len], addr).await {
                        eprintln!("Server error: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
        Ok(())
    }

    async fn handle_packet(&self, data: &[u8], addr: SocketAddr) -> Result<()> {
        if data.len() < HEADER_SIZE {
            return Ok(());
        }

        let packet = Packet::decode(data)?;
        let session_id = packet.header.session_id;

        // Get or create session
        let session = self.sessions.entry(session_id).or_insert_with(|| {
            TestSession {
                noise: RwLock::new(None),
            }
        });

        match packet.header.packet_type {
            PacketType::Handshake => {
                // Create responder
                let mut noise = NoiseSession::new_responder(&self.keypair.secret)?;
                let _ = noise.read_handshake(&packet.payload)?;
                let response = noise.write_handshake(&[])?;

                let response_packet = Packet::new(
                    PacketType::Handshake,
                    packet.header.sequence.next(),
                    session_id,
                    packet.header.uplink_id,
                    response,
                )?;

                self.socket.send_to(&response_packet.encode()?, addr).await?;
                *session.noise.write() = Some(noise);
            }
            PacketType::Data => {
                // Decrypt if encrypted
                let payload = if packet.header.flags.has(PacketFlags::ENCRYPTED) {
                    if let Some(ref mut noise) = *session.noise.write() {
                        if noise.is_transport() {
                            noise.decrypt(&packet.payload)?
                        } else {
                            packet.payload.clone()
                        }
                    } else {
                        packet.payload.clone()
                    }
                } else {
                    packet.payload.clone()
                };

                // Echo back (encrypted if we have session)
                let (response_payload, encrypted) = if let Some(ref mut noise) = *session.noise.write() {
                    if noise.is_transport() {
                        (noise.encrypt(&payload)?, true)
                    } else {
                        (payload.clone(), false)
                    }
                } else {
                    (payload.clone(), false)
                };

                let mut response_packet = Packet::data(
                    self.next_sequence(),
                    session_id,
                    packet.header.uplink_id,
                    response_payload,
                )?;

                if encrypted {
                    response_packet.set_flag(PacketFlags::ENCRYPTED);
                }

                self.socket.send_to(&response_packet.encode()?, addr).await?;
            }
            PacketType::Ping => {
                let pong = Packet::pong(
                    packet.header.sequence.next(),
                    session_id,
                    packet.header.uplink_id,
                    packet.header.timestamp,
                )?;
                self.socket.send_to(&pong.encode()?, addr).await?;
            }
            _ => {}
        }

        Ok(())
    }

    fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

#[tokio::test]
async fn test_handshake_and_echo() {
    // Start server on random port
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    // Run server in background
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    let server_task = tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();

    let manager = MultipathManager::new(config, client_keypair);

    // Add uplink
    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        interface: None,
        local_addr: None,
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink_config).unwrap();

    // Connect to server
    manager.connect(server_public.clone()).await.unwrap();

    // Wait for connection
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send test data
    let test_data = b"Hello, Triglav!";
    manager.send(test_data).await.unwrap();

    // Receive response with timeout
    let result = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;

    match result {
        Ok(Ok((data, _uplink_id))) => {
            assert_eq!(data, test_data.to_vec(), "Echoed data should match");
            println!("Test passed: received {} bytes", data.len());
        }
        Ok(Err(e)) => {
            panic!("Receive error: {}", e);
        }
        Err(_) => {
            panic!("Receive timed out");
        }
    }

    // Cleanup
    server.shutdown();
    let _ = tokio::time::timeout(Duration::from_millis(100), server_task).await;
}

#[tokio::test]
async fn test_multiple_messages() {
    // Start server
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();

    let manager = MultipathManager::new(config, client_keypair);

    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send multiple messages
    for i in 0..5 {
        let msg = format!("Message {}", i);
        manager.send(msg.as_bytes()).await.unwrap();

        let result = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;
        match result {
            Ok(Ok((data, _))) => {
                assert_eq!(String::from_utf8_lossy(&data), msg);
            }
            Ok(Err(e)) => panic!("Receive error on message {}: {}", i, e),
            Err(_) => panic!("Timeout on message {}", i),
        }
    }

    server.shutdown();
}

#[tokio::test]
async fn test_large_payload() {
    // Start server
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();

    let manager = MultipathManager::new(config, client_keypair);

    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send larger payload (within MTU)
    let large_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    manager.send(&large_data).await.unwrap();

    let result = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;
    match result {
        Ok(Ok((data, _))) => {
            assert_eq!(data, large_data, "Large payload should match");
            println!("Large payload test passed: {} bytes", data.len());
        }
        Ok(Err(e)) => panic!("Receive error: {}", e),
        Err(_) => panic!("Timeout"),
    }

    server.shutdown();
}

#[tokio::test]
async fn test_socks5_proxy_startup() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    // Start backend server
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create multipath manager and connect
    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink_config).unwrap();
    manager.connect(server_public).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Start SOCKS5 proxy on random port
    let socks_config = Socks5Config {
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        allow_no_auth: true,
        username: None,
        password: None,
        connect_timeout_secs: 30,
        max_connections: 100,
    };

    // We need to get the actual port - start a listener first
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let socks_addr = listener.local_addr().unwrap();
    drop(listener);

    let socks_config = Socks5Config {
        listen_addr: socks_addr,
        ..socks_config
    };

    let socks_server = Socks5Server::new(socks_config, Arc::clone(&manager));
    tokio::spawn(async move {
        let _ = socks_server.run().await;
    });

    // Wait for proxy to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect to SOCKS5 proxy and perform handshake
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        TcpStream::connect(socks_addr)
    ).await;

    match result {
        Ok(Ok(mut stream)) => {
            // SOCKS5 greeting: version=5, nmethods=1, method=0 (no auth)
            stream.write_all(&[0x05, 0x01, 0x00]).await.unwrap();

            let mut response = [0u8; 2];
            stream.read_exact(&mut response).await.unwrap();

            // Should respond with version=5, method=0 (no auth accepted)
            assert_eq!(response[0], 0x05, "SOCKS version should be 5");
            assert_eq!(response[1], 0x00, "Should accept no auth method");

            // Now send CONNECT request for 127.0.0.1:80 (IPv4)
            // Format: VER=5, CMD=1 (CONNECT), RSV=0, ATYP=1 (IPv4), ADDR (4 bytes), PORT (2 bytes)
            let connect_request = [
                0x05, // version
                0x01, // CONNECT
                0x00, // reserved
                0x01, // IPv4 address type
                127, 0, 0, 1, // 127.0.0.1
                0x00, 0x50, // port 80
            ];
            stream.write_all(&connect_request).await.unwrap();

            // Read CONNECT reply (at least 10 bytes for IPv4)
            let mut connect_reply = [0u8; 10];
            stream.read_exact(&mut connect_reply).await.unwrap();

            assert_eq!(connect_reply[0], 0x05, "Reply version should be 5");
            assert_eq!(connect_reply[1], 0x00, "Reply should indicate success");

            // Now we're in tunnel mode - first we'll receive the echoed CONNECT request
            // from the echo server, then our actual test data

            // Read the echoed CONNECT request first
            let mut connect_echo = vec![0u8; 100];
            let read_result = tokio::time::timeout(
                Duration::from_secs(2),
                stream.read(&mut connect_echo)
            ).await;

            match read_result {
                Ok(Ok(n)) if n > 0 => {
                    let received = String::from_utf8_lossy(&connect_echo[..n]);
                    assert!(received.contains("CONNECT"),
                        "Should receive echoed CONNECT request, got: {}", received);
                    println!("SOCKS5 received CONNECT echo: {:?}", received);
                }
                Ok(Ok(_)) => panic!("Connection closed before receiving CONNECT echo"),
                Ok(Err(e)) => panic!("Read error: {}", e),
                Err(_) => panic!("Timeout waiting for CONNECT echo"),
            }

            // Now send actual test data and verify it echoes back
            let test_data = b"Hello through SOCKS5 tunnel!";
            stream.write_all(test_data).await.unwrap();
            stream.flush().await.unwrap();

            // Read echoed test data
            let mut echo_buf = vec![0u8; test_data.len() + 50];
            let read_result = tokio::time::timeout(
                Duration::from_secs(2),
                stream.read(&mut echo_buf)
            ).await;

            match read_result {
                Ok(Ok(n)) if n > 0 => {
                    let received = &echo_buf[..n];
                    println!("SOCKS5 received {} bytes: {:?}", n, String::from_utf8_lossy(received));
                    assert_eq!(received, test_data,
                        "Echoed data should match test data");
                    println!("SOCKS5 proxy end-to-end data forwarding verified!");
                }
                Ok(Ok(_)) => panic!("Connection closed before receiving test data echo"),
                Ok(Err(e)) => panic!("Read error waiting for test data echo: {}", e),
                Err(_) => panic!("Timeout waiting for test data echo"),
            }
        }
        Ok(Err(e)) => panic!("Failed to connect to SOCKS5 proxy: {}", e),
        Err(_) => panic!("Connection to SOCKS5 proxy timed out"),
    }

    server.shutdown();
}

#[tokio::test]
async fn test_http_proxy_connect() {
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpStream;

    // Start backend server
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create multipath manager and connect
    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink_config).unwrap();
    manager.connect(server_public).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Start HTTP proxy on random port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = listener.local_addr().unwrap();
    drop(listener);

    let http_config = HttpProxyConfig {
        listen_addr: http_addr,
        connect_timeout_secs: 30,
        max_connections: 100,
    };

    let http_server = HttpProxyServer::new(http_config, Arc::clone(&manager));
    tokio::spawn(async move {
        let _ = http_server.run().await;
    });

    // Wait for proxy to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect to HTTP proxy
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        TcpStream::connect(http_addr)
    ).await;

    match result {
        Ok(Ok(stream)) => {
            let (reader, mut writer) = stream.into_split();
            let mut reader = BufReader::new(reader);

            // Send HTTP CONNECT request
            writer.write_all(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n").await.unwrap();
            writer.flush().await.unwrap();

            // Read response line
            let mut response_line = String::new();
            let read_result = tokio::time::timeout(
                Duration::from_secs(2),
                reader.read_line(&mut response_line)
            ).await;

            match read_result {
                Ok(Ok(_)) => {
                    assert!(response_line.contains("200"),
                        "Should get 200 response, got: {}", response_line);
                    println!("HTTP proxy CONNECT response: {}", response_line.trim());

                    // Read remaining headers until empty line
                    loop {
                        let mut line = String::new();
                        reader.read_line(&mut line).await.unwrap();
                        if line.trim().is_empty() {
                            break;
                        }
                    }

                    // Now we're in tunnel mode - send data
                    let test_data = b"Hello through HTTP tunnel!";
                    writer.write_all(test_data).await.unwrap();
                    writer.flush().await.unwrap();

                    // Read echoed data (first the CONNECT request echo, then our data)
                    let mut echo_buf = vec![0u8; 200];
                    let read_result = tokio::time::timeout(
                        Duration::from_secs(2),
                        reader.read(&mut echo_buf)
                    ).await;

                    match read_result {
                        Ok(Ok(n)) if n > 0 => {
                            let received = String::from_utf8_lossy(&echo_buf[..n]);
                            println!("HTTP proxy received: {:?}", received);
                            assert!(received.contains("CONNECT"),
                                "Should receive echoed CONNECT request");
                            println!("HTTP proxy end-to-end test passed!");
                        }
                        Ok(Ok(_)) => panic!("Connection closed before receiving echo"),
                        Ok(Err(e)) => panic!("Read error: {}", e),
                        Err(_) => panic!("Timeout waiting for echo"),
                    }
                }
                Ok(Err(e)) => panic!("Failed to read response: {}", e),
                Err(_) => panic!("Timeout waiting for CONNECT response"),
            }
        }
        Ok(Err(e)) => panic!("Failed to connect to HTTP proxy: {}", e),
        Err(_) => panic!("Connection to HTTP proxy timed out"),
    }

    server.shutdown();
}

// Dublin Traceroute Integration Tests
// These tests verify the flow binding, NAT detection, and path discovery functionality.

#[tokio::test]
async fn test_flow_binding_consistency() {
    //! Verify that packets sent with the same flow_id always use the same uplink.
    //! This is the core Dublin Traceroute ECMP consistency requirement.

    // Start server
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with ECMP awareness enabled
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;

    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add single uplink for this test
    let uplink_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink_id = manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate a flow
    let flow_id = manager.allocate_flow();
    assert!(flow_id > 0, "Flow ID should be non-zero");

    // Send multiple messages on the same flow
    for i in 0..5 {
        let msg = format!("Flow message {}", i);
        manager.send_on_flow(Some(flow_id), msg.as_bytes()).await.unwrap();

        // Verify flow binding is consistent
        let bound_uplink = manager.get_flow_binding(flow_id);
        assert_eq!(bound_uplink, Some(uplink_id),
            "Flow should remain bound to same uplink on message {}", i);
    }

    // Verify flow count
    assert_eq!(manager.active_flow_count(), 1, "Should have exactly 1 active flow");

    // Release the flow
    manager.release_flow(flow_id);
    assert_eq!(manager.active_flow_count(), 0, "Flow count should be 0 after release");

    server.shutdown();
}

#[tokio::test]
async fn test_multiple_flows_different_bindings() {
    //! Verify that different flows can be explicitly bound to different uplinks.
    //! Uses two separate ports on the same server to simulate multiple uplinks.

    // Start server
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with ECMP awareness enabled
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;

    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks pointing to the same server (simulating multiple network paths)
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    // Connect using server's public key
    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flows explicitly on different uplinks
    let flow1 = manager.allocate_flow_on_uplink(uplink1_id);
    let flow2 = manager.allocate_flow_on_uplink(uplink2_id);

    assert!(flow1.is_some(), "Should be able to allocate flow on uplink 1");
    assert!(flow2.is_some(), "Should be able to allocate flow on uplink 2");

    let flow1_id = flow1.unwrap();
    let flow2_id = flow2.unwrap();

    // Verify bindings
    assert_eq!(manager.get_flow_binding(flow1_id), Some(uplink1_id));
    assert_eq!(manager.get_flow_binding(flow2_id), Some(uplink2_id));

    // Verify different flows have different IDs
    assert_ne!(flow1_id, flow2_id, "Different flows should have different IDs");

    // Verify active flow count
    assert_eq!(manager.active_flow_count(), 2);

    server.shutdown();
}

#[tokio::test]
async fn test_nat_state_detection() {
    //! Verify that NAT state is detected during connection and influences routing.

    use triglav::multipath::NatType;

    // Start server
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with ECMP awareness enabled
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;

    let manager = MultipathManager::new(config, client_keypair);

    // Add uplink
    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink_id = manager.add_uplink(uplink_config).unwrap();

    // Connect to server
    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Check NAT summary - since we're on loopback, NAT detection will see private address
    let nat_summary = manager.nat_summary();
    assert!(!nat_summary.is_empty(), "Should have NAT summary for uplinks");

    // Simulate external NAT detection (as if from STUN)
    manager.set_uplink_nat_state(uplink_id, NatType::FullCone, Some("203.0.113.1:12345".parse().unwrap()));

    // Verify NAT state was updated
    let updated_summary = manager.nat_summary();
    let (_, nat_type, is_natted) = updated_summary.iter()
        .find(|(id, _, _)| id.as_str() == "test-uplink")
        .expect("Should find test-uplink");

    assert_eq!(*nat_type, NatType::FullCone, "NAT type should be FullCone");
    assert!(*is_natted, "Should be detected as NATted");

    server.shutdown();
}

#[tokio::test]
async fn test_nat_aware_uplink_selection() {
    //! Verify that NAT-aware selection prefers non-NATted uplinks.

    use triglav::multipath::NatType;

    // Start server
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;

    let manager = MultipathManager::new(config, client_keypair);

    // Add uplink
    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink_id = manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test NAT-aware selection with no NAT
    manager.set_uplink_nat_state(uplink_id, NatType::None, None);

    let selected = manager.select_nat_aware(None);
    assert_eq!(selected, Some(uplink_id), "Should select non-NATted uplink");

    // Get non-NATted uplinks list
    let non_natted = manager.non_natted_uplinks();
    assert_eq!(non_natted.len(), 1, "Should have 1 non-NATted uplink");

    // Now mark as symmetric NAT (worst case)
    manager.set_uplink_nat_state(uplink_id, NatType::Symmetric, Some("203.0.113.1:12345".parse().unwrap()));

    // Non-NATted list should be empty now
    let non_natted = manager.non_natted_uplinks();
    assert!(non_natted.is_empty(), "Non-NATted list should be empty after marking as Symmetric NAT");

    // But NAT-aware selection should still return the uplink (as fallback)
    let selected = manager.select_nat_aware(None);
    assert_eq!(selected, Some(uplink_id), "Should still select uplink as fallback");

    server.shutdown();
}

#[tokio::test]
async fn test_path_discovery_event() {
    //! Verify that path discovery events are emitted when enabled.

    use triglav::multipath::{MultipathEvent};

    // Start server
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with ECMP awareness and short path discovery interval
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    config.path_discovery_interval = Duration::from_millis(100);

    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Subscribe to events before connecting
    let mut event_rx = manager.subscribe();

    // Add uplink
    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Start maintenance loop which includes path discovery
    manager.start_maintenance_loop();

    // Wait for events with timeout
    let mut found_path_discovery = false;
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(2);

    while start.elapsed() < timeout {
        match tokio::time::timeout(Duration::from_millis(100), event_rx.recv()).await {
            Ok(Ok(MultipathEvent::PathDiscoveryComplete { destination, paths_found, diversity_score })) => {
                println!("Path discovery complete: destination={}, paths={}, diversity={}",
                    destination, paths_found, diversity_score);
                found_path_discovery = true;
                break;
            }
            Ok(Ok(event)) => {
                println!("Received event: {:?}", event);
            }
            Ok(Err(_)) => break, // Channel closed
            Err(_) => continue, // Timeout, try again
        }
    }

    assert!(found_path_discovery, "Should receive PathDiscoveryComplete event");

    // Verify path discovery state via the manager
    let path_discovery = manager.path_discovery();
    let diversity = path_discovery.get_diversity(server_addr);
    println!("Path diversity: unique_paths={}, score={}",
        diversity.unique_paths, diversity.diversity_score);

    server.shutdown();
}

#[tokio::test]
async fn test_flow_hash_calculation() {
    //! Verify flow hash calculation produces consistent, non-zero results.

    use std::net::{IpAddr, Ipv4Addr};
    use triglav::multipath::FlowId;

    // Create flow ID
    let flow = FlowId::new(
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
        IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
        12345,
        53,
        17, // UDP
    );

    // Hash should be consistent
    let hash1 = flow.flow_hash();
    let hash2 = flow.flow_hash();
    assert_eq!(hash1, hash2, "Flow hash should be consistent");

    // Hash should never be zero
    assert_ne!(hash1, 0, "Flow hash should never be zero");

    // Different ports should produce different hashes (usually)
    let flow2 = flow.with_src_port(12346);
    let hash3 = flow2.flow_hash();
    assert_ne!(hash1, hash3, "Different ports should produce different hashes");

    // TCP and UDP should produce different hashes
    let flow_tcp = FlowId::new(
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
        IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
        12345,
        80,
        6, // TCP
    );
    let flow_udp = FlowId::new(
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
        IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
        12345,
        80,
        17, // UDP
    );
    assert_ne!(flow_tcp.flow_hash(), flow_udp.flow_hash(),
        "TCP and UDP flows should have different hashes");
}

#[tokio::test]
async fn test_stale_flow_cleanup() {
    //! Verify that stale flows are cleaned up when their uplink becomes unusable.

    // Start server
    let server = TestServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;

    let manager = MultipathManager::new(config, client_keypair);

    // Add uplink
    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink_id = manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate a flow bound to this uplink
    let flow_id = manager.allocate_flow_on_uplink(uplink_id).unwrap();
    assert_eq!(manager.active_flow_count(), 1);

    // Remove the uplink
    manager.remove_uplink(uplink_id);

    // Run cleanup
    manager.cleanup_stale_flows();

    // Flow should be cleaned up since its uplink is gone
    assert_eq!(manager.active_flow_count(), 0, "Stale flow should be cleaned up");
    assert!(manager.get_flow_binding(flow_id).is_none(), "Flow binding should be removed");

    server.shutdown();
}
