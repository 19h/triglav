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
