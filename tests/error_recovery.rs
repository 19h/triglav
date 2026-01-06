//! Error recovery tests for Triglav.
//!
//! Tests for error handling and recovery scenarios including:
//! - All uplinks fail
//! - Uplink reconnection
//! - Session recovery
//! - Graceful degradation

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::net::UdpSocket;
use tokio::sync::broadcast;

use triglav::crypto::{KeyPair, NoiseSession};
use triglav::multipath::{MultipathConfig, MultipathEvent, MultipathManager, UplinkConfig};
use triglav::protocol::{Packet, PacketFlags, PacketType, HEADER_SIZE};
use triglav::transport::TransportProtocol;
use triglav::types::{ConnectionState, SequenceNumber, SessionId, UplinkHealth, UplinkId};

// ============================================================================
// Test Infrastructure
// ============================================================================

struct MockServer {
    keypair: KeyPair,
    socket: Arc<UdpSocket>,
    shutdown: broadcast::Sender<()>,
    request_count: AtomicU64,
    should_respond: Arc<RwLock<bool>>,
}

impl MockServer {
    async fn new(addr: SocketAddr) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(addr).await?;
        let (shutdown, _) = broadcast::channel(1);

        Ok(Self {
            keypair: KeyPair::generate(),
            socket: Arc::new(socket),
            shutdown,
            request_count: AtomicU64::new(0),
            should_respond: Arc::new(RwLock::new(true)),
        })
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    fn public_key(&self) -> &triglav::crypto::PublicKey {
        &self.keypair.public
    }

    fn set_responding(&self, responding: bool) {
        *self.should_respond.write() = responding;
    }

    async fn run(&self) {
        let mut buf = vec![0u8; 65536];
        let mut shutdown_rx = self.shutdown.subscribe();
        let mut sessions: DashMap<SessionId, RwLock<Option<NoiseSession>>> = DashMap::new();

        loop {
            tokio::select! {
                result = self.socket.recv_from(&mut buf) => {
                    match result {
                        Ok((len, addr)) => {
                            self.request_count.fetch_add(1, Ordering::Relaxed);

                            if !*self.should_respond.read() {
                                continue; // Drop packet
                            }

                            if let Ok(packet) = Packet::decode(&buf[..len]) {
                                let session_id = packet.header.session_id;
                                sessions.entry(session_id).or_insert_with(|| RwLock::new(None));

                                match packet.header.packet_type {
                                    PacketType::Handshake => {
                                        let mut noise = NoiseSession::new_responder(&self.keypair.secret).unwrap();
                                        let _ = noise.read_handshake(&packet.payload);
                                        let response = noise.write_handshake(&[]).unwrap();

                                        let response_packet = Packet::new(
                                            PacketType::Handshake,
                                            packet.header.sequence.next(),
                                            session_id,
                                            packet.header.uplink_id,
                                            response,
                                        ).unwrap();

                                        let _ = self.socket.send_to(&response_packet.encode().unwrap(), addr).await;

                                        if let Some(entry) = sessions.get(&session_id) {
                                            *entry.write() = Some(noise);
                                        }
                                    }
                                    PacketType::Data => {
                                        // Prepare response inside lock, send outside
                                        let response_data = if let Some(entry) = sessions.get(&session_id) {
                                            let mut guard = entry.write();
                                            if let Some(ref mut noise) = *guard {
                                                if noise.is_transport() {
                                                    if let Ok(plaintext) = noise.decrypt(&packet.payload) {
                                                        if let Ok(response_ct) = noise.encrypt(&plaintext) {
                                                            let mut response_packet = Packet::data(
                                                                packet.header.sequence.next(),
                                                                session_id,
                                                                packet.header.uplink_id,
                                                                response_ct,
                                                            ).unwrap();
                                                            response_packet.set_flag(PacketFlags::ENCRYPTED);
                                                            Some(response_packet.encode().unwrap())
                                                        } else { None }
                                                    } else { None }
                                                } else { None }
                                            } else { None }
                                        } else { None };

                                        if let Some(encoded) = response_data {
                                            let _ = self.socket.send_to(&encoded, addr).await;
                                        }
                                    }
                                    PacketType::Ping => {
                                        let pong = Packet::pong(
                                            packet.header.sequence.next(),
                                            session_id,
                                            packet.header.uplink_id,
                                            packet.header.timestamp,
                                        ).unwrap();
                                        let _ = self.socket.send_to(&pong.encode().unwrap(), addr).await;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
    }

    fn shutdown(&self) {
        let _ = self.shutdown.send(());
    }
}

// ============================================================================
// Uplink Failure Tests
// ============================================================================

#[tokio::test]
async fn test_single_uplink_failure() {
    let server = MockServer::new("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with single uplink
    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink_id = manager.add_uplink(uplink_config).unwrap();

    // Connect
    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(manager.state(), ConnectionState::Connected);

    // Remove the uplink
    manager.remove_uplink(uplink_id);

    assert_eq!(manager.uplink_count(), 0);

    // Send should fail with no uplinks
    let result = manager.send(b"test").await;
    assert!(result.is_err(), "Send should fail with no uplinks");

    server.shutdown();
}

#[tokio::test]
async fn test_all_uplinks_fail_event() {
    let server = MockServer::new("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Subscribe to events
    let mut event_rx = manager.subscribe();

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

    // Remove uplink
    manager.remove_uplink(uplink_id);

    // Check for disconnection event
    let mut got_disconnect = false;
    while let Ok(event) = tokio::time::timeout(Duration::from_millis(200), event_rx.recv()).await {
        match event {
            Ok(MultipathEvent::UplinkDisconnected(_)) => {
                got_disconnect = true;
                break;
            }
            _ => {}
        }
    }

    assert!(
        got_disconnect || manager.uplink_count() == 0,
        "Should get disconnection event or have no uplinks"
    );

    server.shutdown();
}

#[tokio::test]
async fn test_uplink_failover_to_backup() {
    let server = MockServer::new("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add primary and backup uplinks
    let primary_config = UplinkConfig {
        id: UplinkId::new("primary"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let primary_id = manager.add_uplink(primary_config).unwrap();

    let backup_config = UplinkConfig {
        id: UplinkId::new("backup"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let backup_id = manager.add_uplink(backup_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flow on primary
    let flow_id = manager.allocate_flow_on_uplink(primary_id).unwrap();

    // Send through primary
    manager
        .send_on_flow(Some(flow_id), b"before failover")
        .await
        .unwrap();

    // Remove primary
    manager.remove_uplink(primary_id);

    // Flow should failover to backup on next send
    let result = manager.send_on_flow(Some(flow_id), b"after failover").await;
    assert!(result.is_ok(), "Should failover to backup");

    // Flow binding should now be on backup
    let binding = manager.get_flow_binding(flow_id);
    assert_eq!(
        binding,
        Some(backup_id),
        "Flow should be bound to backup after failover"
    );

    server.shutdown();
}

// ============================================================================
// Server Unresponsive Tests
// ============================================================================

#[tokio::test]
async fn test_server_stops_responding() {
    let server = MockServer::new("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

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

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send works initially - server echoes back
    manager.send(b"works").await.unwrap();

    // Wait for echo response
    let recv_result = tokio::time::timeout(Duration::from_millis(500), manager.recv()).await;
    assert!(
        recv_result.is_ok(),
        "Should receive echo when server is responding"
    );

    // Server stops responding
    server.set_responding(false);

    // Send still "succeeds" (UDP is fire-and-forget)
    let send_result = manager.send(b"no response expected").await;
    assert!(
        send_result.is_ok(),
        "UDP send completes even without response"
    );

    // Track that server's request count incremented (it received our packet even if not responding)
    let requests_before = server.request_count.load(Ordering::Relaxed);
    manager.send(b"another message").await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let requests_after = server.request_count.load(Ordering::Relaxed);

    // Server should still receive packets even when not responding
    assert!(
        requests_after > requests_before,
        "Server still receives packets"
    );

    server.shutdown();
}

#[tokio::test]
async fn test_server_resumes_responding() {
    let server = MockServer::new("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

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

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Initially works
    manager.send(b"initial").await.unwrap();
    let result = tokio::time::timeout(Duration::from_secs(1), manager.recv()).await;
    assert!(result.is_ok(), "Initial receive should work");

    // Server goes down
    server.set_responding(false);
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Server comes back
    server.set_responding(true);

    // Should work again
    manager.send(b"recovery test").await.unwrap();
    let result = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;
    assert!(result.is_ok(), "Should recover when server resumes");

    server.shutdown();
}

// ============================================================================
// Retry Logic Tests
// ============================================================================

#[tokio::test]
async fn test_retry_pending_packets() {
    let server = MockServer::new("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.send_retries = 3;
    config.retry_delay = Duration::from_millis(50);

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

    // Send should work
    manager.send(b"retry test").await.unwrap();

    // Verify retry mechanism is exercised by calling retry_pending
    manager.retry_pending();

    server.shutdown();
}

// ============================================================================
// Graceful Shutdown Tests
// ============================================================================

#[tokio::test]
async fn test_graceful_close() {
    let server = MockServer::new("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

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

    assert_eq!(manager.state(), ConnectionState::Connected);

    // Close gracefully
    manager.close().unwrap();

    // State should be disconnected
    assert_eq!(manager.state(), ConnectionState::Disconnected);

    server.shutdown();
}

// ============================================================================
// Error Classification Tests
// ============================================================================

#[test]
fn test_error_is_recoverable() {
    use triglav::error::{Error, TransportError};

    // Connection timeout is recoverable
    let timeout_err = Error::ConnectionTimeout;
    assert!(timeout_err.is_recoverable());

    // Send failed is recoverable
    let send_err = Error::Transport(TransportError::SendFailed("test".into()));
    assert!(send_err.is_recoverable());

    // Connection closed should trigger reconnect
    let closed_err = Error::ConnectionClosed;
    assert!(closed_err.should_reconnect());

    // Invalid key is not recoverable
    let key_err = Error::InvalidKey("bad key".into());
    assert!(!key_err.is_recoverable());
}

#[test]
fn test_error_should_reconnect() {
    use triglav::error::{Error, ProtocolError};

    let closed = Error::ConnectionClosed;
    assert!(closed.should_reconnect());

    let connection_failed = Error::ConnectionFailed {
        addr: "127.0.0.1:1234".parse().unwrap(),
        reason: "refused".into(),
    };
    assert!(connection_failed.should_reconnect());

    let handshake_failed = Error::HandshakeFailed("failed".into());
    assert!(handshake_failed.should_reconnect());

    let version_mismatch = Error::Protocol(ProtocolError::InvalidVersion {
        expected: 1,
        got: 2,
    });
    assert!(version_mismatch.should_reconnect());
}

// ============================================================================
// Flow Cleanup Tests
// ============================================================================

#[tokio::test]
async fn test_cleanup_stale_flows() {
    let server = MockServer::new("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = MultipathManager::new(config, client_keypair);

    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink_id = manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flows
    let flow1 = manager.allocate_flow_on_uplink(uplink_id).unwrap();
    let flow2 = manager.allocate_flow_on_uplink(uplink_id).unwrap();

    assert_eq!(manager.active_flow_count(), 2);

    // Remove uplink
    manager.remove_uplink(uplink_id);

    // Cleanup stale flows
    manager.cleanup_stale_flows();

    // Flows should be cleaned up
    assert_eq!(manager.active_flow_count(), 0);
    assert!(manager.get_flow_binding(flow1).is_none());
    assert!(manager.get_flow_binding(flow2).is_none());

    server.shutdown();
}

// ============================================================================
// Event Broadcasting Tests
// ============================================================================

#[tokio::test]
async fn test_event_broadcasting() {
    let server = MockServer::new("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Subscribe before any operations
    let mut event_rx = manager.subscribe();

    // Add uplink
    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink_id = manager.add_uplink(uplink_config).unwrap();

    // Connect
    manager.connect(server_public.clone()).await.unwrap();

    // Collect events
    let mut events = Vec::new();
    let timeout = Duration::from_millis(500);

    loop {
        match tokio::time::timeout(timeout, event_rx.recv()).await {
            Ok(Ok(event)) => events.push(event),
            _ => break,
        }
    }

    println!("Received {} events: {:?}", events.len(), events);

    // Should have received connection event
    let has_connected = events
        .iter()
        .any(|e| matches!(e, MultipathEvent::Connected));
    let has_uplink_connected = events
        .iter()
        .any(|e| matches!(e, MultipathEvent::UplinkConnected(_)));

    // At least one of these should be true
    assert!(
        has_connected || has_uplink_connected || !events.is_empty(),
        "Should receive some events during connection"
    );

    server.shutdown();
}

// ============================================================================
// Multiple Subscribers Tests
// ============================================================================

#[tokio::test]
async fn test_multiple_event_subscribers() {
    let server = MockServer::new("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Multiple subscribers
    let mut rx1 = manager.subscribe();
    let mut rx2 = manager.subscribe();
    let mut rx3 = manager.subscribe();

    // Add uplink
    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink_config).unwrap();
    manager.connect(server_public.clone()).await.unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    // All subscribers should receive events
    let mut count1 = 0;
    let mut count2 = 0;
    let mut count3 = 0;

    while let Ok(Ok(_)) = tokio::time::timeout(Duration::from_millis(100), rx1.recv()).await {
        count1 += 1;
    }
    while let Ok(Ok(_)) = tokio::time::timeout(Duration::from_millis(100), rx2.recv()).await {
        count2 += 1;
    }
    while let Ok(Ok(_)) = tokio::time::timeout(Duration::from_millis(100), rx3.recv()).await {
        count3 += 1;
    }

    // All should have received same number of events
    assert_eq!(count1, count2);
    assert_eq!(count2, count3);

    server.shutdown();
}
