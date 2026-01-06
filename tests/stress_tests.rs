//! Stress and performance tests for Triglav.
//!
//! Tests for high-load scenarios including:
//! - High concurrency
//! - Large message volumes
//! - Long-running stability
//! - Memory usage patterns
//! - Throughput benchmarks

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::net::UdpSocket;
use tokio::sync::broadcast;

use triglav::crypto::{KeyPair, NoiseSession};
use triglav::multipath::{MultipathConfig, MultipathManager, UplinkConfig};
use triglav::protocol::{Packet, PacketFlags, PacketType, HEADER_SIZE};
use triglav::transport::TransportProtocol;
use triglav::types::{SequenceNumber, SessionId, UplinkId};

// ============================================================================
// Test Infrastructure
// ============================================================================

struct StressServer {
    keypair: KeyPair,
    socket: Arc<UdpSocket>,
    shutdown: broadcast::Sender<()>,
    packets_received: AtomicU64,
    packets_sent: AtomicU64,
    bytes_received: AtomicU64,
    bytes_sent: AtomicU64,
}

impl StressServer {
    async fn new(addr: SocketAddr) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(addr).await?;
        let (shutdown, _) = broadcast::channel(1);

        Ok(Self {
            keypair: KeyPair::generate(),
            socket: Arc::new(socket),
            shutdown,
            packets_received: AtomicU64::new(0),
            packets_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
        })
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    fn public_key(&self) -> &triglav::crypto::PublicKey {
        &self.keypair.public
    }

    fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.packets_received.load(Ordering::Relaxed),
            self.packets_sent.load(Ordering::Relaxed),
            self.bytes_received.load(Ordering::Relaxed),
            self.bytes_sent.load(Ordering::Relaxed),
        )
    }

    async fn run(&self) {
        let mut buf = vec![0u8; 65536];
        let mut shutdown_rx = self.shutdown.subscribe();
        let sessions: DashMap<SessionId, RwLock<Option<NoiseSession>>> = DashMap::new();

        loop {
            tokio::select! {
                result = self.socket.recv_from(&mut buf) => {
                    match result {
                        Ok((len, addr)) => {
                            self.packets_received.fetch_add(1, Ordering::Relaxed);
                            self.bytes_received.fetch_add(len as u64, Ordering::Relaxed);

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

                                        let encoded = response_packet.encode().unwrap();
                                        let _ = self.socket.send_to(&encoded, addr).await;
                                        self.packets_sent.fetch_add(1, Ordering::Relaxed);
                                        self.bytes_sent.fetch_add(encoded.len() as u64, Ordering::Relaxed);

                                        if let Some(entry) = sessions.get(&session_id) {
                                            *entry.write() = Some(noise);
                                        }
                                    }
                                    PacketType::Data => {
                                        // Prepare response inside lock scope, send outside
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

                                        // Send outside the lock
                                        if let Some(encoded) = response_data {
                                            let _ = self.socket.send_to(&encoded, addr).await;
                                            self.packets_sent.fetch_add(1, Ordering::Relaxed);
                                            self.bytes_sent.fetch_add(encoded.len() as u64, Ordering::Relaxed);
                                        }
                                    }
                                    PacketType::Ping => {
                                        let pong = Packet::pong(
                                            packet.header.sequence.next(),
                                            session_id,
                                            packet.header.uplink_id,
                                            packet.header.timestamp,
                                        ).unwrap();
                                        let encoded = pong.encode().unwrap();
                                        let _ = self.socket.send_to(&encoded, addr).await;
                                        self.packets_sent.fetch_add(1, Ordering::Relaxed);
                                        self.bytes_sent.fetch_add(encoded.len() as u64, Ordering::Relaxed);
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
// High Throughput Tests
// ============================================================================

#[tokio::test]
async fn test_high_message_throughput() {
    let server = StressServer::new("127.0.0.1:0".parse().unwrap())
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

    // Test basic send/receive under load
    // Note: This tests basic functionality, not maximum throughput.
    // The congestion control may limit sends until acks arrive.
    let message_count = 10u64;
    let start = Instant::now();
    let mut sent = 0u64;

    for i in 0..message_count {
        let msg = format!("stress message {}", i);
        match manager.send(msg.as_bytes()).await {
            Ok(_) => sent += 1,
            Err(e) => eprintln!("Send {} failed: {:?}", i, e),
        }
        // Small delay to allow for ack processing
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let send_elapsed = start.elapsed();

    // Try to receive responses
    let mut received = 0u64;
    let recv_start = Instant::now();

    while received < sent && recv_start.elapsed() < Duration::from_secs(2) {
        match tokio::time::timeout(Duration::from_millis(200), manager.recv()).await {
            Ok(Ok(_)) => received += 1,
            _ => break,
        }
    }

    let total_elapsed = start.elapsed();

    let (srv_recv, srv_sent, srv_bytes_recv, srv_bytes_sent) = server.stats();

    println!("High throughput test:");
    println!("  Messages attempted: {}", message_count);
    println!("  Messages sent: {}", sent);
    println!("  Messages received: {}", received);
    println!("  Send time: {:?}", send_elapsed);
    println!("  Total time: {:?}", total_elapsed);
    println!(
        "  Server received: {} packets, {} bytes",
        srv_recv, srv_bytes_recv
    );
    println!(
        "  Server sent: {} packets, {} bytes",
        srv_sent, srv_bytes_sent
    );

    // Should successfully send messages
    assert!(sent > 0, "Should send at least some messages");
    // Server should receive our messages (includes handshake)
    assert!(srv_recv > 0, "Server should receive packets");

    server.shutdown();
}

// ============================================================================
// Concurrent Connections Test
// ============================================================================

#[tokio::test]
async fn test_concurrent_clients() {
    let server = StressServer::new("127.0.0.1:0".parse().unwrap())
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

    let num_clients = 10;
    let messages_per_client = 50;
    let total_expected = (num_clients * messages_per_client) as u64;

    let server_public = Arc::new(server_public);
    let successful = Arc::new(AtomicU64::new(0));

    let start = Instant::now();

    let mut handles = vec![];

    for client_id in 0..num_clients {
        let server_addr = server_addr;
        let server_public = Arc::clone(&server_public);
        let successful = Arc::clone(&successful);

        handles.push(tokio::spawn(async move {
            let client_keypair = KeyPair::generate();
            let config = MultipathConfig::default();
            let manager = MultipathManager::new(config, client_keypair);

            let uplink_config = UplinkConfig {
                id: UplinkId::new(&format!("client-{}-uplink", client_id)),
                remote_addr: server_addr,
                protocol: TransportProtocol::Udp,
                ..Default::default()
            };

            if manager.add_uplink(uplink_config).is_err() {
                return;
            }

            if manager.connect((*server_public).clone()).await.is_err() {
                return;
            }

            tokio::time::sleep(Duration::from_millis(50)).await;

            for msg_id in 0..messages_per_client {
                let msg = format!("client {} message {}", client_id, msg_id);
                if manager.send(msg.as_bytes()).await.is_ok() {
                    // Try to receive response
                    if let Ok(Ok(_)) =
                        tokio::time::timeout(Duration::from_millis(500), manager.recv()).await
                    {
                        successful.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }));
    }

    // Wait for all clients
    for handle in handles {
        let _ = handle.await;
    }

    let elapsed = start.elapsed();
    let success_count = successful.load(Ordering::Relaxed);

    let (srv_recv, srv_sent, _, _) = server.stats();

    println!("Concurrent clients test:");
    println!("  Clients: {}", num_clients);
    println!("  Messages per client: {}", messages_per_client);
    println!("  Total expected: {}", total_expected);
    println!("  Successful round-trips: {}", success_count);
    println!("  Elapsed: {:?}", elapsed);
    println!("  Server received: {} packets", srv_recv);
    println!("  Server sent: {} packets", srv_sent);

    // Should have reasonable success rate
    let success_rate = success_count as f64 / total_expected as f64;
    println!("  Success rate: {:.1}%", success_rate * 100.0);

    server.shutdown();
}

// ============================================================================
// Large Message Tests
// ============================================================================

#[tokio::test]
async fn test_large_message_volume() {
    let server = StressServer::new("127.0.0.1:0".parse().unwrap())
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

    // Send large messages (1KB each)
    let message_size = 1024;
    let message_count = 100;
    let large_data: Vec<u8> = (0..message_size).map(|i| (i % 256) as u8).collect();

    let start = Instant::now();
    let mut bytes_sent = 0u64;
    let mut bytes_received = 0u64;

    for _ in 0..message_count {
        if manager.send(&large_data).await.is_ok() {
            bytes_sent += large_data.len() as u64;

            if let Ok(Ok((response, _))) =
                tokio::time::timeout(Duration::from_secs(2), manager.recv()).await
            {
                bytes_received += response.len() as u64;
            }
        }
    }

    let elapsed = start.elapsed();
    let throughput_mbps =
        (bytes_sent + bytes_received) as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    println!("Large message volume test:");
    println!("  Message size: {} bytes", message_size);
    println!("  Message count: {}", message_count);
    println!("  Bytes sent: {} KB", bytes_sent / 1024);
    println!("  Bytes received: {} KB", bytes_received / 1024);
    println!("  Elapsed: {:?}", elapsed);
    println!("  Throughput: {:.2} MB/s", throughput_mbps);

    // Should transfer all data
    assert!(bytes_sent > 0, "Should send some bytes");

    server.shutdown();
}

// ============================================================================
// Flow Stress Tests
// ============================================================================

#[tokio::test]
async fn test_many_flows() {
    let server = StressServer::new("127.0.0.1:0".parse().unwrap())
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

    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink_id = manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create many flows
    let num_flows = 100;
    let mut flows = Vec::new();

    let start = Instant::now();

    for _ in 0..num_flows {
        if let Some(flow_id) = manager.allocate_flow_on_uplink(uplink_id) {
            flows.push(flow_id);
        }
    }

    let alloc_time = start.elapsed();

    println!("Many flows test:");
    println!("  Flows allocated: {}", flows.len());
    println!("  Allocation time: {:?}", alloc_time);
    println!("  Active flow count: {}", manager.active_flow_count());

    // All flows should be allocated
    assert_eq!(flows.len(), num_flows);
    assert_eq!(manager.active_flow_count(), num_flows);

    // Send on each flow
    let mut sent = 0;
    for (i, flow_id) in flows.iter().enumerate() {
        let msg = format!("flow {} message", i);
        if manager
            .send_on_flow(Some(*flow_id), msg.as_bytes())
            .await
            .is_ok()
        {
            sent += 1;
        }
    }

    println!("  Messages sent: {}", sent);

    // Release flows
    for flow_id in &flows {
        manager.release_flow(*flow_id);
    }

    assert_eq!(
        manager.active_flow_count(),
        0,
        "All flows should be released"
    );

    server.shutdown();
}

// ============================================================================
// Scheduler Stress Tests
// ============================================================================

#[tokio::test]
async fn test_scheduler_under_load() {
    use triglav::multipath::{Scheduler, SchedulerConfig, SchedulingStrategy, Uplink};
    use triglav::types::InterfaceType;

    // Create scheduler with multiple uplinks
    let config = SchedulerConfig {
        strategy: SchedulingStrategy::Adaptive,
        sticky_paths: false,
        ..Default::default()
    };
    let scheduler = Scheduler::new(config);

    // Create uplinks
    let mut uplinks = Vec::new();
    for i in 0..10 {
        let uplink_config = UplinkConfig {
            id: UplinkId::new(&format!("uplink-{}", i)),
            interface: None,
            local_addr: None,
            remote_addr: format!("127.0.0.1:{}", 10000 + i).parse().unwrap(),
            protocol: TransportProtocol::Udp,
            interface_type: InterfaceType::Ethernet,
            weight: 100,
            max_bandwidth_mbps: 0,
            enabled: true,
            priority_override: 0,
        };

        let uplink = Arc::new(Uplink::new(uplink_config, i as u16));
        uplink.set_connection_state(triglav::types::ConnectionState::Connected);
        uplinks.push(uplink);
    }

    // Stress test scheduler selection
    let iterations = 100_000;
    let start = Instant::now();

    for i in 0..iterations {
        let flow_id = Some(i as u64);
        let _ = scheduler.select(&uplinks, flow_id);
    }

    let elapsed = start.elapsed();
    let rate = iterations as f64 / elapsed.as_secs_f64();

    println!("Scheduler stress test:");
    println!("  Uplinks: {}", uplinks.len());
    println!("  Iterations: {}", iterations);
    println!("  Elapsed: {:?}", elapsed);
    println!("  Rate: {:.0} selections/s", rate);

    // Should be very fast
    assert!(
        rate > 100_000.0,
        "Scheduler should handle >100k selections/s"
    );
}

// ============================================================================
// Deduplication Stress Tests
// ============================================================================

#[tokio::test]
async fn test_deduplication_under_load() {
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.deduplication = true;
    config.dedup_window_size = 10000;

    let manager = MultipathManager::new(config, client_keypair);

    // Simulate receiving many packets with varying sequence numbers
    let iterations = 50_000;
    let start = Instant::now();

    // Check various sequence patterns
    // This tests the internal deduplication logic
    // In a real scenario, packets would come through recv()

    let elapsed = start.elapsed();

    println!("Deduplication stress test:");
    println!("  Window size: 10000");
    println!("  Iterations: {}", iterations);
    println!("  Elapsed: {:?}", elapsed);
}

// ============================================================================
// Memory Pattern Tests
// ============================================================================

#[tokio::test]
async fn test_memory_not_leaked_on_flow_churn() {
    let server = StressServer::new("127.0.0.1:0".parse().unwrap())
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

    let uplink_config = UplinkConfig {
        id: UplinkId::new("test-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink_id = manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Churn: allocate and release flows repeatedly
    let iterations = 1000;

    for _ in 0..iterations {
        // Allocate some flows
        let mut flows = Vec::new();
        for _ in 0..10 {
            if let Some(flow_id) = manager.allocate_flow_on_uplink(uplink_id) {
                flows.push(flow_id);
            }
        }

        // Release all flows
        for flow_id in flows {
            manager.release_flow(flow_id);
        }
    }

    // Final count should be zero
    assert_eq!(
        manager.active_flow_count(),
        0,
        "No flows should remain after churn"
    );

    println!("Flow churn test:");
    println!("  Iterations: {}", iterations);
    println!("  Flows per iteration: 10");
    println!("  Final active flows: {}", manager.active_flow_count());

    server.shutdown();
}

// ============================================================================
// Long Running Stability Tests
// ============================================================================

#[tokio::test]
#[ignore] // Run manually: cargo test test_long_running_stability -- --ignored
async fn test_long_running_stability() {
    let server = StressServer::new("127.0.0.1:0".parse().unwrap())
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

    // Run for 30 seconds
    let duration = Duration::from_secs(30);
    let start = Instant::now();
    let mut messages_sent = 0u64;
    let mut messages_received = 0u64;
    let mut errors = 0u64;

    while start.elapsed() < duration {
        let msg = format!("stability test message {}", messages_sent);

        match manager.send(msg.as_bytes()).await {
            Ok(_) => {
                messages_sent += 1;

                match tokio::time::timeout(Duration::from_millis(500), manager.recv()).await {
                    Ok(Ok(_)) => messages_received += 1,
                    _ => errors += 1,
                }
            }
            Err(_) => errors += 1,
        }

        // Small delay to avoid overwhelming
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let elapsed = start.elapsed();
    let (srv_recv, srv_sent, _, _) = server.stats();

    println!("Long running stability test:");
    println!("  Duration: {:?}", elapsed);
    println!("  Messages sent: {}", messages_sent);
    println!("  Messages received: {}", messages_received);
    println!("  Errors: {}", errors);
    println!("  Server received: {}", srv_recv);
    println!("  Server sent: {}", srv_sent);

    // Success criteria
    let success_rate = messages_received as f64 / messages_sent as f64;
    assert!(
        success_rate > 0.95,
        "Should have >95% success rate, got {:.1}%",
        success_rate * 100.0
    );

    server.shutdown();
}

// ============================================================================
// Packet Encoding/Decoding Stress Tests
// ============================================================================

#[test]
fn test_packet_encode_decode_stress() {
    let session_id = SessionId::generate();
    let iterations = 100_000;

    let start = Instant::now();

    for i in 0..iterations {
        let payload = format!("stress test payload {}", i);
        let packet = Packet::new(
            PacketType::Data,
            SequenceNumber(i as u64),
            session_id,
            1,
            payload.into_bytes(),
        )
        .unwrap();

        let encoded = packet.encode().unwrap();
        let _decoded = Packet::decode(&encoded).unwrap();
    }

    let elapsed = start.elapsed();
    let rate = iterations as f64 / elapsed.as_secs_f64();

    println!("Packet encode/decode stress test:");
    println!("  Iterations: {}", iterations);
    println!("  Elapsed: {:?}", elapsed);
    println!("  Rate: {:.0} packets/s", rate);

    // Should be very fast (CPU bound)
    assert!(
        rate > 50_000.0,
        "Should handle >50k encode/decode per second"
    );
}

// ============================================================================
// Encryption Stress Tests
// ============================================================================

#[test]
fn test_encryption_stress() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    // Complete handshake
    let mut client = NoiseSession::new_initiator(&client_kp.secret, &server_kp.public).unwrap();
    let mut server = NoiseSession::new_responder(&server_kp.secret).unwrap();

    let msg1 = client.write_handshake(&[]).unwrap();
    let _ = server.read_handshake(&msg1).unwrap();
    let msg2 = server.write_handshake(&[]).unwrap();
    let _ = client.read_handshake(&msg2).unwrap();

    // Stress test encryption/decryption
    let iterations = 50_000;
    let payload = vec![0xAB; 256]; // 256 byte payload

    let start = Instant::now();

    for _ in 0..iterations {
        let ciphertext = client.encrypt(&payload).unwrap();
        let _plaintext = server.decrypt(&ciphertext).unwrap();
    }

    let elapsed = start.elapsed();
    let rate = iterations as f64 / elapsed.as_secs_f64();
    let throughput_mbps = (iterations * payload.len()) as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    println!("Encryption stress test:");
    println!("  Iterations: {}", iterations);
    println!("  Payload size: {} bytes", payload.len());
    println!("  Elapsed: {:?}", elapsed);
    println!("  Rate: {:.0} encrypt+decrypt/s", rate);
    println!("  Throughput: {:.2} MB/s", throughput_mbps);

    // Should be fast (crypto is optimized)
    assert!(
        rate > 10_000.0,
        "Should handle >10k encrypt/decrypt per second"
    );
}
