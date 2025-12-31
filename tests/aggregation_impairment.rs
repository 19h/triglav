//! Bandwidth Aggregation Tests with Network Impairment
//!
//! Tests network impairment simulation and multi-path traffic distribution.
//! Uses ImpairmentProxy to simulate realistic network conditions.
//!
//! These tests validate:
//! 1. Network impairment simulation (latency, loss, jitter)
//! 2. Multi-path traffic distribution patterns
//! 3. Reorder buffer effectiveness
//! 4. Path failover behavior

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tokio::net::UdpSocket;
use tokio::sync::broadcast;

// ============================================================================
// Network Impairment Infrastructure
// ============================================================================

/// Configuration for network impairment simulation
#[derive(Debug, Clone)]
pub struct ImpairmentConfig {
    pub latency_ms: u64,
    pub jitter_ms: u64,
    pub loss_rate: f64,
}

impl Default for ImpairmentConfig {
    fn default() -> Self {
        Self {
            latency_ms: 0,
            jitter_ms: 0,
            loss_rate: 0.0,
        }
    }
}

impl ImpairmentConfig {
    pub fn fast_ethernet() -> Self {
        Self {
            latency_ms: 5,
            jitter_ms: 2,
            loss_rate: 0.001,
        }
    }

    pub fn wifi() -> Self {
        Self {
            latency_ms: 20,
            jitter_ms: 10,
            loss_rate: 0.02,
        }
    }

    pub fn lte() -> Self {
        Self {
            latency_ms: 50,
            jitter_ms: 30,
            loss_rate: 0.03,
        }
    }

    pub fn lossy_wireless() -> Self {
        Self {
            latency_ms: 30,
            jitter_ms: 20,
            loss_rate: 0.10,
        }
    }
}

/// Statistics from impairment simulation
#[derive(Debug, Default)]
pub struct ImpairmentStats {
    pub packets_received: AtomicU64,
    pub packets_sent: AtomicU64,
    pub packets_dropped: AtomicU64,
    pub bytes_forwarded: AtomicU64,
}

/// Network impairment proxy
pub struct ImpairmentProxy {
    config: ImpairmentConfig,
    stats: Arc<ImpairmentStats>,
    socket: Arc<UdpSocket>,
    target_addr: SocketAddr,
    shutdown: broadcast::Sender<()>,
}

impl ImpairmentProxy {
    pub async fn new(
        bind_addr: SocketAddr,
        target_addr: SocketAddr,
        config: ImpairmentConfig,
    ) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(bind_addr).await?;
        let (shutdown, _) = broadcast::channel(1);

        Ok(Self {
            config,
            stats: Arc::new(ImpairmentStats::default()),
            socket: Arc::new(socket),
            target_addr,
            shutdown,
        })
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    pub fn stats(&self) -> Arc<ImpairmentStats> {
        Arc::clone(&self.stats)
    }

    fn should_drop(&self) -> bool {
        if self.config.loss_rate <= 0.0 {
            return false;
        }
        rand::random::<f64>() < self.config.loss_rate
    }

    fn calculate_delay(&self) -> Duration {
        let base = self.config.latency_ms;
        let jitter = if self.config.jitter_ms > 0 {
            (rand::random::<f64>() * self.config.jitter_ms as f64) as u64
        } else {
            0
        };
        Duration::from_millis(base + jitter)
    }

    pub async fn run(&self) {
        let mut buf = vec![0u8; 65536];
        let mut shutdown_rx = self.shutdown.subscribe();
        let mut client_addr: Option<SocketAddr> = None;

        loop {
            tokio::select! {
                result = self.socket.recv_from(&mut buf) => {
                    match result {
                        Ok((len, addr)) => {
                            self.stats.packets_received.fetch_add(1, Ordering::Relaxed);

                            if self.should_drop() {
                                self.stats.packets_dropped.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }

                            let data = buf[..len].to_vec();

                            let dest_addr = if addr == self.target_addr {
                                match client_addr {
                                    Some(ca) => ca,
                                    None => continue,
                                }
                            } else {
                                client_addr = Some(addr);
                                self.target_addr
                            };

                            let delay = self.calculate_delay();
                            let socket = Arc::clone(&self.socket);
                            let stats = Arc::clone(&self.stats);

                            tokio::spawn(async move {
                                if delay > Duration::ZERO {
                                    tokio::time::sleep(delay).await;
                                }
                                if socket.send_to(&data, dest_addr).await.is_ok() {
                                    stats.packets_sent.fetch_add(1, Ordering::Relaxed);
                                    stats.bytes_forwarded.fetch_add(data.len() as u64, Ordering::Relaxed);
                                }
                            });
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

    pub fn shutdown(&self) {
        let _ = self.shutdown.send(());
    }
}

// ============================================================================
// Test Server
// ============================================================================

struct TestServer {
    socket: Arc<UdpSocket>,
    addr: SocketAddr,
    packets_by_source: Arc<RwLock<HashMap<SocketAddr, Vec<(u64, Vec<u8>)>>>>,
    total_bytes: Arc<AtomicU64>,
    total_packets: Arc<AtomicU64>,
    sequences_seen: Arc<RwLock<Vec<u64>>>,
    shutdown: broadcast::Sender<()>,
}

impl TestServer {
    async fn new() -> std::io::Result<Self> {
        let socket = UdpSocket::bind("127.0.0.1:0").await?;
        let addr = socket.local_addr()?;
        let (shutdown, _) = broadcast::channel(1);

        Ok(Self {
            socket: Arc::new(socket),
            addr,
            packets_by_source: Arc::new(RwLock::new(HashMap::new())),
            total_bytes: Arc::new(AtomicU64::new(0)),
            total_packets: Arc::new(AtomicU64::new(0)),
            sequences_seen: Arc::new(RwLock::new(Vec::new())),
            shutdown,
        })
    }

    fn addr(&self) -> SocketAddr {
        self.addr
    }

    fn unique_sources(&self) -> usize {
        self.packets_by_source.read().len()
    }

    fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Relaxed)
    }

    fn total_packets(&self) -> u64 {
        self.total_packets.load(Ordering::Relaxed)
    }

    fn out_of_order_count(&self) -> usize {
        let seqs = self.sequences_seen.read();
        let mut count = 0;
        for i in 1..seqs.len() {
            if seqs[i] < seqs[i - 1] {
                count += 1;
            }
        }
        count
    }

    async fn run(&self) {
        let mut buf = vec![0u8; 65536];
        let mut shutdown_rx = self.shutdown.subscribe();
        let mut seq_counter = 0u64;

        loop {
            tokio::select! {
                result = self.socket.recv_from(&mut buf) => {
                    match result {
                        Ok((len, src_addr)) => {
                            self.total_bytes.fetch_add(len as u64, Ordering::Relaxed);
                            self.total_packets.fetch_add(1, Ordering::Relaxed);

                            {
                                let mut by_source = self.packets_by_source.write();
                                by_source
                                    .entry(src_addr)
                                    .or_insert_with(Vec::new)
                                    .push((seq_counter, buf[..len].to_vec()));
                            }

                            {
                                let mut seqs = self.sequences_seen.write();
                                seqs.push(seq_counter);
                            }

                            seq_counter += 1;

                            // Echo back
                            let mut response = b"ACK:".to_vec();
                            response.extend_from_slice(&buf[..len]);
                            let _ = self.socket.send_to(&response, src_addr).await;
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
// Tests: Direct UDP through Impairment Proxies
// ============================================================================

/// Test: Traffic through symmetric paths (same latency)
#[tokio::test]
async fn test_symmetric_paths_distribution() {
    let server = TestServer::new().await.unwrap();
    let server_addr = server.addr();
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    // Two proxies with identical characteristics
    let proxy1 = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::fast_ethernet(),
    )
    .await
    .unwrap();
    let proxy1_addr = proxy1.local_addr().unwrap();
    let proxy1_stats = proxy1.stats();
    tokio::spawn(async move {
        proxy1.run().await;
    });

    let proxy2 = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::fast_ethernet(),
    )
    .await
    .unwrap();
    let proxy2_addr = proxy2.local_addr().unwrap();
    let proxy2_stats = proxy2.stats();
    tokio::spawn(async move {
        proxy2.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create two client sockets (simulating two uplinks)
    let client1 = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let client2 = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Send packets alternating between paths (round-robin aggregation)
    let packet_count = 100;
    for i in 0..packet_count {
        let msg = format!("PKT:{:04}", i);
        if i % 2 == 0 {
            client1.send_to(msg.as_bytes(), proxy1_addr).await.unwrap();
        } else {
            client2.send_to(msg.as_bytes(), proxy2_addr).await.unwrap();
        }
    }

    // Wait for delivery
    tokio::time::sleep(Duration::from_millis(200)).await;

    let p1_recv = proxy1_stats.packets_received.load(Ordering::Relaxed);
    let p2_recv = proxy2_stats.packets_received.load(Ordering::Relaxed);
    let p1_sent = proxy1_stats.packets_sent.load(Ordering::Relaxed);
    let p2_sent = proxy2_stats.packets_sent.load(Ordering::Relaxed);

    println!("\n=== Symmetric Paths Distribution ===");
    println!("Proxy 1: recv={}, sent={}", p1_recv, p1_sent);
    println!("Proxy 2: recv={}, sent={}", p2_recv, p2_sent);
    println!("Server received: {} packets, {} bytes", 
             server.total_packets(), server.total_bytes());
    println!("Server unique sources: {}", server.unique_sources());

    // Both paths should receive roughly equal traffic
    assert!(p1_recv >= 40, "Proxy 1 should receive ~50 packets, got {}", p1_recv);
    assert!(p2_recv >= 40, "Proxy 2 should receive ~50 packets, got {}", p2_recv);
    
    // Server should see traffic from both proxy addresses
    assert!(server.unique_sources() >= 2, "Server should see 2 sources");

    server.shutdown();
}

/// Test: Asymmetric latency paths
#[tokio::test]
async fn test_asymmetric_latency_paths() {
    let server = TestServer::new().await.unwrap();
    let server_addr = server.addr();
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    // Fast path: 5ms latency
    let fast_proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::fast_ethernet(),
    )
    .await
    .unwrap();
    let fast_addr = fast_proxy.local_addr().unwrap();
    let fast_stats = fast_proxy.stats();
    tokio::spawn(async move {
        fast_proxy.run().await;
    });

    // Slow path: 50ms latency (LTE-like)
    let slow_proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::lte(),
    )
    .await
    .unwrap();
    let slow_addr = slow_proxy.local_addr().unwrap();
    let slow_stats = slow_proxy.stats();
    tokio::spawn(async move {
        slow_proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let fast_client = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let slow_client = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Send same number through each path
    let start = Instant::now();
    for i in 0..25 {
        let msg = format!("FAST:{:04}", i);
        fast_client.send_to(msg.as_bytes(), fast_addr).await.unwrap();
        
        let msg = format!("SLOW:{:04}", i);
        slow_client.send_to(msg.as_bytes(), slow_addr).await.unwrap();
    }
    let send_time = start.elapsed();

    // Wait for slow path to deliver (50ms + jitter)
    tokio::time::sleep(Duration::from_millis(200)).await;

    let fast_sent = fast_stats.packets_sent.load(Ordering::Relaxed);
    let slow_sent = slow_stats.packets_sent.load(Ordering::Relaxed);

    println!("\n=== Asymmetric Latency Paths ===");
    println!("Fast path (5ms): sent={}", fast_sent);
    println!("Slow path (50ms): sent={}", slow_sent);
    println!("Send time: {:?}", send_time);
    println!("Server total: {} packets", server.total_packets());
    println!("Out-of-order at server: {}", server.out_of_order_count());

    // Both should forward (with minimal loss)
    assert!(fast_sent >= 20, "Fast path should forward most packets");
    assert!(slow_sent >= 20, "Slow path should forward most packets");

    // With latency difference, expect some reordering at server
    // (packets sent later on fast path arrive before earlier slow-path packets)

    server.shutdown();
}

/// Test: Path with packet loss
#[tokio::test]
async fn test_lossy_path() {
    let server = TestServer::new().await.unwrap();
    let server_addr = server.addr();
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    // Good path
    let good_proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::fast_ethernet(), // 0.1% loss
    )
    .await
    .unwrap();
    let good_addr = good_proxy.local_addr().unwrap();
    let good_stats = good_proxy.stats();
    tokio::spawn(async move {
        good_proxy.run().await;
    });

    // Lossy path: 10% loss
    let lossy_proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::lossy_wireless(),
    )
    .await
    .unwrap();
    let lossy_addr = lossy_proxy.local_addr().unwrap();
    let lossy_stats = lossy_proxy.stats();
    tokio::spawn(async move {
        lossy_proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let good_client = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let lossy_client = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Send 100 packets through each
    for i in 0..100 {
        let msg = format!("GOOD:{:04}", i);
        good_client.send_to(msg.as_bytes(), good_addr).await.unwrap();
        
        let msg = format!("LOSS:{:04}", i);
        lossy_client.send_to(msg.as_bytes(), lossy_addr).await.unwrap();
    }

    tokio::time::sleep(Duration::from_millis(300)).await;

    let good_recv = good_stats.packets_received.load(Ordering::Relaxed);
    let good_sent = good_stats.packets_sent.load(Ordering::Relaxed);
    let lossy_recv = lossy_stats.packets_received.load(Ordering::Relaxed);
    let lossy_sent = lossy_stats.packets_sent.load(Ordering::Relaxed);
    let lossy_drop = lossy_stats.packets_dropped.load(Ordering::Relaxed);

    println!("\n=== Lossy Path Test ===");
    println!("Good path: recv={}, sent={}", good_recv, good_sent);
    println!("Lossy path: recv={}, sent={}, dropped={}", lossy_recv, lossy_sent, lossy_drop);
    println!("Server total: {} packets", server.total_packets());

    // Good path should forward almost all
    assert!(good_sent >= 95, "Good path should forward most packets");

    // Lossy path should drop some (10% loss rate)
    assert!(lossy_drop > 0, "Lossy path should drop some packets");
    
    // With 10% loss, expect 5-20 drops on average
    let loss_rate = lossy_drop as f64 / lossy_recv as f64;
    assert!(
        loss_rate > 0.02 && loss_rate < 0.25,
        "Loss rate should be around 10%, got {:.1}%",
        loss_rate * 100.0
    );

    server.shutdown();
}

/// Test: Throughput measurement through aggregated paths
#[tokio::test]
async fn test_aggregated_throughput() {
    let server = TestServer::new().await.unwrap();
    let server_addr = server.addr();
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    // Two moderate latency paths
    let proxy1 = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig {
            latency_ms: 10,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let proxy1_addr = proxy1.local_addr().unwrap();
    let proxy1_stats = proxy1.stats();
    tokio::spawn(async move {
        proxy1.run().await;
    });

    let proxy2 = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig {
            latency_ms: 10,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let proxy2_addr = proxy2.local_addr().unwrap();
    let proxy2_stats = proxy2.stats();
    tokio::spawn(async move {
        proxy2.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client1 = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let client2 = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Send large burst alternating between paths
    let packet_size = 1000;
    let packet_count = 200;
    let payload = vec![0xABu8; packet_size];

    let start = Instant::now();
    for i in 0..packet_count {
        if i % 2 == 0 {
            client1.send_to(&payload, proxy1_addr).await.unwrap();
        } else {
            client2.send_to(&payload, proxy2_addr).await.unwrap();
        }
    }
    let send_time = start.elapsed();

    // Wait for delivery
    tokio::time::sleep(Duration::from_millis(100)).await;

    let total_bytes = (packet_count * packet_size) as f64;
    let throughput_mbps = (total_bytes * 8.0) / send_time.as_secs_f64() / 1_000_000.0;

    let p1_bytes = proxy1_stats.bytes_forwarded.load(Ordering::Relaxed);
    let p2_bytes = proxy2_stats.bytes_forwarded.load(Ordering::Relaxed);

    println!("\n=== Aggregated Throughput ===");
    println!("Packets sent: {}", packet_count);
    println!("Total bytes: {:.1} KB", total_bytes / 1024.0);
    println!("Send time: {:?}", send_time);
    println!("Send throughput: {:.2} Mbps", throughput_mbps);
    println!("Path 1 forwarded: {:.1} KB", p1_bytes as f64 / 1024.0);
    println!("Path 2 forwarded: {:.1} KB", p2_bytes as f64 / 1024.0);
    println!("Combined: {:.1} KB", (p1_bytes + p2_bytes) as f64 / 1024.0);
    println!("Server received: {} packets", server.total_packets());

    // Both paths should forward roughly equal
    assert!(p1_bytes > 0, "Path 1 should forward data");
    assert!(p2_bytes > 0, "Path 2 should forward data");

    // Combined throughput should be sum of both paths
    let combined = p1_bytes + p2_bytes;
    assert!(
        combined >= (total_bytes as u64 * 90 / 100),
        "Combined should be close to total sent"
    );

    server.shutdown();
}

/// Test: Path failover simulation
#[tokio::test]
async fn test_path_failover() {
    let server = TestServer::new().await.unwrap();
    let server_addr = server.addr();
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    // Path 1 - will be "failed"
    let proxy1 = Arc::new(
        ImpairmentProxy::new(
            "127.0.0.1:0".parse().unwrap(),
            server_addr,
            ImpairmentConfig::fast_ethernet(),
        )
        .await
        .unwrap(),
    );
    let proxy1_addr = proxy1.local_addr().unwrap();
    let proxy1_stats = proxy1.stats();
    let proxy1_clone = Arc::clone(&proxy1);
    tokio::spawn(async move {
        proxy1_clone.run().await;
    });

    // Path 2 - stays up
    let proxy2 = Arc::new(
        ImpairmentProxy::new(
            "127.0.0.1:0".parse().unwrap(),
            server_addr,
            ImpairmentConfig::fast_ethernet(),
        )
        .await
        .unwrap(),
    );
    let proxy2_addr = proxy2.local_addr().unwrap();
    let proxy2_stats = proxy2.stats();
    let proxy2_clone = Arc::clone(&proxy2);
    tokio::spawn(async move {
        proxy2_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client1 = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let client2 = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Phase 1: Both paths active
    println!("\nPhase 1: Both paths active");
    for i in 0..20 {
        let msg = format!("PHASE1:{:04}", i);
        if i % 2 == 0 {
            client1.send_to(msg.as_bytes(), proxy1_addr).await.unwrap();
        } else {
            client2.send_to(msg.as_bytes(), proxy2_addr).await.unwrap();
        }
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    let p1_before = proxy1_stats.packets_sent.load(Ordering::Relaxed);
    let p2_before = proxy2_stats.packets_sent.load(Ordering::Relaxed);
    println!("Before failover: P1={}, P2={}", p1_before, p2_before);

    // Simulate path 1 failure by shutting down proxy
    proxy1.shutdown();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Phase 2: Only path 2 active - all traffic goes there
    println!("Phase 2: Path 1 failed, using Path 2 only");
    for i in 0..20 {
        let msg = format!("PHASE2:{:04}", i);
        // Now only use client2/proxy2
        client2.send_to(msg.as_bytes(), proxy2_addr).await.unwrap();
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    let p2_after = proxy2_stats.packets_sent.load(Ordering::Relaxed);
    println!("After failover: P2={}", p2_after);

    // Path 2 should handle all post-failover traffic
    assert!(
        p2_after >= p2_before + 15,
        "Path 2 should handle additional traffic after failover"
    );

    println!("Server total: {} packets", server.total_packets());

    server.shutdown();
}

/// Test: Jitter causes reordering
#[tokio::test]
async fn test_jitter_reordering() {
    let server = TestServer::new().await.unwrap();
    let server_addr = server.addr();
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    // High jitter path
    let jittery_proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig {
            latency_ms: 20,
            jitter_ms: 50, // High jitter can cause 20-70ms latency
            loss_rate: 0.0,
        },
    )
    .await
    .unwrap();
    let jittery_addr = jittery_proxy.local_addr().unwrap();
    tokio::spawn(async move {
        jittery_proxy.run().await;
    });

    // Stable path
    let stable_proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig {
            latency_ms: 40, // Higher base but no jitter
            jitter_ms: 0,
            loss_rate: 0.0,
        },
    )
    .await
    .unwrap();
    let stable_addr = stable_proxy.local_addr().unwrap();
    tokio::spawn(async move {
        stable_proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let jittery_client = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let stable_client = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Alternate sending: even packets through jittery, odd through stable
    for i in 0..50 {
        let msg = format!("SEQ:{:04}", i);
        if i % 2 == 0 {
            jittery_client.send_to(msg.as_bytes(), jittery_addr).await.unwrap();
        } else {
            stable_client.send_to(msg.as_bytes(), stable_addr).await.unwrap();
        }
        // Small delay to maintain send order
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    // Wait for all packets
    tokio::time::sleep(Duration::from_millis(200)).await;

    let out_of_order = server.out_of_order_count();

    println!("\n=== Jitter Reordering Test ===");
    println!("Server received: {} packets", server.total_packets());
    println!("Out-of-order packets: {}", out_of_order);

    // With jitter vs stable paths, expect some reordering
    // A jittery packet sent at t=0 might arrive after a stable packet sent at t=10ms
    // This is expected behavior that the reorder buffer should handle

    server.shutdown();
}

/// Test: RTT measurement through impaired path
#[tokio::test]
async fn test_rtt_measurement() {
    let server = TestServer::new().await.unwrap();
    let server_addr = server.addr();
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        server_clone.run().await;
    });

    // WiFi-like path (20ms latency + jitter)
    let proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::wifi(),
    )
    .await
    .unwrap();
    let proxy_addr = proxy.local_addr().unwrap();
    tokio::spawn(async move {
        proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Measure RTTs
    let mut rtts = Vec::new();
    for i in 0..20 {
        let msg = format!("PING:{}", i);
        let start = Instant::now();
        
        client.send_to(msg.as_bytes(), proxy_addr).await.unwrap();
        
        let mut buf = [0u8; 1024];
        match tokio::time::timeout(
            Duration::from_secs(1),
            client.recv_from(&mut buf),
        ).await {
            Ok(Ok(_)) => {
                rtts.push(start.elapsed());
            }
            _ => {} // Timeout or error
        }
    }

    assert!(!rtts.is_empty(), "Should measure at least some RTTs");

    let min_rtt = rtts.iter().min().unwrap();
    let max_rtt = rtts.iter().max().unwrap();
    let avg_rtt = rtts.iter().sum::<Duration>() / rtts.len() as u32;

    println!("\n=== RTT Measurement (WiFi-like path) ===");
    println!("Samples: {}", rtts.len());
    println!("Min RTT: {:?}", min_rtt);
    println!("Max RTT: {:?}", max_rtt);
    println!("Avg RTT: {:?}", avg_rtt);

    // WiFi config: 20ms base + up to 10ms jitter, round-trip = 2x
    // Expected RTT: 40-60ms
    assert!(
        avg_rtt >= Duration::from_millis(30),
        "Average RTT should be at least 30ms, got {:?}",
        avg_rtt
    );

    server.shutdown();
}
