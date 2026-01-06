//! Automated network impairment simulation tests.
//!
//! These tests simulate real-world network conditions including:
//! - Latency injection
//! - Packet loss
//! - Jitter (variable latency)
//! - Bandwidth throttling
//! - Connection drops
//!
//! Uses in-process simulation rather than kernel-level tools like tc/dnctl.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rand::Rng;
use tokio::net::UdpSocket;
use tokio::sync::broadcast;

use triglav::crypto::KeyPair;
use triglav::multipath::{MultipathConfig, MultipathManager, UplinkConfig};
use triglav::protocol::{Packet, PacketType, HEADER_SIZE};
use triglav::transport::TransportProtocol;
use triglav::types::{SequenceNumber, SessionId, UplinkId};

// ============================================================================
// Network Impairment Simulator
// ============================================================================

/// Configuration for network impairment simulation
#[derive(Debug, Clone)]
pub struct ImpairmentConfig {
    /// Base latency to add (one-way)
    pub latency_ms: u64,
    /// Jitter (random variance in latency)
    pub jitter_ms: u64,
    /// Packet loss rate (0.0 - 1.0)
    pub loss_rate: f64,
    /// Packet duplication rate
    pub duplicate_rate: f64,
    /// Packet reordering rate
    pub reorder_rate: f64,
    /// Bandwidth limit (bytes per second, 0 = unlimited)
    pub bandwidth_bps: u64,
    /// Burst loss (consecutive packets lost)
    pub burst_loss_length: usize,
    /// Probability of entering burst loss mode
    pub burst_loss_probability: f64,
}

impl Default for ImpairmentConfig {
    fn default() -> Self {
        Self {
            latency_ms: 0,
            jitter_ms: 0,
            loss_rate: 0.0,
            duplicate_rate: 0.0,
            reorder_rate: 0.0,
            bandwidth_bps: 0,
            burst_loss_length: 0,
            burst_loss_probability: 0.0,
        }
    }
}

impl ImpairmentConfig {
    /// High latency profile (satellite-like)
    pub fn high_latency() -> Self {
        Self {
            latency_ms: 300,
            jitter_ms: 50,
            ..Default::default()
        }
    }

    /// Lossy network profile (wireless)
    pub fn lossy() -> Self {
        Self {
            latency_ms: 20,
            jitter_ms: 10,
            loss_rate: 0.05, // 5% loss
            ..Default::default()
        }
    }

    /// Burst loss profile (congested network)
    pub fn burst_loss() -> Self {
        Self {
            latency_ms: 30,
            jitter_ms: 20,
            burst_loss_length: 5,
            burst_loss_probability: 0.02,
            ..Default::default()
        }
    }

    /// Mobile network profile (variable quality)
    pub fn mobile() -> Self {
        Self {
            latency_ms: 80,
            jitter_ms: 100, // High jitter
            loss_rate: 0.02,
            duplicate_rate: 0.01,
            reorder_rate: 0.03,
            ..Default::default()
        }
    }

    /// Bandwidth limited profile
    pub fn throttled(bps: u64) -> Self {
        Self {
            latency_ms: 10,
            bandwidth_bps: bps,
            ..Default::default()
        }
    }
}

/// Statistics from impairment simulation
#[derive(Debug, Default)]
pub struct ImpairmentStats {
    pub packets_received: AtomicU64,
    pub packets_sent: AtomicU64,
    pub packets_dropped: AtomicU64,
    pub packets_duplicated: AtomicU64,
    pub packets_reordered: AtomicU64,
    pub total_latency_us: AtomicU64,
    pub bytes_received: AtomicU64,
    pub bytes_sent: AtomicU64,
}

/// Network impairment proxy that sits between client and server
pub struct ImpairmentProxy {
    config: ImpairmentConfig,
    stats: Arc<ImpairmentStats>,
    socket: Arc<UdpSocket>,
    target_addr: SocketAddr,
    shutdown: broadcast::Sender<()>,
    in_burst_loss: AtomicBool,
    burst_remaining: AtomicU64,
    reorder_buffer: Mutex<VecDeque<(Vec<u8>, SocketAddr, Instant)>>,
    last_send_time: Mutex<Instant>,
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
            in_burst_loss: AtomicBool::new(false),
            burst_remaining: AtomicU64::new(0),
            reorder_buffer: Mutex::new(VecDeque::new()),
            last_send_time: Mutex::new(Instant::now()),
        })
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    pub fn stats(&self) -> Arc<ImpairmentStats> {
        Arc::clone(&self.stats)
    }

    fn should_drop(&self) -> bool {
        let mut rng = rand::thread_rng();

        // Check burst loss
        if self.in_burst_loss.load(Ordering::Relaxed) {
            let remaining = self.burst_remaining.fetch_sub(1, Ordering::Relaxed);
            if remaining <= 1 {
                self.in_burst_loss.store(false, Ordering::Relaxed);
            }
            return true;
        }

        // Check for new burst
        if self.config.burst_loss_probability > 0.0 {
            if rng.gen::<f64>() < self.config.burst_loss_probability {
                self.in_burst_loss.store(true, Ordering::Relaxed);
                self.burst_remaining
                    .store(self.config.burst_loss_length as u64, Ordering::Relaxed);
                return true;
            }
        }

        // Random loss
        rng.gen::<f64>() < self.config.loss_rate
    }

    fn should_duplicate(&self) -> bool {
        rand::thread_rng().gen::<f64>() < self.config.duplicate_rate
    }

    fn should_reorder(&self) -> bool {
        rand::thread_rng().gen::<f64>() < self.config.reorder_rate
    }

    fn calculate_delay(&self) -> Duration {
        let mut rng = rand::thread_rng();
        let base = self.config.latency_ms;
        let jitter = if self.config.jitter_ms > 0 {
            rng.gen_range(0..=self.config.jitter_ms)
        } else {
            0
        };
        Duration::from_millis(base + jitter)
    }

    fn check_bandwidth(&self, bytes: usize) -> Duration {
        if self.config.bandwidth_bps == 0 {
            return Duration::ZERO;
        }

        let bits = bytes * 8;
        let seconds = bits as f64 / self.config.bandwidth_bps as f64;
        Duration::from_secs_f64(seconds)
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
                            self.stats.bytes_received.fetch_add(len as u64, Ordering::Relaxed);

                            let data = buf[..len].to_vec();

                            // Determine if this is from client or server
                            let (dest_addr, is_response) = if addr == self.target_addr {
                                // Response from server, send to client
                                match client_addr {
                                    Some(ca) => (ca, true),
                                    None => continue,
                                }
                            } else {
                                // Request from client, send to server
                                client_addr = Some(addr);
                                (self.target_addr, false)
                            };

                            // Apply impairments
                            if self.should_drop() {
                                self.stats.packets_dropped.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }

                            let delay = self.calculate_delay();
                            let bw_delay = self.check_bandwidth(len);
                            let total_delay = delay + bw_delay;

                            self.stats.total_latency_us.fetch_add(
                                total_delay.as_micros() as u64,
                                Ordering::Relaxed
                            );

                            // Handle reordering
                            if self.should_reorder() {
                                self.stats.packets_reordered.fetch_add(1, Ordering::Relaxed);
                                let mut buffer = self.reorder_buffer.lock();
                                buffer.push_back((data.clone(), dest_addr, Instant::now() + total_delay));
                                continue;
                            }

                            // Handle duplication
                            let send_count = if self.should_duplicate() {
                                self.stats.packets_duplicated.fetch_add(1, Ordering::Relaxed);
                                2
                            } else {
                                1
                            };

                            // Send with delay
                            let socket = Arc::clone(&self.socket);
                            let stats = Arc::clone(&self.stats);
                            tokio::spawn(async move {
                                if total_delay > Duration::ZERO {
                                    tokio::time::sleep(total_delay).await;
                                }
                                for _ in 0..send_count {
                                    let _ = socket.send_to(&data, dest_addr).await;
                                    stats.packets_sent.fetch_add(1, Ordering::Relaxed);
                                    stats.bytes_sent.fetch_add(data.len() as u64, Ordering::Relaxed);
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

            // Flush reorder buffer
            let now = Instant::now();
            let mut buffer = self.reorder_buffer.lock();
            while let Some((data, addr, send_time)) = buffer.front() {
                if *send_time <= now {
                    let data = data.clone();
                    let addr = *addr;
                    buffer.pop_front();
                    let socket = Arc::clone(&self.socket);
                    let stats = Arc::clone(&self.stats);
                    tokio::spawn(async move {
                        let _ = socket.send_to(&data, addr).await;
                        stats.packets_sent.fetch_add(1, Ordering::Relaxed);
                    });
                } else {
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
// Test Helpers
// ============================================================================

/// Simple test server that echoes data
async fn run_echo_server(addr: SocketAddr, shutdown: broadcast::Receiver<()>) {
    let socket = UdpSocket::bind(addr).await.unwrap();
    let mut buf = vec![0u8; 65536];
    let mut shutdown = shutdown;

    loop {
        tokio::select! {
            result = socket.recv_from(&mut buf) => {
                if let Ok((len, addr)) = result {
                    let _ = socket.send_to(&buf[..len], addr).await;
                }
            }
            _ = shutdown.recv() => {
                break;
            }
        }
    }
}

// ============================================================================
// High Latency Tests
// ============================================================================

#[tokio::test]
async fn test_high_latency_network() {
    let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);

    // Start echo server
    let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let server = UdpSocket::bind(server_addr).await.unwrap();
    let server_addr = server.local_addr().unwrap();

    tokio::spawn(async move {
        let mut buf = vec![0u8; 65536];
        let mut shutdown = shutdown_rx;
        loop {
            tokio::select! {
                result = server.recv_from(&mut buf) => {
                    if let Ok((len, addr)) = result {
                        let _ = server.send_to(&buf[..len], addr).await;
                    }
                }
                _ = shutdown.recv() => break,
            }
        }
    });

    // Start impairment proxy with high latency
    let proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::high_latency(),
    )
    .await
    .unwrap();

    let proxy_addr = proxy.local_addr().unwrap();
    let stats = proxy.stats();

    tokio::spawn(async move {
        proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Client sends through proxy
    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let start = Instant::now();

    let test_data = b"test message";
    client.send_to(test_data, proxy_addr).await.unwrap();

    let mut buf = [0u8; 1024];
    let result = tokio::time::timeout(Duration::from_secs(5), client.recv_from(&mut buf)).await;

    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Should receive response");
    let (len, _) = result.unwrap().unwrap();
    assert_eq!(&buf[..len], test_data);

    // Verify latency was added (should be at least 2x base latency for round trip)
    assert!(
        elapsed >= Duration::from_millis(500),
        "Round trip should have at least 500ms latency, got {:?}",
        elapsed
    );

    println!("High latency test: RTT = {:?}", elapsed);
    println!(
        "Stats: received={}, sent={}, dropped={}",
        stats.packets_received.load(Ordering::Relaxed),
        stats.packets_sent.load(Ordering::Relaxed),
        stats.packets_dropped.load(Ordering::Relaxed)
    );

    let _ = shutdown_tx.send(());
}

// ============================================================================
// Packet Loss Tests
// ============================================================================

#[tokio::test]
async fn test_lossy_network() {
    let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);

    // Start echo server
    let server = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let server_addr = server.local_addr().unwrap();

    tokio::spawn(async move {
        let mut buf = vec![0u8; 65536];
        let mut shutdown = shutdown_rx;
        loop {
            tokio::select! {
                result = server.recv_from(&mut buf) => {
                    if let Ok((len, addr)) = result {
                        let _ = server.send_to(&buf[..len], addr).await;
                    }
                }
                _ = shutdown.recv() => break,
            }
        }
    });

    // Start impairment proxy with 20% loss
    let proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig {
            loss_rate: 0.20, // 20% loss
            latency_ms: 5,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let proxy_addr = proxy.local_addr().unwrap();
    let stats = proxy.stats();

    tokio::spawn(async move {
        proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Send many packets
    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let total_packets = 100;
    let mut received = 0;

    for i in 0..total_packets {
        let msg = format!("packet {}", i);
        client.send_to(msg.as_bytes(), proxy_addr).await.unwrap();

        let mut buf = [0u8; 1024];
        match tokio::time::timeout(Duration::from_millis(100), client.recv_from(&mut buf)).await {
            Ok(Ok(_)) => received += 1,
            _ => {} // Dropped or timeout
        }
    }

    let dropped = stats.packets_dropped.load(Ordering::Relaxed);
    let loss_rate = dropped as f64 / stats.packets_received.load(Ordering::Relaxed) as f64;

    println!(
        "Lossy network test: sent={}, received={}, dropped={}, loss_rate={:.1}%",
        total_packets,
        received,
        dropped,
        loss_rate * 100.0
    );

    // Verify some packets were dropped (with 20% loss, expect significant drops)
    assert!(dropped > 0, "Should have some dropped packets");
    // Loss rate should be roughly around the configured rate (with variance)
    assert!(
        loss_rate > 0.05 && loss_rate < 0.50,
        "Loss rate should be roughly around 20%, got {:.1}%",
        loss_rate * 100.0
    );

    let _ = shutdown_tx.send(());
}

// ============================================================================
// Burst Loss Tests
// ============================================================================

#[tokio::test]
async fn test_burst_loss() {
    let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);

    let server = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let server_addr = server.local_addr().unwrap();

    tokio::spawn(async move {
        let mut buf = vec![0u8; 65536];
        let mut shutdown = shutdown_rx;
        loop {
            tokio::select! {
                result = server.recv_from(&mut buf) => {
                    if let Ok((len, addr)) = result {
                        let _ = server.send_to(&buf[..len], addr).await;
                    }
                }
                _ = shutdown.recv() => break,
            }
        }
    });

    let proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::burst_loss(),
    )
    .await
    .unwrap();

    let proxy_addr = proxy.local_addr().unwrap();
    let stats = proxy.stats();

    tokio::spawn(async move {
        proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Send packets and track consecutive losses
    let mut consecutive_losses = 0;
    let mut max_consecutive_losses = 0;
    let mut last_received = true;

    for i in 0..200 {
        let msg = format!("burst test {}", i);
        client.send_to(msg.as_bytes(), proxy_addr).await.unwrap();

        let mut buf = [0u8; 1024];
        match tokio::time::timeout(Duration::from_millis(100), client.recv_from(&mut buf)).await {
            Ok(Ok(_)) => {
                max_consecutive_losses = max_consecutive_losses.max(consecutive_losses);
                consecutive_losses = 0;
                last_received = true;
            }
            _ => {
                consecutive_losses += 1;
                last_received = false;
            }
        }
    }

    max_consecutive_losses = max_consecutive_losses.max(consecutive_losses);

    println!(
        "Burst loss test: max consecutive losses = {}",
        max_consecutive_losses
    );
    println!(
        "Stats: dropped = {}",
        stats.packets_dropped.load(Ordering::Relaxed)
    );

    // With burst_loss_length=5, we should see some bursts
    // Note: Due to probabilistic nature, we check for any burst behavior
    assert!(
        stats.packets_dropped.load(Ordering::Relaxed) > 0,
        "Should have some drops"
    );

    let _ = shutdown_tx.send(());
}

// ============================================================================
// Jitter Tests
// ============================================================================

#[tokio::test]
async fn test_high_jitter() {
    let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);

    let server = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let server_addr = server.local_addr().unwrap();

    tokio::spawn(async move {
        let mut buf = vec![0u8; 65536];
        let mut shutdown = shutdown_rx;
        loop {
            tokio::select! {
                result = server.recv_from(&mut buf) => {
                    if let Ok((len, addr)) = result {
                        let _ = server.send_to(&buf[..len], addr).await;
                    }
                }
                _ = shutdown.recv() => break,
            }
        }
    });

    // High jitter: base 50ms, jitter up to 100ms
    let proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig {
            latency_ms: 50,
            jitter_ms: 100,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let proxy_addr = proxy.local_addr().unwrap();

    tokio::spawn(async move {
        proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Measure RTT variance
    let mut rtts: Vec<Duration> = Vec::new();

    for i in 0..20 {
        let start = Instant::now();
        let msg = format!("jitter test {}", i);
        client.send_to(msg.as_bytes(), proxy_addr).await.unwrap();

        let mut buf = [0u8; 1024];
        if let Ok(Ok(_)) =
            tokio::time::timeout(Duration::from_secs(2), client.recv_from(&mut buf)).await
        {
            rtts.push(start.elapsed());
        }
    }

    assert!(
        !rtts.is_empty(),
        "Should have some successful RTT measurements"
    );

    let min_rtt = rtts.iter().min().unwrap();
    let max_rtt = rtts.iter().max().unwrap();
    let variance = *max_rtt - *min_rtt;

    println!(
        "Jitter test: min={:?}, max={:?}, variance={:?}",
        min_rtt, max_rtt, variance
    );

    // With 100ms jitter, we expect significant variance in RTT
    // Each direction can have up to 100ms jitter, so variance could be up to ~200ms
    assert!(
        variance > Duration::from_millis(20),
        "Should have measurable jitter variance, got {:?}",
        variance
    );

    let _ = shutdown_tx.send(());
}

// ============================================================================
// Packet Reordering Tests
// ============================================================================

#[tokio::test]
async fn test_packet_reordering() {
    let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);

    let server = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let server_addr = server.local_addr().unwrap();

    // Server echoes back with sequence number
    tokio::spawn(async move {
        let mut buf = vec![0u8; 65536];
        let mut shutdown = shutdown_rx;
        loop {
            tokio::select! {
                result = server.recv_from(&mut buf) => {
                    if let Ok((len, addr)) = result {
                        // Echo back immediately
                        let _ = server.send_to(&buf[..len], addr).await;
                    }
                }
                _ = shutdown.recv() => break,
            }
        }
    });

    let proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig {
            latency_ms: 10,
            reorder_rate: 0.3, // 30% reordering
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let proxy_addr = proxy.local_addr().unwrap();
    let stats = proxy.stats();

    tokio::spawn(async move {
        proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Send numbered packets
    for i in 0..50u32 {
        let msg = i.to_be_bytes();
        client.send_to(&msg, proxy_addr).await.unwrap();
    }

    // Receive and track order
    let mut received: Vec<u32> = Vec::new();
    let mut buf = [0u8; 4];

    for _ in 0..100 {
        match tokio::time::timeout(Duration::from_millis(200), client.recv_from(&mut buf)).await {
            Ok(Ok((4, _))) => {
                let seq = u32::from_be_bytes(buf);
                received.push(seq);
            }
            _ => break,
        }
    }

    // Check for out-of-order packets
    let mut out_of_order = 0;
    for i in 1..received.len() {
        if received[i] < received[i - 1] {
            out_of_order += 1;
        }
    }

    println!(
        "Reordering test: received={}, out_of_order={}, reordered_by_proxy={}",
        received.len(),
        out_of_order,
        stats.packets_reordered.load(Ordering::Relaxed)
    );

    // With 30% reordering rate, expect some out-of-order
    assert!(
        stats.packets_reordered.load(Ordering::Relaxed) > 0,
        "Proxy should reorder some packets"
    );

    let _ = shutdown_tx.send(());
}

// ============================================================================
// Bandwidth Throttling Tests
// ============================================================================

#[tokio::test]
async fn test_bandwidth_throttling() {
    let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);

    let server = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let server_addr = server.local_addr().unwrap();

    tokio::spawn(async move {
        let mut buf = vec![0u8; 65536];
        let mut shutdown = shutdown_rx;
        loop {
            tokio::select! {
                result = server.recv_from(&mut buf) => {
                    if let Ok((len, addr)) = result {
                        let _ = server.send_to(&buf[..len], addr).await;
                    }
                }
                _ = shutdown.recv() => break,
            }
        }
    });

    // Limit to 10KB/s
    let proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::throttled(10_000), // 10KB/s
    )
    .await
    .unwrap();

    let proxy_addr = proxy.local_addr().unwrap();

    tokio::spawn(async move {
        proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    // Send 5KB of data
    let data = vec![0u8; 1000];
    let start = Instant::now();

    for _ in 0..5 {
        client.send_to(&data, proxy_addr).await.unwrap();
    }

    // Receive all responses
    let mut received = 0;
    let mut buf = [0u8; 2000];
    for _ in 0..5 {
        if let Ok(Ok((len, _))) =
            tokio::time::timeout(Duration::from_secs(5), client.recv_from(&mut buf)).await
        {
            received += len;
        }
    }

    let elapsed = start.elapsed();

    println!(
        "Bandwidth test: received {} bytes in {:?}",
        received, elapsed
    );

    // At 10KB/s, 5KB should take about 500ms each way = ~1s total
    // With some tolerance for setup and processing
    assert!(
        elapsed > Duration::from_millis(200),
        "Bandwidth limiting should slow down transfer, took {:?}",
        elapsed
    );

    let _ = shutdown_tx.send(());
}

// ============================================================================
// Mobile Network Simulation Tests
// ============================================================================

#[tokio::test]
async fn test_mobile_network_conditions() {
    let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);

    let server = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let server_addr = server.local_addr().unwrap();

    tokio::spawn(async move {
        let mut buf = vec![0u8; 65536];
        let mut shutdown = shutdown_rx;
        loop {
            tokio::select! {
                result = server.recv_from(&mut buf) => {
                    if let Ok((len, addr)) = result {
                        let _ = server.send_to(&buf[..len], addr).await;
                    }
                }
                _ = shutdown.recv() => break,
            }
        }
    });

    let proxy = ImpairmentProxy::new(
        "127.0.0.1:0".parse().unwrap(),
        server_addr,
        ImpairmentConfig::mobile(),
    )
    .await
    .unwrap();

    let proxy_addr = proxy.local_addr().unwrap();
    let stats = proxy.stats();

    tokio::spawn(async move {
        proxy.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();

    let mut successful = 0;
    let mut rtts: Vec<Duration> = Vec::new();

    for i in 0..50 {
        let start = Instant::now();
        let msg = format!("mobile test {}", i);
        client.send_to(msg.as_bytes(), proxy_addr).await.unwrap();

        let mut buf = [0u8; 1024];
        match tokio::time::timeout(Duration::from_secs(2), client.recv_from(&mut buf)).await {
            Ok(Ok(_)) => {
                rtts.push(start.elapsed());
                successful += 1;
            }
            _ => {}
        }
    }

    let dropped = stats.packets_dropped.load(Ordering::Relaxed);
    let duplicated = stats.packets_duplicated.load(Ordering::Relaxed);
    let reordered = stats.packets_reordered.load(Ordering::Relaxed);

    println!("Mobile network simulation:");
    println!("  Successful: {}/50", successful);
    println!("  Dropped: {}", dropped);
    println!("  Duplicated: {}", duplicated);
    println!("  Reordered: {}", reordered);

    if !rtts.is_empty() {
        let min_rtt = rtts.iter().min().unwrap();
        let max_rtt = rtts.iter().max().unwrap();
        let avg_rtt = rtts.iter().sum::<Duration>() / rtts.len() as u32;
        println!(
            "  RTT: min={:?}, max={:?}, avg={:?}",
            min_rtt, max_rtt, avg_rtt
        );

        // Mobile config has high jitter, so variance should be significant
        let variance = *max_rtt - *min_rtt;
        assert!(
            variance > Duration::from_millis(50),
            "Mobile network should have high RTT variance"
        );
    }

    // Should have some impairments applied
    assert!(
        dropped > 0 || duplicated > 0 || reordered > 0,
        "Mobile simulation should apply some impairments"
    );

    let _ = shutdown_tx.send(());
}
