//! Physical Multi-Path Tests
//!
//! These tests validate triglav functionality using REAL network interfaces.
//! Unlike the simulated e2e tests, these use actual interface IPs and test
//! real network behavior including failover, aggregation, and latency.
//!
//! Requirements:
//! - At least 2 active network interfaces with IP addresses
//! - A remote server to test against (can be localhost with different IPs)
//! - Root/sudo may be required for some network impairment tests
//!
//! Run with: cargo test --test physical_multipath -- --nocapture --ignored

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::net::UdpSocket;
use tokio::sync::broadcast;
use tokio::time::timeout;

use triglav::crypto::{KeyPair, PublicKey};
use triglav::error::Result;
use triglav::multipath::{MultipathConfig, MultipathManager, UplinkConfig};
use triglav::transport::TransportProtocol;
use triglav::types::UplinkId;

/// Discovers active network interfaces with their IP addresses.
/// Returns a map of interface name -> IP address.
fn discover_interfaces() -> HashMap<String, IpAddr> {
    let mut interfaces = HashMap::new();

    // Use ifconfig to get interface IPs (macOS compatible)
    let output = Command::new("ifconfig")
        .output()
        .expect("Failed to run ifconfig");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut current_iface: Option<String> = None;

    for line in stdout.lines() {
        // Interface line starts without whitespace
        if !line.starts_with('\t') && !line.starts_with(' ') && line.contains(':') {
            let parts: Vec<&str> = line.split(':').collect();
            if !parts.is_empty() {
                current_iface = Some(parts[0].to_string());
            }
        }

        // Look for inet (IPv4) lines
        if let Some(ref iface) = current_iface {
            let trimmed = line.trim();
            if trimmed.starts_with("inet ") && !trimmed.contains("127.0.0.1") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(ip) = parts[1].parse::<Ipv4Addr>() {
                        // Skip link-local and loopback
                        if !ip.is_loopback() && !ip.is_link_local() {
                            interfaces.insert(iface.clone(), IpAddr::V4(ip));
                        }
                    }
                }
            }
        }
    }

    interfaces
}

/// Gets the primary (default route) interface IP.
fn get_primary_interface_ip() -> Option<IpAddr> {
    let output = Command::new("route")
        .args(["-n", "get", "default"])
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("interface:") {
            let iface = trimmed.split(':').nth(1)?.trim();
            let interfaces = discover_interfaces();
            return interfaces.get(iface).copied();
        }
    }
    None
}

/// Test server that tracks which source IPs it receives traffic from.
struct PhysicalTestServer {
    socket: Arc<UdpSocket>,
    addr: SocketAddr,
    keypair: KeyPair,
    /// Tracks unique source addresses seen
    source_addresses: Arc<DashMap<SocketAddr, AtomicU64>>,
    /// Total packets received
    total_packets: Arc<AtomicU64>,
    /// Per-source packet data for verification
    received_data: Arc<DashMap<SocketAddr, Vec<Vec<u8>>>>,
    shutdown: broadcast::Sender<()>,
    running: Arc<AtomicBool>,
}

impl PhysicalTestServer {
    async fn new(bind_addr: SocketAddr) -> Result<Self> {
        let socket = UdpSocket::bind(bind_addr).await?;
        let addr = socket.local_addr()?;
        let (shutdown, _) = broadcast::channel(1);

        Ok(Self {
            socket: Arc::new(socket),
            addr,
            keypair: KeyPair::generate(),
            source_addresses: Arc::new(DashMap::new()),
            total_packets: Arc::new(AtomicU64::new(0)),
            received_data: Arc::new(DashMap::new()),
            shutdown,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    fn addr(&self) -> SocketAddr {
        self.addr
    }

    fn public_key(&self) -> &PublicKey {
        &self.keypair.public
    }

    fn unique_sources(&self) -> usize {
        self.source_addresses.len()
    }

    fn packets_from(&self, addr: &SocketAddr) -> u64 {
        self.source_addresses
            .get(addr)
            .map(|v| v.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    fn total_packets(&self) -> u64 {
        self.total_packets.load(Ordering::Relaxed)
    }

    fn source_addresses(&self) -> Vec<SocketAddr> {
        self.source_addresses
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }

    async fn run(&self) -> Result<()> {
        self.running.store(true, Ordering::SeqCst);
        let mut shutdown_rx = self.shutdown.subscribe();
        let mut buf = vec![0u8; 65535];

        loop {
            tokio::select! {
                result = self.socket.recv_from(&mut buf) => {
                    match result {
                        Ok((len, src_addr)) => {
                            // Track source address
                            self.source_addresses
                                .entry(src_addr)
                                .or_insert_with(|| AtomicU64::new(0))
                                .fetch_add(1, Ordering::Relaxed);

                            self.total_packets.fetch_add(1, Ordering::Relaxed);

                            // Store received data
                            self.received_data
                                .entry(src_addr)
                                .or_insert_with(Vec::new)
                                .push(buf[..len].to_vec());

                            // Echo back with source tracking prefix
                            let mut response = format!("FROM:{}", src_addr).into_bytes();
                            response.extend_from_slice(&buf[..len]);
                            let _ = self.socket.send_to(&response, src_addr).await;
                        }
                        Err(e) => {
                            eprintln!("Server recv error: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }

        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn shutdown(&self) {
        let _ = self.shutdown.send(());
    }
}

/// Test results structure for reporting.
#[derive(Debug, Default)]
struct PhysicalTestResults {
    interfaces_used: Vec<String>,
    packets_sent: u64,
    packets_received: u64,
    unique_paths: usize,
    latencies_ms: Vec<f64>,
    throughput_mbps: f64,
    failover_time_ms: Option<f64>,
    errors: Vec<String>,
}

impl PhysicalTestResults {
    fn avg_latency_ms(&self) -> f64 {
        if self.latencies_ms.is_empty() {
            0.0
        } else {
            self.latencies_ms.iter().sum::<f64>() / self.latencies_ms.len() as f64
        }
    }

    fn print_report(&self) {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║              PHYSICAL TEST RESULTS                       ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║ Interfaces Used: {:40} ║", self.interfaces_used.join(", "));
        println!("║ Packets Sent:    {:40} ║", self.packets_sent);
        println!("║ Packets Received:{:40} ║", self.packets_received);
        println!("║ Unique Paths:    {:40} ║", self.unique_paths);
        println!(
            "║ Avg Latency:     {:37.2} ms ║",
            self.avg_latency_ms()
        );
        println!("║ Throughput:      {:34.2} Mbps ║", self.throughput_mbps);
        if let Some(failover) = self.failover_time_ms {
            println!("║ Failover Time:   {:37.2} ms ║", failover);
        }
        println!("╚══════════════════════════════════════════════════════════╝");

        if !self.errors.is_empty() {
            println!("\nErrors:");
            for err in &self.errors {
                println!("  - {}", err);
            }
        }
    }
}

// ============================================================================
// Physical Tests (marked as ignored so they don't run in CI)
// ============================================================================

/// Test 1: Verify we can discover multiple network interfaces.
#[tokio::test]
#[ignore]
async fn test_interface_discovery() {
    println!("\n=== Physical Test: Interface Discovery ===\n");

    let interfaces = discover_interfaces();

    println!("Discovered {} interfaces:", interfaces.len());
    for (name, ip) in &interfaces {
        println!("  {} -> {}", name, ip);
    }

    if let Some(primary) = get_primary_interface_ip() {
        println!("\nPrimary interface IP: {}", primary);
    }

    assert!(
        !interfaces.is_empty(),
        "Should discover at least one interface"
    );
    println!("\n[PASS] Interface discovery successful");
}

/// Test 2: Basic multi-interface connectivity test.
/// Sends traffic from multiple local IPs to verify path diversity.
#[tokio::test]
#[ignore]
async fn test_multi_interface_connectivity() {
    println!("\n=== Physical Test: Multi-Interface Connectivity ===\n");

    let interfaces = discover_interfaces();
    if interfaces.len() < 2 {
        println!("[SKIP] Need at least 2 interfaces for this test");
        println!("       Found: {:?}", interfaces.keys().collect::<Vec<_>>());
        return;
    }

    // Start test server - bind to 0.0.0.0 to accept from any interface
    let server = PhysicalTestServer::new("0.0.0.0:0".parse().unwrap())
        .await
        .expect("Failed to create server");

    // Use 127.0.0.1 as destination (server accepts on all interfaces)
    let server_port = server.addr().port();
    let server_addr: SocketAddr = format!("127.0.0.1:{}", server_port).parse().unwrap();
    println!("Test server listening on port {}", server_port);

    // Run server in background
    let server_arc = Arc::new(server);
    let server_clone = Arc::clone(&server_arc);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send packets from each interface
    let mut results = PhysicalTestResults::default();
    let mut packets_sent = 0u64;

    for (iface_name, ip) in &interfaces {
        results.interfaces_used.push(iface_name.clone());

        let local_addr = SocketAddr::new(*ip, 0);
        match UdpSocket::bind(local_addr).await {
            Ok(socket) => {
                println!("Sending from {} ({})", iface_name, ip);

                for i in 0..10 {
                    let msg = format!("TEST:{}:{}", iface_name, i);
                    let start = Instant::now();

                    if socket.send_to(msg.as_bytes(), server_addr).await.is_ok() {
                        packets_sent += 1;

                        // Wait for echo
                        let mut buf = vec![0u8; 1024];
                        match timeout(Duration::from_secs(1), socket.recv_from(&mut buf)).await {
                            Ok(Ok((_len, _))) => {
                                let latency = start.elapsed().as_secs_f64() * 1000.0;
                                results.latencies_ms.push(latency);
                                results.packets_received += 1;
                            }
                            _ => {
                                results.errors.push(format!("Timeout on {}", iface_name));
                            }
                        }
                    }
                }
            }
            Err(e) => {
                results
                    .errors
                    .push(format!("Failed to bind {}: {}", iface_name, e));
            }
        }
    }

    results.packets_sent = packets_sent;
    results.unique_paths = server_arc.unique_sources();

    // Cleanup
    server_arc.shutdown();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Report
    results.print_report();

    // Assertions
    assert!(
        results.unique_paths >= 2,
        "Should see traffic from at least 2 different source IPs"
    );
    assert!(
        results.packets_received > 0,
        "Should receive at least some packets"
    );

    println!("\n[PASS] Multi-interface connectivity verified");
}

/// Test 3: Multipath Manager with real interfaces.
#[tokio::test]
#[ignore]
async fn test_multipath_manager_real_interfaces() {
    println!("\n=== Physical Test: MultipathManager with Real Interfaces ===\n");

    let interfaces = discover_interfaces();
    if interfaces.is_empty() {
        println!("[SKIP] No interfaces discovered");
        return;
    }

    // Start test server
    let server = PhysicalTestServer::new("0.0.0.0:0".parse().unwrap())
        .await
        .expect("Failed to create server");

    let server_port = server.addr().port();
    let server_addr: SocketAddr = format!("127.0.0.1:{}", server_port).parse().unwrap();
    let server_pubkey = server.public_key().clone();
    println!("Server: {} (pubkey: {:?})", server_addr, server_pubkey);

    let server_arc = Arc::new(server);
    let server_clone = Arc::clone(&server_arc);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create MultipathManager with real interface configs
    let config = MultipathConfig::default();
    let keypair = KeyPair::generate();

    let manager = MultipathManager::new(config, keypair);

    // Add uplinks for each discovered interface
    for (idx, (name, ip)) in interfaces.iter().enumerate() {
        let uplink_config = UplinkConfig {
            id: UplinkId::new(format!("uplink-{}", idx)),
            interface: Some(name.clone()),
            local_addr: Some(SocketAddr::new(*ip, 0)),
            remote_addr: server_addr,
            protocol: TransportProtocol::Udp,
            weight: 100,
            ..Default::default()
        };

        match manager.add_uplink(uplink_config) {
            Ok(id) => println!("Added uplink {} -> {}", name, id),
            Err(e) => println!("Failed to add uplink {}: {}", name, e),
        }
    }

    let uplinks = manager.uplinks();
    println!("\nConfigured {} uplinks", uplinks.len());

    // Cleanup
    server_arc.shutdown();

    assert!(!uplinks.is_empty(), "Should have at least one uplink");
    println!("\n[PASS] MultipathManager configured with real interfaces");
}

/// Test 4: Bandwidth measurement across interfaces.
#[tokio::test]
#[ignore]
async fn test_bandwidth_measurement() {
    println!("\n=== Physical Test: Bandwidth Measurement ===\n");

    let interfaces = discover_interfaces();
    if interfaces.is_empty() {
        println!("[SKIP] No interfaces discovered");
        return;
    }

    // Use only first interface for this test
    let (iface_name, ip) = interfaces.iter().next().unwrap();
    println!("Testing bandwidth on {} ({})", iface_name, ip);

    // Start test server
    let server = PhysicalTestServer::new("0.0.0.0:0".parse().unwrap())
        .await
        .expect("Failed to create server");

    let server_port = server.addr().port();
    let server_addr: SocketAddr = format!("127.0.0.1:{}", server_port).parse().unwrap();
    let server_arc = Arc::new(server);
    let server_clone = Arc::clone(&server_arc);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create socket bound to interface
    let local_addr = SocketAddr::new(*ip, 0);
    let socket = UdpSocket::bind(local_addr)
        .await
        .expect("Failed to bind socket");

    // Send burst of packets
    let packet_size = 1400; // Typical MTU-safe size
    let packet_count = 1000;
    let payload = vec![0xABu8; packet_size];

    let start = Instant::now();
    let mut sent = 0u64;

    for _ in 0..packet_count {
        if socket.send_to(&payload, server_addr).await.is_ok() {
            sent += 1;
        }
    }

    let elapsed = start.elapsed();
    let bytes_sent = sent * packet_size as u64;
    let throughput_mbps = (bytes_sent as f64 * 8.0) / elapsed.as_secs_f64() / 1_000_000.0;

    println!("\nBandwidth Test Results:");
    println!("  Packets sent: {}", sent);
    println!("  Bytes sent:   {} KB", bytes_sent / 1024);
    println!("  Duration:     {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Throughput:   {:.2} Mbps", throughput_mbps);

    // Cleanup
    server_arc.shutdown();

    println!("\n[PASS] Bandwidth measurement complete");
}

/// Test 5: Simulated failover test (requires manual interface control).
/// This test sends traffic, waits for user to disable an interface,
/// and measures failover time.
#[tokio::test]
#[ignore]
async fn test_manual_failover() {
    println!("\n=== Physical Test: Manual Failover ===\n");
    println!("This test requires manual intervention.");
    println!("When prompted, disable one of your network interfaces.");
    println!("(e.g., turn off WiFi or unplug Ethernet)\n");

    let interfaces = discover_interfaces();
    if interfaces.len() < 2 {
        println!("[SKIP] Need at least 2 interfaces for failover test");
        return;
    }

    println!("Available interfaces:");
    for (name, ip) in &interfaces {
        println!("  {} -> {}", name, ip);
    }

    // Start test server
    let server = PhysicalTestServer::new("0.0.0.0:0".parse().unwrap())
        .await
        .expect("Failed to create server");

    let server_port = server.addr().port();
    let server_addr: SocketAddr = format!("127.0.0.1:{}", server_port).parse().unwrap();
    let server_arc = Arc::new(server);
    let server_clone = Arc::clone(&server_arc);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create sockets for all interfaces
    let mut sockets = Vec::new();
    for (name, ip) in &interfaces {
        let local_addr = SocketAddr::new(*ip, 0);
        if let Ok(socket) = UdpSocket::bind(local_addr).await {
            sockets.push((name.clone(), socket));
        }
    }

    println!("\nSending test traffic on {} interfaces...", sockets.len());
    println!("Press Ctrl+C when ready to start failover test.\n");

    // Continuous send loop with round-robin
    let mut idx = 0;
    let mut last_success = Instant::now();
    let mut failover_detected = false;
    let mut failover_time = None;

    for iteration in 0..100 {
        let (name, socket) = &sockets[idx % sockets.len()];
        let msg = format!("ITER:{}", iteration);

        let result = socket.send_to(msg.as_bytes(), server_addr).await;

        match result {
            Ok(_) => {
                // Try to receive response
                let mut buf = vec![0u8; 1024];
                match timeout(Duration::from_millis(500), socket.recv_from(&mut buf)).await {
                    Ok(Ok(_)) => {
                        if failover_detected && failover_time.is_none() {
                            let recovery = last_success.elapsed().as_secs_f64() * 1000.0;
                            failover_time = Some(recovery);
                            println!("Recovered after {:.2} ms", recovery);
                        }
                        last_success = Instant::now();
                        print!(".");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                    Ok(Err(_)) | Err(_) => {
                        if !failover_detected {
                            println!("\nInterface {} appears down, trying next...", name);
                            failover_detected = true;
                        }
                    }
                }
            }
            Err(_) => {
                if !failover_detected {
                    println!("\nInterface {} failed, failing over...", name);
                    failover_detected = true;
                }
            }
        }

        idx += 1;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Cleanup
    server_arc.shutdown();

    println!("\n\nFailover Test Results:");
    if let Some(time) = failover_time {
        println!("  Failover detected and recovered in {:.2} ms", time);
    } else if failover_detected {
        println!("  Failover detected but not recovered during test");
    } else {
        println!("  No failover event detected");
    }

    println!("\n[DONE] Manual failover test complete");
}

/// Test 6: Latency distribution across interfaces.
#[tokio::test]
#[ignore]
async fn test_latency_distribution() {
    println!("\n=== Physical Test: Latency Distribution ===\n");

    let interfaces = discover_interfaces();
    if interfaces.is_empty() {
        println!("[SKIP] No interfaces discovered");
        return;
    }

    // Start test server
    let server = PhysicalTestServer::new("0.0.0.0:0".parse().unwrap())
        .await
        .expect("Failed to create server");

    let server_port = server.addr().port();
    let server_addr: SocketAddr = format!("127.0.0.1:{}", server_port).parse().unwrap();
    let server_arc = Arc::new(server);
    let server_clone = Arc::clone(&server_arc);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    println!("Measuring latency for each interface (100 samples each):\n");

    let mut results: HashMap<String, Vec<f64>> = HashMap::new();

    for (name, ip) in &interfaces {
        let local_addr = SocketAddr::new(*ip, 0);
        if let Ok(socket) = UdpSocket::bind(local_addr).await {
            let mut latencies = Vec::new();

            for i in 0..100 {
                let msg = format!("PING:{}", i);
                let start = Instant::now();

                if socket.send_to(msg.as_bytes(), server_addr).await.is_ok() {
                    let mut buf = vec![0u8; 1024];
                    if let Ok(Ok(_)) =
                        timeout(Duration::from_secs(1), socket.recv_from(&mut buf)).await
                    {
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        latencies.push(latency_ms);
                    }
                }
            }

            if !latencies.is_empty() {
                let min = latencies.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let avg: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;

                // Calculate stddev
                let variance: f64 = latencies.iter().map(|x| (x - avg).powi(2)).sum::<f64>()
                    / latencies.len() as f64;
                let stddev = variance.sqrt();

                println!("{}:", name);
                println!("  Samples:  {}", latencies.len());
                println!("  Min:      {:.2} ms", min);
                println!("  Max:      {:.2} ms", max);
                println!("  Avg:      {:.2} ms", avg);
                println!("  StdDev:   {:.2} ms", stddev);
                println!();

                results.insert(name.clone(), latencies);
            }
        }
    }

    // Cleanup
    server_arc.shutdown();

    assert!(!results.is_empty(), "Should measure latency on at least one interface");
    println!("[PASS] Latency distribution measured");
}

/// Test 7: Concurrent traffic on multiple interfaces.
#[tokio::test]
#[ignore]
async fn test_concurrent_traffic() {
    println!("\n=== Physical Test: Concurrent Traffic ===\n");

    let interfaces = discover_interfaces();
    if interfaces.len() < 2 {
        println!("[SKIP] Need at least 2 interfaces");
        return;
    }

    // Start test server
    let server = PhysicalTestServer::new("0.0.0.0:0".parse().unwrap())
        .await
        .expect("Failed to create server");

    let server_port = server.addr().port();
    let server_addr: SocketAddr = format!("127.0.0.1:{}", server_port).parse().unwrap();
    let server_arc = Arc::new(server);
    let server_clone = Arc::clone(&server_arc);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    println!("Sending concurrent traffic from {} interfaces...\n", interfaces.len());

    // Spawn concurrent senders
    let mut handles = Vec::new();
    let packets_per_interface = 100;

    for (name, ip) in interfaces.iter().take(3) {
        // Limit to 3 interfaces
        let name = name.clone();
        let ip = *ip;
        let server_addr = server_addr;

        let handle = tokio::spawn(async move {
            let local_addr = SocketAddr::new(ip, 0);
            let socket = match UdpSocket::bind(local_addr).await {
                Ok(s) => s,
                Err(_) => return (name, 0u64, 0u64),
            };

            let mut sent = 0u64;
            let mut received = 0u64;

            for i in 0..packets_per_interface {
                let msg = format!("CONC:{}:{}", name, i);
                if socket.send_to(msg.as_bytes(), server_addr).await.is_ok() {
                    sent += 1;

                    let mut buf = vec![0u8; 1024];
                    if let Ok(Ok(_)) =
                        timeout(Duration::from_millis(200), socket.recv_from(&mut buf)).await
                    {
                        received += 1;
                    }
                }
            }

            (name, sent, received)
        });

        handles.push(handle);
    }

    // Collect results
    let mut total_sent = 0u64;
    let mut total_received = 0u64;

    for handle in handles {
        if let Ok((name, sent, received)) = handle.await {
            println!("  {}: sent={}, received={}", name, sent, received);
            total_sent += sent;
            total_received += received;
        }
    }

    let unique_sources = server_arc.unique_sources();

    // Cleanup
    server_arc.shutdown();

    println!("\nConcurrent Traffic Results:");
    println!("  Total sent:     {}", total_sent);
    println!("  Total received: {}", total_received);
    println!("  Unique sources: {}", unique_sources);
    println!(
        "  Success rate:   {:.1}%",
        (total_received as f64 / total_sent as f64) * 100.0
    );

    assert!(
        unique_sources >= 2,
        "Should see concurrent traffic from multiple sources"
    );
    println!("\n[PASS] Concurrent traffic test successful");
}

/// Test 8: External server connectivity test.
/// Tests against a real external endpoint (requires network access).
#[tokio::test]
#[ignore]
async fn test_external_connectivity() {
    println!("\n=== Physical Test: External Connectivity ===\n");

    let interfaces = discover_interfaces();
    if interfaces.is_empty() {
        println!("[SKIP] No interfaces discovered");
        return;
    }

    // Test DNS resolution and HTTP connectivity from each interface
    // Using Cloudflare's DNS as a simple external endpoint
    let external_addr: SocketAddr = "1.1.1.1:53".parse().unwrap();

    println!("Testing external connectivity to {} from each interface:\n", external_addr);

    for (name, ip) in &interfaces {
        let local_addr = SocketAddr::new(*ip, 0);
        match UdpSocket::bind(local_addr).await {
            Ok(socket) => {
                // Simple DNS query packet (query for example.com)
                let dns_query: [u8; 29] = [
                    0x00, 0x01, // Transaction ID
                    0x01, 0x00, // Flags: standard query
                    0x00, 0x01, // Questions: 1
                    0x00, 0x00, // Answers: 0
                    0x00, 0x00, // Authority: 0
                    0x00, 0x00, // Additional: 0
                    0x07, b'e', b'x', b'a', b'm', b'p', b'l', b'e', // example
                    0x03, b'c', b'o', b'm', // com
                    0x00, // null terminator
                    0x00, 0x01, // Type: A
                    0x00, 0x01, // Class: IN
                ];

                let start = Instant::now();
                if socket.send_to(&dns_query, external_addr).await.is_ok() {
                    let mut buf = vec![0u8; 512];
                    match timeout(Duration::from_secs(2), socket.recv_from(&mut buf)).await {
                        Ok(Ok((len, _from))) => {
                            let latency = start.elapsed().as_secs_f64() * 1000.0;
                            println!(
                                "  {} ({}): OK - {} bytes, {:.2} ms",
                                name, ip, len, latency
                            );
                        }
                        Ok(Err(e)) => {
                            println!("  {} ({}): Error - {}", name, ip, e);
                        }
                        Err(_) => {
                            println!("  {} ({}): Timeout", name, ip);
                        }
                    }
                } else {
                    println!("  {} ({}): Send failed", name, ip);
                }
            }
            Err(e) => {
                println!("  {} ({}): Bind failed - {}", name, ip, e);
            }
        }
    }

    println!("\n[DONE] External connectivity test complete");
}
