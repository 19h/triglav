//! TUN tunnel runner - main execution loop.
//!
//! The TunnelRunner ties together the TUN device, NAT, routing,
//! and the multipath manager to provide a complete VPN tunnel.

use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};

use super::device::{TunConfig, TunDevice};
use super::dns::{DnsConfig, DnsInterceptor};
use super::nat::{NatConfig, NatTable};
use super::packet::IpPacket;
use super::routing::{RouteConfig, RouteManager};
use crate::crypto::{KeyPair, PublicKey};
use crate::error::{Error, Result};
use crate::multipath::{MultipathConfig, MultipathManager, UplinkConfig};

/// Complete tunnel configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelConfig {
    /// TUN device configuration.
    #[serde(default)]
    pub tun: TunConfig,

    /// NAT configuration.
    #[serde(default)]
    pub nat: NatConfig,

    /// Routing configuration.
    #[serde(default)]
    pub routing: RouteConfig,

    /// DNS configuration.
    #[serde(default)]
    pub dns: DnsConfig,

    /// Multipath configuration.
    #[serde(default)]
    pub multipath: MultipathConfig,

    /// Read buffer size.
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,

    /// NAT cleanup interval.
    #[serde(default = "default_nat_cleanup_interval", with = "humantime_serde")]
    pub nat_cleanup_interval: Duration,
}

fn default_buffer_size() -> usize {
    65536
}

fn default_nat_cleanup_interval() -> Duration {
    Duration::from_secs(60)
}

impl Default for TunnelConfig {
    fn default() -> Self {
        Self {
            tun: TunConfig::default(),
            nat: NatConfig::default(),
            routing: RouteConfig::default(),
            dns: DnsConfig::default(),
            multipath: MultipathConfig::default(),
            buffer_size: default_buffer_size(),
            nat_cleanup_interval: default_nat_cleanup_interval(),
        }
    }
}

/// Tunnel statistics.
#[derive(Debug, Clone, Default)]
pub struct TunnelStats {
    /// Packets read from TUN device.
    pub tun_packets_read: u64,
    /// Packets written to TUN device.
    pub tun_packets_written: u64,
    /// Bytes read from TUN device.
    pub tun_bytes_read: u64,
    /// Bytes written to TUN device.
    pub tun_bytes_written: u64,
    /// Packets sent through tunnel.
    pub tunnel_packets_sent: u64,
    /// Packets received from tunnel.
    pub tunnel_packets_received: u64,
    /// Bytes sent through tunnel.
    pub tunnel_bytes_sent: u64,
    /// Bytes received from tunnel.
    pub tunnel_bytes_received: u64,
    /// Packets dropped (various reasons).
    pub packets_dropped: u64,
    /// NAT translation errors.
    pub nat_errors: u64,
    /// Current uptime.
    pub uptime_secs: u64,
}

/// Tunnel event types.
#[derive(Debug, Clone)]
pub enum TunnelEvent {
    /// Tunnel started.
    Started,
    /// Tunnel connected.
    Connected,
    /// Tunnel disconnected.
    Disconnected,
    /// Uplink status changed.
    UplinkChanged { id: String, connected: bool },
    /// Error occurred.
    Error(String),
    /// Statistics updated.
    StatsUpdated(TunnelStats),
}

/// TUN tunnel runner.
pub struct TunnelRunner {
    /// Configuration.
    config: TunnelConfig,

    /// TUN device.
    tun: TunDevice,

    /// NAT table.
    nat: Arc<NatTable>,

    /// Route manager.
    routes: RouteManager,

    /// DNS interceptor.
    dns: Arc<DnsInterceptor>,

    /// Multipath manager.
    manager: Arc<MultipathManager>,

    /// Local keypair.
    keypair: KeyPair,

    /// Whether the tunnel is running.
    running: AtomicBool,

    /// Start time.
    started_at: RwLock<Option<Instant>>,

    /// Statistics.
    stats: RwLock<TunnelStats>,

    /// Event sender.
    event_tx: broadcast::Sender<TunnelEvent>,
}

impl TunnelRunner {
    /// Create a new tunnel runner.
    pub fn new(config: TunnelConfig) -> Result<Self> {
        // Create TUN device
        let mut tun = TunDevice::create(config.tun.clone())?;

        // Create NAT table
        let nat = Arc::new(NatTable::new(config.nat.clone()));

        // Create route manager
        let routes = RouteManager::new(config.routing.clone(), tun.name().to_string());

        // Create DNS interceptor
        let dns = Arc::new(DnsInterceptor::new(config.dns.clone()));

        // Create keypair
        let keypair = KeyPair::generate();

        // Create multipath manager
        let manager = Arc::new(MultipathManager::new(
            config.multipath.clone(),
            keypair.clone(),
        ));

        let (event_tx, _) = broadcast::channel(256);

        Ok(Self {
            config,
            tun,
            nat,
            routes,
            dns,
            manager,
            keypair,
            running: AtomicBool::new(false),
            started_at: RwLock::new(None),
            stats: RwLock::new(TunnelStats::default()),
            event_tx,
        })
    }

    /// Get the TUN device name.
    pub fn tun_name(&self) -> &str {
        self.tun.name()
    }

    /// Get the multipath manager.
    pub fn manager(&self) -> &Arc<MultipathManager> {
        &self.manager
    }

    /// Subscribe to tunnel events.
    pub fn subscribe(&self) -> broadcast::Receiver<TunnelEvent> {
        self.event_tx.subscribe()
    }

    /// Get current statistics.
    pub fn stats(&self) -> TunnelStats {
        let mut stats = self.stats.read().clone();
        if let Some(started) = *self.started_at.read() {
            stats.uptime_secs = started.elapsed().as_secs();
        }
        stats
    }

    /// Add an uplink.
    pub fn add_uplink(&self, config: UplinkConfig) -> Result<u16> {
        self.manager.add_uplink(config)
    }

    /// Connect to a remote server.
    pub async fn connect(&self, remote_public: PublicKey) -> Result<()> {
        // Configure TUN device
        self.tun.configure_addresses()?;

        // Connect multipath manager
        self.manager.connect(remote_public).await?;

        let _ = self.event_tx.send(TunnelEvent::Connected);
        Ok(())
    }

    /// Run the tunnel.
    ///
    /// This starts the main packet processing loop and blocks until
    /// the tunnel is stopped.
    pub async fn run(&mut self) -> Result<()> {
        // Bring up TUN interface
        self.tun.up()?;

        // Set up routes
        self.routes.setup()?;

        self.running.store(true, Ordering::SeqCst);
        *self.started_at.write() = Some(Instant::now());

        let _ = self.event_tx.send(TunnelEvent::Started);

        tracing::info!(
            tun = %self.tun.name(),
            "Tunnel started"
        );

        // Start background tasks
        self.start_background_tasks();

        // Run main packet loop
        let result = self.packet_loop().await;

        // Cleanup
        self.running.store(false, Ordering::SeqCst);
        self.routes.teardown()?;

        let _ = self.event_tx.send(TunnelEvent::Disconnected);

        result
    }

    /// Stop the tunnel.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the tunnel is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Main packet processing loop.
    async fn packet_loop(&self) -> Result<()> {
        let tun_handle = self.tun.handle();
        let mut tun_buf = vec![0u8; self.config.buffer_size];
        let mut tunnel_buf = vec![0u8; self.config.buffer_size];

        // Start the receive loop from multipath manager
        let mut recv_rx = self.manager.start_recv_loop();

        // Start maintenance loop
        self.manager.start_maintenance_loop();

        loop {
            if !self.running.load(Ordering::SeqCst) {
                break;
            }

            tokio::select! {
                // Read from TUN -> send through tunnel
                result = tun_handle.read(&mut tun_buf) => {
                    match result {
                        Ok(len) if len > 0 => {
                            self.handle_tun_packet(&mut tun_buf[..len]).await;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::debug!(error = %e, "TUN read error");
                        }
                    }
                }

                // Receive from tunnel -> write to TUN
                Some((data, uplink_id)) = recv_rx.recv() => {
                    self.handle_tunnel_packet(&data).await;
                }

                // Periodic maintenance
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    // Check if we should stop
                    if !self.running.load(Ordering::SeqCst) {
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle a packet read from the TUN device.
    async fn handle_tun_packet(&self, packet: &mut [u8]) {
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.tun_packets_read += 1;
            stats.tun_bytes_read += packet.len() as u64;
        }

        // Parse packet to extract flow info
        let flow = match IpPacket::parse(packet) {
            Ok(p) => p.flow_tuple(),
            Err(e) => {
                tracing::trace!(error = %e, "Failed to parse IP packet from TUN");
                self.stats.write().packets_dropped += 1;
                return;
            }
        };

        // NAT outbound translation
        let mut packet_vec = packet.to_vec();
        if let Err(e) = self.nat.translate_outbound(&mut packet_vec) {
            tracing::trace!(error = %e, "NAT outbound translation failed");
            self.stats.write().nat_errors += 1;
            return;
        }

        // Calculate flow hash for ECMP consistency
        let flow_id = flow.flow_hash();

        // Send through multipath manager
        match self.manager.send_on_flow(Some(flow_id), &packet_vec).await {
            Ok(_) => {
                let mut stats = self.stats.write();
                stats.tunnel_packets_sent += 1;
                stats.tunnel_bytes_sent += packet_vec.len() as u64;
            }
            Err(e) => {
                tracing::debug!(error = %e, "Failed to send packet through tunnel");
                self.stats.write().packets_dropped += 1;
            }
        }
    }

    /// Handle a packet received from the tunnel.
    async fn handle_tunnel_packet(&self, data: &[u8]) {
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.tunnel_packets_received += 1;
            stats.tunnel_bytes_received += data.len() as u64;
        }

        // NAT inbound translation
        let mut packet = data.to_vec();
        if let Err(e) = self.nat.translate_inbound(&mut packet) {
            tracing::trace!(error = %e, "NAT inbound translation failed");
            self.stats.write().nat_errors += 1;
            return;
        }

        // Write to TUN device
        match self.tun.handle().write(&packet).await {
            Ok(len) => {
                let mut stats = self.stats.write();
                stats.tun_packets_written += 1;
                stats.tun_bytes_written += len as u64;
            }
            Err(e) => {
                tracing::debug!(error = %e, "Failed to write packet to TUN");
                self.stats.write().packets_dropped += 1;
            }
        }
    }

    /// Start background maintenance tasks.
    fn start_background_tasks(&self) {
        let nat = Arc::clone(&self.nat);
        let cleanup_interval = self.config.nat_cleanup_interval;
        let running = self.running.load(Ordering::SeqCst);

        // NAT cleanup task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;
                nat.cleanup_expired();
            }
        });

        // DNS server task (if configured)
        if self.config.dns.listen_addr.is_some() {
            let dns = Arc::clone(&self.dns);
            tokio::spawn(async move {
                if let Err(e) = dns.run().await {
                    tracing::error!(error = %e, "DNS interceptor error");
                }
            });
        }
    }
}

impl Drop for TunnelRunner {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        // Route cleanup happens in RouteManager::drop
    }
}

/// Builder for TunnelRunner with fluent API.
pub struct TunnelBuilder {
    config: TunnelConfig,
    uplinks: Vec<UplinkConfig>,
}

impl TunnelBuilder {
    /// Create a new tunnel builder with default config.
    pub fn new() -> Self {
        Self {
            config: TunnelConfig::default(),
            uplinks: Vec::new(),
        }
    }

    /// Set the TUN device name.
    pub fn tun_name(mut self, name: &str) -> Self {
        self.config.tun.name = name.to_string();
        self
    }

    /// Set the tunnel IPv4 address.
    pub fn ipv4(mut self, addr: std::net::Ipv4Addr) -> Self {
        self.config.tun.ipv4_addr = Some(addr);
        self.config.nat.tunnel_ipv4 = addr;
        self
    }

    /// Enable full tunnel (route all traffic).
    pub fn full_tunnel(mut self, enable: bool) -> Self {
        self.config.routing.full_tunnel = enable;
        self
    }

    /// Add a network route to tunnel.
    pub fn route(mut self, network: &str) -> Self {
        self.config.routing.include_routes.push(network.to_string());
        self
    }

    /// Exclude a network from the tunnel.
    pub fn exclude(mut self, network: &str) -> Self {
        self.config.routing.exclude_routes.push(network.to_string());
        self
    }

    /// Add DNS server.
    pub fn dns_server(mut self, server: std::net::SocketAddr) -> Self {
        self.config.dns.upstream_servers.push(server);
        self
    }

    /// Enable local DNS interception.
    pub fn local_dns(mut self, addr: std::net::SocketAddr) -> Self {
        self.config.dns.listen_addr = Some(addr);
        self
    }

    /// Add an uplink configuration.
    pub fn uplink(mut self, config: UplinkConfig) -> Self {
        self.uplinks.push(config);
        self
    }

    /// Set the scheduling strategy.
    pub fn strategy(mut self, strategy: crate::multipath::SchedulingStrategy) -> Self {
        self.config.multipath.scheduler.strategy = strategy;
        self
    }

    /// Build the tunnel runner.
    pub fn build(self) -> Result<TunnelRunner> {
        let mut runner = TunnelRunner::new(self.config)?;

        for uplink in self.uplinks {
            runner.add_uplink(uplink)?;
        }

        Ok(runner)
    }
}

impl Default for TunnelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tunnel_config_default() {
        let config = TunnelConfig::default();
        assert_eq!(config.buffer_size, 65536);
    }

    #[test]
    fn test_tunnel_builder() {
        let builder = TunnelBuilder::new()
            .tun_name("tg0")
            .full_tunnel(true)
            .route("10.0.0.0/8");

        assert_eq!(builder.config.tun.name, "tg0");
        assert!(builder.config.routing.full_tunnel);
        assert!(builder
            .config
            .routing
            .include_routes
            .contains(&"10.0.0.0/8".to_string()));
    }
}
