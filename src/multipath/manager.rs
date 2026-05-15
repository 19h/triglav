//! Multi-path connection manager.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use futures::stream::{FuturesUnordered, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};

use super::aggregator::{AggregationMode, AggregatorConfig, BandwidthAggregator};
use super::throughput::{ThroughputConfig, ThroughputOptimizer};
use super::{calculate_flow_hash, FlowId, PathDiscovery, PathDiscoveryConfig};
use super::{Scheduler, SchedulerConfig, Uplink, UplinkConfig};
use crate::crypto::{KeyPair, NoiseSession, PublicKey};
use crate::error::{Error, Result};
use crate::protocol::{Packet, PacketFlags, PacketType};
use crate::transport::{connect, Transport, TransportConfig};
use crate::tun::IpPacket;
use crate::types::{
    ConnectionId, ConnectionState, SequenceNumber, SessionId, TrafficStats, UplinkHealth, UplinkId,
};
use std::net::IpAddr;

/// Multi-path manager configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipathConfig {
    /// Scheduler configuration.
    #[serde(default)]
    pub scheduler: SchedulerConfig,

    /// Transport configuration.
    #[serde(default)]
    pub transport: TransportConfig,

    /// Throughput optimization configuration.
    #[serde(default)]
    pub throughput: ThroughputConfig,

    /// Maximum number of uplinks.
    #[serde(default = "default_max_uplinks")]
    pub max_uplinks: usize,

    /// Retry count for failed sends.
    #[serde(default = "default_retries")]
    pub send_retries: u32,

    /// Retry delay base.
    #[serde(default = "default_retry_delay", with = "humantime_serde")]
    pub retry_delay: Duration,

    /// Enable automatic uplink discovery.
    #[serde(default)]
    pub auto_discover: bool,

    /// Uplink health check interval.
    #[serde(default = "default_health_interval", with = "humantime_serde")]
    pub health_check_interval: Duration,

    /// Path discovery interval for ECMP path probing.
    /// Set to 0 to disable automatic path discovery.
    #[serde(default = "default_path_discovery_interval", with = "humantime_serde")]
    pub path_discovery_interval: Duration,

    /// Maximum age for discovered path data before cleanup.
    #[serde(default = "default_path_max_age", with = "humantime_serde")]
    pub path_max_age: Duration,

    /// Enable packet deduplication.
    #[serde(default = "default_dedup")]
    pub deduplication: bool,

    /// Deduplication window size.
    #[serde(default = "default_dedup_window")]
    pub dedup_window_size: usize,

    /// Enable ECMP-aware flow hashing for path selection.
    /// When enabled, packets are inspected for IP headers and flows
    /// with the same 5-tuple will use the same uplink (Dublin Traceroute technique).
    #[serde(default)]
    pub ecmp_aware: bool,

    /// Path discovery configuration.
    #[serde(default)]
    pub path_discovery: PathDiscoveryConfig,

    /// Enable throughput optimization.
    #[serde(default = "default_throughput_optimization")]
    pub throughput_optimization: bool,

    /// Bandwidth probe interval for active throughput measurement.
    #[serde(default = "default_bandwidth_probe_interval", with = "humantime_serde")]
    pub bandwidth_probe_interval: Duration,

    /// Bandwidth aggregation mode.
    /// - None: Single uplink per flow (load balancing only)
    /// - Full: Stripe packets across ALL uplinks (true bandwidth aggregation)
    /// - Adaptive: Aggregate only when latency spread is acceptable
    #[serde(default = "default_aggregation_mode")]
    pub aggregation_mode: AggregationMode,

    /// Aggregator configuration for bandwidth aggregation.
    #[serde(default)]
    pub aggregator: AggregatorConfig,

    /// Enable selective redundancy for latency-sensitive or connection-critical packets.
    #[serde(default = "default_hedged_packets")]
    pub hedged_packets: bool,

    /// Maximum copies for a hedged packet, including the primary send.
    #[serde(default = "default_hedged_max_copies")]
    pub hedged_max_copies: usize,

    /// Payloads at or below this size are treated as interactive and hedged.
    #[serde(default = "default_hedged_small_packet_threshold")]
    pub hedged_small_packet_threshold: usize,
}

fn default_throughput_optimization() -> bool {
    true
}
fn default_bandwidth_probe_interval() -> Duration {
    Duration::from_secs(5)
}
fn default_aggregation_mode() -> AggregationMode {
    AggregationMode::None
}
fn default_hedged_packets() -> bool {
    true
}
fn default_hedged_max_copies() -> usize {
    2
}
fn default_hedged_small_packet_threshold() -> usize {
    256
}

fn default_max_uplinks() -> usize {
    16
}
fn default_retries() -> u32 {
    3
}
fn default_retry_delay() -> Duration {
    Duration::from_millis(100)
}
fn default_health_interval() -> Duration {
    Duration::from_secs(1)
}
fn default_path_discovery_interval() -> Duration {
    Duration::from_secs(60)
}
fn default_path_max_age() -> Duration {
    Duration::from_secs(300)
}
fn default_dedup() -> bool {
    true
}
fn default_dedup_window() -> usize {
    1000
}

impl Default for MultipathConfig {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            transport: TransportConfig::default(),
            throughput: ThroughputConfig::default(),
            max_uplinks: default_max_uplinks(),
            send_retries: default_retries(),
            retry_delay: default_retry_delay(),
            auto_discover: false,
            health_check_interval: default_health_interval(),
            path_discovery_interval: default_path_discovery_interval(),
            path_max_age: default_path_max_age(),
            deduplication: default_dedup(),
            dedup_window_size: default_dedup_window(),
            ecmp_aware: false,
            path_discovery: PathDiscoveryConfig::default(),
            throughput_optimization: default_throughput_optimization(),
            bandwidth_probe_interval: default_bandwidth_probe_interval(),
            aggregation_mode: AggregationMode::default(),
            aggregator: AggregatorConfig::default(),
            hedged_packets: default_hedged_packets(),
            hedged_max_copies: default_hedged_max_copies(),
            hedged_small_packet_threshold: default_hedged_small_packet_threshold(),
        }
    }
}

/// Event types for multipath manager.
#[derive(Debug, Clone)]
pub enum MultipathEvent {
    /// Uplink connected.
    UplinkConnected(UplinkId),
    /// Uplink disconnected.
    UplinkDisconnected(UplinkId),
    /// Uplink health changed.
    UplinkHealthChanged(UplinkId, UplinkHealth),
    /// Packet received.
    PacketReceived(Packet),
    /// Send failed on all uplinks.
    SendFailed(u64), // sequence number
    /// All uplinks down.
    AllUplinksDown,
    /// Connection established.
    Connected,
    /// Connection closed.
    Disconnected,
    /// Path discovery completed for a destination.
    PathDiscoveryComplete {
        /// Destination that was probed.
        destination: std::net::SocketAddr,
        /// Number of paths discovered.
        paths_found: usize,
        /// Path diversity score (0.0 - 1.0).
        diversity_score: f64,
    },
    /// Flow bindings were repaired after a path degraded or failed.
    PathRepaired {
        /// Failed/degraded uplink numeric ID.
        failed_uplink: u16,
        /// Replacement uplink numeric ID, if one was available.
        replacement_uplink: Option<u16>,
        /// Number of flow bindings moved or cleared.
        moved_flows: usize,
    },
}

/// Pending packet for retry.
#[derive(Debug)]
struct PendingPacket {
    packet: Packet,
    sent_at: Instant,
    retries: u32,
    uplink_id: u16,
}

enum ReceiveLoopEvent {
    Packet {
        uplink_id: u16,
        uplink_name: UplinkId,
        data: Vec<u8>,
    },
    Stopped {
        uplink_id: u16,
    },
}

/// Sliding window for packet deduplication with O(1) operations.
#[derive(Debug)]
struct DeduplicationWindow {
    /// Set for O(1) duplicate detection.
    seen: HashSet<u64>,
    /// Queue for maintaining insertion order (for cleanup).
    order: VecDeque<u64>,
    /// Maximum window size.
    max_size: usize,
}

impl DeduplicationWindow {
    /// Create a new deduplication window.
    fn new(max_size: usize) -> Self {
        Self {
            seen: HashSet::with_capacity(max_size),
            order: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Check if sequence number is a duplicate. Returns true if duplicate, false if new.
    /// If new, adds the sequence number to the window.
    fn is_duplicate(&mut self, seq: u64) -> bool {
        if self.seen.contains(&seq) {
            return true; // Duplicate
        }

        // New sequence number - add to window
        self.seen.insert(seq);
        self.order.push_back(seq);

        // Cleanup: remove oldest entries if window is full
        while self.order.len() > self.max_size {
            if let Some(old_seq) = self.order.pop_front() {
                self.seen.remove(&old_seq);
            }
        }

        false // Not a duplicate
    }
}

/// Multi-path connection manager.
pub struct MultipathManager {
    /// Configuration.
    config: MultipathConfig,
    /// Session ID.
    session_id: SessionId,
    /// Connection ID.
    connection_id: ConnectionId,
    /// Active uplinks.
    uplinks: DashMap<u16, Arc<Uplink>>,
    /// Uplink ID to numeric ID mapping.
    uplink_ids: DashMap<UplinkId, u16>,
    /// Scheduler.
    scheduler: Scheduler,
    /// Next sequence number.
    next_seq: AtomicU64,
    /// Next uplink numeric ID.
    next_uplink_id: AtomicU16,
    /// Pending packets (for retry).
    pending: DashMap<u64, PendingPacket>,
    /// Deduplication window for received packets.
    dedup_window: RwLock<DeduplicationWindow>,
    /// Aggregate stats.
    stats: RwLock<TrafficStats>,
    /// Event sender.
    event_tx: broadcast::Sender<MultipathEvent>,
    /// State.
    state: RwLock<ConnectionState>,
    /// Local keypair.
    local_keypair: KeyPair,
    /// Path discovery for ECMP-aware routing (Dublin Traceroute technique).
    path_discovery: PathDiscovery,
    /// Connection-to-uplink flow bindings (Dublin Traceroute: flow stickiness).
    /// Maps connection/stream IDs to their assigned uplink for ECMP consistency.
    flow_bindings: DashMap<u64, u16>,
    /// Next flow ID for new connections.
    next_flow_id: AtomicU64,
    /// Throughput optimizer for maximum bandwidth utilization.
    throughput_optimizer: ThroughputOptimizer,
    /// Next probe ID for bandwidth probing.
    next_probe_id: AtomicU64,
    /// Bandwidth aggregator for true multi-path aggregation.
    aggregator: BandwidthAggregator,
}

impl MultipathManager {
    /// Create a new multipath manager.
    pub fn new(config: MultipathConfig, local_keypair: KeyPair) -> Self {
        let (event_tx, _) = broadcast::channel(256);

        let dedup_size = config.dedup_window_size;
        let throughput_optimizer = ThroughputOptimizer::new(config.throughput.clone());
        let aggregator = BandwidthAggregator::new(config.aggregator.clone());

        Self {
            scheduler: Scheduler::new(config.scheduler.clone()),
            path_discovery: PathDiscovery::new(config.path_discovery.clone()),
            throughput_optimizer,
            aggregator,
            config,
            session_id: SessionId::generate(),
            connection_id: ConnectionId::new(),
            uplinks: DashMap::new(),
            uplink_ids: DashMap::new(),
            next_seq: AtomicU64::new(1),
            next_uplink_id: AtomicU16::new(0),
            pending: DashMap::new(),
            dedup_window: RwLock::new(DeduplicationWindow::new(dedup_size)),
            stats: RwLock::new(TrafficStats::default()),
            event_tx,
            state: RwLock::new(ConnectionState::Disconnected),
            local_keypair,
            flow_bindings: DashMap::new(),
            next_flow_id: AtomicU64::new(1),
            next_probe_id: AtomicU64::new(1),
        }
    }

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Get the connection ID.
    pub fn connection_id(&self) -> ConnectionId {
        self.connection_id
    }

    /// Get current state.
    pub fn state(&self) -> ConnectionState {
        *self.state.read()
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> TrafficStats {
        *self.stats.read()
    }

    /// Subscribe to events.
    pub fn subscribe(&self) -> broadcast::Receiver<MultipathEvent> {
        self.event_tx.subscribe()
    }

    /// Get uplink count.
    pub fn uplink_count(&self) -> usize {
        self.uplinks.len()
    }

    /// Get all uplinks.
    pub fn uplinks(&self) -> Vec<Arc<Uplink>> {
        self.uplinks.iter().map(|r| r.value().clone()).collect()
    }

    /// Get uplink by numeric ID.
    pub fn get_uplink(&self, id: u16) -> Option<Arc<Uplink>> {
        self.uplinks.get(&id).map(|r| r.value().clone())
    }

    /// Get uplink by name.
    pub fn get_uplink_by_name(&self, name: &UplinkId) -> Option<Arc<Uplink>> {
        self.uplink_ids
            .get(name)
            .and_then(|id| self.uplinks.get(&*id).map(|r| r.value().clone()))
    }

    /// Add an uplink.
    pub fn add_uplink(&self, config: UplinkConfig) -> Result<u16> {
        if self.uplinks.len() >= self.config.max_uplinks {
            return Err(Error::Config(format!(
                "maximum uplinks ({}) reached",
                self.config.max_uplinks
            )));
        }

        let numeric_id = self.next_uplink_id.fetch_add(1, Ordering::SeqCst);
        let uplink = Arc::new(Uplink::new(config.clone(), numeric_id));

        self.uplinks.insert(numeric_id, uplink);
        self.uplink_ids.insert(config.id, numeric_id);

        // Register with throughput optimizer
        if self.config.throughput_optimization {
            self.throughput_optimizer.register_uplink(numeric_id);
        }

        Ok(numeric_id)
    }

    /// Remove an uplink.
    pub fn remove_uplink(&self, id: u16) -> Option<Arc<Uplink>> {
        if let Some((_, uplink)) = self.uplinks.remove(&id) {
            self.uplink_ids.remove(uplink.id());
            self.clear_flow_bindings_for_uplink(id);
            uplink.set_connection_state(ConnectionState::Disconnected);

            // Unregister from throughput optimizer
            if self.config.throughput_optimization {
                self.throughput_optimizer.unregister_uplink(id);
            }

            let _ = self
                .event_tx
                .send(MultipathEvent::UplinkDisconnected(uplink.id().clone()));
            Some(uplink)
        } else {
            None
        }
    }

    /// Remove all uplinks associated with an interface.
    pub fn remove_uplinks_for_interface(&self, interface: &str) -> Vec<Arc<Uplink>> {
        let ids: Vec<_> = self
            .uplinks
            .iter()
            .filter(|entry| entry.value().config().interface.as_deref() == Some(interface))
            .map(|entry| *entry.key())
            .collect();

        ids.into_iter()
            .filter_map(|id| self.remove_uplink(id))
            .collect()
    }

    /// Add one uplink and immediately connect it if possible.
    pub async fn add_and_connect_uplink(
        &self,
        config: UplinkConfig,
        remote_public: &PublicKey,
    ) -> Result<u16> {
        if let Some(existing_id) = self.uplink_ids.get(&config.id).map(|id| *id) {
            if let Some(existing) = self.get_uplink(existing_id) {
                if existing.config().local_addr == config.local_addr
                    && existing.config().remote_addr == config.remote_addr
                    && existing.config().protocol == config.protocol
                {
                    if existing.state().connection_state != ConnectionState::Connected {
                        if let Err(e) = self.connect_uplink(&existing, remote_public).await {
                            existing.record_failure(&e.to_string());
                            return Err(e);
                        }
                        self.send_fast_probe(existing_id).await;
                    }
                    return Ok(existing_id);
                }

                self.remove_uplink(existing_id);
            }
        }

        let id = self.add_uplink(config)?;
        if let Some(uplink) = self.get_uplink(id) {
            if let Err(e) = self.connect_uplink(&uplink, remote_public).await {
                uplink.record_failure(&e.to_string());
                return Err(e);
            }
            self.send_fast_probe(id).await;
        }

        if self
            .uplinks
            .iter()
            .any(|entry| entry.value().state().connection_state == ConnectionState::Connected)
        {
            *self.state.write() = ConnectionState::Connected;
        }

        Ok(id)
    }

    fn clear_flow_bindings_for_uplink(&self, uplink_id: u16) {
        self.flow_bindings.retain(|_, bound| *bound != uplink_id);
    }

    /// Move flow bindings off a degraded/failed path and quickly probe the replacement.
    async fn repair_uplink_paths(&self, failed_uplink: u16) {
        let replacement = self.select_repair_uplink(failed_uplink);
        let affected_flows: Vec<u64> = self
            .flow_bindings
            .iter()
            .filter(|entry| *entry.value() == failed_uplink)
            .map(|entry| *entry.key())
            .collect();

        if affected_flows.is_empty() {
            if let Some(replacement) = replacement {
                self.send_fast_probe(replacement).await;
            }
            return;
        }

        match replacement {
            Some(replacement) => {
                for flow_id in &affected_flows {
                    self.flow_bindings.insert(*flow_id, replacement);
                }

                self.send_fast_probe(replacement).await;

                tracing::info!(
                    failed_uplink,
                    replacement_uplink = replacement,
                    moved_flows = affected_flows.len(),
                    "Repaired flow bindings after uplink degradation"
                );
            }
            None => {
                for flow_id in &affected_flows {
                    self.flow_bindings.remove(flow_id);
                }

                tracing::warn!(
                    failed_uplink,
                    moved_flows = affected_flows.len(),
                    "Cleared flow bindings after uplink degradation; no replacement path available"
                );
            }
        }

        let _ = self.event_tx.send(MultipathEvent::PathRepaired {
            failed_uplink,
            replacement_uplink: replacement,
            moved_flows: affected_flows.len(),
        });
    }

    fn select_repair_uplink(&self, failed_uplink: u16) -> Option<u16> {
        self.uplinks
            .iter()
            .filter(|entry| *entry.key() != failed_uplink)
            .filter(|entry| entry.value().is_usable())
            .max_by_key(|entry| entry.value().priority_score())
            .map(|entry| *entry.key())
    }

    async fn send_fast_probe(&self, uplink_id: u16) {
        let Some(uplink) = self.get_uplink(uplink_id) else {
            return;
        };
        let Some(transport) = uplink.get_transport() else {
            return;
        };

        let seq = SequenceNumber(self.next_seq.fetch_add(1, Ordering::SeqCst));
        let Ok(ping) = Packet::ping(seq, self.session_id, uplink.numeric_id()) else {
            return;
        };
        let Ok(encoded) = ping.encode() else {
            return;
        };

        if let Err(e) = transport.send(&encoded).await {
            tracing::debug!(
                uplink = %uplink.id(),
                error = %e,
                "Fast path repair probe failed"
            );
            uplink.record_failure(&e.to_string());
        } else {
            tracing::trace!(
                uplink = %uplink.id(),
                "Sent fast path repair probe"
            );
        }
    }

    /// Connect all uplinks to the remote server.
    pub async fn connect(self: &Arc<Self>, remote_public: PublicKey) -> Result<()> {
        *self.state.write() = ConnectionState::Connecting;

        let uplinks: Vec<_> = self.uplinks.iter().map(|r| r.value().clone()).collect();

        if uplinks.is_empty() {
            *self.state.write() = ConnectionState::Failed;
            return Err(Error::NoAvailableUplinks);
        }

        let (result_tx, mut result_rx) = mpsc::channel(uplinks.len());

        for uplink in uplinks {
            let manager = Arc::clone(self);
            let remote_public = remote_public.clone();
            let result_tx = result_tx.clone();

            tokio::spawn(async move {
                let result = manager.connect_uplink(&uplink, &remote_public).await;
                if let Err(e) = &result {
                    tracing::warn!(
                        uplink = %uplink.id(),
                        error = %e,
                        "Failed to connect uplink"
                    );
                    uplink.record_failure(&e.to_string());
                }
                let _ = result_tx.send((uplink.numeric_id(), result)).await;
            });
        }
        drop(result_tx);

        let mut last_error = None;
        while let Some((_uplink_id, result)) = result_rx.recv().await {
            match result {
                Ok(()) => {
                    *self.state.write() = ConnectionState::Connected;
                    let _ = self.event_tx.send(MultipathEvent::Connected);

                    tokio::spawn(async move { while result_rx.recv().await.is_some() {} });

                    return Ok(());
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        *self.state.write() = ConnectionState::Failed;
        Err(last_error.unwrap_or(Error::NoAvailableUplinks))
    }

    /// Connect a single uplink.
    async fn connect_uplink(&self, uplink: &Uplink, remote_public: &PublicKey) -> Result<()> {
        uplink.set_connection_state(ConnectionState::Connecting);

        let config = uplink.config();

        // Create transport
        let transport = connect(
            config.protocol,
            config.remote_addr,
            config.local_addr,
            &self.config.transport,
        )
        .await?;

        // Wrap in Arc for sharing across async operations
        let transport: Arc<dyn Transport> = Arc::from(transport);
        uplink.set_transport(transport.clone());

        // Perform Noise handshake
        uplink.set_connection_state(ConnectionState::Handshaking);

        let mut noise = NoiseSession::new_initiator(&self.local_keypair.secret, remote_public)?;

        // Write handshake message 1 (e, es)
        let msg1 = noise.write_handshake(&[])?;

        // Create handshake packet
        let handshake_packet = Packet::new(
            PacketType::Handshake,
            SequenceNumber(0),
            self.session_id,
            uplink.numeric_id(),
            msg1,
        )?;

        // Send handshake message
        let encoded = handshake_packet.encode()?;
        transport.send(&encoded).await?;

        tracing::debug!(
            uplink = %uplink.id(),
            "Sent handshake message 1"
        );

        // Receive handshake response
        let mut buf = vec![0u8; 65536];
        let len = tokio::time::timeout(
            self.config.transport.connect_timeout,
            transport.recv(&mut buf),
        )
        .await
        .map_err(|_| Error::ConnectionTimeout)??;

        // Decode response packet
        let response_packet = Packet::decode(&buf[..len])?;

        if response_packet.header.packet_type != PacketType::Handshake {
            return Err(Error::Protocol(
                crate::error::ProtocolError::UnexpectedMessage {
                    expected: "Handshake".into(),
                    got: format!("{:?}", response_packet.header.packet_type),
                },
            ));
        }

        // Process handshake message 2
        let _ = noise.read_handshake(&response_packet.payload)?;

        if !noise.is_transport() {
            return Err(Error::Crypto(crate::error::CryptoError::NoiseProtocol(
                "Handshake did not complete".into(),
            )));
        }

        tracing::debug!(
            uplink = %uplink.id(),
            "Handshake complete, noise session ready"
        );

        uplink.set_noise_session(noise);
        uplink.set_connection_state(ConnectionState::Connected);

        // Detect NAT by comparing configured vs actual local address
        // (Dublin Traceroute technique: NAT detection via address comparison)
        self.detect_uplink_nat(uplink, &*transport);

        let _ = self
            .event_tx
            .send(MultipathEvent::UplinkConnected(uplink.id().clone()));

        Ok(())
    }

    /// Detect NAT on an uplink by comparing configured and actual addresses.
    fn detect_uplink_nat(&self, uplink: &Uplink, transport: &dyn Transport) {
        let config = uplink.config();

        // Get actual local address from transport
        let actual_local = transport.local_addr();

        // Compare with configured local address
        let is_natted = match (config.local_addr, actual_local) {
            (Some(configured), Ok(actual)) => {
                // If port changed (ephemeral port assigned), might be NAT
                // If IP is different, definitely NAT
                if configured.ip() != actual.ip() {
                    tracing::info!(
                        uplink = %uplink.id(),
                        configured = %configured,
                        actual = %actual,
                        "NAT detected: IP address differs"
                    );
                    true
                } else if configured.port() != 0 && configured.port() != actual.port() {
                    tracing::debug!(
                        uplink = %uplink.id(),
                        configured_port = configured.port(),
                        actual_port = actual.port(),
                        "Port-only NAT possible"
                    );
                    // Port change alone isn't definitive NAT, could be ephemeral port
                    false
                } else {
                    false
                }
            }
            (None, Ok(actual)) => {
                // No configured address - check if it's a private address
                // which would indicate we're behind NAT
                let is_private = match actual.ip() {
                    IpAddr::V4(ip) => ip.is_private() || ip.is_loopback(),
                    IpAddr::V6(ip) => ip.is_loopback(),
                };
                if is_private {
                    tracing::debug!(
                        uplink = %uplink.id(),
                        local = %actual,
                        "Private address detected, likely behind NAT"
                    );
                }
                is_private
            }
            _ => false,
        };

        // Update uplink NAT state
        if is_natted {
            uplink.update_nat_state(|state| {
                state.set_nat_type(super::nat::NatType::Unknown);
            });
        }
    }

    /// Set NAT state for an uplink from external detection (e.g., STUN).
    ///
    /// Use this when you have external NAT type information from a STUN server
    /// or similar mechanism.
    pub fn set_uplink_nat_state(
        &self,
        uplink_id: u16,
        nat_type: super::nat::NatType,
        external_addr: Option<std::net::SocketAddr>,
    ) {
        if let Some(uplink) = self.get_uplink(uplink_id) {
            uplink.update_nat_state(|state| {
                state.set_nat_type(nat_type);
                if let Some(addr) = external_addr {
                    state.set_external_addr(addr);
                }
            });

            tracing::info!(
                uplink = %uplink.id(),
                nat_type = ?nat_type,
                external_addr = ?external_addr,
                "NAT state updated from external detection"
            );
        }
    }

    /// Send data through the multi-path connection.
    /// For ECMP-aware routing, use `send_on_flow()` instead.
    pub async fn send(&self, data: &[u8]) -> Result<u64> {
        self.send_on_flow(None, data).await
    }

    /// Send data on a specific flow for ECMP path consistency.
    ///
    /// When aggregation mode is enabled (Full or Adaptive), packets are STRIPED
    /// across ALL uplinks proportionally to their bandwidth. This provides true
    /// bandwidth aggregation - a single flow can use the combined bandwidth of
    /// all uplinks.
    ///
    /// When aggregation is disabled (None), the traditional flow-sticky behavior
    /// is used where each flow is bound to a single uplink.
    pub async fn send_on_flow(&self, flow_id: Option<u64>, data: &[u8]) -> Result<u64> {
        // Check if bandwidth aggregation is enabled
        let use_aggregation = match self.config.aggregation_mode {
            AggregationMode::Full => true,
            AggregationMode::Adaptive => {
                // Use aggregation if latency spread is acceptable
                let spread = self.aggregator.latency_spread();
                spread < Duration::from_millis(100)
            }
            AggregationMode::None => false,
        };

        if use_aggregation {
            self.send_aggregated(data).await
        } else {
            self.send_single_path(flow_id, data).await
        }
    }

    /// Send using bandwidth aggregation (packet striping).
    /// Distributes packets across ALL uplinks weighted by their bandwidth.
    async fn send_aggregated(&self, data: &[u8]) -> Result<u64> {
        let uplinks = self.uplinks();

        // Get next stripe (uplink + sequence)
        let (uplink_id, agg_seq) = self
            .aggregator
            .next_stripe(&uplinks)
            .ok_or(Error::NoAvailableUplinks)?;

        // Use aggregator sequence in packet header for reordering
        let seq = SequenceNumber(agg_seq);

        if let Some(uplink) = self.get_uplink(uplink_id) {
            // Apply pacing if enabled
            if self.config.throughput_optimization {
                let pacing_interval = self
                    .throughput_optimizer
                    .pacing_interval(uplink_id, data.len());
                if pacing_interval > Duration::ZERO {
                    tokio::time::sleep(pacing_interval).await;
                }
            }

            match self.send_on_uplink(&uplink, seq, data).await {
                Ok(_) => {
                    // Record send with throughput optimizer
                    if self.config.throughput_optimization {
                        self.throughput_optimizer
                            .record_send(uplink_id, data.len() as u64);
                    }

                    let mut stats = self.stats.write();
                    stats.bytes_sent += data.len() as u64;
                    stats.packets_sent += 1;

                    Ok(agg_seq)
                }
                Err(e) => {
                    uplink.record_failure(&e.to_string());
                    self.repair_uplink_paths(uplink_id).await;

                    // Try fallback to another uplink
                    self.send_aggregated_fallback(data, uplink_id).await
                }
            }
        } else {
            Err(Error::NoAvailableUplinks)
        }
    }

    /// Fallback when primary stripe uplink fails - try next available.
    async fn send_aggregated_fallback(&self, data: &[u8], failed_uplink: u16) -> Result<u64> {
        let uplinks = self.uplinks();

        for uplink in &uplinks {
            if uplink.numeric_id() == failed_uplink {
                continue;
            }
            if !uplink.is_usable() {
                continue;
            }

            let seq = SequenceNumber(self.aggregator.current_seq());

            match self.send_on_uplink(uplink, seq, data).await {
                Ok(_) => {
                    let mut stats = self.stats.write();
                    stats.bytes_sent += data.len() as u64;
                    stats.packets_sent += 1;
                    return Ok(seq.0);
                }
                Err(_) => continue,
            }
        }

        Err(Error::NoAvailableUplinks)
    }

    /// Send using single-path (flow-sticky) mode.
    /// Traditional behavior where each flow uses one uplink.
    async fn send_single_path(&self, flow_id: Option<u64>, data: &[u8]) -> Result<u64> {
        let seq = SequenceNumber(self.next_seq.fetch_add(1, Ordering::SeqCst));
        let data_size = data.len() as u64;

        // Check for existing flow binding first
        let bound_uplink = flow_id.and_then(|fid| self.flow_bindings.get(&fid).map(|r| *r));

        // Create packet
        let uplinks = self.uplinks();

        // If we have a bound uplink that's still usable, use it
        let mut selected = if let Some(uplink_id) = bound_uplink {
            if self.get_uplink(uplink_id).is_some_and(|u| u.is_usable()) {
                vec![uplink_id]
            } else {
                // Bound uplink no longer usable, select new one and update binding
                let new_selection =
                    self.scheduler
                        .select_for_size(&uplinks, flow_id, Some(data_size));
                if let (Some(fid), Some(&new_id)) = (flow_id, new_selection.first()) {
                    self.flow_bindings.insert(fid, new_id);
                }
                new_selection
            }
        } else {
            // No binding - select using size-aware scheduling
            let selection = self
                .scheduler
                .select_for_size(&uplinks, flow_id, Some(data_size));
            if let (Some(fid), Some(&uplink_id)) = (flow_id, selection.first()) {
                self.flow_bindings.insert(fid, uplink_id);
            }
            selection
        };

        let hedge_packet = self.should_hedge_packet(data)
            && self.config.scheduler.strategy != super::SchedulingStrategy::Redundant;
        if hedge_packet {
            self.extend_with_hedged_uplinks(&mut selected);
        }

        if selected.is_empty() {
            return Err(Error::NoAvailableUplinks);
        }

        // Send on selected uplinks
        let mut last_error = None;
        let mut sent = false;
        let send_all_selected =
            self.config.scheduler.strategy == super::SchedulingStrategy::Redundant || hedge_packet;

        for uplink_id in &selected {
            if let Some(uplink) = self.get_uplink(*uplink_id) {
                // Apply pacing if enabled
                if self.config.throughput_optimization {
                    let pacing_interval = self
                        .throughput_optimizer
                        .pacing_interval(*uplink_id, data.len());
                    if pacing_interval > Duration::ZERO {
                        tokio::time::sleep(pacing_interval).await;
                    }
                }

                match self.send_on_uplink(&uplink, seq, data).await {
                    Ok(_) => {
                        sent = true;

                        // Record send with throughput optimizer
                        if self.config.throughput_optimization {
                            self.throughput_optimizer
                                .record_send(*uplink_id, data.len() as u64);
                        }

                        // For normal single-path strategies, stop after first success.
                        if !send_all_selected {
                            break;
                        }
                    }
                    Err(e) => {
                        uplink.record_failure(&e.to_string());
                        self.repair_uplink_paths(*uplink_id).await;

                        // Record MTU failure if applicable
                        if self.config.throughput_optimization
                            && e.to_string().contains("too large")
                        {
                            self.throughput_optimizer.record_mtu_result(
                                *uplink_id,
                                data.len() as u32,
                                false,
                            );
                        }

                        last_error = Some(e);
                    }
                }
            }
        }

        if sent {
            let mut stats = self.stats.write();
            stats.bytes_sent += data.len() as u64;
            stats.packets_sent += 1;
            Ok(seq.0)
        } else if let Some(e) = last_error {
            Err(e)
        } else {
            Err(Error::NoAvailableUplinks)
        }
    }

    fn should_hedge_packet(&self, data: &[u8]) -> bool {
        self.config.hedged_packets
            && is_hedged_packet(data, self.config.hedged_small_packet_threshold)
    }

    fn extend_with_hedged_uplinks(&self, selected: &mut Vec<u16>) {
        let target_copies = self.config.hedged_max_copies.max(1);
        if selected.len() >= target_copies {
            return;
        }

        let mut candidates: Vec<_> = self
            .uplinks()
            .into_iter()
            .filter(|uplink| {
                uplink.is_usable() && uplink.can_send() && !selected.contains(&uplink.numeric_id())
            })
            .collect();

        candidates.sort_by(|a, b| {
            hedge_uplink_score(b)
                .partial_cmp(&hedge_uplink_score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for uplink in candidates {
            if selected.len() >= target_copies {
                break;
            }
            selected.push(uplink.numeric_id());
        }
    }

    /// Send packet on a specific uplink.
    async fn send_on_uplink(
        &self,
        uplink: &Uplink,
        seq: SequenceNumber,
        data: &[u8],
    ) -> Result<()> {
        // Encrypt the payload using the uplink's noise session
        let encrypted_payload = uplink.encrypt(data)?;

        // Create packet with encrypted payload
        let mut packet =
            Packet::data(seq, self.session_id, uplink.numeric_id(), encrypted_payload)?;

        // Mark as encrypted
        packet.set_flag(PacketFlags::ENCRYPTED);

        let encoded = packet.encode()?;

        // Get transport and send
        let transport = uplink.get_transport().ok_or_else(|| {
            Error::Transport(crate::error::TransportError::SendFailed(
                "no transport".into(),
            ))
        })?;

        transport.send(&encoded).await?;
        uplink.record_send(encoded.len());

        tracing::trace!(
            uplink = %uplink.id(),
            seq = seq.0,
            len = data.len(),
            "Sent encrypted packet"
        );

        // Track pending packet for retry (store original data for potential retransmission)
        self.pending.insert(
            seq.0,
            PendingPacket {
                packet: Packet::data(seq, self.session_id, uplink.numeric_id(), data.to_vec())?,
                sent_at: Instant::now(),
                retries: 0,
                uplink_id: uplink.numeric_id(),
            },
        );

        Ok(())
    }

    /// Handle incoming packet.
    /// When aggregation is enabled, packets are passed through the reorder buffer
    /// to handle out-of-order arrival from different uplinks.
    pub fn handle_packet(&self, data: &[u8], from_uplink: u16) -> Result<Option<Vec<u8>>> {
        let packet = Packet::decode(data)?;

        // Verify session
        if packet.header.session_id != self.session_id {
            return Err(Error::Protocol(
                crate::error::ProtocolError::InvalidVersion {
                    expected: 0,
                    got: 1,
                },
            ));
        }

        // Deduplication using O(1) sliding window
        if self.config.deduplication {
            let mut dedup = self.dedup_window.write();
            if dedup.is_duplicate(packet.header.sequence.0) {
                return Ok(None); // Duplicate
            }
        }

        // Update uplink stats
        if let Some(uplink) = self.get_uplink(from_uplink) {
            uplink.record_recv(data.len());
            uplink.record_success();
        }

        // Update aggregate stats
        {
            let mut stats = self.stats.write();
            stats.bytes_received += data.len() as u64;
            stats.packets_received += 1;
        }

        // Handle by packet type
        match packet.header.packet_type {
            PacketType::Data => {
                // Decrypt if encrypted
                let payload = if packet.header.flags.has(PacketFlags::ENCRYPTED) {
                    if let Some(uplink) = self.get_uplink(from_uplink) {
                        uplink.decrypt(&packet.payload)?
                    } else {
                        packet.payload.clone()
                    }
                } else {
                    packet.payload.clone()
                };

                // Check if aggregation is enabled - use reorder buffer
                let use_aggregation =
                    !matches!(self.config.aggregation_mode, AggregationMode::None);

                if use_aggregation {
                    // Pass through reorder buffer for proper sequencing
                    let ready_packets =
                        self.aggregator
                            .receive(packet.header.sequence.0, payload, from_uplink);

                    // Return the first ready packet (others will be picked up by poll)
                    // In practice, we'd want to deliver all ready packets
                    if let Some(first) = ready_packets.into_iter().next() {
                        let decrypted_packet = Packet::data(
                            packet.header.sequence,
                            packet.header.session_id,
                            packet.header.uplink_id,
                            first.clone(),
                        )?;
                        let _ = self
                            .event_tx
                            .send(MultipathEvent::PacketReceived(decrypted_packet));
                        return Ok(Some(first));
                    } else {
                        // Packet buffered, waiting for earlier packets
                        return Ok(None);
                    }
                }

                let decrypted_packet = Packet::data(
                    packet.header.sequence,
                    packet.header.session_id,
                    packet.header.uplink_id,
                    payload.clone(),
                )?;

                let _ = self
                    .event_tx
                    .send(MultipathEvent::PacketReceived(decrypted_packet));
                Ok(Some(payload))
            }
            PacketType::Ack => {
                self.handle_ack(&packet)?;
                Ok(None)
            }
            PacketType::Pong => {
                self.handle_pong(&packet, from_uplink);
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Poll the reorder buffer for packets that are ready due to timeout.
    /// Call this periodically when using bandwidth aggregation.
    pub fn poll_reorder_buffer(&self) -> Vec<Vec<u8>> {
        if matches!(self.config.aggregation_mode, AggregationMode::None) {
            return vec![];
        }
        self.aggregator.poll_timeout()
    }

    /// Get aggregator statistics.
    pub fn aggregator_stats(&self) -> super::aggregator::AggregatorStats {
        self.aggregator.stats()
    }

    /// Get reorder buffer statistics.
    pub fn reorder_stats(&self) -> super::aggregator::ReorderStats {
        self.aggregator.reorder_stats()
    }

    /// Receive data from any uplink.
    /// Returns the decrypted payload and the uplink ID it came from.
    pub async fn recv(&self) -> Result<(Vec<u8>, u16)> {
        let uplinks: Vec<_> = self
            .uplinks
            .iter()
            .filter(|r| r.value().is_usable())
            .map(|r| r.value().clone())
            .collect();

        if uplinks.is_empty() {
            return Err(Error::NoAvailableUplinks);
        }

        let mut receives = FuturesUnordered::new();
        for uplink in uplinks {
            receives.push(Self::recv_once_from_uplink(uplink));
        }

        while let Some((uplink, uplink_id, uplink_name, result)) = receives.next().await {
            match result {
                Ok(data) => {
                    let ack_sequence = self.data_sequence_to_ack(&data)?;
                    let payload = self.handle_packet(&data, uplink_id)?;

                    if let Some(sequence) = ack_sequence {
                        if let Err(e) = self.send_ack(uplink_id, sequence).await {
                            tracing::debug!(
                                uplink = %uplink_name,
                                sequence,
                                error = %e,
                                "Failed to send ACK"
                            );
                        }
                    }

                    if let Some(payload) = payload {
                        return Ok((payload, uplink_id));
                    }

                    if uplink.is_usable() {
                        receives.push(Self::recv_once_from_uplink(uplink));
                    }
                }
                Err(e) => {
                    tracing::debug!(uplink = %uplink_name, error = %e, "Receive error");
                    if let Some(uplink) = self.get_uplink(uplink_id) {
                        uplink.record_failure(&e.to_string());
                    }
                    self.repair_uplink_paths(uplink_id).await;
                }
            }
        }

        Err(Error::NoAvailableUplinks)
    }

    async fn recv_once_from_uplink(
        uplink: Arc<Uplink>,
    ) -> (Arc<Uplink>, u16, UplinkId, Result<Vec<u8>>) {
        let uplink_id = uplink.numeric_id();
        let uplink_name = uplink.id().clone();
        let mut buf = vec![0u8; 65536];
        let result = uplink.recv(&mut buf).await;
        (
            uplink,
            uplink_id,
            uplink_name,
            result.map(|len| buf[..len].to_vec()),
        )
    }

    fn data_sequence_to_ack(&self, data: &[u8]) -> Result<Option<u64>> {
        let packet = Packet::decode(data)?;
        if packet.header.session_id != self.session_id {
            return Err(Error::Protocol(
                crate::error::ProtocolError::InvalidVersion {
                    expected: 0,
                    got: 1,
                },
            ));
        }

        Ok((packet.header.packet_type == PacketType::Data).then_some(packet.header.sequence.0))
    }

    async fn send_ack(&self, uplink_id: u16, acked_sequence: u64) -> Result<()> {
        let uplink = self
            .get_uplink(uplink_id)
            .ok_or(Error::NoAvailableUplinks)?;
        let transport = uplink.get_transport().ok_or_else(|| {
            Error::Transport(crate::error::TransportError::SendFailed(
                "no transport".into(),
            ))
        })?;

        let seq = SequenceNumber(self.next_seq.fetch_add(1, Ordering::SeqCst));
        let packet = Packet::ack(seq, self.session_id, uplink.numeric_id(), &[acked_sequence])?;
        let encoded = packet.encode()?;

        transport.send(&encoded).await?;

        tracing::trace!(
            uplink = %uplink.id(),
            sequence = acked_sequence,
            "Sent ACK"
        );

        Ok(())
    }

    /// Start the receive loop in a background task.
    /// Returns a channel to receive decrypted data from.
    pub fn start_recv_loop(self: &Arc<Self>) -> mpsc::Receiver<(Vec<u8>, u16)> {
        let (tx, rx) = mpsc::channel(256);
        let manager = Arc::clone(self);

        tokio::spawn(async move {
            let (receive_tx, mut receive_rx) = mpsc::channel(256);
            let mut spawned = HashSet::new();
            let mut scan_interval = tokio::time::interval(Duration::from_millis(250));

            loop {
                tokio::select! {
                    _ = scan_interval.tick() => {
                        if *manager.state.read() != ConnectionState::Connected {
                            break;
                        }

                        for uplink in manager
                            .uplinks
                            .iter()
                            .filter(|entry| entry.value().is_usable())
                            .map(|entry| entry.value().clone())
                        {
                            let uplink_id = uplink.numeric_id();
                            if spawned.insert(uplink_id) {
                                Self::spawn_uplink_recv_task(
                                    Arc::clone(&manager),
                                    uplink,
                                    receive_tx.clone(),
                                );
                            }
                        }
                    }
                    event = receive_rx.recv() => {
                        match event {
                            Some(ReceiveLoopEvent::Packet { uplink_id, uplink_name, data }) => {
                                let ack_sequence = match manager.data_sequence_to_ack(&data) {
                                    Ok(sequence) => sequence,
                                    Err(e) => {
                                        tracing::debug!(
                                            uplink = %uplink_name,
                                            error = %e,
                                            "Packet decode error"
                                        );
                                        continue;
                                    }
                                };

                                match manager.handle_packet(&data, uplink_id) {
                                    Ok(Some(payload)) => {
                                        if let Some(sequence) = ack_sequence {
                                            if let Err(e) = manager.send_ack(uplink_id, sequence).await {
                                                tracing::debug!(
                                                    uplink = %uplink_name,
                                                    sequence,
                                                    error = %e,
                                                    "Failed to send ACK"
                                                );
                                            }
                                        }

                                        if tx.send((payload, uplink_id)).await.is_err() {
                                            return;
                                        }
                                    }
                                    Ok(None) => {
                                        if let Some(sequence) = ack_sequence {
                                            if let Err(e) = manager.send_ack(uplink_id, sequence).await {
                                                tracing::debug!(
                                                    uplink = %uplink_name,
                                                    sequence,
                                                    error = %e,
                                                    "Failed to send ACK"
                                                );
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::debug!(
                                            uplink = %uplink_name,
                                            error = %e,
                                            "Packet handling error"
                                        );
                                    }
                                }
                            }
                            Some(ReceiveLoopEvent::Stopped { uplink_id }) => {
                                spawned.remove(&uplink_id);
                            }
                            None => break,
                        }
                    }
                }
            }
        });

        rx
    }

    fn spawn_uplink_recv_task(
        manager: Arc<Self>,
        uplink: Arc<Uplink>,
        receive_tx: mpsc::Sender<ReceiveLoopEvent>,
    ) {
        tokio::spawn(async move {
            let uplink_id = uplink.numeric_id();
            let uplink_name = uplink.id().clone();
            let mut buf = vec![0u8; 65536];

            loop {
                if *manager.state.read() != ConnectionState::Connected || !uplink.is_usable() {
                    break;
                }

                match tokio::time::timeout(Duration::from_secs(1), uplink.recv(&mut buf)).await {
                    Ok(Ok(len)) => {
                        if receive_tx
                            .send(ReceiveLoopEvent::Packet {
                                uplink_id,
                                uplink_name: uplink_name.clone(),
                                data: buf[..len].to_vec(),
                            })
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::debug!(uplink = %uplink_name, error = %e, "Receive error");
                        uplink.record_failure(&e.to_string());
                        manager.repair_uplink_paths(uplink_id).await;
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    Err(_) => {}
                }
            }

            let _ = receive_tx
                .send(ReceiveLoopEvent::Stopped { uplink_id })
                .await;
        });
    }

    /// Handle ACK packet.
    fn handle_ack(&self, packet: &Packet) -> Result<()> {
        let acked: Vec<u64> = bincode::deserialize(&packet.payload)
            .map_err(|e| crate::error::ProtocolError::Deserialization(e.to_string()))?;

        for seq in acked {
            if let Some((_, pending)) = self.pending.remove(&seq) {
                // Calculate RTT
                let rtt = pending.sent_at.elapsed();
                let uplink_id = pending.uplink_id;
                let packet_size = pending.packet.payload.len() as u64;

                if let Some(uplink) = self.get_uplink(uplink_id) {
                    uplink.record_rtt(rtt);

                    // Record with throughput optimizer for BBR-style congestion control
                    if self.config.throughput_optimization {
                        self.throughput_optimizer
                            .record_ack(uplink_id, packet_size, rtt);

                        // Record bandwidth measurement
                        if rtt.as_secs_f64() > 0.0 {
                            let bandwidth = packet_size as f64 / rtt.as_secs_f64();
                            self.throughput_optimizer
                                .record_bandwidth(uplink_id, bandwidth);
                        }

                        // Record MTU success
                        self.throughput_optimizer.record_mtu_result(
                            uplink_id,
                            packet_size as u32,
                            true,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle pong packet.
    fn handle_pong(&self, packet: &Packet, from_uplink: u16) {
        if packet.payload.len() >= 8 {
            let probe_or_ts = u64::from_be_bytes(packet.payload[..8].try_into().unwrap());
            let payload_size = packet.payload.len() as u64;

            // Check if this is a bandwidth probe response
            if self.config.throughput_optimization && payload_size > 64 {
                // This is likely a bandwidth probe response
                let probe_id = probe_or_ts;
                self.throughput_optimizer.record_probe_response(
                    from_uplink,
                    probe_id,
                    payload_size,
                );

                if let Some(uplink) = self.get_uplink(from_uplink) {
                    uplink.record_success();
                }
                return;
            }

            // Regular ping/pong for RTT measurement
            let ping_ts = probe_or_ts;
            let now_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;

            if now_us > ping_ts {
                let rtt = Duration::from_micros(now_us - ping_ts);
                if let Some(uplink) = self.get_uplink(from_uplink) {
                    uplink.record_rtt(rtt);

                    // Also record with throughput optimizer
                    if self.config.throughput_optimization {
                        self.throughput_optimizer
                            .record_ack(from_uplink, payload_size, rtt);
                    }
                }
            }
        }
    }

    /// Retry pending packets.
    pub async fn retry_pending(&self) {
        let now = Instant::now();
        let mut to_retry: Vec<(u64, Vec<u8>, u16)> = Vec::new();
        let mut to_fail = Vec::new();

        // Find packets that need retry
        for entry in &self.pending {
            let pending = entry.value();
            let uplink = self.get_uplink(pending.uplink_id);

            let rto = uplink.as_ref().map_or(Duration::from_millis(500), |u| {
                Duration::from_millis((u.rtt().as_millis() as u64 * 2).max(100))
            });

            if now.duration_since(pending.sent_at) > rto {
                if pending.retries < self.config.send_retries {
                    // Clone data for retry
                    to_retry.push((
                        *entry.key(),
                        pending.packet.payload.clone(),
                        pending.uplink_id,
                    ));
                } else {
                    to_fail.push(*entry.key());
                }
            }
        }

        // Retry packets on different uplinks
        for (seq, data, old_uplink_id) in to_retry {
            // Record loss on previous uplink
            if let Some(old_uplink) = self.get_uplink(old_uplink_id) {
                old_uplink.record_loss();
                self.repair_uplink_paths(old_uplink_id).await;
            }

            // Select new uplink (prefer different one from the failed one)
            let uplinks = self.uplinks();
            let selected = self.scheduler.select(&uplinks, Some(seq));

            // Try to find an uplink that's different from the one that failed
            let new_uplink_id = selected
                .iter()
                .find(|&&id| id != old_uplink_id)
                .copied()
                .or_else(|| selected.first().copied());

            if let Some(new_uplink_id) = new_uplink_id {
                let mut retry_uplinks = vec![new_uplink_id];
                if self.config.hedged_packets {
                    self.extend_with_hedged_uplinks(&mut retry_uplinks);
                }

                let mut sent_on = None;
                let mut attempted = false;
                for retry_uplink_id in retry_uplinks {
                    if let Some(uplink) = self.get_uplink(retry_uplink_id) {
                        attempted = true;
                        match self
                            .send_on_uplink(&uplink, SequenceNumber(seq), &data)
                            .await
                        {
                            Ok(_) => {
                                sent_on = Some(retry_uplink_id);
                                tracing::debug!(
                                    seq = seq,
                                    old_uplink = old_uplink_id,
                                    new_uplink = retry_uplink_id,
                                    "Retransmitted packet"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    seq = seq,
                                    uplink = %uplink.id(),
                                    error = %e,
                                    "Retry failed"
                                );
                                uplink.record_failure(&e.to_string());
                                self.repair_uplink_paths(retry_uplink_id).await;
                            }
                        }
                    }
                }

                if let Some(sent_uplink_id) = sent_on {
                    if let Some(mut entry) = self.pending.get_mut(&seq) {
                        entry.retries += 1;
                        entry.uplink_id = sent_uplink_id;
                        entry.sent_at = Instant::now();
                    }

                    let mut stats = self.stats.write();
                    stats.packets_retransmitted += 1;
                } else if attempted {
                    if let Some(mut entry) = self.pending.get_mut(&seq) {
                        entry.retries += 1;
                        entry.sent_at = Instant::now();
                    }
                }
            }
        }

        // Fail packets that exceeded retries
        for seq in to_fail {
            if let Some((_, pending)) = self.pending.remove(&seq) {
                {
                    let mut stats = self.stats.write();
                    stats.packets_dropped += 1;
                }

                if let Some(uplink) = self.get_uplink(pending.uplink_id) {
                    uplink.record_loss();
                    self.repair_uplink_paths(pending.uplink_id).await;
                }

                tracing::warn!(
                    seq = seq,
                    "Packet failed after {} retries",
                    self.config.send_retries
                );
                let _ = self.event_tx.send(MultipathEvent::SendFailed(seq));
            }
        }
    }

    /// Periodic maintenance.
    pub fn periodic_maintenance(&self) {
        // Update all uplinks
        for entry in &self.uplinks {
            entry.value().periodic_update();
        }

        // Check for all uplinks down
        let any_usable = self.uplinks.iter().any(|r| r.value().is_usable());
        if !any_usable && *self.state.read() == ConnectionState::Connected {
            let _ = self.event_tx.send(MultipathEvent::AllUplinksDown);
        }

        // Cleanup scheduler state
        self.scheduler.cleanup();

        // Cleanup stale flow bindings (Dublin Traceroute: flow maintenance)
        self.cleanup_stale_flows();

        // Cleanup stale path discovery data (Dublin Traceroute: path maintenance)
        self.path_discovery.cleanup(self.config.path_max_age);

        // Update aggregator weights for bandwidth-proportional striping
        if !matches!(self.config.aggregation_mode, AggregationMode::None) {
            let uplinks = self.uplinks();
            self.aggregator.update_weights(&uplinks);
        }
    }

    /// Start the maintenance loop in a background task.
    /// This handles retries, keepalives, uplink health monitoring, and throughput optimization.
    pub fn start_maintenance_loop(self: &Arc<Self>) {
        let manager = Arc::clone(self);

        tokio::spawn(async move {
            let mut retry_interval = tokio::time::interval(Duration::from_millis(100));
            let mut maintenance_interval =
                tokio::time::interval(manager.config.health_check_interval);
            let mut ping_interval = tokio::time::interval(Duration::from_secs(5));

            // Path discovery interval (Dublin Traceroute: periodic ECMP probing)
            let path_discovery_enabled = !manager.config.path_discovery_interval.is_zero();
            let mut path_discovery_interval = tokio::time::interval(if path_discovery_enabled {
                manager.config.path_discovery_interval
            } else {
                Duration::from_secs(3600) // Fallback, won't trigger if disabled
            });

            // Bandwidth probing interval for throughput optimization
            let bandwidth_probe_enabled =
                manager.config.throughput_optimization && manager.config.throughput.probing_enabled;
            let mut bandwidth_probe_interval = tokio::time::interval(if bandwidth_probe_enabled {
                manager.config.bandwidth_probe_interval
            } else {
                Duration::from_secs(3600)
            });

            loop {
                if *manager.state.read() != ConnectionState::Connected {
                    break;
                }

                tokio::select! {
                    _ = retry_interval.tick() => {
                        manager.retry_pending().await;
                    }
                    _ = maintenance_interval.tick() => {
                        manager.periodic_maintenance();
                    }
                    _ = ping_interval.tick() => {
                        // Send ping on all uplinks for quality measurement
                        for uplink_ref in &manager.uplinks {
                            let uplink = uplink_ref.value();
                            if uplink.is_usable() {
                                if let Some(transport) = uplink.get_transport() {
                                    let seq = SequenceNumber(manager.next_seq.fetch_add(1, Ordering::SeqCst));
                                    if let Ok(ping) = Packet::ping(seq, manager.session_id, uplink.numeric_id()) {
                                        if let Ok(encoded) = ping.encode() {
                                            let _ = transport.send(&encoded).await;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ = path_discovery_interval.tick(), if path_discovery_enabled => {
                        manager.run_path_discovery().await;
                    }
                    _ = bandwidth_probe_interval.tick(), if bandwidth_probe_enabled => {
                        manager.run_bandwidth_probes().await;
                    }
                }
            }
        });
    }

    /// Run active bandwidth probes on uplinks that need probing.
    async fn run_bandwidth_probes(&self) {
        let uplinks_to_probe = self.throughput_optimizer.uplinks_needing_probe();

        for uplink_id in uplinks_to_probe {
            if let Some(uplink) = self.get_uplink(uplink_id) {
                if let Some(transport) = uplink.get_transport() {
                    let probe_id = self.next_probe_id.fetch_add(1, Ordering::SeqCst);
                    let probe_size = self.config.throughput.probe_size;

                    // Create probe packet (ping with larger payload for bandwidth estimation)
                    let seq = SequenceNumber(self.next_seq.fetch_add(1, Ordering::SeqCst));

                    // Use a ping packet with a larger probe payload
                    let mut probe_payload = vec![0u8; probe_size];
                    probe_payload[0..8].copy_from_slice(&probe_id.to_be_bytes());

                    if let Ok(mut probe_packet) = Packet::new(
                        PacketType::Ping,
                        seq,
                        self.session_id,
                        uplink_id,
                        probe_payload,
                    ) {
                        probe_packet.set_flag(PacketFlags::ENCRYPTED);

                        if let Ok(encoded) = probe_packet.encode() {
                            // Record probe sent
                            self.throughput_optimizer
                                .record_probe_sent(uplink_id, probe_id);

                            // Send probe burst for more accurate measurement
                            for _ in 0..self.config.throughput.probe_burst_count {
                                let _ = transport.send(&encoded).await;
                            }

                            tracing::trace!(
                                uplink_id = uplink_id,
                                probe_id = probe_id,
                                probe_size = probe_size,
                                "Sent bandwidth probe"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Run path discovery for all connected uplinks.
    ///
    /// This probes paths to discover ECMP diversity and updates path quality metrics.
    /// (Dublin Traceroute: active ECMP path probing)
    async fn run_path_discovery(&self) {
        if !self.config.ecmp_aware {
            return;
        }

        let uplinks: Vec<_> = self
            .uplinks
            .iter()
            .filter(|r| r.value().is_usable())
            .map(|r| r.value().clone())
            .collect();

        for uplink in &uplinks {
            if let Some(transport) = uplink.get_transport() {
                // Get local and remote addresses for probe generation
                let local_addr = match transport.local_addr() {
                    Ok(addr) => addr,
                    Err(_) => continue,
                };
                let remote_addr = uplink.config().remote_addr;

                // Generate probes for this destination
                let probes = self.path_discovery.generate_probes(local_addr, remote_addr);

                tracing::debug!(
                    uplink = %uplink.id(),
                    probe_count = probes.len(),
                    "Generated path discovery probes"
                );

                // Send a sample of probes to different TTLs
                // Full probing would require raw socket access for ICMP;
                // here we record the intent and allow external probe injection
                for (ttl, flow_id, marker) in probes
                    .iter()
                    .take(self.config.path_discovery.num_paths as usize)
                {
                    // Record that we're probing this path
                    // Actual probe transmission depends on transport capability
                    tracing::trace!(
                        uplink = %uplink.id(),
                        ttl = ttl,
                        flow_hash = flow_id.flow_hash(),
                        ip_id = marker.ip_id,
                        "Path probe scheduled"
                    );
                }

                // Update NAT state from path discovery results
                self.path_discovery
                    .update_nat_state(uplink.numeric_id(), |state| {
                        // Sync with uplink's NAT state
                        state.set_nat_type(uplink.nat_type());
                    });

                // Emit path discovery event with current diversity metrics
                let diversity = self.path_discovery.get_diversity(remote_addr);
                let paths = self.path_discovery.get_paths(remote_addr);

                let _ = self.event_tx.send(MultipathEvent::PathDiscoveryComplete {
                    destination: remote_addr,
                    paths_found: paths.len(),
                    diversity_score: diversity.diversity_score,
                });
            }
        }
    }

    /// Process an incoming path probe response (e.g., ICMP Time Exceeded).
    ///
    /// Call this when you receive an ICMP response that correlates to a path probe.
    /// This is typically used when implementing custom probe transmission/reception.
    pub fn process_probe_response(
        &self,
        destination: std::net::SocketAddr,
        hop: super::path_discovery::Hop,
    ) {
        self.path_discovery.record_hop(destination, hop);

        // Recalculate diversity after recording hop
        let diversity = self.path_discovery.get_diversity(destination);
        let paths = self.path_discovery.get_paths(destination);

        tracing::debug!(
            destination = %destination,
            paths = paths.len(),
            diversity = diversity.diversity_score,
            "Processed path probe response"
        );

        let _ = self.event_tx.send(MultipathEvent::PathDiscoveryComplete {
            destination,
            paths_found: paths.len(),
            diversity_score: diversity.diversity_score,
        });
    }

    /// Close the connection.
    pub fn close(&self) -> Result<()> {
        *self.state.write() = ConnectionState::Disconnecting;

        // Close all uplinks
        for entry in &self.uplinks {
            entry
                .value()
                .set_connection_state(ConnectionState::Disconnecting);
        }

        self.pending.clear();
        *self.state.write() = ConnectionState::Disconnected;
        let _ = self.event_tx.send(MultipathEvent::Disconnected);

        Ok(())
    }

    /// Get access to the throughput optimizer.
    pub fn throughput_optimizer(&self) -> &ThroughputOptimizer {
        &self.throughput_optimizer
    }

    /// Get throughput summary across all uplinks.
    pub fn throughput_summary(&self) -> super::throughput::ThroughputSummary {
        self.throughput_optimizer.summary(&self.uplinks())
    }

    /// Get optimal packet size for an uplink (considering path MTU).
    pub fn optimal_packet_size(&self, uplink_id: u16) -> u32 {
        self.throughput_optimizer.optimal_packet_size(uplink_id)
    }

    /// Get effective throughput score for an uplink.
    pub fn effective_throughput(
        &self,
        uplink_id: u16,
    ) -> Option<super::throughput::EffectiveThroughput> {
        self.get_uplink(uplink_id)
            .map(|u| self.throughput_optimizer.effective_throughput(&u))
    }

    /// Select the best uplink for a transfer of given size.
    /// Considers both bandwidth and latency to find the fastest path.
    pub fn best_uplink_for_size(&self, size_bytes: u64) -> Option<u16> {
        self.throughput_optimizer
            .best_for_size(&self.uplinks(), size_bytes)
    }

    /// Get quality summary.
    pub fn quality_summary(&self) -> QualitySummary {
        let uplinks: Vec<_> = self.uplinks.iter().map(|r| r.value().clone()).collect();

        let usable = uplinks.iter().filter(|u| u.is_usable()).count();
        let total = uplinks.len();

        let avg_rtt = if uplinks.is_empty() {
            Duration::ZERO
        } else {
            let sum: Duration = uplinks.iter().map(|u| u.rtt()).sum();
            sum / uplinks.len() as u32
        };

        let avg_loss = if uplinks.is_empty() {
            0.0
        } else {
            let sum: f64 = uplinks.iter().map(|u| u.loss_ratio()).sum();
            sum / uplinks.len() as f64
        };

        let total_bandwidth = uplinks
            .iter()
            .filter(|u| u.is_usable())
            .map(|u| u.bandwidth())
            .fold(crate::types::Bandwidth::ZERO, |acc, b| {
                crate::types::Bandwidth::from_bps(acc.bytes_per_sec + b.bytes_per_sec)
            });

        QualitySummary {
            usable_uplinks: usable,
            total_uplinks: total,
            avg_rtt,
            avg_loss,
            total_bandwidth,
            stats: *self.stats.read(),
        }
    }

    // Dublin Traceroute-inspired ECMP and NAT-aware routing methods

    /// Extract flow ID from packet data by parsing IP headers.
    /// Returns a flow hash as u64 for use in ECMP-aware path selection.
    ///
    /// Supports both IPv4 and IPv6 packets with TCP/UDP headers.
    #[allow(dead_code)]
    fn extract_flow_id(data: &[u8]) -> Option<u64> {
        if data.len() < 20 {
            return None;
        }

        let version = (data[0] >> 4) & 0x0f;

        match version {
            4 => Self::extract_ipv4_flow_id(data),
            6 => Self::extract_ipv6_flow_id(data),
            _ => None,
        }
    }

    /// Extract flow ID from IPv4 packet.
    #[allow(dead_code)]
    fn extract_ipv4_flow_id(data: &[u8]) -> Option<u64> {
        if data.len() < 20 {
            return None;
        }

        let ihl = (data[0] & 0x0f) as usize * 4;
        if data.len() < ihl + 4 {
            return None;
        }

        let protocol = data[9];
        let src_ip = IpAddr::V4(std::net::Ipv4Addr::new(
            data[12], data[13], data[14], data[15],
        ));
        let dst_ip = IpAddr::V4(std::net::Ipv4Addr::new(
            data[16], data[17], data[18], data[19],
        ));

        // Extract ports for TCP (6) and UDP (17)
        let (src_port, dst_port) = if (protocol == 6 || protocol == 17) && data.len() >= ihl + 4 {
            let sport = u16::from_be_bytes([data[ihl], data[ihl + 1]]);
            let dport = u16::from_be_bytes([data[ihl + 2], data[ihl + 3]]);
            (sport, dport)
        } else {
            (0, 0)
        };

        // Calculate flow hash and return as u64
        let hash = calculate_flow_hash(src_ip, dst_ip, src_port, dst_port, protocol);
        Some(u64::from(hash))
    }

    /// Extract flow ID from IPv6 packet.
    #[allow(dead_code)]
    fn extract_ipv6_flow_id(data: &[u8]) -> Option<u64> {
        if data.len() < 40 {
            return None;
        }

        let next_header = data[6];
        let mut src_bytes = [0u8; 16];
        let mut dst_bytes = [0u8; 16];
        src_bytes.copy_from_slice(&data[8..24]);
        dst_bytes.copy_from_slice(&data[24..40]);

        let src_ip = IpAddr::V6(std::net::Ipv6Addr::from(src_bytes));
        let dst_ip = IpAddr::V6(std::net::Ipv6Addr::from(dst_bytes));

        // Extract ports for TCP (6) and UDP (17)
        // Note: This is simplified and doesn't handle extension headers
        let header_len = 40;
        let (src_port, dst_port, protocol) =
            if (next_header == 6 || next_header == 17) && data.len() >= header_len + 4 {
                let sport = u16::from_be_bytes([data[header_len], data[header_len + 1]]);
                let dport = u16::from_be_bytes([data[header_len + 2], data[header_len + 3]]);
                (sport, dport, next_header)
            } else {
                (0, 0, next_header)
            };

        let hash = calculate_flow_hash(src_ip, dst_ip, src_port, dst_port, protocol);
        Some(u64::from(hash))
    }

    /// Get access to path discovery for advanced routing decisions.
    pub fn path_discovery(&self) -> &PathDiscovery {
        &self.path_discovery
    }

    /// Get uplinks that are not behind NAT (useful for peer-to-peer connections).
    pub fn non_natted_uplinks(&self) -> Vec<Arc<Uplink>> {
        self.uplinks
            .iter()
            .filter(|r| r.value().is_usable() && !r.value().is_natted())
            .map(|r| r.value().clone())
            .collect()
    }

    /// Get NAT type summary for all uplinks.
    pub fn nat_summary(&self) -> Vec<(UplinkId, super::nat::NatType, bool)> {
        self.uplinks
            .iter()
            .map(|r| {
                let uplink = r.value();
                (uplink.id().clone(), uplink.nat_type(), uplink.is_natted())
            })
            .collect()
    }

    /// Select uplink considering NAT state.
    /// Prefers non-NATted uplinks when available, unless the NATted uplink has
    /// significantly better performance.
    pub fn select_nat_aware(&self, flow_id: Option<u64>) -> Option<u16> {
        let uplinks = self.uplinks();
        if uplinks.is_empty() {
            return None;
        }

        // Separate uplinks by NAT status
        let non_natted: Vec<_> = uplinks
            .iter()
            .filter(|u| u.is_usable() && !u.is_natted())
            .collect();

        let natted: Vec<_> = uplinks
            .iter()
            .filter(|u| u.is_usable() && u.is_natted())
            .collect();

        // If we have non-NATted options, prefer them unless NATted is much better
        if !non_natted.is_empty() {
            // Use scheduler to select among non-NATted
            let selected = self.scheduler.select(
                &non_natted.into_iter().cloned().collect::<Vec<_>>(),
                flow_id,
            );
            if !selected.is_empty() {
                return Some(selected[0]);
            }
        }

        // Fall back to NATted uplinks
        if !natted.is_empty() {
            let selected = self
                .scheduler
                .select(&natted.into_iter().cloned().collect::<Vec<_>>(), flow_id);
            if !selected.is_empty() {
                return Some(selected[0]);
            }
        }

        None
    }

    /// Calculate flow hash for a specific 5-tuple.
    /// This is a convenience wrapper around the flow_hash module.
    pub fn calculate_flow_hash(
        &self,
        src_ip: IpAddr,
        dst_ip: IpAddr,
        src_port: u16,
        dst_port: u16,
        protocol: u8,
    ) -> u16 {
        calculate_flow_hash(src_ip, dst_ip, src_port, dst_port, protocol)
    }

    /// Create a FlowId for manual flow management.
    pub fn create_flow_id(
        &self,
        src_ip: IpAddr,
        dst_ip: IpAddr,
        src_port: u16,
        dst_port: u16,
        protocol: u8,
    ) -> FlowId {
        FlowId::new(src_ip, dst_ip, src_port, dst_port, protocol)
    }

    // Flow binding management for ECMP path consistency

    /// Allocate a new flow ID for a connection/stream.
    ///
    /// Use this when establishing a new connection that should have consistent
    /// path routing. Pass the returned flow ID to `send_on_flow()`.
    pub fn allocate_flow(&self) -> u64 {
        self.next_flow_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Allocate a flow and immediately bind it to a specific uplink.
    ///
    /// This is useful when you want to ensure a connection uses a specific path.
    pub fn allocate_flow_on_uplink(&self, uplink_id: u16) -> Option<u64> {
        // Verify the uplink exists and is usable
        if !self.get_uplink(uplink_id).is_some_and(|u| u.is_usable()) {
            return None;
        }

        let flow_id = self.allocate_flow();
        self.flow_bindings.insert(flow_id, uplink_id);
        Some(flow_id)
    }

    /// Get the uplink bound to a flow.
    pub fn get_flow_binding(&self, flow_id: u64) -> Option<u16> {
        self.flow_bindings.get(&flow_id).map(|r| *r)
    }

    /// Release a flow binding when connection ends.
    pub fn release_flow(&self, flow_id: u64) {
        self.flow_bindings.remove(&flow_id);
    }

    /// Get number of active flow bindings.
    pub fn active_flow_count(&self) -> usize {
        self.flow_bindings.len()
    }

    /// Cleanup stale flow bindings (for flows whose uplinks are no longer usable).
    pub fn cleanup_stale_flows(&self) {
        let stale: Vec<u64> = self
            .flow_bindings
            .iter()
            .filter(|entry| {
                !self
                    .get_uplink(*entry.value())
                    .is_some_and(|u| u.is_usable())
            })
            .map(|entry| *entry.key())
            .collect();

        for flow_id in stale {
            self.flow_bindings.remove(&flow_id);
        }
    }
}

/// Quality summary.
#[derive(Debug, Clone)]
pub struct QualitySummary {
    pub usable_uplinks: usize,
    pub total_uplinks: usize,
    pub avg_rtt: Duration,
    pub avg_loss: f64,
    pub total_bandwidth: crate::types::Bandwidth,
    pub stats: TrafficStats,
}

// Intentionally abbreviated Debug output - internal state not useful for debugging
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for MultipathManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultipathManager")
            .field("session_id", &self.session_id)
            .field("connection_id", &self.connection_id)
            .field("state", &*self.state.read())
            .field("uplinks", &self.uplinks.len())
            .finish()
    }
}

fn is_hedged_packet(data: &[u8], small_packet_threshold: usize) -> bool {
    if data.len() <= small_packet_threshold {
        return true;
    }

    let Ok(packet) = IpPacket::parse(data) else {
        return false;
    };

    packet.is_dns() || packet.is_tcp_syn() || packet.is_tcp_fin_or_rst()
}

fn hedge_uplink_score(uplink: &Uplink) -> f64 {
    let health = uplink.health().priority_modifier();
    let rtt_penalty = uplink.rtt().as_millis() as f64;
    let loss_penalty = uplink.loss_ratio() * 10_000.0;

    health * 10_000.0 - rtt_penalty - loss_penalty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::TransportProtocol;

    #[test]
    fn test_dedup_new_sequences() {
        let mut window = DeduplicationWindow::new(10);
        assert!(!window.is_duplicate(1));
        assert!(!window.is_duplicate(2));
        assert!(!window.is_duplicate(3));
    }

    #[test]
    fn test_dedup_detects_duplicates() {
        let mut window = DeduplicationWindow::new(10);
        assert!(!window.is_duplicate(1));
        assert!(window.is_duplicate(1)); // Duplicate
        assert!(!window.is_duplicate(2));
        assert!(window.is_duplicate(2)); // Duplicate
    }

    #[test]
    fn test_dedup_window_cleanup() {
        let mut window = DeduplicationWindow::new(3);
        assert!(!window.is_duplicate(1));
        assert!(!window.is_duplicate(2));
        assert!(!window.is_duplicate(3));
        // Window full: [1, 2, 3]
        assert!(!window.is_duplicate(4)); // Pushes out 1
                                          // Window now: [2, 3, 4]
        assert!(!window.is_duplicate(1)); // 1 is no longer in window, so not duplicate
        assert!(window.is_duplicate(3)); // Still in window
        assert!(window.is_duplicate(4)); // Still in window
    }

    #[test]
    fn test_dedup_out_of_order() {
        let mut window = DeduplicationWindow::new(10);
        assert!(!window.is_duplicate(5));
        assert!(!window.is_duplicate(3));
        assert!(!window.is_duplicate(7));
        assert!(!window.is_duplicate(1));
        // All unique, should detect duplicates regardless of order
        assert!(window.is_duplicate(5));
        assert!(window.is_duplicate(3));
        assert!(window.is_duplicate(7));
        assert!(window.is_duplicate(1));
    }

    #[test]
    fn hedged_packet_classifier_matches_interactive_packets() {
        assert!(is_hedged_packet(b"ping", 8));
        assert!(!is_hedged_packet(&vec![0xaa; 512], 8));

        let mut dns_packet = vec![0u8; 32];
        dns_packet[0] = 0x45;
        dns_packet[2..4].copy_from_slice(&32u16.to_be_bytes());
        dns_packet[8] = 64;
        dns_packet[9] = 17;
        dns_packet[12..16].copy_from_slice(&[10, 77, 0, 2]);
        dns_packet[16..20].copy_from_slice(&[8, 8, 8, 8]);
        dns_packet[20..22].copy_from_slice(&49152u16.to_be_bytes());
        dns_packet[22..24].copy_from_slice(&53u16.to_be_bytes());
        dns_packet[24..26].copy_from_slice(&12u16.to_be_bytes());
        dns_packet[28..32].copy_from_slice(b"dns?");

        assert!(is_hedged_packet(&dns_packet, 0));

        let tcp_syn = [
            0x45, 0x00, 0x00, 0x3c, 0x1c, 0x46, 0x40, 0x00, 0x40, 0x06, 0x00, 0x00, 0xc0, 0xa8,
            0x01, 0x01, 0x08, 0x08, 0x08, 0x08, 0x04, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x50, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        assert!(is_hedged_packet(&tcp_syn, 0));
    }

    #[test]
    fn hedging_extends_selection_with_backup_uplink() {
        let mut config = MultipathConfig::default();
        config.hedged_packets = true;
        config.hedged_max_copies = 2;
        let manager = MultipathManager::new(config, KeyPair::generate());

        let primary_id = manager
            .add_uplink(UplinkConfig {
                id: UplinkId::new("primary"),
                remote_addr: "127.0.0.1:10000".parse().unwrap(),
                protocol: TransportProtocol::Udp,
                ..Default::default()
            })
            .unwrap();
        let backup_id = manager
            .add_uplink(UplinkConfig {
                id: UplinkId::new("backup"),
                remote_addr: "127.0.0.1:10001".parse().unwrap(),
                protocol: TransportProtocol::Udp,
                ..Default::default()
            })
            .unwrap();

        manager
            .get_uplink(primary_id)
            .unwrap()
            .set_connection_state(ConnectionState::Connected);
        manager
            .get_uplink(backup_id)
            .unwrap()
            .set_connection_state(ConnectionState::Connected);

        let mut selected = vec![primary_id];
        manager.extend_with_hedged_uplinks(&mut selected);

        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&primary_id));
        assert!(selected.contains(&backup_id));
    }

    #[tokio::test]
    async fn path_repair_moves_flow_bindings_to_usable_replacement() {
        let manager = MultipathManager::new(MultipathConfig::default(), KeyPair::generate());

        let failed_id = manager
            .add_uplink(UplinkConfig {
                id: UplinkId::new("failed"),
                remote_addr: "127.0.0.1:10000".parse().unwrap(),
                protocol: TransportProtocol::Udp,
                ..Default::default()
            })
            .unwrap();
        let replacement_id = manager
            .add_uplink(UplinkConfig {
                id: UplinkId::new("replacement"),
                remote_addr: "127.0.0.1:10001".parse().unwrap(),
                protocol: TransportProtocol::Udp,
                ..Default::default()
            })
            .unwrap();

        manager
            .get_uplink(failed_id)
            .unwrap()
            .set_connection_state(ConnectionState::Connected);
        manager
            .get_uplink(replacement_id)
            .unwrap()
            .set_connection_state(ConnectionState::Connected);

        let flow_id = manager.allocate_flow_on_uplink(failed_id).unwrap();

        manager.repair_uplink_paths(failed_id).await;

        assert_eq!(manager.get_flow_binding(flow_id), Some(replacement_id));
    }

    #[tokio::test]
    async fn path_repair_clears_flow_bindings_without_replacement() {
        let manager = MultipathManager::new(MultipathConfig::default(), KeyPair::generate());

        let failed_id = manager
            .add_uplink(UplinkConfig {
                id: UplinkId::new("failed"),
                remote_addr: "127.0.0.1:10000".parse().unwrap(),
                protocol: TransportProtocol::Udp,
                ..Default::default()
            })
            .unwrap();

        manager
            .get_uplink(failed_id)
            .unwrap()
            .set_connection_state(ConnectionState::Connected);

        let flow_id = manager.allocate_flow_on_uplink(failed_id).unwrap();

        manager.repair_uplink_paths(failed_id).await;

        assert_eq!(manager.get_flow_binding(flow_id), None);
    }
}
