//! Multi-path connection manager.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use futures::stream::{FuturesUnordered, StreamExt};
use parking_lot::{Mutex, RwLock};
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

    /// Maximum logical packets awaiting acknowledgement (including redundant copies).
    #[serde(default = "default_max_pending_packets")]
    pub max_pending_packets: usize,

    /// Maximum retained plaintext bytes awaiting acknowledgement.
    #[serde(default = "default_max_pending_bytes")]
    pub max_pending_bytes: usize,

    /// Absolute retention limit, including periods with no usable uplinks.
    #[serde(default = "default_max_pending_age", with = "humantime_serde")]
    pub max_pending_age: Duration,

    /// Maximum inner IP packet size; must fit a protected datagram.
    #[serde(default = "default_inner_mtu")]
    pub inner_mtu: u16,

    /// Retry failed path establishment starting at this delay.
    #[serde(default = "default_reconnect_initial", with = "humantime_serde")]
    pub reconnect_initial_delay: Duration,
    #[serde(default = "default_reconnect_max", with = "humantime_serde")]
    pub reconnect_max_delay: Duration,
    /// Re-establish keys after this long without an authenticated record.
    #[serde(default = "default_peer_silence", with = "humantime_serde")]
    pub peer_silence_timeout: Duration,

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
fn default_max_pending_packets() -> usize {
    4096
}
fn default_max_pending_bytes() -> usize {
    8 * 1024 * 1024
}
fn default_max_pending_age() -> Duration {
    Duration::from_secs(10)
}
fn default_inner_mtu() -> u16 {
    1340
}

fn default_reconnect_initial() -> Duration {
    Duration::from_millis(250)
}
fn default_reconnect_max() -> Duration {
    Duration::from_secs(15)
}
fn default_peer_silence() -> Duration {
    Duration::from_secs(15)
}

fn reconnect_delay(config: &MultipathConfig, failures: u32, jitter: f64) -> Duration {
    let initial = config
        .reconnect_initial_delay
        .max(Duration::from_millis(20));
    let ceiling = config.reconnect_max_delay.max(initial);
    let base = initial
        .saturating_mul(1u32 << failures.min(16))
        .min(ceiling);
    base.mul_f64(jitter.clamp(0.8, 1.2)).min(ceiling)
}

#[derive(Debug)]
struct ReconnectState {
    failures: u32,
    next_attempt: Instant,
    in_progress: bool,
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
            max_pending_packets: default_max_pending_packets(),
            max_pending_bytes: default_max_pending_bytes(),
            max_pending_age: default_max_pending_age(),
            inner_mtu: default_inner_mtu(),
            reconnect_initial_delay: default_reconnect_initial(),
            reconnect_max_delay: default_reconnect_max(),
            peer_silence_timeout: default_peer_silence(),
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
    created_at: Instant,
    sent_at: Instant,
    retries: u32,
    uplink_id: u16,
    attempts: HashMap<u16, Instant>,
}

#[derive(Default)]
struct PendingBuffer {
    packets: HashMap<u64, PendingPacket>,
    bytes: usize,
}

impl PendingBuffer {
    fn remove(&mut self, seq: u64) -> Option<PendingPacket> {
        let packet = self.packets.remove(&seq)?;
        self.bytes -= packet.packet.payload.len();
        Some(packet)
    }
}

/// Bounded retransmission buffer occupancy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingStats {
    pub packets: usize,
    pub bytes: usize,
    pub max_packets: usize,
    pub max_bytes: usize,
    pub oldest_age_ms: u64,
}

enum ReceiveLoopEvent {
    Packet {
        generation: u64,
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
    pending: Mutex<PendingBuffer>,
    retry_lock: tokio::sync::Mutex<()>,
    reconnect: Mutex<HashMap<u16, ReconnectState>>,
    remote_public: RwLock<Option<PublicKey>>,
    next_return_sequence: AtomicU64,
    next_handshake_epoch: AtomicU64,
    negotiated_mtu: AtomicU16,
    closing: AtomicBool,
    lifecycle: AtomicU64,
    maintenance_generation: AtomicU64,
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
    join_token: [u8; 32],
    access_token: RwLock<Option<[u8; 32]>>,
    assigned_ipv4: RwLock<Option<std::net::Ipv4Addr>>,
    assigned_ipv6: RwLock<Option<std::net::Ipv6Addr>>,
    ready_packets: parking_lot::Mutex<VecDeque<Vec<u8>>>,
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
            negotiated_mtu: AtomicU16::new(config.inner_mtu),
            config,
            session_id: SessionId::generate(),
            connection_id: ConnectionId::new(),
            uplinks: DashMap::new(),
            uplink_ids: DashMap::new(),
            next_seq: AtomicU64::new(1),
            next_uplink_id: AtomicU16::new(0),
            pending: Mutex::new(PendingBuffer::default()),
            retry_lock: tokio::sync::Mutex::new(()),
            reconnect: Mutex::new(HashMap::new()),
            remote_public: RwLock::new(None),
            next_return_sequence: AtomicU64::new(0),
            next_handshake_epoch: AtomicU64::new(1),
            closing: AtomicBool::new(false),
            lifecycle: AtomicU64::new(0),
            maintenance_generation: AtomicU64::new(u64::MAX),
            dedup_window: RwLock::new(DeduplicationWindow::new(dedup_size)),
            stats: RwLock::new(TrafficStats::default()),
            event_tx,
            state: RwLock::new(ConnectionState::Disconnected),
            local_keypair,
            join_token: rand::random(),
            access_token: RwLock::new(None),
            assigned_ipv4: RwLock::new(None),
            assigned_ipv6: RwLock::new(None),
            ready_packets: parking_lot::Mutex::new(VecDeque::new()),
            flow_bindings: DashMap::new(),
            next_flow_id: AtomicU64::new(1),
            next_probe_id: AtomicU64::new(1),
        }
    }

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Address allocated by the production exit for this logical client.
    pub fn assigned_ipv4(&self) -> Option<std::net::Ipv4Addr> {
        *self.assigned_ipv4.read()
    }

    pub fn assigned_ipv6(&self) -> Option<std::net::Ipv6Addr> {
        *self.assigned_ipv6.read()
    }

    pub fn set_access_token(&self, token: Option<[u8; 32]>) {
        *self.access_token.write() = token;
    }

    pub(crate) fn access_token(&self) -> Option<[u8; 32]> {
        *self.access_token.read()
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
        if self.closing.load(Ordering::Acquire) {
            return Err(Error::ConnectionClosed);
        }
        *self.remote_public.write() = Some(*remote_public);
        if let Some(existing_id) = self.uplink_ids.get(&config.id).map(|id| *id) {
            if let Some(existing) = self.get_uplink(existing_id) {
                if existing.config().local_addr == config.local_addr
                    && existing.config().remote_addr == config.remote_addr
                    && existing.config().protocol == config.protocol
                {
                    if matches!(
                        existing.state().connection_state,
                        ConnectionState::Connecting | ConnectionState::Handshaking
                    ) {
                        // The concurrent startup handshake already owns this path.
                        return Ok(existing_id);
                    }
                    if existing.state().connection_state != ConnectionState::Connected {
                        if let Err(e) = self.connect_uplink(&existing, remote_public).await {
                            existing.set_connection_state(ConnectionState::Failed);
                            existing.record_failure(&e.to_string());
                            return Err(e);
                        }
                    }
                    *self.state.write() = ConnectionState::Connected;
                    self.send_fast_probe(existing_id).await;
                    return Ok(existing_id);
                }

                existing.set_connection_state(ConnectionState::Disconnected);
                self.clear_flow_bindings_for_uplink(existing_id);
                let replacement = Arc::new(Uplink::new(config, existing_id));
                {
                    let mut buffer = self.pending.lock();
                    for pending in buffer.packets.values_mut() {
                        if pending.attempts.remove(&existing_id).is_some() {
                            existing.cancel_in_flight();
                        }
                    }
                }
                self.uplinks.insert(existing_id, Arc::clone(&replacement));
                self.reconnect.lock().remove(&existing_id);
                if let Err(error) = self.connect_uplink(&replacement, remote_public).await {
                    replacement.set_connection_state(ConnectionState::Failed);
                    replacement.record_failure(&error.to_string());
                    return Err(error);
                }
                *self.state.write() = ConnectionState::Connected;
                self.send_fast_probe(existing_id).await;
                return Ok(existing_id);
            }
        }

        let id = self.add_uplink(config)?;
        if let Some(uplink) = self.get_uplink(id) {
            if let Err(e) = self.connect_uplink(&uplink, remote_public).await {
                uplink.set_connection_state(ConnectionState::Failed);
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

    /// Retain established exit identities through a temporary interface outage.
    /// A successful authenticated probe re-enables the same encrypted path.
    pub fn suspend_uplinks_for_interface(&self, interface: &str) -> usize {
        let mut count = 0;
        for uplink in self.uplinks() {
            if uplink.config().interface.as_deref() == Some(interface) {
                uplink.mark_unavailable();
                self.clear_flow_bindings_for_uplink(uplink.numeric_id());
                count += 1;
            }
        }
        count
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

    pub fn negotiated_mtu(&self) -> u16 {
        self.negotiated_mtu.load(Ordering::Acquire)
    }

    fn offered_mtu(&self, uplink: &Uplink) -> Result<u16> {
        use crate::protocol::mtu;
        let mut offered = mtu::validate_mtu(self.config.inner_mtu)?.min(self.negotiated_mtu());
        if let Some(interface) = &uplink.config().interface {
            if let Some(info) = crate::util::get_network_interfaces()
                .into_iter()
                .find(|i| &i.name == interface)
            {
                offered = offered.min(
                    mtu::path_inner_mtu(info.mtu, uplink.config().remote_addr.is_ipv6()).map_err(
                        |error| Error::ConnectionFailed {
                            addr: uplink.config().remote_addr,
                            reason: error.to_string(),
                        },
                    )?,
                );
            }
        }
        Ok(offered)
    }

    async fn send_fast_probe(&self, uplink_id: u16) {
        let Some(uplink) = self.get_uplink(uplink_id) else {
            return;
        };
        let Some(transport) = uplink.get_transport() else {
            return;
        };

        let seq = SequenceNumber(self.next_seq.fetch_add(1, Ordering::SeqCst));
        let Ok(mut ping) = Packet::ping(seq, self.session_id, uplink.numeric_id()) else {
            return;
        };
        ping.payload = ping.header.timestamp.to_be_bytes().to_vec();
        if self.assigned_ipv4().is_some() {
            let Ok(offered) = self.offered_mtu(&uplink) else {
                uplink.mark_unavailable();
                return;
            };
            let mut challenge = [0; 16];
            challenge[..8].copy_from_slice(&ping.header.timestamp.to_be_bytes());
            challenge[8..].copy_from_slice(&rand::random::<[u8; 8]>());
            ping.payload = crate::protocol::mtu::request(&challenge, offered);
            let Ok(payload) = uplink.encrypt_packet(&ping.payload, seq.0, PacketType::Ping as u8)
            else {
                return;
            };
            ping.payload = payload;
            ping.header.payload_len = ping.payload.len() as u16;
            ping.set_flag(PacketFlags::ENCRYPTED);
        } else {
            ping.header.payload_len = ping.payload.len() as u16;
            ping.set_flag(0);
        }
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
        crate::protocol::mtu::validate_mtu(self.config.inner_mtu)?;
        self.closing.store(false, Ordering::Release);
        *self.remote_public.write() = Some(remote_public);
        let lifecycle = self.lifecycle.load(Ordering::Acquire);
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
                    uplink.set_connection_state(ConnectionState::Failed);
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
                    if self.closing.load(Ordering::Acquire)
                        || self.lifecycle.load(Ordering::Acquire) != lifecycle
                    {
                        return Err(Error::ConnectionClosed);
                    }
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

        if self.closing.load(Ordering::Acquire)
            || self.lifecycle.load(Ordering::Acquire) != lifecycle
        {
            return Err(Error::ConnectionClosed);
        }
        *self.state.write() = ConnectionState::Failed;
        Err(last_error.unwrap_or(Error::NoAvailableUplinks))
    }

    /// Wait for an endpoint, retaining interface discovery and a bounded retry rate.
    pub async fn connect_until_ready(self: &Arc<Self>, remote_public: PublicKey) -> Result<()> {
        self.closing.store(false, Ordering::Release);
        let lifecycle = self.lifecycle.load(Ordering::Acquire);
        let mut failures = 0;
        loop {
            if self.closing.load(Ordering::Acquire)
                || self.lifecycle.load(Ordering::Acquire) != lifecycle
            {
                return Err(Error::ConnectionClosed);
            }
            match self.connect(remote_public).await {
                Ok(()) => return Ok(()),
                Err(Error::ConnectionClosed) => return Err(Error::ConnectionClosed),
                Err(
                    error @ (Error::Config(_) | Error::InvalidConfig(_) | Error::InvalidKey(_)),
                ) => return Err(error),
                Err(error) => tracing::warn!(%error, "Waiting for an available exit path"),
            }
            let delay = reconnect_delay(&self.config, failures, 0.8 + rand::random::<f64>() * 0.4);
            failures = failures.saturating_add(1);
            tokio::time::sleep(delay).await;
        }
    }

    /// Connect a single uplink.
    async fn connect_uplink(&self, uplink: &Uplink, remote_public: &PublicKey) -> Result<()> {
        let lifecycle = self.lifecycle.load(Ordering::Acquire);
        let _handshake = uplink.handshake_lock.lock().await;
        if self.closing.load(Ordering::Acquire)
            || !self
                .get_uplink(uplink.numeric_id())
                .is_some_and(|current| std::ptr::eq(current.as_ref(), uplink))
        {
            return Err(Error::ConnectionClosed);
        }
        if uplink.state().connection_state == ConnectionState::Connected
            && uplink.is_noise_ready()
            && uplink.silence() < self.config.peer_silence_timeout
        {
            return Ok(());
        }

        uplink.set_connection_state(ConnectionState::Connecting);

        let config = uplink.config();

        // Create transport
        let local_addr = if config.local_addr.is_some() {
            config.local_addr
        } else if let Some(interface) = &config.interface {
            Some(crate::util::interface_bind_address(
                interface,
                config.remote_addr,
            )?)
        } else {
            None
        };
        let transport: Box<dyn Transport> =
            if config.protocol == crate::transport::TransportProtocol::Udp {
                Box::new(
                    crate::transport::UdpTransport::connect_on_interface(
                        config.remote_addr,
                        local_addr,
                        &self.config.transport,
                        config.interface.as_deref(),
                    )
                    .await?,
                )
            } else {
                connect(
                    config.protocol,
                    config.remote_addr,
                    local_addr,
                    &self.config.transport,
                )
                .await?
            };

        // Wrap in Arc for sharing across async operations
        let transport: Arc<dyn Transport> = Arc::from(transport);

        // Perform Noise handshake
        uplink.set_connection_state(ConnectionState::Handshaking);

        let mut noise = NoiseSession::new_initiator(&self.local_keypair.secret, remote_public)?;

        // Write handshake message 1 (e, es)
        let hello = crate::protocol::recovery::ExitHello {
            token: self.join_token,
            session: self.session_id,
            uplink: uplink.numeric_id(),
            preferred_ipv4: self.assigned_ipv4(),
            next_return_sequence: self.next_return_sequence.load(Ordering::Acquire),
            path_epoch: self.next_handshake_epoch.fetch_add(1, Ordering::AcqRel),
            access_token: self.access_token(),
            version: 3,
        }
        .encode();
        let msg1 = noise.write_handshake(&hello)?;

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

        // Retransmit the same hello: the exit caches its reply. A lost UDP
        // handshake packet does not require waiting for the entire reconnect cycle.
        let mut buf = vec![0u8; 65536];
        let deadline = tokio::time::Instant::now() + self.config.transport.connect_timeout;
        let len = loop {
            let retry_at = (tokio::time::Instant::now() + Duration::from_millis(500)).min(deadline);
            match tokio::time::timeout_at(retry_at, transport.recv(&mut buf)).await {
                Ok(result) => break result?,
                Err(_) if tokio::time::Instant::now() < deadline => {
                    transport.send(&encoded).await?;
                }
                Err(_) => return Err(Error::ConnectionTimeout),
            }
        };

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
        if response_packet.header.session_id != self.session_id
            || response_packet.header.uplink_id != uplink.numeric_id()
        {
            return Err(Error::InvalidPacket("Handshake identity mismatch".into()));
        }
        let response = noise.read_handshake(&response_packet.payload)?;
        let recovery_peer = response.starts_with(b"TG3");
        if recovery_peer || response.starts_with(b"TG2") {
            if !matches!(response.len(), 7 | 23) {
                return Err(Error::InvalidPacket("Invalid exit handshake".into()));
            }
            let address =
                std::net::Ipv4Addr::new(response[3], response[4], response[5], response[6]);
            let ipv6 = if response.len() == 23 {
                Some(std::net::Ipv6Addr::from(
                    <[u8; 16]>::try_from(&response[7..23]).unwrap(),
                ))
            } else {
                None
            };
            let mut assigned = self.assigned_ipv4.write();
            if assigned.is_some() && *self.assigned_ipv6.read() != ipv6 {
                return Err(Error::InvalidPacket(
                    "Exit changed client IPv6 address".into(),
                ));
            }
            if assigned.is_some_and(|old| old != address) {
                return Err(Error::InvalidPacket("Exit changed client address".into()));
            }
            *assigned = Some(address);
            *self.assigned_ipv6.write() = ipv6;
            noise.enable_datagrams();
        }

        if !noise.is_transport() {
            return Err(Error::Crypto(crate::error::CryptoError::NoiseProtocol(
                "Handshake did not complete".into(),
            )));
        }

        if !recovery_peer && response.starts_with(b"TG2") {
            if self.offered_mtu(uplink)? < crate::protocol::mtu::DEFAULT_INNER_MTU {
                return Err(Error::HandshakeFailed(
                    "TG2 exit cannot negotiate the reduced path MTU".into(),
                ));
            }
            self.negotiated_mtu
                .fetch_min(crate::protocol::mtu::DEFAULT_INNER_MTU, Ordering::AcqRel);
        }

        tracing::debug!(
            uplink = %uplink.id(),
            "Handshake complete, noise session ready"
        );

        let mut confirmed_rtt = None;
        if recovery_peer {
            if !self
                .get_uplink(uplink.numeric_id())
                .is_some_and(|current| std::ptr::eq(current.as_ref(), uplink))
            {
                return Err(Error::ConnectionClosed);
            }
            // Proof of possession commits a prepared rekey on the exit. Read the
            // encrypted confirmation before exposing the new transport to receivers.
            let sequence = self.next_seq.fetch_add(1, Ordering::SeqCst);
            let challenge: [u8; 16] = rand::random();
            let offered = self.offered_mtu(uplink)?;
            let confirmation = crate::protocol::mtu::request(&challenge, offered);
            let payload = noise.encrypt_packet(&confirmation, sequence, PacketType::Ping as u8)?;
            let mut probe = Packet::new(
                PacketType::Ping,
                SequenceNumber(sequence),
                self.session_id,
                uplink.numeric_id(),
                payload,
            )?;
            probe.set_flag(PacketFlags::ENCRYPTED);
            let mut wire = probe.encode()?;
            let deadline = tokio::time::Instant::now() + self.config.transport.connect_timeout;
            loop {
                // Every retry needs a fresh record nonce. The challenge remains bound
                // to this handshake; unrelated data cannot confirm its keys.
                let sent_at = Instant::now();
                transport.send(&wire).await?;
                let end = (tokio::time::Instant::now() + Duration::from_millis(500)).min(deadline);
                if let Ok(Ok(len)) = tokio::time::timeout_at(end, transport.recv(&mut buf)).await {
                    if let Ok(packet) = Packet::decode(&buf[..len]) {
                        if packet.header.session_id == self.session_id
                            && packet.header.uplink_id == uplink.numeric_id()
                            && packet.header.packet_type == PacketType::Pong
                            && packet.header.flags.has(PacketFlags::ENCRYPTED)
                        {
                            if let Ok(payload) = noise.decrypt_packet(
                                &packet.payload,
                                packet.header.sequence.0,
                                PacketType::Pong as u8,
                            ) {
                                let accepted =
                                    crate::protocol::mtu::confirm(&payload, &challenge, offered)?;
                                self.negotiated_mtu.fetch_min(accepted, Ordering::AcqRel);
                                confirmed_rtt = Some(sent_at.elapsed());
                                break;
                            }
                        }
                    }
                }
                if tokio::time::Instant::now() >= deadline {
                    return Err(Error::ConnectionTimeout);
                }
                // Re-encoding below uses a new nonce so the exit can respond after
                // a lost Pong without weakening replay protection.
                let payload =
                    noise.encrypt_packet(&confirmation, sequence, PacketType::Ping as u8)?;
                probe.payload = payload;
                probe.header.payload_len = probe.payload.len() as u16;
                probe.set_flag(PacketFlags::ENCRYPTED);
                wire = probe.encode()?;
            }
        }
        let _lifecycle_guard = self.state.read();
        if self.closing.load(Ordering::Acquire)
            || self.lifecycle.load(Ordering::Acquire) != lifecycle
            || !self
                .get_uplink(uplink.numeric_id())
                .is_some_and(|current| std::ptr::eq(current.as_ref(), uplink))
        {
            return Err(Error::ConnectionClosed);
        }
        uplink.set_transport(Arc::clone(&transport));
        uplink.set_noise_session(noise);
        uplink.set_connection_state(ConnectionState::Connected);
        if let Some(rtt) = confirmed_rtt {
            uplink.record_probe_success(rtt);
        }

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
        if self.assigned_ipv4().is_some() && data.len() > usize::from(self.negotiated_mtu()) {
            return Err(crate::error::TransportError::MtuExceeded {
                size: data.len(),
                max: usize::from(self.negotiated_mtu()),
            }
            .into());
        }
        if self.closing.load(Ordering::Acquire) {
            return Err(Error::ConnectionClosed);
        }
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
        self.track_packet(seq, data, uplink_id)?;

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
                    let result = self.send_aggregated_fallback(data, uplink_id, seq).await;
                    if result.is_err() {
                        self.cancel_pending(seq.0);
                    }
                    result
                }
            }
        } else {
            self.cancel_pending(seq.0);
            Err(Error::NoAvailableUplinks)
        }
    }

    /// Fallback when primary stripe uplink fails - try next available.
    async fn send_aggregated_fallback(
        &self,
        data: &[u8],
        failed_uplink: u16,
        seq: SequenceNumber,
    ) -> Result<u64> {
        let uplinks = self.uplinks();

        for uplink in &uplinks {
            if uplink.numeric_id() == failed_uplink {
                continue;
            }
            if !uplink.is_usable() {
                continue;
            }

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
        self.track_packet(seq, data, selected[0])?;

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
            self.cancel_pending(seq.0);
            Err(e)
        } else {
            self.cancel_pending(seq.0);
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

    fn track_packet(&self, seq: SequenceNumber, data: &[u8], uplink_id: u16) -> Result<()> {
        let mut buffer = self.pending.lock();
        if buffer.packets.len() >= self.config.max_pending_packets
            || data.len() > self.config.max_pending_bytes.saturating_sub(buffer.bytes)
        {
            return Err(Error::Backpressure);
        }
        if buffer.packets.contains_key(&seq.0) {
            return Err(Error::Internal("duplicate pending sequence".into()));
        }
        let packet = Packet::data(seq, self.session_id, uplink_id, data.to_vec())?;
        buffer.bytes += data.len();
        buffer.packets.insert(
            seq.0,
            PendingPacket {
                packet,
                created_at: Instant::now(),
                sent_at: Instant::now(),
                retries: 0,
                uplink_id,
                attempts: HashMap::new(),
            },
        );
        Ok(())
    }

    fn track_attempt(&self, seq: u64, uplink: &Uplink) -> bool {
        let mut buffer = self.pending.lock();
        let Some(pending) = buffer.packets.get_mut(&seq) else {
            return false; // An earlier hedge was already acknowledged or expired.
        };
        if pending
            .attempts
            .insert(uplink.numeric_id(), Instant::now())
            .is_none()
        {
            uplink.begin_in_flight();
        }
        true
    }

    fn cancel_pending(&self, seq: u64) {
        if let Some(pending) = self.pending.lock().remove(seq) {
            for id in pending.attempts.keys() {
                if let Some(uplink) = self.get_uplink(*id) {
                    uplink.cancel_in_flight();
                }
            }
        }
    }

    pub fn pending_stats(&self) -> PendingStats {
        let buffer = self.pending.lock();
        PendingStats {
            packets: buffer.packets.len(),
            bytes: buffer.bytes,
            max_packets: self.config.max_pending_packets,
            max_bytes: self.config.max_pending_bytes,
            oldest_age_ms: buffer
                .packets
                .values()
                .map(|p| p.created_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64)
                .max()
                .unwrap_or(0),
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
        let encrypted_payload = uplink.encrypt_packet(data, seq.0, PacketType::Data as u8)?;

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

        if !self.track_attempt(seq.0, uplink) {
            return Ok(());
        }
        if let Err(error) = transport.send(&encoded).await {
            let mut buffer = self.pending.lock();
            if let Some(pending) = buffer.packets.get_mut(&seq.0) {
                if pending.attempts.remove(&uplink.numeric_id()).is_some() {
                    uplink.cancel_in_flight();
                }
            }
            return Err(error);
        }
        uplink.record_transmission(encoded.len());

        tracing::trace!(
            uplink = %uplink.id(),
            seq = seq.0,
            len = data.len(),
            "Sent encrypted packet"
        );

        Ok(())
    }

    /// Handle incoming packet.
    /// When aggregation is enabled, packets are passed through the reorder buffer
    /// to handle out-of-order arrival from different uplinks.
    pub fn handle_packet(&self, data: &[u8], from_uplink: u16) -> Result<Option<Vec<u8>>> {
        let mut packet = Packet::decode(data)?;

        // Verify session
        if packet.header.session_id != self.session_id {
            return Err(Error::Protocol(
                crate::error::ProtocolError::InvalidVersion {
                    expected: 0,
                    got: 1,
                },
            ));
        }

        if packet.header.uplink_id != from_uplink {
            return Err(Error::InvalidPacket("Uplink identity mismatch".into()));
        }
        if packet.header.flags.has(PacketFlags::ENCRYPTED) {
            let uplink = self
                .get_uplink(from_uplink)
                .ok_or(Error::NoAvailableUplinks)?;
            packet.payload = uplink.decrypt_packet(
                &packet.payload,
                packet.header.sequence.0,
                packet.header.packet_type as u8,
            )?;
        } else if self.assigned_ipv4().is_some() {
            return Err(Error::InvalidPacket("Unauthenticated exit packet".into()));
        }
        if packet.header.packet_type == PacketType::Data {
            self.next_return_sequence
                .fetch_max(packet.header.sequence.0.saturating_add(1), Ordering::AcqRel);
        }
        if packet.header.packet_type == PacketType::Data
            && self.config.deduplication
            && self
                .dedup_window
                .write()
                .is_duplicate(packet.header.sequence.0)
        {
            return Ok(None);
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
                let payload = packet.payload.clone();

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
                    let mut ready_packets = ready_packets.into_iter();
                    if let Some(first) = ready_packets.next() {
                        self.ready_packets.lock().extend(ready_packets);
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
                self.handle_ack(&packet, from_uplink)?;
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
        // The TUN exit uses the inner transport's reliability. The older relay
        // runtime requires outer acknowledgements for its retransmit queue.
        if self.assigned_ipv4().is_some() {
            return Ok(None);
        }
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
            let mut scan_interval = tokio::time::interval(Duration::from_millis(100));

            loop {
                tokio::select! {
                    _ = scan_interval.tick() => {
                        if *manager.state.read() != ConnectionState::Connected || tx.is_closed() {
                            break;
                        }

                        for uplink in manager
                            .uplinks
                            .iter()
                            .filter(|entry| entry.value().state().connection_state == ConnectionState::Connected)
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
                        let mut ready: Vec<_> = manager.ready_packets.lock().drain(..).collect();
                        ready.extend(manager.poll_reorder_buffer());
                        for payload in ready {
                            if tx.send((payload, 0)).await.is_err() {
                                return;
                            }
                        }
                    }
                    event = receive_rx.recv() => {
                        match event {
                            Some(ReceiveLoopEvent::Packet { generation, uplink_id, uplink_name, data }) => {
                                if !manager.get_uplink(uplink_id).is_some_and(|u| u.generation() == generation) { continue; }
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
                if *manager.state.read() != ConnectionState::Connected
                    || uplink.state().connection_state != ConnectionState::Connected
                    || receive_tx.is_closed()
                {
                    break;
                }

                let generation = uplink.generation();
                match tokio::time::timeout(Duration::from_secs(1), uplink.recv(&mut buf)).await {
                    Ok(Ok(len)) => {
                        if receive_tx
                            .send(ReceiveLoopEvent::Packet {
                                generation,
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

    /// Retire every copy on the first authenticated acknowledgement.
    fn handle_ack(&self, packet: &Packet, from_uplink: u16) -> Result<()> {
        // ACK encoding is an eight-byte count followed by that many u64 values.
        // Validate the count before deserialization can allocate from untrusted input.
        if packet.payload.len() < 8 || (packet.payload.len() - 8) % 8 != 0 {
            return Err(Error::InvalidPacket(
                "invalid acknowledgement length".into(),
            ));
        }
        let count = u64::from_le_bytes(packet.payload[..8].try_into().unwrap());
        if count != ((packet.payload.len() - 8) / 8) as u64 {
            return Err(Error::InvalidPacket("invalid acknowledgement count".into()));
        }
        for chunk in packet.payload[8..].chunks_exact(8) {
            let seq = u64::from_le_bytes(chunk.try_into().unwrap());
            let pending = self.pending.lock().remove(seq);
            if let Some(pending) = pending {
                let packet_size = pending.packet.payload.len() as u64;
                for (id, sent_at) in pending.attempts {
                    if let Some(uplink) = self.get_uplink(id) {
                        if id == from_uplink {
                            let rtt = sent_at.elapsed();
                            uplink.record_rtt(rtt);
                            if self.config.throughput_optimization {
                                self.throughput_optimizer.record_ack(id, packet_size, rtt);
                                if !rtt.is_zero() {
                                    self.throughput_optimizer.record_bandwidth(
                                        id,
                                        packet_size as f64 / rtt.as_secs_f64(),
                                    );
                                }
                                self.throughput_optimizer.record_mtu_result(
                                    id,
                                    packet_size as u32,
                                    true,
                                );
                            }
                        } else {
                            uplink.cancel_in_flight();
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Handle pong packet.
    fn handle_pong(&self, packet: &Packet, from_uplink: u16) {
        if let Some(result) = crate::protocol::mtu::acknowledged(&packet.payload) {
            let Ok(mtu) = result else {
                return;
            };
            self.negotiated_mtu.fetch_min(mtu, Ordering::AcqRel);
        } else if crate::protocol::mtu::requested(&packet.payload).is_some_and(|result| {
            result.is_ok_and(|mtu| mtu < crate::protocol::mtu::DEFAULT_INNER_MTU)
        }) {
            // An older exit echoes the request but cannot confirm a lower cap.
            // Quarantine this path so it cannot keep sending oversized records.
            if let Some(uplink) = self.get_uplink(from_uplink) {
                uplink.record_failure("exit cannot negotiate the reduced path MTU");
                uplink.set_connection_state(ConnectionState::Failed);
            }
            return;
        }
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
                    uplink.record_probe_success(rtt);

                    // Also record with throughput optimizer
                    if self.config.throughput_optimization {
                        self.throughput_optimizer
                            .record_ack(from_uplink, payload_size, rtt);
                    }
                }
            }
        }
    }

    /// Retry a bounded snapshot. No buffer lock survives a network operation.
    pub async fn retry_pending(&self) {
        let Ok(_retry_guard) = self.retry_lock.try_lock() else {
            return;
        };
        let now = Instant::now();
        let mut retry = Vec::new();
        let mut failed = Vec::new();
        {
            let mut buffer = self.pending.lock();
            let mut expired = Vec::new();
            for (&seq, pending) in &mut buffer.packets {
                let rto = self
                    .get_uplink(pending.uplink_id)
                    .map_or(Duration::from_millis(500), |u| {
                        u.rtt().saturating_mul(2).max(Duration::from_millis(100))
                    });
                if (self.assigned_ipv4().is_some()
                    && pending.packet.payload.len() > usize::from(self.negotiated_mtu()))
                    || now.duration_since(pending.created_at) >= self.config.max_pending_age
                    || (now.duration_since(pending.sent_at) >= rto
                        && pending.retries >= self.config.send_retries)
                {
                    expired.push(seq);
                } else if now.duration_since(pending.sent_at) >= rto {
                    // Count the retry even during a total outage. Otherwise expired
                    // attempts would be charged as lost on every maintenance tick.
                    pending.retries += 1;
                    pending.sent_at = now;
                    retry.push((
                        seq,
                        pending.packet.payload.clone(),
                        pending.uplink_id,
                        std::mem::take(&mut pending.attempts),
                    ));
                }
            }
            for seq in expired {
                if let Some(packet) = buffer.remove(seq) {
                    failed.push((seq, packet.attempts));
                }
            }
        }
        for (seq, attempts) in failed {
            for id in attempts.keys() {
                if let Some(uplink) = self.get_uplink(*id) {
                    uplink.record_loss();
                }
            }
            self.stats.write().packets_dropped += 1;
            let _ = self.event_tx.send(MultipathEvent::SendFailed(seq));
        }
        for (seq, data, old_id, attempts) in retry {
            for id in attempts.keys() {
                if let Some(uplink) = self.get_uplink(*id) {
                    uplink.record_loss();
                }
                self.repair_uplink_paths(*id).await;
            }
            let selected = self.scheduler.select(&self.uplinks(), Some(seq));
            let preferred = selected
                .iter()
                .find(|&&id| id != old_id)
                .copied()
                .or_else(|| selected.first().copied());
            let Some(id) = preferred else { continue };
            let mut copies = vec![id];
            if self.config.hedged_packets {
                self.extend_with_hedged_uplinks(&mut copies);
            }
            let mut sent = false;
            for id in copies {
                if let Some(uplink) = self.get_uplink(id) {
                    match self
                        .send_on_uplink(&uplink, SequenceNumber(seq), &data)
                        .await
                    {
                        Ok(()) => {
                            sent = true;
                            if let Some(pending) = self.pending.lock().packets.get_mut(&seq) {
                                pending.uplink_id = id;
                            }
                        }
                        Err(e) => {
                            uplink.record_failure(&e.to_string());
                            self.repair_uplink_paths(id).await;
                        }
                    }
                }
            }
            if sent {
                self.stats.write().packets_retransmitted += 1;
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

    fn start_due_reconnects(self: &Arc<Self>, tasks: &mut tokio::task::JoinSet<()>) {
        if self.closing.load(Ordering::Acquire) {
            return;
        }
        let Some(public) = *self.remote_public.read() else {
            return;
        };
        let lifecycle = self.lifecycle.load(Ordering::Acquire);
        for uplink in self.uplinks() {
            let state = uplink.state().connection_state;
            if matches!(
                state,
                ConnectionState::Connecting | ConnectionState::Handshaking
            ) || (state == ConnectionState::Connected
                && uplink.silence() < self.config.peer_silence_timeout)
            {
                continue;
            }
            let id = uplink.numeric_id();
            {
                let mut retries = self.reconnect.lock();
                let retry = retries.entry(id).or_insert(ReconnectState {
                    failures: 0,
                    next_attempt: Instant::now(),
                    in_progress: false,
                });
                if retry.in_progress || retry.next_attempt > Instant::now() {
                    continue;
                }
                retry.in_progress = true;
            }
            let manager = Arc::clone(self);
            tasks.spawn(async move {
                let result = manager.connect_uplink(&uplink, &public).await;
                if manager.closing.load(Ordering::Acquire)
                    || manager.lifecycle.load(Ordering::Acquire) != lifecycle
                {
                    return;
                }
                let success = result.is_ok();
                if let Err(error) = result {
                    uplink.set_connection_state(ConnectionState::Failed);
                    uplink.record_failure(&error.to_string());
                    tracing::debug!(uplink = %uplink.id(), %error, "Path re-establishment failed");
                } else {
                    tracing::info!(uplink = %uplink.id(), "Path re-established");
                }
                let mut retries = manager.reconnect.lock();
                if let Some(retry) = retries.get_mut(&id) {
                    retry.in_progress = false;
                    retry.failures = if success {
                        0
                    } else {
                        retry.failures.saturating_add(1)
                    };
                    retry.next_attempt = Instant::now()
                        + reconnect_delay(
                            &manager.config,
                            retry.failures,
                            0.8 + rand::random::<f64>() * 0.4,
                        );
                }
            });
        }
    }

    /// Start the maintenance loop in a background task.
    /// This handles retries, keepalives, uplink health monitoring, and throughput optimization.
    pub fn start_maintenance_loop(self: &Arc<Self>) {
        let lifecycle = self.lifecycle.load(Ordering::Acquire);
        if self
            .maintenance_generation
            .swap(lifecycle, Ordering::AcqRel)
            == lifecycle
        {
            return;
        }
        let manager = Arc::clone(self);

        tokio::spawn(async move {
            let mut reconnect_tasks = tokio::task::JoinSet::new();
            let mut retry_interval = tokio::time::interval(Duration::from_millis(100));
            let mut maintenance_interval = tokio::time::interval(
                manager
                    .config
                    .health_check_interval
                    .max(Duration::from_millis(20)),
            );
            let mut ping_interval = tokio::time::interval(Duration::from_secs(5));

            // Path discovery interval (Dublin Traceroute: periodic ECMP probing)
            let path_discovery_enabled = manager.assigned_ipv4().is_none()
                && !manager.config.path_discovery_interval.is_zero();
            let mut path_discovery_interval = tokio::time::interval(if path_discovery_enabled {
                manager.config.path_discovery_interval
            } else {
                Duration::from_secs(3600) // Fallback, won't trigger if disabled
            });

            // Bandwidth probing interval for throughput optimization
            let bandwidth_probe_enabled = manager.assigned_ipv4().is_none()
                && manager.config.throughput_optimization
                && manager.config.throughput.probing_enabled;
            let mut bandwidth_probe_interval = tokio::time::interval(if bandwidth_probe_enabled {
                manager.config.bandwidth_probe_interval
            } else {
                Duration::from_secs(3600)
            });

            loop {
                if *manager.state.read() != ConnectionState::Connected
                    || manager.closing.load(Ordering::Acquire)
                    || manager.lifecycle.load(Ordering::Acquire) != lifecycle
                {
                    break;
                }

                tokio::select! {
                    _ = reconnect_tasks.join_next(), if !reconnect_tasks.is_empty() => {}
                    _ = retry_interval.tick() => {
                        manager.retry_pending().await;
                    }
                    _ = maintenance_interval.tick() => {
                        manager.periodic_maintenance();
                        manager.start_due_reconnects(&mut reconnect_tasks);
                    }
                    _ = ping_interval.tick() => {
                        for uplink in manager.uplinks() {
                            if uplink.is_noise_ready() {
                                manager.send_fast_probe(uplink.numeric_id()).await;
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
            let _ = manager.maintenance_generation.compare_exchange(
                lifecycle,
                u64::MAX,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
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

    /// Notify the production exit on every path before stopping local tasks.
    pub async fn disconnect(&self) -> Result<()> {
        self.closing.store(true, Ordering::Release);
        if self.assigned_ipv4().is_some() {
            for uplink in self.uplinks() {
                if let Some(transport) = uplink.get_transport() {
                    let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
                    if let Ok(payload) = uplink.encrypt_packet(&[], seq, PacketType::Close as u8) {
                        let mut packet = Packet::new(
                            PacketType::Close,
                            SequenceNumber(seq),
                            self.session_id,
                            uplink.numeric_id(),
                            payload,
                        )?;
                        packet.set_flag(PacketFlags::ENCRYPTED);
                        let _ = transport.send(&packet.encode()?).await;
                    }
                }
            }
        }
        self.close()
    }

    pub(crate) fn stream_uplinks(&self) -> Vec<Arc<Uplink>> {
        let mut uplinks = self.uplinks();
        let selected = self.scheduler.select(&uplinks, Some(self.allocate_flow()));
        uplinks.retain(|u| u.is_usable());
        uplinks.sort_by_key(|u| {
            (
                selected.first().copied() != Some(u.numeric_id()),
                u.numeric_id(),
            )
        });
        uplinks
    }

    /// Close the connection.
    pub fn close(&self) -> Result<()> {
        self.closing.store(true, Ordering::Release);
        self.lifecycle.fetch_add(1, Ordering::AcqRel);
        let mut state = self.state.write();
        *state = ConnectionState::Disconnecting;

        // Close all uplinks
        for entry in &self.uplinks {
            entry
                .value()
                .set_connection_state(ConnectionState::Disconnecting);
        }

        let sequences: Vec<_> = self.pending.lock().packets.keys().copied().collect();
        for seq in sequences {
            self.cancel_pending(seq);
        }
        self.ready_packets.lock().clear();
        *state = ConnectionState::Disconnected;
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
            0x45, 0x00, 0x00, 0x28, 0x1c, 0x46, 0x40, 0x00, 0x40, 0x06, 0x00, 0x00, 0xc0, 0xa8,
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

#[cfg(test)]
#[path = "resilience_tests.rs"]
mod resilience_tests;
