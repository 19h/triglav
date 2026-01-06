//! Individual uplink management.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use super::nat::UplinkNatState;
use crate::crypto::NoiseSession;
use crate::metrics::QualityMetrics;
use crate::transport::{Transport, TransportProtocol};
use crate::types::{
    Bandwidth, ConnectionState, InterfaceType, Latency, TrafficStats, UplinkHealth, UplinkId,
};

/// Uplink configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UplinkConfig {
    /// Unique identifier.
    pub id: UplinkId,
    /// Network interface name (e.g., "en0", "wlan0").
    pub interface: Option<String>,
    /// Local bind address.
    pub local_addr: Option<SocketAddr>,
    /// Remote server address.
    pub remote_addr: SocketAddr,
    /// Transport protocol.
    #[serde(default)]
    pub protocol: TransportProtocol,
    /// Interface type (for priority calculation).
    #[serde(default)]
    pub interface_type: InterfaceType,
    /// Weight for load balancing (higher = more traffic).
    #[serde(default = "default_weight")]
    pub weight: u32,
    /// Maximum bandwidth limit (0 = unlimited).
    #[serde(default)]
    pub max_bandwidth_mbps: u32,
    /// Enable this uplink.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Priority override (0 = auto-calculate).
    #[serde(default)]
    pub priority_override: u32,
}

fn default_weight() -> u32 {
    100
}
fn default_enabled() -> bool {
    true
}

impl Default for UplinkConfig {
    fn default() -> Self {
        Self {
            id: UplinkId::new("uplink-0"),
            interface: None,
            local_addr: None,
            remote_addr: SocketAddr::from(([0, 0, 0, 0], 0)),
            protocol: TransportProtocol::default(),
            interface_type: InterfaceType::Unknown,
            weight: 100,
            max_bandwidth_mbps: 0,
            enabled: true,
            priority_override: 0,
        }
    }
}

/// Uplink state.
#[derive(Debug)]
pub struct UplinkState {
    /// Current connection state.
    pub connection_state: ConnectionState,
    /// Health status.
    pub health: UplinkHealth,
    /// Last activity time.
    pub last_activity: Instant,
    /// Last successful send.
    pub last_send: Option<Instant>,
    /// Last successful receive.
    pub last_recv: Option<Instant>,
    /// Consecutive failures.
    pub consecutive_failures: u32,
    /// Time of last failure.
    pub last_failure: Option<Instant>,
    /// Error message for last failure.
    pub last_error: Option<String>,
}

impl Default for UplinkState {
    fn default() -> Self {
        Self {
            connection_state: ConnectionState::Disconnected,
            health: UplinkHealth::Unknown,
            last_activity: Instant::now(),
            last_send: None,
            last_recv: None,
            consecutive_failures: 0,
            last_failure: None,
            last_error: None,
        }
    }
}

/// Comprehensive connection parameters for throughput optimization.
/// Contains all metrics needed to make intelligent routing and scheduling decisions.
#[derive(Debug, Clone)]
pub struct ConnectionParams {
    // Identity
    /// Numeric uplink ID.
    pub uplink_id: u16,
    /// Network interface type.
    pub interface_type: InterfaceType,
    /// Transport protocol.
    pub protocol: TransportProtocol,

    // Timing
    /// Smoothed round-trip time.
    pub rtt: Duration,
    /// RTT variance.
    pub rtt_variance: Duration,
    /// Minimum observed RTT.
    pub rtt_min: Duration,
    /// Jitter (variance in delay).
    pub jitter: Duration,

    // Bandwidth
    /// Upload bandwidth estimate.
    pub bandwidth_up: Bandwidth,
    /// Download bandwidth estimate.
    pub bandwidth_down: Bandwidth,
    /// Effective throughput (considering loss).
    pub effective_throughput: Bandwidth,
    /// Goodput (useful throughput after retransmissions).
    pub goodput: Bandwidth,
    /// Maximum bandwidth limit (0 = unlimited).
    pub max_bandwidth_mbps: u32,

    // Capacity
    /// Bandwidth-delay product in bytes.
    pub bdp: u64,
    /// Congestion window in packets.
    pub cwnd: u32,
    /// Packets in flight.
    pub in_flight: u32,

    // Loss
    /// Packet loss ratio (0.0-1.0).
    pub packet_loss: f64,
    /// Retransmission rate (0.0-1.0).
    pub retransmission_rate: f64,

    // Health
    /// Health status.
    pub health: UplinkHealth,
    /// Consecutive failures.
    pub consecutive_failures: u32,
    /// Whether uplink is usable.
    pub is_usable: bool,

    // NAT
    /// Whether behind NAT.
    pub is_natted: bool,
    /// NAT type.
    pub nat_type: super::nat::NatType,

    // Transfer estimates
    /// Estimated time to transfer 1MB.
    pub time_for_1mb: Duration,
    /// Computed priority score.
    pub priority_score: u32,

    // Statistics
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total packets received.
    pub packets_received: u64,
    /// Packets dropped.
    pub packets_dropped: u64,
    /// Packets retransmitted.
    pub packets_retransmitted: u64,
}

impl ConnectionParams {
    /// Check if this uplink is suitable for latency-sensitive traffic.
    pub fn is_low_latency(&self) -> bool {
        self.rtt < Duration::from_millis(50) && self.jitter < Duration::from_millis(10)
    }

    /// Check if this uplink is suitable for high-bandwidth traffic.
    pub fn is_high_bandwidth(&self) -> bool {
        self.effective_throughput.bytes_per_sec > 10_000_000.0 // 10 MB/s
    }

    /// Check if this uplink has low packet loss.
    pub fn has_low_loss(&self) -> bool {
        self.packet_loss < 0.01 // Less than 1%
    }

    /// Calculate a quality score (0.0-1.0) for this uplink.
    pub fn quality_score(&self) -> f64 {
        let rtt_score = 1.0 / (1.0 + self.rtt.as_secs_f64() * 10.0);
        let loss_score = 1.0 - self.packet_loss.min(1.0);
        let jitter_score = 1.0 / (1.0 + self.jitter.as_secs_f64() * 100.0);
        let health_score = self.health.priority_modifier();

        (rtt_score + loss_score + jitter_score + health_score) / 4.0
    }

    /// Estimate transfer time for given bytes.
    pub fn transfer_time(&self, bytes: u64) -> Duration {
        if self.effective_throughput.bytes_per_sec <= 0.0 {
            return Duration::MAX;
        }

        let transfer_secs = bytes as f64 / self.effective_throughput.bytes_per_sec;
        // Cap at 1 hour to avoid Duration overflow
        let total_secs = (transfer_secs + self.rtt.as_secs_f64()).min(3600.0);
        Duration::from_secs_f64(total_secs)
    }

    /// Check if this uplink can complete a transfer faster than another.
    pub fn faster_than(&self, other: &Self, bytes: u64) -> bool {
        self.transfer_time(bytes) < other.transfer_time(bytes)
    }
}

/// RTT tracking with sliding window.
#[derive(Debug)]
struct RttTracker {
    samples: VecDeque<Duration>,
    max_samples: usize,
    smoothed_rtt: Duration,
    rtt_var: Duration,
}

impl RttTracker {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            smoothed_rtt: Duration::ZERO,
            rtt_var: Duration::ZERO,
        }
    }

    fn add_sample(&mut self, rtt: Duration) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(rtt);

        // Update smoothed RTT using Jacobson/Karels algorithm
        if self.smoothed_rtt == Duration::ZERO {
            self.smoothed_rtt = rtt;
            self.rtt_var = rtt / 2;
        } else {
            let rtt_f = rtt.as_secs_f64();
            let srtt_f = self.smoothed_rtt.as_secs_f64();
            let rttvar_f = self.rtt_var.as_secs_f64();

            let delta = (rtt_f - srtt_f).abs();
            self.rtt_var = Duration::from_secs_f64(rttvar_f * 0.75 + delta * 0.25);
            self.smoothed_rtt = Duration::from_secs_f64(srtt_f * 0.875 + rtt_f * 0.125);
        }
    }

    fn get_latency(&self) -> Option<Latency> {
        if self.samples.is_empty() {
            return None;
        }

        let samples: Vec<_> = self.samples.iter().copied().collect();
        Latency::from_samples(&samples)
    }

    fn smoothed(&self) -> Duration {
        self.smoothed_rtt
    }

    fn variance(&self) -> Duration {
        self.rtt_var
    }
}

/// Bandwidth estimator using exponential moving average.
#[derive(Debug)]
struct BandwidthEstimator {
    bytes_sent: AtomicU64,
    bytes_recv: AtomicU64,
    last_update: RwLock<Instant>,
    send_rate: RwLock<f64>,
    recv_rate: RwLock<f64>,
}

impl BandwidthEstimator {
    fn new() -> Self {
        Self {
            bytes_sent: AtomicU64::new(0),
            bytes_recv: AtomicU64::new(0),
            last_update: RwLock::new(Instant::now()),
            send_rate: RwLock::new(0.0),
            recv_rate: RwLock::new(0.0),
        }
    }

    fn record_sent(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_recv(&self, bytes: u64) {
        self.bytes_recv.fetch_add(bytes, Ordering::Relaxed);
    }

    fn update(&self) {
        let now = Instant::now();
        let mut last = self.last_update.write();
        let elapsed = now.duration_since(*last).as_secs_f64();

        if elapsed < 0.1 {
            return; // Don't update too frequently
        }

        let sent = self.bytes_sent.swap(0, Ordering::Relaxed) as f64;
        let recv = self.bytes_recv.swap(0, Ordering::Relaxed) as f64;

        let new_send_rate = sent / elapsed;
        let new_recv_rate = recv / elapsed;

        // EMA update
        let alpha = super::EMA_ALPHA;
        let mut send_rate = self.send_rate.write();
        let mut recv_rate = self.recv_rate.write();

        *send_rate = alpha * new_send_rate + (1.0 - alpha) * *send_rate;
        *recv_rate = alpha * new_recv_rate + (1.0 - alpha) * *recv_rate;

        *last = now;
    }

    fn send_bandwidth(&self) -> Bandwidth {
        Bandwidth::from_bps(*self.send_rate.read())
    }

    fn recv_bandwidth(&self) -> Bandwidth {
        Bandwidth::from_bps(*self.recv_rate.read())
    }
}

/// Loss tracker for packet loss estimation.
#[derive(Debug)]
struct LossTracker {
    packets_sent: AtomicU64,
    packets_acked: AtomicU64,
    packets_lost: AtomicU64,
    window: RwLock<VecDeque<(u64, u64, u64)>>, // (sent, acked, lost) per window
    window_size: usize,
}

impl LossTracker {
    fn new(window_size: usize) -> Self {
        Self {
            packets_sent: AtomicU64::new(0),
            packets_acked: AtomicU64::new(0),
            packets_lost: AtomicU64::new(0),
            window: RwLock::new(VecDeque::with_capacity(window_size)),
            window_size,
        }
    }

    fn record_sent(&self) {
        self.packets_sent.fetch_add(1, Ordering::Relaxed);
    }

    fn record_acked(&self) {
        self.packets_acked.fetch_add(1, Ordering::Relaxed);
    }

    fn record_lost(&self) {
        self.packets_lost.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot_window(&self) {
        let sent = self.packets_sent.swap(0, Ordering::Relaxed);
        let acked = self.packets_acked.swap(0, Ordering::Relaxed);
        let lost = self.packets_lost.swap(0, Ordering::Relaxed);

        let mut window = self.window.write();
        if window.len() >= self.window_size {
            window.pop_front();
        }
        window.push_back((sent, acked, lost));
    }

    fn loss_ratio(&self) -> f64 {
        let window = self.window.read();
        let total_sent: u64 = window.iter().map(|(s, _, _)| s).sum();
        let total_lost: u64 = window.iter().map(|(_, _, l)| l).sum();

        if total_sent == 0 {
            0.0
        } else {
            total_lost as f64 / total_sent as f64
        }
    }
}

/// An individual network uplink.
pub struct Uplink {
    /// Configuration.
    config: UplinkConfig,
    /// Numeric ID for packet headers.
    numeric_id: u16,
    /// Current state.
    state: RwLock<UplinkState>,
    /// RTT tracker.
    rtt: RwLock<RttTracker>,
    /// Bandwidth estimator.
    bandwidth: BandwidthEstimator,
    /// Loss tracker.
    loss: LossTracker,
    /// Traffic statistics.
    stats: RwLock<TrafficStats>,
    /// Transport (set after connection). Using Arc so we can clone for async operations.
    transport: RwLock<Option<Arc<dyn Transport>>>,
    /// Noise session (set after handshake). Using Mutex for async compatibility.
    noise_session: Mutex<Option<NoiseSession>>,
    /// Computed priority score.
    priority_score: AtomicU32,
    /// In-flight packets count.
    in_flight: AtomicU32,
    /// Maximum in-flight (congestion window).
    cwnd: AtomicU32,
    /// NAT detection state (Dublin Traceroute-inspired).
    nat_state: RwLock<UplinkNatState>,
}

impl Uplink {
    /// Create a new uplink.
    pub fn new(config: UplinkConfig, numeric_id: u16) -> Self {
        Self {
            config,
            numeric_id,
            state: RwLock::new(UplinkState::default()),
            rtt: RwLock::new(RttTracker::new(100)),
            bandwidth: BandwidthEstimator::new(),
            loss: LossTracker::new(10),
            stats: RwLock::new(TrafficStats::default()),
            transport: RwLock::new(None),
            noise_session: Mutex::new(None),
            priority_score: AtomicU32::new(0),
            in_flight: AtomicU32::new(0),
            cwnd: AtomicU32::new(10), // Initial cwnd
            nat_state: RwLock::new(UplinkNatState::default()),
        }
    }

    /// Get the uplink ID.
    pub fn id(&self) -> &UplinkId {
        &self.config.id
    }

    /// Get the numeric ID (for packet headers).
    pub fn numeric_id(&self) -> u16 {
        self.numeric_id
    }

    /// Get the configuration.
    pub fn config(&self) -> &UplinkConfig {
        &self.config
    }

    /// Get current state.
    pub fn state(&self) -> UplinkState {
        let state = self.state.read();
        UplinkState {
            connection_state: state.connection_state,
            health: state.health,
            last_activity: state.last_activity,
            last_send: state.last_send,
            last_recv: state.last_recv,
            consecutive_failures: state.consecutive_failures,
            last_failure: state.last_failure,
            last_error: state.last_error.clone(),
        }
    }

    /// Get health status.
    pub fn health(&self) -> UplinkHealth {
        self.state.read().health
    }

    /// Check if uplink is usable.
    pub fn is_usable(&self) -> bool {
        let state = self.state.read();
        self.config.enabled
            && state.connection_state == ConnectionState::Connected
            && state.health.is_usable()
    }

    /// Get the priority score.
    pub fn priority_score(&self) -> u32 {
        self.priority_score.load(Ordering::Relaxed)
    }

    /// Get smoothed RTT.
    pub fn rtt(&self) -> Duration {
        self.rtt.read().smoothed()
    }

    /// Get current bandwidth estimate.
    pub fn bandwidth(&self) -> Bandwidth {
        self.bandwidth.recv_bandwidth()
    }

    /// Get packet loss ratio.
    pub fn loss_ratio(&self) -> f64 {
        self.loss.loss_ratio()
    }

    /// Get traffic statistics.
    pub fn stats(&self) -> TrafficStats {
        *self.stats.read()
    }

    /// Get congestion window.
    pub fn cwnd(&self) -> u32 {
        self.cwnd.load(Ordering::Relaxed)
    }

    /// Get in-flight packet count.
    pub fn in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Check if congestion window allows sending.
    pub fn can_send(&self) -> bool {
        self.in_flight.load(Ordering::Relaxed) < self.cwnd.load(Ordering::Relaxed)
    }

    /// Set the transport.
    pub fn set_transport(&self, transport: Arc<dyn Transport>) {
        *self.transport.write() = Some(transport);
    }

    /// Get the transport (cloned Arc).
    pub fn get_transport(&self) -> Option<Arc<dyn Transport>> {
        self.transport.read().clone()
    }

    /// Set the noise session.
    pub fn set_noise_session(&self, session: NoiseSession) {
        *self.noise_session.lock() = Some(session);
    }

    /// Send raw data through the transport (no encryption).
    pub async fn send_raw(&self, data: &[u8]) -> crate::error::Result<usize> {
        let transport = self
            .get_transport()
            .ok_or_else(|| crate::error::TransportError::SendFailed("no transport".into()))?;

        let len = transport.send(data).await?;
        self.record_send(len);
        Ok(len)
    }

    /// Send encrypted data through the transport.
    /// Returns the number of bytes sent (original data size).
    pub async fn send_encrypted(&self, data: &[u8]) -> crate::error::Result<usize> {
        // Encrypt the data
        let ciphertext = {
            let mut noise = self.noise_session.lock();
            let session = noise.as_mut().ok_or_else(|| {
                crate::error::CryptoError::NoiseProtocol("no noise session".into())
            })?;

            if !session.is_transport() {
                return Err(crate::error::CryptoError::NoiseProtocol(
                    "handshake not complete".into(),
                )
                .into());
            }

            session.encrypt(data)?
        };

        // Get transport and send
        let transport = self
            .get_transport()
            .ok_or_else(|| crate::error::TransportError::SendFailed("no transport".into()))?;

        transport.send(&ciphertext).await?;
        self.record_send(ciphertext.len());
        Ok(data.len())
    }

    /// Receive data from transport.
    pub async fn recv(&self, buf: &mut [u8]) -> crate::error::Result<usize> {
        let transport = self
            .get_transport()
            .ok_or_else(|| crate::error::TransportError::ReceiveFailed("no transport".into()))?;

        let len = transport.recv(buf).await?;
        self.record_recv(len);
        Ok(len)
    }

    /// Receive data from transport with source address.
    pub async fn recv_from(
        &self,
        buf: &mut [u8],
    ) -> crate::error::Result<(usize, std::net::SocketAddr)> {
        let transport = self
            .get_transport()
            .ok_or_else(|| crate::error::TransportError::ReceiveFailed("no transport".into()))?;

        let (len, addr) = transport.recv_from(buf).await?;
        self.record_recv(len);
        Ok((len, addr))
    }

    /// Decrypt received data.
    pub fn decrypt(&self, ciphertext: &[u8]) -> crate::error::Result<Vec<u8>> {
        let mut noise = self.noise_session.lock();
        let session = noise
            .as_mut()
            .ok_or_else(|| crate::error::CryptoError::NoiseProtocol("no noise session".into()))?;

        if !session.is_transport() {
            return Err(
                crate::error::CryptoError::NoiseProtocol("handshake not complete".into()).into(),
            );
        }

        Ok(session.decrypt(ciphertext)?)
    }

    /// Check if the noise session is ready for transport.
    pub fn is_noise_ready(&self) -> bool {
        self.noise_session
            .lock()
            .as_ref()
            .is_some_and(crate::crypto::NoiseSession::is_transport)
    }

    /// Encrypt data (get ciphertext without sending).
    pub fn encrypt(&self, data: &[u8]) -> crate::error::Result<Vec<u8>> {
        let mut noise = self.noise_session.lock();
        let session = noise
            .as_mut()
            .ok_or_else(|| crate::error::CryptoError::NoiseProtocol("no noise session".into()))?;

        if !session.is_transport() {
            return Err(
                crate::error::CryptoError::NoiseProtocol("handshake not complete".into()).into(),
            );
        }

        Ok(session.encrypt(data)?)
    }

    /// Write handshake message.
    pub fn write_handshake(&self, payload: &[u8]) -> crate::error::Result<Vec<u8>> {
        let mut noise = self.noise_session.lock();
        let session = noise
            .as_mut()
            .ok_or_else(|| crate::error::CryptoError::NoiseProtocol("no noise session".into()))?;
        Ok(session.write_handshake(payload)?)
    }

    /// Read handshake message.
    pub fn read_handshake(&self, message: &[u8]) -> crate::error::Result<Vec<u8>> {
        let mut noise = self.noise_session.lock();
        let session = noise
            .as_mut()
            .ok_or_else(|| crate::error::CryptoError::NoiseProtocol("no noise session".into()))?;
        Ok(session.read_handshake(message)?)
    }

    /// Update connection state.
    pub fn set_connection_state(&self, state: ConnectionState) {
        let mut s = self.state.write();
        s.connection_state = state;
        s.last_activity = Instant::now();

        // Update health based on state
        if state == ConnectionState::Connected {
            s.consecutive_failures = 0;
            if s.health == UplinkHealth::Down || s.health == UplinkHealth::Unknown {
                s.health = UplinkHealth::Healthy;
            }
        } else if state == ConnectionState::Failed {
            s.health = UplinkHealth::Down;
        }
    }

    /// Record a successful send.
    pub fn record_send(&self, bytes: usize) {
        {
            let mut uplink_state = self.state.write();
            uplink_state.last_send = Some(Instant::now());
            uplink_state.last_activity = Instant::now();
        }

        self.bandwidth.record_sent(bytes as u64);
        self.loss.record_sent();
        self.in_flight.fetch_add(1, Ordering::Relaxed);

        let mut traffic = self.stats.write();
        traffic.bytes_sent += bytes as u64;
        traffic.packets_sent += 1;
    }

    /// Record a successful receive.
    pub fn record_recv(&self, bytes: usize) {
        {
            let mut uplink_state = self.state.write();
            uplink_state.last_recv = Some(Instant::now());
            uplink_state.last_activity = Instant::now();
        }

        self.bandwidth.record_recv(bytes as u64);

        let mut traffic = self.stats.write();
        traffic.bytes_received += bytes as u64;
        traffic.packets_received += 1;
    }

    /// Record an RTT measurement.
    pub fn record_rtt(&self, rtt: Duration) {
        self.rtt.write().add_sample(rtt);
        self.loss.record_acked();
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
        self.update_priority();
    }

    /// Record a packet loss.
    pub fn record_loss(&self) {
        self.loss.record_lost();
        self.in_flight.fetch_sub(1, Ordering::Relaxed);

        let mut stats = self.stats.write();
        stats.packets_dropped += 1;

        // Reduce cwnd on loss (multiplicative decrease)
        let cwnd = self.cwnd.load(Ordering::Relaxed);
        self.cwnd.store((cwnd / 2).max(2), Ordering::Relaxed);

        self.update_health();
    }

    /// Record a failure.
    pub fn record_failure(&self, error: &str) {
        let mut state = self.state.write();
        state.consecutive_failures += 1;
        state.last_failure = Some(Instant::now());
        state.last_error = Some(error.to_string());

        // Update health based on consecutive failures
        state.health = match state.consecutive_failures {
            0..=2 => UplinkHealth::Healthy,
            3..=5 => UplinkHealth::Degraded,
            6..=10 => UplinkHealth::Unhealthy,
            _ => UplinkHealth::Down,
        };
    }

    /// Record a successful operation (resets failure counter).
    pub fn record_success(&self) {
        let mut state = self.state.write();
        state.consecutive_failures = 0;
        state.last_activity = Instant::now();

        // Slowly increase cwnd on success (additive increase)
        let cwnd = self.cwnd.load(Ordering::Relaxed);
        self.cwnd.store((cwnd + 1).min(1000), Ordering::Relaxed);

        // Improve health if currently degraded
        if state.health == UplinkHealth::Degraded || state.health == UplinkHealth::Unhealthy {
            state.health = UplinkHealth::Healthy;
        }
    }

    /// Update priority score based on current metrics.
    fn update_priority(&self) {
        let base = if self.config.priority_override > 0 {
            self.config.priority_override
        } else {
            self.config.interface_type.base_priority()
        };

        let health_mod = self.state.read().health.priority_modifier();
        let rtt_score = self.rtt_score();
        let loss_score = self.loss_score();

        let score = (f64::from(base) * health_mod * rtt_score * loss_score) as u32;
        self.priority_score.store(score, Ordering::Relaxed);
    }

    /// Calculate RTT-based score (lower RTT = higher score).
    fn rtt_score(&self) -> f64 {
        let rtt = self.rtt.read().smoothed();
        if rtt == Duration::ZERO {
            return 1.0;
        }

        let rtt_ms = rtt.as_secs_f64() * 1000.0;
        // Score decreases with RTT: 1.0 at 0ms, 0.5 at 100ms, 0.1 at 500ms
        1.0 / (1.0 + rtt_ms / 100.0)
    }

    /// Calculate loss-based score (lower loss = higher score).
    fn loss_score(&self) -> f64 {
        let loss = self.loss.loss_ratio();
        // Score decreases with loss: 1.0 at 0%, 0.5 at 5%, 0.1 at 20%
        1.0 - loss.min(1.0)
    }

    /// Update health based on current metrics.
    fn update_health(&self) {
        let loss = self.loss.loss_ratio();
        let rtt = self.rtt.read().smoothed();

        let mut state = self.state.write();

        // Determine health based on metrics
        let new_health = if loss > 0.3 || rtt > Duration::from_secs(5) {
            UplinkHealth::Unhealthy
        } else if loss > 0.1 || rtt > Duration::from_secs(1) {
            UplinkHealth::Degraded
        } else if state.consecutive_failures > 5 {
            UplinkHealth::Unhealthy
        } else {
            UplinkHealth::Healthy
        };

        state.health = new_health;
    }

    /// Periodic update (call from timer).
    pub fn periodic_update(&self) {
        self.bandwidth.update();
        self.loss.snapshot_window();
        self.update_priority();
        self.update_health();

        // Check for timeout
        let state = self.state.read();
        if state.connection_state == ConnectionState::Connected {
            if let Some(last_recv) = state.last_recv {
                if last_recv.elapsed() > super::DEFAULT_UPLINK_TIMEOUT {
                    drop(state);
                    self.record_failure("receive timeout");
                }
            }
        }
    }

    /// Get quality metrics.
    pub fn quality_metrics(&self) -> QualityMetrics {
        let rtt = self.rtt.read();
        QualityMetrics {
            rtt: rtt.smoothed(),
            rtt_variance: rtt.variance(),
            latency: rtt.get_latency(),
            bandwidth_up: self.bandwidth.send_bandwidth(),
            bandwidth_down: self.bandwidth.recv_bandwidth(),
            packet_loss: self.loss.loss_ratio(),
            jitter: rtt.variance(),
            health: self.state.read().health,
        }
    }

    /// Get comprehensive connection parameters for throughput optimization.
    /// This provides all metrics needed to make intelligent routing decisions.
    pub fn connection_params(&self) -> ConnectionParams {
        let rtt = self.rtt.read();
        let state = self.state.read();
        let stats = *self.stats.read();

        let bandwidth_up = self.bandwidth.send_bandwidth();
        let bandwidth_down = self.bandwidth.recv_bandwidth();
        let smoothed_rtt = rtt.smoothed();
        let loss = self.loss.loss_ratio();

        // Calculate bandwidth-delay product
        let bdp = if smoothed_rtt > Duration::ZERO {
            (bandwidth_down.bytes_per_sec * smoothed_rtt.as_secs_f64()) as u64
        } else {
            0
        };

        // Calculate effective throughput (considering loss)
        let loss_factor = (1.0 - loss).max(0.01);
        let effective_throughput = bandwidth_down.bytes_per_sec * loss_factor;

        // Estimate time to transfer 1MB
        let time_for_1mb = if effective_throughput > 0.0 {
            let secs =
                ((1024.0 * 1024.0) / effective_throughput + smoothed_rtt.as_secs_f64()).min(3600.0);
            Duration::from_secs_f64(secs)
        } else {
            Duration::from_secs(3600) // 1 hour max
        };

        // Calculate goodput (actual useful throughput)
        let total_sent = stats.bytes_sent.max(1);
        let retransmission_rate =
            stats.packets_retransmitted as f64 / stats.packets_sent.max(1) as f64;
        let goodput = bandwidth_down.bytes_per_sec * (1.0 - retransmission_rate);

        ConnectionParams {
            // Identity
            uplink_id: self.numeric_id,
            interface_type: self.config.interface_type,
            protocol: self.config.protocol,

            // Timing
            rtt: smoothed_rtt,
            rtt_variance: rtt.variance(),
            rtt_min: rtt.samples.front().copied().unwrap_or(Duration::ZERO),
            jitter: rtt.variance(),

            // Bandwidth
            bandwidth_up,
            bandwidth_down,
            effective_throughput: Bandwidth::from_bps(effective_throughput),
            goodput: Bandwidth::from_bps(goodput),
            max_bandwidth_mbps: self.config.max_bandwidth_mbps,

            // Capacity
            bdp,
            cwnd: self.cwnd.load(Ordering::Relaxed),
            in_flight: self.in_flight.load(Ordering::Relaxed),

            // Loss
            packet_loss: loss,
            retransmission_rate,

            // Health
            health: state.health,
            consecutive_failures: state.consecutive_failures,
            is_usable: self.config.enabled
                && state.connection_state == ConnectionState::Connected
                && state.health.is_usable(),

            // NAT
            is_natted: self.nat_state.read().is_natted(),
            nat_type: self.nat_state.read().nat_type(),

            // Transfer estimates
            time_for_1mb,
            priority_score: self.priority_score.load(Ordering::Relaxed),

            // Statistics
            bytes_sent: stats.bytes_sent,
            bytes_received: stats.bytes_received,
            packets_sent: stats.packets_sent,
            packets_received: stats.packets_received,
            packets_dropped: stats.packets_dropped,
            packets_retransmitted: stats.packets_retransmitted,
        }
    }

    /// Get send bandwidth.
    pub fn send_bandwidth(&self) -> Bandwidth {
        self.bandwidth.send_bandwidth()
    }

    /// Get receive bandwidth.
    pub fn recv_bandwidth(&self) -> Bandwidth {
        self.bandwidth.recv_bandwidth()
    }

    /// Get RTT variance (jitter).
    pub fn rtt_variance(&self) -> Duration {
        self.rtt.read().variance()
    }

    // NAT detection methods (Dublin Traceroute-inspired)

    /// Check if this uplink is behind NAT.
    pub fn is_natted(&self) -> bool {
        self.nat_state.read().is_natted()
    }

    /// Get the NAT type detected on this uplink.
    pub fn nat_type(&self) -> super::nat::NatType {
        self.nat_state.read().nat_type()
    }

    /// Get the external address (after NAT translation).
    pub fn external_addr(&self) -> Option<std::net::SocketAddr> {
        self.nat_state.read().external_addr()
    }

    /// Get NAT detection state for advanced operations.
    pub fn nat_detection_state(&self) -> super::nat::NatDetectionState {
        self.nat_state.read().detection_state().clone()
    }

    /// Update NAT state with access to mutable reference.
    pub fn update_nat_state<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut UplinkNatState) -> R,
    {
        f(&mut self.nat_state.write())
    }

    /// Reset NAT detection state.
    pub fn reset_nat_state(&self) {
        self.nat_state.write().reset();
    }
}

// Intentionally abbreviated Debug output - internal state is not relevant for debugging
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Uplink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.read();
        f.debug_struct("Uplink")
            .field("id", &self.config.id)
            .field("numeric_id", &self.numeric_id)
            .field("state", &state.connection_state)
            .field("health", &state.health)
            .field("rtt", &self.rtt())
            .field("loss", &format!("{:.1}%", self.loss_ratio() * 100.0))
            .field("priority", &self.priority_score())
            .finish()
    }
}
