//! Throughput optimization for maximum bandwidth utilization.
//!
//! This module implements comprehensive throughput optimization including:
//! - Bandwidth-delay product (BDP) awareness
//! - Active bandwidth probing and capacity estimation
//! - MTU/packet size optimization with path MTU discovery
//! - Effective throughput scoring combining bandwidth and latency
//! - Send-side pacing based on measured capacity
//! - Latency-aware flow assignment to prevent high-latency blocking
//! - Frame size optimization

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::Uplink;
use crate::types::Bandwidth;

/// Default MTU for most networks.
pub const DEFAULT_MTU: u32 = 1500;

/// Minimum MTU we'll use (IPv6 minimum).
pub const MIN_MTU: u32 = 1280;

/// Maximum MTU for jumbo frames.
pub const MAX_MTU: u32 = 9000;

/// Default pacing interval.
pub const DEFAULT_PACING_INTERVAL: Duration = Duration::from_micros(100);

/// Throughput optimizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputConfig {
    /// Enable active bandwidth probing.
    #[serde(default = "default_probing_enabled")]
    pub probing_enabled: bool,

    /// Probe interval for bandwidth measurement.
    #[serde(default = "default_probe_interval", with = "humantime_serde")]
    pub probe_interval: Duration,

    /// Probe packet size for bandwidth estimation.
    #[serde(default = "default_probe_size")]
    pub probe_size: usize,

    /// Number of probe packets per burst.
    #[serde(default = "default_probe_burst")]
    pub probe_burst_count: usize,

    /// Enable path MTU discovery.
    #[serde(default = "default_pmtud_enabled")]
    pub pmtud_enabled: bool,

    /// Enable send-side pacing.
    #[serde(default = "default_pacing_enabled")]
    pub pacing_enabled: bool,

    /// Target buffer occupancy as fraction of BDP (0.0-1.0).
    #[serde(default = "default_buffer_target")]
    pub buffer_target_fraction: f64,

    /// Maximum latency threshold for uplink selection (ms).
    /// Uplinks with latency above this are penalized heavily.
    #[serde(default = "default_max_latency_ms")]
    pub max_acceptable_latency_ms: u32,

    /// Latency weight in effective throughput calculation.
    #[serde(default = "default_latency_weight")]
    pub latency_weight: f64,

    /// Enable BBR-style bandwidth estimation.
    #[serde(default = "default_bbr_enabled")]
    pub bbr_estimation: bool,

    /// Minimum RTT filter window size.
    #[serde(default = "default_min_rtt_window")]
    pub min_rtt_window_samples: usize,

    /// Bandwidth probe gain (multiplier during probing phase).
    #[serde(default = "default_probe_gain")]
    pub probe_gain: f64,

    /// Drain gain (multiplier during drain phase).
    #[serde(default = "default_drain_gain")]
    pub drain_gain: f64,

    /// Enable frame batching (Nagle-like optimization).
    #[serde(default = "default_batching_enabled")]
    pub frame_batching: bool,

    /// Maximum batch delay before forcing send.
    #[serde(default = "default_batch_delay", with = "humantime_serde")]
    pub max_batch_delay: Duration,

    /// Maximum batch size in bytes.
    #[serde(default = "default_batch_size")]
    pub max_batch_size: usize,
}

fn default_probing_enabled() -> bool {
    true
}
fn default_probe_interval() -> Duration {
    Duration::from_secs(1)
}
fn default_probe_size() -> usize {
    1400
}
fn default_probe_burst() -> usize {
    10
}
fn default_pmtud_enabled() -> bool {
    true
}
fn default_pacing_enabled() -> bool {
    true
}
fn default_buffer_target() -> f64 {
    0.5
}
fn default_max_latency_ms() -> u32 {
    500
}
fn default_latency_weight() -> f64 {
    0.4
}
fn default_bbr_enabled() -> bool {
    true
}
fn default_min_rtt_window() -> usize {
    10
}
fn default_probe_gain() -> f64 {
    1.25
}
fn default_drain_gain() -> f64 {
    0.75
}
fn default_batching_enabled() -> bool {
    false
}
fn default_batch_delay() -> Duration {
    Duration::from_millis(1)
}
fn default_batch_size() -> usize {
    16384
}

impl Default for ThroughputConfig {
    fn default() -> Self {
        Self {
            probing_enabled: default_probing_enabled(),
            probe_interval: default_probe_interval(),
            probe_size: default_probe_size(),
            probe_burst_count: default_probe_burst(),
            pmtud_enabled: default_pmtud_enabled(),
            pacing_enabled: default_pacing_enabled(),
            buffer_target_fraction: default_buffer_target(),
            max_acceptable_latency_ms: default_max_latency_ms(),
            latency_weight: default_latency_weight(),
            bbr_estimation: default_bbr_enabled(),
            min_rtt_window_samples: default_min_rtt_window(),
            probe_gain: default_probe_gain(),
            drain_gain: default_drain_gain(),
            frame_batching: default_batching_enabled(),
            max_batch_delay: default_batch_delay(),
            max_batch_size: default_batch_size(),
        }
    }
}

/// BBR-style congestion control state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BbrState {
    /// Startup: exponential bandwidth search
    Startup,
    /// Drain: reduce queue after startup
    Drain,
    /// ProbeBW: steady-state bandwidth probing
    ProbeBandwidth,
    /// ProbeRTT: periodic RTT measurement
    ProbeRtt,
}

impl Default for BbrState {
    fn default() -> Self {
        Self::Startup
    }
}

/// Bandwidth-delay product estimator for an uplink.
#[derive(Debug)]
pub struct BdpEstimator {
    /// Maximum observed bandwidth (bottleneck bandwidth).
    max_bandwidth: RwLock<f64>,
    /// Minimum observed RTT (propagation delay).
    min_rtt: RwLock<Duration>,
    /// RTT samples for min_rtt filter.
    rtt_samples: RwLock<VecDeque<(Instant, Duration)>>,
    /// Bandwidth samples for max filter.
    bw_samples: RwLock<VecDeque<(Instant, f64)>>,
    /// Sample window duration.
    window_duration: Duration,
    /// Current BBR state.
    bbr_state: RwLock<BbrState>,
    /// Cycles in current state.
    state_cycles: AtomicU32,
    /// Last state transition time.
    last_transition: RwLock<Instant>,
    /// Pacing rate (bytes per second).
    pacing_rate: AtomicU64,
    /// Congestion window in bytes.
    cwnd_bytes: AtomicU64,
    /// In-flight bytes.
    inflight_bytes: AtomicU64,
    /// Delivery rate samples.
    delivery_rate: RwLock<f64>,
    /// Round-trip counter for ProbeRTT timing.
    round_count: AtomicU64,
    /// Whether we're in a ProbeRTT round.
    probe_rtt_round: RwLock<Option<u64>>,
}

impl BdpEstimator {
    /// Create a new BDP estimator.
    pub fn new(window_duration: Duration) -> Self {
        Self {
            max_bandwidth: RwLock::new(0.0),
            min_rtt: RwLock::new(Duration::MAX),
            rtt_samples: RwLock::new(VecDeque::with_capacity(100)),
            bw_samples: RwLock::new(VecDeque::with_capacity(100)),
            window_duration,
            bbr_state: RwLock::new(BbrState::default()),
            state_cycles: AtomicU32::new(0),
            last_transition: RwLock::new(Instant::now()),
            pacing_rate: AtomicU64::new(0),
            cwnd_bytes: AtomicU64::new(10 * 1500), // Initial cwnd: 10 packets
            inflight_bytes: AtomicU64::new(0),
            delivery_rate: RwLock::new(0.0),
            round_count: AtomicU64::new(0),
            probe_rtt_round: RwLock::new(None),
        }
    }

    /// Record an RTT sample.
    pub fn record_rtt(&self, rtt: Duration) {
        let now = Instant::now();

        // Update min RTT if this is lower
        {
            let mut min = self.min_rtt.write();
            if rtt < *min {
                *min = rtt;
            }
        }

        // Add to sample window
        {
            let mut samples = self.rtt_samples.write();
            samples.push_back((now, rtt));

            // Remove old samples
            while let Some((ts, _)) = samples.front() {
                if now.duration_since(*ts) > self.window_duration {
                    samples.pop_front();
                } else {
                    break;
                }
            }
        }

        // Check for BBR state transitions
        self.maybe_transition_state();
    }

    /// Record a bandwidth sample (bytes delivered / time).
    pub fn record_bandwidth(&self, bytes_per_sec: f64) {
        let now = Instant::now();

        // Update max bandwidth if this is higher
        {
            let mut max = self.max_bandwidth.write();
            if bytes_per_sec > *max {
                *max = bytes_per_sec;
            }
        }

        // Update delivery rate
        *self.delivery_rate.write() = bytes_per_sec;

        // Add to sample window
        {
            let mut samples = self.bw_samples.write();
            samples.push_back((now, bytes_per_sec));

            // Remove old samples
            while let Some((ts, _)) = samples.front() {
                if now.duration_since(*ts) > self.window_duration {
                    samples.pop_front();
                } else {
                    break;
                }
            }
        }

        // Update pacing rate based on BBR state
        self.update_pacing_rate();
    }

    /// Record bytes being sent (for inflight tracking).
    pub fn record_send(&self, bytes: u64) {
        self.inflight_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record bytes acknowledged.
    pub fn record_ack(&self, bytes: u64) {
        let prev = self.inflight_bytes.fetch_sub(bytes, Ordering::Relaxed);
        // Prevent underflow
        if bytes > prev {
            self.inflight_bytes.store(0, Ordering::Relaxed);
        }
        self.round_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the current bandwidth-delay product in bytes.
    pub fn bdp(&self) -> u64 {
        let bandwidth = *self.max_bandwidth.read();
        let rtt = *self.min_rtt.read();

        if rtt == Duration::MAX || bandwidth == 0.0 {
            return 10 * 1500; // Default: 10 packets
        }

        (bandwidth * rtt.as_secs_f64()) as u64
    }

    /// Get the optimal congestion window in bytes.
    pub fn optimal_cwnd(&self, target_fraction: f64) -> u64 {
        let bdp = self.bdp();
        ((bdp as f64) * target_fraction).max(2.0 * 1500.0) as u64
    }

    /// Get the current pacing rate in bytes per second.
    pub fn pacing_rate(&self) -> u64 {
        self.pacing_rate.load(Ordering::Relaxed)
    }

    /// Get the current BBR state.
    pub fn state(&self) -> BbrState {
        *self.bbr_state.read()
    }

    /// Get minimum RTT.
    pub fn min_rtt(&self) -> Duration {
        let rtt = *self.min_rtt.read();
        if rtt == Duration::MAX {
            Duration::ZERO
        } else {
            rtt
        }
    }

    /// Get maximum observed bandwidth.
    pub fn max_bandwidth(&self) -> Bandwidth {
        Bandwidth::from_bps(*self.max_bandwidth.read())
    }

    /// Get current inflight bytes.
    pub fn inflight(&self) -> u64 {
        self.inflight_bytes.load(Ordering::Relaxed)
    }

    /// Check if we can send more data (inflight < cwnd).
    pub fn can_send(&self) -> bool {
        self.inflight_bytes.load(Ordering::Relaxed) < self.cwnd_bytes.load(Ordering::Relaxed)
    }

    /// Get the pacing interval for the next packet of given size.
    pub fn pacing_interval(&self, packet_size: usize) -> Duration {
        let rate = self.pacing_rate.load(Ordering::Relaxed);
        if rate == 0 {
            return Duration::ZERO;
        }

        let seconds = packet_size as f64 / rate as f64;
        Duration::from_secs_f64(seconds)
    }

    /// Update pacing rate based on current state.
    fn update_pacing_rate(&self) {
        let bw = *self.max_bandwidth.read();
        let state = *self.bbr_state.read();

        let gain = match state {
            BbrState::Startup => 2.0,        // Aggressive probing
            BbrState::Drain => 0.5,          // Drain the queue
            BbrState::ProbeBandwidth => 1.0, // Steady state
            BbrState::ProbeRtt => 1.0,       // Normal rate during RTT probe
        };

        let rate = (bw * gain) as u64;
        self.pacing_rate.store(rate, Ordering::Relaxed);

        // Update cwnd based on BDP
        let bdp = self.bdp();
        let cwnd = match state {
            BbrState::Startup => bdp * 2,    // Allow queue buildup
            BbrState::Drain => bdp,          // Target BDP
            BbrState::ProbeBandwidth => bdp, // Target BDP
            BbrState::ProbeRtt => 4 * 1500,  // Minimal cwnd for RTT measurement
        };
        self.cwnd_bytes.store(cwnd.max(4 * 1500), Ordering::Relaxed);
    }

    /// Maybe transition BBR state.
    fn maybe_transition_state(&self) {
        let mut state = self.bbr_state.write();
        let cycles = self.state_cycles.load(Ordering::Relaxed);
        let transition_elapsed = self.last_transition.read().elapsed();

        let new_state = match *state {
            BbrState::Startup => {
                // Exit startup when bandwidth growth slows
                let bw_samples = self.bw_samples.read();
                if bw_samples.len() >= 3 {
                    let recent: Vec<f64> =
                        bw_samples.iter().rev().take(3).map(|(_, b)| *b).collect();
                    let growth = if recent.len() >= 2 && recent[1] > 0.0 {
                        recent[0] / recent[1]
                    } else {
                        2.0
                    };
                    // Exit startup if growth < 25%
                    if growth < 1.25 {
                        Some(BbrState::Drain)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            BbrState::Drain => {
                // Exit drain when inflight <= BDP
                let inflight = self.inflight_bytes.load(Ordering::Relaxed);
                let bdp = self.bdp();
                if inflight <= bdp {
                    Some(BbrState::ProbeBandwidth)
                } else {
                    None
                }
            }
            BbrState::ProbeBandwidth => {
                // Enter ProbeRTT periodically (every 10 seconds)
                if transition_elapsed > Duration::from_secs(10) {
                    Some(BbrState::ProbeRtt)
                } else {
                    None
                }
            }
            BbrState::ProbeRtt => {
                // Exit ProbeRTT after one RTT with minimal inflight
                let inflight = self.inflight_bytes.load(Ordering::Relaxed);
                let probe_round = self.probe_rtt_round.read();
                let round = self.round_count.load(Ordering::Relaxed);

                if inflight <= 4 * 1500 {
                    if let Some(start_round) = *probe_round {
                        if round > start_round {
                            drop(probe_round);
                            *self.probe_rtt_round.write() = None;
                            Some(BbrState::ProbeBandwidth)
                        } else {
                            None
                        }
                    } else {
                        drop(probe_round);
                        *self.probe_rtt_round.write() = Some(round);
                        None
                    }
                } else {
                    None
                }
            }
        };

        if let Some(new) = new_state {
            *state = new;
            self.state_cycles.store(0, Ordering::Relaxed);
            *self.last_transition.write() = Instant::now();
        } else {
            self.state_cycles.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Reset the estimator (e.g., after path change).
    pub fn reset(&self) {
        *self.max_bandwidth.write() = 0.0;
        *self.min_rtt.write() = Duration::MAX;
        self.rtt_samples.write().clear();
        self.bw_samples.write().clear();
        *self.bbr_state.write() = BbrState::Startup;
        self.state_cycles.store(0, Ordering::Relaxed);
        *self.last_transition.write() = Instant::now();
        self.pacing_rate.store(0, Ordering::Relaxed);
        self.cwnd_bytes.store(10 * 1500, Ordering::Relaxed);
        self.inflight_bytes.store(0, Ordering::Relaxed);
    }
}

/// Path MTU discovery state.
#[derive(Debug)]
pub struct PmtudState {
    /// Current MTU estimate.
    current_mtu: AtomicU32,
    /// Last successful MTU.
    last_good_mtu: AtomicU32,
    /// MTU being probed.
    probe_mtu: AtomicU32,
    /// Probe state.
    state: RwLock<PmtudPhase>,
    /// Last probe time.
    last_probe: RwLock<Instant>,
    /// Consecutive failures at current probe MTU.
    failures: AtomicU32,
    /// Probe history: (mtu, success).
    history: RwLock<VecDeque<(u32, bool)>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PmtudPhase {
    /// Searching for maximum MTU.
    Search,
    /// Stable MTU found.
    Stable,
    /// Verifying current MTU still works.
    Verify,
}

impl PmtudState {
    /// Create new PMTUD state.
    pub fn new(initial_mtu: u32) -> Self {
        Self {
            current_mtu: AtomicU32::new(initial_mtu),
            last_good_mtu: AtomicU32::new(MIN_MTU),
            probe_mtu: AtomicU32::new(initial_mtu),
            state: RwLock::new(PmtudPhase::Search),
            last_probe: RwLock::new(Instant::now()),
            failures: AtomicU32::new(0),
            history: RwLock::new(VecDeque::with_capacity(20)),
        }
    }

    /// Get current MTU estimate.
    pub fn mtu(&self) -> u32 {
        self.current_mtu.load(Ordering::Relaxed)
    }

    /// Get optimal payload size (MTU minus headers).
    pub fn optimal_payload_size(&self, header_overhead: u32) -> u32 {
        self.current_mtu
            .load(Ordering::Relaxed)
            .saturating_sub(header_overhead)
    }

    /// Record a successful transmission at given size.
    pub fn record_success(&self, size: u32) {
        let current = self.current_mtu.load(Ordering::Relaxed);

        if size >= current {
            // Update last good MTU
            self.last_good_mtu.store(size, Ordering::Relaxed);
            self.failures.store(0, Ordering::Relaxed);

            // Record in history
            let mut history = self.history.write();
            history.push_back((size, true));
            if history.len() > 20 {
                history.pop_front();
            }
        }

        // If we're in search mode and succeeded, try higher
        let mut state = self.state.write();
        if *state == PmtudPhase::Search {
            let probe = self.probe_mtu.load(Ordering::Relaxed);
            if size >= probe && probe < MAX_MTU {
                // Try even larger
                let new_probe = ((probe + MAX_MTU) / 2).min(MAX_MTU);
                self.probe_mtu.store(new_probe, Ordering::Relaxed);
            } else if size >= probe {
                // Found maximum
                self.current_mtu.store(probe, Ordering::Relaxed);
                *state = PmtudPhase::Stable;
            }
        }
    }

    /// Record a transmission failure (likely MTU too large).
    pub fn record_failure(&self, size: u32) {
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;

        // Record in history
        {
            let mut history = self.history.write();
            history.push_back((size, false));
            if history.len() > 20 {
                history.pop_front();
            }
        }

        let mut state = self.state.write();

        match *state {
            PmtudPhase::Search => {
                // Binary search down
                let probe = self.probe_mtu.load(Ordering::Relaxed);
                let good = self.last_good_mtu.load(Ordering::Relaxed);

                if probe - good < 32 {
                    // Close enough, use last good
                    self.current_mtu.store(good, Ordering::Relaxed);
                    self.probe_mtu.store(good, Ordering::Relaxed);
                    *state = PmtudPhase::Stable;
                } else {
                    // Try smaller
                    let new_probe = (probe + good) / 2;
                    self.probe_mtu.store(new_probe, Ordering::Relaxed);
                }
            }
            PmtudPhase::Stable => {
                // Path MTU may have decreased
                if failures >= 3 {
                    // Restart search from lower value
                    let current = self.current_mtu.load(Ordering::Relaxed);
                    let reduced = (current * 3 / 4).max(MIN_MTU);
                    self.current_mtu.store(reduced, Ordering::Relaxed);
                    self.probe_mtu.store(reduced, Ordering::Relaxed);
                    self.failures.store(0, Ordering::Relaxed);
                    *state = PmtudPhase::Search;
                }
            }
            PmtudPhase::Verify => {
                // Verification failed, reduce MTU
                let current = self.current_mtu.load(Ordering::Relaxed);
                let reduced = (current * 3 / 4).max(MIN_MTU);
                self.current_mtu.store(reduced, Ordering::Relaxed);
                self.probe_mtu.store(reduced, Ordering::Relaxed);
                self.failures.store(0, Ordering::Relaxed);
                *state = PmtudPhase::Search;
            }
        }
    }

    /// Get the next MTU to probe.
    pub fn next_probe_mtu(&self) -> u32 {
        self.probe_mtu.load(Ordering::Relaxed)
    }

    /// Check if we should send a probe.
    pub fn should_probe(&self) -> bool {
        let state = *self.state.read();
        match state {
            PmtudPhase::Search => true,
            PmtudPhase::Stable => {
                // Periodically verify (every 60 seconds)
                self.last_probe.read().elapsed() > Duration::from_secs(60)
            }
            PmtudPhase::Verify => true,
        }
    }

    /// Record that a probe was sent.
    pub fn probe_sent(&self) {
        *self.last_probe.write() = Instant::now();
    }
}

impl Default for PmtudState {
    fn default() -> Self {
        Self::new(DEFAULT_MTU)
    }
}

/// Effective throughput calculator that combines bandwidth and latency.
#[derive(Debug, Clone, Copy)]
pub struct EffectiveThroughput {
    /// Raw bandwidth in bytes per second.
    pub bandwidth_bps: f64,
    /// Round-trip time.
    pub rtt: Duration,
    /// Packet loss ratio (0.0-1.0).
    pub loss_ratio: f64,
    /// Jitter (RTT variance).
    pub jitter: Duration,
    /// Calculated effective throughput score.
    pub score: f64,
    /// Estimated time to transfer 1MB.
    pub time_for_1mb: Duration,
    /// Bandwidth-delay product.
    pub bdp: u64,
}

impl EffectiveThroughput {
    /// Calculate effective throughput from connection parameters.
    ///
    /// This considers:
    /// - Raw bandwidth
    /// - Latency impact (high latency reduces effective throughput)
    /// - Packet loss (retransmissions reduce throughput)
    /// - Jitter (unpredictable timing affects throughput)
    pub fn calculate(
        bandwidth_bps: f64,
        rtt: Duration,
        loss_ratio: f64,
        jitter: Duration,
        config: &ThroughputConfig,
    ) -> Self {
        // Mathis equation for TCP throughput:
        // throughput = (MSS / RTT) * (C / sqrt(loss))
        // We use a modified version that accounts for UDP/custom protocols

        let rtt_secs = rtt.as_secs_f64().max(0.001); // Minimum 1ms RTT
        let loss_adjusted = (1.0 - loss_ratio).max(0.01);
        let jitter_penalty = 1.0 / (1.0 + jitter.as_secs_f64() * 10.0);

        // Calculate effective bandwidth considering loss
        // With loss, we effectively lose bandwidth due to retransmissions
        let loss_throughput_factor = if loss_ratio > 0.0 {
            // Simplified Mathis model
            1.0 / (1.0 + loss_ratio * 2.0)
        } else {
            1.0
        };

        let effective_bandwidth = bandwidth_bps * loss_throughput_factor;

        // Latency penalty: higher latency means slower feedback loop
        // This affects how quickly we can fill the pipe
        let max_latency_secs = config.max_acceptable_latency_ms as f64 / 1000.0;
        let latency_penalty = if rtt_secs > max_latency_secs {
            // Severe penalty for exceeding threshold
            (max_latency_secs / rtt_secs).powi(2)
        } else {
            // Mild penalty proportional to latency
            1.0 - (rtt_secs / max_latency_secs) * config.latency_weight
        };

        // BDP (bandwidth-delay product)
        let bdp = (bandwidth_bps * rtt_secs) as u64;

        // Time to transfer 1MB considering all factors
        let bytes_1mb = 1024.0 * 1024.0;
        let transfer_time = if effective_bandwidth > 0.0 {
            bytes_1mb / effective_bandwidth
        } else {
            // Use a large but valid duration (1 hour in seconds)
            3600.0
        };
        // Add RTT for connection overhead, cap at reasonable max (1 hour)
        let total_transfer_time = (transfer_time + rtt_secs).min(3600.0);

        // Combined score (higher is better)
        // Normalizes bandwidth to 0-1 range (assuming max 1Gbps)
        let normalized_bw = (effective_bandwidth / 125_000_000.0).min(1.0);
        let score = normalized_bw * latency_penalty * loss_adjusted * jitter_penalty;

        Self {
            bandwidth_bps,
            rtt,
            loss_ratio,
            jitter,
            score,
            time_for_1mb: Duration::from_secs_f64(total_transfer_time),
            bdp,
        }
    }

    /// Compare two throughput measurements and determine which is better.
    /// Returns true if self is better than other.
    pub fn is_better_than(&self, other: &Self) -> bool {
        self.score > other.score
    }

    /// Check if this uplink can complete a transfer faster than another,
    /// considering both bandwidth and latency.
    pub fn faster_for_size(&self, other: &Self, size_bytes: u64) -> bool {
        let self_time = self.transfer_time(size_bytes);
        let other_time = other.transfer_time(size_bytes);
        self_time < other_time
    }

    /// Estimate time to transfer given bytes.
    pub fn transfer_time(&self, bytes: u64) -> Duration {
        let loss_factor = 1.0 / (1.0 - self.loss_ratio).max(0.01);
        let bandwidth = self.bandwidth_bps.max(1.0);
        let transfer_secs = (bytes as f64 * loss_factor) / bandwidth;
        // Cap at 1 hour to avoid Duration overflow
        let total_secs = (transfer_secs + self.rtt.as_secs_f64()).min(3600.0);
        Duration::from_secs_f64(total_secs)
    }
}

/// Frame batcher for optimizing small packet sends.
#[derive(Debug)]
pub struct FrameBatcher {
    /// Pending frames.
    pending: RwLock<Vec<Vec<u8>>>,
    /// Total pending size.
    pending_size: AtomicU64,
    /// Time of first pending frame.
    first_pending: RwLock<Option<Instant>>,
    /// Configuration.
    max_delay: Duration,
    max_size: usize,
}

impl FrameBatcher {
    /// Create a new frame batcher.
    pub fn new(max_delay: Duration, max_size: usize) -> Self {
        Self {
            pending: RwLock::new(Vec::new()),
            pending_size: AtomicU64::new(0),
            first_pending: RwLock::new(None),
            max_delay,
            max_size,
        }
    }

    /// Add a frame to the batch.
    /// Returns Some(batch) if batch should be sent immediately.
    pub fn add(&self, frame: Vec<u8>) -> Option<Vec<Vec<u8>>> {
        let frame_size = frame.len();

        let mut pending = self.pending.write();
        let prev_size = self
            .pending_size
            .fetch_add(frame_size as u64, Ordering::Relaxed);

        // Set first_pending time if this is the first frame
        if pending.is_empty() {
            *self.first_pending.write() = Some(Instant::now());
        }

        pending.push(frame);

        // Check if we should flush
        let total_size = prev_size as usize + frame_size;
        let first_time = *self.first_pending.read();
        let should_flush = total_size >= self.max_size
            || first_time.is_some_and(|t| t.elapsed() >= self.max_delay);

        if should_flush {
            self.pending_size.store(0, Ordering::Relaxed);
            *self.first_pending.write() = None;
            Some(std::mem::take(&mut *pending))
        } else {
            None
        }
    }

    /// Force flush all pending frames.
    pub fn flush(&self) -> Vec<Vec<u8>> {
        self.pending_size.store(0, Ordering::Relaxed);
        *self.first_pending.write() = None;
        std::mem::take(&mut *self.pending.write())
    }

    /// Check if batch is ready to send (due to timeout).
    pub fn is_ready(&self) -> bool {
        let first = *self.first_pending.read();
        first.is_some_and(|t| t.elapsed() >= self.max_delay)
    }

    /// Get pending size.
    pub fn pending_size(&self) -> usize {
        self.pending_size.load(Ordering::Relaxed) as usize
    }
}

/// Per-uplink throughput state.
#[derive(Debug)]
pub struct UplinkThroughputState {
    /// BDP estimator.
    pub bdp: BdpEstimator,
    /// Path MTU discovery state.
    pub pmtud: PmtudState,
    /// Frame batcher (if enabled).
    pub batcher: Option<FrameBatcher>,
    /// Last bandwidth probe time.
    pub last_probe: RwLock<Instant>,
    /// Probe in flight.
    pub probe_inflight: AtomicU32,
    /// Probe sent timestamps for RTT calculation.
    pub probe_timestamps: RwLock<HashMap<u64, Instant>>,
    /// Active bandwidth probing enabled.
    pub probing_active: bool,
}

impl UplinkThroughputState {
    /// Create new uplink throughput state.
    pub fn new(config: &ThroughputConfig) -> Self {
        Self {
            bdp: BdpEstimator::new(Duration::from_secs(10)),
            pmtud: PmtudState::new(DEFAULT_MTU),
            batcher: if config.frame_batching {
                Some(FrameBatcher::new(
                    config.max_batch_delay,
                    config.max_batch_size,
                ))
            } else {
                None
            },
            last_probe: RwLock::new(Instant::now()),
            probe_inflight: AtomicU32::new(0),
            probe_timestamps: RwLock::new(HashMap::new()),
            probing_active: config.probing_enabled,
        }
    }

    /// Check if this uplink needs bandwidth probing.
    pub fn needs_probe(&self, interval: Duration) -> bool {
        self.probing_active
            && self.last_probe.read().elapsed() >= interval
            && self.probe_inflight.load(Ordering::Relaxed) == 0
    }

    /// Record probe sent.
    pub fn record_probe_sent(&self, probe_id: u64) {
        self.probe_timestamps
            .write()
            .insert(probe_id, Instant::now());
        self.probe_inflight.fetch_add(1, Ordering::Relaxed);
        *self.last_probe.write() = Instant::now();
    }

    /// Record probe response.
    pub fn record_probe_response(&self, probe_id: u64, bytes: u64) {
        self.probe_inflight.fetch_sub(1, Ordering::Relaxed);

        if let Some(sent_time) = self.probe_timestamps.write().remove(&probe_id) {
            let rtt = sent_time.elapsed();
            self.bdp.record_rtt(rtt);

            // Calculate bandwidth from probe
            if rtt.as_secs_f64() > 0.0 {
                let bw = bytes as f64 / rtt.as_secs_f64();
                self.bdp.record_bandwidth(bw);
            }
        }
    }
}

/// Throughput optimizer for the multipath manager.
pub struct ThroughputOptimizer {
    /// Configuration.
    config: ThroughputConfig,
    /// Per-uplink state.
    uplink_state: RwLock<HashMap<u16, UplinkThroughputState>>,
    /// Last optimization run.
    last_optimization: RwLock<Instant>,
}

impl ThroughputOptimizer {
    /// Create a new throughput optimizer.
    pub fn new(config: ThroughputConfig) -> Self {
        Self {
            config,
            uplink_state: RwLock::new(HashMap::new()),
            last_optimization: RwLock::new(Instant::now()),
        }
    }

    /// Register an uplink with the optimizer.
    pub fn register_uplink(&self, uplink_id: u16) {
        self.uplink_state
            .write()
            .insert(uplink_id, UplinkThroughputState::new(&self.config));
    }

    /// Unregister an uplink.
    pub fn unregister_uplink(&self, uplink_id: u16) {
        self.uplink_state.write().remove(&uplink_id);
    }

    /// Get configuration.
    pub fn config(&self) -> &ThroughputConfig {
        &self.config
    }

    /// Calculate effective throughput for an uplink.
    pub fn effective_throughput(&self, uplink: &Uplink) -> EffectiveThroughput {
        let bandwidth = uplink.bandwidth();
        let rtt = uplink.rtt();
        let loss = uplink.loss_ratio();

        // Get jitter from RTT variance
        let metrics = uplink.quality_metrics();
        let jitter = metrics.jitter;

        EffectiveThroughput::calculate(bandwidth.bytes_per_sec, rtt, loss, jitter, &self.config)
    }

    /// Select the best uplink based on effective throughput.
    /// Returns uplinks sorted by effective throughput (best first).
    pub fn rank_uplinks(&self, uplinks: &[Arc<Uplink>]) -> Vec<(u16, EffectiveThroughput)> {
        let mut ranked: Vec<_> = uplinks
            .iter()
            .filter(|u| u.is_usable())
            .map(|u| {
                let throughput = self.effective_throughput(u);
                (u.numeric_id(), throughput)
            })
            .collect();

        // Sort by score (descending)
        ranked.sort_by(|a, b| {
            b.1.score
                .partial_cmp(&a.1.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ranked
    }

    /// Select the best uplink for a transfer of given size.
    /// Considers both bandwidth and latency to find the fastest path.
    pub fn best_for_size(&self, uplinks: &[Arc<Uplink>], size_bytes: u64) -> Option<u16> {
        uplinks
            .iter()
            .filter(|u| u.is_usable())
            .map(|u| {
                let throughput = self.effective_throughput(u);
                let time = throughput.transfer_time(size_bytes);
                (u.numeric_id(), time)
            })
            .min_by(|a, b| a.1.cmp(&b.1))
            .map(|(id, _)| id)
    }

    /// Get optimal packet size for an uplink.
    pub fn optimal_packet_size(&self, uplink_id: u16) -> u32 {
        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            // Subtract protocol overhead (IP + UDP + Triglav headers ~ 60 bytes)
            uplink_state.pmtud.optimal_payload_size(60)
        } else {
            DEFAULT_MTU - 60
        }
    }

    /// Get pacing interval for sending on an uplink.
    pub fn pacing_interval(&self, uplink_id: u16, packet_size: usize) -> Duration {
        if !self.config.pacing_enabled {
            return Duration::ZERO;
        }

        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            uplink_state.bdp.pacing_interval(packet_size)
        } else {
            Duration::ZERO
        }
    }

    /// Check if we can send on an uplink (respects congestion control).
    pub fn can_send(&self, uplink_id: u16) -> bool {
        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            uplink_state.bdp.can_send()
        } else {
            true // Default to allowing sends
        }
    }

    /// Record a successful send.
    pub fn record_send(&self, uplink_id: u16, bytes: u64) {
        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            uplink_state.bdp.record_send(bytes);
        }
    }

    /// Record an acknowledgment.
    pub fn record_ack(&self, uplink_id: u16, bytes: u64, rtt: Duration) {
        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            uplink_state.bdp.record_ack(bytes);
            uplink_state.bdp.record_rtt(rtt);
        }
    }

    /// Record a bandwidth measurement.
    pub fn record_bandwidth(&self, uplink_id: u16, bytes_per_sec: f64) {
        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            uplink_state.bdp.record_bandwidth(bytes_per_sec);
        }
    }

    /// Record MTU probe result.
    pub fn record_mtu_result(&self, uplink_id: u16, size: u32, success: bool) {
        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            if success {
                uplink_state.pmtud.record_success(size);
            } else {
                uplink_state.pmtud.record_failure(size);
            }
        }
    }

    /// Get uplinks that need bandwidth probing.
    pub fn uplinks_needing_probe(&self) -> Vec<u16> {
        self.uplink_state
            .read()
            .iter()
            .filter(|(_, state)| state.needs_probe(self.config.probe_interval))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Record that a probe was sent.
    pub fn record_probe_sent(&self, uplink_id: u16, probe_id: u64) {
        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            uplink_state.record_probe_sent(probe_id);
        }
    }

    /// Record a probe response.
    pub fn record_probe_response(&self, uplink_id: u16, probe_id: u64, bytes: u64) {
        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            uplink_state.record_probe_response(probe_id, bytes);
        }
    }

    /// Get BBR state for an uplink.
    pub fn bbr_state(&self, uplink_id: u16) -> Option<BbrState> {
        self.uplink_state
            .read()
            .get(&uplink_id)
            .map(|s| s.bdp.state())
    }

    /// Get BDP for an uplink.
    pub fn bdp(&self, uplink_id: u16) -> Option<u64> {
        self.uplink_state
            .read()
            .get(&uplink_id)
            .map(|s| s.bdp.bdp())
    }

    /// Add a frame to the batcher for an uplink.
    /// Returns Some(frames) if batch should be sent.
    pub fn batch_frame(&self, uplink_id: u16, frame: Vec<u8>) -> Option<Vec<Vec<u8>>> {
        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            if let Some(ref batcher) = uplink_state.batcher {
                return batcher.add(frame);
            }
        }
        // No batching, return single frame immediately
        Some(vec![frame])
    }

    /// Flush pending batched frames for an uplink.
    pub fn flush_batch(&self, uplink_id: u16) -> Vec<Vec<u8>> {
        let state = self.uplink_state.read();
        if let Some(uplink_state) = state.get(&uplink_id) {
            if let Some(ref batcher) = uplink_state.batcher {
                return batcher.flush();
            }
        }
        vec![]
    }

    /// Check all uplinks for batches ready to send.
    pub fn ready_batches(&self) -> Vec<(u16, Vec<Vec<u8>>)> {
        let state = self.uplink_state.read();
        let mut result = Vec::new();

        for (&uplink_id, uplink_state) in state.iter() {
            if let Some(ref batcher) = uplink_state.batcher {
                if batcher.is_ready() {
                    result.push((uplink_id, batcher.flush()));
                }
            }
        }

        result
    }

    /// Get throughput summary for all uplinks.
    pub fn summary(&self, uplinks: &[Arc<Uplink>]) -> ThroughputSummary {
        let ranked = self.rank_uplinks(uplinks);

        let total_bandwidth: f64 = ranked.iter().map(|(_, t)| t.bandwidth_bps).sum();

        let best_score = ranked.first().map(|(_, t)| t.score).unwrap_or(0.0);
        let worst_score = ranked.last().map(|(_, t)| t.score).unwrap_or(0.0);

        let avg_rtt = if ranked.is_empty() {
            Duration::ZERO
        } else {
            let total: Duration = ranked.iter().map(|(_, t)| t.rtt).sum();
            total / ranked.len() as u32
        };

        ThroughputSummary {
            uplink_count: ranked.len(),
            total_bandwidth: Bandwidth::from_bps(total_bandwidth),
            best_score,
            worst_score,
            avg_rtt,
            ranked_uplinks: ranked,
        }
    }
}

impl std::fmt::Debug for ThroughputOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThroughputOptimizer")
            .field("config", &self.config)
            .field("uplink_count", &self.uplink_state.read().len())
            .finish()
    }
}

/// Summary of throughput across all uplinks.
#[derive(Debug, Clone)]
pub struct ThroughputSummary {
    /// Number of usable uplinks.
    pub uplink_count: usize,
    /// Total aggregate bandwidth.
    pub total_bandwidth: Bandwidth,
    /// Best effective throughput score.
    pub best_score: f64,
    /// Worst effective throughput score.
    pub worst_score: f64,
    /// Average RTT across uplinks.
    pub avg_rtt: Duration,
    /// Uplinks ranked by effective throughput.
    pub ranked_uplinks: Vec<(u16, EffectiveThroughput)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_throughput_calculation() {
        let config = ThroughputConfig::default();

        // High bandwidth, low latency
        let good = EffectiveThroughput::calculate(
            125_000_000.0, // 1 Gbps
            Duration::from_millis(10),
            0.0,
            Duration::from_millis(1),
            &config,
        );

        // Low bandwidth, high latency
        let bad = EffectiveThroughput::calculate(
            12_500_000.0, // 100 Mbps
            Duration::from_millis(500),
            0.05,
            Duration::from_millis(50),
            &config,
        );

        assert!(good.score > bad.score);
        assert!(good.is_better_than(&bad));
    }

    #[test]
    fn test_transfer_time_comparison() {
        let config = ThroughputConfig::default();

        // High bandwidth but high latency
        let high_bw_high_lat = EffectiveThroughput::calculate(
            125_000_000.0,          // 1 Gbps
            Duration::from_secs(3), // 3 second RTT!
            0.0,
            Duration::ZERO,
            &config,
        );

        // Lower bandwidth but low latency
        let low_bw_low_lat = EffectiveThroughput::calculate(
            12_500_000.0, // 100 Mbps
            Duration::from_millis(10),
            0.0,
            Duration::ZERO,
            &config,
        );

        // For small transfers, low latency should win
        let small_size = 10 * 1024; // 10 KB
        assert!(low_bw_low_lat.faster_for_size(&high_bw_high_lat, small_size));

        // For very large transfers, high bandwidth should win despite latency
        let huge_size = 1024 * 1024 * 1024; // 1 GB
        assert!(high_bw_high_lat.faster_for_size(&low_bw_low_lat, huge_size));
    }

    #[test]
    fn test_bdp_calculation() {
        let estimator = BdpEstimator::new(Duration::from_secs(10));

        // Record some measurements
        estimator.record_rtt(Duration::from_millis(100));
        estimator.record_bandwidth(125_000_000.0); // 1 Gbps

        let bdp = estimator.bdp();
        // BDP = 1 Gbps * 100ms = 12.5 MB
        assert!(bdp > 10_000_000);
        assert!(bdp < 20_000_000);
    }

    #[test]
    fn test_pmtud_binary_search() {
        let pmtud = PmtudState::new(DEFAULT_MTU);

        // Simulate MTU discovery
        pmtud.record_success(1400);
        assert!(pmtud.mtu() <= DEFAULT_MTU);

        // Simulate failure at high MTU
        pmtud.record_failure(1500);

        // Should have reduced probe target
        assert!(pmtud.next_probe_mtu() < 1500);
    }

    #[test]
    fn test_frame_batcher() {
        let batcher = FrameBatcher::new(Duration::from_millis(10), 1000);

        // Add small frames
        assert!(batcher.add(vec![0; 100]).is_none());
        assert!(batcher.add(vec![0; 100]).is_none());

        // Force flush
        let frames = batcher.flush();
        assert_eq!(frames.len(), 2);

        // Add frames until size limit
        for _ in 0..10 {
            let result = batcher.add(vec![0; 200]);
            if result.is_some() {
                // Should trigger around 5 frames (1000 bytes)
                break;
            }
        }
    }
}
