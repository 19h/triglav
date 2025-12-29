//! Intelligent packet scheduler for multi-path distribution.
//!
//! Implements multiple scheduling strategies:
//! - Weighted round-robin
//! - Lowest latency first
//! - Lowest loss first
//! - Adaptive (combines multiple signals)
//! - Redundant (send on multiple paths)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::Uplink;

/// Scheduling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingStrategy {
    /// Weighted round-robin based on configured weights.
    WeightedRoundRobin,
    /// Always choose lowest latency uplink.
    LowestLatency,
    /// Always choose lowest loss uplink.
    LowestLoss,
    /// Adaptive selection based on multiple factors.
    #[default]
    Adaptive,
    /// Send on all uplinks for redundancy.
    Redundant,
    /// Send on primary, failover to secondary.
    PrimaryBackup,
    /// Load balance based on available bandwidth.
    BandwidthProportional,
}

/// Scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduling strategy.
    #[serde(default)]
    pub strategy: SchedulingStrategy,

    /// Minimum RTT difference to prefer one uplink (ms).
    #[serde(default = "default_rtt_threshold")]
    pub rtt_threshold_ms: u32,

    /// Minimum loss difference to prefer one uplink (%).
    #[serde(default = "default_loss_threshold")]
    pub loss_threshold_percent: f32,

    /// Weight for RTT in adaptive scoring (0-1).
    #[serde(default = "default_rtt_weight")]
    pub rtt_weight: f32,

    /// Weight for loss in adaptive scoring (0-1).
    #[serde(default = "default_loss_weight")]
    pub loss_weight: f32,

    /// Weight for bandwidth in adaptive scoring (0-1).
    #[serde(default = "default_bw_weight")]
    pub bandwidth_weight: f32,

    /// Enable path stickiness (prefer same path for related packets).
    #[serde(default = "default_sticky")]
    pub sticky_paths: bool,

    /// Stickiness timeout.
    #[serde(default = "default_sticky_timeout", with = "humantime_serde")]
    pub sticky_timeout: Duration,

    /// Enable proactive probing of backup paths.
    #[serde(default = "default_probe")]
    pub probe_backup_paths: bool,

    /// Probe interval for backup paths.
    #[serde(default = "default_probe_interval", with = "humantime_serde")]
    pub probe_interval: Duration,
}

fn default_rtt_threshold() -> u32 { 10 }
fn default_loss_threshold() -> f32 { 2.0 }
fn default_rtt_weight() -> f32 { 0.4 }
fn default_loss_weight() -> f32 { 0.4 }
fn default_bw_weight() -> f32 { 0.2 }
fn default_sticky() -> bool { true }
fn default_sticky_timeout() -> Duration { Duration::from_secs(5) }
fn default_probe() -> bool { true }
fn default_probe_interval() -> Duration { Duration::from_secs(1) }

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            strategy: SchedulingStrategy::default(),
            rtt_threshold_ms: default_rtt_threshold(),
            loss_threshold_percent: default_loss_threshold(),
            rtt_weight: default_rtt_weight(),
            loss_weight: default_loss_weight(),
            bandwidth_weight: default_bw_weight(),
            sticky_paths: default_sticky(),
            sticky_timeout: default_sticky_timeout(),
            probe_backup_paths: default_probe(),
            probe_interval: default_probe_interval(),
        }
    }
}

/// State for weighted round-robin.
#[derive(Debug, Default)]
struct WrrState {
    current_index: usize,
    current_weight: u32,
}

/// Path stickiness tracking.
#[derive(Debug)]
struct PathStickiness {
    /// Flow -> (uplink_id, last_used)
    flows: HashMap<u64, (u16, Instant)>,
}

impl PathStickiness {
    fn new() -> Self {
        Self {
            flows: HashMap::new(),
        }
    }

    fn get(&self, flow_id: u64, timeout: Duration) -> Option<u16> {
        self.flows.get(&flow_id).and_then(|(uplink, last)| {
            if last.elapsed() < timeout {
                Some(*uplink)
            } else {
                None
            }
        })
    }

    fn set(&mut self, flow_id: u64, uplink_id: u16) {
        self.flows.insert(flow_id, (uplink_id, Instant::now()));
    }

    fn cleanup(&mut self, timeout: Duration) {
        self.flows.retain(|_, (_, last)| last.elapsed() < timeout);
    }
}

/// Packet scheduler.
pub struct Scheduler {
    config: SchedulerConfig,
    wrr_state: RwLock<WrrState>,
    stickiness: RwLock<PathStickiness>,
    last_probe: RwLock<HashMap<u16, Instant>>,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            wrr_state: RwLock::new(WrrState::default()),
            stickiness: RwLock::new(PathStickiness::new()),
            last_probe: RwLock::new(HashMap::new()),
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Select uplink(s) for a packet.
    ///
    /// Returns a list of uplink IDs in priority order.
    /// For most strategies, this returns a single uplink.
    /// For Redundant strategy, returns multiple uplinks.
    pub fn select(&self, uplinks: &[Arc<Uplink>], flow_id: Option<u64>) -> Vec<u16> {
        // Filter to usable uplinks
        let usable: Vec<_> = uplinks.iter()
            .filter(|u| u.is_usable())
            .collect();

        if usable.is_empty() {
            return vec![];
        }

        // Check stickiness first
        if self.config.sticky_paths {
            if let Some(flow) = flow_id {
                let sticky = self.stickiness.read().get(flow, self.config.sticky_timeout);
                if let Some(sticky_uplink) = sticky {
                    if usable.iter().any(|u| u.numeric_id() == sticky_uplink) {
                        return vec![sticky_uplink];
                    }
                }
            }
        }

        let selected = match self.config.strategy {
            SchedulingStrategy::WeightedRoundRobin => {
                self.select_wrr(&usable)
            }
            SchedulingStrategy::LowestLatency => {
                Self::select_lowest_latency(&usable)
            }
            SchedulingStrategy::LowestLoss => {
                Self::select_lowest_loss(&usable)
            }
            SchedulingStrategy::Adaptive => {
                self.select_adaptive(&usable)
            }
            SchedulingStrategy::Redundant => {
                Self::select_redundant(&usable)
            }
            SchedulingStrategy::PrimaryBackup => {
                Self::select_primary_backup(&usable)
            }
            SchedulingStrategy::BandwidthProportional => {
                self.select_bandwidth_proportional(&usable)
            }
        };

        // Update stickiness
        if self.config.sticky_paths && !selected.is_empty() {
            if let Some(flow) = flow_id {
                self.stickiness.write().set(flow, selected[0]);
            }
        }

        selected
    }

    /// Weighted round-robin selection.
    fn select_wrr(&self, uplinks: &[&Arc<Uplink>]) -> Vec<u16> {
        if uplinks.is_empty() {
            return vec![];
        }

        let mut state = self.wrr_state.write();

        // Find uplink with remaining weight
        let max_weight: u32 = uplinks.iter().map(|u| u.config().weight).max().unwrap_or(1);

        loop {
            state.current_index = (state.current_index + 1) % uplinks.len();

            if state.current_index == 0 {
                if state.current_weight == 0 {
                    state.current_weight = max_weight;
                } else {
                    state.current_weight -= 1;
                }
            }

            let uplink = &uplinks[state.current_index];
            if uplink.config().weight >= state.current_weight && uplink.can_send() {
                return vec![uplink.numeric_id()];
            }

            // Safety: prevent infinite loop
            if state.current_weight == 0 && state.current_index == 0 {
                break;
            }
        }

        // Fallback to first usable
        uplinks.first().map(|u| vec![u.numeric_id()]).unwrap_or_default()
    }

    /// Select lowest latency uplink.
    fn select_lowest_latency(uplinks: &[&Arc<Uplink>]) -> Vec<u16> {
        uplinks.iter()
            .filter(|u| u.can_send())
            .min_by_key(|u| u.rtt())
            .map(|u| vec![u.numeric_id()])
            .unwrap_or_default()
    }

    /// Select lowest loss uplink.
    fn select_lowest_loss(uplinks: &[&Arc<Uplink>]) -> Vec<u16> {
        uplinks.iter()
            .filter(|u| u.can_send())
            .min_by(|a, b| {
                a.loss_ratio().partial_cmp(&b.loss_ratio())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|u| vec![u.numeric_id()])
            .unwrap_or_default()
    }

    /// Adaptive selection based on multiple factors.
    fn select_adaptive(&self, uplinks: &[&Arc<Uplink>]) -> Vec<u16> {
        // Calculate scores for each uplink
        let mut scored: Vec<_> = uplinks.iter()
            .filter(|u| u.can_send())
            .map(|u| {
                let rtt_score = Self::rtt_score(u);
                let loss_score = Self::loss_score(u);
                let bw_score = Self::bandwidth_score(u, uplinks);

                let total_score =
                    rtt_score * self.config.rtt_weight +
                    loss_score * self.config.loss_weight +
                    bw_score * self.config.bandwidth_weight;

                (u.numeric_id(), total_score)
            })
            .collect();

        // Sort by score (highest first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().map(|(id, _)| id).take(1).collect()
    }

    /// Redundant selection (all usable uplinks).
    fn select_redundant(uplinks: &[&Arc<Uplink>]) -> Vec<u16> {
        uplinks.iter()
            .filter(|u| u.can_send())
            .map(|u| u.numeric_id())
            .collect()
    }

    /// Primary-backup selection.
    fn select_primary_backup(uplinks: &[&Arc<Uplink>]) -> Vec<u16> {
        // Sort by priority (highest first)
        let mut sorted: Vec<_> = uplinks.iter().collect();
        sorted.sort_by_key(|u| std::cmp::Reverse(u.priority_score()));

        // Return first usable, with backup
        let mut result = Vec::new();
        for uplink in sorted {
            if uplink.can_send() {
                result.push(uplink.numeric_id());
                if result.len() >= 2 {
                    break;
                }
            }
        }
        result
    }

    /// Bandwidth-proportional selection.
    fn select_bandwidth_proportional(&self, uplinks: &[&Arc<Uplink>]) -> Vec<u16> {
        // Select based on available bandwidth ratio
        let total_bw: f64 = uplinks.iter()
            .filter(|u| u.can_send())
            .map(|u| u.bandwidth().bytes_per_sec)
            .sum();

        if total_bw == 0.0 {
            return self.select_wrr(uplinks);
        }

        // Use randomized selection weighted by bandwidth
        let r: f64 = rand::random();
        let mut cumulative = 0.0;

        for uplink in uplinks.iter().filter(|u| u.can_send()) {
            cumulative += uplink.bandwidth().bytes_per_sec / total_bw;
            if r <= cumulative {
                return vec![uplink.numeric_id()];
            }
        }

        uplinks.first().map(|u| vec![u.numeric_id()]).unwrap_or_default()
    }

    /// Calculate RTT score (0-1, higher is better).
    fn rtt_score(uplink: &Uplink) -> f32 {
        let rtt = uplink.rtt().as_secs_f32() * 1000.0; // ms
        // Score decreases with RTT
        1.0 / (1.0 + rtt / 50.0)
    }

    /// Calculate loss score (0-1, higher is better).
    fn loss_score(uplink: &Uplink) -> f32 {
        let loss = uplink.loss_ratio() as f32;
        1.0 - loss.min(1.0)
    }

    /// Calculate bandwidth score (0-1, higher is better).
    fn bandwidth_score(uplink: &Uplink, all: &[&Arc<Uplink>]) -> f32 {
        let bw = uplink.bandwidth().bytes_per_sec;
        let max_bw: f64 = all.iter()
            .map(|u| u.bandwidth().bytes_per_sec)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        if max_bw == 0.0 {
            0.5
        } else {
            (bw / max_bw) as f32
        }
    }

    /// Check if an uplink needs probing.
    pub fn needs_probe(&self, uplink: &Uplink) -> bool {
        if !self.config.probe_backup_paths {
            return false;
        }

        let probes = self.last_probe.read();
        match probes.get(&uplink.numeric_id()) {
            Some(last) => last.elapsed() >= self.config.probe_interval,
            None => true,
        }
    }

    /// Record that an uplink was probed.
    pub fn record_probe(&self, uplink_id: u16) {
        self.last_probe.write().insert(uplink_id, Instant::now());
    }

    /// Cleanup stale state.
    pub fn cleanup(&self) {
        self.stickiness.write().cleanup(self.config.sticky_timeout);

        // Cleanup old probe records
        let timeout = self.config.probe_interval * 10;
        self.last_probe.write().retain(|_, last| last.elapsed() < timeout);
    }

    /// Get uplinks that should be probed.
    pub fn uplinks_to_probe(&self, uplinks: &[Arc<Uplink>]) -> Vec<u16> {
        uplinks.iter()
            .filter(|u| u.is_usable() && self.needs_probe(u))
            .map(|u| u.numeric_id())
            .collect()
    }
}

// Intentionally abbreviated Debug output - internal state not useful for debugging
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("strategy", &self.config.strategy)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = Scheduler::new(SchedulerConfig::default());
        assert_eq!(scheduler.config.strategy, SchedulingStrategy::Adaptive);
    }

    #[test]
    fn test_empty_uplinks() {
        let scheduler = Scheduler::new(SchedulerConfig::default());
        let result = scheduler.select(&[], None);
        assert!(result.is_empty());
    }
}
