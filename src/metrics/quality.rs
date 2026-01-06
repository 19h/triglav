//! Quality metrics tracking.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::types::{Bandwidth, Latency, UplinkHealth};

/// Quality metrics for an uplink.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Smoothed round-trip time.
    pub rtt: Duration,
    /// RTT variance.
    pub rtt_variance: Duration,
    /// Latency statistics (if available).
    pub latency: Option<Latency>,
    /// Upload bandwidth.
    pub bandwidth_up: Bandwidth,
    /// Download bandwidth.
    pub bandwidth_down: Bandwidth,
    /// Packet loss ratio (0.0 - 1.0).
    pub packet_loss: f64,
    /// Jitter (RTT variance).
    pub jitter: Duration,
    /// Current health status.
    pub health: UplinkHealth,
}

impl QualityMetrics {
    /// Calculate an overall quality score (0-100).
    pub fn score(&self) -> u32 {
        let rtt_score = self.rtt_score();
        let loss_score = self.loss_score();
        let jitter_score = self.jitter_score();

        // Weighted average
        let score = rtt_score * 0.4 + loss_score * 0.4 + jitter_score * 0.2;

        // Apply health modifier
        let health_mod = self.health.priority_modifier();

        ((score * health_mod) * 100.0) as u32
    }

    /// RTT score (0-1, lower RTT = higher score).
    fn rtt_score(&self) -> f64 {
        let rtt_ms = self.rtt.as_secs_f64() * 1000.0;
        if rtt_ms == 0.0 {
            1.0
        } else {
            1.0 / (1.0 + rtt_ms / 100.0)
        }
    }

    /// Loss score (0-1, lower loss = higher score).
    fn loss_score(&self) -> f64 {
        1.0 - self.packet_loss.min(1.0)
    }

    /// Jitter score (0-1, lower jitter = higher score).
    fn jitter_score(&self) -> f64 {
        let jitter_ms = self.jitter.as_secs_f64() * 1000.0;
        if jitter_ms == 0.0 {
            1.0
        } else {
            1.0 / (1.0 + jitter_ms / 50.0)
        }
    }

    /// Check if metrics indicate a problem.
    pub fn has_issues(&self) -> bool {
        self.packet_loss > 0.1
            || self.rtt > Duration::from_secs(1)
            || self.health == UplinkHealth::Degraded
            || self.health == UplinkHealth::Unhealthy
    }

    /// Get a human-readable status.
    pub fn status(&self) -> &'static str {
        match self.score() {
            90..=100 => "excellent",
            70..=89 => "good",
            50..=69 => "fair",
            30..=49 => "poor",
            _ => "critical",
        }
    }

    /// Get a brief summary.
    pub fn summary(&self) -> String {
        format!(
            "rtt={:.1}ms loss={:.1}% score={}",
            self.rtt.as_secs_f64() * 1000.0,
            self.packet_loss * 100.0,
            self.score()
        )
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            rtt: Duration::ZERO,
            rtt_variance: Duration::ZERO,
            latency: None,
            bandwidth_up: Bandwidth::ZERO,
            bandwidth_down: Bandwidth::ZERO,
            packet_loss: 0.0,
            jitter: Duration::ZERO,
            health: UplinkHealth::Unknown,
        }
    }
}
