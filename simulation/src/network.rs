//! Network Layer Models
//!
//! Models latency distributions, packet loss patterns, jitter, reordering,
//! and end-to-end network behavior including routing and congestion.

use rand::Rng;
use rand_distr::{Distribution, Normal, LogNormal, Exp, Pareto, Uniform, Weibull};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::physical::{RadioTechnology, SignalQuality};

// ============================================================================
// Latency Models
// ============================================================================

/// Latency distribution types based on empirical network measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyDistribution {
    /// Constant latency (ideal)
    Constant { base_ms: f64 },
    
    /// Normal distribution (symmetric variation)
    Normal { mean_ms: f64, std_dev_ms: f64 },
    
    /// Log-normal distribution (long tail, common in real networks)
    LogNormal { mu: f64, sigma: f64 },
    
    /// Pareto distribution (heavy tail, models congestion spikes)
    Pareto { scale: f64, shape: f64 },
    
    /// Weibull distribution (models aging/degradation effects)
    Weibull { scale: f64, shape: f64 },
    
    /// Bimodal (models queueing with occasional congestion)
    Bimodal {
        low_mean_ms: f64,
        low_std_ms: f64,
        high_mean_ms: f64,
        high_std_ms: f64,
        high_probability: f64,
    },
    
    /// Empirical (based on measured CDF)
    Empirical { percentiles: Vec<(f64, f64)> },
}

impl LatencyDistribution {
    /// Sample a latency value in milliseconds
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            LatencyDistribution::Constant { base_ms } => *base_ms,
            
            LatencyDistribution::Normal { mean_ms, std_dev_ms } => {
                let normal = Normal::new(*mean_ms, *std_dev_ms).unwrap();
                normal.sample(rng).max(0.1)
            }
            
            LatencyDistribution::LogNormal { mu, sigma } => {
                let lognormal = LogNormal::new(*mu, *sigma).unwrap();
                lognormal.sample(rng)
            }
            
            LatencyDistribution::Pareto { scale, shape } => {
                let pareto = Pareto::new(*scale, *shape).unwrap();
                pareto.sample(rng)
            }
            
            LatencyDistribution::Weibull { scale, shape } => {
                let weibull = Weibull::new(*scale, *shape).unwrap();
                weibull.sample(rng)
            }
            
            LatencyDistribution::Bimodal {
                low_mean_ms,
                low_std_ms,
                high_mean_ms,
                high_std_ms,
                high_probability,
            } => {
                if rng.gen::<f64>() < *high_probability {
                    let normal = Normal::new(*high_mean_ms, *high_std_ms).unwrap();
                    normal.sample(rng).max(0.1)
                } else {
                    let normal = Normal::new(*low_mean_ms, *low_std_ms).unwrap();
                    normal.sample(rng).max(0.1)
                }
            }
            
            LatencyDistribution::Empirical { percentiles } => {
                let p = rng.gen::<f64>();
                for window in percentiles.windows(2) {
                    if p >= window[0].0 && p < window[1].0 {
                        // Linear interpolation
                        let ratio = (p - window[0].0) / (window[1].0 - window[0].0);
                        return window[0].1 + ratio * (window[1].1 - window[0].1);
                    }
                }
                percentiles.last().map(|p| p.1).unwrap_or(50.0)
            }
        }
    }

    /// Create distribution for typical LTE network
    pub fn typical_lte() -> Self {
        LatencyDistribution::LogNormal {
            mu: 3.5,  // ~33ms median
            sigma: 0.5,
        }
    }

    /// Create distribution for typical 5G network
    pub fn typical_5g() -> Self {
        LatencyDistribution::LogNormal {
            mu: 2.3,  // ~10ms median
            sigma: 0.4,
        }
    }

    /// Create distribution for WiFi network
    pub fn typical_wifi() -> Self {
        LatencyDistribution::Bimodal {
            low_mean_ms: 5.0,
            low_std_ms: 2.0,
            high_mean_ms: 50.0,
            high_std_ms: 30.0,
            high_probability: 0.1,
        }
    }

    /// Create distribution for congested network
    pub fn congested() -> Self {
        LatencyDistribution::Pareto {
            scale: 100.0,
            shape: 1.5,
        }
    }
}

// ============================================================================
// Packet Loss Models
// ============================================================================

/// Packet loss pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossPattern {
    /// Independent random loss (Bernoulli)
    Random { probability: f64 },
    
    /// Bursty loss (Gilbert-Elliott model)
    GilbertElliott {
        /// Probability of transitioning from good to bad state
        p_good_to_bad: f64,
        /// Probability of transitioning from bad to good state
        p_bad_to_good: f64,
        /// Loss probability in good state
        loss_in_good: f64,
        /// Loss probability in bad state
        loss_in_bad: f64,
    },
    
    /// Correlated loss (based on previous packet)
    Markov {
        /// Base loss probability
        base_loss: f64,
        /// Additional loss probability if previous packet was lost
        correlation: f64,
    },
    
    /// Periodic loss (e.g., interference pattern)
    Periodic {
        /// Period in packets
        period: usize,
        /// Loss probability during burst
        burst_loss: f64,
        /// Burst length in packets
        burst_length: usize,
    },
    
    /// Signal-quality dependent loss
    SignalDependent {
        /// Coefficients for SINR to loss mapping
        sinr_threshold_db: f64,
        slope: f64,
    },
}

impl LossPattern {
    /// Determine if a packet is lost
    pub fn is_lost<R: Rng>(&mut self, rng: &mut R, state: &mut LossState) -> bool {
        match self {
            LossPattern::Random { probability } => {
                rng.gen::<f64>() < *probability
            }
            
            LossPattern::GilbertElliott {
                p_good_to_bad,
                p_bad_to_good,
                loss_in_good,
                loss_in_bad,
            } => {
                // State transition
                if state.in_bad_state {
                    if rng.gen::<f64>() < *p_bad_to_good {
                        state.in_bad_state = false;
                    }
                } else {
                    if rng.gen::<f64>() < *p_good_to_bad {
                        state.in_bad_state = true;
                    }
                }
                
                // Loss decision
                let loss_prob = if state.in_bad_state { *loss_in_bad } else { *loss_in_good };
                let lost = rng.gen::<f64>() < loss_prob;
                state.previous_lost = lost;
                lost
            }
            
            LossPattern::Markov { base_loss, correlation } => {
                let loss_prob = if state.previous_lost {
                    (*base_loss + *correlation).min(1.0)
                } else {
                    *base_loss
                };
                let lost = rng.gen::<f64>() < loss_prob;
                state.previous_lost = lost;
                lost
            }
            
            LossPattern::Periodic { period, burst_loss, burst_length } => {
                state.packet_count += 1;
                let phase = state.packet_count % *period;
                if phase < *burst_length {
                    rng.gen::<f64>() < *burst_loss
                } else {
                    false
                }
            }
            
            LossPattern::SignalDependent { sinr_threshold_db, slope } => {
                let sinr = state.current_sinr_db;
                let loss_prob = if sinr < *sinr_threshold_db {
                    ((*sinr_threshold_db - sinr) * *slope / 100.0).min(1.0)
                } else {
                    0.001 // Minimum loss
                };
                rng.gen::<f64>() < loss_prob
            }
        }
    }
}

/// State for stateful loss patterns
#[derive(Debug, Clone, Default)]
pub struct LossState {
    pub in_bad_state: bool,
    pub previous_lost: bool,
    pub packet_count: usize,
    pub current_sinr_db: f64,
    pub consecutive_losses: usize,
}

// ============================================================================
// Jitter Models
// ============================================================================

/// Jitter characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterModel {
    /// Base jitter standard deviation (ms)
    pub base_jitter_ms: f64,
    /// Correlation with previous sample (0-1)
    pub correlation: f64,
    /// Maximum jitter bound (ms)
    pub max_jitter_ms: f64,
    /// Spike probability
    pub spike_probability: f64,
    /// Spike magnitude multiplier
    pub spike_multiplier: f64,
}

impl JitterModel {
    pub fn new(base_jitter_ms: f64) -> Self {
        Self {
            base_jitter_ms,
            correlation: 0.3,
            max_jitter_ms: base_jitter_ms * 10.0,
            spike_probability: 0.01,
            spike_multiplier: 5.0,
        }
    }

    /// Sample jitter value
    pub fn sample<R: Rng>(&self, rng: &mut R, previous_jitter: f64) -> f64 {
        let normal = Normal::new(0.0, self.base_jitter_ms).unwrap();
        let innovation = normal.sample(rng);
        
        // Correlated jitter
        let mut jitter = self.correlation * previous_jitter + (1.0 - self.correlation) * innovation;
        
        // Occasional spikes
        if rng.gen::<f64>() < self.spike_probability {
            jitter *= self.spike_multiplier;
        }
        
        // Bound jitter
        jitter.abs().min(self.max_jitter_ms)
    }

    /// Typical LTE jitter
    pub fn typical_lte() -> Self {
        Self {
            base_jitter_ms: 8.0,
            correlation: 0.4,
            max_jitter_ms: 100.0,
            spike_probability: 0.02,
            spike_multiplier: 5.0,
        }
    }

    /// Typical 5G jitter
    pub fn typical_5g() -> Self {
        Self {
            base_jitter_ms: 3.0,
            correlation: 0.3,
            max_jitter_ms: 50.0,
            spike_probability: 0.01,
            spike_multiplier: 4.0,
        }
    }

    /// Typical WiFi jitter (more variable)
    pub fn typical_wifi() -> Self {
        Self {
            base_jitter_ms: 5.0,
            correlation: 0.2,
            max_jitter_ms: 200.0,
            spike_probability: 0.05,
            spike_multiplier: 8.0,
        }
    }
}

// ============================================================================
// Bandwidth Model
// ============================================================================

/// Bandwidth variation model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthModel {
    /// Maximum bandwidth (Mbps)
    pub max_bandwidth_mbps: f64,
    /// Minimum guaranteed bandwidth (Mbps)
    pub min_bandwidth_mbps: f64,
    /// Current available bandwidth (Mbps)
    pub current_bandwidth_mbps: f64,
    /// Variation pattern
    pub variation: BandwidthVariation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BandwidthVariation {
    /// Constant bandwidth
    Constant,
    /// Random walk
    RandomWalk { step_size_mbps: f64 },
    /// Time-of-day pattern (congestion during peak hours)
    TimeOfDay { peak_reduction_factor: f64 },
    /// Signal-quality dependent
    SignalDependent,
    /// TCP-like AIMD pattern
    AIMD { additive_increase: f64, multiplicative_decrease: f64 },
}

impl BandwidthModel {
    pub fn new(max_mbps: f64, min_mbps: f64, variation: BandwidthVariation) -> Self {
        Self {
            max_bandwidth_mbps: max_mbps,
            min_bandwidth_mbps: min_mbps,
            current_bandwidth_mbps: max_mbps,
            variation,
        }
    }

    /// Update bandwidth based on conditions
    pub fn update<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        time_of_day_hours: f64,
        signal_quality: Option<&SignalQuality>,
        congestion_event: bool,
    ) {
        match &self.variation {
            BandwidthVariation::Constant => {}
            
            BandwidthVariation::RandomWalk { step_size_mbps } => {
                let step = if rng.gen::<bool>() { *step_size_mbps } else { -*step_size_mbps };
                self.current_bandwidth_mbps = (self.current_bandwidth_mbps + step)
                    .max(self.min_bandwidth_mbps)
                    .min(self.max_bandwidth_mbps);
            }
            
            BandwidthVariation::TimeOfDay { peak_reduction_factor } => {
                // Peak hours: 8-10, 12-14, 18-22
                let is_peak = (time_of_day_hours >= 8.0 && time_of_day_hours <= 10.0)
                    || (time_of_day_hours >= 12.0 && time_of_day_hours <= 14.0)
                    || (time_of_day_hours >= 18.0 && time_of_day_hours <= 22.0);
                
                if is_peak {
                    self.current_bandwidth_mbps = self.max_bandwidth_mbps * (1.0 - peak_reduction_factor);
                } else {
                    self.current_bandwidth_mbps = self.max_bandwidth_mbps;
                }
            }
            
            BandwidthVariation::SignalDependent => {
                if let Some(sq) = signal_quality {
                    // Bandwidth scales with SINR
                    let factor = ((sq.sinr_db + 10.0) / 30.0).max(0.1).min(1.0);
                    self.current_bandwidth_mbps = self.max_bandwidth_mbps * factor;
                }
            }
            
            BandwidthVariation::AIMD { additive_increase, multiplicative_decrease } => {
                if congestion_event {
                    self.current_bandwidth_mbps *= *multiplicative_decrease;
                } else {
                    self.current_bandwidth_mbps += *additive_increase;
                }
                self.current_bandwidth_mbps = self.current_bandwidth_mbps
                    .max(self.min_bandwidth_mbps)
                    .min(self.max_bandwidth_mbps);
            }
        }
    }
}

// ============================================================================
// Network Path Model
// ============================================================================

/// Complete network path model
#[derive(Debug, Clone)]
pub struct NetworkPath {
    pub id: String,
    pub technology: RadioTechnology,
    pub latency_dist: LatencyDistribution,
    pub loss_pattern: LossPattern,
    pub jitter_model: JitterModel,
    pub bandwidth_model: BandwidthModel,
    
    // State
    pub loss_state: LossState,
    pub previous_jitter: f64,
    pub is_active: bool,
    
    // Metrics history (for algorithms)
    pub rtt_history: VecDeque<f64>,
    pub loss_history: VecDeque<bool>,
    pub throughput_history: VecDeque<f64>,
}

impl NetworkPath {
    pub fn new(
        id: String,
        technology: RadioTechnology,
        latency_dist: LatencyDistribution,
        loss_pattern: LossPattern,
        jitter_model: JitterModel,
        bandwidth_model: BandwidthModel,
    ) -> Self {
        Self {
            id,
            technology,
            latency_dist,
            loss_pattern,
            jitter_model,
            bandwidth_model,
            loss_state: LossState::default(),
            previous_jitter: 0.0,
            is_active: true,
            rtt_history: VecDeque::with_capacity(1000),
            loss_history: VecDeque::with_capacity(1000),
            throughput_history: VecDeque::with_capacity(1000),
        }
    }

    /// Simulate sending a packet, returns (latency_ms, lost)
    pub fn send_packet<R: Rng>(&mut self, rng: &mut R) -> (f64, bool) {
        if !self.is_active {
            return (f64::INFINITY, true);
        }

        // Check for loss
        let lost = self.loss_pattern.is_lost(rng, &mut self.loss_state);
        self.loss_history.push_back(lost);
        if self.loss_history.len() > 1000 {
            self.loss_history.pop_front();
        }

        if lost {
            self.loss_state.consecutive_losses += 1;
            return (f64::INFINITY, true);
        }

        self.loss_state.consecutive_losses = 0;

        // Calculate latency
        let base_latency = self.latency_dist.sample(rng);
        let jitter = self.jitter_model.sample(rng, self.previous_jitter);
        self.previous_jitter = jitter;

        let total_latency = base_latency + jitter;

        // Record RTT (round trip = 2x one-way)
        let rtt = total_latency * 2.0;
        self.rtt_history.push_back(rtt);
        if self.rtt_history.len() > 1000 {
            self.rtt_history.pop_front();
        }

        (total_latency, false)
    }

    /// Get current path quality metrics
    pub fn get_metrics(&self) -> PathMetrics {
        let recent_rtts: Vec<f64> = self.rtt_history.iter().rev().take(20).copied().collect();
        let recent_losses: Vec<bool> = self.loss_history.iter().rev().take(100).copied().collect();

        let avg_rtt = if recent_rtts.is_empty() {
            0.0
        } else {
            recent_rtts.iter().sum::<f64>() / recent_rtts.len() as f64
        };

        let rtt_variance = if recent_rtts.len() > 1 {
            let mean = avg_rtt;
            recent_rtts.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (recent_rtts.len() - 1) as f64
        } else {
            0.0
        };

        let loss_rate = if recent_losses.is_empty() {
            0.0
        } else {
            recent_losses.iter().filter(|&&x| x).count() as f64 / recent_losses.len() as f64
        };

        let jitter = rtt_variance.sqrt();

        PathMetrics {
            avg_rtt_ms: avg_rtt,
            rtt_variance_ms2: rtt_variance,
            jitter_ms: jitter,
            loss_rate,
            bandwidth_mbps: self.bandwidth_model.current_bandwidth_mbps,
            is_active: self.is_active,
            consecutive_losses: self.loss_state.consecutive_losses,
        }
    }

    /// Create typical LTE path
    pub fn typical_lte(id: String) -> Self {
        Self::new(
            id,
            RadioTechnology::LTE,
            LatencyDistribution::typical_lte(),
            LossPattern::GilbertElliott {
                p_good_to_bad: 0.01,
                p_bad_to_good: 0.1,
                loss_in_good: 0.001,
                loss_in_bad: 0.1,
            },
            JitterModel::typical_lte(),
            BandwidthModel::new(100.0, 1.0, BandwidthVariation::SignalDependent),
        )
    }

    /// Create typical 5G path
    pub fn typical_5g(id: String) -> Self {
        Self::new(
            id,
            RadioTechnology::NR5G,
            LatencyDistribution::typical_5g(),
            LossPattern::Random { probability: 0.001 },
            JitterModel::typical_5g(),
            BandwidthModel::new(500.0, 10.0, BandwidthVariation::SignalDependent),
        )
    }

    /// Create typical WiFi path
    pub fn typical_wifi(id: String) -> Self {
        Self::new(
            id,
            RadioTechnology::WiFi6,
            LatencyDistribution::typical_wifi(),
            LossPattern::Markov {
                base_loss: 0.005,
                correlation: 0.3,
            },
            JitterModel::typical_wifi(),
            BandwidthModel::new(200.0, 5.0, BandwidthVariation::RandomWalk { step_size_mbps: 10.0 }),
        )
    }
}

/// Path quality metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PathMetrics {
    pub avg_rtt_ms: f64,
    pub rtt_variance_ms2: f64,
    pub jitter_ms: f64,
    pub loss_rate: f64,
    pub bandwidth_mbps: f64,
    pub is_active: bool,
    pub consecutive_losses: usize,
}

impl PathMetrics {
    /// Calculate a composite quality score (0-1, higher is better)
    pub fn quality_score(&self) -> f64 {
        if !self.is_active {
            return 0.0;
        }

        // Weighted scoring
        let rtt_score = (1.0 - (self.avg_rtt_ms / 500.0).min(1.0)).max(0.0);
        let jitter_score = (1.0 - (self.jitter_ms / 100.0).min(1.0)).max(0.0);
        let loss_score = (1.0 - self.loss_rate * 10.0).max(0.0);
        let bw_score = (self.bandwidth_mbps / 100.0).min(1.0);

        // Weights
        0.3 * rtt_score + 0.2 * jitter_score + 0.3 * loss_score + 0.2 * bw_score
    }
}

// ============================================================================
// Internet Backbone Model
// ============================================================================

/// Models internet backbone behavior (IX, transit, peering)
#[derive(Debug, Clone)]
pub struct BackboneModel {
    /// Number of hops
    pub hop_count: usize,
    /// Per-hop latency distribution
    pub per_hop_latency: LatencyDistribution,
    /// Probability of routing change per minute
    pub routing_instability: f64,
    /// BGP convergence time (ms)
    pub bgp_convergence_ms: f64,
    /// Current state
    pub is_converged: bool,
    pub convergence_timer_ms: f64,
}

impl BackboneModel {
    pub fn new(hop_count: usize) -> Self {
        Self {
            hop_count,
            per_hop_latency: LatencyDistribution::LogNormal {
                mu: 1.0,  // ~2.7ms per hop
                sigma: 0.3,
            },
            routing_instability: 0.001, // 0.1% chance per minute
            bgp_convergence_ms: 30000.0, // 30 seconds
            is_converged: true,
            convergence_timer_ms: 0.0,
        }
    }

    /// Calculate backbone latency
    pub fn latency<R: Rng>(&mut self, rng: &mut R, delta_ms: f64) -> f64 {
        // Check for routing events
        if self.is_converged {
            let event_prob = self.routing_instability * delta_ms / 60000.0;
            if rng.gen::<f64>() < event_prob {
                self.is_converged = false;
                self.convergence_timer_ms = self.bgp_convergence_ms;
            }
        } else {
            self.convergence_timer_ms -= delta_ms;
            if self.convergence_timer_ms <= 0.0 {
                self.is_converged = true;
            }
        }

        // During convergence, latency is much higher
        let multiplier = if self.is_converged { 1.0 } else { 10.0 };

        let mut total_latency = 0.0;
        for _ in 0..self.hop_count {
            total_latency += self.per_hop_latency.sample(rng) * multiplier;
        }

        total_latency
    }

    /// Check if backbone is experiencing issues
    pub fn has_issues(&self) -> bool {
        !self.is_converged
    }
}

// ============================================================================
// Complete Network Link
// ============================================================================

/// Complete network link including radio and backbone
#[derive(Debug, Clone)]
pub struct NetworkLink {
    pub id: String,
    pub path: NetworkPath,
    pub backbone: BackboneModel,
    pub provider: String,
    
    // Aggregate metrics
    pub total_packets_sent: u64,
    pub total_packets_lost: u64,
    pub total_bytes_sent: u64,
}

impl NetworkLink {
    pub fn new(id: String, path: NetworkPath, backbone: BackboneModel, provider: String) -> Self {
        Self {
            id,
            path,
            backbone,
            provider,
            total_packets_sent: 0,
            total_packets_lost: 0,
            total_bytes_sent: 0,
        }
    }

    /// Send a packet through the complete link
    pub fn send_packet<R: Rng>(&mut self, rng: &mut R, delta_ms: f64, packet_size_bytes: usize) -> PacketResult {
        self.total_packets_sent += 1;

        // Radio path
        let (radio_latency, radio_lost) = self.path.send_packet(rng);
        
        if radio_lost {
            self.total_packets_lost += 1;
            return PacketResult {
                latency_ms: f64::INFINITY,
                lost: true,
                loss_reason: LossReason::RadioLoss,
            };
        }

        // Backbone
        let backbone_latency = self.backbone.latency(rng, delta_ms);
        
        // Check for backbone-induced loss during convergence
        if !self.backbone.is_converged && rng.gen::<f64>() < 0.1 {
            self.total_packets_lost += 1;
            return PacketResult {
                latency_ms: f64::INFINITY,
                lost: true,
                loss_reason: LossReason::BackboneLoss,
            };
        }

        self.total_bytes_sent += packet_size_bytes as u64;

        PacketResult {
            latency_ms: radio_latency + backbone_latency,
            lost: false,
            loss_reason: LossReason::None,
        }
    }

    /// Get comprehensive link metrics
    pub fn get_metrics(&self) -> LinkMetrics {
        let path_metrics = self.path.get_metrics();
        
        LinkMetrics {
            path: path_metrics,
            backbone_converged: self.backbone.is_converged,
            total_packets_sent: self.total_packets_sent,
            total_packets_lost: self.total_packets_lost,
            overall_loss_rate: if self.total_packets_sent > 0 {
                self.total_packets_lost as f64 / self.total_packets_sent as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PacketResult {
    pub latency_ms: f64,
    pub lost: bool,
    pub loss_reason: LossReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossReason {
    None,
    RadioLoss,
    BackboneLoss,
    Congestion,
    Handover,
    OutOfCoverage,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinkMetrics {
    pub path: PathMetrics,
    pub backbone_converged: bool,
    pub total_packets_sent: u64,
    pub total_packets_lost: u64,
    pub overall_loss_rate: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_latency_distributions() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let dists = [
            LatencyDistribution::typical_lte(),
            LatencyDistribution::typical_5g(),
            LatencyDistribution::typical_wifi(),
        ];

        for dist in &dists {
            let samples: Vec<f64> = (0..1000).map(|_| dist.sample(&mut rng)).collect();
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            assert!(mean > 0.0 && mean < 500.0, "Mean latency should be reasonable");
        }
    }

    #[test]
    fn test_loss_patterns() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut state = LossState::default();

        let mut pattern = LossPattern::GilbertElliott {
            p_good_to_bad: 0.05,
            p_bad_to_good: 0.2,
            loss_in_good: 0.01,
            loss_in_bad: 0.5,
        };

        let mut losses = 0;
        let trials = 10000;
        for _ in 0..trials {
            if pattern.is_lost(&mut rng, &mut state) {
                losses += 1;
            }
        }

        let loss_rate = losses as f64 / trials as f64;
        assert!(loss_rate > 0.01 && loss_rate < 0.3, "Loss rate should be moderate");
    }

    #[test]
    fn test_network_path() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut path = NetworkPath::typical_5g("test_5g".to_string());

        for _ in 0..100 {
            let (latency, lost) = path.send_packet(&mut rng);
            if !lost {
                assert!(latency > 0.0 && latency < 1000.0);
            }
        }

        let metrics = path.get_metrics();
        assert!(metrics.avg_rtt_ms > 0.0);
        assert!(metrics.loss_rate < 0.5);
    }
}
