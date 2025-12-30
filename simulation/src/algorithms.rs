//! Failover Algorithm Implementations
//!
//! Various failover algorithms from simple threshold-based to advanced ML-based.
//! Each algorithm is evaluated across all scenarios.

use std::collections::{HashMap, VecDeque};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::network::{PathMetrics, LinkMetrics};

// ============================================================================
// Algorithm Trait
// ============================================================================

/// Failover decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailoverDecision {
    /// Stay on current path
    StayCurrent,
    /// Switch to specified path
    SwitchTo(usize),
    /// Use multiple paths simultaneously
    MultiPath(u32), // Bitmask of paths
}

/// Failover algorithm interface
pub trait FailoverAlgorithm: Send + Sync {
    /// Algorithm identifier
    fn id(&self) -> &str;
    
    /// Human-readable description
    fn description(&self) -> String;
    
    /// Initialize algorithm state
    fn init(&mut self, path_count: usize);
    
    /// Update algorithm with new metrics and decide on failover
    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision;
    
    /// Get algorithm-specific parameters for reporting
    fn get_parameters(&self) -> HashMap<String, f64>;
    
    /// Reset algorithm state
    fn reset(&mut self);
}

// ============================================================================
// 1. Simple Threshold-Based Failover
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    pub rtt_threshold_ms: f64,
    pub loss_threshold_percent: f64,
    pub consecutive_failures: usize,
    pub recovery_probe_interval_ms: u64,
    pub hysteresis_ms: u64,
}

pub struct ThresholdFailover {
    id: String,
    config: ThresholdConfig,
    failure_counts: Vec<usize>,
    last_switch_time_ms: f64,
    current_time_ms: f64,
}

impl ThresholdFailover {
    pub fn new(id: &str, config: ThresholdConfig) -> Self {
        Self {
            id: id.to_string(),
            config,
            failure_counts: Vec::new(),
            last_switch_time_ms: 0.0,
            current_time_ms: 0.0,
        }
    }
}

impl FailoverAlgorithm for ThresholdFailover {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "Threshold: RTT>{:.0}ms or loss>{:.1}%, {} failures, {}ms hysteresis",
            self.config.rtt_threshold_ms,
            self.config.loss_threshold_percent,
            self.config.consecutive_failures,
            self.config.hysteresis_ms
        )
    }

    fn init(&mut self, path_count: usize) {
        self.failure_counts = vec![0; path_count];
        self.last_switch_time_ms = 0.0;
        self.current_time_ms = 0.0;
    }

    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        self.current_time_ms += 10.0; // Assume 10ms ticks

        // Check if we're in hysteresis period
        if self.current_time_ms - self.last_switch_time_ms < self.config.hysteresis_ms as f64 {
            return FailoverDecision::StayCurrent;
        }

        // Check current path health
        let current_metrics = &metrics[current_path];
        let current_failed = !current_metrics.is_active
            || current_metrics.avg_rtt_ms > self.config.rtt_threshold_ms
            || current_metrics.loss_rate * 100.0 > self.config.loss_threshold_percent;

        if current_failed {
            self.failure_counts[current_path] += 1;
        } else {
            self.failure_counts[current_path] = 0;
        }

        // Check if we need to failover
        if self.failure_counts[current_path] >= self.config.consecutive_failures {
            // Find best alternative
            let mut best_path = current_path;
            let mut best_score = f64::MIN;

            for (i, m) in metrics.iter().enumerate() {
                if i == current_path || !m.is_active {
                    continue;
                }

                let score = m.quality_score();
                if score > best_score {
                    best_score = score;
                    best_path = i;
                }
            }

            if best_path != current_path {
                self.last_switch_time_ms = self.current_time_ms;
                self.failure_counts[current_path] = 0;
                return FailoverDecision::SwitchTo(best_path);
            }
        }

        FailoverDecision::StayCurrent
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("rtt_threshold_ms".to_string(), self.config.rtt_threshold_ms);
        params.insert("loss_threshold_percent".to_string(), self.config.loss_threshold_percent);
        params.insert("consecutive_failures".to_string(), self.config.consecutive_failures as f64);
        params.insert("hysteresis_ms".to_string(), self.config.hysteresis_ms as f64);
        params
    }

    fn reset(&mut self) {
        self.failure_counts.fill(0);
        self.last_switch_time_ms = 0.0;
        self.current_time_ms = 0.0;
    }
}

// ============================================================================
// 2. Weighted Moving Average Failover
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WMAConfig {
    pub window_size: usize,
    pub weights: Vec<f64>,
    pub adaptive_threshold: bool,
    pub baseline_learning_period_s: u64,
    pub threshold_multiplier: f64,
}

pub struct WeightedMovingAverageFailover {
    id: String,
    config: WMAConfig,
    rtt_windows: Vec<VecDeque<f64>>,
    baselines: Vec<f64>,
    learning: bool,
    samples_collected: usize,
}

impl WeightedMovingAverageFailover {
    pub fn new(id: &str, config: WMAConfig) -> Self {
        Self {
            id: id.to_string(),
            config,
            rtt_windows: Vec::new(),
            baselines: Vec::new(),
            learning: true,
            samples_collected: 0,
        }
    }

    fn weighted_average(&self, values: &VecDeque<f64>) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut weight_sum = 0.0;
        
        for (i, &val) in values.iter().rev().take(self.config.weights.len()).enumerate() {
            let weight = self.config.weights.get(i).copied().unwrap_or(0.1);
            sum += val * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            0.0
        }
    }
}

impl FailoverAlgorithm for WeightedMovingAverageFailover {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "WMA: window={}, adaptive={}, multiplier={:.1}",
            self.config.window_size,
            self.config.adaptive_threshold,
            self.config.threshold_multiplier
        )
    }

    fn init(&mut self, path_count: usize) {
        self.rtt_windows = vec![VecDeque::with_capacity(self.config.window_size); path_count];
        self.baselines = vec![0.0; path_count];
        self.learning = true;
        self.samples_collected = 0;
    }

    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        // Update RTT windows
        for (i, m) in metrics.iter().enumerate() {
            if m.is_active {
                self.rtt_windows[i].push_back(m.avg_rtt_ms);
                if self.rtt_windows[i].len() > self.config.window_size {
                    self.rtt_windows[i].pop_front();
                }
            }
        }

        self.samples_collected += 1;

        // Learning phase
        if self.learning {
            if self.samples_collected >= (self.config.baseline_learning_period_s * 100) as usize {
                for (i, window) in self.rtt_windows.iter().enumerate() {
                    self.baselines[i] = self.weighted_average(window);
                }
                self.learning = false;
            }
            return FailoverDecision::StayCurrent;
        }

        // Check current path
        let current_wma = self.weighted_average(&self.rtt_windows[current_path]);
        let threshold = if self.config.adaptive_threshold {
            self.baselines[current_path] * self.config.threshold_multiplier
        } else {
            200.0 // Fixed fallback
        };

        if current_wma > threshold || !metrics[current_path].is_active {
            // Find best alternative
            let mut best_path = current_path;
            let mut best_wma = f64::MAX;

            for (i, m) in metrics.iter().enumerate() {
                if i == current_path || !m.is_active {
                    continue;
                }

                let wma = self.weighted_average(&self.rtt_windows[i]);
                if wma < best_wma {
                    best_wma = wma;
                    best_path = i;
                }
            }

            if best_path != current_path && best_wma < current_wma * 0.8 {
                return FailoverDecision::SwitchTo(best_path);
            }
        }

        FailoverDecision::StayCurrent
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("window_size".to_string(), self.config.window_size as f64);
        params.insert("threshold_multiplier".to_string(), self.config.threshold_multiplier);
        params
    }

    fn reset(&mut self) {
        for window in &mut self.rtt_windows {
            window.clear();
        }
        self.baselines.fill(0.0);
        self.learning = true;
        self.samples_collected = 0;
    }
}

// ============================================================================
// 3. EWMA (Exponential Weighted Moving Average) Failover
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWMAConfig {
    pub alpha_rtt: f64,
    pub alpha_variance: f64,
    pub rtt_threshold_factor: f64,
    pub min_samples: usize,
}

pub struct EWMAFailover {
    id: String,
    config: EWMAConfig,
    srtt: Vec<f64>,      // Smoothed RTT
    rttvar: Vec<f64>,    // RTT variance
    sample_counts: Vec<usize>,
}

impl EWMAFailover {
    pub fn new(id: &str, config: EWMAConfig) -> Self {
        Self {
            id: id.to_string(),
            config,
            srtt: Vec::new(),
            rttvar: Vec::new(),
            sample_counts: Vec::new(),
        }
    }

    fn rto(&self, path: usize) -> f64 {
        // TCP-style RTO calculation
        self.srtt[path] + self.config.rtt_threshold_factor * self.rttvar[path]
    }
}

impl FailoverAlgorithm for EWMAFailover {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "EWMA: alpha_rtt={:.3}, alpha_var={:.3}, factor={:.1}",
            self.config.alpha_rtt,
            self.config.alpha_variance,
            self.config.rtt_threshold_factor
        )
    }

    fn init(&mut self, path_count: usize) {
        self.srtt = vec![0.0; path_count];
        self.rttvar = vec![0.0; path_count];
        self.sample_counts = vec![0; path_count];
    }

    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        // Update EWMA for all paths
        for (i, m) in metrics.iter().enumerate() {
            if !m.is_active {
                continue;
            }

            let rtt = m.avg_rtt_ms;
            self.sample_counts[i] += 1;

            if self.sample_counts[i] == 1 {
                // First sample
                self.srtt[i] = rtt;
                self.rttvar[i] = rtt / 2.0;
            } else {
                // EWMA update (RFC 6298)
                let err = (rtt - self.srtt[i]).abs();
                self.rttvar[i] = (1.0 - self.config.alpha_variance) * self.rttvar[i]
                    + self.config.alpha_variance * err;
                self.srtt[i] = (1.0 - self.config.alpha_rtt) * self.srtt[i]
                    + self.config.alpha_rtt * rtt;
            }
        }

        // Check if current path exceeds threshold
        if self.sample_counts[current_path] < self.config.min_samples {
            return FailoverDecision::StayCurrent;
        }

        let current_rto = self.rto(current_path);
        let current_rtt = metrics[current_path].avg_rtt_ms;

        if !metrics[current_path].is_active || current_rtt > current_rto {
            // Find best alternative
            let mut best_path = current_path;
            let mut best_srtt = f64::MAX;

            for (i, m) in metrics.iter().enumerate() {
                if i == current_path || !m.is_active {
                    continue;
                }
                if self.sample_counts[i] >= self.config.min_samples && self.srtt[i] < best_srtt {
                    best_srtt = self.srtt[i];
                    best_path = i;
                }
            }

            if best_path != current_path {
                return FailoverDecision::SwitchTo(best_path);
            }
        }

        FailoverDecision::StayCurrent
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("alpha_rtt".to_string(), self.config.alpha_rtt);
        params.insert("alpha_variance".to_string(), self.config.alpha_variance);
        params.insert("rtt_threshold_factor".to_string(), self.config.rtt_threshold_factor);
        params
    }

    fn reset(&mut self) {
        self.srtt.fill(0.0);
        self.rttvar.fill(0.0);
        self.sample_counts.fill(0);
    }
}

// ============================================================================
// 4. Statistical Outlier Detection (Z-Score)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZScoreConfig {
    pub z_threshold: f64,
    pub window_size: usize,
    pub min_samples: usize,
    pub outlier_streak_threshold: usize,
}

pub struct StatisticalOutlierFailover {
    id: String,
    config: ZScoreConfig,
    windows: Vec<VecDeque<f64>>,
    outlier_streaks: Vec<usize>,
}

impl StatisticalOutlierFailover {
    pub fn new(id: &str, config: ZScoreConfig) -> Self {
        Self {
            id: id.to_string(),
            config,
            windows: Vec::new(),
            outlier_streaks: Vec::new(),
        }
    }

    fn z_score(&self, path: usize, value: f64) -> f64 {
        let window = &self.windows[path];
        if window.len() < 2 {
            return 0.0;
        }

        let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
        let variance: f64 = window.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (window.len() - 1) as f64;
        let std_dev = variance.sqrt();

        if std_dev < 0.001 {
            return 0.0;
        }

        (value - mean) / std_dev
    }
}

impl FailoverAlgorithm for StatisticalOutlierFailover {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "Z-Score: threshold={:.1}, window={}, streak={}",
            self.config.z_threshold,
            self.config.window_size,
            self.config.outlier_streak_threshold
        )
    }

    fn init(&mut self, path_count: usize) {
        self.windows = vec![VecDeque::with_capacity(self.config.window_size); path_count];
        self.outlier_streaks = vec![0; path_count];
    }

    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        // Update windows
        for (i, m) in metrics.iter().enumerate() {
            if m.is_active {
                self.windows[i].push_back(m.avg_rtt_ms);
                if self.windows[i].len() > self.config.window_size {
                    self.windows[i].pop_front();
                }
            }
        }

        if self.windows[current_path].len() < self.config.min_samples {
            return FailoverDecision::StayCurrent;
        }

        // Check for outliers
        let current_rtt = metrics[current_path].avg_rtt_ms;
        let z = self.z_score(current_path, current_rtt);

        if !metrics[current_path].is_active || z > self.config.z_threshold {
            self.outlier_streaks[current_path] += 1;
        } else {
            self.outlier_streaks[current_path] = 0;
        }

        // Failover if streak threshold exceeded
        if self.outlier_streaks[current_path] >= self.config.outlier_streak_threshold {
            let mut best_path = current_path;
            let mut best_mean = f64::MAX;

            for (i, window) in self.windows.iter().enumerate() {
                if i == current_path || !metrics[i].is_active || window.len() < self.config.min_samples {
                    continue;
                }
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                if mean < best_mean {
                    best_mean = mean;
                    best_path = i;
                }
            }

            if best_path != current_path {
                self.outlier_streaks[current_path] = 0;
                return FailoverDecision::SwitchTo(best_path);
            }
        }

        FailoverDecision::StayCurrent
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("z_threshold".to_string(), self.config.z_threshold);
        params.insert("window_size".to_string(), self.config.window_size as f64);
        params.insert("outlier_streak_threshold".to_string(), self.config.outlier_streak_threshold as f64);
        params
    }

    fn reset(&mut self) {
        for window in &mut self.windows {
            window.clear();
        }
        self.outlier_streaks.fill(0);
    }
}

// ============================================================================
// 5. Kalman Filter Based Prediction
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanConfig {
    pub process_noise: f64,
    pub measurement_noise: f64,
    pub prediction_horizon_ms: u64,
    pub confidence_threshold: f64,
}

pub struct KalmanFilterFailover {
    id: String,
    config: KalmanConfig,
    // State: [rtt, rtt_velocity]
    states: Vec<[f64; 2]>,
    // Covariance matrices
    covariances: Vec<[[f64; 2]; 2]>,
}

impl KalmanFilterFailover {
    pub fn new(id: &str, config: KalmanConfig) -> Self {
        Self {
            id: id.to_string(),
            config,
            states: Vec::new(),
            covariances: Vec::new(),
        }
    }

    fn predict(&self, path: usize) -> (f64, f64) {
        let state = self.states[path];
        let cov = self.covariances[path];
        
        // Predict future RTT
        let dt = self.config.prediction_horizon_ms as f64 / 1000.0;
        let predicted_rtt = state[0] + state[1] * dt;
        
        // Confidence based on covariance
        let confidence = 1.0 / (1.0 + cov[0][0].sqrt());
        
        (predicted_rtt, confidence)
    }

    fn update_kalman(&mut self, path: usize, measurement: f64) {
        let q = self.config.process_noise;
        let r = self.config.measurement_noise;
        
        // Predict step
        let mut state = self.states[path];
        let mut cov = self.covariances[path];
        
        // Process model: x' = x + v*dt, v' = v
        let dt = 0.01; // 10ms
        state[0] += state[1] * dt;
        cov[0][0] += cov[0][1] * dt + cov[1][0] * dt + cov[1][1] * dt * dt + q;
        cov[0][1] += cov[1][1] * dt;
        cov[1][0] += cov[1][1] * dt;
        cov[1][1] += q;
        
        // Update step
        let y = measurement - state[0]; // Innovation
        let s = cov[0][0] + r; // Innovation covariance
        let k = [cov[0][0] / s, cov[1][0] / s]; // Kalman gain
        
        state[0] += k[0] * y;
        state[1] += k[1] * y;
        
        cov[0][0] -= k[0] * cov[0][0];
        cov[0][1] -= k[0] * cov[0][1];
        cov[1][0] -= k[1] * cov[0][0];
        cov[1][1] -= k[1] * cov[0][1];
        
        self.states[path] = state;
        self.covariances[path] = cov;
    }
}

impl FailoverAlgorithm for KalmanFilterFailover {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "Kalman: Q={:.3}, R={:.3}, horizon={}ms",
            self.config.process_noise,
            self.config.measurement_noise,
            self.config.prediction_horizon_ms
        )
    }

    fn init(&mut self, path_count: usize) {
        self.states = vec![[50.0, 0.0]; path_count]; // Initial RTT estimate
        self.covariances = vec![[[100.0, 0.0], [0.0, 10.0]]; path_count];
    }

    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        // Update Kalman filters for all paths
        for (i, m) in metrics.iter().enumerate() {
            if m.is_active {
                self.update_kalman(i, m.avg_rtt_ms);
            }
        }

        // Predict current path
        let (current_predicted, current_conf) = self.predict(current_path);
        
        // Check if current path is degrading
        if !metrics[current_path].is_active 
            || (current_predicted > 200.0 && current_conf > self.config.confidence_threshold) {
            
            // Find best predicted alternative
            let mut best_path = current_path;
            let mut best_predicted = f64::MAX;

            for (i, m) in metrics.iter().enumerate() {
                if i == current_path || !m.is_active {
                    continue;
                }
                let (predicted, conf) = self.predict(i);
                if conf > self.config.confidence_threshold && predicted < best_predicted {
                    best_predicted = predicted;
                    best_path = i;
                }
            }

            if best_path != current_path && best_predicted < current_predicted * 0.7 {
                return FailoverDecision::SwitchTo(best_path);
            }
        }

        FailoverDecision::StayCurrent
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("process_noise".to_string(), self.config.process_noise);
        params.insert("measurement_noise".to_string(), self.config.measurement_noise);
        params.insert("prediction_horizon_ms".to_string(), self.config.prediction_horizon_ms as f64);
        params
    }

    fn reset(&mut self) {
        for state in &mut self.states {
            *state = [50.0, 0.0];
        }
        for cov in &mut self.covariances {
            *cov = [[100.0, 0.0], [0.0, 10.0]];
        }
    }
}

// ============================================================================
// 6. Multi-Armed Bandit (UCB1)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditConfig {
    pub exploration_factor: f64,
    pub reward_decay: f64,
    pub min_exploration_rounds: usize,
}

pub struct BanditFailover {
    id: String,
    config: BanditConfig,
    plays: Vec<usize>,
    rewards: Vec<f64>,
    total_plays: usize,
}

impl BanditFailover {
    pub fn new(id: &str, config: BanditConfig) -> Self {
        Self {
            id: id.to_string(),
            config,
            plays: Vec::new(),
            rewards: Vec::new(),
            total_plays: 0,
        }
    }

    fn ucb1_score(&self, arm: usize) -> f64 {
        if self.plays[arm] == 0 {
            return f64::MAX;
        }

        let avg_reward = self.rewards[arm] / self.plays[arm] as f64;
        let exploration = self.config.exploration_factor
            * ((self.total_plays as f64).ln() / self.plays[arm] as f64).sqrt();

        avg_reward + exploration
    }
}

impl FailoverAlgorithm for BanditFailover {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "UCB1 Bandit: exploration={:.1}, decay={:.3}",
            self.config.exploration_factor,
            self.config.reward_decay
        )
    }

    fn init(&mut self, path_count: usize) {
        self.plays = vec![0; path_count];
        self.rewards = vec![0.0; path_count];
        self.total_plays = 0;
    }

    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        // Calculate reward for current path (inverse of RTT + loss penalty)
        let reward = if metrics[current_path].is_active {
            let rtt_reward = 1.0 / (1.0 + metrics[current_path].avg_rtt_ms / 100.0);
            let loss_penalty = 1.0 - metrics[current_path].loss_rate * 10.0;
            (rtt_reward * loss_penalty).max(0.0)
        } else {
            0.0
        };

        // Decay old rewards and add new
        self.rewards[current_path] *= self.config.reward_decay;
        self.rewards[current_path] += reward;
        self.plays[current_path] += 1;
        self.total_plays += 1;

        // Exploration phase
        if self.total_plays < self.config.min_exploration_rounds * metrics.len() {
            // Round-robin exploration
            let next = (current_path + 1) % metrics.len();
            if metrics[next].is_active && self.plays[next] < self.config.min_exploration_rounds {
                return FailoverDecision::SwitchTo(next);
            }
        }

        // UCB1 selection
        let mut best_arm = current_path;
        let mut best_score = f64::MIN;

        for (i, m) in metrics.iter().enumerate() {
            if !m.is_active {
                continue;
            }
            let score = self.ucb1_score(i);
            if score > best_score {
                best_score = score;
                best_arm = i;
            }
        }

        if best_arm != current_path {
            FailoverDecision::SwitchTo(best_arm)
        } else {
            FailoverDecision::StayCurrent
        }
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("exploration_factor".to_string(), self.config.exploration_factor);
        params.insert("reward_decay".to_string(), self.config.reward_decay);
        params
    }

    fn reset(&mut self) {
        self.plays.fill(0);
        self.rewards.fill(0.0);
        self.total_plays = 0;
    }
}

// ============================================================================
// 7. Q-Learning Failover
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QLearningConfig {
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: f64,
    pub state_discretization: StateDiscretization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDiscretization {
    pub rtt_buckets: Vec<f64>,
    pub loss_buckets: Vec<f64>,
    pub jitter_buckets: Vec<f64>,
}

pub struct QLearningFailover {
    id: String,
    config: QLearningConfig,
    q_table: HashMap<(usize, usize), f64>, // (state, action) -> Q-value
    epsilon: f64,
    previous_state: Option<usize>,
    previous_action: Option<usize>,
}

impl QLearningFailover {
    pub fn new(id: &str, config: QLearningConfig) -> Self {
        Self {
            id: id.to_string(),
            config,
            q_table: HashMap::new(),
            epsilon: 1.0,
            previous_state: None,
            previous_action: None,
        }
    }

    fn discretize_state(&self, metrics: &[PathMetrics]) -> usize {
        let mut state = 0;
        
        for (i, m) in metrics.iter().enumerate() {
            // RTT bucket
            let rtt_bucket = self.config.state_discretization.rtt_buckets.iter()
                .position(|&t| m.avg_rtt_ms < t)
                .unwrap_or(self.config.state_discretization.rtt_buckets.len());
            
            // Loss bucket
            let loss_bucket = self.config.state_discretization.loss_buckets.iter()
                .position(|&t| m.loss_rate * 100.0 < t)
                .unwrap_or(self.config.state_discretization.loss_buckets.len());
            
            // Combine into state
            state = state * 100 + rtt_bucket * 10 + loss_bucket + i * 1000;
        }
        
        state
    }

    fn get_q_value(&self, state: usize, action: usize) -> f64 {
        *self.q_table.get(&(state, action)).unwrap_or(&0.0)
    }

    fn set_q_value(&mut self, state: usize, action: usize, value: f64) {
        self.q_table.insert((state, action), value);
    }
}

impl FailoverAlgorithm for QLearningFailover {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "Q-Learning: lr={:.2}, gamma={:.2}, eps={:.2}->{:.2}",
            self.config.learning_rate,
            self.config.discount_factor,
            self.config.epsilon_start,
            self.config.epsilon_end
        )
    }

    fn init(&mut self, _path_count: usize) {
        self.q_table.clear();
        self.epsilon = self.config.epsilon_start;
        self.previous_state = None;
        self.previous_action = None;
    }

    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        let current_state = self.discretize_state(metrics);
        
        // Calculate reward
        let reward = if metrics[current_path].is_active {
            let rtt_reward = 100.0 / (1.0 + metrics[current_path].avg_rtt_ms);
            let loss_penalty = -metrics[current_path].loss_rate * 100.0;
            rtt_reward + loss_penalty
        } else {
            -100.0
        };

        // Q-learning update
        if let (Some(prev_state), Some(prev_action)) = (self.previous_state, self.previous_action) {
            let max_future_q = (0..metrics.len())
                .map(|a| self.get_q_value(current_state, a))
                .fold(f64::MIN, f64::max);
            
            let old_q = self.get_q_value(prev_state, prev_action);
            let new_q = old_q + self.config.learning_rate 
                * (reward + self.config.discount_factor * max_future_q - old_q);
            
            self.set_q_value(prev_state, prev_action, new_q);
        }

        // Epsilon-greedy action selection
        let action = if rand::random::<f64>() < self.epsilon {
            // Explore
            rand::random::<usize>() % metrics.len()
        } else {
            // Exploit
            (0..metrics.len())
                .filter(|&i| metrics[i].is_active)
                .max_by(|&a, &b| {
                    self.get_q_value(current_state, a)
                        .partial_cmp(&self.get_q_value(current_state, b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(current_path)
        };

        // Decay epsilon
        self.epsilon = (self.epsilon * self.config.epsilon_decay)
            .max(self.config.epsilon_end);

        self.previous_state = Some(current_state);
        self.previous_action = Some(action);

        if action != current_path && metrics[action].is_active {
            FailoverDecision::SwitchTo(action)
        } else {
            FailoverDecision::StayCurrent
        }
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.config.learning_rate);
        params.insert("discount_factor".to_string(), self.config.discount_factor);
        params.insert("epsilon".to_string(), self.epsilon);
        params.insert("q_table_size".to_string(), self.q_table.len() as f64);
        params
    }

    fn reset(&mut self) {
        self.q_table.clear();
        self.epsilon = self.config.epsilon_start;
        self.previous_state = None;
        self.previous_action = None;
    }
}

// ============================================================================
// 8. Ensemble Failover
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleComponent {
    EWMA { weight: f64, alpha: f64 },
    ZScore { weight: f64, threshold: f64 },
    Kalman { weight: f64, process_noise: f64 },
    Threshold { weight: f64, rtt_ms: f64 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VotingStrategy {
    WeightedMajority,
    Unanimous,
    Any,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    pub components: Vec<EnsembleComponent>,
    pub voting_strategy: VotingStrategy,
    pub confidence_threshold: f64,
}

pub struct EnsembleFailover {
    id: String,
    config: EnsembleConfig,
    // Simplified: just track votes
    votes: Vec<(usize, f64)>, // (path_index, weight)
}

impl EnsembleFailover {
    pub fn new(id: &str, config: EnsembleConfig) -> Self {
        Self {
            id: id.to_string(),
            config,
            votes: Vec::new(),
        }
    }
}

impl FailoverAlgorithm for EnsembleFailover {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "Ensemble: {} components, {:?} voting",
            self.config.components.len(),
            self.config.voting_strategy
        )
    }

    fn init(&mut self, _path_count: usize) {
        self.votes.clear();
    }

    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        self.votes.clear();

        // Collect votes from each component
        for component in &self.config.components {
            let (vote, weight) = match component {
                EnsembleComponent::EWMA { weight, alpha: _ } => {
                    // Simple: vote for path with lowest RTT
                    let best = metrics.iter().enumerate()
                        .filter(|(_, m)| m.is_active)
                        .min_by(|(_, a), (_, b)| 
                            a.avg_rtt_ms.partial_cmp(&b.avg_rtt_ms).unwrap())
                        .map(|(i, _)| i);
                    (best, *weight)
                }
                EnsembleComponent::ZScore { weight, threshold } => {
                    // Vote against current if RTT is high
                    if metrics[current_path].avg_rtt_ms > *threshold * 2.0 {
                        let best = metrics.iter().enumerate()
                            .filter(|(i, m)| *i != current_path && m.is_active)
                            .min_by(|(_, a), (_, b)|
                                a.avg_rtt_ms.partial_cmp(&b.avg_rtt_ms).unwrap())
                            .map(|(i, _)| i);
                        (best, *weight)
                    } else {
                        (Some(current_path), *weight)
                    }
                }
                EnsembleComponent::Kalman { weight, process_noise: _ } => {
                    // Vote for path with best quality score
                    let best = metrics.iter().enumerate()
                        .filter(|(_, m)| m.is_active)
                        .max_by(|(_, a), (_, b)|
                            a.quality_score().partial_cmp(&b.quality_score()).unwrap())
                        .map(|(i, _)| i);
                    (best, *weight)
                }
                EnsembleComponent::Threshold { weight, rtt_ms } => {
                    if metrics[current_path].avg_rtt_ms > *rtt_ms {
                        let best = metrics.iter().enumerate()
                            .filter(|(i, m)| *i != current_path && m.is_active && m.avg_rtt_ms < *rtt_ms)
                            .min_by(|(_, a), (_, b)|
                                a.avg_rtt_ms.partial_cmp(&b.avg_rtt_ms).unwrap())
                            .map(|(i, _)| i);
                        (best, *weight)
                    } else {
                        (Some(current_path), *weight)
                    }
                }
            };

            if let Some(v) = vote {
                self.votes.push((v, weight));
            }
        }

        // Aggregate votes
        let mut path_scores: Vec<f64> = vec![0.0; metrics.len()];
        for (path, weight) in &self.votes {
            path_scores[*path] += weight;
        }

        // Decision based on voting strategy
        match self.config.voting_strategy {
            VotingStrategy::WeightedMajority => {
                let total_weight: f64 = self.votes.iter().map(|(_, w)| w).sum();
                let best_path = path_scores.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(current_path);

                if best_path != current_path 
                    && path_scores[best_path] / total_weight > self.config.confidence_threshold {
                    FailoverDecision::SwitchTo(best_path)
                } else {
                    FailoverDecision::StayCurrent
                }
            }
            VotingStrategy::Unanimous => {
                if self.votes.iter().all(|(p, _)| *p == self.votes[0].0) 
                    && self.votes[0].0 != current_path {
                    FailoverDecision::SwitchTo(self.votes[0].0)
                } else {
                    FailoverDecision::StayCurrent
                }
            }
            VotingStrategy::Any => {
                if let Some((path, _)) = self.votes.iter()
                    .find(|(p, _)| *p != current_path && metrics[*p].is_active) {
                    FailoverDecision::SwitchTo(*path)
                } else {
                    FailoverDecision::StayCurrent
                }
            }
        }
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("component_count".to_string(), self.config.components.len() as f64);
        params.insert("confidence_threshold".to_string(), self.config.confidence_threshold);
        params
    }

    fn reset(&mut self) {
        self.votes.clear();
    }
}

// ============================================================================
// 9. Proactive Pattern-Based Failover
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PatternType {
    Periodic,
    Trending,
    Seasonal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactiveConfig {
    pub pattern_window_s: u64,
    pub prediction_horizon_s: u64,
    pub confidence_threshold: f64,
    pub preemptive_switch_enabled: bool,
    pub pattern_types: Vec<PatternType>,
}

pub struct ProactiveFailover {
    id: String,
    config: ProactiveConfig,
    history: Vec<VecDeque<(f64, f64)>>, // (time_s, rtt_ms)
}

impl ProactiveFailover {
    pub fn new(id: &str, config: ProactiveConfig) -> Self {
        Self {
            id: id.to_string(),
            config,
            history: Vec::new(),
        }
    }

    fn detect_trend(&self, path: usize) -> Option<f64> {
        let hist = &self.history[path];
        if hist.len() < 10 {
            return None;
        }

        // Simple linear regression
        let n = hist.len() as f64;
        let sum_x: f64 = hist.iter().map(|(t, _)| t).sum();
        let sum_y: f64 = hist.iter().map(|(_, r)| r).sum();
        let sum_xy: f64 = hist.iter().map(|(t, r)| t * r).sum();
        let sum_x2: f64 = hist.iter().map(|(t, _)| t * t).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        Some(slope)
    }
}

impl FailoverAlgorithm for ProactiveFailover {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "Proactive: window={}s, horizon={}s, preemptive={}",
            self.config.pattern_window_s,
            self.config.prediction_horizon_s,
            self.config.preemptive_switch_enabled
        )
    }

    fn init(&mut self, path_count: usize) {
        self.history = vec![VecDeque::with_capacity(1000); path_count];
    }

    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        // Record history
        let current_time = self.history[current_path].back()
            .map(|(t, _)| t + 0.01)
            .unwrap_or(0.0);

        for (i, m) in metrics.iter().enumerate() {
            if m.is_active {
                self.history[i].push_back((current_time, m.avg_rtt_ms));
                
                // Limit history size
                let max_samples = (self.config.pattern_window_s * 100) as usize;
                while self.history[i].len() > max_samples {
                    self.history[i].pop_front();
                }
            }
        }

        // Detect trends
        if let Some(trend) = self.detect_trend(current_path) {
            // Predict future RTT
            let predicted_rtt = metrics[current_path].avg_rtt_ms 
                + trend * self.config.prediction_horizon_s as f64;

            // Preemptive switch if degradation predicted
            if self.config.preemptive_switch_enabled 
                && predicted_rtt > 200.0 
                && trend > 5.0 { // RTT increasing by >5ms/s
                
                // Find stable alternative
                let best = metrics.iter().enumerate()
                    .filter(|(i, m)| *i != current_path && m.is_active)
                    .filter(|(i, _)| {
                        self.detect_trend(*i).map(|t| t < 1.0).unwrap_or(true)
                    })
                    .min_by(|(_, a), (_, b)|
                        a.avg_rtt_ms.partial_cmp(&b.avg_rtt_ms).unwrap())
                    .map(|(i, _)| i);

                if let Some(best_path) = best {
                    return FailoverDecision::SwitchTo(best_path);
                }
            }
        }

        FailoverDecision::StayCurrent
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("pattern_window_s".to_string(), self.config.pattern_window_s as f64);
        params.insert("prediction_horizon_s".to_string(), self.config.prediction_horizon_s as f64);
        params
    }

    fn reset(&mut self) {
        for h in &mut self.history {
            h.clear();
        }
    }
}

// ============================================================================
// 10. Military-Grade Paranoid Failover
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParanoidConfig {
    pub heartbeat_interval_ms: u64,
    pub max_silent_period_ms: u64,
    pub parallel_probing: bool,
    pub path_diversity_required: usize,
    pub instant_failover_on_loss: bool,
    pub return_delay_ms: u64,
}

pub struct ParanoidFailover {
    id: String,
    config: ParanoidConfig,
    last_response_times: Vec<f64>,
    current_time: f64,
    last_switch_time: f64,
    active_paths: Vec<bool>,
}

impl ParanoidFailover {
    pub fn new(id: &str, config: ParanoidConfig) -> Self {
        Self {
            id: id.to_string(),
            config,
            last_response_times: Vec::new(),
            current_time: 0.0,
            last_switch_time: 0.0,
            active_paths: Vec::new(),
        }
    }
}

impl FailoverAlgorithm for ParanoidFailover {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "Paranoid MilSpec: heartbeat={}ms, max_silent={}ms, diversity={}",
            self.config.heartbeat_interval_ms,
            self.config.max_silent_period_ms,
            self.config.path_diversity_required
        )
    }

    fn init(&mut self, path_count: usize) {
        self.last_response_times = vec![0.0; path_count];
        self.active_paths = vec![true; path_count];
        self.current_time = 0.0;
        self.last_switch_time = 0.0;
    }

    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        self.current_time += 10.0; // 10ms ticks

        // Update path status
        for (i, m) in metrics.iter().enumerate() {
            if m.is_active && m.consecutive_losses == 0 {
                self.last_response_times[i] = self.current_time;
                self.active_paths[i] = true;
            } else {
                // Check for silent period
                let silent_time = self.current_time - self.last_response_times[i];
                if silent_time > self.config.max_silent_period_ms as f64 {
                    self.active_paths[i] = false;
                }
            }
        }

        // Instant failover on any packet loss
        if self.config.instant_failover_on_loss && metrics[current_path].consecutive_losses > 0 {
            // Find any working alternative
            for (i, &active) in self.active_paths.iter().enumerate() {
                if i != current_path && active && metrics[i].is_active {
                    self.last_switch_time = self.current_time;
                    return FailoverDecision::SwitchTo(i);
                }
            }
        }

        // Check current path health
        let current_silent = self.current_time - self.last_response_times[current_path];
        if current_silent > self.config.max_silent_period_ms as f64 / 2.0 
            || !metrics[current_path].is_active {
            
            // Aggressive failover
            let best = self.active_paths.iter().enumerate()
                .filter(|(i, &active)| *i != current_path && active && metrics[*i].is_active)
                .min_by(|(i, _), (j, _)|
                    metrics[*i].avg_rtt_ms.partial_cmp(&metrics[*j].avg_rtt_ms).unwrap())
                .map(|(i, _)| i);

            if let Some(best_path) = best {
                self.last_switch_time = self.current_time;
                return FailoverDecision::SwitchTo(best_path);
            }
        }

        // Multi-path mode if diversity required
        if self.config.path_diversity_required > 1 {
            let active_count = self.active_paths.iter().filter(|&&x| x).count();
            if active_count >= self.config.path_diversity_required {
                let mut mask = 0u32;
                for (i, &active) in self.active_paths.iter().enumerate() {
                    if active && metrics[i].is_active {
                        mask |= 1 << i;
                    }
                }
                return FailoverDecision::MultiPath(mask);
            }
        }

        FailoverDecision::StayCurrent
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("heartbeat_interval_ms".to_string(), self.config.heartbeat_interval_ms as f64);
        params.insert("max_silent_period_ms".to_string(), self.config.max_silent_period_ms as f64);
        params.insert("path_diversity_required".to_string(), self.config.path_diversity_required as f64);
        params.insert("return_delay_ms".to_string(), self.config.return_delay_ms as f64);
        params
    }

    fn reset(&mut self) {
        self.last_response_times.fill(0.0);
        self.active_paths.fill(true);
        self.current_time = 0.0;
        self.last_switch_time = 0.0;
    }
}

// ============================================================================
// 11. Neural Network Failover Adapter
// ============================================================================

use crate::neural::{NeuralPredictor, NetworkFeatures};

/// Adapter to use neural network models as failover algorithms
pub struct NeuralFailover {
    id: String,
    predictor: Box<dyn NeuralPredictor>,
    feature_history: NetworkFeatures,
    rtt_history: VecDeque<f64>,
    loss_history: VecDeque<f64>,
    jitter_history: VecDeque<f64>,
    bandwidth_history: VecDeque<f64>,
    failover_history: VecDeque<f64>,
    current_time_ms: f64,
    last_prediction_time: f64,
    prediction_interval_ms: f64,
    num_paths: usize,
}

impl NeuralFailover {
    pub fn new(id: &str, predictor: Box<dyn NeuralPredictor>) -> Self {
        Self {
            id: id.to_string(),
            predictor,
            feature_history: NetworkFeatures::empty(3),
            rtt_history: VecDeque::with_capacity(256),
            loss_history: VecDeque::with_capacity(256),
            jitter_history: VecDeque::with_capacity(256),
            bandwidth_history: VecDeque::with_capacity(256),
            failover_history: VecDeque::with_capacity(256),
            current_time_ms: 0.0,
            last_prediction_time: 0.0,
            prediction_interval_ms: 10.0,
            num_paths: 3,
        }
    }
    
    fn update_features(&mut self, metrics: &[PathMetrics], current_path: usize) {
        // Update RTT history
        let current_rtt = metrics[current_path].avg_rtt_ms;
        self.rtt_history.push_back(current_rtt);
        if self.rtt_history.len() > 256 {
            self.rtt_history.pop_front();
        }
        
        // Update loss history
        let current_loss = metrics[current_path].loss_rate;
        self.loss_history.push_back(current_loss);
        if self.loss_history.len() > 256 {
            self.loss_history.pop_front();
        }
        
        // Update jitter history
        let current_jitter = metrics[current_path].jitter_ms;
        self.jitter_history.push_back(current_jitter);
        if self.jitter_history.len() > 256 {
            self.jitter_history.pop_front();
        }
        
        // Update bandwidth history (estimated)
        let bandwidth_est = if current_loss > 0.5 { 0.0 } else { 100.0 / (1.0 + current_rtt / 50.0) };
        self.bandwidth_history.push_back(bandwidth_est);
        if self.bandwidth_history.len() > 256 {
            self.bandwidth_history.pop_front();
        }
        
        // Build features
        self.feature_history.rtt_history = self.rtt_history.iter().cloned().collect();
        self.feature_history.loss_history = self.loss_history.iter().cloned().collect();
        self.feature_history.jitter_history = self.jitter_history.iter().cloned().collect();
        self.feature_history.bandwidth_history = self.bandwidth_history.iter().cloned().collect();
        
        // Signal strength per path
        self.feature_history.signal_strength = metrics.iter()
            .map(|m| if m.is_active { -70.0 - m.avg_rtt_ms / 10.0 } else { -120.0 })
            .collect();
        
        // Time features
        let hour = (self.current_time_ms / 3600000.0) % 24.0;
        self.feature_history.time_of_day = hour / 24.0;
        
        let day = ((self.current_time_ms / 86400000.0) as usize) % 7;
        self.feature_history.day_of_week = [0.0; 7];
        self.feature_history.day_of_week[day] = 1.0;
        
        self.feature_history.current_path = current_path;
        self.feature_history.num_paths = metrics.len();
        
        // Path features
        self.feature_history.path_features = metrics.iter()
            .flat_map(|m| vec![
                m.avg_rtt_ms / 500.0,
                m.loss_rate,
                m.jitter_ms / 100.0,
                if m.is_active { 1.0 } else { 0.0 },
                m.consecutive_losses as f64 / 10.0,
                m.bandwidth_mbps / 1000.0,
                m.rtt_variance_ms2.sqrt() / 100.0,
                m.quality_score(),
            ])
            .collect();
        
        // Recent failovers
        self.feature_history.recent_failovers = self.failover_history.iter().cloned().collect();
        
        // Topology embedding (simplified)
        self.feature_history.topology_embedding = vec![0.0; 32];
        for i in 0..metrics.len().min(32) {
            self.feature_history.topology_embedding[i] = if metrics[i].is_active { 1.0 } else { 0.0 };
        }
    }
}

impl FailoverAlgorithm for NeuralFailover {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn description(&self) -> String {
        format!("Neural/{}: {}", self.predictor.name(), self.predictor.description())
    }
    
    fn init(&mut self, path_count: usize) {
        self.num_paths = path_count;
        self.feature_history = NetworkFeatures::empty(path_count);
        self.rtt_history.clear();
        self.loss_history.clear();
        self.jitter_history.clear();
        self.bandwidth_history.clear();
        self.failover_history.clear();
        self.current_time_ms = 0.0;
        self.last_prediction_time = 0.0;
        self.predictor.reset();
    }
    
    fn update(&mut self, metrics: &[PathMetrics], current_path: usize) -> FailoverDecision {
        self.current_time_ms += 10.0; // 10ms tick
        
        // Update feature history
        self.update_features(metrics, current_path);
        
        // Only make predictions at specified interval
        if self.current_time_ms - self.last_prediction_time < self.prediction_interval_ms {
            return FailoverDecision::StayCurrent;
        }
        self.last_prediction_time = self.current_time_ms;
        
        // Get neural network prediction
        let prediction = self.predictor.predict(&self.feature_history);
        
        // Decision logic based on prediction
        if prediction.failover_probability > 0.7 {
            // High probability of needing failover
            let recommended = prediction.recommended_path.min(metrics.len() - 1);
            
            if recommended != current_path && metrics[recommended].is_active {
                // Record failover
                self.failover_history.push_back(1.0);
                if self.failover_history.len() > 100 {
                    self.failover_history.pop_front();
                }
                return FailoverDecision::SwitchTo(recommended);
            }
            
            // Find best alternative based on path scores
            let best_alt = prediction.path_scores.iter()
                .enumerate()
                .filter(|(i, _)| *i < metrics.len() && *i != current_path && metrics[*i].is_active)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i);
            
            if let Some(alt) = best_alt {
                self.failover_history.push_back(1.0);
                if self.failover_history.len() > 100 {
                    self.failover_history.pop_front();
                }
                return FailoverDecision::SwitchTo(alt);
            }
        }
        
        // Record no failover
        self.failover_history.push_back(0.0);
        if self.failover_history.len() > 100 {
            self.failover_history.pop_front();
        }
        
        FailoverDecision::StayCurrent
    }
    
    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("num_parameters".to_string(), self.predictor.num_parameters() as f64);
        params.insert("inference_latency_us".to_string(), self.predictor.inference_latency_us() as f64);
        params.insert("prediction_interval_ms".to_string(), self.prediction_interval_ms);
        params
    }
    
    fn reset(&mut self) {
        self.rtt_history.clear();
        self.loss_history.clear();
        self.jitter_history.clear();
        self.bandwidth_history.clear();
        self.failover_history.clear();
        self.current_time_ms = 0.0;
        self.last_prediction_time = 0.0;
        self.predictor.reset();
    }
}
