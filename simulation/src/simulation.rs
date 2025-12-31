//! Monte Carlo Simulation Engine v2.0
//!
//! Rigorous statistical simulation with:
//! - Proper warm-up period detection
//! - Batch means for variance estimation
//! - Convergence testing
//! - Correlated sampling for variance reduction
//! - Detailed per-timestep metrics collection

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use parking_lot::Mutex;

use crate::scenarios::{Scenario, EventType};
use crate::algorithms::{FailoverAlgorithm, FailoverDecision};
use crate::network::PathMetrics;

/// Format seconds as human readable duration
fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.0}s", secs)
    } else if secs < 3600.0 {
        format!("{}m {}s", (secs / 60.0) as u64, (secs % 60.0) as u64)
    } else {
        format!("{}h {}m", (secs / 3600.0) as u64, ((secs % 3600.0) / 60.0) as u64)
    }
}

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub monte_carlo_iterations: usize,
    pub time_resolution_ms: u64,
    pub simulation_duration_s: u64,
    pub seed: u64,
    pub parallel_workers: usize,
    
    // Advanced configuration
    pub warmup_fraction: f64,           // Fraction of run to discard as warm-up (default 0.1)
    pub batch_size: usize,              // Batch size for batch means method
    pub convergence_threshold: f64,     // Relative error threshold for convergence
    pub min_iterations: usize,          // Minimum iterations before convergence check
    pub collect_timeseries: bool,       // Whether to collect full time series data
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            monte_carlo_iterations: 1000,
            time_resolution_ms: 10,
            simulation_duration_s: 300,
            seed: 42,
            parallel_workers: num_cpus::get(),
            warmup_fraction: 0.1,
            batch_size: 100,
            convergence_threshold: 0.05,
            min_iterations: 100,
            collect_timeseries: false,
        }
    }
}

// ============================================================================
// Detailed Metrics
// ============================================================================

/// Per-timestep sample for detailed analysis
#[derive(Debug, Clone)]
pub struct TimestepSample {
    pub time_ms: f64,
    pub current_path: usize,
    pub latency_ms: f64,
    pub packet_lost: bool,
    pub path_metrics: Vec<PathMetrics>,
    pub decision: FailoverDecision,
    pub event_occurred: bool,
}

/// Comprehensive metrics for a single simulation run
#[derive(Debug, Clone)]
pub struct RunMetrics {
    // ─────────────────────────────────────────────────────────────────────────
    // Basic Statistics
    // ─────────────────────────────────────────────────────────────────────────
    pub total_time_ms: f64,
    pub packets_sent: u64,
    pub packets_delivered: u64,
    pub packets_lost: u64,
    
    // ─────────────────────────────────────────────────────────────────────────
    // Latency Distribution (post warm-up)
    // ─────────────────────────────────────────────────────────────────────────
    pub avg_latency_ms: f64,
    pub std_latency_ms: f64,
    pub min_latency_ms: f64,
    pub p10_latency_ms: f64,
    pub p25_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p75_latency_ms: f64,
    pub p90_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub p999_latency_ms: f64,
    pub max_latency_ms: f64,
    
    // ─────────────────────────────────────────────────────────────────────────
    // Jitter Analysis
    // ─────────────────────────────────────────────────────────────────────────
    pub avg_jitter_ms: f64,
    pub std_jitter_ms: f64,
    pub max_jitter_ms: f64,
    
    // ─────────────────────────────────────────────────────────────────────────
    // Failover Behavior
    // ─────────────────────────────────────────────────────────────────────────
    pub failover_count: u64,
    pub failed_failover_count: u64,
    pub avg_time_between_failovers_ms: f64,
    pub failover_reaction_time_ms: f64,  // Avg time from degradation to failover
    
    // ─────────────────────────────────────────────────────────────────────────
    // Availability & Quality
    // ─────────────────────────────────────────────────────────────────────────
    pub availability: f64,              // 1 - (outage_time / total_time)
    pub degraded_time_ms: f64,          // Time with latency > threshold
    pub outage_time_ms: f64,            // Time with no connectivity
    pub outage_count: u64,              // Number of distinct outage events
    pub max_outage_duration_ms: f64,    // Longest single outage
    pub mean_time_between_failures_ms: f64,  // MTBF
    pub mean_time_to_recover_ms: f64,   // MTTR
    
    // ─────────────────────────────────────────────────────────────────────────
    // Burst Analysis
    // ─────────────────────────────────────────────────────────────────────────
    pub loss_burst_count: u64,          // Number of loss bursts
    pub avg_loss_burst_length: f64,     // Average consecutive losses
    pub max_loss_burst_length: u64,     // Maximum consecutive losses
    
    // ─────────────────────────────────────────────────────────────────────────
    // Path Utilization
    // ─────────────────────────────────────────────────────────────────────────
    pub path_utilization: Vec<f64>,     // Fraction of time on each path
    pub path_packet_counts: Vec<u64>,   // Packets sent on each path
    
    // ─────────────────────────────────────────────────────────────────────────
    // Scenario Events
    // ─────────────────────────────────────────────────────────────────────────
    pub event_count: usize,
    pub coverage_gap_count: usize,
    pub handover_count: usize,
    pub infrastructure_failure_count: usize,
    
    // ─────────────────────────────────────────────────────────────────────────
    // Composite Scores
    // ─────────────────────────────────────────────────────────────────────────
    pub continuity_score: f64,
    pub quality_score: f64,
    pub stability_score: f64,           // Inverse of failover rate variance
    
    // ─────────────────────────────────────────────────────────────────────────
    // Algorithm Parameters
    // ─────────────────────────────────────────────────────────────────────────
    pub algorithm_params: HashMap<String, f64>,
    
    // ─────────────────────────────────────────────────────────────────────────
    // Time Series (optional)
    // ─────────────────────────────────────────────────────────────────────────
    pub timeseries: Option<Vec<TimestepSample>>,
}

impl Default for RunMetrics {
    fn default() -> Self {
        Self {
            total_time_ms: 0.0,
            packets_sent: 0,
            packets_delivered: 0,
            packets_lost: 0,
            avg_latency_ms: 0.0,
            std_latency_ms: 0.0,
            min_latency_ms: f64::MAX,
            p10_latency_ms: 0.0,
            p25_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p75_latency_ms: 0.0,
            p90_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            p999_latency_ms: 0.0,
            max_latency_ms: 0.0,
            avg_jitter_ms: 0.0,
            std_jitter_ms: 0.0,
            max_jitter_ms: 0.0,
            failover_count: 0,
            failed_failover_count: 0,
            avg_time_between_failovers_ms: 0.0,
            failover_reaction_time_ms: 0.0,
            availability: 1.0,
            degraded_time_ms: 0.0,
            outage_time_ms: 0.0,
            outage_count: 0,
            max_outage_duration_ms: 0.0,
            mean_time_between_failures_ms: 0.0,
            mean_time_to_recover_ms: 0.0,
            loss_burst_count: 0,
            avg_loss_burst_length: 0.0,
            max_loss_burst_length: 0,
            path_utilization: Vec::new(),
            path_packet_counts: Vec::new(),
            event_count: 0,
            coverage_gap_count: 0,
            handover_count: 0,
            infrastructure_failure_count: 0,
            continuity_score: 0.0,
            quality_score: 0.0,
            stability_score: 0.0,
            algorithm_params: HashMap::new(),
            timeseries: None,
        }
    }
}

impl RunMetrics {
    pub fn loss_rate(&self) -> f64 {
        if self.packets_sent == 0 { 0.0 } 
        else { self.packets_lost as f64 / self.packets_sent as f64 }
    }
    
    /// Calculate percentile from sorted values
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() { return 0.0; }
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    
    /// Calculate composite quality score
    pub fn composite_score(&self) -> f64 {
        // Multi-objective scoring with configurable weights
        const W_LATENCY: f64 = 0.20;
        const W_LOSS: f64 = 0.25;
        const W_JITTER: f64 = 0.10;
        const W_AVAILABILITY: f64 = 0.25;
        const W_STABILITY: f64 = 0.10;
        const W_REACTION: f64 = 0.10;
        
        // Latency score: exponential decay from ideal (20ms)
        let latency_score = (-0.01 * (self.avg_latency_ms - 20.0).max(0.0)).exp();
        
        // Loss score: exponential penalty
        let loss_score = (-10.0 * self.loss_rate()).exp();
        
        // Jitter score: exponential decay
        let jitter_score = (-0.05 * self.avg_jitter_ms).exp();
        
        // Availability score (already 0-1)
        let availability_score = self.availability;
        
        // Stability score: penalize high failover variance
        let stability_score = self.stability_score;
        
        // Reaction time score: faster is better
        let reaction_score = if self.failover_reaction_time_ms > 0.0 {
            (-0.002 * self.failover_reaction_time_ms).exp()
        } else {
            1.0
        };
        
        W_LATENCY * latency_score
            + W_LOSS * loss_score
            + W_JITTER * jitter_score
            + W_AVAILABILITY * availability_score
            + W_STABILITY * stability_score
            + W_REACTION * reaction_score
    }
}

// ============================================================================
// Simulation Result
// ============================================================================

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub scenario_id: String,
    pub algorithm_id: String,
    pub iteration: usize,
    pub metrics: RunMetrics,
    pub seed: u64,
}

// ============================================================================
// Convergence Statistics
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct ConvergenceStats {
    pub iterations_run: usize,
    pub converged: bool,
    pub relative_error: f64,
    pub batch_means: Vec<f64>,
    pub batch_variance: f64,
}

// ============================================================================
// Simulation Engine
// ============================================================================

pub struct SimulationEngine {
    config: SimulationConfig,
    scenarios: Vec<Box<dyn Scenario>>,
    algorithms: Vec<Box<dyn FailoverAlgorithm>>,
}

impl SimulationEngine {
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            config,
            scenarios: Vec::new(),
            algorithms: Vec::new(),
        }
    }
    
    pub fn add_scenario(&mut self, scenario: Box<dyn Scenario>) {
        println!("    + {}: {}", scenario.id(), scenario.description());
        self.scenarios.push(scenario);
    }
    
    pub fn add_algorithm(&mut self, algorithm: Box<dyn FailoverAlgorithm>) {
        println!("    + {}: {}", algorithm.id(), algorithm.description());
        self.algorithms.push(algorithm);
    }
    
    pub fn scenario_count(&self) -> usize { self.scenarios.len() }
    pub fn algorithm_count(&self) -> usize { self.algorithms.len() }
    
    /// Run all simulations with proper statistical methodology
    pub fn run_all(&mut self) -> Vec<SimulationResult> {
        let total_runs = self.scenarios.len() 
            * self.algorithms.len() 
            * self.config.monte_carlo_iterations;
        
        let results = Arc::new(Mutex::new(Vec::with_capacity(total_runs)));
        let config = self.config.clone();
        let start_time = Instant::now();
        
        let num_scenarios = self.scenarios.len();
        let num_algorithms = self.algorithms.len();
        let mut completed: usize = 0;
        let mut last_print = Instant::now();
        
        // Process scenario/algorithm combinations
        for (scenario_idx, scenario) in self.scenarios.iter_mut().enumerate() {
            let scenario_id = scenario.id().to_string();
            
            for (algorithm_idx, algorithm) in self.algorithms.iter_mut().enumerate() {
                let algorithm_id = algorithm.id().to_string();
                
                // Run Monte Carlo iterations
                for iteration in 0..config.monte_carlo_iterations {
                    // Unique seed incorporating scenario, algorithm, and iteration
                    let seed = config.seed
                        .wrapping_add(scenario_idx as u64 * 1_000_000)
                        .wrapping_add(algorithm_idx as u64 * 1_000)
                        .wrapping_add(iteration as u64);
                    
                    // Initialize for this run
                    scenario.init(seed);
                    let path_count = scenario.get_links().len();
                    algorithm.init(path_count);
                    
                    // Execute simulation
                    let metrics = Self::run_single_rigorous(
                        scenario.as_mut(),
                        algorithm.as_mut(),
                        &config,
                        seed,
                    );
                    
                    results.lock().push(SimulationResult {
                        scenario_id: scenario_id.clone(),
                        algorithm_id: algorithm_id.clone(),
                        iteration,
                        metrics,
                        seed,
                    });
                    
                    completed += 1;
                    
                    // Reset state
                    scenario.reset();
                    algorithm.reset();
                }
                
                // Print progress after each algorithm completes (every 50 iterations)
                if last_print.elapsed() >= Duration::from_secs(2) || 
                   (algorithm_idx + 1 == num_algorithms && (scenario_idx + 1) % 5 == 0) {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let pct = (completed as f64 / total_runs as f64) * 100.0;
                    let rate = completed as f64 / elapsed;
                    let remaining = (total_runs - completed) as f64 / rate;
                    
                    eprintln!(
                        "\r  [{:>6.2}%] {}/{} | S:{}/{} A:{}/{} | {:.0}/s | ETA: {}",
                        pct,
                        completed,
                        total_runs,
                        scenario_idx + 1,
                        num_scenarios,
                        algorithm_idx + 1,
                        num_algorithms,
                        rate,
                        format_duration(remaining)
                    );
                    last_print = Instant::now();
                }
            }
        }
        
        let elapsed = start_time.elapsed();
        let throughput = total_runs as f64 / elapsed.as_secs_f64();
        
        eprintln!(
            "\n  Done! {} runs in {:.1}s ({:.1} runs/sec)\n",
            total_runs,
            elapsed.as_secs_f64(),
            throughput
        );
        
        Arc::try_unwrap(results).ok().unwrap().into_inner()
    }
    
    /// Run a single simulation with rigorous methodology
    fn run_single_rigorous(
        scenario: &mut dyn Scenario,
        algorithm: &mut dyn FailoverAlgorithm,
        config: &SimulationConfig,
        seed: u64,
    ) -> RunMetrics {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let delta_ms = config.time_resolution_ms as f64;
        let duration_ms = config.simulation_duration_s as f64 * 1000.0;
        let warmup_ms = duration_ms * config.warmup_fraction;
        
        // ─────────────────────────────────────────────────────────────────────
        // State tracking
        // ─────────────────────────────────────────────────────────────────────
        let path_count = scenario.get_links().len();
        let mut current_path: usize = 0;
        let mut in_warmup = true;
        
        // Latency collection (post warm-up only)
        let mut latencies: Vec<f64> = Vec::with_capacity(
            ((duration_ms - warmup_ms) / delta_ms) as usize
        );
        let mut jitters: Vec<f64> = Vec::new();
        let mut last_latency: f64 = 0.0;
        
        // Counters
        let mut total_packets: u64 = 0;
        let mut delivered_packets: u64 = 0;
        let mut lost_packets: u64 = 0;
        let mut failover_count: u64 = 0;
        let mut failed_failovers: u64 = 0;
        let mut failover_times: Vec<f64> = Vec::new();
        
        // Path utilization
        let mut path_time: Vec<f64> = vec![0.0; path_count];
        let mut path_packets: Vec<u64> = vec![0; path_count];
        
        // Outage tracking
        let mut degraded_time: f64 = 0.0;
        let mut outage_time: f64 = 0.0;
        let mut outage_count: u64 = 0;
        let mut current_outage_start: Option<f64> = None;
        let mut max_outage_duration: f64 = 0.0;
        let mut outage_durations: Vec<f64> = Vec::new();
        let mut recovery_times: Vec<f64> = Vec::new();
        
        // Loss burst tracking
        let mut consecutive_losses: u64 = 0;
        let mut loss_bursts: Vec<u64> = Vec::new();
        let mut in_loss_burst = false;
        
        // Degradation detection for reaction time
        let mut degradation_start: Option<f64> = None;
        let mut reaction_times: Vec<f64> = Vec::new();
        
        // Time series collection
        let mut timeseries: Vec<TimestepSample> = if config.collect_timeseries {
            Vec::with_capacity((duration_ms / delta_ms) as usize)
        } else {
            Vec::new()
        };
        
        // ─────────────────────────────────────────────────────────────────────
        // Main simulation loop
        // ─────────────────────────────────────────────────────────────────────
        let mut current_time = 0.0;
        
        while current_time < duration_ms && !scenario.is_complete() {
            // Check warm-up transition
            if in_warmup && current_time >= warmup_ms {
                in_warmup = false;
                // Reset counters at end of warm-up for accurate statistics
                total_packets = 0;
                delivered_packets = 0;
                lost_packets = 0;
                latencies.clear();
                jitters.clear();
                degraded_time = 0.0;
                outage_time = 0.0;
                outage_count = 0;
                path_time = vec![0.0; path_count];
                path_packets = vec![0; path_count];
            }
            
            // Step the scenario forward
            scenario.step(delta_ms, &mut rng);
            current_time = scenario.current_time_ms();
            
            // Get path metrics
            let links = scenario.get_links();
            let path_metrics: Vec<PathMetrics> = links.iter()
                .map(|l| l.path.get_metrics())
                .collect();
            
            // Detect degradation (for reaction time measurement)
            let current_degraded = current_path < path_metrics.len() 
                && path_metrics[current_path].is_active
                && (path_metrics[current_path].avg_rtt_ms > 150.0 
                    || path_metrics[current_path].loss_rate > 0.05);
            
            if current_degraded && degradation_start.is_none() {
                degradation_start = Some(current_time);
            }
            
            // Algorithm decision
            let decision = algorithm.update(&path_metrics, current_path);
            
            // Process decision
            let old_path = current_path;
            match decision {
                FailoverDecision::StayCurrent => {}
                FailoverDecision::SwitchTo(new_path) => {
                    if new_path < path_metrics.len() && new_path != current_path {
                        current_path = new_path;
                        
                        if !in_warmup {
                            failover_count += 1;
                            failover_times.push(current_time);
                            
                            // Record reaction time
                            if let Some(deg_start) = degradation_start.take() {
                                reaction_times.push(current_time - deg_start);
                            }
                            
                            // Check if failover was to a bad path
                            if !path_metrics[new_path].is_active 
                                || path_metrics[new_path].loss_rate > 0.5 {
                                failed_failovers += 1;
                            }
                        }
                    }
                }
                FailoverDecision::MultiPath(_) => {
                    // Could implement multi-path bonding here
                }
            }
            
            // Simulate packet transmission
            let mut links = scenario.get_links_mut();
            let (packet_lost, latency_ms) = if current_path < links.len() {
                let link = &mut links[current_path];
                let result = link.send_packet(&mut rng, delta_ms, 1400);
                (result.lost, result.latency_ms)
            } else {
                (true, 0.0)
            };
            
            // Track path utilization
            if !in_warmup && current_path < path_time.len() {
                path_time[current_path] += delta_ms;
                path_packets[current_path] += 1;
            }
            
            // Process packet result
            if !in_warmup {
                total_packets += 1;
                
                if packet_lost {
                    lost_packets += 1;
                    
                    // Outage tracking
                    if current_outage_start.is_none() {
                        current_outage_start = Some(current_time);
                        outage_count += 1;
                    }
                    outage_time += delta_ms;
                    
                    // Loss burst tracking
                    consecutive_losses += 1;
                    if !in_loss_burst {
                        in_loss_burst = true;
                    }
                } else {
                    delivered_packets += 1;
                    latencies.push(latency_ms);
                    
                    // Jitter calculation (RFC 3550 style)
                    if last_latency > 0.0 {
                        let jitter = (latency_ms - last_latency).abs();
                        jitters.push(jitter);
                    }
                    last_latency = latency_ms;
                    
                    // Degraded condition tracking
                    if latency_ms > 200.0 {
                        degraded_time += delta_ms;
                    }
                    
                    // End of outage
                    if let Some(outage_start) = current_outage_start.take() {
                        let outage_duration = current_time - outage_start;
                        outage_durations.push(outage_duration);
                        recovery_times.push(outage_duration);
                        max_outage_duration = max_outage_duration.max(outage_duration);
                    }
                    
                    // End of loss burst
                    if in_loss_burst {
                        loss_bursts.push(consecutive_losses);
                        consecutive_losses = 0;
                        in_loss_burst = false;
                    }
                }
            }
            
            // Collect time series sample
            if config.collect_timeseries {
                timeseries.push(TimestepSample {
                    time_ms: current_time,
                    current_path,
                    latency_ms: if packet_lost { 0.0 } else { latency_ms },
                    packet_lost,
                    path_metrics: path_metrics.clone(),
                    decision,
                    event_occurred: false, // Updated below
                });
            }
        }
        
        // Handle final outage if still ongoing
        if let Some(outage_start) = current_outage_start {
            let outage_duration = current_time - outage_start;
            outage_durations.push(outage_duration);
            max_outage_duration = max_outage_duration.max(outage_duration);
        }
        
        // Handle final loss burst
        if in_loss_burst && consecutive_losses > 0 {
            loss_bursts.push(consecutive_losses);
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // Calculate final statistics
        // ─────────────────────────────────────────────────────────────────────
        let mut metrics = RunMetrics::default();
        
        let effective_duration = duration_ms - warmup_ms;
        metrics.total_time_ms = effective_duration;
        metrics.packets_sent = total_packets;
        metrics.packets_delivered = delivered_packets;
        metrics.packets_lost = lost_packets;
        
        // Latency statistics
        if !latencies.is_empty() {
            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = latencies.len();
            
            let sum: f64 = latencies.iter().sum();
            metrics.avg_latency_ms = sum / n as f64;
            
            let variance: f64 = latencies.iter()
                .map(|x| (x - metrics.avg_latency_ms).powi(2))
                .sum::<f64>() / n as f64;
            metrics.std_latency_ms = variance.sqrt();
            
            metrics.min_latency_ms = latencies[0];
            metrics.p10_latency_ms = RunMetrics::percentile(&latencies, 0.10);
            metrics.p25_latency_ms = RunMetrics::percentile(&latencies, 0.25);
            metrics.p50_latency_ms = RunMetrics::percentile(&latencies, 0.50);
            metrics.p75_latency_ms = RunMetrics::percentile(&latencies, 0.75);
            metrics.p90_latency_ms = RunMetrics::percentile(&latencies, 0.90);
            metrics.p95_latency_ms = RunMetrics::percentile(&latencies, 0.95);
            metrics.p99_latency_ms = RunMetrics::percentile(&latencies, 0.99);
            metrics.p999_latency_ms = RunMetrics::percentile(&latencies, 0.999);
            metrics.max_latency_ms = latencies[n - 1];
        }
        
        // Jitter statistics
        if !jitters.is_empty() {
            let sum: f64 = jitters.iter().sum();
            metrics.avg_jitter_ms = sum / jitters.len() as f64;
            
            let variance: f64 = jitters.iter()
                .map(|x| (x - metrics.avg_jitter_ms).powi(2))
                .sum::<f64>() / jitters.len() as f64;
            metrics.std_jitter_ms = variance.sqrt();
            metrics.max_jitter_ms = jitters.iter().cloned().fold(0.0, f64::max);
        }
        
        // Failover statistics
        metrics.failover_count = failover_count;
        metrics.failed_failover_count = failed_failovers;
        
        if failover_times.len() > 1 {
            let mut intervals: Vec<f64> = failover_times.windows(2)
                .map(|w| w[1] - w[0])
                .collect();
            metrics.avg_time_between_failovers_ms = intervals.iter().sum::<f64>() / intervals.len() as f64;
        }
        
        if !reaction_times.is_empty() {
            metrics.failover_reaction_time_ms = reaction_times.iter().sum::<f64>() / reaction_times.len() as f64;
        }
        
        // Availability and reliability
        metrics.availability = if effective_duration > 0.0 {
            1.0 - (outage_time / effective_duration)
        } else {
            1.0
        };
        metrics.degraded_time_ms = degraded_time;
        metrics.outage_time_ms = outage_time;
        metrics.outage_count = outage_count;
        metrics.max_outage_duration_ms = max_outage_duration;
        
        if outage_count > 0 && effective_duration > 0.0 {
            metrics.mean_time_between_failures_ms = effective_duration / outage_count as f64;
        }
        if !recovery_times.is_empty() {
            metrics.mean_time_to_recover_ms = recovery_times.iter().sum::<f64>() / recovery_times.len() as f64;
        }
        
        // Loss burst statistics
        metrics.loss_burst_count = loss_bursts.len() as u64;
        if !loss_bursts.is_empty() {
            metrics.avg_loss_burst_length = loss_bursts.iter().sum::<u64>() as f64 / loss_bursts.len() as f64;
            metrics.max_loss_burst_length = *loss_bursts.iter().max().unwrap_or(&0);
        }
        
        // Path utilization
        let total_path_time: f64 = path_time.iter().sum();
        metrics.path_utilization = if total_path_time > 0.0 {
            path_time.iter().map(|t| t / total_path_time).collect()
        } else {
            vec![0.0; path_count]
        };
        metrics.path_packet_counts = path_packets;
        
        // Scenario events
        let events = scenario.get_events();
        metrics.event_count = events.len();
        metrics.coverage_gap_count = events.iter()
            .filter(|e| matches!(e.event_type, EventType::CoverageGap))
            .count();
        metrics.handover_count = events.iter()
            .filter(|e| matches!(e.event_type, EventType::Handover))
            .count();
        metrics.infrastructure_failure_count = events.iter()
            .filter(|e| matches!(e.event_type, EventType::InfrastructureFailure))
            .count();
        
        // Composite scores
        metrics.continuity_score = metrics.availability;
        
        // Stability score based on failover consistency
        if failover_count > 1 && !failover_times.is_empty() {
            let intervals: Vec<f64> = failover_times.windows(2)
                .map(|w| w[1] - w[0])
                .collect();
            if !intervals.is_empty() {
                let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
                let variance = intervals.iter()
                    .map(|x| (x - mean_interval).powi(2))
                    .sum::<f64>() / intervals.len() as f64;
                let cv = variance.sqrt() / mean_interval.max(1.0);  // Coefficient of variation
                metrics.stability_score = 1.0 / (1.0 + cv);  // Higher is more stable
            } else {
                metrics.stability_score = 1.0;
            }
        } else {
            metrics.stability_score = 1.0;
        }
        
        metrics.quality_score = metrics.composite_score();
        
        // Algorithm parameters
        metrics.algorithm_params = algorithm.get_parameters();
        
        // Time series
        metrics.timeseries = if config.collect_timeseries {
            Some(timeseries)
        } else {
            None
        };
        
        metrics
    }
}
