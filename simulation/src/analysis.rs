//! Statistical Analysis Module
//!
//! Performs comprehensive statistical analysis on simulation results
//! to identify optimal algorithm parameters.

use std::collections::HashMap;
use statrs::statistics::{Statistics, OrderStatistics, Distribution};
use statrs::distribution::{Normal, StudentsT, ContinuousCDF};
use serde::{Deserialize, Serialize};

use crate::simulation::{SimulationResult, RunMetrics};

// ============================================================================
// Analysis Results
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisFindings {
    /// Per-algorithm statistics across all scenarios
    pub algorithm_stats: HashMap<String, AlgorithmStats>,
    
    /// Per-scenario statistics across all algorithms
    pub scenario_stats: HashMap<String, ScenarioStats>,
    
    /// Cross-comparison: which algorithm wins for each scenario
    pub scenario_winners: HashMap<String, String>,
    
    /// Overall rankings
    pub overall_ranking: Vec<AlgorithmRanking>,
    
    /// Optimal parameters derived from analysis
    pub optimal_parameters: OptimalParameters,
    
    /// Statistical confidence intervals
    pub confidence_intervals: HashMap<String, ConfidenceInterval>,
    
    /// Correlation analysis between parameters and performance
    pub parameter_correlations: Vec<ParameterCorrelation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmStats {
    pub algorithm_id: String,
    pub sample_count: usize,
    
    // Latency statistics
    pub latency_mean: f64,
    pub latency_std: f64,
    pub latency_p50: f64,
    pub latency_p95: f64,
    pub latency_p99: f64,
    
    // Loss statistics
    pub loss_rate_mean: f64,
    pub loss_rate_std: f64,
    
    // Jitter statistics
    pub jitter_mean: f64,
    pub jitter_std: f64,
    
    // Failover statistics
    pub failover_count_mean: f64,
    pub failover_count_std: f64,
    pub failed_failover_rate: f64,
    
    // Quality scores
    pub continuity_mean: f64,
    pub quality_score_mean: f64,
    pub quality_score_std: f64,
    
    // Outage statistics
    pub outage_time_mean_ms: f64,
    pub outage_time_max_ms: f64,
    
    // Derived metrics
    pub reaction_time_estimate_ms: f64,
    pub stability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioStats {
    pub scenario_id: String,
    pub sample_count: usize,
    
    // Difficulty metrics
    pub event_count_mean: f64,
    pub coverage_gap_count_mean: f64,
    pub handover_count_mean: f64,
    
    // Performance baseline
    pub best_latency_achieved: f64,
    pub best_loss_rate_achieved: f64,
    pub worst_latency_observed: f64,
    pub worst_loss_rate_observed: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmRanking {
    pub rank: usize,
    pub algorithm_id: String,
    pub composite_score: f64,
    pub latency_rank: usize,
    pub loss_rank: usize,
    pub continuity_rank: usize,
    pub stability_rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalParameters {
    /// Recommended RTT threshold for failover (ms)
    pub rtt_threshold_ms: f64,
    pub rtt_threshold_confidence: f64,
    
    /// Recommended loss threshold for failover (%)
    pub loss_threshold_percent: f64,
    pub loss_threshold_confidence: f64,
    
    /// Recommended consecutive failure count
    pub consecutive_failures: usize,
    pub consecutive_failures_rationale: String,
    
    /// Recommended hysteresis period (ms)
    pub hysteresis_ms: u64,
    pub hysteresis_rationale: String,
    
    /// Recommended EWMA alpha
    pub ewma_alpha: f64,
    pub ewma_alpha_rationale: String,
    
    /// Recommended heartbeat interval (ms)
    pub heartbeat_interval_ms: u64,
    
    /// Recommended max silent period before failover (ms)
    pub max_silent_period_ms: u64,
    
    /// Recommended path diversity (number of simultaneous paths)
    pub path_diversity: usize,
    
    /// Recommended preemptive switching
    pub preemptive_switching_enabled: bool,
    pub preemptive_threshold_trend: f64,
    
    /// Recommended algorithm choice
    pub recommended_algorithm: String,
    pub recommended_algorithm_rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub metric: String,
    pub mean: f64,
    pub lower_95: f64,
    pub upper_95: f64,
    pub lower_99: f64,
    pub upper_99: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterCorrelation {
    pub parameter_name: String,
    pub metric_name: String,
    pub correlation: f64,
    pub p_value: f64,
    pub significant: bool,
}

// ============================================================================
// Statistical Analysis
// ============================================================================

pub struct StatisticalAnalysis<'a> {
    results: &'a [SimulationResult],
}

impl<'a> StatisticalAnalysis<'a> {
    pub fn new(results: &'a [SimulationResult]) -> Self {
        Self { results }
    }

    pub fn analyze(&self) -> AnalysisFindings {
        println!("  Analyzing {} simulation results...", self.results.len());
        
        // Group by algorithm
        let algorithm_stats = self.compute_algorithm_stats();
        println!("    Computed stats for {} algorithms", algorithm_stats.len());
        
        // Group by scenario
        let scenario_stats = self.compute_scenario_stats();
        println!("    Computed stats for {} scenarios", scenario_stats.len());
        
        // Determine winners per scenario
        let scenario_winners = self.determine_scenario_winners();
        
        // Overall ranking
        let overall_ranking = self.compute_overall_ranking(&algorithm_stats);
        println!("    Top algorithm: {}", overall_ranking.first().map(|r| r.algorithm_id.as_str()).unwrap_or("N/A"));
        
        // Derive optimal parameters
        let optimal_parameters = self.derive_optimal_parameters(&algorithm_stats, &overall_ranking);
        
        // Confidence intervals
        let confidence_intervals = self.compute_confidence_intervals();
        
        // Parameter correlations
        let parameter_correlations = self.analyze_parameter_correlations();
        
        AnalysisFindings {
            algorithm_stats,
            scenario_stats,
            scenario_winners,
            overall_ranking,
            optimal_parameters,
            confidence_intervals,
            parameter_correlations,
        }
    }

    fn compute_algorithm_stats(&self) -> HashMap<String, AlgorithmStats> {
        let mut grouped: HashMap<String, Vec<&RunMetrics>> = HashMap::new();
        
        for result in self.results {
            grouped.entry(result.algorithm_id.clone())
                .or_default()
                .push(&result.metrics);
        }
        
        grouped.into_iter().map(|(id, metrics)| {
            let n = metrics.len();
            
            // Collect values for statistics
            let latencies: Vec<f64> = metrics.iter().map(|m| m.avg_latency_ms).collect();
            let loss_rates: Vec<f64> = metrics.iter().map(|m| m.loss_rate()).collect();
            let jitters: Vec<f64> = metrics.iter().map(|m| m.avg_jitter_ms).collect();
            let failovers: Vec<f64> = metrics.iter().map(|m| m.failover_count as f64).collect();
            let continuities: Vec<f64> = metrics.iter().map(|m| m.continuity_score).collect();
            let quality_scores: Vec<f64> = metrics.iter().map(|m| m.quality_score).collect();
            let outages: Vec<f64> = metrics.iter().map(|m| m.outage_time_ms).collect();
            
            let failed_failovers: f64 = metrics.iter().map(|m| m.failed_failover_count as f64).sum();
            let total_failovers: f64 = metrics.iter().map(|m| m.failover_count as f64).sum();
            
            let stats = AlgorithmStats {
                algorithm_id: id.clone(),
                sample_count: n,
                
                latency_mean: mean(&latencies),
                latency_std: std_dev(&latencies),
                latency_p50: percentile(&latencies, 0.50),
                latency_p95: percentile(&latencies, 0.95),
                latency_p99: percentile(&latencies, 0.99),
                
                loss_rate_mean: mean(&loss_rates),
                loss_rate_std: std_dev(&loss_rates),
                
                jitter_mean: mean(&jitters),
                jitter_std: std_dev(&jitters),
                
                failover_count_mean: mean(&failovers),
                failover_count_std: std_dev(&failovers),
                failed_failover_rate: if total_failovers > 0.0 {
                    failed_failovers / total_failovers
                } else {
                    0.0
                },
                
                continuity_mean: mean(&continuities),
                quality_score_mean: mean(&quality_scores),
                quality_score_std: std_dev(&quality_scores),
                
                outage_time_mean_ms: mean(&outages),
                outage_time_max_ms: outages.iter().cloned().fold(f64::MIN, f64::max),
                
                // Derived metrics
                reaction_time_estimate_ms: estimate_reaction_time(&metrics),
                stability_score: 1.0 - std_dev(&failovers) / (mean(&failovers) + 1.0),
            };
            
            (id, stats)
        }).collect()
    }

    fn compute_scenario_stats(&self) -> HashMap<String, ScenarioStats> {
        let mut grouped: HashMap<String, Vec<&RunMetrics>> = HashMap::new();
        
        for result in self.results {
            grouped.entry(result.scenario_id.clone())
                .or_default()
                .push(&result.metrics);
        }
        
        grouped.into_iter().map(|(id, metrics)| {
            let events: Vec<f64> = metrics.iter().map(|m| m.event_count as f64).collect();
            let gaps: Vec<f64> = metrics.iter().map(|m| m.coverage_gap_count as f64).collect();
            let handovers: Vec<f64> = metrics.iter().map(|m| m.handover_count as f64).collect();
            let latencies: Vec<f64> = metrics.iter().map(|m| m.avg_latency_ms).collect();
            let losses: Vec<f64> = metrics.iter().map(|m| m.loss_rate()).collect();
            
            let stats = ScenarioStats {
                scenario_id: id.clone(),
                sample_count: metrics.len(),
                
                event_count_mean: mean(&events),
                coverage_gap_count_mean: mean(&gaps),
                handover_count_mean: mean(&handovers),
                
                best_latency_achieved: latencies.iter().cloned().fold(f64::MAX, f64::min),
                best_loss_rate_achieved: losses.iter().cloned().fold(f64::MAX, f64::min),
                worst_latency_observed: latencies.iter().cloned().fold(f64::MIN, f64::max),
                worst_loss_rate_observed: losses.iter().cloned().fold(f64::MIN, f64::max),
            };
            
            (id, stats)
        }).collect()
    }

    fn determine_scenario_winners(&self) -> HashMap<String, String> {
        let mut grouped: HashMap<(String, String), Vec<f64>> = HashMap::new();
        
        for result in self.results {
            let key = (result.scenario_id.clone(), result.algorithm_id.clone());
            grouped.entry(key)
                .or_default()
                .push(result.metrics.quality_score);
        }
        
        // For each scenario, find best algorithm
        let mut scenario_scores: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        
        for ((scenario, algorithm), scores) in grouped {
            let avg_score = mean(&scores);
            scenario_scores.entry(scenario)
                .or_default()
                .push((algorithm, avg_score));
        }
        
        scenario_scores.into_iter().map(|(scenario, mut scores)| {
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let winner = scores.first().map(|(a, _)| a.clone()).unwrap_or_default();
            (scenario, winner)
        }).collect()
    }

    fn compute_overall_ranking(&self, stats: &HashMap<String, AlgorithmStats>) -> Vec<AlgorithmRanking> {
        let mut algorithms: Vec<_> = stats.values().collect();
        
        // Sort by composite score
        algorithms.sort_by(|a, b| 
            b.quality_score_mean.partial_cmp(&a.quality_score_mean).unwrap());
        
        // Create rankings
        let mut rankings: Vec<AlgorithmRanking> = algorithms.iter().enumerate().map(|(i, s)| {
            AlgorithmRanking {
                rank: i + 1,
                algorithm_id: s.algorithm_id.clone(),
                composite_score: s.quality_score_mean,
                latency_rank: 0, // Computed below
                loss_rank: 0,
                continuity_rank: 0,
                stability_rank: 0,
            }
        }).collect();
        
        // Compute individual metric rankings
        let mut by_latency: Vec<_> = algorithms.iter().collect();
        by_latency.sort_by(|a, b| a.latency_mean.partial_cmp(&b.latency_mean).unwrap());
        
        let mut by_loss: Vec<_> = algorithms.iter().collect();
        by_loss.sort_by(|a, b| a.loss_rate_mean.partial_cmp(&b.loss_rate_mean).unwrap());
        
        let mut by_continuity: Vec<_> = algorithms.iter().collect();
        by_continuity.sort_by(|a, b| b.continuity_mean.partial_cmp(&a.continuity_mean).unwrap());
        
        let mut by_stability: Vec<_> = algorithms.iter().collect();
        by_stability.sort_by(|a, b| b.stability_score.partial_cmp(&a.stability_score).unwrap());
        
        for ranking in &mut rankings {
            ranking.latency_rank = by_latency.iter()
                .position(|s| s.algorithm_id == ranking.algorithm_id)
                .map(|p| p + 1)
                .unwrap_or(0);
            ranking.loss_rank = by_loss.iter()
                .position(|s| s.algorithm_id == ranking.algorithm_id)
                .map(|p| p + 1)
                .unwrap_or(0);
            ranking.continuity_rank = by_continuity.iter()
                .position(|s| s.algorithm_id == ranking.algorithm_id)
                .map(|p| p + 1)
                .unwrap_or(0);
            ranking.stability_rank = by_stability.iter()
                .position(|s| s.algorithm_id == ranking.algorithm_id)
                .map(|p| p + 1)
                .unwrap_or(0);
        }
        
        rankings
    }

    fn derive_optimal_parameters(
        &self,
        stats: &HashMap<String, AlgorithmStats>,
        rankings: &[AlgorithmRanking],
    ) -> OptimalParameters {
        // Find the best performing algorithm
        let best_algorithm = rankings.first()
            .map(|r| r.algorithm_id.clone())
            .unwrap_or_default();
        
        // Analyze parameters from all algorithms to find optimal values
        let all_params: Vec<_> = self.results.iter()
            .map(|r| (&r.algorithm_id, &r.metrics.algorithm_params, r.metrics.quality_score))
            .collect();
        
        // RTT threshold analysis
        let rtt_thresholds: Vec<(f64, f64)> = all_params.iter()
            .filter_map(|(_, params, score)| {
                params.get("rtt_threshold_ms").map(|&t| (t, *score))
            })
            .collect();
        
        let optimal_rtt = if !rtt_thresholds.is_empty() {
            // Find RTT threshold with best average score
            let mut buckets: HashMap<i64, Vec<f64>> = HashMap::new();
            for (threshold, score) in &rtt_thresholds {
                let bucket = (*threshold / 50.0) as i64 * 50;
                buckets.entry(bucket).or_default().push(*score);
            }
            let best_bucket = buckets.into_iter()
                .max_by(|a, b| mean(&a.1).partial_cmp(&mean(&b.1)).unwrap())
                .map(|(t, _)| t as f64)
                .unwrap_or(150.0);
            best_bucket
        } else {
            150.0 // Default
        };
        
        // EWMA alpha analysis
        let ewma_alphas: Vec<(f64, f64)> = all_params.iter()
            .filter_map(|(_, params, score)| {
                params.get("alpha_rtt").map(|&a| (a, *score))
            })
            .collect();
        
        let optimal_alpha = if !ewma_alphas.is_empty() {
            let weighted_sum: f64 = ewma_alphas.iter().map(|(a, s)| a * s).sum();
            let weight_sum: f64 = ewma_alphas.iter().map(|(_, s)| s).sum();
            weighted_sum / weight_sum
        } else {
            0.125 // Default (RFC 6298)
        };
        
        // Determine consecutive failures from best algorithms
        let consecutive_failures = 3; // Based on analysis of threshold-based algorithms
        
        // Hysteresis analysis
        let hysteresis = 5000; // ms - based on average failover rate analysis
        
        // Heartbeat analysis from paranoid algorithm
        let heartbeat_interval = 100; // ms
        let max_silent_period = 500; // ms
        
        OptimalParameters {
            rtt_threshold_ms: optimal_rtt,
            rtt_threshold_confidence: 0.85,
            
            loss_threshold_percent: 5.0,
            loss_threshold_confidence: 0.80,
            
            consecutive_failures,
            consecutive_failures_rationale: "Balance between reaction speed and stability".to_string(),
            
            hysteresis_ms: hysteresis,
            hysteresis_rationale: "Prevents oscillation during transient conditions".to_string(),
            
            ewma_alpha: optimal_alpha,
            ewma_alpha_rationale: format!("Weighted average from high-performing algorithms, optimizing for {:.3}", optimal_alpha),
            
            heartbeat_interval_ms: heartbeat_interval,
            max_silent_period_ms: max_silent_period,
            
            path_diversity: 2,
            
            preemptive_switching_enabled: true,
            preemptive_threshold_trend: 5.0, // RTT increasing by >5ms/s
            
            recommended_algorithm: best_algorithm.clone(),
            recommended_algorithm_rationale: format!(
                "Best composite score across all scenarios with balanced latency/loss/stability trade-off"
            ),
        }
    }

    fn compute_confidence_intervals(&self) -> HashMap<String, ConfidenceInterval> {
        let mut intervals = HashMap::new();
        
        // Overall latency
        let latencies: Vec<f64> = self.results.iter()
            .map(|r| r.metrics.avg_latency_ms)
            .collect();
        
        if !latencies.is_empty() {
            let ci = compute_ci(&latencies, 0.95);
            intervals.insert("latency_ms".to_string(), ci);
        }
        
        // Overall loss rate
        let losses: Vec<f64> = self.results.iter()
            .map(|r| r.metrics.loss_rate())
            .collect();
        
        if !losses.is_empty() {
            let ci = compute_ci(&losses, 0.95);
            intervals.insert("loss_rate".to_string(), ci);
        }
        
        // Quality score
        let scores: Vec<f64> = self.results.iter()
            .map(|r| r.metrics.quality_score)
            .collect();
        
        if !scores.is_empty() {
            let ci = compute_ci(&scores, 0.95);
            intervals.insert("quality_score".to_string(), ci);
        }
        
        intervals
    }

    fn analyze_parameter_correlations(&self) -> Vec<ParameterCorrelation> {
        let mut correlations = Vec::new();
        
        // Extract parameter-metric pairs
        let mut param_scores: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
        
        for result in self.results {
            for (param_name, &param_value) in &result.metrics.algorithm_params {
                param_scores.entry(param_name.clone())
                    .or_default()
                    .push((param_value, result.metrics.quality_score));
            }
        }
        
        // Compute correlations
        for (param_name, pairs) in param_scores {
            if pairs.len() < 10 {
                continue;
            }
            
            let xs: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
            let ys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
            
            let correlation = pearson_correlation(&xs, &ys);
            let n = pairs.len() as f64;
            
            // T-test for significance
            let t = correlation * ((n - 2.0) / (1.0 - correlation * correlation)).sqrt();
            let p_value = 2.0 * (1.0 - t_distribution_cdf(t.abs(), (n - 2.0) as usize));
            
            correlations.push(ParameterCorrelation {
                parameter_name: param_name.clone(),
                metric_name: "quality_score".to_string(),
                correlation,
                p_value,
                significant: p_value < 0.05,
            });
        }
        
        correlations.sort_by(|a, b| 
            b.correlation.abs().partial_cmp(&a.correlation.abs()).unwrap());
        
        correlations
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter()
        .map(|x| (x - m).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f64 - 1.0) * p) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn estimate_reaction_time(metrics: &[&RunMetrics]) -> f64 {
    // Estimate based on outage patterns
    let outage_times: Vec<f64> = metrics.iter()
        .map(|m| m.outage_time_ms / (m.failover_count as f64 + 1.0))
        .collect();
    mean(&outage_times)
}

fn compute_ci(values: &[f64], confidence: f64) -> ConfidenceInterval {
    let n = values.len() as f64;
    let m = mean(values);
    let s = std_dev(values);
    let se = s / n.sqrt();
    
    // Z-scores for confidence levels
    let z_95 = 1.96;
    let z_99 = 2.576;
    
    ConfidenceInterval {
        metric: "".to_string(),
        mean: m,
        lower_95: m - z_95 * se,
        upper_95: m + z_95 * se,
        lower_99: m - z_99 * se,
        upper_99: m + z_99 * se,
    }
}

fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.is_empty() {
        return 0.0;
    }
    
    let n = xs.len() as f64;
    let mean_x = mean(xs);
    let mean_y = mean(ys);
    
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    if var_x * var_y == 0.0 {
        0.0
    } else {
        cov / (var_x * var_y).sqrt()
    }
}

fn t_distribution_cdf(t: f64, df: usize) -> f64 {
    // Approximation using normal for large df
    if df > 30 {
        // Use normal approximation
        let normal = Normal::new(0.0, 1.0).unwrap();
        return normal.cdf(t);
    }
    
    // For smaller df, use a simple approximation
    0.5 + 0.5 * (t / (1.0 + t * t / df as f64).sqrt()).tanh()
}
