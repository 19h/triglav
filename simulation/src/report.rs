//! Report Generation Module
//!
//! Generates actionable reports with optimal algorithm parameters.

use std::fs::File;
use std::io::{Write, BufWriter};
use colored::Colorize;
use comfy_table::{Table, ContentArrangement, presets::UTF8_FULL};

use crate::analysis::AnalysisFindings;
use crate::simulation::SimulationConfig;

// ============================================================================
// Report
// ============================================================================

pub struct Report<'a> {
    findings: &'a AnalysisFindings,
    config: &'a SimulationConfig,
}

impl<'a> Report<'a> {
    pub fn new(findings: &'a AnalysisFindings, config: &'a SimulationConfig) -> Self {
        Self { findings, config }
    }

    /// Print summary to console
    pub fn print_summary(&self) {
        println!();
        println!("{}", "═".repeat(80).bright_blue());
        println!("{}", "  SIMULATION RESULTS SUMMARY".bright_white().bold());
        println!("{}", "═".repeat(80).bright_blue());
        println!();

        // Algorithm Rankings
        self.print_algorithm_rankings();
        println!();

        // Scenario Winners
        self.print_scenario_winners();
        println!();

        // Optimal Parameters
        self.print_optimal_parameters();
        println!();

        // Key Correlations
        self.print_key_correlations();
        println!();

        // Actionable Recommendations
        self.print_recommendations();
    }

    fn print_algorithm_rankings(&self) {
        println!("{}", "▶ ALGORITHM RANKINGS".bright_yellow().bold());
        println!();

        let mut table = Table::new();
        table.load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                "Rank", "Algorithm", "Score", "Latency", "Loss", "Continuity", "Stability"
            ]);

        for ranking in &self.findings.overall_ranking {
            let score_color = if ranking.composite_score > 0.7 {
                format!("{:.3}", ranking.composite_score).green().to_string()
            } else if ranking.composite_score > 0.5 {
                format!("{:.3}", ranking.composite_score).yellow().to_string()
            } else {
                format!("{:.3}", ranking.composite_score).red().to_string()
            };

            table.add_row(vec![
                format!("#{}", ranking.rank),
                ranking.algorithm_id.clone(),
                score_color,
                format!("#{}", ranking.latency_rank),
                format!("#{}", ranking.loss_rank),
                format!("#{}", ranking.continuity_rank),
                format!("#{}", ranking.stability_rank),
            ]);
        }

        println!("{table}");
    }

    fn print_scenario_winners(&self) {
        println!("{}", "▶ SCENARIO WINNERS".bright_yellow().bold());
        println!();

        let mut table = Table::new();
        table.load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec!["Scenario", "Best Algorithm", "Difficulty"]);

        for (scenario, winner) in &self.findings.scenario_winners {
            let difficulty = self.findings.scenario_stats.get(scenario)
                .map(|s| {
                    let score = s.event_count_mean / 10.0 + s.coverage_gap_count_mean * 2.0;
                    if score > 10.0 { "Hard" }
                    else if score > 5.0 { "Medium" }
                    else { "Easy" }
                })
                .unwrap_or("Unknown");

            table.add_row(vec![
                scenario.clone(),
                winner.clone(),
                difficulty.to_string(),
            ]);
        }

        println!("{table}");
    }

    fn print_optimal_parameters(&self) {
        println!("{}", "▶ OPTIMAL PARAMETERS".bright_yellow().bold());
        println!();
        
        let params = &self.findings.optimal_parameters;

        println!("  {} Detection Parameters:", "●".bright_cyan());
        println!("    RTT Threshold:        {} (confidence: {:.0}%)",
            format!("{:.0}ms", params.rtt_threshold_ms).bright_white(),
            params.rtt_threshold_confidence * 100.0);
        println!("    Loss Threshold:       {} (confidence: {:.0}%)",
            format!("{:.1}%", params.loss_threshold_percent).bright_white(),
            params.loss_threshold_confidence * 100.0);
        println!("    Consecutive Failures: {}",
            format!("{}", params.consecutive_failures).bright_white());
        println!("      └─ {}", params.consecutive_failures_rationale.bright_black());
        println!();

        println!("  {} Timing Parameters:", "●".bright_cyan());
        println!("    EWMA Alpha:           {}",
            format!("{:.3}", params.ewma_alpha).bright_white());
        println!("      └─ {}", params.ewma_alpha_rationale.bright_black());
        println!("    Hysteresis Period:    {}",
            format!("{}ms", params.hysteresis_ms).bright_white());
        println!("      └─ {}", params.hysteresis_rationale.bright_black());
        println!("    Heartbeat Interval:   {}",
            format!("{}ms", params.heartbeat_interval_ms).bright_white());
        println!("    Max Silent Period:    {}",
            format!("{}ms", params.max_silent_period_ms).bright_white());
        println!();

        println!("  {} Advanced Parameters:", "●".bright_cyan());
        println!("    Path Diversity:       {} paths",
            format!("{}", params.path_diversity).bright_white());
        println!("    Preemptive Switching: {}",
            if params.preemptive_switching_enabled { "Enabled".green() } else { "Disabled".red() });
        println!("    Preemptive Threshold: RTT trend > {}ms/s",
            format!("{:.1}", params.preemptive_threshold_trend).bright_white());
        println!();

        println!("  {} Recommended Algorithm:", "★".bright_green());
        println!("    {}",
            params.recommended_algorithm.bright_white().bold());
        println!("      └─ {}", params.recommended_algorithm_rationale.bright_black());
    }

    fn print_key_correlations(&self) {
        println!("{}", "▶ PARAMETER-PERFORMANCE CORRELATIONS".bright_yellow().bold());
        println!();

        let significant: Vec<_> = self.findings.parameter_correlations.iter()
            .filter(|c| c.significant)
            .take(5)
            .collect();

        if significant.is_empty() {
            println!("  No statistically significant correlations found.");
            return;
        }

        let mut table = Table::new();
        table.load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec!["Parameter", "Correlation", "p-value", "Impact"]);

        for corr in significant {
            let impact = if corr.correlation > 0.5 {
                "Strong Positive".green().to_string()
            } else if corr.correlation > 0.2 {
                "Moderate Positive".yellow().to_string()
            } else if corr.correlation < -0.5 {
                "Strong Negative".red().to_string()
            } else if corr.correlation < -0.2 {
                "Moderate Negative".yellow().to_string()
            } else {
                "Weak".bright_black().to_string()
            };

            table.add_row(vec![
                corr.parameter_name.clone(),
                format!("{:.3}", corr.correlation),
                format!("{:.4}", corr.p_value),
                impact,
            ]);
        }

        println!("{table}");
    }

    fn print_recommendations(&self) {
        println!("{}", "▶ ACTIONABLE RECOMMENDATIONS".bright_yellow().bold());
        println!();

        let params = &self.findings.optimal_parameters;

        println!("  {} PRIMARY RECOMMENDATION:", "1.".bright_green().bold());
        println!("     Implement {} algorithm as the primary failover mechanism.",
            params.recommended_algorithm.bright_white().bold());
        println!();

        println!("  {} CONFIGURATION:", "2.".bright_green().bold());
        println!("     Use the following parameters in production:");
        println!();
        println!("     ```toml");
        println!("     [failover]");
        println!("     algorithm = \"{}\"", params.recommended_algorithm);
        println!("     rtt_threshold_ms = {:.0}", params.rtt_threshold_ms);
        println!("     loss_threshold_percent = {:.1}", params.loss_threshold_percent);
        println!("     consecutive_failures = {}", params.consecutive_failures);
        println!("     hysteresis_ms = {}", params.hysteresis_ms);
        println!("     ewma_alpha = {:.3}", params.ewma_alpha);
        println!();
        println!("     [probing]");
        println!("     heartbeat_interval_ms = {}", params.heartbeat_interval_ms);
        println!("     max_silent_period_ms = {}", params.max_silent_period_ms);
        println!("     path_diversity = {}", params.path_diversity);
        println!();
        println!("     [preemptive]");
        println!("     enabled = {}", params.preemptive_switching_enabled);
        println!("     trend_threshold_ms_per_s = {:.1}", params.preemptive_threshold_trend);
        println!("     ```");
        println!();

        println!("  {} MONITORING:", "3.".bright_green().bold());
        println!("     Track these metrics to validate failover performance:");
        println!("     - Failover count per hour (expect 2-10 in variable conditions)");
        println!("     - Failed failover rate (target <5%)");
        println!("     - Connection continuity score (target >95%)");
        println!("     - P99 latency during failover events");
        println!();

        println!("  {} TUNING GUIDANCE:", "4.".bright_green().bold());
        println!("     - If too many failovers: Increase hysteresis_ms by 50%");
        println!("     - If too slow to failover: Decrease consecutive_failures to 2");
        println!("     - If path oscillation: Increase ewma_alpha to 0.2");
        println!("     - For VoIP/real-time: Decrease max_silent_period_ms to 300");
    }

    /// Export results to CSV
    pub fn export_csv(&self, filename: &str) -> std::io::Result<()> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        // Header
        writeln!(writer, "algorithm_id,rank,composite_score,latency_mean,latency_std,loss_rate_mean,continuity_mean,failover_count_mean,stability_score")?;

        // Data
        for ranking in &self.findings.overall_ranking {
            if let Some(stats) = self.findings.algorithm_stats.get(&ranking.algorithm_id) {
                writeln!(
                    writer,
                    "{},{},{:.4},{:.2},{:.2},{:.6},{:.4},{:.2},{:.4}",
                    stats.algorithm_id,
                    ranking.rank,
                    ranking.composite_score,
                    stats.latency_mean,
                    stats.latency_std,
                    stats.loss_rate_mean,
                    stats.continuity_mean,
                    stats.failover_count_mean,
                    stats.stability_score,
                )?;
            }
        }

        println!("  Exported: {}", filename.bright_cyan());
        Ok(())
    }

    /// Export results to JSON
    pub fn export_json(&self, filename: &str) -> std::io::Result<()> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        let json = serde_json::json!({
            "config": {
                "monte_carlo_iterations": self.config.monte_carlo_iterations,
                "simulation_duration_s": self.config.simulation_duration_s,
                "time_resolution_ms": self.config.time_resolution_ms,
            },
            "rankings": self.findings.overall_ranking,
            "algorithm_stats": self.findings.algorithm_stats,
            "scenario_stats": self.findings.scenario_stats,
            "scenario_winners": self.findings.scenario_winners,
            "optimal_parameters": self.findings.optimal_parameters,
            "confidence_intervals": self.findings.confidence_intervals,
            "parameter_correlations": self.findings.parameter_correlations,
        });

        serde_json::to_writer_pretty(&mut writer, &json)?;
        println!("  Exported: {}", filename.bright_cyan());
        Ok(())
    }

    /// Export optimal parameters as TOML config
    pub fn export_algorithm_parameters(&self, filename: &str) -> std::io::Result<()> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        let params = &self.findings.optimal_parameters;

        writeln!(writer, "# Triglav Optimal Failover Parameters")?;
        writeln!(writer, "# Generated from {} Monte Carlo iterations", self.config.monte_carlo_iterations)?;
        writeln!(writer, "# Simulation duration: {}s per scenario", self.config.simulation_duration_s)?;
        writeln!(writer, "")?;

        writeln!(writer, "[failover]")?;
        writeln!(writer, "# Recommended algorithm: {}", params.recommended_algorithm)?;
        writeln!(writer, "algorithm = \"{}\"", params.recommended_algorithm)?;
        writeln!(writer, "")?;
        writeln!(writer, "# Detection thresholds")?;
        writeln!(writer, "rtt_threshold_ms = {:.0}", params.rtt_threshold_ms)?;
        writeln!(writer, "loss_threshold_percent = {:.1}", params.loss_threshold_percent)?;
        writeln!(writer, "consecutive_failures = {}", params.consecutive_failures)?;
        writeln!(writer, "")?;
        writeln!(writer, "# Timing parameters")?;
        writeln!(writer, "hysteresis_ms = {}", params.hysteresis_ms)?;
        writeln!(writer, "ewma_alpha = {:.3}", params.ewma_alpha)?;
        writeln!(writer, "")?;

        writeln!(writer, "[probing]")?;
        writeln!(writer, "heartbeat_interval_ms = {}", params.heartbeat_interval_ms)?;
        writeln!(writer, "max_silent_period_ms = {}", params.max_silent_period_ms)?;
        writeln!(writer, "path_diversity = {}", params.path_diversity)?;
        writeln!(writer, "")?;

        writeln!(writer, "[preemptive]")?;
        writeln!(writer, "enabled = {}", params.preemptive_switching_enabled)?;
        writeln!(writer, "trend_threshold_ms_per_s = {:.1}", params.preemptive_threshold_trend)?;
        writeln!(writer, "")?;

        writeln!(writer, "# Confidence metrics")?;
        writeln!(writer, "[confidence]")?;
        writeln!(writer, "rtt_threshold = {:.2}", params.rtt_threshold_confidence)?;
        writeln!(writer, "loss_threshold = {:.2}", params.loss_threshold_confidence)?;

        println!("  Exported: {}", filename.bright_cyan());
        Ok(())
    }
}
