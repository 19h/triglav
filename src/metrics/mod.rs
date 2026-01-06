//! Quality metrics, predictive analytics, and Prometheus export.
//!
//! This module provides:
//! - Real-time quality measurement
//! - Historical data tracking
//! - Predictive quality estimation
//! - Anomaly detection
//! - Prometheus metrics export
//! - HTTP health endpoints

#[cfg(feature = "metrics")]
mod http_server;
mod predictor;
#[cfg(feature = "metrics")]
mod prometheus_export;
mod quality;

#[cfg(feature = "metrics")]
pub use http_server::*;
pub use predictor::{Prediction, QualityPredictor};
#[cfg(feature = "metrics")]
pub use prometheus_export::*;
pub use quality::QualityMetrics;

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Metrics configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Sample window for averaging.
    #[serde(default = "default_sample_window", with = "humantime_serde")]
    pub sample_window: Duration,

    /// History retention period.
    #[serde(default = "default_history_retention", with = "humantime_serde")]
    pub history_retention: Duration,

    /// Anomaly detection threshold (standard deviations).
    #[serde(default = "default_anomaly_threshold")]
    pub anomaly_threshold: f64,

    /// Enable predictive analytics.
    #[serde(default = "default_prediction")]
    pub prediction_enabled: bool,

    /// Prediction horizon.
    #[serde(default = "default_prediction_horizon", with = "humantime_serde")]
    pub prediction_horizon: Duration,

    /// Enable Prometheus metrics export.
    #[serde(default = "default_prometheus")]
    pub prometheus_enabled: bool,

    /// HTTP server bind address for metrics/health.
    #[serde(default = "default_http_bind")]
    pub http_bind: String,

    /// Enable per-user metrics.
    #[serde(default)]
    pub per_user_metrics: bool,
}

fn default_sample_window() -> Duration {
    Duration::from_secs(10)
}
fn default_history_retention() -> Duration {
    Duration::from_secs(3600)
}
fn default_anomaly_threshold() -> f64 {
    3.0
}
fn default_prediction() -> bool {
    true
}
fn default_prediction_horizon() -> Duration {
    Duration::from_secs(60)
}
fn default_prometheus() -> bool {
    true
}
fn default_http_bind() -> String {
    "0.0.0.0:9090".to_string()
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            sample_window: default_sample_window(),
            history_retention: default_history_retention(),
            anomaly_threshold: default_anomaly_threshold(),
            prediction_enabled: default_prediction(),
            prediction_horizon: default_prediction_horizon(),
            prometheus_enabled: default_prometheus(),
            http_bind: default_http_bind(),
            per_user_metrics: false,
        }
    }
}
