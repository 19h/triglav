//! Predictive quality analytics using time series analysis.

use std::collections::VecDeque;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Quality prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Predicted RTT.
    pub rtt: Duration,
    /// Confidence (0-1).
    pub confidence: f64,
    /// Predicted degradation in next period.
    pub degradation_likely: bool,
    /// Predicted improvement in next period.
    pub improvement_likely: bool,
    /// Recommended action.
    pub recommendation: PredictionRecommendation,
}

/// Recommended action based on prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionRecommendation {
    /// Continue current strategy.
    Continue,
    /// Consider switching to backup uplink.
    ConsiderSwitch,
    /// Reduce traffic on this uplink.
    ReduceLoad,
    /// Increase traffic on this uplink.
    IncreaseLoad,
    /// Immediate failover recommended.
    Failover,
}

/// Time series sample.
#[derive(Debug, Clone)]
struct Sample {
    rtt_ms: f64,
    loss: f64,
}

/// Quality predictor using exponential smoothing.
pub struct QualityPredictor {
    /// Historical samples.
    samples: VecDeque<Sample>,
    /// Maximum samples to keep.
    max_samples: usize,
    /// Smoothing factor for level.
    alpha: f64,
    /// Smoothing factor for trend.
    beta: f64,
    /// Current level estimate.
    level: f64,
    /// Current trend estimate.
    trend: f64,
    /// Variance estimate.
    variance: f64,
}

impl QualityPredictor {
    /// Create a new predictor.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            alpha: 0.3, // Level smoothing
            beta: 0.1,  // Trend smoothing
            level: 0.0,
            trend: 0.0,
            variance: 0.0,
        }
    }

    /// Add a sample.
    pub fn add_sample(&mut self, rtt: Duration, loss: f64) {
        let rtt_ms = rtt.as_secs_f64() * 1000.0;

        let sample = Sample { rtt_ms, loss };

        // Holt's linear exponential smoothing
        if self.samples.is_empty() {
            self.level = rtt_ms;
            self.trend = 0.0;
        } else {
            let prev_level = self.level;
            self.level = self.alpha * rtt_ms + (1.0 - self.alpha) * (prev_level + self.trend);
            self.trend = self.beta * (self.level - prev_level) + (1.0 - self.beta) * self.trend;

            // Update variance estimate
            let error = rtt_ms - prev_level;
            self.variance = self.alpha * error * error + (1.0 - self.alpha) * self.variance;
        }

        // Add sample
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Predict quality for future time.
    pub fn predict(&self, horizon: Duration) -> Prediction {
        if self.samples.len() < 3 {
            return Prediction {
                rtt: Duration::ZERO,
                confidence: 0.0,
                degradation_likely: false,
                improvement_likely: false,
                recommendation: PredictionRecommendation::Continue,
            };
        }

        // Predict RTT at horizon
        let steps = horizon.as_secs_f64();
        let predicted_rtt_ms = self.level + self.trend * steps;

        // Confidence based on variance and sample count
        let std_dev = self.variance.sqrt();
        let sample_confidence = (self.samples.len() as f64 / self.max_samples as f64).min(1.0);
        let variance_confidence = 1.0 / (1.0 + std_dev / 100.0);
        let confidence = sample_confidence * variance_confidence;

        // Analyze trend
        let degradation_likely = self.trend > 5.0 && predicted_rtt_ms > self.level * 1.2;
        let improvement_likely = self.trend < -5.0 && predicted_rtt_ms < self.level * 0.8;

        // Determine recommendation
        let recommendation = self.recommend(predicted_rtt_ms, degradation_likely);

        Prediction {
            rtt: Duration::from_secs_f64(predicted_rtt_ms.max(0.0) / 1000.0),
            confidence,
            degradation_likely,
            improvement_likely,
            recommendation,
        }
    }

    /// Generate recommendation.
    fn recommend(&self, predicted_rtt_ms: f64, degrading: bool) -> PredictionRecommendation {
        // Check recent loss rate
        let recent_loss: f64 = self
            .samples
            .iter()
            .rev()
            .take(5)
            .map(|s| s.loss)
            .sum::<f64>()
            / 5.0;

        if recent_loss > 0.2 || predicted_rtt_ms > 1000.0 {
            PredictionRecommendation::Failover
        } else if degrading && predicted_rtt_ms > 200.0 {
            PredictionRecommendation::ConsiderSwitch
        } else if recent_loss > 0.05 || predicted_rtt_ms > 100.0 {
            PredictionRecommendation::ReduceLoad
        } else if recent_loss < 0.01 && predicted_rtt_ms < 50.0 {
            PredictionRecommendation::IncreaseLoad
        } else {
            PredictionRecommendation::Continue
        }
    }

    /// Detect anomalies in recent samples.
    pub fn detect_anomaly(&self, threshold_stddev: f64) -> bool {
        if self.samples.len() < 10 {
            return false;
        }

        let std_dev = self.variance.sqrt();
        if std_dev == 0.0 {
            return false;
        }

        // Check if latest sample is an outlier
        if let Some(latest) = self.samples.back() {
            let z_score = (latest.rtt_ms - self.level).abs() / std_dev;
            z_score > threshold_stddev
        } else {
            false
        }
    }

    /// Get trend direction.
    pub fn trend_direction(&self) -> TrendDirection {
        if self.trend > 1.0 {
            TrendDirection::Increasing
        } else if self.trend < -1.0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    /// Get current smoothed estimate.
    pub fn current_estimate(&self) -> Duration {
        Duration::from_secs_f64(self.level.max(0.0) / 1000.0)
    }
}

/// Trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

impl Default for QualityPredictor {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_basic() {
        let mut predictor = QualityPredictor::new(100);

        // Add some samples
        for i in 0..20 {
            let rtt = Duration::from_millis(50 + i);
            predictor.add_sample(rtt, 0.0);
        }

        let prediction = predictor.predict(Duration::from_secs(1));
        assert!(prediction.confidence > 0.0);
        assert!(prediction.rtt > Duration::ZERO);
    }

    #[test]
    fn test_trend_detection() {
        let mut predictor = QualityPredictor::new(100);

        // Add increasing samples
        for i in 0..20 {
            let rtt = Duration::from_millis(50 + i * 10);
            predictor.add_sample(rtt, 0.0);
        }

        assert_eq!(predictor.trend_direction(), TrendDirection::Increasing);
    }
}
