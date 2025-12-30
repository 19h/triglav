//! Neural Network Module for Advanced Failover Prediction
//!
//! Implements state-of-the-art deep learning architectures:
//! - LSTM networks for sequential pattern recognition
//! - Transformer models for attention-based prediction
//! - Temporal Convolutional Networks (TCN) for efficient sequence modeling

pub mod lstm;
pub mod transformer;
pub mod tcn;

pub use lstm::LSTMNetwork;
pub use transformer::TransformerNetwork;
pub use tcn::TCNNetwork;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Input features for neural network models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkFeatures {
    /// RTT measurements (ms) - last N samples
    pub rtt_history: Vec<f64>,
    /// Packet loss rate history
    pub loss_history: Vec<f64>,
    /// Jitter measurements
    pub jitter_history: Vec<f64>,
    /// Bandwidth measurements (Mbps)
    pub bandwidth_history: Vec<f64>,
    /// Signal quality (dBm) per path
    pub signal_strength: Vec<f64>,
    /// Time of day (normalized 0-1)
    pub time_of_day: f64,
    /// Day of week (one-hot encoded)
    pub day_of_week: [f64; 7],
    /// Current path index
    pub current_path: usize,
    /// Number of available paths
    pub num_paths: usize,
    /// Path-specific features (flattened)
    pub path_features: Vec<f64>,
    /// Recent failover events
    pub recent_failovers: Vec<f64>,
    /// Network topology embedding
    pub topology_embedding: Vec<f64>,
}

impl NetworkFeatures {
    /// Convert to normalized tensor for neural network input
    pub fn to_tensor(&self, sequence_length: usize) -> Vec<f64> {
        let mut features = Vec::with_capacity(512);
        
        // Pad or truncate histories to sequence_length
        let pad_or_truncate = |v: &[f64], len: usize| -> Vec<f64> {
            if v.len() >= len {
                v[v.len() - len..].to_vec()
            } else {
                let mut padded = vec![0.0; len - v.len()];
                padded.extend(v);
                padded
            }
        };
        
        features.extend(pad_or_truncate(&self.rtt_history, sequence_length));
        features.extend(pad_or_truncate(&self.loss_history, sequence_length));
        features.extend(pad_or_truncate(&self.jitter_history, sequence_length));
        features.extend(pad_or_truncate(&self.bandwidth_history, sequence_length));
        features.extend(&self.signal_strength);
        features.push(self.time_of_day);
        features.extend(&self.day_of_week);
        features.push(self.current_path as f64 / self.num_paths.max(1) as f64);
        features.push(self.num_paths as f64 / 10.0);
        features.extend(&self.path_features);
        features.extend(pad_or_truncate(&self.recent_failovers, 10));
        features.extend(&self.topology_embedding);
        
        features
    }
    
    /// Create empty features with given dimensions
    pub fn empty(num_paths: usize) -> Self {
        Self {
            rtt_history: Vec::new(),
            loss_history: Vec::new(),
            jitter_history: Vec::new(),
            bandwidth_history: Vec::new(),
            signal_strength: vec![0.0; num_paths],
            time_of_day: 0.0,
            day_of_week: [0.0; 7],
            current_path: 0,
            num_paths,
            path_features: vec![0.0; num_paths * 8],
            recent_failovers: Vec::new(),
            topology_embedding: vec![0.0; 32],
        }
    }
}

/// Output prediction from neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverPrediction {
    /// Probability of needing failover in next time window
    pub failover_probability: f64,
    /// Predicted time until failure (ms)
    pub time_to_failure_ms: f64,
    /// Confidence interval for time to failure
    pub ttf_confidence: (f64, f64),
    /// Recommended path index
    pub recommended_path: usize,
    /// Path scores (higher is better)
    pub path_scores: Vec<f64>,
    /// Uncertainty estimate (epistemic)
    pub uncertainty: f64,
    /// Attention weights (for interpretability)
    pub attention_weights: Option<Vec<f64>>,
    /// Anomaly score (0-1, higher is more anomalous)
    pub anomaly_score: f64,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Sequence length for temporal models
    pub sequence_length: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads (for transformer)
    pub num_heads: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Weight decay for regularization
    pub weight_decay: f64,
    /// Gradient clipping norm
    pub grad_clip_norm: f64,
    /// Use layer normalization
    pub layer_norm: bool,
    /// Use residual connections
    pub residual: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            sequence_length: 64,
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 8,
            dropout: 0.1,
            learning_rate: 1e-4,
            batch_size: 64,
            epochs: 100,
            weight_decay: 1e-5,
            grad_clip_norm: 1.0,
            layer_norm: true,
            residual: true,
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub auc_roc: f64,
    pub calibration_error: f64,
}

/// Abstract trait for all neural network failover predictors
pub trait NeuralPredictor: Send + Sync {
    /// Get model name
    fn name(&self) -> &str;
    
    /// Get model description
    fn description(&self) -> String;
    
    /// Make prediction from features
    fn predict(&self, features: &NetworkFeatures) -> FailoverPrediction;
    
    /// Update model with new observation (online learning)
    fn update(&mut self, features: &NetworkFeatures, actual_failover: bool, actual_path: usize);
    
    /// Get model complexity (number of parameters)
    fn num_parameters(&self) -> usize;
    
    /// Get inference latency estimate (microseconds)
    fn inference_latency_us(&self) -> u64;
    
    /// Reset model state
    fn reset(&mut self);
    
    /// Clone as boxed trait object
    fn clone_box(&self) -> Box<dyn NeuralPredictor>;
}

/// Statistics for neural network performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NeuralStats {
    pub predictions: usize,
    pub correct_predictions: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub total_inference_time_us: u64,
    pub prediction_confidences: Vec<f64>,
    pub calibration_bins: Vec<(f64, f64)>,  // (predicted_prob, actual_rate)
}

impl NeuralStats {
    pub fn accuracy(&self) -> f64 {
        if self.predictions == 0 {
            return 0.0;
        }
        self.correct_predictions as f64 / self.predictions as f64
    }
    
    pub fn precision(&self) -> f64 {
        let tp = self.correct_predictions.saturating_sub(self.false_negatives);
        let fp = self.false_positives;
        if tp + fp == 0 {
            return 0.0;
        }
        tp as f64 / (tp + fp) as f64
    }
    
    pub fn recall(&self) -> f64 {
        let tp = self.correct_predictions.saturating_sub(self.false_negatives);
        let fn_ = self.false_negatives;
        if tp + fn_ == 0 {
            return 0.0;
        }
        tp as f64 / (tp + fn_) as f64
    }
    
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }
    
    pub fn avg_inference_time_us(&self) -> f64 {
        if self.predictions == 0 {
            return 0.0;
        }
        self.total_inference_time_us as f64 / self.predictions as f64
    }
}
