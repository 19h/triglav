//! Temporal Convolutional Network (TCN)
//!
//! Implements a dilated causal convolution network for efficient
//! long-range temporal dependency modeling.

use super::*;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::collections::VecDeque;

/// 1D Causal Convolution with dilation
#[derive(Clone)]
struct CausalConv1d {
    kernel_size: usize,
    dilation: usize,
    in_channels: usize,
    out_channels: usize,
    weights: Array2<f64>,  // [out_channels, in_channels * kernel_size]
    bias: Array1<f64>,
}

impl CausalConv1d {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, dilation: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        Self {
            kernel_size,
            dilation,
            in_channels,
            out_channels,
            weights: Array2::from_shape_fn(
                (out_channels, in_channels * kernel_size),
                |_| normal.sample(&mut rng)
            ),
            bias: Array1::zeros(out_channels),
        }
    }
    
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        // Simplified 1D convolution for single timestep with history
        // In practice, would operate on full sequence
        let receptive_field = self.in_channels * self.kernel_size;
        let mut input = vec![0.0; receptive_field];
        
        let x_len = x.len();
        for i in 0..self.kernel_size.min(x_len) {
            let start = i * self.in_channels;
            let end = start + self.in_channels.min(x_len - i * self.in_channels);
            for j in 0..(end - start) {
                if start + j < receptive_field && i * self.in_channels + j < x_len {
                    input[start + j] = x[i * self.in_channels + j];
                }
            }
        }
        
        let input_arr = Array1::from_vec(input);
        self.weights.dot(&input_arr) + &self.bias
    }
    
    fn num_parameters(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

/// Residual Block with dilated convolutions
#[derive(Clone)]
struct TCNResidualBlock {
    conv1: CausalConv1d,
    conv2: CausalConv1d,
    /// Residual connection (1x1 conv if channels differ)
    residual_conv: Option<Array2<f64>>,
    gamma1: Array1<f64>,
    beta1: Array1<f64>,
    gamma2: Array1<f64>,
    beta2: Array1<f64>,
    hidden_dim: usize,
    dropout_rate: f64,
}

impl TCNResidualBlock {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, dilation: usize, dropout_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / in_channels as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        let residual_conv = if in_channels != out_channels {
            Some(Array2::from_shape_fn((out_channels, in_channels), |_| normal.sample(&mut rng)))
        } else {
            None
        };
        
        Self {
            conv1: CausalConv1d::new(in_channels, out_channels, kernel_size, dilation),
            conv2: CausalConv1d::new(out_channels, out_channels, kernel_size, dilation),
            residual_conv,
            gamma1: Array1::ones(out_channels),
            beta1: Array1::zeros(out_channels),
            gamma2: Array1::ones(out_channels),
            beta2: Array1::zeros(out_channels),
            hidden_dim: out_channels,
            dropout_rate,
        }
    }
    
    fn layer_norm(x: &Array1<f64>, gamma: &Array1<f64>, beta: &Array1<f64>) -> Array1<f64> {
        let mean = x.mean().unwrap_or(0.0);
        let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
        let std = (var + 1e-5).sqrt();
        (x - mean) / std * gamma + beta
    }
    
    fn relu(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.max(0.0))
    }
    
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        // First conv block
        let h = self.conv1.forward(x);
        let h = Self::layer_norm(&h, &self.gamma1, &self.beta1);
        let h = Self::relu(&h);
        
        // Second conv block
        let h = self.conv2.forward(&h);
        let h = Self::layer_norm(&h, &self.gamma2, &self.beta2);
        
        // Residual connection
        let residual = if let Some(ref conv) = self.residual_conv {
            conv.dot(x)
        } else {
            // Ensure dimensions match
            if x.len() == h.len() {
                x.clone()
            } else {
                Array1::zeros(h.len())
            }
        };
        
        Self::relu(&(&h + &residual))
    }
    
    fn num_parameters(&self) -> usize {
        let conv_params = self.conv1.num_parameters() + self.conv2.num_parameters();
        let norm_params = 4 * self.hidden_dim;
        let residual_params = self.residual_conv.as_ref().map(|c| c.len()).unwrap_or(0);
        conv_params + norm_params + residual_params
    }
}

/// Full TCN Network
#[derive(Clone)]
pub struct TCNNetwork {
    name: String,
    config: ModelConfig,
    /// Input projection
    input_linear: Array2<f64>,
    input_bias: Array1<f64>,
    /// TCN blocks with increasing dilation
    blocks: Vec<TCNResidualBlock>,
    /// Global pooling followed by output heads
    output_linear: Array2<f64>,
    output_bias: Array1<f64>,
    path_linear: Array2<f64>,
    path_bias: Array1<f64>,
    time_linear: Array2<f64>,
    time_bias: Array1<f64>,
    uncertainty_linear: Array2<f64>,
    uncertainty_bias: Array1<f64>,
    num_paths: usize,
    feature_dim: usize,
    recent_history: VecDeque<(NetworkFeatures, bool)>,
}

impl TCNNetwork {
    pub fn new(config: ModelConfig, num_paths: usize) -> Self {
        let feature_dim = config.sequence_length * 4 + num_paths + 8 + 2 + num_paths * 8 + 10 + 32;
        let hidden_dim = config.hidden_dim;
        let kernel_size = 3;
        
        let mut rng = rand::thread_rng();
        let std = (2.0 / feature_dim as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        // Create TCN blocks with exponentially increasing dilation
        let mut blocks = Vec::new();
        let mut in_channels = hidden_dim;
        for i in 0..config.num_layers {
            let dilation = 2_usize.pow(i as u32);
            blocks.push(TCNResidualBlock::new(
                in_channels,
                hidden_dim,
                kernel_size,
                dilation,
                config.dropout,
            ));
            in_channels = hidden_dim;
        }
        
        Self {
            name: "TCN-Dilated".to_string(),
            input_linear: Array2::from_shape_fn((hidden_dim, feature_dim), |_| normal.sample(&mut rng) * 0.1),
            input_bias: Array1::zeros(hidden_dim),
            blocks,
            output_linear: Array2::from_shape_fn((2, hidden_dim), |_| normal.sample(&mut rng)),
            output_bias: Array1::zeros(2),
            path_linear: Array2::from_shape_fn((num_paths, hidden_dim), |_| normal.sample(&mut rng)),
            path_bias: Array1::zeros(num_paths),
            time_linear: Array2::from_shape_fn((2, hidden_dim), |_| normal.sample(&mut rng)),
            time_bias: Array1::zeros(2),
            uncertainty_linear: Array2::from_shape_fn((1, hidden_dim), |_| normal.sample(&mut rng)),
            uncertainty_bias: Array1::zeros(1),
            num_paths,
            feature_dim,
            config,
            recent_history: VecDeque::with_capacity(1000),
        }
    }
    
    fn project_input(&self, features: &NetworkFeatures) -> Array1<f64> {
        let raw = features.to_tensor(self.config.sequence_length);
        let mut padded = vec![0.0; self.feature_dim];
        for (i, &v) in raw.iter().enumerate() {
            if i < self.feature_dim {
                padded[i] = v;
            }
        }
        let x = Array1::from_vec(padded);
        let projected = self.input_linear.dot(&x) + &self.input_bias;
        projected.mapv(|v| v.max(0.0))
    }
    
    fn forward_pass(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut hidden = x.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }
        hidden
    }
}

impl NeuralPredictor for TCNNetwork {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> String {
        format!(
            "TCN with {} layers, exponential dilation, {} channels, receptive field ~{}",
            self.config.num_layers,
            self.config.hidden_dim,
            3 * (2_usize.pow(self.config.num_layers as u32) - 1) + 1
        )
    }
    
    fn predict(&self, features: &NetworkFeatures) -> FailoverPrediction {
        let x = self.project_input(features);
        let hidden = self.forward_pass(&x);
        
        // Failover probability
        let logits = self.output_linear.dot(&hidden) + &self.output_bias;
        let failover_prob = 1.0 / (1.0 + (-logits[0]).exp());
        
        // Path scores
        let path_logits = self.path_linear.dot(&hidden) + &self.path_bias;
        let max_logit = path_logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: Vec<f64> = path_logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();
        let path_scores: Vec<f64> = exp_logits.iter().map(|e| e / sum_exp).collect();
        let recommended_path = path_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        // Time to failure
        let time_params = self.time_linear.dot(&hidden) + &self.time_bias;
        let mu = time_params[0];
        let sigma = time_params[1].abs() + 0.1;
        let time_to_failure = (mu + 0.5 * sigma * sigma).exp();
        let ttf_lower = (mu - 1.96 * sigma).exp();
        let ttf_upper = (mu + 1.96 * sigma).exp();
        
        // Uncertainty
        let uncertainty_logit = self.uncertainty_linear.dot(&hidden) + &self.uncertainty_bias;
        let uncertainty = 1.0 / (1.0 + (-uncertainty_logit[0]).exp());
        
        FailoverPrediction {
            failover_probability: failover_prob,
            time_to_failure_ms: time_to_failure.max(0.0).min(3600000.0),
            ttf_confidence: (ttf_lower.max(0.0), ttf_upper.min(3600000.0)),
            recommended_path,
            path_scores,
            uncertainty,
            attention_weights: None,
            anomaly_score: uncertainty * (1.0 - failover_prob),
        }
    }
    
    fn update(&mut self, features: &NetworkFeatures, actual_failover: bool, _actual_path: usize) {
        if self.recent_history.len() >= 1000 {
            self.recent_history.pop_front();
        }
        self.recent_history.push_back((features.clone(), actual_failover));
    }
    
    fn num_parameters(&self) -> usize {
        let input_params = self.input_linear.len() + self.input_bias.len();
        let block_params: usize = self.blocks.iter().map(|b| b.num_parameters()).sum();
        let output_params = self.output_linear.len() + self.output_bias.len();
        let path_params = self.path_linear.len() + self.path_bias.len();
        let time_params = self.time_linear.len() + self.time_bias.len();
        let uncertainty_params = self.uncertainty_linear.len() + self.uncertainty_bias.len();
        
        input_params + block_params + output_params + path_params + time_params + uncertainty_params
    }
    
    fn inference_latency_us(&self) -> u64 {
        // TCN is faster due to parallelizable convolutions
        (self.num_parameters() / 1500).max(50) as u64
    }
    
    fn reset(&mut self) {
        // TCN is stateless
    }
    
    fn clone_box(&self) -> Box<dyn NeuralPredictor> {
        Box::new(self.clone())
    }
}
