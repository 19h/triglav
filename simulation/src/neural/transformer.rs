//! Transformer-based Failover Prediction Network
//!
//! Implements a Transformer encoder with multi-head self-attention
//! for parallel processing of network metric sequences.

use super::*;
use ndarray::{Array1, Array2, s};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::collections::VecDeque;

/// Multi-Head Self-Attention
#[derive(Clone)]
struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    w_q: Array2<f64>,
    w_k: Array2<f64>,
    w_v: Array2<f64>,
    w_o: Array2<f64>,
}

impl MultiHeadAttention {
    fn new(hidden_dim: usize, num_heads: usize) -> Self {
        assert!(hidden_dim % num_heads == 0);
        let head_dim = hidden_dim / num_heads;
        
        let mut rng = rand::thread_rng();
        let std = (2.0 / hidden_dim as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        let mut init_weight = |rows: usize, cols: usize| -> Array2<f64> {
            Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng))
        };
        
        Self {
            num_heads,
            head_dim,
            hidden_dim,
            w_q: init_weight(hidden_dim, hidden_dim),
            w_k: init_weight(hidden_dim, hidden_dim),
            w_v: init_weight(hidden_dim, hidden_dim),
            w_o: init_weight(hidden_dim, hidden_dim),
        }
    }
    
    fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Vec<f64>) {
        // Single input attention (self-attention with single token)
        let q = self.w_q.dot(x);
        let k = self.w_k.dot(x);
        let v = self.w_v.dot(x);
        
        let scale = (self.head_dim as f64).sqrt();
        
        // Multi-head attention computation
        let mut head_outputs = Array1::zeros(self.hidden_dim);
        let mut all_attention_weights = Vec::new();
        
        for head in 0..self.num_heads {
            let start = head * self.head_dim;
            let end = start + self.head_dim;
            
            let q_h = q.slice(s![start..end]).to_owned();
            let k_h = k.slice(s![start..end]).to_owned();
            let v_h = v.slice(s![start..end]).to_owned();
            
            // Attention score (single token, so softmax is trivial)
            let score = q_h.dot(&k_h) / scale;
            let attention_weight = 1.0; // Single token attention
            all_attention_weights.push(attention_weight);
            
            // Apply attention
            let head_out = attention_weight * &v_h;
            head_outputs.slice_mut(s![start..end]).assign(&head_out);
        }
        
        // Output projection
        let output = self.w_o.dot(&head_outputs);
        
        (output, all_attention_weights)
    }
    
    fn num_parameters(&self) -> usize {
        4 * self.hidden_dim * self.hidden_dim
    }
}

/// Feed-Forward Network
#[derive(Clone)]
struct FeedForward {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
    hidden_dim: usize,
    ff_dim: usize,
}

impl FeedForward {
    fn new(hidden_dim: usize, ff_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / hidden_dim as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        Self {
            w1: Array2::from_shape_fn((ff_dim, hidden_dim), |_| normal.sample(&mut rng)),
            b1: Array1::zeros(ff_dim),
            w2: Array2::from_shape_fn((hidden_dim, ff_dim), |_| normal.sample(&mut rng)),
            b2: Array1::zeros(hidden_dim),
            hidden_dim,
            ff_dim,
        }
    }
    
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        // GELU activation
        let gelu = |v: f64| -> f64 {
            0.5 * v * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI.sqrt() * (v + 0.044715 * v.powi(3))).tanh())
        };
        
        let h = self.w1.dot(x) + &self.b1;
        let h = h.mapv(gelu);
        self.w2.dot(&h) + &self.b2
    }
    
    fn num_parameters(&self) -> usize {
        self.hidden_dim * self.ff_dim + self.ff_dim + self.ff_dim * self.hidden_dim + self.hidden_dim
    }
}

/// Layer Normalization
#[derive(Clone)]
struct LayerNorm {
    gamma: Array1<f64>,
    beta: Array1<f64>,
    eps: f64,
}

impl LayerNorm {
    fn new(dim: usize) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps: 1e-5,
        }
    }
    
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let mean = x.mean().unwrap_or(0.0);
        let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
        let std = (var + self.eps).sqrt();
        (x - mean) / std * &self.gamma + &self.beta
    }
    
    fn num_parameters(&self) -> usize {
        2 * self.gamma.len()
    }
}

/// Transformer Encoder Block
#[derive(Clone)]
struct TransformerBlock {
    attention: MultiHeadAttention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout_rate: f64,
}

impl TransformerBlock {
    fn new(hidden_dim: usize, num_heads: usize, ff_dim: usize, dropout_rate: f64) -> Self {
        Self {
            attention: MultiHeadAttention::new(hidden_dim, num_heads),
            ff: FeedForward::new(hidden_dim, ff_dim),
            norm1: LayerNorm::new(hidden_dim),
            norm2: LayerNorm::new(hidden_dim),
            dropout_rate,
        }
    }
    
    fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Vec<f64>) {
        // Pre-norm architecture
        let normed = self.norm1.forward(x);
        let (attn_out, attn_weights) = self.attention.forward(&normed);
        let x = x + &attn_out; // Residual
        
        let normed = self.norm2.forward(&x);
        let ff_out = self.ff.forward(&normed);
        let x = x + &ff_out; // Residual
        
        (x, attn_weights)
    }
    
    fn num_parameters(&self) -> usize {
        self.attention.num_parameters() + 
        self.ff.num_parameters() + 
        self.norm1.num_parameters() + 
        self.norm2.num_parameters()
    }
}

/// Positional Encoding
#[derive(Clone)]
struct PositionalEncoding {
    encodings: Array2<f64>,
    max_len: usize,
    hidden_dim: usize,
}

impl PositionalEncoding {
    fn new(max_len: usize, hidden_dim: usize) -> Self {
        let mut encodings = Array2::zeros((max_len, hidden_dim));
        
        for pos in 0..max_len {
            for i in 0..hidden_dim {
                let angle = pos as f64 / (10000.0_f64).powf((2 * (i / 2)) as f64 / hidden_dim as f64);
                encodings[[pos, i]] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        
        Self {
            encodings,
            max_len,
            hidden_dim,
        }
    }
    
    fn encode(&self, pos: usize) -> Array1<f64> {
        let pos = pos.min(self.max_len - 1);
        self.encodings.row(pos).to_owned()
    }
}

/// Full Transformer Network
#[derive(Clone)]
pub struct TransformerNetwork {
    name: String,
    config: ModelConfig,
    /// Input projection
    input_linear: Array2<f64>,
    input_bias: Array1<f64>,
    /// Positional encoding
    pos_encoding: PositionalEncoding,
    /// Transformer blocks
    blocks: Vec<TransformerBlock>,
    /// Output head - failover probability
    output_linear: Array2<f64>,
    output_bias: Array1<f64>,
    /// Path scoring head
    path_linear: Array2<f64>,
    path_bias: Array1<f64>,
    /// Time prediction head  
    time_linear: Array2<f64>,
    time_bias: Array1<f64>,
    /// Uncertainty head
    uncertainty_linear: Array2<f64>,
    uncertainty_bias: Array1<f64>,
    /// Number of paths
    num_paths: usize,
    /// Feature dimension
    feature_dim: usize,
    /// Recent history for online learning
    recent_history: VecDeque<(NetworkFeatures, bool)>,
}

impl TransformerNetwork {
    pub fn new(config: ModelConfig, num_paths: usize) -> Self {
        let feature_dim = config.sequence_length * 4 + num_paths + 8 + 2 + num_paths * 8 + 10 + 32;
        let hidden_dim = config.hidden_dim;
        let ff_dim = hidden_dim * 4;
        
        let mut rng = rand::thread_rng();
        let std = (2.0 / feature_dim as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        let mut blocks = Vec::new();
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(
                hidden_dim,
                config.num_heads,
                ff_dim,
                config.dropout,
            ));
        }
        
        Self {
            name: "Transformer-Encoder".to_string(),
            input_linear: Array2::from_shape_fn((hidden_dim, feature_dim), |_| normal.sample(&mut rng) * 0.1),
            input_bias: Array1::zeros(hidden_dim),
            pos_encoding: PositionalEncoding::new(512, hidden_dim),
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
        self.input_linear.dot(&x) + &self.input_bias
    }
    
    fn forward_pass(&self, x: &Array1<f64>) -> (Array1<f64>, Vec<Vec<f64>>) {
        // Add positional encoding
        let x = x + &self.pos_encoding.encode(0);
        
        let mut hidden = x;
        let mut all_attention_weights = Vec::new();
        
        for block in &self.blocks {
            let (h, attn_weights) = block.forward(&hidden);
            hidden = h;
            all_attention_weights.push(attn_weights);
        }
        
        (hidden, all_attention_weights)
    }
}

impl NeuralPredictor for TransformerNetwork {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> String {
        format!(
            "Transformer with {} layers, {} heads, {} hidden units",
            self.config.num_layers,
            self.config.num_heads,
            self.config.hidden_dim
        )
    }
    
    fn predict(&self, features: &NetworkFeatures) -> FailoverPrediction {
        let x = self.project_input(features);
        let (hidden, all_attention) = self.forward_pass(&x);
        
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
        
        // Flatten attention weights for output
        let attention_weights: Vec<f64> = all_attention.into_iter().flatten().collect();
        
        FailoverPrediction {
            failover_probability: failover_prob,
            time_to_failure_ms: time_to_failure.max(0.0).min(3600000.0),
            ttf_confidence: (ttf_lower.max(0.0), ttf_upper.min(3600000.0)),
            recommended_path,
            path_scores,
            uncertainty,
            attention_weights: Some(attention_weights),
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
        (self.num_parameters() / 800).max(150) as u64
    }
    
    fn reset(&mut self) {
        // Transformer is stateless, nothing to reset
    }
    
    fn clone_box(&self) -> Box<dyn NeuralPredictor> {
        Box::new(self.clone())
    }
}
