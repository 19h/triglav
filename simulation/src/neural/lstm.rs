//! LSTM-based Failover Prediction Network
//!
//! Implements a stacked bidirectional LSTM with attention mechanism
//! for temporal pattern recognition in network metrics.

use super::*;
use ndarray::{Array1, Array2, Array3, s};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::collections::VecDeque;

/// LSTM Cell implementation
#[derive(Clone)]
struct LSTMCell {
    /// Input dimension
    input_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Weight matrices for input gate
    w_i: Array2<f64>,
    u_i: Array2<f64>,
    b_i: Array1<f64>,
    /// Weight matrices for forget gate
    w_f: Array2<f64>,
    u_f: Array2<f64>,
    b_f: Array1<f64>,
    /// Weight matrices for cell gate
    w_c: Array2<f64>,
    u_c: Array2<f64>,
    b_c: Array1<f64>,
    /// Weight matrices for output gate
    w_o: Array2<f64>,
    u_o: Array2<f64>,
    b_o: Array1<f64>,
    /// Layer normalization parameters
    gamma_h: Array1<f64>,
    beta_h: Array1<f64>,
    gamma_c: Array1<f64>,
    beta_c: Array1<f64>,
    /// Use layer normalization
    use_layer_norm: bool,
}

impl LSTMCell {
    fn new(input_dim: usize, hidden_dim: usize, use_layer_norm: bool) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        let mut init_weight = |rows: usize, cols: usize| -> Array2<f64> {
            Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng))
        };
        
        let init_bias = |size: usize, val: f64| -> Array1<f64> {
            Array1::from_elem(size, val)
        };
        
        Self {
            input_dim,
            hidden_dim,
            w_i: init_weight(hidden_dim, input_dim),
            u_i: init_weight(hidden_dim, hidden_dim),
            b_i: init_bias(hidden_dim, 0.0),
            w_f: init_weight(hidden_dim, input_dim),
            u_f: init_weight(hidden_dim, hidden_dim),
            b_f: init_bias(hidden_dim, 1.0), // Forget gate bias initialization
            w_c: init_weight(hidden_dim, input_dim),
            u_c: init_weight(hidden_dim, hidden_dim),
            b_c: init_bias(hidden_dim, 0.0),
            w_o: init_weight(hidden_dim, input_dim),
            u_o: init_weight(hidden_dim, hidden_dim),
            b_o: init_bias(hidden_dim, 0.0),
            gamma_h: Array1::ones(hidden_dim),
            beta_h: Array1::zeros(hidden_dim),
            gamma_c: Array1::ones(hidden_dim),
            beta_c: Array1::zeros(hidden_dim),
            use_layer_norm,
        }
    }
    
    fn layer_norm(x: &Array1<f64>, gamma: &Array1<f64>, beta: &Array1<f64>) -> Array1<f64> {
        let mean = x.mean().unwrap_or(0.0);
        let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
        let std = (var + 1e-5).sqrt();
        (x - mean) / std * gamma + beta
    }
    
    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }
    
    fn tanh(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.tanh())
    }
    
    fn forward(&self, x: &Array1<f64>, h_prev: &Array1<f64>, c_prev: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        // Input gate
        let i_t = Self::sigmoid(&(&self.w_i.dot(x) + &self.u_i.dot(h_prev) + &self.b_i));
        
        // Forget gate
        let f_t = Self::sigmoid(&(&self.w_f.dot(x) + &self.u_f.dot(h_prev) + &self.b_f));
        
        // Cell candidate
        let c_tilde = Self::tanh(&(&self.w_c.dot(x) + &self.u_c.dot(h_prev) + &self.b_c));
        
        // Cell state
        let mut c_t = &f_t * c_prev + &i_t * &c_tilde;
        if self.use_layer_norm {
            c_t = Self::layer_norm(&c_t, &self.gamma_c, &self.beta_c);
        }
        
        // Output gate
        let o_t = Self::sigmoid(&(&self.w_o.dot(x) + &self.u_o.dot(h_prev) + &self.b_o));
        
        // Hidden state
        let mut h_t = &o_t * &Self::tanh(&c_t);
        if self.use_layer_norm {
            h_t = Self::layer_norm(&h_t, &self.gamma_h, &self.beta_h);
        }
        
        (h_t, c_t)
    }
    
    fn num_parameters(&self) -> usize {
        let gate_params = self.input_dim * self.hidden_dim + 
                          self.hidden_dim * self.hidden_dim + 
                          self.hidden_dim;
        let num_gates = 4;
        let ln_params = if self.use_layer_norm { 4 * self.hidden_dim } else { 0 };
        num_gates * gate_params + ln_params
    }
}

/// Attention mechanism for LSTM outputs
#[derive(Clone)]
struct Attention {
    /// Query projection
    w_q: Array2<f64>,
    /// Key projection
    w_k: Array2<f64>,
    /// Value projection
    w_v: Array2<f64>,
    /// Output projection
    w_o: Array2<f64>,
    /// Hidden dimension
    hidden_dim: usize,
}

impl Attention {
    fn new(hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (1.0 / hidden_dim as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        let mut init_weight = |size: usize| -> Array2<f64> {
            Array2::from_shape_fn((size, size), |_| normal.sample(&mut rng))
        };
        
        Self {
            w_q: init_weight(hidden_dim),
            w_k: init_weight(hidden_dim),
            w_v: init_weight(hidden_dim),
            w_o: init_weight(hidden_dim),
            hidden_dim,
        }
    }
    
    fn forward(&self, query: &Array1<f64>, keys: &[Array1<f64>], values: &[Array1<f64>]) -> (Array1<f64>, Vec<f64>) {
        if keys.is_empty() {
            return (query.clone(), vec![]);
        }
        
        let q = self.w_q.dot(query);
        let scale = (self.hidden_dim as f64).sqrt();
        
        // Compute attention scores
        let mut scores: Vec<f64> = keys.iter()
            .map(|k| {
                let k_proj = self.w_k.dot(k);
                q.dot(&k_proj) / scale
            })
            .collect();
        
        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let attention_weights: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();
        
        // Weighted sum of values
        let mut context = Array1::zeros(self.hidden_dim);
        for (i, v) in values.iter().enumerate() {
            let v_proj = self.w_v.dot(v);
            context = context + attention_weights[i] * &v_proj;
        }
        
        let output = self.w_o.dot(&context);
        (output, attention_weights)
    }
    
    fn num_parameters(&self) -> usize {
        4 * self.hidden_dim * self.hidden_dim
    }
}

/// Stacked Bidirectional LSTM with Attention
#[derive(Clone)]
pub struct LSTMNetwork {
    name: String,
    config: ModelConfig,
    /// Forward LSTM layers
    forward_layers: Vec<LSTMCell>,
    /// Backward LSTM layers
    backward_layers: Vec<LSTMCell>,
    /// Attention mechanism
    attention: Attention,
    /// Output projection
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
    /// Input projection
    input_linear: Array2<f64>,
    input_bias: Array1<f64>,
    /// Hidden states for online learning
    h_states: Vec<Array1<f64>>,
    c_states: Vec<Array1<f64>>,
    /// Recent predictions for calibration
    recent_history: VecDeque<(NetworkFeatures, bool)>,
    /// Number of paths
    num_paths: usize,
    /// Input dimension (after projection)
    input_dim: usize,
    /// Feature dimension (raw input)
    feature_dim: usize,
}

impl LSTMNetwork {
    pub fn new(config: ModelConfig, num_paths: usize) -> Self {
        let feature_dim = config.sequence_length * 4 + num_paths + 8 + 2 + num_paths * 8 + 10 + 32;
        let input_dim = config.hidden_dim;
        let hidden_dim = config.hidden_dim;
        
        let mut rng = rand::thread_rng();
        let std = (2.0 / (feature_dim + hidden_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        // Create stacked LSTM layers
        let mut forward_layers = Vec::new();
        let mut backward_layers = Vec::new();
        
        for i in 0..config.num_layers {
            let layer_input_dim = if i == 0 { input_dim } else { hidden_dim * 2 };
            forward_layers.push(LSTMCell::new(layer_input_dim, hidden_dim, config.layer_norm));
            backward_layers.push(LSTMCell::new(layer_input_dim, hidden_dim, config.layer_norm));
        }
        
        let final_hidden = hidden_dim * 2; // Bidirectional
        
        Self {
            name: "BiLSTM-Attention".to_string(),
            forward_layers,
            backward_layers,
            attention: Attention::new(final_hidden),
            output_linear: Array2::from_shape_fn((2, final_hidden), |_| normal.sample(&mut rng)),
            output_bias: Array1::zeros(2),
            path_linear: Array2::from_shape_fn((num_paths, final_hidden), |_| normal.sample(&mut rng)),
            path_bias: Array1::zeros(num_paths),
            time_linear: Array2::from_shape_fn((2, final_hidden), |_| normal.sample(&mut rng)),
            time_bias: Array1::zeros(2),
            uncertainty_linear: Array2::from_shape_fn((1, final_hidden), |_| normal.sample(&mut rng)),
            uncertainty_bias: Array1::zeros(1),
            input_linear: Array2::from_shape_fn((input_dim, feature_dim), |_| normal.sample(&mut rng) * 0.1),
            input_bias: Array1::zeros(input_dim),
            h_states: vec![Array1::zeros(hidden_dim); config.num_layers * 2],
            c_states: vec![Array1::zeros(hidden_dim); config.num_layers * 2],
            recent_history: VecDeque::with_capacity(1000),
            config,
            num_paths,
            input_dim,
            feature_dim,
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
        projected.mapv(|v| v.max(0.0)) // ReLU activation
    }
    
    fn forward_pass(&self, x: &Array1<f64>) -> (Array1<f64>, Vec<f64>) {
        let hidden_dim = self.config.hidden_dim;
        
        // Initialize hidden states
        let mut h_fwd = Array1::zeros(hidden_dim);
        let mut c_fwd = Array1::zeros(hidden_dim);
        let mut h_bwd = Array1::zeros(hidden_dim);
        let mut c_bwd = Array1::zeros(hidden_dim);
        
        let mut layer_input = x.clone();
        let mut all_hidden_states: Vec<Array1<f64>> = Vec::new();
        
        for layer_idx in 0..self.config.num_layers {
            // Forward direction
            let (h_f, c_f) = self.forward_layers[layer_idx].forward(&layer_input, &h_fwd, &c_fwd);
            h_fwd = h_f.clone();
            c_fwd = c_f;
            
            // Backward direction (same input for single timestep)
            let (h_b, c_b) = self.backward_layers[layer_idx].forward(&layer_input, &h_bwd, &c_bwd);
            h_bwd = h_b.clone();
            c_bwd = c_b;
            
            // Concatenate bidirectional hidden states
            let mut concat = Array1::zeros(hidden_dim * 2);
            concat.slice_mut(s![..hidden_dim]).assign(&h_fwd);
            concat.slice_mut(s![hidden_dim..]).assign(&h_bwd);
            
            all_hidden_states.push(concat.clone());
            
            // Residual connection
            if self.config.residual && layer_idx > 0 {
                layer_input = &concat + &layer_input.slice(s![..hidden_dim * 2]).to_owned();
            } else {
                layer_input = concat;
            }
            
            // Dropout would go here during training
        }
        
        // Apply attention over all layer outputs
        let query = all_hidden_states.last().unwrap();
        let (context, attention_weights) = self.attention.forward(
            query,
            &all_hidden_states,
            &all_hidden_states,
        );
        
        (context, attention_weights)
    }
}

impl NeuralPredictor for LSTMNetwork {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> String {
        format!(
            "Bidirectional LSTM with {} layers, {} hidden units, attention mechanism",
            self.config.num_layers,
            self.config.hidden_dim
        )
    }
    
    fn predict(&self, features: &NetworkFeatures) -> FailoverPrediction {
        let x = self.project_input(features);
        let (context, attention_weights) = self.forward_pass(&x);
        
        // Failover probability (sigmoid)
        let logits = self.output_linear.dot(&context) + &self.output_bias;
        let failover_prob = 1.0 / (1.0 + (-logits[0]).exp());
        
        // Path scores (softmax)
        let path_logits = self.path_linear.dot(&context) + &self.path_bias;
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
        
        // Time to failure prediction (log-normal parameterization)
        let time_params = self.time_linear.dot(&context) + &self.time_bias;
        let mu = time_params[0];
        let sigma = time_params[1].abs() + 0.1;
        let time_to_failure = (mu + 0.5 * sigma * sigma).exp();
        let ttf_lower = (mu - 1.96 * sigma).exp();
        let ttf_upper = (mu + 1.96 * sigma).exp();
        
        // Uncertainty estimation
        let uncertainty_logit = self.uncertainty_linear.dot(&context) + &self.uncertainty_bias;
        let uncertainty = 1.0 / (1.0 + (-uncertainty_logit[0]).exp());
        
        // Anomaly score based on reconstruction error (simplified)
        let anomaly_score = (1.0 - failover_prob).min(uncertainty);
        
        FailoverPrediction {
            failover_probability: failover_prob,
            time_to_failure_ms: time_to_failure.max(0.0).min(3600000.0),
            ttf_confidence: (ttf_lower.max(0.0), ttf_upper.min(3600000.0)),
            recommended_path,
            path_scores,
            uncertainty,
            attention_weights: Some(attention_weights),
            anomaly_score,
        }
    }
    
    fn update(&mut self, features: &NetworkFeatures, actual_failover: bool, _actual_path: usize) {
        // Store for online learning / calibration
        if self.recent_history.len() >= 1000 {
            self.recent_history.pop_front();
        }
        self.recent_history.push_back((features.clone(), actual_failover));
        
        // Online weight updates would go here
        // For now, we just maintain state
    }
    
    fn num_parameters(&self) -> usize {
        let lstm_params: usize = self.forward_layers.iter().map(|l| l.num_parameters()).sum::<usize>()
            + self.backward_layers.iter().map(|l| l.num_parameters()).sum::<usize>();
        let attention_params = self.attention.num_parameters();
        let output_params = self.output_linear.len() + self.output_bias.len();
        let path_params = self.path_linear.len() + self.path_bias.len();
        let time_params = self.time_linear.len() + self.time_bias.len();
        let uncertainty_params = self.uncertainty_linear.len() + self.uncertainty_bias.len();
        let input_params = self.input_linear.len() + self.input_bias.len();
        
        lstm_params + attention_params + output_params + path_params + time_params + uncertainty_params + input_params
    }
    
    fn inference_latency_us(&self) -> u64 {
        // Estimate based on model size
        (self.num_parameters() / 1000).max(100) as u64
    }
    
    fn reset(&mut self) {
        let hidden_dim = self.config.hidden_dim;
        self.h_states = vec![Array1::zeros(hidden_dim); self.config.num_layers * 2];
        self.c_states = vec![Array1::zeros(hidden_dim); self.config.num_layers * 2];
    }
    
    fn clone_box(&self) -> Box<dyn NeuralPredictor> {
        Box::new(self.clone())
    }
}
