//! Physical Layer Models
//!
//! Models radio propagation, interference, signal strength, and handover mechanics
//! based on 3GPP specifications and empirical data.

use rand::Rng;
use rand_distr::{Distribution, Normal, LogNormal, Exp, Uniform};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ============================================================================
// Radio Technologies
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RadioTechnology {
    /// 4G LTE (various bands)
    LTE,
    /// 5G NR Sub-6GHz
    NR5G,
    /// 5G NR mmWave
    NR5G_mmWave,
    /// WiFi 5 (802.11ac)
    WiFi5,
    /// WiFi 6 (802.11ax)
    WiFi6,
    /// WiFi 6E (6GHz)
    WiFi6E,
    /// Legacy 3G (for rural fallback)
    UMTS,
    /// LEO Satellite (Starlink, etc.)
    Satellite,
    /// Fiber optic
    Fiber,
}

impl RadioTechnology {
    /// Typical frequency band (MHz) for path loss calculations
    pub fn frequency_mhz(&self) -> f64 {
        match self {
            RadioTechnology::LTE => 1800.0,
            RadioTechnology::NR5G => 3500.0,
            RadioTechnology::NR5G_mmWave => 28000.0,
            RadioTechnology::WiFi5 => 5000.0,
            RadioTechnology::WiFi6 => 5000.0,
            RadioTechnology::WiFi6E => 6000.0,
            RadioTechnology::UMTS => 2100.0,
            RadioTechnology::Satellite => 12000.0,
            RadioTechnology::Fiber => 0.0, // Not applicable
        }
    }

    /// Maximum theoretical throughput (Mbps)
    pub fn max_throughput_mbps(&self) -> f64 {
        match self {
            RadioTechnology::LTE => 150.0,
            RadioTechnology::NR5G => 1000.0,
            RadioTechnology::Satellite => 300.0,
            RadioTechnology::Fiber => 10000.0,
            RadioTechnology::NR5G_mmWave => 4000.0,
            RadioTechnology::WiFi5 => 866.0,
            RadioTechnology::WiFi6 => 1200.0,
            RadioTechnology::WiFi6E => 2400.0,
            RadioTechnology::UMTS => 42.0,
        }
    }

    /// Typical handover time (ms)
    pub fn handover_time_ms(&self) -> (f64, f64) {
        match self {
            RadioTechnology::LTE => (50.0, 150.0),
            RadioTechnology::NR5G => (20.0, 80.0),
            RadioTechnology::NR5G_mmWave => (10.0, 50.0),
            RadioTechnology::WiFi5 => (100.0, 500.0),
            RadioTechnology::WiFi6 => (50.0, 200.0),
            RadioTechnology::WiFi6E => (50.0, 200.0),
            RadioTechnology::UMTS => (200.0, 500.0),
            RadioTechnology::Satellite => (200.0, 2000.0),
            RadioTechnology::Fiber => (1.0, 10.0),
        }
    }

    /// Minimum signal strength for connectivity (dBm)
    pub fn min_signal_dbm(&self) -> f64 {
        match self {
            RadioTechnology::LTE => -110.0,
            RadioTechnology::NR5G => -105.0,
            RadioTechnology::NR5G_mmWave => -95.0,
            RadioTechnology::WiFi5 => -80.0,
            RadioTechnology::WiFi6 => -82.0,
            RadioTechnology::WiFi6E => -78.0,
            RadioTechnology::UMTS => -105.0,
            RadioTechnology::Satellite => -120.0,
            RadioTechnology::Fiber => -30.0, // Optical power
        }
    }
}

// ============================================================================
// Path Loss Models
// ============================================================================

/// Path loss model types based on 3GPP TR 38.901
#[derive(Debug, Clone, Copy)]
pub enum PathLossModel {
    /// Free space path loss (ideal)
    FreeSpace,
    /// Urban macro cell (UMa)
    UrbanMacro,
    /// Urban micro cell (UMi) - street canyon
    UrbanMicro,
    /// Rural macro cell (RMa)
    RuralMacro,
    /// Indoor hotspot (InH)
    IndoorHotspot,
    /// Indoor-to-outdoor
    IndoorOutdoor,
    /// Vehicle penetration loss
    VehiclePenetration,
}

impl PathLossModel {
    /// Calculate path loss in dB
    /// 
    /// # Arguments
    /// * `distance_m` - Distance between transmitter and receiver in meters
    /// * `frequency_mhz` - Carrier frequency in MHz
    /// * `height_tx_m` - Transmitter height in meters
    /// * `height_rx_m` - Receiver height in meters
    pub fn calculate(&self, distance_m: f64, frequency_mhz: f64, height_tx_m: f64, height_rx_m: f64) -> f64 {
        let d = distance_m.max(1.0);
        let f = frequency_mhz;
        let h_bs = height_tx_m;
        let h_ut = height_rx_m;

        match self {
            PathLossModel::FreeSpace => {
                // Friis free space: PL = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
                20.0 * d.log10() + 20.0 * f.log10() - 27.55
            }
            PathLossModel::UrbanMacro => {
                // 3GPP UMa LOS model (simplified)
                let d_bp = 4.0 * (h_bs - 1.0) * (h_ut - 1.0) * f / 300.0;
                if d < d_bp {
                    22.0 * d.log10() + 28.0 + 20.0 * (f / 1000.0).log10()
                } else {
                    40.0 * d.log10() + 28.0 + 20.0 * (f / 1000.0).log10() 
                        - 9.0 * (d_bp.powi(2) + (h_bs - h_ut).powi(2)).log10()
                }
            }
            PathLossModel::UrbanMicro => {
                // 3GPP UMi street canyon
                32.4 + 21.0 * d.log10() + 20.0 * (f / 1000.0).log10()
            }
            PathLossModel::RuralMacro => {
                // 3GPP RMa model
                let d_bp = 2.0 * PI * h_bs * h_ut * f / 300.0;
                if d < d_bp {
                    20.0 * d.log10() + 20.0 * (f / 1000.0).log10() + 31.84
                } else {
                    20.0 * d.log10() + 20.0 * (f / 1000.0).log10() + 31.84 + 10.0
                }
            }
            PathLossModel::IndoorHotspot => {
                // Indoor office/shopping mall
                32.4 + 17.3 * d.log10() + 20.0 * (f / 1000.0).log10()
            }
            PathLossModel::IndoorOutdoor => {
                // Outdoor to indoor penetration
                let outdoor_loss = PathLossModel::UrbanMicro.calculate(d, f, h_bs, h_ut);
                let penetration_loss = 20.0 + 0.5 * (f / 1000.0); // Building penetration
                outdoor_loss + penetration_loss
            }
            PathLossModel::VehiclePenetration => {
                // Vehicle (train, car) penetration
                let outdoor_loss = PathLossModel::RuralMacro.calculate(d, f, h_bs, h_ut);
                let vehicle_loss = match f {
                    f if f < 1000.0 => 8.0,   // Low band: ~8 dB
                    f if f < 3000.0 => 15.0,  // Mid band: ~15 dB
                    _ => 25.0,                 // High band: ~25 dB
                };
                outdoor_loss + vehicle_loss
            }
        }
    }
}

// ============================================================================
// Shadow Fading
// ============================================================================

/// Shadow fading generator (log-normal distribution)
pub struct ShadowFading {
    /// Standard deviation in dB
    std_dev_db: f64,
    /// Correlation distance in meters
    correlation_distance_m: f64,
    /// Current fading value
    current_value: f64,
    /// Random generator
    rng: rand_chacha::ChaCha8Rng,
}

impl ShadowFading {
    pub fn new(std_dev_db: f64, correlation_distance_m: f64, seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            std_dev_db,
            correlation_distance_m,
            current_value: 0.0,
            rng: rand_chacha::ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Update fading based on movement
    pub fn update(&mut self, distance_moved_m: f64) -> f64 {
        let normal = Normal::new(0.0, self.std_dev_db).unwrap();
        let correlation = (-distance_moved_m / self.correlation_distance_m).exp();
        let innovation = normal.sample(&mut self.rng);
        
        self.current_value = correlation * self.current_value 
            + (1.0 - correlation * correlation).sqrt() * innovation;
        
        self.current_value
    }

    pub fn current(&self) -> f64 {
        self.current_value
    }
}

// ============================================================================
// Fast Fading (Small-scale)
// ============================================================================

/// Fast fading channel model
#[derive(Debug, Clone, Copy)]
pub enum FastFadingModel {
    /// Rayleigh fading (no LOS)
    Rayleigh,
    /// Rician fading (with LOS component)
    Rician { k_factor: f64 },
    /// Nakagami-m fading (generalized)
    Nakagami { m: f64 },
}

impl FastFadingModel {
    /// Generate a fading sample (linear power factor)
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            FastFadingModel::Rayleigh => {
                let exp = Exp::new(1.0).unwrap();
                exp.sample(rng)
            }
            FastFadingModel::Rician { k_factor } => {
                let k = *k_factor;
                let normal = Normal::new(0.0, 1.0).unwrap();
                let x = (k / (k + 1.0)).sqrt() + normal.sample(rng) / (2.0 * (k + 1.0)).sqrt();
                let y = normal.sample(rng) / (2.0 * (k + 1.0)).sqrt();
                x * x + y * y
            }
            FastFadingModel::Nakagami { m } => {
                // Approximation using gamma distribution
                let shape = *m;
                let scale = 1.0 / *m;
                let gamma = rand_distr::Gamma::new(shape, scale).unwrap();
                gamma.sample(rng)
            }
        }
    }

    /// Convert linear power to dB
    pub fn sample_db<R: Rng>(&self, rng: &mut R) -> f64 {
        10.0 * self.sample(rng).log10()
    }
}

// ============================================================================
// Interference Model
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterferenceLevel {
    None,
    Low,
    Medium,
    High,
    Severe,
    Variable,
}

impl InterferenceLevel {
    /// Get interference power in dB relative to noise floor
    pub fn to_db(&self) -> f64 {
        match self {
            InterferenceLevel::None => 0.0,
            InterferenceLevel::Low => 3.0,
            InterferenceLevel::Medium => 6.0,
            InterferenceLevel::High => 10.0,
            InterferenceLevel::Severe => 15.0,
            InterferenceLevel::Variable => 6.0, // Average
        }
    }

    /// Generate a sample for variable interference
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            InterferenceLevel::Variable => {
                let uniform = Uniform::new(0.0, 15.0);
                uniform.sample(rng)
            }
            _ => self.to_db(),
        }
    }
}

// ============================================================================
// Signal Quality Metrics
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SignalQuality {
    /// Received signal strength (dBm)
    pub rssi_dbm: f64,
    /// Signal-to-interference-plus-noise ratio (dB)
    pub sinr_db: f64,
    /// Reference signal received power (dBm) - for LTE/5G
    pub rsrp_dbm: f64,
    /// Reference signal received quality (dB) - for LTE/5G
    pub rsrq_db: f64,
    /// Channel quality indicator (0-15) - for LTE/5G
    pub cqi: u8,
}

impl SignalQuality {
    /// Calculate from physical parameters
    pub fn calculate(
        tx_power_dbm: f64,
        path_loss_db: f64,
        shadow_fading_db: f64,
        fast_fading_db: f64,
        noise_floor_dbm: f64,
        interference_db: f64,
    ) -> Self {
        let rssi_dbm = tx_power_dbm - path_loss_db + shadow_fading_db + fast_fading_db;
        let noise_plus_interference_dbm = 10.0 * (
            10_f64.powf(noise_floor_dbm / 10.0) + 
            10_f64.powf((noise_floor_dbm + interference_db) / 10.0)
        ).log10();
        let sinr_db = rssi_dbm - noise_plus_interference_dbm;
        
        // RSRP is typically 10-15 dB below RSSI
        let rsrp_dbm = rssi_dbm - 12.0;
        
        // RSRQ depends on resource block utilization
        let rsrq_db = (sinr_db - 10.0).max(-20.0).min(-3.0);
        
        // CQI mapping (simplified)
        let cqi = match sinr_db {
            s if s < -6.0 => 0,
            s if s < -4.0 => 1,
            s if s < -2.0 => 2,
            s if s < 0.0 => 3,
            s if s < 2.0 => 4,
            s if s < 4.0 => 5,
            s if s < 6.0 => 6,
            s if s < 8.0 => 7,
            s if s < 10.0 => 8,
            s if s < 12.0 => 9,
            s if s < 14.0 => 10,
            s if s < 16.0 => 11,
            s if s < 18.0 => 12,
            s if s < 20.0 => 13,
            s if s < 22.0 => 14,
            _ => 15,
        };

        Self {
            rssi_dbm,
            sinr_db,
            rsrp_dbm,
            rsrq_db,
            cqi,
        }
    }

    /// Estimate achievable throughput (Mbps) based on signal quality and technology
    pub fn estimate_throughput(&self, technology: RadioTechnology) -> f64 {
        let max_throughput = technology.max_throughput_mbps();
        
        // Shannon capacity approximation with practical efficiency
        let spectral_efficiency = if self.sinr_db > 0.0 {
            (1.0 + 10_f64.powf(self.sinr_db / 10.0)).log2()
        } else {
            0.1
        };
        
        // Practical efficiency factor (accounts for overhead, retransmissions)
        let efficiency_factor = 0.7;
        
        // CQI-based scaling
        let cqi_factor = self.cqi as f64 / 15.0;
        
        (max_throughput * spectral_efficiency * efficiency_factor * cqi_factor / 6.0)
            .min(max_throughput)
            .max(0.1)
    }

    /// Estimate packet loss rate based on signal quality
    pub fn estimate_packet_loss(&self) -> f64 {
        // Block error rate approximation
        let bler = match self.sinr_db {
            s if s > 20.0 => 0.0001,
            s if s > 15.0 => 0.001,
            s if s > 10.0 => 0.005,
            s if s > 5.0 => 0.01,
            s if s > 0.0 => 0.03,
            s if s > -5.0 => 0.10,
            _ => 0.30,
        };
        
        // Packet loss is higher due to multiple block errors per packet
        (1.0_f64 - (1.0_f64 - bler).powi(10)).min(1.0_f64)
    }
}

// ============================================================================
// Handover Model
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoverType {
    /// Intra-frequency handover (same band)
    IntraFrequency,
    /// Inter-frequency handover (different band, same technology)
    InterFrequency,
    /// Inter-RAT handover (different technology, e.g., 5G to LTE)
    InterRAT,
    /// Inter-operator handover (roaming)
    InterOperator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoverState {
    /// Normal connected state
    Connected,
    /// Handover preparation phase
    Preparing,
    /// Handover execution (brief outage)
    Executing,
    /// Handover completion
    Completing,
    /// Handover failed, reconnecting
    Failed,
}

pub struct HandoverModel {
    /// Current state
    state: HandoverState,
    /// Time in current state (ms)
    state_time_ms: f64,
    /// Handover type
    handover_type: HandoverType,
    /// Technology
    technology: RadioTechnology,
}

impl HandoverModel {
    pub fn new(technology: RadioTechnology) -> Self {
        Self {
            state: HandoverState::Connected,
            state_time_ms: 0.0,
            handover_type: HandoverType::IntraFrequency,
            technology,
        }
    }

    /// Trigger a handover
    pub fn trigger(&mut self, handover_type: HandoverType) {
        self.state = HandoverState::Preparing;
        self.state_time_ms = 0.0;
        self.handover_type = handover_type;
    }

    /// Update handover state (returns packet loss factor 0.0-1.0)
    pub fn update<R: Rng>(&mut self, delta_ms: f64, rng: &mut R) -> f64 {
        self.state_time_ms += delta_ms;

        let (prep_time, exec_time, complete_time, failure_prob) = match self.handover_type {
            HandoverType::IntraFrequency => {
                let (min, max) = self.technology.handover_time_ms();
                (min * 0.3, min * 0.4, min * 0.3, 0.01)
            }
            HandoverType::InterFrequency => {
                let (min, max) = self.technology.handover_time_ms();
                (max * 0.3, max * 0.5, max * 0.2, 0.02)
            }
            HandoverType::InterRAT => {
                (100.0, 200.0, 100.0, 0.05)
            }
            HandoverType::InterOperator => {
                (500.0, 2000.0, 500.0, 0.10)
            }
        };

        match self.state {
            HandoverState::Connected => 0.0,
            HandoverState::Preparing => {
                if self.state_time_ms >= prep_time {
                    self.state = HandoverState::Executing;
                    self.state_time_ms = 0.0;
                }
                0.0 // No loss during preparation
            }
            HandoverState::Executing => {
                if self.state_time_ms >= exec_time {
                    if rng.gen::<f64>() < failure_prob {
                        self.state = HandoverState::Failed;
                    } else {
                        self.state = HandoverState::Completing;
                    }
                    self.state_time_ms = 0.0;
                }
                1.0 // Complete loss during execution
            }
            HandoverState::Completing => {
                if self.state_time_ms >= complete_time {
                    self.state = HandoverState::Connected;
                    self.state_time_ms = 0.0;
                }
                0.1 // Some loss during completion
            }
            HandoverState::Failed => {
                // Recovery takes longer
                if self.state_time_ms >= exec_time * 3.0 {
                    self.state = HandoverState::Connected;
                    self.state_time_ms = 0.0;
                }
                1.0 // Complete loss during failure recovery
            }
        }
    }

    pub fn state(&self) -> HandoverState {
        self.state
    }

    pub fn is_connected(&self) -> bool {
        matches!(self.state, HandoverState::Connected)
    }
}

// ============================================================================
// Radio Environment
// ============================================================================

/// Complete radio environment simulation
pub struct RadioEnvironment {
    pub technology: RadioTechnology,
    pub path_loss_model: PathLossModel,
    pub shadow_fading: ShadowFading,
    pub fast_fading_model: FastFadingModel,
    pub interference_level: InterferenceLevel,
    pub handover_model: HandoverModel,
    
    // Physical parameters
    pub tx_power_dbm: f64,
    pub noise_floor_dbm: f64,
    pub distance_m: f64,
    pub height_tx_m: f64,
    pub height_rx_m: f64,
    
    // Current state
    pub signal_quality: SignalQuality,
}

impl RadioEnvironment {
    pub fn new(
        technology: RadioTechnology,
        path_loss_model: PathLossModel,
        interference_level: InterferenceLevel,
        seed: u64,
    ) -> Self {
        let shadow_std_dev = match path_loss_model {
            PathLossModel::UrbanMacro => 4.0,
            PathLossModel::UrbanMicro => 3.0,
            PathLossModel::RuralMacro => 6.0,
            PathLossModel::IndoorHotspot => 3.0,
            _ => 4.0,
        };

        Self {
            technology,
            path_loss_model,
            shadow_fading: ShadowFading::new(shadow_std_dev, 50.0, seed),
            fast_fading_model: FastFadingModel::Rician { k_factor: 3.0 },
            interference_level,
            handover_model: HandoverModel::new(technology),
            tx_power_dbm: 43.0, // Typical macro cell
            noise_floor_dbm: -104.0,
            distance_m: 500.0,
            height_tx_m: 25.0,
            height_rx_m: 1.5,
            signal_quality: SignalQuality {
                rssi_dbm: -80.0,
                sinr_db: 10.0,
                rsrp_dbm: -92.0,
                rsrq_db: -10.0,
                cqi: 10,
            },
        }
    }

    /// Update environment state
    pub fn update<R: Rng>(&mut self, delta_ms: f64, distance_moved_m: f64, rng: &mut R) {
        // Update fading
        let shadow_db = self.shadow_fading.update(distance_moved_m);
        let fast_db = self.fast_fading_model.sample_db(rng);
        let interference_db = self.interference_level.sample(rng);

        // Calculate path loss
        let path_loss_db = self.path_loss_model.calculate(
            self.distance_m,
            self.technology.frequency_mhz(),
            self.height_tx_m,
            self.height_rx_m,
        );

        // Update signal quality
        self.signal_quality = SignalQuality::calculate(
            self.tx_power_dbm,
            path_loss_db,
            shadow_db,
            fast_db,
            self.noise_floor_dbm,
            interference_db,
        );
    }

    /// Get current achievable throughput
    pub fn throughput_mbps(&self) -> f64 {
        self.signal_quality.estimate_throughput(self.technology)
    }

    /// Get current packet loss rate
    pub fn packet_loss_rate(&self) -> f64 {
        self.signal_quality.estimate_packet_loss()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_loss_models() {
        let models = [
            PathLossModel::FreeSpace,
            PathLossModel::UrbanMacro,
            PathLossModel::UrbanMicro,
            PathLossModel::RuralMacro,
        ];

        for model in &models {
            let loss = model.calculate(1000.0, 3500.0, 25.0, 1.5);
            assert!(loss > 0.0, "Path loss should be positive");
            assert!(loss < 200.0, "Path loss should be reasonable");
        }
    }

    #[test]
    fn test_signal_quality() {
        let sq = SignalQuality::calculate(43.0, 100.0, 0.0, 0.0, -104.0, 0.0);
        assert!(sq.rssi_dbm > -100.0 && sq.rssi_dbm < 0.0);
        assert!(sq.cqi <= 15);
    }

    #[test]
    fn test_handover() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut ho = HandoverModel::new(RadioTechnology::LTE);
        
        ho.trigger(HandoverType::IntraFrequency);
        assert!(!ho.is_connected());
        
        // Simulate time passing
        for _ in 0..100 {
            ho.update(10.0, &mut rng);
        }
        
        // Should eventually reconnect
        assert!(ho.is_connected());
    }
}
