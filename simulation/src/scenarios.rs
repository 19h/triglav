//! Scenario Generators
//!
//! Comprehensive real-world scenario models including:
//! - Train journey with WiFi + tethering
//! - Stationary with multiple uplinks
//! - Cross-border driving
//! - Infrastructure failures
//! - Urban mobility

use std::ops::Range;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform, Exp};
use serde::{Deserialize, Serialize};

use crate::physical::*;
use crate::network::*;

// ============================================================================
// Common Types
// ============================================================================

pub use crate::physical::{RadioTechnology, InterferenceLevel};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UplinkConfig {
    pub technology: RadioTechnology,
    pub provider: String,
    pub signal_strength_dbm: f64,
    pub interference_level: InterferenceLevel,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LocationType {
    DenseUrban,
    Urban,
    Suburban,
    Rural,
    Highway,
    Tunnel,
    Industrial,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DevicePosition {
    Window,
    Aisle,
    Middle,
    Interior,
    Pocket,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BackhaulType {
    LTE,
    NR5G,
    Satellite,
    Fiber,
    WiFi,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MovementType {
    Stationary,
    Walking,
    Running,
    Cycling,
    Scooter,
    Bus,
    Tram,
    Driving,
    Train,
    Flying,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AreaType {
    DenseUrban,
    Urban,
    Suburban,
    Rural,
    Industrial,
    Park,
    ShoppingMall,
}



// ============================================================================
// Scenario Trait
// ============================================================================

/// A simulation scenario that generates network events over time
pub trait Scenario: Send + Sync {
    /// Scenario identifier
    fn id(&self) -> &str;
    
    /// Human-readable description
    fn description(&self) -> String;
    
    /// Initialize scenario state
    fn init(&mut self, seed: u64);
    
    /// Get available network links at current time
    fn get_links(&self) -> Vec<&NetworkLink>;
    
    /// Get mutable links for updates
    fn get_links_mut(&mut self) -> Vec<&mut NetworkLink>;
    
    /// Advance simulation by delta_ms milliseconds (uses boxed RNG for dyn compatibility)
    fn step(&mut self, delta_ms: f64, rng: &mut dyn RngCore);
    
    /// Check if scenario is complete
    fn is_complete(&self) -> bool;
    
    /// Get current simulation time (ms)
    fn current_time_ms(&self) -> f64;
    
    /// Get scenario duration (ms)
    fn duration_ms(&self) -> f64;
    
    /// Get scenario-specific events that occurred
    fn get_events(&self) -> Vec<ScenarioEvent>;
    
    /// Reset scenario to initial state
    fn reset(&mut self);
}

// Re-export RngCore for implementations
pub use rand::RngCore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioEvent {
    pub time_ms: f64,
    pub event_type: EventType,
    pub description: String,
    pub affected_links: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    CoverageGap,
    CoverageRestored,
    Handover,
    HandoverFailed,
    SignalDegraded,
    SignalImproved,
    InfrastructureFailure,
    InfrastructureRecovered,
    BorderCrossing,
    OperatorChange,
    TunnelEntry,
    TunnelExit,
    Congestion,
    CongestionCleared,
}

// ============================================================================
// Train Journey Scenario
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainJourneyConfig {
    pub route_km: f64,
    pub speed_kmh: f64,
    pub tunnel_count: usize,
    pub avg_tunnel_length_km: f64,
    pub coverage_gap_probability: f64,
    pub coverage_gap_duration_s: Range<f64>,
    pub train_wifi_config: TrainWiFiConfig,
    pub tethering_config: TetheringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainWiFiConfig {
    pub backhaul_type: BackhaulType,
    pub passenger_load: f64,
    pub contention_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetheringConfig {
    pub technology: RadioTechnology,
    pub device_position: DevicePosition,
}

pub struct TrainJourneyScenario {
    id: String,
    config: TrainJourneyConfig,
    
    // State
    current_time_ms: f64,
    current_position_km: f64,
    in_tunnel: bool,
    in_coverage_gap: bool,
    coverage_gap_end_km: f64,
    
    // Tunnel positions (km)
    tunnels: Vec<(f64, f64)>, // (start_km, end_km)
    
    // Links
    wifi_link: NetworkLink,
    tethering_link: NetworkLink,
    
    // Events
    events: Vec<ScenarioEvent>,
}

impl TrainJourneyScenario {
    pub fn new(id: &str, config: TrainJourneyConfig) -> Self {
        // Create WiFi link (train's backhaul)
        let wifi_path = NetworkPath::new(
            "train_wifi".to_string(),
            RadioTechnology::WiFi6,
            LatencyDistribution::Bimodal {
                low_mean_ms: 30.0,  // Good conditions
                low_std_ms: 10.0,
                high_mean_ms: 200.0, // Congested
                high_std_ms: 100.0,
                high_probability: config.train_wifi_config.passenger_load * 0.3,
            },
            LossPattern::Markov {
                base_loss: 0.01 * config.train_wifi_config.contention_factor,
                correlation: 0.4,
            },
            JitterModel {
                base_jitter_ms: 15.0,
                correlation: 0.3,
                max_jitter_ms: 200.0,
                spike_probability: 0.05,
                spike_multiplier: 5.0,
            },
            BandwidthModel::new(50.0, 1.0, BandwidthVariation::TimeOfDay {
                peak_reduction_factor: 0.5,
            }),
        );

        // Create tethering link (5G/LTE)
        let tethering_latency = match config.tethering_config.technology {
            RadioTechnology::NR5G => LatencyDistribution::typical_5g(),
            RadioTechnology::LTE => LatencyDistribution::typical_lte(),
            _ => LatencyDistribution::typical_lte(),
        };

        let device_loss_factor = match config.tethering_config.device_position {
            DevicePosition::Window => 1.0,
            DevicePosition::Aisle => 1.5,
            DevicePosition::Middle => 2.0,
            DevicePosition::Interior => 2.5,
            DevicePosition::Pocket => 3.0,
        };

        let tethering_path = NetworkPath::new(
            "tethering_5g".to_string(),
            config.tethering_config.technology,
            tethering_latency,
            LossPattern::GilbertElliott {
                p_good_to_bad: 0.02 * device_loss_factor,
                p_bad_to_good: 0.1,
                loss_in_good: 0.005,
                loss_in_bad: 0.3,
            },
            JitterModel::typical_5g(),
            BandwidthModel::new(200.0, 5.0, BandwidthVariation::SignalDependent),
        );

        Self {
            id: id.to_string(),
            config,
            current_time_ms: 0.0,
            current_position_km: 0.0,
            in_tunnel: false,
            in_coverage_gap: false,
            coverage_gap_end_km: 0.0,
            tunnels: Vec::new(),
            wifi_link: NetworkLink::new(
                "train_wifi".to_string(),
                wifi_path,
                BackboneModel::new(8),
                "train_operator".to_string(),
            ),
            tethering_link: NetworkLink::new(
                "tethering_5g".to_string(),
                tethering_path,
                BackboneModel::new(5),
                "mobile_operator".to_string(),
            ),
            events: Vec::new(),
        }
    }

    fn generate_tunnels(&mut self, rng: &mut impl Rng) {
        self.tunnels.clear();
        let mut position = 20.0; // First tunnel after 20km
        
        for _ in 0..self.config.tunnel_count {
            let gap = Uniform::new(20.0, 80.0).sample(rng);
            let start = position + gap;
            let length = Normal::new(self.config.avg_tunnel_length_km, 1.0)
                .unwrap()
                .sample(rng)
                .max(0.5);
            let end = start + length;
            
            if end < self.config.route_km {
                self.tunnels.push((start, end));
                position = end;
            }
        }
    }

    fn check_tunnel_status(&self, position_km: f64) -> bool {
        for (start, end) in &self.tunnels {
            if position_km >= *start && position_km <= *end {
                return true;
            }
        }
        false
    }
}

impl Scenario for TrainJourneyScenario {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!(
            "Train journey: {}km at {}km/h with {} tunnels, WiFi ({:?}) + tethering ({:?})",
            self.config.route_km,
            self.config.speed_kmh,
            self.config.tunnel_count,
            self.config.train_wifi_config.backhaul_type,
            self.config.tethering_config.technology
        )
    }

    fn init(&mut self, seed: u64) {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        self.generate_tunnels(&mut rng);
        self.current_time_ms = 0.0;
        self.current_position_km = 0.0;
        self.events.clear();
    }

    fn get_links(&self) -> Vec<&NetworkLink> {
        vec![&self.wifi_link, &self.tethering_link]
    }

    fn get_links_mut(&mut self) -> Vec<&mut NetworkLink> {
        vec![&mut self.wifi_link, &mut self.tethering_link]
    }

    fn step(&mut self, delta_ms: f64, rng: &mut dyn RngCore) {
        self.current_time_ms += delta_ms;
        
        // Update position
        let delta_hours = delta_ms / 3_600_000.0;
        let delta_km = self.config.speed_kmh * delta_hours;
        self.current_position_km += delta_km;

        // Check tunnel status
        let was_in_tunnel = self.in_tunnel;
        self.in_tunnel = self.check_tunnel_status(self.current_position_km);

        if self.in_tunnel && !was_in_tunnel {
            // Entering tunnel
            self.events.push(ScenarioEvent {
                time_ms: self.current_time_ms,
                event_type: EventType::TunnelEntry,
                description: format!("Entering tunnel at {:.1}km", self.current_position_km),
                affected_links: vec!["tethering_5g".to_string()],
            });
            
            // Tethering goes down in tunnel
            self.tethering_link.path.is_active = false;
        } else if !self.in_tunnel && was_in_tunnel {
            // Exiting tunnel
            self.events.push(ScenarioEvent {
                time_ms: self.current_time_ms,
                event_type: EventType::TunnelExit,
                description: format!("Exiting tunnel at {:.1}km", self.current_position_km),
                affected_links: vec!["tethering_5g".to_string()],
            });
            
            // Tethering recovers (with delay for handover)
            // Simulate handover delay
            self.tethering_link.path.is_active = true;
        }

        // Random coverage gaps for tethering
        if !self.in_tunnel && !self.in_coverage_gap {
            let gap_prob = self.config.coverage_gap_probability * delta_ms / 60_000.0;
            if rng.gen::<f64>() < gap_prob {
                self.in_coverage_gap = true;
                let gap_duration_s = Uniform::new(
                    self.config.coverage_gap_duration_s.start,
                    self.config.coverage_gap_duration_s.end,
                ).sample(rng);
                let gap_km = gap_duration_s * self.config.speed_kmh / 3600.0;
                self.coverage_gap_end_km = self.current_position_km + gap_km;

                self.events.push(ScenarioEvent {
                    time_ms: self.current_time_ms,
                    event_type: EventType::CoverageGap,
                    description: format!("Coverage gap at {:.1}km, duration {:.0}s", 
                        self.current_position_km, gap_duration_s),
                    affected_links: vec!["tethering_5g".to_string()],
                });

                self.tethering_link.path.is_active = false;
            }
        } else if self.in_coverage_gap && self.current_position_km >= self.coverage_gap_end_km {
            self.in_coverage_gap = false;
            self.tethering_link.path.is_active = true;

            self.events.push(ScenarioEvent {
                time_ms: self.current_time_ms,
                event_type: EventType::CoverageRestored,
                description: format!("Coverage restored at {:.1}km", self.current_position_km),
                affected_links: vec!["tethering_5g".to_string()],
            });
        }

        // Degraded signal quality based on speed and position
        let signal_degradation = if self.in_tunnel {
            30.0 // Heavy attenuation
        } else if self.in_coverage_gap {
            40.0
        } else {
            // Speed-dependent degradation (Doppler effect)
            (self.config.speed_kmh / 100.0) * 5.0
        };

        self.tethering_link.path.loss_state.current_sinr_db = 15.0 - signal_degradation;
    }

    fn is_complete(&self) -> bool {
        self.current_position_km >= self.config.route_km
    }

    fn current_time_ms(&self) -> f64 {
        self.current_time_ms
    }

    fn duration_ms(&self) -> f64 {
        (self.config.route_km / self.config.speed_kmh) * 3_600_000.0
    }

    fn get_events(&self) -> Vec<ScenarioEvent> {
        self.events.clone()
    }

    fn reset(&mut self) {
        self.current_time_ms = 0.0;
        self.current_position_km = 0.0;
        self.in_tunnel = false;
        self.in_coverage_gap = false;
        self.events.clear();
        self.wifi_link.path.is_active = true;
        self.tethering_link.path.is_active = true;
    }
}

// ============================================================================
// Stationary Dual Uplink Scenario
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationaryConfig {
    pub uplinks: Vec<UplinkConfig>,
    pub location_type: LocationType,
    pub time_of_day_variation: bool,
}

pub struct StationaryDualUplinkScenario {
    id: String,
    config: StationaryConfig,
    current_time_ms: f64,
    duration_ms: f64,
    links: Vec<NetworkLink>,
    events: Vec<ScenarioEvent>,
}

impl StationaryDualUplinkScenario {
    pub fn new(id: &str, config: StationaryConfig) -> Self {
        let links = config.uplinks.iter().enumerate().map(|(i, uplink)| {
            let latency = match uplink.technology {
                RadioTechnology::NR5G | RadioTechnology::NR5G_mmWave => {
                    LatencyDistribution::typical_5g()
                }
                RadioTechnology::LTE => LatencyDistribution::typical_lte(),
                RadioTechnology::WiFi5 | RadioTechnology::WiFi6 | RadioTechnology::WiFi6E => {
                    LatencyDistribution::typical_wifi()
                }
                _ => LatencyDistribution::typical_lte(),
            };

            let loss_pattern = match uplink.interference_level {
                InterferenceLevel::None | InterferenceLevel::Low => {
                    LossPattern::Random { probability: 0.001 }
                }
                InterferenceLevel::Medium => {
                    LossPattern::Markov { base_loss: 0.005, correlation: 0.3 }
                }
                InterferenceLevel::High | InterferenceLevel::Severe => {
                    LossPattern::GilbertElliott {
                        p_good_to_bad: 0.05,
                        p_bad_to_good: 0.1,
                        loss_in_good: 0.01,
                        loss_in_bad: 0.2,
                    }
                }
                InterferenceLevel::Variable => {
                    LossPattern::GilbertElliott {
                        p_good_to_bad: 0.03,
                        p_bad_to_good: 0.15,
                        loss_in_good: 0.005,
                        loss_in_bad: 0.15,
                    }
                }
            };

            let path = NetworkPath::new(
                format!("uplink_{}", i),
                uplink.technology,
                latency,
                loss_pattern,
                JitterModel::new(5.0),
                BandwidthModel::new(
                    uplink.technology.max_throughput_mbps() * 0.5,
                    1.0,
                    if config.time_of_day_variation {
                        BandwidthVariation::TimeOfDay { peak_reduction_factor: 0.3 }
                    } else {
                        BandwidthVariation::Constant
                    },
                ),
            );

            NetworkLink::new(
                format!("uplink_{}", i),
                path,
                BackboneModel::new(6),
                uplink.provider.clone(),
            )
        }).collect();

        Self {
            id: id.to_string(),
            config,
            current_time_ms: 0.0,
            duration_ms: 3_600_000.0, // 1 hour default
            links,
            events: Vec::new(),
        }
    }
}

impl Scenario for StationaryDualUplinkScenario {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        let uplink_desc: Vec<String> = self.config.uplinks.iter()
            .map(|u| format!("{:?}", u.technology))
            .collect();
        format!("Stationary: {:?} location with {} uplinks ({})",
            self.config.location_type,
            self.config.uplinks.len(),
            uplink_desc.join(", "))
    }

    fn init(&mut self, _seed: u64) {
        self.current_time_ms = 0.0;
        self.events.clear();
    }

    fn get_links(&self) -> Vec<&NetworkLink> {
        self.links.iter().collect()
    }

    fn get_links_mut(&mut self) -> Vec<&mut NetworkLink> {
        self.links.iter_mut().collect()
    }

    fn step(&mut self, delta_ms: f64, rng: &mut dyn RngCore) {
        self.current_time_ms += delta_ms;

        // Time of day (hours, 0-24)
        let time_of_day = (self.current_time_ms / 3_600_000.0) % 24.0;

        // Update each link
        for link in &mut self.links {
            // Update bandwidth based on time of day
            link.path.bandwidth_model.update(rng, time_of_day, None, false);

            // Occasional interference spikes (e.g., microwave, other WiFi)
            if matches!(link.path.technology, RadioTechnology::WiFi5 | RadioTechnology::WiFi6 | RadioTechnology::WiFi6E) {
                if rng.gen::<f64>() < 0.0001 { // ~0.36 events per hour
                    self.events.push(ScenarioEvent {
                        time_ms: self.current_time_ms,
                        event_type: EventType::SignalDegraded,
                        description: "WiFi interference spike".to_string(),
                        affected_links: vec![link.id.clone()],
                    });
                }
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.current_time_ms >= self.duration_ms
    }

    fn current_time_ms(&self) -> f64 {
        self.current_time_ms
    }

    fn duration_ms(&self) -> f64 {
        self.duration_ms
    }

    fn get_events(&self) -> Vec<ScenarioEvent> {
        self.events.clone()
    }

    fn reset(&mut self) {
        self.current_time_ms = 0.0;
        self.events.clear();
        for link in &mut self.links {
            link.path.is_active = true;
        }
    }
}

// ============================================================================
// Infrastructure Failure Scenario
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FailureType {
    InternetExchange,
    BackboneLink,
    BackboneFiber,
    DNSServer,
    DNS,
    ProviderOutage,
    DDOSAttack,
    CellTower,
    PowerOutage,
    SatelliteConstellation,
    CDN,
    Cascading,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfrastructureFailureConfig {
    pub failure_type: FailureType,
    pub affected_paths: Vec<String>,
    pub failure_duration_s: Range<f64>,
    pub failure_probability_per_hour: f64,
    pub graceful_degradation: bool,
    pub bgp_reconvergence_time_s: Range<f64>,
}

pub struct InfrastructureFailureScenario {
    id: String,
    config: InfrastructureFailureConfig,
    current_time_ms: f64,
    duration_ms: f64,
    
    // Failure state
    failure_active: bool,
    failure_end_ms: f64,
    degradation_factor: f64,
    
    links: Vec<NetworkLink>,
    events: Vec<ScenarioEvent>,
}

impl InfrastructureFailureScenario {
    pub fn new(id: &str, config: InfrastructureFailureConfig) -> Self {
        // Create two links - one affected, one not
        let affected_link = NetworkLink::new(
            "primary_path".to_string(),
            NetworkPath::typical_5g("primary".to_string()),
            BackboneModel::new(8),
            "primary_provider".to_string(),
        );

        let backup_link = NetworkLink::new(
            "backup_path".to_string(),
            NetworkPath::typical_lte("backup".to_string()),
            BackboneModel::new(10), // Longer path
            "backup_provider".to_string(),
        );

        Self {
            id: id.to_string(),
            config,
            current_time_ms: 0.0,
            duration_ms: 3_600_000.0,
            failure_active: false,
            failure_end_ms: 0.0,
            degradation_factor: 1.0,
            links: vec![affected_link, backup_link],
            events: Vec::new(),
        }
    }
}

impl Scenario for InfrastructureFailureScenario {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!("Infrastructure failure: {:?}, probability {:.3}/hour",
            self.config.failure_type,
            self.config.failure_probability_per_hour)
    }

    fn init(&mut self, _seed: u64) {
        self.current_time_ms = 0.0;
        self.failure_active = false;
        self.events.clear();
    }

    fn get_links(&self) -> Vec<&NetworkLink> {
        self.links.iter().collect()
    }

    fn get_links_mut(&mut self) -> Vec<&mut NetworkLink> {
        self.links.iter_mut().collect()
    }

    fn step(&mut self, delta_ms: f64, rng: &mut dyn RngCore) {
        self.current_time_ms += delta_ms;

        // Check for new failure
        if !self.failure_active {
            let failure_prob = self.config.failure_probability_per_hour * delta_ms / 3_600_000.0;
            if rng.gen::<f64>() < failure_prob {
                self.failure_active = true;
                let duration_s = Uniform::new(
                    self.config.failure_duration_s.start,
                    self.config.failure_duration_s.end,
                ).sample(rng);
                self.failure_end_ms = self.current_time_ms + duration_s * 1000.0;

                self.events.push(ScenarioEvent {
                    time_ms: self.current_time_ms,
                    event_type: EventType::InfrastructureFailure,
                    description: format!("{:?} failure, expected duration {:.0}s",
                        self.config.failure_type, duration_s),
                    affected_links: vec!["primary_path".to_string()],
                });

                // Affect primary link
                if self.config.graceful_degradation {
                    // Gradual degradation
                    self.degradation_factor = 0.3;
                } else {
                    // Complete failure
                    self.links[0].path.is_active = false;
                }

                // BGP reconvergence affects both paths initially
                self.links[0].backbone.is_converged = false;
                self.links[0].backbone.convergence_timer_ms = 
                    Uniform::new(
                        self.config.bgp_reconvergence_time_s.start * 1000.0,
                        self.config.bgp_reconvergence_time_s.end * 1000.0,
                    ).sample(rng);
            }
        } else if self.current_time_ms >= self.failure_end_ms {
            // Recovery
            self.failure_active = false;
            self.degradation_factor = 1.0;
            self.links[0].path.is_active = true;

            self.events.push(ScenarioEvent {
                time_ms: self.current_time_ms,
                event_type: EventType::InfrastructureRecovered,
                description: format!("{:?} recovered", self.config.failure_type),
                affected_links: vec!["primary_path".to_string()],
            });
        }
    }

    fn is_complete(&self) -> bool {
        self.current_time_ms >= self.duration_ms
    }

    fn current_time_ms(&self) -> f64 {
        self.current_time_ms
    }

    fn duration_ms(&self) -> f64 {
        self.duration_ms
    }

    fn get_events(&self) -> Vec<ScenarioEvent> {
        self.events.clone()
    }

    fn reset(&mut self) {
        self.current_time_ms = 0.0;
        self.failure_active = false;
        self.degradation_factor = 1.0;
        self.events.clear();
        for link in &mut self.links {
            link.path.is_active = true;
            link.backbone.is_converged = true;
        }
    }
}

// ============================================================================
// Cross-Border Driving Scenario
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrivingConfig {
    pub route_km: f64,
    pub avg_speed_kmh: f64,
    pub border_crossing_km: f64,
    pub uplink: UplinkConfig,
    pub operator_handover_config: OperatorHandoverConfig,
    pub rural_coverage_gaps: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorHandoverConfig {
    pub polish_operators: Vec<String>,
    pub german_operators: Vec<String>,
    pub handover_duration_s: Range<f64>,
    pub registration_failure_probability: f64,
}

pub struct DrivingCrossBorderScenario {
    id: String,
    config: DrivingConfig,
    current_time_ms: f64,
    current_position_km: f64,
    
    // Border state
    in_poland: bool,
    border_transition_active: bool,
    border_transition_end_ms: f64,
    current_operator: String,
    
    // Coverage
    in_coverage_gap: bool,
    coverage_gap_end_km: f64,
    
    link: NetworkLink,
    events: Vec<ScenarioEvent>,
}

impl DrivingCrossBorderScenario {
    pub fn new(id: &str, config: DrivingConfig) -> Self {
        let path = NetworkPath::new(
            "mobile".to_string(),
            config.uplink.technology,
            LatencyDistribution::typical_lte(), // Rural areas often fall back to LTE
            LossPattern::GilbertElliott {
                p_good_to_bad: 0.03,
                p_bad_to_good: 0.08,
                loss_in_good: 0.005,
                loss_in_bad: 0.25,
            },
            JitterModel::typical_lte(),
            BandwidthModel::new(50.0, 0.5, BandwidthVariation::SignalDependent),
        );

        let link = NetworkLink::new(
            "mobile".to_string(),
            path,
            BackboneModel::new(7),
            "roaming".to_string(),
        );

        Self {
            id: id.to_string(),
            config,
            current_time_ms: 0.0,
            current_position_km: 0.0,
            in_poland: true,
            border_transition_active: false,
            border_transition_end_ms: 0.0,
            current_operator: "play".to_string(), // Default Polish operator
            in_coverage_gap: false,
            coverage_gap_end_km: 0.0,
            link,
            events: Vec::new(),
        }
    }
}

impl Scenario for DrivingCrossBorderScenario {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!("Cross-border drive: {}km, border at {}km, {:?}",
            self.config.route_km,
            self.config.border_crossing_km,
            self.config.uplink.technology)
    }

    fn init(&mut self, _seed: u64) {
        self.current_time_ms = 0.0;
        self.current_position_km = 0.0;
        self.in_poland = true;
        self.border_transition_active = false;
        self.current_operator = "play".to_string();
        self.events.clear();
    }

    fn get_links(&self) -> Vec<&NetworkLink> {
        vec![&self.link]
    }

    fn get_links_mut(&mut self) -> Vec<&mut NetworkLink> {
        vec![&mut self.link]
    }

    fn step(&mut self, delta_ms: f64, rng: &mut dyn RngCore) {
        self.current_time_ms += delta_ms;

        // Update position
        let delta_hours = delta_ms / 3_600_000.0;
        let delta_km = self.config.avg_speed_kmh * delta_hours;
        self.current_position_km += delta_km;

        // Check for border crossing
        if self.in_poland && self.current_position_km >= self.config.border_crossing_km 
            && !self.border_transition_active {
            
            self.border_transition_active = true;
            let handover_duration_s = Uniform::new(
                self.config.operator_handover_config.handover_duration_s.start,
                self.config.operator_handover_config.handover_duration_s.end,
            ).sample(rng);
            self.border_transition_end_ms = self.current_time_ms + handover_duration_s * 1000.0;

            self.events.push(ScenarioEvent {
                time_ms: self.current_time_ms,
                event_type: EventType::BorderCrossing,
                description: format!("Border crossing at {:.1}km, operator handover starting",
                    self.current_position_km),
                affected_links: vec!["mobile".to_string()],
            });

            // Connection drops during operator handover
            self.link.path.is_active = false;
        }

        // Complete border transition
        if self.border_transition_active && self.current_time_ms >= self.border_transition_end_ms {
            self.border_transition_active = false;
            self.in_poland = false;

            // Check for registration failure
            if rng.gen::<f64>() < self.config.operator_handover_config.registration_failure_probability {
                // Retry handover (additional delay)
                self.events.push(ScenarioEvent {
                    time_ms: self.current_time_ms,
                    event_type: EventType::HandoverFailed,
                    description: "Network registration failed, retrying".to_string(),
                    affected_links: vec!["mobile".to_string()],
                });

                let retry_duration_s = Uniform::new(5.0, 15.0).sample(rng);
                self.border_transition_end_ms = self.current_time_ms + retry_duration_s * 1000.0;
                self.border_transition_active = true;
            } else {
                // Successful handover
                self.link.path.is_active = true;
                self.current_operator = self.config.operator_handover_config.german_operators
                    [rng.gen_range(0..self.config.operator_handover_config.german_operators.len())]
                    .to_string();

                self.events.push(ScenarioEvent {
                    time_ms: self.current_time_ms,
                    event_type: EventType::OperatorChange,
                    description: format!("Registered with {}", self.current_operator),
                    affected_links: vec!["mobile".to_string()],
                });
            }
        }

        // Rural coverage gaps
        if self.config.rural_coverage_gaps && !self.border_transition_active {
            if !self.in_coverage_gap {
                let gap_prob = 0.1 * delta_ms / 60_000.0; // ~6 gaps per hour of driving
                if rng.gen::<f64>() < gap_prob {
                    self.in_coverage_gap = true;
                    let gap_duration_s = Uniform::new(10.0, 60.0).sample(rng);
                    let gap_km = gap_duration_s * self.config.avg_speed_kmh / 3600.0;
                    self.coverage_gap_end_km = self.current_position_km + gap_km;

                    self.events.push(ScenarioEvent {
                        time_ms: self.current_time_ms,
                        event_type: EventType::CoverageGap,
                        description: format!("Rural coverage gap at {:.1}km", self.current_position_km),
                        affected_links: vec!["mobile".to_string()],
                    });

                    self.link.path.is_active = false;
                }
            } else if self.current_position_km >= self.coverage_gap_end_km {
                self.in_coverage_gap = false;
                self.link.path.is_active = true;

                self.events.push(ScenarioEvent {
                    time_ms: self.current_time_ms,
                    event_type: EventType::CoverageRestored,
                    description: format!("Coverage restored at {:.1}km", self.current_position_km),
                    affected_links: vec!["mobile".to_string()],
                });
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.current_position_km >= self.config.route_km
    }

    fn current_time_ms(&self) -> f64 {
        self.current_time_ms
    }

    fn duration_ms(&self) -> f64 {
        (self.config.route_km / self.config.avg_speed_kmh) * 3_600_000.0
    }

    fn get_events(&self) -> Vec<ScenarioEvent> {
        self.events.clone()
    }

    fn reset(&mut self) {
        self.current_time_ms = 0.0;
        self.current_position_km = 0.0;
        self.in_poland = true;
        self.border_transition_active = false;
        self.in_coverage_gap = false;
        self.events.clear();
        self.link.path.is_active = true;
    }
}

// ============================================================================
// Urban Mobility Scenario
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrbanMobilityConfig {
    pub movement_type: MovementType,
    pub area_type: AreaType,
    pub duration_s: f64,
    pub uplinks: Vec<UplinkConfig>,
    pub handover_frequency_per_km: f64,
    pub building_entry_probability: f64,
}

pub struct UrbanMobilityScenario {
    id: String,
    config: UrbanMobilityConfig,
    current_time_ms: f64,
    distance_traveled_m: f64,
    last_handover_m: f64,
    inside_building: bool,
    links: Vec<NetworkLink>,
    events: Vec<ScenarioEvent>,
}

impl UrbanMobilityScenario {
    pub fn new(id: &str, config: UrbanMobilityConfig) -> Self {
        let links = config.uplinks.iter().enumerate().map(|(i, uplink)| {
            NetworkLink::new(
                format!("uplink_{}", i),
                NetworkPath::typical_5g(format!("path_{}", i)),
                BackboneModel::new(5),
                uplink.provider.clone(),
            )
        }).collect();

        Self {
            id: id.to_string(),
            config,
            current_time_ms: 0.0,
            distance_traveled_m: 0.0,
            last_handover_m: 0.0,
            inside_building: false,
            links,
            events: Vec::new(),
        }
    }
}

impl Scenario for UrbanMobilityScenario {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> String {
        format!("Urban mobility: {:?} in {:?} area for {:.0}s",
            self.config.movement_type,
            self.config.area_type,
            self.config.duration_s)
    }

    fn init(&mut self, _seed: u64) {
        self.current_time_ms = 0.0;
        self.distance_traveled_m = 0.0;
        self.events.clear();
    }

    fn get_links(&self) -> Vec<&NetworkLink> {
        self.links.iter().collect()
    }

    fn get_links_mut(&mut self) -> Vec<&mut NetworkLink> {
        self.links.iter_mut().collect()
    }

    fn step(&mut self, delta_ms: f64, rng: &mut dyn RngCore) {
        self.current_time_ms += delta_ms;

        // Speed based on movement type
        let speed_mps = match self.config.movement_type {
            MovementType::Stationary => 0.0,
            MovementType::Walking => 1.4,   // ~5 km/h
            MovementType::Running => 3.0,   // ~11 km/h
            MovementType::Cycling => 5.5,   // ~20 km/h
            MovementType::Scooter => 4.2,   // ~15 km/h
            MovementType::Bus => 6.9,       // ~25 km/h
            MovementType::Tram => 5.5,      // ~20 km/h
            MovementType::Driving => 8.3,   // ~30 km/h urban
            MovementType::Train => 22.0,    // ~80 km/h urban
            MovementType::Flying => 0.0,
        };

        let delta_m = speed_mps * delta_ms / 1000.0;
        self.distance_traveled_m += delta_m;

        // Check for handover
        let handover_distance_m = 1000.0 / self.config.handover_frequency_per_km;
        if self.distance_traveled_m - self.last_handover_m >= handover_distance_m {
            self.last_handover_m = self.distance_traveled_m;

            self.events.push(ScenarioEvent {
                time_ms: self.current_time_ms,
                event_type: EventType::Handover,
                description: format!("Cell handover at {:.0}m", self.distance_traveled_m),
                affected_links: self.links.iter().map(|l| l.id.clone()).collect(),
            });

            // Brief disruption during handover
            for link in &mut self.links {
                // Increase loss temporarily
                link.path.loss_state.in_bad_state = true;
            }
        }

        // Building entry/exit
        if rng.gen::<f64>() < self.config.building_entry_probability * delta_ms / 60_000.0 {
            self.inside_building = !self.inside_building;

            if self.inside_building {
                self.events.push(ScenarioEvent {
                    time_ms: self.current_time_ms,
                    event_type: EventType::SignalDegraded,
                    description: "Entered building".to_string(),
                    affected_links: self.links.iter().map(|l| l.id.clone()).collect(),
                });

                // Degrade signal
                for link in &mut self.links {
                    link.path.loss_state.current_sinr_db -= 15.0;
                }
            } else {
                self.events.push(ScenarioEvent {
                    time_ms: self.current_time_ms,
                    event_type: EventType::SignalImproved,
                    description: "Exited building".to_string(),
                    affected_links: self.links.iter().map(|l| l.id.clone()).collect(),
                });

                for link in &mut self.links {
                    link.path.loss_state.current_sinr_db += 15.0;
                }
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.current_time_ms >= self.config.duration_s * 1000.0
    }

    fn current_time_ms(&self) -> f64 {
        self.current_time_ms
    }

    fn duration_ms(&self) -> f64 {
        self.config.duration_s * 1000.0
    }

    fn get_events(&self) -> Vec<ScenarioEvent> {
        self.events.clone()
    }

    fn reset(&mut self) {
        self.current_time_ms = 0.0;
        self.distance_traveled_m = 0.0;
        self.last_handover_m = 0.0;
        self.inside_building = false;
        self.events.clear();
        for link in &mut self.links {
            link.path.is_active = true;
        }
    }
}

// ============================================================================
// Edge Case Scenarios
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorstCaseConfig {
    pub simultaneous_failures: usize,
    pub failure_correlation: f64,
    pub recovery_time_s: Range<f64>,
}

pub struct WorstCaseScenario {
    id: String,
    config: WorstCaseConfig,
    current_time_ms: f64,
    links: Vec<NetworkLink>,
    events: Vec<ScenarioEvent>,
}

impl WorstCaseScenario {
    pub fn new(id: &str, config: WorstCaseConfig) -> Self {
        let links = (0..config.simultaneous_failures)
            .map(|i| {
                NetworkLink::new(
                    format!("path_{}", i),
                    NetworkPath::typical_lte(format!("path_{}", i)),
                    BackboneModel::new(8),
                    format!("provider_{}", i),
                )
            })
            .collect();

        Self {
            id: id.to_string(),
            config,
            current_time_ms: 0.0,
            links,
            events: Vec::new(),
        }
    }
}

impl Scenario for WorstCaseScenario {
    fn id(&self) -> &str { &self.id }
    fn description(&self) -> String {
        format!("Worst case: {} simultaneous failures", self.config.simultaneous_failures)
    }
    fn init(&mut self, _seed: u64) { self.current_time_ms = 0.0; self.events.clear(); }
    fn get_links(&self) -> Vec<&NetworkLink> { self.links.iter().collect() }
    fn get_links_mut(&mut self) -> Vec<&mut NetworkLink> { self.links.iter_mut().collect() }
    
    fn step(&mut self, delta_ms: f64, rng: &mut dyn RngCore) {
        self.current_time_ms += delta_ms;
        
        // Cascade failures with correlation
        if self.current_time_ms > 10_000.0 && self.current_time_ms < 10_100.0 {
            for (i, link) in self.links.iter_mut().enumerate() {
                if rng.gen::<f64>() < self.config.failure_correlation {
                    link.path.is_active = false;
                    self.events.push(ScenarioEvent {
                        time_ms: self.current_time_ms,
                        event_type: EventType::InfrastructureFailure,
                        description: format!("Cascade failure: path_{}", i),
                        affected_links: vec![link.id.clone()],
                    });
                }
            }
        }
    }
    
    fn is_complete(&self) -> bool { self.current_time_ms >= 60_000.0 }
    fn current_time_ms(&self) -> f64 { self.current_time_ms }
    fn duration_ms(&self) -> f64 { 60_000.0 }
    fn get_events(&self) -> Vec<ScenarioEvent> { self.events.clone() }
    fn reset(&mut self) {
        self.current_time_ms = 0.0;
        self.events.clear();
        for link in &mut self.links {
            link.path.is_active = true;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestCaseConfig {
    pub uplink_count: usize,
    pub all_stable: bool,
    pub latency_variation_percent: f64,
}

pub struct BestCaseScenario {
    id: String,
    config: BestCaseConfig,
    current_time_ms: f64,
    links: Vec<NetworkLink>,
    events: Vec<ScenarioEvent>,
}

impl BestCaseScenario {
    pub fn new(id: &str, config: BestCaseConfig) -> Self {
        let links = (0..config.uplink_count)
            .map(|i| {
                let mut link = NetworkLink::new(
                    format!("stable_path_{}", i),
                    NetworkPath::typical_5g(format!("stable_{}", i)),
                    BackboneModel::new(4),
                    format!("stable_provider_{}", i),
                );
                // Override with very stable characteristics
                link.path.loss_pattern = LossPattern::Random { probability: 0.0001 };
                link.path.jitter_model = JitterModel::new(1.0);
                link
            })
            .collect();

        Self {
            id: id.to_string(),
            config,
            current_time_ms: 0.0,
            links,
            events: Vec::new(),
        }
    }
}

impl Scenario for BestCaseScenario {
    fn id(&self) -> &str { &self.id }
    fn description(&self) -> String {
        format!("Best case: {} stable uplinks", self.config.uplink_count)
    }
    fn init(&mut self, _seed: u64) { self.current_time_ms = 0.0; }
    fn get_links(&self) -> Vec<&NetworkLink> { self.links.iter().collect() }
    fn get_links_mut(&mut self) -> Vec<&mut NetworkLink> { self.links.iter_mut().collect() }
    fn step(&mut self, delta_ms: f64, _rng: &mut dyn RngCore) { self.current_time_ms += delta_ms; }
    fn is_complete(&self) -> bool { self.current_time_ms >= 60_000.0 }
    fn current_time_ms(&self) -> f64 { self.current_time_ms }
    fn duration_ms(&self) -> f64 { 60_000.0 }
    fn get_events(&self) -> Vec<ScenarioEvent> { self.events.clone() }
    fn reset(&mut self) { self.current_time_ms = 0.0; self.events.clear(); }
}
