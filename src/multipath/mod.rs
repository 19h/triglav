//! Multi-path connection management with intelligent uplink selection.
//!
//! This module implements the core multi-path functionality:
//! - Uplink discovery and management
//! - Intelligent packet scheduling across uplinks
//! - Automatic failover and recovery
//! - Bandwidth aggregation
//! - Quality-based path selection
//! - ECMP-aware flow hashing (Dublin Traceroute technique)
//! - NAT detection and traversal support
//! - Path discovery and diversity assessment

mod uplink;
mod scheduler;
mod manager;
mod flow_hash;
mod nat;
mod path_discovery;

pub use uplink::{Uplink, UplinkConfig, UplinkState};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulingStrategy};
pub use manager::{MultipathManager, MultipathConfig, MultipathEvent};

// Dublin Traceroute-inspired modules
pub use flow_hash::{
    FlowId, FlowHashBucket, EcmpPathEnumerator,
    calculate_flow_hash, flow_hash_from_addrs,
};
pub use nat::{
    NatId, NatType, NatDetectionState, UplinkNatState,
    IpIdMarker, NatProbe, NatProbeResponse, ProbeMatcher,
    compute_udp_checksum,
};
pub use path_discovery::{
    PathDiscovery, PathDiscoveryConfig, PathDiversity,
    DiscoveredPath, Hop, EcmpFlowSelector,
};

use std::time::Duration;

/// Default probe interval for uplink quality measurement.
pub const DEFAULT_PROBE_INTERVAL: Duration = Duration::from_secs(1);

/// Default timeout for considering an uplink dead.
pub const DEFAULT_UPLINK_TIMEOUT: Duration = Duration::from_secs(10);

/// Minimum RTT samples before making scheduling decisions.
pub const MIN_RTT_SAMPLES: usize = 3;

/// Weight decay factor for exponential moving average.
pub const EMA_ALPHA: f64 = 0.2;
