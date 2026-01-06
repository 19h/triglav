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
//! - Throughput optimization with BBR-style congestion control
//! - Path MTU discovery
//! - Effective throughput scoring (bandwidth + latency combined)

pub mod aggregator;
mod flow_hash;
mod manager;
mod nat;
mod path_discovery;
mod scheduler;
mod throughput;
mod uplink;

pub use aggregator::{
    AggregationMode, AggregatorConfig, AggregatorStats, BandwidthAggregator, ReorderBuffer,
    ReorderStats,
};
pub use manager::{MultipathConfig, MultipathEvent, MultipathManager};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulingStrategy};
pub use throughput::{
    BbrState, BdpEstimator, EffectiveThroughput, FrameBatcher, PmtudState, ThroughputConfig,
    ThroughputOptimizer, ThroughputSummary, DEFAULT_MTU, MAX_MTU, MIN_MTU,
};
pub use uplink::{ConnectionParams, Uplink, UplinkConfig, UplinkState};

// Dublin Traceroute-inspired modules
pub use flow_hash::{
    calculate_flow_hash, flow_hash_from_addrs, EcmpPathEnumerator, FlowHashBucket, FlowId,
};
pub use nat::{
    compute_udp_checksum, IpIdMarker, NatDetectionState, NatId, NatProbe, NatProbeResponse,
    NatType, ProbeMatcher, UplinkNatState,
};
pub use path_discovery::{
    DiscoveredPath, EcmpFlowSelector, Hop, PathDiscovery, PathDiscoveryConfig, PathDiversity,
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
