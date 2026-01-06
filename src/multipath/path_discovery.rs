//! Path discovery for multi-path optimization.
//!
//! Combines flow hash calculation and NAT detection to discover and characterize
//! network paths. This module enables:
//!
//! - ECMP path enumeration (discovering multiple paths to a destination)
//! - Path quality probing with NAT-aware packet correlation
//! - Path diversity assessment for optimal multi-path utilization
//!
//! Based on Dublin Traceroute techniques adapted for Triglav's multi-path architecture.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use super::flow_hash::{EcmpPathEnumerator, FlowId};
use super::nat::{IpIdMarker, NatId, NatProbe, NatProbeResponse, UplinkNatState};

/// Discovered hop information.
#[derive(Debug, Clone)]
pub struct Hop {
    /// TTL at which this hop was discovered.
    pub ttl: u8,
    /// IP address of the responding router.
    pub addr: Option<SocketAddr>,
    /// Round-trip time to this hop.
    pub rtt: Option<Duration>,
    /// NAT ID at this hop.
    pub nat_id: NatId,
    /// Flow hash used to reach this hop.
    pub flow_hash: u16,
    /// Whether NAT was detected at this hop.
    pub nat_detected: bool,
    /// ICMP response type.
    pub icmp_type: Option<u8>,
    /// Whether this is the final hop.
    pub is_last: bool,
    /// Discovery timestamp.
    pub discovered_at: Instant,
}

impl Hop {
    /// Create a hop from a probe response.
    pub fn from_response(
        probe: &NatProbe,
        response: &NatProbeResponse,
        prev_nat_id: Option<NatId>,
    ) -> Self {
        let nat_id = response.nat_id(probe.marker.udp_checksum);
        let nat_detected = prev_nat_id.is_some_and(|prev| prev != nat_id);

        Self {
            ttl: probe.marker.ttl,
            addr: Some(response.responder_addr),
            rtt: Some(response.rtt(probe.sent_at)),
            nat_id,
            flow_hash: FlowId::from_udp(probe.src_addr, probe.dst_addr).flow_hash(),
            nat_detected,
            icmp_type: Some(response.icmp_type),
            is_last: response.is_destination(),
            discovered_at: response.received_at,
        }
    }

    /// Create a placeholder for a non-responding hop.
    pub fn non_responding(ttl: u8, flow_hash: u16) -> Self {
        Self {
            ttl,
            addr: None,
            rtt: None,
            nat_id: NatId::default(),
            flow_hash,
            nat_detected: false,
            icmp_type: None,
            is_last: false,
            discovered_at: Instant::now(),
        }
    }
}

/// A discovered path (sequence of hops for a specific flow).
#[derive(Debug, Clone)]
pub struct DiscoveredPath {
    /// Flow ID used for this path.
    pub flow_id: FlowId,
    /// Flow hash for this path.
    pub flow_hash: u16,
    /// Hops along this path.
    pub hops: Vec<Hop>,
    /// Whether the path reaches the destination.
    pub reaches_destination: bool,
    /// Total path RTT (to destination or last responding hop).
    pub total_rtt: Option<Duration>,
    /// Number of NAT devices detected on this path.
    pub nat_count: usize,
    /// Path quality score (0.0 - 1.0).
    pub quality_score: f64,
    /// Last update time.
    pub last_updated: Instant,
}

impl DiscoveredPath {
    /// Create a new empty path.
    pub fn new(flow_id: FlowId) -> Self {
        Self {
            flow_hash: flow_id.flow_hash(),
            flow_id,
            hops: Vec::new(),
            reaches_destination: false,
            total_rtt: None,
            nat_count: 0,
            quality_score: 0.0,
            last_updated: Instant::now(),
        }
    }

    /// Add a hop to the path.
    pub fn add_hop(&mut self, hop: Hop) {
        if hop.nat_detected {
            self.nat_count += 1;
        }
        if hop.is_last {
            self.reaches_destination = true;
            self.total_rtt = hop.rtt;
        }
        self.hops.push(hop);
        self.hops.sort_by_key(|h| h.ttl);
        self.update_quality();
        self.last_updated = Instant::now();
    }

    /// Update path quality score.
    fn update_quality(&mut self) {
        if self.hops.is_empty() {
            self.quality_score = 0.0;
            return;
        }

        let mut score = 1.0;

        // Penalize for non-responding hops
        let responding = self.hops.iter().filter(|h| h.addr.is_some()).count();
        let response_ratio = responding as f64 / self.hops.len() as f64;
        score *= response_ratio;

        // Penalize for high latency
        if let Some(rtt) = self.total_rtt {
            let rtt_ms = rtt.as_secs_f64() * 1000.0;
            score *= 1.0 / (1.0 + rtt_ms / 100.0);
        }

        // Penalize for NAT traversal
        score *= 1.0 / (1.0 + self.nat_count as f64 * 0.1);

        // Bonus for reaching destination
        if self.reaches_destination {
            score *= 1.2;
        }

        self.quality_score = score.clamp(0.0, 1.0);
    }

    /// Get the path length (hop count).
    pub fn length(&self) -> usize {
        self.hops.len()
    }

    /// Check if this path is complete.
    pub fn is_complete(&self) -> bool {
        self.reaches_destination
    }

    /// Get hop at a specific TTL.
    pub fn hop_at_ttl(&self, ttl: u8) -> Option<&Hop> {
        self.hops.iter().find(|h| h.ttl == ttl)
    }
}

/// Path diversity metrics for a destination.
#[derive(Debug, Default, Clone)]
pub struct PathDiversity {
    /// Number of unique paths discovered.
    pub unique_paths: usize,
    /// Number of unique first hops.
    pub unique_first_hops: usize,
    /// Number of unique intermediate hops.
    pub unique_intermediate_hops: usize,
    /// Path diversity score (0.0 - 1.0).
    pub diversity_score: f64,
    /// Recommended number of paths to use.
    pub recommended_paths: usize,
}

impl PathDiversity {
    /// Calculate diversity metrics from discovered paths.
    pub fn from_paths(paths: &[DiscoveredPath]) -> Self {
        if paths.is_empty() {
            return Self::default();
        }

        let unique_paths = paths.len();

        // Count unique first hops
        let first_hops: std::collections::HashSet<_> = paths
            .iter()
            .filter_map(|p| p.hops.first())
            .filter_map(|h| h.addr)
            .collect();
        let unique_first_hops = first_hops.len();

        // Count unique intermediate hops (excluding first and last)
        let intermediate_hops: std::collections::HashSet<_> = paths
            .iter()
            .flat_map(|p| {
                p.hops
                    .iter()
                    .skip(1)
                    .filter(|h| !h.is_last)
                    .filter_map(|h| h.addr)
            })
            .collect();
        let unique_intermediate_hops = intermediate_hops.len();

        // Calculate diversity score
        let path_diversity = if unique_paths > 1 {
            (unique_first_hops as f64 / unique_paths as f64).min(1.0)
        } else {
            0.0
        };

        let hop_diversity = if intermediate_hops.is_empty() {
            0.5
        } else {
            (unique_intermediate_hops as f64 / (unique_paths * 3) as f64).min(1.0)
        };

        let diversity_score = (path_diversity * 0.6 + hop_diversity * 0.4).clamp(0.0, 1.0);

        // Recommend paths based on diversity
        let recommended_paths = match unique_first_hops {
            0 => 1,
            1 => unique_paths.min(2),
            2..=3 => unique_first_hops,
            _ => 4.min(unique_first_hops),
        };

        Self {
            unique_paths,
            unique_first_hops,
            unique_intermediate_hops,
            diversity_score,
            recommended_paths,
        }
    }
}

/// Configuration for path discovery.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PathDiscoveryConfig {
    /// Minimum TTL to probe.
    #[serde(default = "default_min_ttl")]
    pub min_ttl: u8,
    /// Maximum TTL to probe.
    #[serde(default = "default_max_ttl")]
    pub max_ttl: u8,
    /// Number of paths to enumerate.
    #[serde(default = "default_num_paths")]
    pub num_paths: u16,
    /// Base source port for probing.
    #[serde(default = "default_base_src_port")]
    pub base_src_port: u16,
    /// Use source port (vs destination port) for path variation.
    #[serde(default = "default_use_src_port")]
    pub use_src_port: bool,
    /// Probe timeout.
    #[serde(default = "default_probe_timeout", with = "humantime_serde")]
    pub probe_timeout: Duration,
    /// Delay between probes.
    #[serde(default = "default_probe_delay", with = "humantime_serde")]
    pub probe_delay: Duration,
    /// Number of retries for non-responding hops.
    #[serde(default = "default_retries")]
    pub retries: u8,
}

fn default_min_ttl() -> u8 {
    1
}
fn default_max_ttl() -> u8 {
    32
}
fn default_num_paths() -> u16 {
    8
}
fn default_base_src_port() -> u16 {
    33434
}
fn default_use_src_port() -> bool {
    true
}
fn default_probe_timeout() -> Duration {
    Duration::from_secs(3)
}
fn default_probe_delay() -> Duration {
    Duration::from_millis(50)
}
fn default_retries() -> u8 {
    2
}

impl Default for PathDiscoveryConfig {
    fn default() -> Self {
        Self {
            min_ttl: default_min_ttl(),
            max_ttl: default_max_ttl(),
            num_paths: default_num_paths(),
            base_src_port: default_base_src_port(),
            use_src_port: default_use_src_port(),
            probe_timeout: default_probe_timeout(),
            probe_delay: default_probe_delay(),
            retries: default_retries(),
        }
    }
}

/// Path discovery engine.
///
/// Discovers and tracks multiple paths to destinations using ECMP enumeration
/// and NAT-aware packet correlation.
#[derive(Debug)]
pub struct PathDiscovery {
    /// Configuration.
    config: PathDiscoveryConfig,
    /// Discovered paths per destination.
    paths: RwLock<HashMap<SocketAddr, Vec<DiscoveredPath>>>,
    /// NAT state per uplink.
    nat_states: RwLock<HashMap<u16, UplinkNatState>>,
    /// Path diversity cache.
    diversity_cache: RwLock<HashMap<SocketAddr, PathDiversity>>,
}

impl PathDiscovery {
    /// Create a new path discovery engine.
    pub fn new(config: PathDiscoveryConfig) -> Self {
        Self {
            config,
            paths: RwLock::new(HashMap::new()),
            nat_states: RwLock::new(HashMap::new()),
            diversity_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create NAT state for an uplink.
    pub fn get_nat_state(&self, uplink_id: u16) -> UplinkNatState {
        let states = self.nat_states.read();
        states.get(&uplink_id).cloned().unwrap_or_default()
    }

    /// Update NAT state for an uplink.
    pub fn update_nat_state<F>(&self, uplink_id: u16, f: F)
    where
        F: FnOnce(&mut UplinkNatState),
    {
        let mut states = self.nat_states.write();
        let state = states.entry(uplink_id).or_default();
        f(state);
    }

    /// Generate probes for path discovery to a destination.
    pub fn generate_probes(
        &self,
        src_addr: SocketAddr,
        dst_addr: SocketAddr,
    ) -> Vec<(u8, FlowId, IpIdMarker)> {
        let base_flow = FlowId::from_udp(src_addr, dst_addr);
        let enumerator = EcmpPathEnumerator::new(
            base_flow,
            self.config.base_src_port,
            self.config.num_paths,
            self.config.use_src_port,
        );

        let mut probes = Vec::new();

        for flow in enumerator.flows() {
            let flow_id = if self.config.use_src_port {
                flow.src_port
            } else {
                flow.dst_port
            };

            for ttl in self.config.min_ttl..=self.config.max_ttl {
                let marker = IpIdMarker::from_probe(ttl, flow_id, self.config.use_src_port);
                probes.push((ttl, flow, marker));
            }
        }

        probes
    }

    /// Record a discovered hop.
    pub fn record_hop(&self, dst_addr: SocketAddr, hop: Hop) {
        let mut paths = self.paths.write();
        let dest_paths = paths.entry(dst_addr).or_default();

        // Find or create path for this flow hash
        let path = dest_paths.iter_mut().find(|p| p.flow_hash == hop.flow_hash);

        if let Some(path) = path {
            path.add_hop(hop);
        } else {
            // Create new path with a synthesized flow ID
            let flow_id = FlowId::new(
                dst_addr.ip(), // We don't know the actual src here
                dst_addr.ip(),
                0,
                0,
                17,
            );
            let mut new_path = DiscoveredPath::new(flow_id);
            new_path.flow_hash = hop.flow_hash;
            new_path.add_hop(hop);
            dest_paths.push(new_path);
        }

        // Invalidate diversity cache
        self.diversity_cache.write().remove(&dst_addr);
    }

    /// Get discovered paths for a destination.
    pub fn get_paths(&self, dst_addr: SocketAddr) -> Vec<DiscoveredPath> {
        self.paths
            .read()
            .get(&dst_addr)
            .cloned()
            .unwrap_or_default()
    }

    /// Get path diversity for a destination.
    pub fn get_diversity(&self, dst_addr: SocketAddr) -> PathDiversity {
        // Check cache
        if let Some(diversity) = self.diversity_cache.read().get(&dst_addr) {
            return diversity.clone();
        }

        // Calculate diversity
        let paths = self.get_paths(dst_addr);
        let diversity = PathDiversity::from_paths(&paths);

        // Cache result
        self.diversity_cache
            .write()
            .insert(dst_addr, diversity.clone());

        diversity
    }

    /// Get the best paths for a destination (sorted by quality).
    pub fn get_best_paths(&self, dst_addr: SocketAddr, count: usize) -> Vec<DiscoveredPath> {
        let mut paths = self.get_paths(dst_addr);
        paths.sort_by(|a, b| {
            b.quality_score
                .partial_cmp(&a.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        paths.truncate(count);
        paths
    }

    /// Get flow hashes for different ECMP paths to a destination.
    pub fn get_ecmp_flow_hashes(&self, dst_addr: SocketAddr) -> Vec<u16> {
        let paths = self.paths.read();
        paths
            .get(&dst_addr)
            .map(|p| p.iter().map(|path| path.flow_hash).collect())
            .unwrap_or_default()
    }

    /// Check if a destination has NAT on any discovered path.
    pub fn has_nat(&self, dst_addr: SocketAddr) -> bool {
        self.paths
            .read()
            .get(&dst_addr)
            .is_some_and(|paths| paths.iter().any(|p| p.nat_count > 0))
    }

    /// Clear discovered paths for a destination.
    pub fn clear_paths(&self, dst_addr: SocketAddr) {
        self.paths.write().remove(&dst_addr);
        self.diversity_cache.write().remove(&dst_addr);
    }

    /// Clear all discovered paths.
    pub fn clear_all(&self) {
        self.paths.write().clear();
        self.diversity_cache.write().clear();
    }

    /// Get configuration.
    pub fn config(&self) -> &PathDiscoveryConfig {
        &self.config
    }

    /// Cleanup stale data.
    pub fn cleanup(&self, max_age: Duration) {
        let now = Instant::now();

        // Remove stale paths
        self.paths.write().retain(|_, paths| {
            paths.retain(|p| now.duration_since(p.last_updated) < max_age);
            !paths.is_empty()
        });

        // NAT states don't need cleanup - they just track state

        // Clear diversity cache (will be recalculated as needed)
        self.diversity_cache.write().clear();
    }
}

impl Default for PathDiscovery {
    fn default() -> Self {
        Self::new(PathDiscoveryConfig::default())
    }
}

/// ECMP-aware flow selector.
///
/// Selects flows to use specific ECMP paths based on discovered path characteristics.
#[derive(Debug)]
pub struct EcmpFlowSelector {
    /// Flow hash to uplink mapping.
    hash_to_uplink: RwLock<HashMap<u16, u16>>,
    /// Preferred hashes per destination.
    preferred_hashes: RwLock<HashMap<SocketAddr, Vec<u16>>>,
}

impl Default for EcmpFlowSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl EcmpFlowSelector {
    /// Create a new ECMP flow selector.
    pub fn new() -> Self {
        Self {
            hash_to_uplink: RwLock::new(HashMap::new()),
            preferred_hashes: RwLock::new(HashMap::new()),
        }
    }

    /// Map a flow hash to a specific uplink.
    pub fn set_mapping(&self, flow_hash: u16, uplink_id: u16) {
        self.hash_to_uplink.write().insert(flow_hash, uplink_id);
    }

    /// Get the uplink for a flow hash.
    pub fn get_uplink(&self, flow_hash: u16) -> Option<u16> {
        self.hash_to_uplink.read().get(&flow_hash).copied()
    }

    /// Set preferred hashes for a destination.
    pub fn set_preferred(&self, dst: SocketAddr, hashes: Vec<u16>) {
        self.preferred_hashes.write().insert(dst, hashes);
    }

    /// Get preferred hash for a destination.
    pub fn get_preferred(&self, dst: SocketAddr) -> Option<u16> {
        self.preferred_hashes
            .read()
            .get(&dst)
            .and_then(|h| h.first().copied())
    }

    /// Suggest a source port that will produce the desired flow hash.
    ///
    /// This is useful for ECMP path selection - by choosing the right source port,
    /// we can influence which ECMP path the packet takes.
    pub fn suggest_port_for_path(&self, base_flow: FlowId, target_hash: u16) -> Option<u16> {
        // Try ports around the base to find one that produces the target hash
        for offset in 0..1000u16 {
            let port = base_flow.src_port.wrapping_add(offset);
            let test_flow = base_flow.with_src_port(port);
            if test_flow.flow_hash() == target_hash {
                return Some(port);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_discovered_path() {
        let flow = FlowId::new(
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            12345,
            53,
            17,
        );

        let mut path = DiscoveredPath::new(flow);
        assert!(!path.is_complete());
        assert_eq!(path.length(), 0);

        let hop = Hop {
            ttl: 1,
            addr: Some(SocketAddr::from(([192, 168, 1, 254], 0))),
            rtt: Some(Duration::from_millis(5)),
            nat_id: NatId(0),
            flow_hash: flow.flow_hash(),
            nat_detected: false,
            icmp_type: Some(11),
            is_last: false,
            discovered_at: Instant::now(),
        };

        path.add_hop(hop);
        assert_eq!(path.length(), 1);
        assert!(!path.is_complete());
    }

    #[test]
    fn test_path_diversity() {
        let flow1 = FlowId::new(
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            12345,
            53,
            17,
        );

        let flow2 = flow1.with_src_port(12346);

        let path1 = DiscoveredPath::new(flow1);
        let path2 = DiscoveredPath::new(flow2);

        let diversity = PathDiversity::from_paths(&[path1, path2]);
        assert_eq!(diversity.unique_paths, 2);
    }

    #[test]
    fn test_path_discovery_config() {
        let config = PathDiscoveryConfig::default();
        assert_eq!(config.min_ttl, 1);
        assert_eq!(config.max_ttl, 32);
        assert_eq!(config.num_paths, 8);
    }

    #[test]
    fn test_ecmp_flow_selector() {
        let selector = EcmpFlowSelector::new();

        selector.set_mapping(0x1234, 1);
        assert_eq!(selector.get_uplink(0x1234), Some(1));
        assert_eq!(selector.get_uplink(0x5678), None);
    }

    #[test]
    fn test_hop_creation() {
        let hop = Hop::non_responding(5, 0x1234);
        assert!(hop.addr.is_none());
        assert_eq!(hop.ttl, 5);
        assert_eq!(hop.flow_hash, 0x1234);
    }
}
