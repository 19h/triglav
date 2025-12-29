//! NAT detection and traversal support.
//!
//! Implements NAT detection techniques from Dublin Traceroute:
//! - IP ID matching for packet correlation through NAT
//! - Checksum-based NAT ID detection
//! - NAT traversal state tracking
//!
//! These techniques allow Triglav to:
//! 1. Detect when packets traverse NAT devices
//! 2. Correlate sent and received packets even after NAT rewriting
//! 3. Track NAT behavior changes along a path

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

/// NAT identifier calculated from checksum difference.
///
/// When a NAT rewrites a packet (changing IP/port), the UDP checksum changes.
/// The NAT ID is the difference between the received and sent checksums.
/// If the NAT ID changes between hops, NAT was detected at that hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NatId(pub u16);

impl NatId {
    /// Calculate NAT ID from sent and received UDP checksums.
    ///
    /// NAT ID = received_checksum - sent_checksum
    ///
    /// If NAT rewrites the packet, the checksum changes and NAT ID will be non-zero.
    /// If NAT ID differs between hops, a NAT device exists at that point.
    pub fn from_checksums(sent_checksum: u16, received_checksum: u16) -> Self {
        Self(received_checksum.wrapping_sub(sent_checksum))
    }

    /// Check if this represents a NATted connection (non-zero ID).
    pub fn is_natted(&self) -> bool {
        self.0 != 0
    }

    /// Get the raw NAT ID value.
    pub fn value(&self) -> u16 {
        self.0
    }
}

/// IP ID matching for packet correlation through NAT.
///
/// Dublin Traceroute's key insight: store the UDP checksum in the IP ID field.
/// When NAT rewrites the packet, the IP ID is preserved in ICMP error responses,
/// allowing us to match packets even after NAT translation.
#[derive(Debug, Clone, Copy)]
pub struct IpIdMarker {
    /// The IP ID value (derived from UDP checksum).
    pub ip_id: u16,
    /// The original UDP checksum.
    pub udp_checksum: u16,
    /// TTL used for this probe.
    pub ttl: u8,
    /// Flow identifier (port-based).
    pub flow_id: u16,
}

impl IpIdMarker {
    /// Create a new IP ID marker.
    ///
    /// The IP ID is set to the UDP checksum, allowing packet matching
    /// through NAT devices that preserve IP ID in ICMP responses.
    pub fn new(udp_checksum: u16, ttl: u8, flow_id: u16) -> Self {
        Self {
            ip_id: udp_checksum,
            udp_checksum,
            ttl,
            flow_id,
        }
    }

    /// Create marker from packet metadata.
    ///
    /// Computes a predictable IP ID based on TTL and flow ID for NAT correlation.
    /// The `_use_src_port` parameter is reserved for future use when different
    /// encoding schemes are needed.
    pub fn from_probe(ttl: u8, flow_id: u16, _use_src_port: bool) -> Self {
        // IP ID encoding: TTL + flow_id for correlation
        let id = u16::from(ttl).wrapping_add(flow_id);
        Self {
            ip_id: id,
            udp_checksum: id, // Will be overwritten when actual checksum is computed
            ttl,
            flow_id,
        }
    }

    /// Match this marker against a received IP ID.
    ///
    /// Returns true if the received IP ID matches our marker, indicating
    /// this is a response to our probe.
    pub fn matches(&self, received_ip_id: u16) -> bool {
        self.ip_id == received_ip_id
    }

    /// Match using UDP checksum comparison (original paris-traceroute method).
    ///
    /// This works when there's no NAT, as the checksum is preserved.
    pub fn matches_checksum(&self, received_checksum: u16) -> bool {
        self.udp_checksum == received_checksum
    }
}

/// Probe packet for NAT detection.
#[derive(Debug, Clone)]
pub struct NatProbe {
    /// IP ID marker for correlation.
    pub marker: IpIdMarker,
    /// Destination address.
    pub dst_addr: SocketAddr,
    /// Source address (before NAT).
    pub src_addr: SocketAddr,
    /// Time sent.
    pub sent_at: Instant,
    /// Payload used (for checksum computation).
    pub payload: Vec<u8>,
}

impl NatProbe {
    /// Create a new NAT probe.
    pub fn new(
        marker: IpIdMarker,
        src_addr: SocketAddr,
        dst_addr: SocketAddr,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            marker,
            dst_addr,
            src_addr,
            sent_at: Instant::now(),
            payload,
        }
    }

    /// Create probe payload with embedded correlation data.
    ///
    /// The last two bytes encode TTL + flow_id for matching.
    pub fn create_payload(base_payload: &[u8], ttl: u8, flow_id: u16) -> Vec<u8> {
        let mut payload = base_payload.to_vec();
        let id = u16::from(ttl).wrapping_add(flow_id);
        payload.push((id >> 8) as u8);
        payload.push((id & 0xff) as u8);
        payload
    }
}

/// Response from a NAT probe.
#[derive(Debug, Clone)]
pub struct NatProbeResponse {
    /// IP ID from the response (in ICMP inner packet).
    pub inner_ip_id: u16,
    /// UDP checksum from the response (in ICMP inner packet).
    pub inner_udp_checksum: u16,
    /// Source IP of the ICMP response (the responding hop).
    pub responder_addr: SocketAddr,
    /// Response IP ID (for NAT loop detection).
    pub response_ip_id: u16,
    /// Time received.
    pub received_at: Instant,
    /// ICMP type (11 = TTL exceeded, 3 = unreachable).
    pub icmp_type: u8,
    /// ICMP code.
    pub icmp_code: u8,
}

impl NatProbeResponse {
    /// Calculate NAT ID by comparing with original probe.
    pub fn nat_id(&self, original_checksum: u16) -> NatId {
        NatId::from_checksums(original_checksum, self.inner_udp_checksum)
    }

    /// Calculate RTT from probe send time.
    pub fn rtt(&self, sent_at: Instant) -> Duration {
        self.received_at.saturating_duration_since(sent_at)
    }

    /// Check if this is a TTL exceeded response.
    pub fn is_ttl_exceeded(&self) -> bool {
        self.icmp_type == 11 && self.icmp_code == 0
    }

    /// Check if this is the final destination (port unreachable).
    pub fn is_destination(&self) -> bool {
        self.icmp_type == 3 && self.icmp_code == 3
    }
}

/// NAT detection state for tracking NAT changes across a path.
#[derive(Debug, Clone)]
pub struct NatDetectionState {
    /// Current NAT ID.
    current_nat_id: NatId,
    /// NAT ID history per hop.
    hop_nat_ids: Vec<Option<NatId>>,
    /// Detected NAT locations (hop indices where NAT ID changed).
    nat_locations: Vec<usize>,
    /// Last update time.
    last_update: Instant,
}

impl Default for NatDetectionState {
    fn default() -> Self {
        Self::new()
    }
}

impl NatDetectionState {
    /// Create a new NAT detection state.
    pub fn new() -> Self {
        Self {
            current_nat_id: NatId::default(),
            hop_nat_ids: Vec::new(),
            nat_locations: Vec::new(),
            last_update: Instant::now(),
        }
    }

    /// Update state with a new hop's NAT ID.
    ///
    /// Returns true if NAT was detected at this hop (NAT ID changed).
    pub fn update_hop(&mut self, hop: usize, nat_id: NatId) -> bool {
        // Ensure vector is large enough
        while self.hop_nat_ids.len() <= hop {
            self.hop_nat_ids.push(None);
        }

        let nat_detected = hop > 0
            && self.hop_nat_ids.get(hop - 1).is_some_and(|prev| {
                prev.is_some_and(|prev_id| prev_id != nat_id)
            });

        if nat_detected && !self.nat_locations.contains(&hop) {
            self.nat_locations.push(hop);
        }

        self.hop_nat_ids[hop] = Some(nat_id);
        self.current_nat_id = nat_id;
        self.last_update = Instant::now();

        nat_detected
    }

    /// Check if any NAT was detected on the path.
    pub fn has_nat(&self) -> bool {
        !self.nat_locations.is_empty() || self.current_nat_id.is_natted()
    }

    /// Get the number of detected NAT devices.
    pub fn nat_count(&self) -> usize {
        self.nat_locations.len()
    }

    /// Get the hop indices where NAT was detected.
    pub fn nat_locations(&self) -> &[usize] {
        &self.nat_locations
    }

    /// Get the current NAT ID.
    pub fn current_nat_id(&self) -> NatId {
        self.current_nat_id
    }

    /// Reset the state.
    pub fn reset(&mut self) {
        self.current_nat_id = NatId::default();
        self.hop_nat_ids.clear();
        self.nat_locations.clear();
        self.last_update = Instant::now();
    }
}

/// Probe matcher for correlating sent probes with received responses.
#[derive(Debug)]
pub struct ProbeMatcher {
    /// Outstanding probes indexed by IP ID.
    probes_by_ip_id: HashMap<u16, NatProbe>,
    /// Outstanding probes indexed by checksum (fallback).
    probes_by_checksum: HashMap<u16, u16>, // checksum -> ip_id
    /// Probe timeout.
    timeout: Duration,
}

impl ProbeMatcher {
    /// Create a new probe matcher.
    pub fn new(timeout: Duration) -> Self {
        Self {
            probes_by_ip_id: HashMap::new(),
            probes_by_checksum: HashMap::new(),
            timeout,
        }
    }

    /// Register a sent probe.
    pub fn register_probe(&mut self, probe: NatProbe) {
        let ip_id = probe.marker.ip_id;
        let checksum = probe.marker.udp_checksum;
        self.probes_by_checksum.insert(checksum, ip_id);
        self.probes_by_ip_id.insert(ip_id, probe);
    }

    /// Match a response to its original probe.
    ///
    /// First tries IP ID matching (works through NAT), then falls back
    /// to checksum matching (works without NAT).
    pub fn match_response(&mut self, response: &NatProbeResponse) -> Option<NatProbe> {
        // Try IP ID matching first (works through NAT)
        if let Some(probe) = self.probes_by_ip_id.remove(&response.inner_ip_id) {
            self.probes_by_checksum.remove(&probe.marker.udp_checksum);
            return Some(probe);
        }

        // Fall back to checksum matching (original paris-traceroute method)
        if let Some(&ip_id) = self.probes_by_checksum.get(&response.inner_udp_checksum) {
            if let Some(probe) = self.probes_by_ip_id.remove(&ip_id) {
                self.probes_by_checksum.remove(&probe.marker.udp_checksum);
                return Some(probe);
            }
        }

        None
    }

    /// Remove expired probes.
    pub fn cleanup(&mut self) {
        let now = Instant::now();
        let timeout = self.timeout;

        self.probes_by_ip_id.retain(|_, probe| {
            now.duration_since(probe.sent_at) < timeout
        });

        // Rebuild checksum index
        self.probes_by_checksum.clear();
        for probe in self.probes_by_ip_id.values() {
            self.probes_by_checksum.insert(probe.marker.udp_checksum, probe.marker.ip_id);
        }
    }

    /// Get the number of outstanding probes.
    pub fn pending_count(&self) -> usize {
        self.probes_by_ip_id.len()
    }
}

/// Per-uplink NAT tracking state.
#[derive(Debug, Clone)]
pub struct UplinkNatState {
    /// NAT detection state.
    detection: NatDetectionState,
    /// Whether the uplink is behind NAT.
    is_natted: bool,
    /// Last known external address (after NAT).
    external_addr: Option<SocketAddr>,
    /// NAT type (if detected).
    nat_type: NatType,
}

impl Default for UplinkNatState {
    fn default() -> Self {
        Self::new()
    }
}

impl UplinkNatState {
    /// Create a new uplink NAT state.
    pub fn new() -> Self {
        Self {
            detection: NatDetectionState::new(),
            is_natted: false,
            external_addr: None,
            nat_type: NatType::Unknown,
        }
    }

    /// Check if this uplink is behind NAT.
    pub fn is_natted(&self) -> bool {
        self.is_natted
    }

    /// Get the external address (after NAT translation).
    pub fn external_addr(&self) -> Option<SocketAddr> {
        self.external_addr
    }

    /// Get the detected NAT type.
    pub fn nat_type(&self) -> NatType {
        self.nat_type
    }

    /// Update NAT state from a probe response.
    pub fn update(&mut self, probe: &NatProbe, response: &NatProbeResponse) {
        let nat_id = response.nat_id(probe.marker.udp_checksum);

        self.is_natted = nat_id.is_natted();

        // Infer external address from response if possible
        // The responding hop saw our packet with the NATted source
        if self.is_natted {
            self.detection.update_hop(probe.marker.ttl as usize, nat_id);
        }
    }

    /// Set the external address discovered via other means (e.g., STUN).
    pub fn set_external_addr(&mut self, addr: SocketAddr) {
        self.external_addr = Some(addr);
    }

    /// Set the detected NAT type.
    pub fn set_nat_type(&mut self, nat_type: NatType) {
        self.nat_type = nat_type;
        self.is_natted = nat_type != NatType::None && nat_type != NatType::Unknown;
    }

    /// Get NAT detection state.
    pub fn detection_state(&self) -> &NatDetectionState {
        &self.detection
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.detection.reset();
        self.is_natted = false;
        self.external_addr = None;
        self.nat_type = NatType::Unknown;
    }
}

/// NAT type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NatType {
    /// NAT type not yet determined.
    #[default]
    Unknown,
    /// No NAT detected (public IP).
    None,
    /// Full cone NAT (least restrictive).
    FullCone,
    /// Restricted cone NAT.
    RestrictedCone,
    /// Port-restricted cone NAT.
    PortRestrictedCone,
    /// Symmetric NAT (most restrictive).
    Symmetric,
}

impl NatType {
    /// Check if this NAT type allows direct connections.
    pub fn allows_direct_connect(&self) -> bool {
        matches!(self, Self::None | Self::FullCone)
    }

    /// Check if this NAT type requires hole punching.
    pub fn requires_hole_punch(&self) -> bool {
        matches!(self, Self::RestrictedCone | Self::PortRestrictedCone)
    }

    /// Check if this NAT type requires a relay.
    pub fn requires_relay(&self) -> bool {
        matches!(self, Self::Symmetric)
    }
}

/// UDP checksum computation for NAT ID matching.
///
/// Computes the UDP checksum for a packet, which is used for NAT detection.
pub fn compute_udp_checksum(
    src_addr: &[u8],
    dst_addr: &[u8],
    src_port: u16,
    dst_port: u16,
    payload: &[u8],
) -> u16 {
    let udp_len = 8 + payload.len();
    let mut sum: u32 = 0;

    // Pseudo-header
    for chunk in src_addr.chunks(2) {
        sum = sum.wrapping_add(u32::from(u16::from_be_bytes([
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
        ])));
    }
    for chunk in dst_addr.chunks(2) {
        sum = sum.wrapping_add(u32::from(u16::from_be_bytes([
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
        ])));
    }
    sum = sum.wrapping_add(17); // UDP protocol
    sum = sum.wrapping_add(udp_len as u32);

    // UDP header
    sum = sum.wrapping_add(u32::from(src_port));
    sum = sum.wrapping_add(u32::from(dst_port));
    sum = sum.wrapping_add(udp_len as u32);
    // checksum field is 0 during computation

    // Payload
    for chunk in payload.chunks(2) {
        let word = if chunk.len() == 2 {
            u16::from_be_bytes([chunk[0], chunk[1]])
        } else {
            u16::from_be_bytes([chunk[0], 0])
        };
        sum = sum.wrapping_add(u32::from(word));
    }

    // Fold to 16 bits
    while sum > 0xffff {
        sum = (sum & 0xffff) + (sum >> 16);
    }

    let checksum = !sum as u16;
    if checksum == 0 { 0xffff } else { checksum }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nat_id_from_checksums() {
        let nat_id = NatId::from_checksums(0x1234, 0x1234);
        assert!(!nat_id.is_natted());
        assert_eq!(nat_id.value(), 0);

        let nat_id = NatId::from_checksums(0x1234, 0x1235);
        assert!(nat_id.is_natted());
        assert_eq!(nat_id.value(), 1);
    }

    #[test]
    fn test_ip_id_marker() {
        let marker = IpIdMarker::new(0xABCD, 64, 12345);
        assert_eq!(marker.ip_id, 0xABCD);
        assert!(marker.matches(0xABCD));
        assert!(!marker.matches(0x1234));
    }

    #[test]
    fn test_nat_detection_state() {
        let mut state = NatDetectionState::new();

        // First hop with no NAT
        let nat_detected = state.update_hop(0, NatId(0));
        assert!(!nat_detected);
        assert!(!state.has_nat());

        // Second hop with same NAT ID
        let nat_detected = state.update_hop(1, NatId(0));
        assert!(!nat_detected);

        // Third hop with different NAT ID (NAT detected!)
        let nat_detected = state.update_hop(2, NatId(100));
        assert!(nat_detected);
        assert!(state.has_nat());
        assert_eq!(state.nat_count(), 1);
        assert_eq!(state.nat_locations(), &[2]);
    }

    #[test]
    fn test_probe_matcher() {
        let mut matcher = ProbeMatcher::new(Duration::from_secs(5));

        let marker = IpIdMarker::new(0xABCD, 64, 12345);
        let probe = NatProbe::new(
            marker,
            SocketAddr::from(([192, 168, 1, 1], 12345)),
            SocketAddr::from(([8, 8, 8, 8], 53)),
            vec![0; 8],
        );

        matcher.register_probe(probe);

        // Match by IP ID
        let response = NatProbeResponse {
            inner_ip_id: 0xABCD,
            inner_udp_checksum: 0xABCD,
            responder_addr: SocketAddr::from(([1, 2, 3, 4], 0)),
            response_ip_id: 0,
            received_at: Instant::now(),
            icmp_type: 11,
            icmp_code: 0,
        };

        let matched = matcher.match_response(&response);
        assert!(matched.is_some());
        assert_eq!(matcher.pending_count(), 0);
    }

    #[test]
    fn test_uplink_nat_state() {
        let mut state = UplinkNatState::new();
        assert!(!state.is_natted());
        assert_eq!(state.nat_type(), NatType::Unknown);

        state.set_nat_type(NatType::FullCone);
        assert!(state.is_natted());
        assert_eq!(state.nat_type(), NatType::FullCone);

        state.set_external_addr(SocketAddr::from(([1, 2, 3, 4], 12345)));
        assert!(state.external_addr().is_some());

        state.reset();
        assert!(!state.is_natted());
        assert!(state.external_addr().is_none());
    }

    #[test]
    fn test_nat_type() {
        assert!(NatType::None.allows_direct_connect());
        assert!(NatType::FullCone.allows_direct_connect());
        assert!(!NatType::Symmetric.allows_direct_connect());

        assert!(NatType::RestrictedCone.requires_hole_punch());
        assert!(NatType::Symmetric.requires_relay());
    }

    #[test]
    fn test_udp_checksum() {
        // Simple test case
        let src = [192, 168, 1, 1];
        let dst = [8, 8, 8, 8];
        let checksum = compute_udp_checksum(&src, &dst, 12345, 53, b"test");
        assert_ne!(checksum, 0);
    }

    #[test]
    fn test_probe_payload_creation() {
        let payload = NatProbe::create_payload(b"HELLO", 64, 12345);
        assert_eq!(payload.len(), 7); // 5 + 2 bytes for ID
    }
}
