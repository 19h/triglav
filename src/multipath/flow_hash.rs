//! Flow hash calculation for ECMP-aware path selection.
//!
//! Implements flow hash calculation compatible with common ECMP implementations
//! used by network devices. Packets with the same flow hash will traverse the
//! same network path through ECMP routers.
//!
//! Based on Dublin Traceroute's flow hash calculation technique.

use std::net::{IpAddr, SocketAddr};

/// Flow identifier containing the 5-tuple that determines ECMP path selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlowId {
    /// Source IP address.
    pub src_ip: IpAddr,
    /// Destination IP address.
    pub dst_ip: IpAddr,
    /// Source port.
    pub src_port: u16,
    /// Destination port.
    pub dst_port: u16,
    /// IP protocol (6 = TCP, 17 = UDP).
    pub protocol: u8,
}

impl FlowId {
    /// Create a new flow identifier.
    pub fn new(src_ip: IpAddr, dst_ip: IpAddr, src_port: u16, dst_port: u16, protocol: u8) -> Self {
        Self {
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            protocol,
        }
    }

    /// Create from socket addresses with UDP protocol.
    pub fn from_udp(src: SocketAddr, dst: SocketAddr) -> Self {
        Self::new(src.ip(), dst.ip(), src.port(), dst.port(), 17)
    }

    /// Create from socket addresses with TCP protocol.
    pub fn from_tcp(src: SocketAddr, dst: SocketAddr) -> Self {
        Self::new(src.ip(), dst.ip(), src.port(), dst.port(), 6)
    }

    /// Calculate the ECMP-compatible flow hash.
    ///
    /// This hash mimics common router ECMP implementations:
    /// - Uses all 5-tuple fields
    /// - Produces consistent results for the same flow
    /// - Returns 0xffff if the calculated hash is 0 (never returns 0)
    ///
    /// Two packets with the same flow hash will follow the same path through
    /// ECMP-enabled network devices.
    pub fn flow_hash(&self) -> u16 {
        let hash = self.compute_hash();
        if hash == 0 {
            0xffff
        } else {
            hash
        }
    }

    /// Internal hash computation matching Dublin Traceroute algorithm.
    fn compute_hash(&self) -> u16 {
        let mut hash: u32 = 0;

        // Add TOS (0 for our purposes) + protocol
        hash = hash.wrapping_add(u32::from(self.protocol));

        // Add source IP
        match self.src_ip {
            IpAddr::V4(addr) => {
                let octets = addr.octets();
                hash = hash.wrapping_add(u32::from(u16::from_be_bytes([octets[0], octets[1]])));
                hash = hash.wrapping_add(u32::from(u16::from_be_bytes([octets[2], octets[3]])));
            }
            IpAddr::V6(addr) => {
                let octets = addr.octets();
                for chunk in octets.chunks(2) {
                    hash = hash.wrapping_add(u32::from(u16::from_be_bytes([chunk[0], chunk[1]])));
                }
            }
        }

        // Add destination IP
        match self.dst_ip {
            IpAddr::V4(addr) => {
                let octets = addr.octets();
                hash = hash.wrapping_add(u32::from(u16::from_be_bytes([octets[0], octets[1]])));
                hash = hash.wrapping_add(u32::from(u16::from_be_bytes([octets[2], octets[3]])));
            }
            IpAddr::V6(addr) => {
                let octets = addr.octets();
                for chunk in octets.chunks(2) {
                    hash = hash.wrapping_add(u32::from(u16::from_be_bytes([chunk[0], chunk[1]])));
                }
            }
        }

        // Add ports
        hash = hash.wrapping_add(u32::from(self.src_port));
        hash = hash.wrapping_add(u32::from(self.dst_port));

        // Fold to 16 bits
        while hash > 0xffff {
            hash = (hash & 0xffff) + (hash >> 16);
        }

        hash as u16
    }

    /// Generate a variant flow that will take a different ECMP path.
    ///
    /// By modifying the source port, we can probe different ECMP paths
    /// while keeping the same destination.
    pub fn with_src_port(&self, port: u16) -> Self {
        Self {
            src_port: port,
            ..*self
        }
    }

    /// Generate a variant flow with different destination port.
    pub fn with_dst_port(&self, port: u16) -> Self {
        Self {
            dst_port: port,
            ..*self
        }
    }
}

/// Calculate flow hash for a packet's 5-tuple.
///
/// This is a convenience function for quick hash calculation.
pub fn calculate_flow_hash(
    src_ip: IpAddr,
    dst_ip: IpAddr,
    src_port: u16,
    dst_port: u16,
    protocol: u8,
) -> u16 {
    FlowId::new(src_ip, dst_ip, src_port, dst_port, protocol).flow_hash()
}

/// Calculate flow hash from socket addresses.
pub fn flow_hash_from_addrs(src: SocketAddr, dst: SocketAddr, is_tcp: bool) -> u16 {
    if is_tcp {
        FlowId::from_tcp(src, dst).flow_hash()
    } else {
        FlowId::from_udp(src, dst).flow_hash()
    }
}

/// ECMP path enumerator.
///
/// Generates a sequence of port values that will produce different flow hashes,
/// allowing enumeration of multiple ECMP paths to a destination.
#[derive(Debug)]
pub struct EcmpPathEnumerator {
    base_flow: FlowId,
    base_port: u16,
    num_paths: u16,
    use_src_port: bool,
}

impl EcmpPathEnumerator {
    /// Create a new ECMP path enumerator.
    ///
    /// # Arguments
    /// * `base_flow` - The base flow to vary
    /// * `base_port` - Starting port number
    /// * `num_paths` - Number of paths to enumerate
    /// * `use_src_port` - If true, vary source port; if false, vary destination port
    pub fn new(base_flow: FlowId, base_port: u16, num_paths: u16, use_src_port: bool) -> Self {
        Self {
            base_flow,
            base_port,
            num_paths,
            use_src_port,
        }
    }

    /// Get flow IDs for all paths.
    pub fn flows(&self) -> Vec<FlowId> {
        (0..self.num_paths)
            .map(|i| {
                let port = self.base_port.wrapping_add(i);
                if self.use_src_port {
                    self.base_flow.with_src_port(port)
                } else {
                    self.base_flow.with_dst_port(port)
                }
            })
            .collect()
    }

    /// Get unique flow hashes (deduplicated).
    pub fn unique_hashes(&self) -> Vec<u16> {
        let mut hashes: Vec<_> = self.flows().iter().map(FlowId::flow_hash).collect();
        hashes.sort_unstable();
        hashes.dedup();
        hashes
    }

    /// Estimate number of unique ECMP paths based on hash diversity.
    pub fn estimated_path_count(&self) -> usize {
        self.unique_hashes().len()
    }
}

/// Flow hash bucket for grouping packets by ECMP path.
#[derive(Debug, Default)]
pub struct FlowHashBucket {
    /// Flows grouped by their hash.
    buckets: std::collections::HashMap<u16, Vec<FlowId>>,
}

impl FlowHashBucket {
    /// Create a new flow hash bucket.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a flow to its bucket.
    pub fn add(&mut self, flow: FlowId) {
        let hash = flow.flow_hash();
        self.buckets.entry(hash).or_default().push(flow);
    }

    /// Get the number of unique buckets (paths).
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Get flows in a specific bucket.
    pub fn get_bucket(&self, hash: u16) -> Option<&Vec<FlowId>> {
        self.buckets.get(&hash)
    }

    /// Iterate over all buckets.
    pub fn iter(&self) -> impl Iterator<Item = (&u16, &Vec<FlowId>)> {
        self.buckets.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn test_flow_hash_consistency() {
        let flow = FlowId::new(
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            12345,
            80,
            6,
        );

        let hash1 = flow.flow_hash();
        let hash2 = flow.flow_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_different_ports_different_hash() {
        let flow1 = FlowId::new(
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            12345,
            80,
            17,
        );

        let flow2 = flow1.with_src_port(12346);

        // Different ports should (usually) produce different hashes
        // Note: hash collisions are possible but unlikely for adjacent ports
        let hash1 = flow1.flow_hash();
        let hash2 = flow2.flow_hash();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_flow_hash_never_zero() {
        // Test many flows to ensure we never get 0
        for port in 1..1000 {
            let flow = FlowId::new(
                IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
                IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
                port,
                port,
                0,
            );
            assert_ne!(flow.flow_hash(), 0);
        }
    }

    #[test]
    fn test_ipv6_flow_hash() {
        let flow = FlowId::new(
            IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1)),
            IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 2)),
            12345,
            443,
            6,
        );

        let hash = flow.flow_hash();
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_ecmp_enumerator() {
        let base = FlowId::new(
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            10000,
            53,
            17,
        );

        let enumerator = EcmpPathEnumerator::new(base, 10000, 16, true);
        let flows = enumerator.flows();
        assert_eq!(flows.len(), 16);

        // Should have multiple unique hashes
        let unique = enumerator.estimated_path_count();
        assert!(unique > 1);
    }

    #[test]
    fn test_flow_hash_bucket() {
        let mut bucket = FlowHashBucket::new();

        for port in 10000..10016 {
            let flow = FlowId::new(
                IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
                IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
                port,
                80,
                6,
            );
            bucket.add(flow);
        }

        // Should have grouped flows by hash
        assert!(bucket.bucket_count() > 0);
    }

    #[test]
    fn test_from_socket_addrs() {
        let src = SocketAddr::from(([192, 168, 1, 1], 12345));
        let dst = SocketAddr::from(([10, 0, 0, 1], 80));

        let flow_tcp = FlowId::from_tcp(src, dst);
        let flow_udp = FlowId::from_udp(src, dst);

        assert_eq!(flow_tcp.protocol, 6);
        assert_eq!(flow_udp.protocol, 17);

        // Different protocols should produce different hashes
        assert_ne!(flow_tcp.flow_hash(), flow_udp.flow_hash());
    }
}
