//! IP packet parsing and manipulation.
//!
//! Provides utilities for parsing IPv4/IPv6 packets and extracting
//! flow information (5-tuple) for ECMP-aware routing.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use crate::error::{Error, Result};

/// IP version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IpVersion {
    V4,
    V6,
}

/// Transport layer protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportProtocol {
    Tcp,
    Udp,
    Icmp,
    Icmpv6,
    Other(u8),
}

impl TransportProtocol {
    /// Get the IP protocol number.
    pub fn protocol_number(&self) -> u8 {
        match self {
            TransportProtocol::Tcp => 6,
            TransportProtocol::Udp => 17,
            TransportProtocol::Icmp => 1,
            TransportProtocol::Icmpv6 => 58,
            TransportProtocol::Other(n) => *n,
        }
    }

    /// Create from IP protocol number.
    pub fn from_protocol_number(n: u8) -> Self {
        match n {
            6 => TransportProtocol::Tcp,
            17 => TransportProtocol::Udp,
            1 => TransportProtocol::Icmp,
            58 => TransportProtocol::Icmpv6,
            _ => TransportProtocol::Other(n),
        }
    }
}

/// 5-tuple flow identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FlowTuple {
    /// Source IP address.
    pub src_addr: IpAddr,
    /// Destination IP address.
    pub dst_addr: IpAddr,
    /// Source port (0 for ICMP).
    pub src_port: u16,
    /// Destination port (0 for ICMP).
    pub dst_port: u16,
    /// Transport protocol.
    pub protocol: TransportProtocol,
}

impl FlowTuple {
    /// Create a new flow tuple.
    pub fn new(
        src_addr: IpAddr,
        dst_addr: IpAddr,
        src_port: u16,
        dst_port: u16,
        protocol: TransportProtocol,
    ) -> Self {
        Self {
            src_addr,
            dst_addr,
            src_port,
            dst_port,
            protocol,
        }
    }

    /// Compute a hash for ECMP load balancing.
    ///
    /// This produces a consistent hash that can be used to select
    /// an uplink, ensuring packets of the same flow use the same path.
    pub fn flow_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        self.src_addr.hash(&mut hasher);
        self.dst_addr.hash(&mut hasher);
        self.src_port.hash(&mut hasher);
        self.dst_port.hash(&mut hasher);
        self.protocol.protocol_number().hash(&mut hasher);
        hasher.finish()
    }

    /// Get the reverse flow (swap src/dst).
    pub fn reverse(&self) -> Self {
        Self {
            src_addr: self.dst_addr,
            dst_addr: self.src_addr,
            src_port: self.dst_port,
            dst_port: self.src_port,
            protocol: self.protocol,
        }
    }
}

/// Parsed IP packet header information.
#[derive(Debug, Clone)]
pub struct IpPacket<'a> {
    /// IP version.
    pub version: IpVersion,
    /// Header length in bytes.
    pub header_len: usize,
    /// Total packet length.
    pub total_len: usize,
    /// Time to live / hop limit.
    pub ttl: u8,
    /// Transport protocol.
    pub protocol: TransportProtocol,
    /// Source address.
    pub src_addr: IpAddr,
    /// Destination address.
    pub dst_addr: IpAddr,
    /// Source port (if TCP/UDP).
    pub src_port: Option<u16>,
    /// Destination port (if TCP/UDP).
    pub dst_port: Option<u16>,
    /// Reference to the raw packet data.
    pub data: &'a [u8],
    /// Offset where payload begins.
    pub payload_offset: usize,
}

impl<'a> IpPacket<'a> {
    /// Parse an IP packet from raw bytes.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::Protocol(crate::error::ProtocolError::MalformedPacket(
                "Empty packet".into()
            )));
        }

        let version = (data[0] >> 4) & 0x0f;

        match version {
            4 => Self::parse_ipv4(data),
            6 => Self::parse_ipv6(data),
            _ => Err(Error::Protocol(crate::error::ProtocolError::MalformedPacket(
                format!("Unknown IP version: {}", version)
            ))),
        }
    }

    /// Parse an IPv4 packet.
    fn parse_ipv4(data: &'a [u8]) -> Result<Self> {
        if data.len() < 20 {
            return Err(Error::Protocol(crate::error::ProtocolError::MalformedPacket(
                "IPv4 packet too short".into()
            )));
        }

        let ihl = (data[0] & 0x0f) as usize;
        let header_len = ihl * 4;

        if data.len() < header_len {
            return Err(Error::Protocol(crate::error::ProtocolError::MalformedPacket(
                "IPv4 header truncated".into()
            )));
        }

        let total_len = u16::from_be_bytes([data[2], data[3]]) as usize;
        let ttl = data[8];
        let protocol_num = data[9];
        let protocol = TransportProtocol::from_protocol_number(protocol_num);

        let src_addr = IpAddr::V4(Ipv4Addr::new(data[12], data[13], data[14], data[15]));
        let dst_addr = IpAddr::V4(Ipv4Addr::new(data[16], data[17], data[18], data[19]));

        // Parse transport layer ports if TCP or UDP
        let (src_port, dst_port, payload_offset) = if data.len() >= header_len + 4 {
            match protocol {
                TransportProtocol::Tcp | TransportProtocol::Udp => {
                    let sport = u16::from_be_bytes([data[header_len], data[header_len + 1]]);
                    let dport = u16::from_be_bytes([data[header_len + 2], data[header_len + 3]]);
                    
                    // For TCP, skip the header (minimum 20 bytes)
                    // For UDP, skip 8 bytes
                    let transport_header = match protocol {
                        TransportProtocol::Tcp => {
                            if data.len() >= header_len + 12 {
                                let data_offset = ((data[header_len + 12] >> 4) & 0x0f) as usize * 4;
                                data_offset
                            } else {
                                20
                            }
                        }
                        TransportProtocol::Udp => 8,
                        _ => 0,
                    };
                    
                    (Some(sport), Some(dport), header_len + transport_header)
                }
                _ => (None, None, header_len),
            }
        } else {
            (None, None, header_len)
        };

        Ok(Self {
            version: IpVersion::V4,
            header_len,
            total_len,
            ttl,
            protocol,
            src_addr,
            dst_addr,
            src_port,
            dst_port,
            data,
            payload_offset,
        })
    }

    /// Parse an IPv6 packet.
    fn parse_ipv6(data: &'a [u8]) -> Result<Self> {
        if data.len() < 40 {
            return Err(Error::Protocol(crate::error::ProtocolError::MalformedPacket(
                "IPv6 packet too short".into()
            )));
        }

        let payload_len = u16::from_be_bytes([data[4], data[5]]) as usize;
        let next_header = data[6];
        let hop_limit = data[7];

        let mut src_bytes = [0u8; 16];
        let mut dst_bytes = [0u8; 16];
        src_bytes.copy_from_slice(&data[8..24]);
        dst_bytes.copy_from_slice(&data[24..40]);

        let src_addr = IpAddr::V6(Ipv6Addr::from(src_bytes));
        let dst_addr = IpAddr::V6(Ipv6Addr::from(dst_bytes));

        // Handle extension headers (simplified - just check immediate next header)
        let (protocol, header_len) = Self::skip_ipv6_extension_headers(data, next_header, 40)?;

        // Parse transport layer ports
        let (src_port, dst_port, payload_offset) = if data.len() >= header_len + 4 {
            match protocol {
                TransportProtocol::Tcp | TransportProtocol::Udp => {
                    let sport = u16::from_be_bytes([data[header_len], data[header_len + 1]]);
                    let dport = u16::from_be_bytes([data[header_len + 2], data[header_len + 3]]);
                    
                    let transport_header = match protocol {
                        TransportProtocol::Tcp => {
                            if data.len() >= header_len + 12 {
                                let data_offset = ((data[header_len + 12] >> 4) & 0x0f) as usize * 4;
                                data_offset
                            } else {
                                20
                            }
                        }
                        TransportProtocol::Udp => 8,
                        _ => 0,
                    };
                    
                    (Some(sport), Some(dport), header_len + transport_header)
                }
                _ => (None, None, header_len),
            }
        } else {
            (None, None, header_len)
        };

        Ok(Self {
            version: IpVersion::V6,
            header_len,
            total_len: 40 + payload_len,
            ttl: hop_limit,
            protocol,
            src_addr,
            dst_addr,
            src_port,
            dst_port,
            data,
            payload_offset,
        })
    }

    /// Skip IPv6 extension headers to find the transport protocol.
    fn skip_ipv6_extension_headers(
        data: &[u8],
        next_header: u8,
        mut offset: usize,
    ) -> Result<(TransportProtocol, usize)> {
        let mut current_header = next_header;

        // Extension header types that we need to skip
        const HOP_BY_HOP: u8 = 0;
        const ROUTING: u8 = 43;
        const FRAGMENT: u8 = 44;
        const DESTINATION: u8 = 60;

        loop {
            match current_header {
                HOP_BY_HOP | ROUTING | DESTINATION => {
                    if data.len() < offset + 2 {
                        break;
                    }
                    current_header = data[offset];
                    let ext_len = (data[offset + 1] as usize + 1) * 8;
                    offset += ext_len;
                }
                FRAGMENT => {
                    if data.len() < offset + 8 {
                        break;
                    }
                    current_header = data[offset];
                    offset += 8;
                }
                _ => break,
            }

            if offset >= data.len() {
                break;
            }
        }

        Ok((TransportProtocol::from_protocol_number(current_header), offset))
    }

    /// Get the flow tuple for this packet.
    pub fn flow_tuple(&self) -> FlowTuple {
        FlowTuple::new(
            self.src_addr,
            self.dst_addr,
            self.src_port.unwrap_or(0),
            self.dst_port.unwrap_or(0),
            self.protocol,
        )
    }

    /// Get the payload (data after IP and transport headers).
    pub fn payload(&self) -> &[u8] {
        if self.payload_offset < self.data.len() {
            &self.data[self.payload_offset..]
        } else {
            &[]
        }
    }

    /// Check if this is a TCP SYN packet (connection start).
    pub fn is_tcp_syn(&self) -> bool {
        if self.protocol != TransportProtocol::Tcp {
            return false;
        }

        let tcp_offset = self.header_len;
        if self.data.len() < tcp_offset + 14 {
            return false;
        }

        let flags = self.data[tcp_offset + 13];
        // SYN flag is bit 1 (0x02), check SYN set and ACK not set
        (flags & 0x02) != 0 && (flags & 0x10) == 0
    }

    /// Check if this is a TCP FIN or RST packet (connection end).
    pub fn is_tcp_fin_or_rst(&self) -> bool {
        if self.protocol != TransportProtocol::Tcp {
            return false;
        }

        let tcp_offset = self.header_len;
        if self.data.len() < tcp_offset + 14 {
            return false;
        }

        let flags = self.data[tcp_offset + 13];
        // FIN is bit 0 (0x01), RST is bit 2 (0x04)
        (flags & 0x01) != 0 || (flags & 0x04) != 0
    }

    /// Check if this is a DNS packet (UDP port 53).
    pub fn is_dns(&self) -> bool {
        self.protocol == TransportProtocol::Udp
            && (self.src_port == Some(53) || self.dst_port == Some(53))
    }
}

/// Mutable IP packet for modification.
pub struct IpPacketMut<'a> {
    data: &'a mut [u8],
    version: IpVersion,
    header_len: usize,
}

impl<'a> IpPacketMut<'a> {
    /// Create a mutable packet wrapper.
    pub fn new(data: &'a mut [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::Protocol(crate::error::ProtocolError::MalformedPacket(
                "Empty packet".into()
            )));
        }

        let version = match (data[0] >> 4) & 0x0f {
            4 => IpVersion::V4,
            6 => IpVersion::V6,
            v => return Err(Error::Protocol(crate::error::ProtocolError::MalformedPacket(
                format!("Unknown IP version: {}", v)
            ))),
        };

        let header_len = match version {
            IpVersion::V4 => ((data[0] & 0x0f) as usize) * 4,
            IpVersion::V6 => 40,
        };

        Ok(Self {
            data,
            version,
            header_len,
        })
    }

    /// Set the source address.
    pub fn set_src_addr(&mut self, addr: IpAddr) -> Result<()> {
        match (self.version, addr) {
            (IpVersion::V4, IpAddr::V4(ipv4)) => {
                self.data[12..16].copy_from_slice(&ipv4.octets());
                self.update_ipv4_checksum();
                Ok(())
            }
            (IpVersion::V6, IpAddr::V6(ipv6)) => {
                self.data[8..24].copy_from_slice(&ipv6.octets());
                Ok(())
            }
            _ => Err(Error::Protocol(crate::error::ProtocolError::MalformedPacket(
                "Address version mismatch".into()
            ))),
        }
    }

    /// Set the destination address.
    pub fn set_dst_addr(&mut self, addr: IpAddr) -> Result<()> {
        match (self.version, addr) {
            (IpVersion::V4, IpAddr::V4(ipv4)) => {
                self.data[16..20].copy_from_slice(&ipv4.octets());
                self.update_ipv4_checksum();
                Ok(())
            }
            (IpVersion::V6, IpAddr::V6(ipv6)) => {
                self.data[24..40].copy_from_slice(&ipv6.octets());
                Ok(())
            }
            _ => Err(Error::Protocol(crate::error::ProtocolError::MalformedPacket(
                "Address version mismatch".into()
            ))),
        }
    }

    /// Set the TTL/hop limit.
    pub fn set_ttl(&mut self, ttl: u8) {
        match self.version {
            IpVersion::V4 => {
                self.data[8] = ttl;
                self.update_ipv4_checksum();
            }
            IpVersion::V6 => {
                self.data[7] = ttl;
            }
        }
    }

    /// Update the IPv4 header checksum.
    fn update_ipv4_checksum(&mut self) {
        if self.version != IpVersion::V4 || self.data.len() < self.header_len {
            return;
        }

        // Zero out existing checksum
        self.data[10] = 0;
        self.data[11] = 0;

        // Calculate new checksum
        let mut sum: u32 = 0;
        for i in (0..self.header_len).step_by(2) {
            let word = if i + 1 < self.header_len {
                u16::from_be_bytes([self.data[i], self.data[i + 1]])
            } else {
                u16::from_be_bytes([self.data[i], 0])
            };
            sum += word as u32;
        }

        // Fold 32-bit sum to 16 bits
        while sum >> 16 != 0 {
            sum = (sum & 0xffff) + (sum >> 16);
        }

        let checksum = !(sum as u16);
        self.data[10..12].copy_from_slice(&checksum.to_be_bytes());
    }

    /// Update TCP/UDP checksum after NAT modifications.
    ///
    /// This uses incremental checksum update for efficiency.
    pub fn update_transport_checksum(&mut self, old_addr: IpAddr, new_addr: IpAddr) -> Result<()> {
        // Determine transport protocol and checksum offset
        let (protocol, checksum_offset) = match self.version {
            IpVersion::V4 => {
                let proto = self.data[9];
                let ihl = ((self.data[0] & 0x0f) as usize) * 4;
                match proto {
                    6 => (TransportProtocol::Tcp, ihl + 16),  // TCP checksum at offset 16
                    17 => (TransportProtocol::Udp, ihl + 6), // UDP checksum at offset 6
                    _ => return Ok(()), // No checksum to update
                }
            }
            IpVersion::V6 => {
                let proto = self.data[6];
                match proto {
                    6 => (TransportProtocol::Tcp, 40 + 16),
                    17 => (TransportProtocol::Udp, 40 + 6),
                    _ => return Ok(()),
                }
            }
        };

        if self.data.len() < checksum_offset + 2 {
            return Ok(());
        }

        // Get old checksum
        let old_checksum = u16::from_be_bytes([
            self.data[checksum_offset],
            self.data[checksum_offset + 1],
        ]);

        // Skip if checksum is zero (UDP optional checksum)
        if protocol == TransportProtocol::Udp && old_checksum == 0 {
            return Ok(());
        }

        // Incremental checksum update (RFC 1624)
        let new_checksum = incremental_checksum_update(old_checksum, old_addr, new_addr);

        self.data[checksum_offset..checksum_offset + 2]
            .copy_from_slice(&new_checksum.to_be_bytes());

        Ok(())
    }

    /// Get the underlying data.
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Get mutable access to the underlying data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }
}

/// Incremental checksum update when changing an IP address.
fn incremental_checksum_update(old_checksum: u16, old_addr: IpAddr, new_addr: IpAddr) -> u16 {
    // Convert checksum to host order
    let mut sum = !old_checksum as u32;

    // Subtract old address words
    match old_addr {
        IpAddr::V4(ipv4) => {
            let octets = ipv4.octets();
            sum = sum.wrapping_sub(u16::from_be_bytes([octets[0], octets[1]]) as u32);
            sum = sum.wrapping_sub(u16::from_be_bytes([octets[2], octets[3]]) as u32);
        }
        IpAddr::V6(ipv6) => {
            let octets = ipv6.octets();
            for i in (0..16).step_by(2) {
                sum = sum.wrapping_sub(u16::from_be_bytes([octets[i], octets[i + 1]]) as u32);
            }
        }
    }

    // Add new address words
    match new_addr {
        IpAddr::V4(ipv4) => {
            let octets = ipv4.octets();
            sum = sum.wrapping_add(u16::from_be_bytes([octets[0], octets[1]]) as u32);
            sum = sum.wrapping_add(u16::from_be_bytes([octets[2], octets[3]]) as u32);
        }
        IpAddr::V6(ipv6) => {
            let octets = ipv6.octets();
            for i in (0..16).step_by(2) {
                sum = sum.wrapping_add(u16::from_be_bytes([octets[i], octets[i + 1]]) as u32);
            }
        }
    }

    // Fold and complement
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }

    !(sum as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sample IPv4 TCP SYN packet
    const IPV4_TCP_SYN: &[u8] = &[
        0x45, 0x00, 0x00, 0x3c, // Version, IHL, TOS, Total Length
        0x1c, 0x46, 0x40, 0x00, // ID, Flags, Fragment Offset
        0x40, 0x06, 0x00, 0x00, // TTL, Protocol (TCP), Checksum (placeholder)
        0xc0, 0xa8, 0x01, 0x01, // Source IP: 192.168.1.1
        0x08, 0x08, 0x08, 0x08, // Dest IP: 8.8.8.8
        // TCP header (simplified)
        0x04, 0x00, // Source port: 1024
        0x00, 0x50, // Dest port: 80
        0x00, 0x00, 0x00, 0x00, // Sequence number
        0x00, 0x00, 0x00, 0x00, // Ack number
        0x50, 0x02, 0x00, 0x00, // Data offset, flags (SYN), window
        0x00, 0x00, 0x00, 0x00, // Checksum, urgent pointer
    ];

    #[test]
    fn test_parse_ipv4() {
        let packet = IpPacket::parse(IPV4_TCP_SYN).unwrap();
        
        assert_eq!(packet.version, IpVersion::V4);
        assert_eq!(packet.header_len, 20);
        assert_eq!(packet.protocol, TransportProtocol::Tcp);
        assert_eq!(packet.src_addr, IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)));
        assert_eq!(packet.dst_addr, IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)));
        assert_eq!(packet.src_port, Some(1024));
        assert_eq!(packet.dst_port, Some(80));
        assert_eq!(packet.ttl, 64);
    }

    #[test]
    fn test_flow_tuple() {
        let packet = IpPacket::parse(IPV4_TCP_SYN).unwrap();
        let flow = packet.flow_tuple();
        
        assert_eq!(flow.src_addr, IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)));
        assert_eq!(flow.dst_addr, IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)));
        assert_eq!(flow.src_port, 1024);
        assert_eq!(flow.dst_port, 80);
        assert_eq!(flow.protocol, TransportProtocol::Tcp);
    }

    #[test]
    fn test_flow_hash_consistency() {
        let packet = IpPacket::parse(IPV4_TCP_SYN).unwrap();
        let flow = packet.flow_tuple();
        
        // Same flow should produce same hash
        let hash1 = flow.flow_hash();
        let hash2 = flow.flow_hash();
        assert_eq!(hash1, hash2);
        
        // Reverse flow should produce different hash
        let reverse = flow.reverse();
        let hash3 = reverse.flow_hash();
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_is_tcp_syn() {
        let packet = IpPacket::parse(IPV4_TCP_SYN).unwrap();
        assert!(packet.is_tcp_syn());
    }

    #[test]
    fn test_transport_protocol() {
        assert_eq!(TransportProtocol::Tcp.protocol_number(), 6);
        assert_eq!(TransportProtocol::Udp.protocol_number(), 17);
        assert_eq!(TransportProtocol::from_protocol_number(6), TransportProtocol::Tcp);
    }
}
