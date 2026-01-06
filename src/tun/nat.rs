//! NAT (Network Address Translation) for TUN tunnel.
//!
//! Provides address translation between local tunnel addresses and
//! addresses visible to the remote server.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::packet::{FlowTuple, IpPacket, IpPacketMut, TransportProtocol};
use crate::error::{Error, Result};

/// NAT configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatConfig {
    /// Tunnel IPv4 address (local side).
    #[serde(default = "default_tunnel_ipv4")]
    pub tunnel_ipv4: Ipv4Addr,

    /// Tunnel IPv6 address (local side).
    #[serde(default)]
    pub tunnel_ipv6: Option<Ipv6Addr>,

    /// Server-side IPv4 address (what the server sees).
    #[serde(default = "default_server_ipv4")]
    pub server_ipv4: Ipv4Addr,

    /// Server-side IPv6 address (what the server sees).
    #[serde(default)]
    pub server_ipv6: Option<Ipv6Addr>,

    /// NAT entry timeout for UDP flows.
    #[serde(default = "default_udp_timeout", with = "humantime_serde")]
    pub udp_timeout: Duration,

    /// NAT entry timeout for TCP flows.
    #[serde(default = "default_tcp_timeout", with = "humantime_serde")]
    pub tcp_timeout: Duration,

    /// NAT entry timeout for ICMP.
    #[serde(default = "default_icmp_timeout", with = "humantime_serde")]
    pub icmp_timeout: Duration,

    /// Port range start for NAT port allocation.
    #[serde(default = "default_port_start")]
    pub port_range_start: u16,

    /// Port range end for NAT port allocation.
    #[serde(default = "default_port_end")]
    pub port_range_end: u16,

    /// Enable hairpin NAT (traffic between local clients).
    #[serde(default)]
    pub hairpin_nat: bool,
}

fn default_tunnel_ipv4() -> Ipv4Addr {
    "10.0.85.1".parse().unwrap()
}

fn default_server_ipv4() -> Ipv4Addr {
    "10.0.85.2".parse().unwrap()
}

fn default_udp_timeout() -> Duration {
    Duration::from_secs(180)
}

fn default_tcp_timeout() -> Duration {
    Duration::from_secs(7200)
}

fn default_icmp_timeout() -> Duration {
    Duration::from_secs(60)
}

fn default_port_start() -> u16 {
    32768
}

fn default_port_end() -> u16 {
    61000
}

impl Default for NatConfig {
    fn default() -> Self {
        Self {
            tunnel_ipv4: default_tunnel_ipv4(),
            tunnel_ipv6: None,
            server_ipv4: default_server_ipv4(),
            server_ipv6: None,
            udp_timeout: default_udp_timeout(),
            tcp_timeout: default_tcp_timeout(),
            icmp_timeout: default_icmp_timeout(),
            port_range_start: default_port_start(),
            port_range_end: default_port_end(),
            hairpin_nat: false,
        }
    }
}

/// NAT entry representing a translated flow.
#[derive(Debug, Clone)]
pub struct NatEntry {
    /// Original source address (local client).
    pub original_src: IpAddr,
    /// Original source port.
    pub original_src_port: u16,
    /// Translated source address (tunnel address).
    pub translated_src: IpAddr,
    /// Translated source port.
    pub translated_src_port: u16,
    /// Destination address.
    pub dst_addr: IpAddr,
    /// Destination port.
    pub dst_port: u16,
    /// Protocol.
    pub protocol: TransportProtocol,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry was last used.
    pub last_used: Instant,
    /// Number of packets translated.
    pub packet_count: u64,
    /// Number of bytes translated.
    pub byte_count: u64,
}

impl NatEntry {
    fn new(
        original_src: IpAddr,
        original_src_port: u16,
        translated_src: IpAddr,
        translated_src_port: u16,
        dst_addr: IpAddr,
        dst_port: u16,
        protocol: TransportProtocol,
    ) -> Self {
        let now = Instant::now();
        Self {
            original_src,
            original_src_port,
            translated_src,
            translated_src_port,
            dst_addr,
            dst_port,
            protocol,
            created_at: now,
            last_used: now,
            packet_count: 0,
            byte_count: 0,
        }
    }

    fn touch(&mut self, bytes: usize) {
        self.last_used = Instant::now();
        self.packet_count += 1;
        self.byte_count += bytes as u64;
    }

    fn is_expired(&self, timeout: Duration) -> bool {
        self.last_used.elapsed() > timeout
    }
}

/// Key for NAT lookup (outbound direction).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct OutboundKey {
    src_addr: IpAddr,
    src_port: u16,
    dst_addr: IpAddr,
    dst_port: u16,
    protocol: u8,
}

/// Key for NAT lookup (inbound direction).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct InboundKey {
    translated_port: u16,
    src_addr: IpAddr,
    src_port: u16,
    protocol: u8,
}

/// NAT table for bidirectional address translation.
pub struct NatTable {
    config: NatConfig,

    /// Outbound mappings: original flow -> NAT entry.
    outbound: DashMap<OutboundKey, NatEntry>,

    /// Inbound mappings: translated port + remote -> original.
    inbound: DashMap<InboundKey, OutboundKey>,

    /// Next port to allocate.
    next_port: AtomicU16,

    /// Statistics.
    stats: RwLock<NatStats>,
}

/// NAT statistics.
#[derive(Debug, Clone, Default)]
pub struct NatStats {
    pub active_entries: usize,
    pub total_translations: u64,
    pub packets_outbound: u64,
    pub packets_inbound: u64,
    pub bytes_outbound: u64,
    pub bytes_inbound: u64,
}

impl NatTable {
    /// Create a new NAT table.
    pub fn new(config: NatConfig) -> Self {
        Self {
            next_port: AtomicU16::new(config.port_range_start),
            config,
            outbound: DashMap::new(),
            inbound: DashMap::new(),
            stats: RwLock::new(NatStats::default()),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &NatConfig {
        &self.config
    }

    /// Get statistics.
    pub fn stats(&self) -> NatStats {
        let mut stats = self.stats.read().clone();
        stats.active_entries = self.outbound.len();
        stats
    }

    /// Translate an outbound packet (local -> tunnel).
    ///
    /// Modifies the packet in place and returns the NAT entry.
    pub fn translate_outbound(&self, packet: &mut [u8]) -> Result<Option<NatEntry>> {
        // Parse packet to extract flow info, then drop borrow
        let (flow, header_len, protocol) = {
            let parsed = IpPacket::parse(packet)?;
            (parsed.flow_tuple(), parsed.header_len, parsed.protocol)
        };

        // Check if this needs NAT (only translate local addresses)
        if !self.is_local_address(flow.src_addr) {
            return Ok(None);
        }

        let key = OutboundKey {
            src_addr: flow.src_addr,
            src_port: flow.src_port,
            dst_addr: flow.dst_addr,
            dst_port: flow.dst_port,
            protocol: flow.protocol.protocol_number(),
        };

        // Look up or create NAT entry
        let entry = if let Some(mut existing) = self.outbound.get_mut(&key) {
            existing.touch(packet.len());
            existing.clone()
        } else {
            // Create new entry
            let translated_port = self.allocate_port()?;
            let translated_addr = self.get_tunnel_address(flow.src_addr);

            let entry = NatEntry::new(
                flow.src_addr,
                flow.src_port,
                translated_addr,
                translated_port,
                flow.dst_addr,
                flow.dst_port,
                flow.protocol,
            );

            // Store in both tables
            let inbound_key = InboundKey {
                translated_port,
                src_addr: flow.dst_addr,
                src_port: flow.dst_port,
                protocol: flow.protocol.protocol_number(),
            };

            self.outbound.insert(key.clone(), entry.clone());
            self.inbound.insert(inbound_key, key);

            tracing::debug!(
                original = %flow.src_addr,
                original_port = flow.src_port,
                translated = %entry.translated_src,
                translated_port = entry.translated_src_port,
                dst = %flow.dst_addr,
                dst_port = flow.dst_port,
                "Created NAT entry"
            );

            entry
        };

        // Modify packet
        let mut pkt = IpPacketMut::new(packet)?;

        // Update transport checksum first (before modifying addresses)
        pkt.update_transport_checksum(entry.original_src, entry.translated_src)?;

        // Set new source address
        pkt.set_src_addr(entry.translated_src)?;

        // Update source port in transport header
        self.set_src_port_direct(
            pkt.data_mut(),
            header_len,
            protocol,
            entry.translated_src_port,
        )?;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.packets_outbound += 1;
            stats.bytes_outbound += packet.len() as u64;
            stats.total_translations += 1;
        }

        Ok(Some(entry))
    }

    /// Translate an inbound packet (tunnel -> local).
    ///
    /// Modifies the packet in place and returns the original destination.
    pub fn translate_inbound(&self, packet: &mut [u8]) -> Result<Option<(IpAddr, u16)>> {
        // Parse packet to extract flow info, then drop borrow
        let (flow, header_len, protocol) = {
            let parsed = IpPacket::parse(packet)?;
            (parsed.flow_tuple(), parsed.header_len, parsed.protocol)
        };

        // The destination should be our tunnel address
        if !self.is_tunnel_address(flow.dst_addr) {
            return Ok(None);
        }

        // Look up by translated port + source
        let inbound_key = InboundKey {
            translated_port: flow.dst_port,
            src_addr: flow.src_addr,
            src_port: flow.src_port,
            protocol: flow.protocol.protocol_number(),
        };

        let outbound_key = match self.inbound.get(&inbound_key) {
            Some(key) => key.clone(),
            None => {
                tracing::trace!(
                    dst_port = flow.dst_port,
                    src = %flow.src_addr,
                    src_port = flow.src_port,
                    "No NAT entry for inbound packet"
                );
                return Ok(None);
            }
        };

        let entry = match self.outbound.get_mut(&outbound_key) {
            Some(mut e) => {
                e.touch(packet.len());
                e.clone()
            }
            None => return Ok(None),
        };

        // Modify packet to restore original destination
        let mut pkt = IpPacketMut::new(packet)?;

        // Update transport checksum first
        pkt.update_transport_checksum(entry.translated_src, entry.original_src)?;

        // Restore original destination address
        pkt.set_dst_addr(entry.original_src)?;

        // Restore original destination port
        self.set_dst_port_direct(
            pkt.data_mut(),
            header_len,
            protocol,
            entry.original_src_port,
        )?;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.packets_inbound += 1;
            stats.bytes_inbound += packet.len() as u64;
        }

        Ok(Some((entry.original_src, entry.original_src_port)))
    }

    /// Remove expired NAT entries.
    pub fn cleanup_expired(&self) {
        let mut to_remove = Vec::new();

        for entry in self.outbound.iter() {
            let timeout = match entry.protocol {
                TransportProtocol::Tcp => self.config.tcp_timeout,
                TransportProtocol::Udp => self.config.udp_timeout,
                TransportProtocol::Icmp | TransportProtocol::Icmpv6 => self.config.icmp_timeout,
                _ => self.config.udp_timeout,
            };

            if entry.is_expired(timeout) {
                to_remove.push(entry.key().clone());
            }
        }

        for key in to_remove {
            if let Some((_, entry)) = self.outbound.remove(&key) {
                // Also remove from inbound table
                let inbound_key = InboundKey {
                    translated_port: entry.translated_src_port,
                    src_addr: entry.dst_addr,
                    src_port: entry.dst_port,
                    protocol: entry.protocol.protocol_number(),
                };
                self.inbound.remove(&inbound_key);

                tracing::trace!(
                    original = %entry.original_src,
                    original_port = entry.original_src_port,
                    "Removed expired NAT entry"
                );
            }
        }
    }

    /// Get count of active NAT entries.
    pub fn entry_count(&self) -> usize {
        self.outbound.len()
    }

    /// Clear all NAT entries.
    pub fn clear(&self) {
        self.outbound.clear();
        self.inbound.clear();
    }

    // Private helpers

    fn is_local_address(&self, addr: IpAddr) -> bool {
        match addr {
            IpAddr::V4(ipv4) => {
                // RFC 1918 private addresses + link-local
                ipv4.is_private() || ipv4.is_loopback() || ipv4.is_link_local()
            }
            IpAddr::V6(ipv6) => {
                // ULA + link-local
                ipv6.is_loopback()
                    || (ipv6.segments()[0] & 0xfe00) == 0xfc00 // ULA
                    || (ipv6.segments()[0] & 0xffc0) == 0xfe80 // Link-local
            }
        }
    }

    fn is_tunnel_address(&self, addr: IpAddr) -> bool {
        match addr {
            IpAddr::V4(ipv4) => ipv4 == self.config.tunnel_ipv4,
            IpAddr::V6(ipv6) => self.config.tunnel_ipv6.map_or(false, |t| t == ipv6),
        }
    }

    fn get_tunnel_address(&self, original: IpAddr) -> IpAddr {
        match original {
            IpAddr::V4(_) => IpAddr::V4(self.config.tunnel_ipv4),
            IpAddr::V6(_) => self
                .config
                .tunnel_ipv6
                .map(IpAddr::V6)
                .unwrap_or_else(|| IpAddr::V4(self.config.tunnel_ipv4)),
        }
    }

    fn allocate_port(&self) -> Result<u16> {
        let range_size = self.config.port_range_end - self.config.port_range_start;

        for _ in 0..range_size {
            let port = self.next_port.fetch_add(1, Ordering::SeqCst);

            // Wrap around
            if port >= self.config.port_range_end {
                self.next_port
                    .store(self.config.port_range_start, Ordering::SeqCst);
            }

            // Check if port is available
            let in_use = self.outbound.iter().any(|e| e.translated_src_port == port);
            if !in_use {
                return Ok(port);
            }
        }

        Err(Error::Config("NAT port exhaustion".into()))
    }

    fn set_src_port_direct(
        &self,
        packet: &mut [u8],
        header_len: usize,
        protocol: TransportProtocol,
        port: u16,
    ) -> Result<()> {
        if packet.len() < header_len + 2 {
            return Ok(());
        }

        match protocol {
            TransportProtocol::Tcp | TransportProtocol::Udp => {
                packet[header_len..header_len + 2].copy_from_slice(&port.to_be_bytes());
                // Note: checksum update is handled separately
            }
            _ => {}
        }

        Ok(())
    }

    fn set_dst_port_direct(
        &self,
        packet: &mut [u8],
        header_len: usize,
        protocol: TransportProtocol,
        port: u16,
    ) -> Result<()> {
        if packet.len() < header_len + 4 {
            return Ok(());
        }

        match protocol {
            TransportProtocol::Tcp | TransportProtocol::Udp => {
                packet[header_len + 2..header_len + 4].copy_from_slice(&port.to_be_bytes());
                // Note: checksum update is handled separately
            }
            _ => {}
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nat_config_default() {
        let config = NatConfig::default();
        assert_eq!(config.tunnel_ipv4, Ipv4Addr::new(10, 0, 85, 1));
        assert_eq!(config.port_range_start, 32768);
    }

    #[test]
    fn test_nat_entry_timeout() {
        let entry = NatEntry::new(
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            12345,
            IpAddr::V4(Ipv4Addr::new(10, 0, 85, 1)),
            32768,
            IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            80,
            TransportProtocol::Tcp,
        );

        assert!(!entry.is_expired(Duration::from_secs(60)));
    }

    #[test]
    fn test_port_allocation() {
        let config = NatConfig {
            port_range_start: 1000,
            port_range_end: 1010,
            ..Default::default()
        };
        let nat = NatTable::new(config);

        let port1 = nat.allocate_port().unwrap();
        let port2 = nat.allocate_port().unwrap();

        assert!(port1 >= 1000 && port1 < 1010);
        assert!(port2 >= 1000 && port2 < 1010);
        assert_ne!(port1, port2);
    }
}
