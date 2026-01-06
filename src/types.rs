//! Core types used throughout Triglav.

use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConnectionId(pub Uuid);

impl ConnectionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(Uuid::from_bytes(bytes))
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }
}

impl Default for ConnectionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.0.to_string()[..8])
    }
}

/// Unique identifier for an uplink (network interface).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UplinkId(pub String);

impl UplinkId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for UplinkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for UplinkId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for UplinkId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Unique identifier for a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub [u8; 32]);

impl SessionId {
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut bytes);
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(&self.0[..8]))
    }
}

/// Packet sequence number for ordering and deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SequenceNumber(pub u64);

impl SequenceNumber {
    pub const ZERO: Self = Self(0);

    pub fn new(n: u64) -> Self {
        Self(n)
    }

    pub fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }

    pub fn distance(self, other: Self) -> i64 {
        (self.0 as i64).wrapping_sub(other.0 as i64)
    }
}

impl fmt::Display for SequenceNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Network interface type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InterfaceType {
    /// Wired Ethernet connection
    Ethernet,
    /// WiFi connection
    Wifi,
    /// Cellular data (4G/5G/LTE)
    Cellular,
    /// USB tethering
    Tethering,
    /// VPN or tunnel interface
    Tunnel,
    /// Loopback interface
    Loopback,
    /// Unknown interface type
    #[default]
    Unknown,
}

impl InterfaceType {
    /// Base priority score (higher = preferred).
    pub fn base_priority(self) -> u32 {
        match self {
            Self::Ethernet => 100,
            Self::Wifi => 80,
            Self::Cellular => 60,
            Self::Tethering => 50,
            Self::Tunnel => 40,
            Self::Loopback => 10,
            Self::Unknown => 30,
        }
    }

    /// Expected latency characteristics in milliseconds.
    pub fn expected_latency_ms(self) -> (u32, u32) {
        match self {
            Self::Ethernet => (1, 10),
            Self::Wifi => (5, 50),
            Self::Cellular => (20, 200),
            Self::Tethering => (30, 150),
            Self::Tunnel => (10, 100),
            Self::Loopback => (0, 1),
            Self::Unknown => (10, 100),
        }
    }

    /// Expected bandwidth characteristics in Mbps.
    pub fn expected_bandwidth_mbps(self) -> (u32, u32) {
        match self {
            Self::Ethernet => (100, 10000),
            Self::Wifi => (10, 1000),
            Self::Cellular => (1, 100),
            Self::Tethering => (5, 50),
            Self::Tunnel => (10, 1000),
            Self::Loopback => (10000, 100000),
            Self::Unknown => (1, 100),
        }
    }
}

impl fmt::Display for InterfaceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ethernet => write!(f, "ethernet"),
            Self::Wifi => write!(f, "wifi"),
            Self::Cellular => write!(f, "cellular"),
            Self::Tethering => write!(f, "tethering"),
            Self::Tunnel => write!(f, "tunnel"),
            Self::Loopback => write!(f, "loopback"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// IP version preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum IpVersion {
    /// IPv4 only
    V4Only,
    /// IPv6 only
    V6Only,
    /// Prefer IPv6, fallback to IPv4
    #[default]
    PreferV6,
    /// Prefer IPv4, fallback to IPv6
    PreferV4,
    /// Dual-stack, use both equally
    DualStack,
}

impl IpVersion {
    pub fn accepts_v4(self) -> bool {
        !matches!(self, Self::V6Only)
    }

    pub fn accepts_v6(self) -> bool {
        !matches!(self, Self::V4Only)
    }

    pub fn prefers_v6(self) -> bool {
        matches!(self, Self::PreferV6 | Self::V6Only)
    }

    pub fn filter_addr(self, addr: IpAddr) -> bool {
        match addr {
            IpAddr::V4(_) => self.accepts_v4(),
            IpAddr::V6(_) => self.accepts_v6(),
        }
    }
}

/// Uplink health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UplinkHealth {
    /// Uplink is healthy and performing well
    Healthy,
    /// Uplink is degraded but usable
    Degraded,
    /// Uplink is unhealthy and should be avoided
    Unhealthy,
    /// Uplink is completely down
    Down,
    /// Uplink health is unknown (initializing)
    Unknown,
}

impl UplinkHealth {
    pub fn is_usable(self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded | Self::Unknown)
    }

    pub fn priority_modifier(self) -> f64 {
        match self {
            Self::Healthy => 1.0,
            Self::Degraded => 0.5,
            Self::Unknown => 0.7,
            Self::Unhealthy => 0.1,
            Self::Down => 0.0,
        }
    }
}

impl fmt::Display for UplinkHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
            Self::Down => write!(f, "down"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConnectionState {
    /// Initial state, not yet connected
    Disconnected,
    /// Currently establishing connection
    Connecting,
    /// Connection established, performing handshake
    Handshaking,
    /// Fully connected and operational
    Connected,
    /// Connection is being gracefully closed
    Disconnecting,
    /// Connection failed
    Failed,
}

impl ConnectionState {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Connecting | Self::Handshaking | Self::Connected)
    }

    pub fn is_established(self) -> bool {
        matches!(self, Self::Connected)
    }
}

impl fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disconnected => write!(f, "disconnected"),
            Self::Connecting => write!(f, "connecting"),
            Self::Handshaking => write!(f, "handshaking"),
            Self::Connected => write!(f, "connected"),
            Self::Disconnecting => write!(f, "disconnecting"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Bandwidth measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Bandwidth {
    /// Bytes per second
    pub bytes_per_sec: f64,
}

impl Bandwidth {
    pub const ZERO: Self = Self { bytes_per_sec: 0.0 };

    pub fn from_bps(bytes_per_sec: f64) -> Self {
        Self { bytes_per_sec }
    }

    pub fn from_mbps(megabits_per_sec: f64) -> Self {
        Self {
            bytes_per_sec: megabits_per_sec * 125_000.0,
        }
    }

    pub fn as_mbps(self) -> f64 {
        self.bytes_per_sec / 125_000.0
    }

    pub fn as_human_readable(self) -> String {
        let bps = self.bytes_per_sec * 8.0;
        if bps >= 1_000_000_000.0 {
            let gbps = bps / 1_000_000_000.0;
            format!("{gbps:.2} Gbps")
        } else if bps >= 1_000_000.0 {
            let mbps = bps / 1_000_000.0;
            format!("{mbps:.2} Mbps")
        } else if bps >= 1_000.0 {
            let kbps = bps / 1_000.0;
            format!("{kbps:.2} Kbps")
        } else {
            format!("{bps:.0} bps")
        }
    }
}

impl fmt::Display for Bandwidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_human_readable())
    }
}

/// Latency measurement with statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Latency {
    /// Minimum observed latency
    pub min: Duration,
    /// Average latency
    pub avg: Duration,
    /// Maximum observed latency
    pub max: Duration,
    /// Standard deviation
    pub stddev: Duration,
    /// 95th percentile latency
    pub p95: Duration,
    /// 99th percentile latency
    pub p99: Duration,
}

impl Latency {
    pub fn from_samples(samples: &[Duration]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut sorted: Vec<_> = samples.to_vec();
        sorted.sort();

        let n = sorted.len();
        let sum: Duration = sorted.iter().sum();
        let avg = sum / n as u32;

        let variance: f64 = sorted
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - avg.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / n as f64;

        Some(Self {
            min: sorted[0],
            avg,
            max: sorted[n - 1],
            stddev: Duration::from_secs_f64(variance.sqrt()),
            p95: sorted[(n * 95 / 100).min(n - 1)],
            p99: sorted[(n * 99 / 100).min(n - 1)],
        })
    }

    pub fn as_human_readable(&self) -> String {
        format!(
            "avg={:.1}ms, p95={:.1}ms, p99={:.1}ms",
            self.avg.as_secs_f64() * 1000.0,
            self.p95.as_secs_f64() * 1000.0,
            self.p99.as_secs_f64() * 1000.0
        )
    }
}

impl fmt::Display for Latency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}ms", self.avg.as_secs_f64() * 1000.0)
    }
}

/// Authentication key for client-server communication.
#[derive(Clone, Serialize, Deserialize)]
pub struct AuthKey {
    /// The encoded key string
    key: String,
    /// Decoded server public key
    #[serde(skip)]
    server_pubkey: Option<[u8; 32]>,
    /// Decoded server addresses
    #[serde(skip)]
    server_addrs: Vec<SocketAddr>,
}

impl AuthKey {
    /// Key prefix for identification
    const PREFIX: &'static str = "tg1";

    /// Create a new auth key from components.
    pub fn new(server_pubkey: [u8; 32], server_addrs: Vec<SocketAddr>) -> Self {
        let mut data = Vec::with_capacity(32 + server_addrs.len() * 18);
        data.extend_from_slice(&server_pubkey);

        for addr in &server_addrs {
            match addr {
                SocketAddr::V4(v4) => {
                    data.push(4);
                    data.extend_from_slice(&v4.ip().octets());
                    data.extend_from_slice(&v4.port().to_be_bytes());
                }
                SocketAddr::V6(v6) => {
                    data.push(6);
                    data.extend_from_slice(&v6.ip().octets());
                    data.extend_from_slice(&v6.port().to_be_bytes());
                }
            }
        }

        let encoded =
            base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, &data);

        Self {
            key: format!("{}_{}", Self::PREFIX, encoded),
            server_pubkey: Some(server_pubkey),
            server_addrs,
        }
    }

    /// Parse an auth key from string.
    pub fn parse(key: &str) -> Result<Self, crate::Error> {
        use base64::Engine;

        let parts: Vec<&str> = key.split('_').collect();
        if parts.len() != 2 || parts[0] != Self::PREFIX {
            return Err(crate::Error::InvalidKey("invalid key format".into()));
        }

        let data = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(parts[1])
            .map_err(|e| crate::Error::InvalidKey(format!("base64 decode failed: {e}")))?;

        if data.len() < 32 {
            return Err(crate::Error::InvalidKey("key too short".into()));
        }

        let mut server_pubkey = [0u8; 32];
        server_pubkey.copy_from_slice(&data[..32]);

        let mut server_addrs = Vec::new();
        let mut pos = 32;

        while pos < data.len() {
            let version = data[pos];
            pos += 1;

            match version {
                4 => {
                    if pos + 6 > data.len() {
                        break;
                    }
                    let ip = std::net::Ipv4Addr::new(
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                    );
                    let port = u16::from_be_bytes([data[pos + 4], data[pos + 5]]);
                    server_addrs.push(SocketAddr::from((ip, port)));
                    pos += 6;
                }
                6 => {
                    if pos + 18 > data.len() {
                        break;
                    }
                    let mut octets = [0u8; 16];
                    octets.copy_from_slice(&data[pos..pos + 16]);
                    let ip = std::net::Ipv6Addr::from(octets);
                    let port = u16::from_be_bytes([data[pos + 16], data[pos + 17]]);
                    server_addrs.push(SocketAddr::from((ip, port)));
                    pos += 18;
                }
                _ => break,
            }
        }

        if server_addrs.is_empty() {
            return Err(crate::Error::InvalidKey(
                "no server addresses in key".into(),
            ));
        }

        Ok(Self {
            key: key.to_string(),
            server_pubkey: Some(server_pubkey),
            server_addrs,
        })
    }

    /// Get the server's public key.
    pub fn server_pubkey(&self) -> [u8; 32] {
        self.server_pubkey.expect("key not parsed")
    }

    /// Get the server addresses.
    pub fn server_addrs(&self) -> &[SocketAddr] {
        &self.server_addrs
    }

    /// Get the encoded key string.
    pub fn as_str(&self) -> &str {
        &self.key
    }
}

impl fmt::Display for AuthKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.key)
    }
}

// Intentionally abbreviated Debug output - sensitive data should not be fully printed
#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for AuthKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthKey")
            .field(
                "key",
                &format!("{}...", &self.key[..16.min(self.key.len())]),
            )
            .field("server_addrs", &self.server_addrs)
            .finish()
    }
}

impl FromStr for AuthKey {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Traffic statistics.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TrafficStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub packets_dropped: u64,
    pub packets_retransmitted: u64,
}

impl TrafficStats {
    pub fn add(&mut self, other: &Self) {
        self.bytes_sent += other.bytes_sent;
        self.bytes_received += other.bytes_received;
        self.packets_sent += other.packets_sent;
        self.packets_received += other.packets_received;
        self.packets_dropped += other.packets_dropped;
        self.packets_retransmitted += other.packets_retransmitted;
    }
}
