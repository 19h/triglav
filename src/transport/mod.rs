//! Transport layer for Triglav.
//!
//! Provides UDP fast-path and TCP fallback transport with dual-stack IPv4/IPv6 support.
//! Designed for high performance with zero-copy operations where possible.

mod buffer;
mod socket;
mod tcp;
mod udp;

pub use buffer::{BufferPool, PacketBuffer};
pub use socket::{bind_socket, SocketConfig};
pub use tcp::{TcpStream, TcpTransport};
pub use udp::{UdpSocket, UdpTransport};

use std::net::SocketAddr;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::types::IpVersion;

/// Transport configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    /// IP version preference.
    #[serde(default)]
    pub ip_version: IpVersion,

    /// Send buffer size in bytes.
    #[serde(default = "default_send_buffer")]
    pub send_buffer_size: usize,

    /// Receive buffer size in bytes.
    #[serde(default = "default_recv_buffer")]
    pub recv_buffer_size: usize,

    /// Connection timeout.
    #[serde(default = "default_connect_timeout", with = "humantime_serde")]
    pub connect_timeout: Duration,

    /// Read/write timeout.
    #[serde(default = "default_io_timeout", with = "humantime_serde")]
    pub io_timeout: Duration,

    /// Keep-alive interval for TCP.
    #[serde(default = "default_keepalive", with = "humantime_serde")]
    pub keepalive_interval: Duration,

    /// Enable TCP_NODELAY.
    #[serde(default = "default_nodelay")]
    pub tcp_nodelay: bool,

    /// Enable SO_REUSEADDR.
    #[serde(default = "default_reuse_addr")]
    pub reuse_addr: bool,

    /// Enable SO_REUSEPORT (where available).
    #[serde(default)]
    pub reuse_port: bool,

    /// Maximum segment size override (0 = system default).
    #[serde(default)]
    pub mss: u32,

    /// Enable fast-open for TCP (where available).
    #[serde(default)]
    pub tcp_fast_open: bool,
}

fn default_send_buffer() -> usize {
    2 * 1024 * 1024
} // 2 MB
fn default_recv_buffer() -> usize {
    2 * 1024 * 1024
} // 2 MB
fn default_connect_timeout() -> Duration {
    Duration::from_secs(10)
}
fn default_io_timeout() -> Duration {
    Duration::from_secs(30)
}
fn default_keepalive() -> Duration {
    Duration::from_secs(30)
}
fn default_nodelay() -> bool {
    true
}
fn default_reuse_addr() -> bool {
    true
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            ip_version: IpVersion::default(),
            send_buffer_size: default_send_buffer(),
            recv_buffer_size: default_recv_buffer(),
            connect_timeout: default_connect_timeout(),
            io_timeout: default_io_timeout(),
            keepalive_interval: default_keepalive(),
            tcp_nodelay: default_nodelay(),
            reuse_addr: default_reuse_addr(),
            reuse_port: false,
            mss: 0,
            tcp_fast_open: false,
        }
    }
}

/// Transport trait for sending and receiving packets.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Get the local address.
    fn local_addr(&self) -> Result<SocketAddr>;

    /// Get the remote address (if connected).
    fn remote_addr(&self) -> Option<SocketAddr>;

    /// Send data to a specific address.
    async fn send_to(&self, data: &[u8], addr: SocketAddr) -> Result<usize>;

    /// Send data (for connected transports).
    async fn send(&self, data: &[u8]) -> Result<usize>;

    /// Receive data with source address.
    async fn recv_from(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)>;

    /// Receive data (for connected transports).
    async fn recv(&self, buf: &mut [u8]) -> Result<usize>;

    /// Close the transport.
    async fn close(&self) -> Result<()>;

    /// Check if transport is connected.
    fn is_connected(&self) -> bool;

    /// Get transport type name.
    fn transport_type(&self) -> &'static str;
}

/// Transport protocol selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransportProtocol {
    /// UDP for low-latency, high-throughput (default fast path).
    #[default]
    Udp,
    /// TCP for reliability through firewalls.
    Tcp,
}

impl std::fmt::Display for TransportProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Udp => write!(f, "udp"),
            Self::Tcp => write!(f, "tcp"),
        }
    }
}

/// Create a transport based on protocol.
pub fn create_transport(
    protocol: TransportProtocol,
    bind_addr: SocketAddr,
    config: &TransportConfig,
) -> Result<Box<dyn Transport>> {
    match protocol {
        TransportProtocol::Udp => {
            let transport = UdpTransport::bind(bind_addr, config)?;
            Ok(Box::new(transport))
        }
        TransportProtocol::Tcp => {
            // For TCP, we create a listener-capable transport
            let transport = TcpTransport::bind(bind_addr, config)?;
            Ok(Box::new(transport))
        }
    }
}

/// Connect to a remote address using the specified protocol.
pub async fn connect(
    protocol: TransportProtocol,
    remote_addr: SocketAddr,
    bind_addr: Option<SocketAddr>,
    config: &TransportConfig,
) -> Result<Box<dyn Transport>> {
    match protocol {
        TransportProtocol::Udp => {
            let transport = UdpTransport::connect(remote_addr, bind_addr, config).await?;
            Ok(Box::new(transport))
        }
        TransportProtocol::Tcp => {
            let transport = TcpTransport::connect(remote_addr, bind_addr, config).await?;
            Ok(Box::new(transport))
        }
    }
}

/// Packet with metadata for internal processing.
#[derive(Debug)]
pub struct TransportPacket {
    /// Packet data.
    pub data: Vec<u8>,
    /// Source address.
    pub src_addr: SocketAddr,
    /// Destination address.
    pub dst_addr: SocketAddr,
    /// Receive timestamp.
    pub timestamp: std::time::Instant,
}
