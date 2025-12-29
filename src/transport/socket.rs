//! Low-level socket creation with platform-specific optimizations.

use std::net::SocketAddr;

use socket2::{Domain, Protocol, Socket, Type};

use super::TransportConfig;
use crate::error::{Result, TransportError};

/// Socket configuration options.
#[derive(Debug, Clone)]
pub struct SocketConfig {
    pub send_buffer_size: usize,
    pub recv_buffer_size: usize,
    pub reuse_addr: bool,
    pub reuse_port: bool,
    pub nodelay: bool,
    pub keepalive: Option<std::time::Duration>,
}

impl SocketConfig {
    /// Create from transport config.
    pub fn from_transport_config(config: &TransportConfig) -> Self {
        Self {
            send_buffer_size: config.send_buffer_size,
            recv_buffer_size: config.recv_buffer_size,
            reuse_addr: config.reuse_addr,
            reuse_port: config.reuse_port,
            nodelay: config.tcp_nodelay,
            keepalive: Some(config.keepalive_interval),
        }
    }
}

impl Default for SocketConfig {
    fn default() -> Self {
        Self {
            send_buffer_size: 2 * 1024 * 1024,
            recv_buffer_size: 2 * 1024 * 1024,
            reuse_addr: true,
            reuse_port: false,
            nodelay: true,
            keepalive: Some(std::time::Duration::from_secs(30)),
        }
    }
}

/// Create a UDP socket with optimizations.
pub fn create_udp_socket(addr: SocketAddr, config: &SocketConfig) -> Result<Socket> {
    let domain = if addr.is_ipv6() {
        Domain::IPV6
    } else {
        Domain::IPV4
    };

    let socket = Socket::new(domain, Type::DGRAM, Some(Protocol::UDP))
        .map_err(|e| TransportError::SocketError(e.to_string()))?;

    configure_socket(&socket, config)?;

    // For IPv6, allow IPv4 mapped addresses
    if addr.is_ipv6() {
        socket.set_only_v6(false)
            .map_err(|e| TransportError::SocketError(e.to_string()))?;
    }

    // Bind
    socket.bind(&addr.into())
        .map_err(|e| TransportError::BindFailed {
            addr,
            reason: e.to_string(),
        })?;

    // Set non-blocking
    socket.set_nonblocking(true)
        .map_err(|e| TransportError::SocketError(e.to_string()))?;

    Ok(socket)
}

/// Create a TCP socket with optimizations.
pub fn create_tcp_socket(addr: SocketAddr, config: &SocketConfig) -> Result<Socket> {
    let domain = if addr.is_ipv6() {
        Domain::IPV6
    } else {
        Domain::IPV4
    };

    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))
        .map_err(|e| TransportError::SocketError(e.to_string()))?;

    configure_socket(&socket, config)?;

    // TCP-specific options
    if config.nodelay {
        socket.set_nodelay(true)
            .map_err(|e| TransportError::SocketError(e.to_string()))?;
    }

    if let Some(keepalive) = config.keepalive {
        let ka = socket2::TcpKeepalive::new()
            .with_time(keepalive)
            .with_interval(keepalive / 3);

        socket.set_tcp_keepalive(&ka)
            .map_err(|e| TransportError::SocketError(e.to_string()))?;
    }

    // For IPv6, allow IPv4 mapped addresses
    if addr.is_ipv6() {
        socket.set_only_v6(false)
            .map_err(|e| TransportError::SocketError(e.to_string()))?;
    }

    // Bind
    socket.bind(&addr.into())
        .map_err(|e| TransportError::BindFailed {
            addr,
            reason: e.to_string(),
        })?;

    Ok(socket)
}

/// Apply common socket configuration.
fn configure_socket(socket: &Socket, config: &SocketConfig) -> Result<()> {
    // Buffer sizes
    socket.set_send_buffer_size(config.send_buffer_size)
        .map_err(|e| TransportError::SocketError(format!("set send buffer: {e}")))?;

    socket.set_recv_buffer_size(config.recv_buffer_size)
        .map_err(|e| TransportError::SocketError(format!("set recv buffer: {e}")))?;

    // Address reuse
    if config.reuse_addr {
        socket.set_reuse_address(true)
            .map_err(|e| TransportError::SocketError(format!("set reuse addr: {e}")))?;
    }

    // Port reuse (for load balancing, where supported)
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "freebsd"))]
    if config.reuse_port {
        socket.set_reuse_port(true)
            .map_err(|e| TransportError::SocketError(format!("set reuse port: {e}")))?;
    }

    Ok(())
}

/// Create a socket and bind to any available address.
pub fn bind_socket(
    is_ipv6: bool,
    is_udp: bool,
    config: &SocketConfig,
) -> Result<Socket> {
    let addr = if is_ipv6 {
        SocketAddr::from(([0u8; 16], 0))
    } else {
        SocketAddr::from(([0u8; 4], 0))
    };

    if is_udp {
        create_udp_socket(addr, config)
    } else {
        create_tcp_socket(addr, config)
    }
}

