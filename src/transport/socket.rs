//! Low-level socket creation with platform-specific optimizations.
//!
//! Supports interface-level binding on Linux (SO_BINDTODEVICE) and macOS (IP_BOUND_IF).

use std::net::SocketAddr;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;

use socket2::{Domain, Protocol, Socket, Type};
use tracing::{debug, warn};

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
    /// Interface to bind to (for multi-homing).
    pub bind_interface: Option<String>,
    /// Mark for policy routing (Linux only).
    pub fwmark: Option<u32>,
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
            bind_interface: None,
            fwmark: None,
        }
    }

    /// Set the interface to bind to.
    pub fn with_interface(mut self, interface: impl Into<String>) -> Self {
        self.bind_interface = Some(interface.into());
        self
    }

    /// Set the fwmark for policy routing.
    pub fn with_fwmark(mut self, mark: u32) -> Self {
        self.fwmark = Some(mark);
        self
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
            bind_interface: None,
            fwmark: None,
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

    // Bind to interface if specified
    bind_to_interface(&socket, config)?;

    // Set fwmark if specified (Linux only)
    #[cfg(target_os = "linux")]
    if let Some(mark) = config.fwmark {
        set_socket_mark(&socket, mark)?;
    }

    // Bind to address
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

/// Create a UDP socket bound to an interface.
#[allow(dead_code)]
pub fn create_udp_socket_on_interface(
    addr: SocketAddr, 
    interface: &str,
    config: &SocketConfig,
) -> Result<Socket> {
    let mut config = config.clone();
    config.bind_interface = Some(interface.to_string());
    create_udp_socket(addr, &config)
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

    // Bind to interface if specified
    bind_to_interface(&socket, config)?;

    // Set fwmark if specified (Linux only)
    #[cfg(target_os = "linux")]
    if let Some(mark) = config.fwmark {
        set_socket_mark(&socket, mark)?;
    }

    // Bind to address
    socket.bind(&addr.into())
        .map_err(|e| TransportError::BindFailed {
            addr,
            reason: e.to_string(),
        })?;

    Ok(socket)
}

/// Create a TCP socket bound to an interface.
#[allow(dead_code)]
pub fn create_tcp_socket_on_interface(
    addr: SocketAddr,
    interface: &str,
    config: &SocketConfig,
) -> Result<Socket> {
    let mut config = config.clone();
    config.bind_interface = Some(interface.to_string());
    create_tcp_socket(addr, &config)
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

/// Bind socket to a specific interface.
fn bind_to_interface(socket: &Socket, config: &SocketConfig) -> Result<()> {
    let interface = match &config.bind_interface {
        Some(iface) => iface,
        None => return Ok(()),
    };

    #[cfg(target_os = "linux")]
    {
        use std::ffi::CString;
        
        let cname = CString::new(interface.as_str())
            .map_err(|_| TransportError::SocketError("Invalid interface name".to_string()))?;
        
        let ret = unsafe {
            libc::setsockopt(
                socket.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_BINDTODEVICE,
                cname.as_ptr() as *const libc::c_void,
                (interface.len() + 1) as libc::socklen_t,
            )
        };
        
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            // EPERM means we don't have CAP_NET_RAW
            if err.raw_os_error() == Some(libc::EPERM) {
                warn!(
                    "SO_BINDTODEVICE requires CAP_NET_RAW or root for interface {}. \
                     Falling back to source address binding only.",
                    interface
                );
            } else {
                return Err(TransportError::SocketError(
                    format!("SO_BINDTODEVICE failed for {}: {}", interface, err)
                ).into());
            }
        } else {
            debug!("Bound socket to interface {} via SO_BINDTODEVICE", interface);
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Get interface index
        let idx = crate::util::if_nametoindex(interface)
            .ok_or_else(|| TransportError::SocketError(
                format!("Interface not found: {}", interface)
            ))?;
        
        let ret = unsafe {
            libc::setsockopt(
                socket.as_raw_fd(),
                libc::IPPROTO_IP,
                libc::IP_BOUND_IF,
                &idx as *const u32 as *const libc::c_void,
                std::mem::size_of::<u32>() as libc::socklen_t,
            )
        };
        
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            return Err(TransportError::SocketError(
                format!("IP_BOUND_IF failed for {}: {}", interface, err)
            ).into());
        }
        
        debug!("Bound socket to interface {} via IP_BOUND_IF", interface);
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        warn!("Interface binding not supported on this platform, using address binding only");
    }

    Ok(())
}

/// Set SO_MARK on socket (Linux only).
#[cfg(target_os = "linux")]
fn set_socket_mark(socket: &Socket, mark: u32) -> Result<()> {
    let ret = unsafe {
        libc::setsockopt(
            socket.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_MARK,
            &mark as *const u32 as *const libc::c_void,
            std::mem::size_of::<u32>() as libc::socklen_t,
        )
    };
    
    if ret != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EPERM) {
            warn!("SO_MARK requires CAP_NET_ADMIN or root");
        }
        return Err(TransportError::SocketError(
            format!("SO_MARK failed: {}", err)
        ).into());
    }
    
    debug!("Set socket mark to {}", mark);
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

/// Create a socket bound to an interface.
#[allow(dead_code)]
pub fn bind_socket_on_interface(
    is_ipv6: bool,
    is_udp: bool,
    interface: &str,
    config: &SocketConfig,
) -> Result<Socket> {
    let addr = if is_ipv6 {
        SocketAddr::from(([0u8; 16], 0))
    } else {
        SocketAddr::from(([0u8; 4], 0))
    };

    if is_udp {
        create_udp_socket_on_interface(addr, interface, config)
    } else {
        create_tcp_socket_on_interface(addr, interface, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_socket_config_with_interface() {
        let config = SocketConfig::default()
            .with_interface("eth0")
            .with_fwmark(100);
        
        assert_eq!(config.bind_interface, Some("eth0".to_string()));
        assert_eq!(config.fwmark, Some(100));
    }

    #[test]
    fn test_create_udp_socket() {
        let addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
        let config = SocketConfig::default();
        
        let socket = create_udp_socket(addr, &config).unwrap();
        let local = socket.local_addr().unwrap();
        assert!(local.as_socket().is_some());
    }

    #[test]
    fn test_create_tcp_socket() {
        let addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
        let config = SocketConfig::default();
        
        let socket = create_tcp_socket(addr, &config).unwrap();
        let local = socket.local_addr().unwrap();
        assert!(local.as_socket().is_some());
    }

    #[test]
    fn test_bind_socket() {
        let config = SocketConfig::default();
        
        let udp = bind_socket(false, true, &config).unwrap();
        assert!(udp.local_addr().unwrap().as_socket().is_some());
        
        let tcp = bind_socket(false, false, &config).unwrap();
        assert!(tcp.local_addr().unwrap().as_socket().is_some());
    }
}
