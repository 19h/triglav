//! TUN/TAP virtual network interface module.
//!
//! This module provides a cross-platform abstraction for creating and managing
//! TUN (Layer 3) virtual network interfaces. Unlike proxy-based approaches,
//! TUN interfaces operate at the IP packet level, allowing transparent tunneling
//! of all network traffic without requiring application-level proxy configuration.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         Application Layer                               │
//! │                    (Any TCP/UDP/ICMP application)                       │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                         Kernel TCP/IP Stack                              │
//! │                    (OS handles transport protocols)                      │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                         TUN Device (utun/tun0)                          │
//! │                    [IP Packets read/written here]                        │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                              Triglav                                     │
//! │  ┌──────────────────────────────────────────────────────────────────┐   │
//! │  │                        TunnelRunner                               │   │
//! │  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │   │
//! │  │  │  IP Parser  │──│     NAT      │──│   MultipathManager      │  │   │
//! │  │  │ (5-tuple)   │  │ Translation  │  │  (encryption, routing)  │  │   │
//! │  │  └─────────────┘  └──────────────┘  └─────────────────────────┘  │   │
//! │  └──────────────────────────────────────────────────────────────────┘   │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                     Multiple Physical Uplinks                            │
//! │               (WiFi, Cellular, Ethernet, etc.)                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Platform Support
//!
//! - **Linux**: Uses `/dev/net/tun` with `ioctl(TUNSETIFF)`
//! - **macOS**: Uses `utun` kernel control socket
//! - **Windows**: Uses WinTUN driver (requires installation)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use triglav::tun::{TunDevice, TunConfig};
//!
//! # async fn example() -> triglav::Result<()> {
//! // Create TUN device
//! let config = TunConfig::default();
//! let tun = TunDevice::create(config)?;
//!
//! // Read/write IP packets
//! let mut buf = vec![0u8; 1500];
//! let len = tun.read(&mut buf).await?;
//! // Process IP packet in buf[..len]
//! # Ok(())
//! # }
//! ```

mod device;
mod dns;
mod nat;
mod packet;
mod routing;
mod runner;

pub use device::{TunConfig, TunDevice, TunHandle};
pub use dns::{DnsConfig, DnsInterceptor};
pub use nat::{NatConfig, NatEntry, NatTable};
pub use packet::{FlowTuple, IpPacket, IpVersion, TransportProtocol as IpTransportProtocol};
pub use routing::{Route, RouteConfig, RouteManager};
pub use runner::{TunnelConfig, TunnelRunner, TunnelStats};

use crate::error::Result;

/// Default MTU for TUN device.
pub const DEFAULT_TUN_MTU: u16 = 1420;

/// Default tunnel IP address (internal side).
pub const DEFAULT_TUNNEL_IPV4: &str = "10.0.85.1";
pub const DEFAULT_TUNNEL_IPV6: &str = "fd00:7472:6967::1";

/// Default tunnel network.
pub const DEFAULT_TUNNEL_NETWORK_V4: &str = "10.0.85.0/24";
pub const DEFAULT_TUNNEL_NETWORK_V6: &str = "fd00:7472:6967::/64";

/// Check if the current process has sufficient privileges to create TUN devices.
pub fn check_privileges() -> Result<bool> {
    #[cfg(unix)]
    {
        // On Unix, we need root or CAP_NET_ADMIN
        let uid = unsafe { libc::getuid() };
        if uid == 0 {
            return Ok(true);
        }

        // Check for CAP_NET_ADMIN on Linux
        #[cfg(target_os = "linux")]
        {
            // Try to check capabilities - simplified check
            // A full implementation would use libcap
            Ok(false)
        }

        #[cfg(target_os = "macos")]
        {
            // On macOS, only root can create utun devices
            Ok(false)
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Ok(false)
        }
    }

    #[cfg(windows)]
    {
        // On Windows, check if we can access WinTUN
        // This is a simplified check
        Ok(true) // Assume we have access if WinTUN is installed
    }

    #[cfg(not(any(unix, windows)))]
    {
        Ok(false)
    }
}

/// Get the recommended TUN device name for this platform.
pub fn recommended_device_name() -> &'static str {
    #[cfg(target_os = "linux")]
    {
        "tg0"
    }
    #[cfg(target_os = "macos")]
    {
        "utun" // macOS will append a number
    }
    #[cfg(target_os = "windows")]
    {
        "Triglav"
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        "tun0"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_privileges() {
        // Should not panic
        let _ = check_privileges();
    }

    #[test]
    fn test_recommended_device_name() {
        let name = recommended_device_name();
        assert!(!name.is_empty());
    }
}
