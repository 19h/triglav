//! Utility functions and helpers.

use std::net::{IpAddr, SocketAddr};

/// Get all network interfaces with their addresses.
#[cfg(target_os = "linux")]
pub fn get_network_interfaces() -> Vec<NetworkInterface> {
    // Use netlink on Linux
    vec![] // TODO: Implement with rtnetlink
}

#[cfg(target_os = "macos")]
pub fn get_network_interfaces() -> Vec<NetworkInterface> {
    use std::ffi::CStr;

    let mut interfaces = Vec::new();

    unsafe {
        let mut ifaddrs: *mut libc::ifaddrs = std::ptr::null_mut();
        if libc::getifaddrs(std::ptr::addr_of_mut!(ifaddrs)) != 0 {
            return interfaces;
        }

        let mut current = ifaddrs;
        while !current.is_null() {
            let ifa = &*current;

            if !ifa.ifa_name.is_null() && !ifa.ifa_addr.is_null() {
                let name = CStr::from_ptr(ifa.ifa_name).to_string_lossy().into_owned();
                let family = i32::from((*ifa.ifa_addr).sa_family);

                // Pointer casts are necessary for socket address type matching
                #[allow(clippy::cast_ptr_alignment)]
                let addr = match family {
                    libc::AF_INET => {
                        let sockaddr = ifa.ifa_addr.cast::<libc::sockaddr_in>();
                        let ip = std::net::Ipv4Addr::from(u32::from_be((*sockaddr).sin_addr.s_addr));
                        Some(IpAddr::V4(ip))
                    }
                    libc::AF_INET6 => {
                        let sockaddr = ifa.ifa_addr.cast::<libc::sockaddr_in6>();
                        let ip = std::net::Ipv6Addr::from((*sockaddr).sin6_addr.s6_addr);
                        Some(IpAddr::V6(ip))
                    }
                    _ => None,
                };

                if let Some(ip) = addr {
                    let is_up = (ifa.ifa_flags as i32 & libc::IFF_UP) != 0;
                    let is_loopback = (ifa.ifa_flags as i32 & libc::IFF_LOOPBACK) != 0;

                    interfaces.push(NetworkInterface {
                        name: name.clone(),
                        address: ip,
                        is_up,
                        is_loopback,
                        interface_type: guess_interface_type(&name),
                    });
                }
            }

            current = ifa.ifa_next;
        }

        libc::freeifaddrs(ifaddrs);
    }

    interfaces
}

/// Network interface information.
#[derive(Debug, Clone)]
pub struct NetworkInterface {
    pub name: String,
    pub address: IpAddr,
    pub is_up: bool,
    pub is_loopback: bool,
    pub interface_type: crate::types::InterfaceType,
}

/// Guess interface type from name.
pub fn guess_interface_type(name: &str) -> crate::types::InterfaceType {
    use crate::types::InterfaceType;

    let name = name.to_lowercase();

    if name.starts_with("lo") {
        InterfaceType::Loopback
    } else if name.starts_with("en") || name.starts_with("eth") {
        // Could be ethernet or wifi on macOS
        if name.contains("wifi") || name == "en0" {
            InterfaceType::Wifi
        } else {
            InterfaceType::Ethernet
        }
    } else if name.starts_with("wlan") || name.starts_with("wl") {
        InterfaceType::Wifi
    } else if name.starts_with("cell") || name.starts_with("pdp") || name.starts_with("rmnet") {
        InterfaceType::Cellular
    } else if name.starts_with("tun") || name.starts_with("tap") || name.starts_with("utun") {
        InterfaceType::Tunnel
    } else if name.starts_with("bridge") || name.starts_with("br") {
        InterfaceType::Ethernet
    } else {
        InterfaceType::Unknown
    }
}

/// Format bytes as human-readable.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format duration as human-readable.
pub fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    let ms = duration.subsec_millis();

    if secs >= 3600 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else if secs > 0 {
        format!("{secs}.{ms:03}s")
    } else {
        format!("{ms}ms")
    }
}

/// Parse socket address with optional port.
pub fn parse_addr_with_default_port(s: &str, default_port: u16) -> Result<SocketAddr, std::net::AddrParseError> {
    if s.contains(':') {
        s.parse()
    } else {
        format!("{s}:{default_port}").parse()
    }
}

/// Check if running with elevated privileges.
#[cfg(unix)]
pub fn is_root() -> bool {
    unsafe { libc::geteuid() == 0 }
}

#[cfg(not(unix))]
pub fn is_root() -> bool {
    false
}

/// Get the hostname.
pub fn hostname() -> Option<String> {
    let mut buf = [0u8; 256];
    unsafe {
        if libc::gethostname(buf.as_mut_ptr().cast::<i8>(), buf.len()) == 0 {
            let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            String::from_utf8(buf[..len].to_vec()).ok()
        } else {
            None
        }
    }
}
