//! Utility functions and helpers.

use std::net::{IpAddr, SocketAddr};

use crate::types::InterfaceType;

// Re-export submodules
mod connectivity;
mod interface;
mod netwatch;

pub use connectivity::*;
pub use interface::*;
pub use netwatch::*;

/// Network interface information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkInterface {
    pub name: String,
    pub index: u32,
    pub address: IpAddr,
    pub netmask: Option<IpAddr>,
    pub broadcast: Option<IpAddr>,
    pub is_up: bool,
    pub is_running: bool,
    pub is_loopback: bool,
    pub mtu: u32,
    pub interface_type: InterfaceType,
    pub mac_address: Option<[u8; 6]>,
}

impl NetworkInterface {
    /// Check if this interface has a default gateway.
    pub fn has_gateway(&self) -> bool {
        // Will be set by gateway detection
        false
    }

    /// Get a string representation of the MAC address.
    pub fn mac_string(&self) -> Option<String> {
        self.mac_address.map(|mac| {
            format!(
                "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
            )
        })
    }
}

/// Guess interface type from name.
pub fn guess_interface_type(name: &str) -> InterfaceType {
    let name = name.to_lowercase();

    if name.starts_with("lo") {
        InterfaceType::Loopback
    } else if name.starts_with("eth") || name.starts_with("enp") || name.starts_with("eno") {
        InterfaceType::Ethernet
    } else if name.starts_with("en") {
        // macOS: en0 is usually WiFi on laptops, Ethernet on desktops
        // We'll guess WiFi as it's more common for multipath scenarios
        InterfaceType::Wifi
    } else if name.starts_with("wlan") || name.starts_with("wl") || name.starts_with("wlp") {
        InterfaceType::Wifi
    } else if name.starts_with("cell")
        || name.starts_with("pdp")
        || name.starts_with("rmnet")
        || name.starts_with("wwan")
        || name.starts_with("usb")
    {
        InterfaceType::Cellular
    } else if name.starts_with("tun")
        || name.starts_with("tap")
        || name.starts_with("utun")
        || name.starts_with("wg")
    {
        InterfaceType::Tunnel
    } else if name.starts_with("bridge")
        || name.starts_with("br")
        || name.starts_with("virbr")
        || name.starts_with("docker")
    {
        InterfaceType::Ethernet
    } else if name.starts_with("veth") || name.starts_with("vnet") {
        InterfaceType::Tunnel
    } else if name.starts_with("bond") || name.starts_with("team") {
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
pub fn parse_addr_with_default_port(
    s: &str,
    default_port: u16,
) -> Result<SocketAddr, std::net::AddrParseError> {
    if s.contains(':') && !s.starts_with('[') {
        // IPv4 with port or just IPv4
        if s.matches(':').count() == 1 {
            s.parse()
        } else {
            // Multiple colons - might be IPv6
            format!("[{s}]:{default_port}").parse()
        }
    } else if s.starts_with('[') {
        // IPv6 with port
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
        if libc::gethostname(buf.as_mut_ptr().cast::<libc::c_char>(), buf.len()) == 0 {
            let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            String::from_utf8(buf[..len].to_vec()).ok()
        } else {
            None
        }
    }
}

/// Get interface index by name.
#[cfg(unix)]
pub fn if_nametoindex(name: &str) -> Option<u32> {
    use std::ffi::CString;
    let cname = CString::new(name).ok()?;
    let idx = unsafe { libc::if_nametoindex(cname.as_ptr()) };
    if idx == 0 {
        None
    } else {
        Some(idx)
    }
}

#[cfg(not(unix))]
pub fn if_nametoindex(_name: &str) -> Option<u32> {
    None
}

/// Get interface name by index.
#[cfg(unix)]
pub fn if_indextoname(index: u32) -> Option<String> {
    let mut buf = [0u8; libc::IF_NAMESIZE];
    let result = unsafe { libc::if_indextoname(index, buf.as_mut_ptr().cast::<libc::c_char>()) };
    if result.is_null() {
        None
    } else {
        let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8(buf[..len].to_vec()).ok()
    }
}

#[cfg(not(unix))]
pub fn if_indextoname(_index: u32) -> Option<String> {
    None
}
