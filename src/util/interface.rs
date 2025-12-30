//! Network interface discovery.
//!
//! Platform-specific implementations for enumerating network interfaces.

use std::collections::HashMap;
use std::net::IpAddr;

use super::{guess_interface_type, NetworkInterface};

/// Get all network interfaces with their addresses.
#[cfg(target_os = "linux")]
pub fn get_network_interfaces() -> Vec<NetworkInterface> {
    get_linux_interfaces()
}

#[cfg(target_os = "macos")]
pub fn get_network_interfaces() -> Vec<NetworkInterface> {
    get_macos_interfaces()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn get_network_interfaces() -> Vec<NetworkInterface> {
    vec![]
}

/// Get interface addresses only (name -> list of IPs).
pub fn get_interface_addresses() -> HashMap<String, Vec<IpAddr>> {
    let interfaces = get_network_interfaces();
    let mut map: HashMap<String, Vec<IpAddr>> = HashMap::new();
    
    for iface in interfaces {
        map.entry(iface.name.clone())
            .or_default()
            .push(iface.address);
    }
    
    map
}

/// Get primary address for an interface.
pub fn get_interface_primary_address(name: &str) -> Option<IpAddr> {
    let interfaces = get_network_interfaces();
    
    // Prefer IPv4, then IPv6
    let mut ipv4 = None;
    let mut ipv6 = None;
    
    for iface in interfaces {
        if iface.name == name && iface.is_up && !iface.is_loopback {
            match iface.address {
                IpAddr::V4(v4) if !v4.is_loopback() && !v4.is_link_local() => {
                    ipv4 = Some(IpAddr::V4(v4));
                }
                IpAddr::V6(v6) if !v6.is_loopback() => {
                    // Skip link-local IPv6
                    let segments = v6.segments();
                    if segments[0] != 0xfe80 {
                        ipv6 = Some(IpAddr::V6(v6));
                    }
                }
                _ => {}
            }
        }
    }
    
    ipv4.or(ipv6)
}

/// Get all usable interfaces (up, not loopback, has addresses).
pub fn get_usable_interfaces() -> Vec<NetworkInterface> {
    get_network_interfaces()
        .into_iter()
        .filter(|i| {
            i.is_up && !i.is_loopback && 
            match i.address {
                IpAddr::V4(v4) => !v4.is_loopback() && !v4.is_link_local(),
                IpAddr::V6(v6) => {
                    !v6.is_loopback() && {
                        let segments = v6.segments();
                        segments[0] != 0xfe80 // not link-local
                    }
                }
            }
        })
        .collect()
}

// ============================================================================
// Linux Implementation using /proc and /sys
// ============================================================================

#[cfg(target_os = "linux")]
fn get_linux_interfaces() -> Vec<NetworkInterface> {
    use std::fs;
    use std::path::Path;
    
    let mut interfaces = Vec::new();
    let mut seen: HashMap<(String, IpAddr), bool> = HashMap::new();
    
    // First, get interface list from /sys/class/net
    let net_path = Path::new("/sys/class/net");
    let entries = match fs::read_dir(net_path) {
        Ok(e) => e,
        Err(_) => return get_linux_interfaces_via_ioctl(),
    };
    
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let iface_path = entry.path();
        
        // Read interface index
        let index = fs::read_to_string(iface_path.join("ifindex"))
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(0);
        
        // Read flags to determine if up
        let flags = fs::read_to_string(iface_path.join("flags"))
            .ok()
            .and_then(|s| {
                let s = s.trim().trim_start_matches("0x");
                u32::from_str_radix(s, 16).ok()
            })
            .unwrap_or(0);
        
        let is_up = (flags & libc::IFF_UP as u32) != 0;
        let is_running = (flags & libc::IFF_RUNNING as u32) != 0;
        let is_loopback = (flags & libc::IFF_LOOPBACK as u32) != 0;
        
        // Read MTU
        let mtu = fs::read_to_string(iface_path.join("mtu"))
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(1500);
        
        // Read MAC address
        let mac_address = fs::read_to_string(iface_path.join("address"))
            .ok()
            .and_then(|s| parse_mac_address(s.trim()));
        
        // Read addresses from /proc/net/if_inet6 and via ioctl
        let addresses = get_interface_addresses_linux(&name);
        
        for (addr, netmask, broadcast) in addresses {
            let key = (name.clone(), addr);
            if seen.contains_key(&key) {
                continue;
            }
            seen.insert(key, true);
            
            interfaces.push(NetworkInterface {
                name: name.clone(),
                index,
                address: addr,
                netmask,
                broadcast,
                is_up,
                is_running,
                is_loopback,
                mtu,
                interface_type: guess_interface_type(&name),
                mac_address,
            });
        }
    }
    
    interfaces
}

#[cfg(target_os = "linux")]
fn get_interface_addresses_linux(name: &str) -> Vec<(IpAddr, Option<IpAddr>, Option<IpAddr>)> {
    let mut addresses = Vec::new();
    
    // Get IPv4 addresses via ioctl
    if let Some(v4_addrs) = get_ipv4_addresses_ioctl(name) {
        addresses.extend(v4_addrs);
    }
    
    // Get IPv6 addresses from /proc/net/if_inet6
    if let Some(v6_addrs) = get_ipv6_addresses_proc(name) {
        addresses.extend(v6_addrs);
    }
    
    addresses
}

#[cfg(target_os = "linux")]
fn get_ipv4_addresses_ioctl(name: &str) -> Option<Vec<(IpAddr, Option<IpAddr>, Option<IpAddr>)>> {
    use std::ffi::CString;
    use std::mem::MaybeUninit;
    use std::os::fd::AsRawFd;
    
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    let fd = socket.as_raw_fd();
    
    let mut ifr: libc::ifreq = unsafe { MaybeUninit::zeroed().assume_init() };
    let name_bytes = name.as_bytes();
    let copy_len = name_bytes.len().min(libc::IFNAMSIZ - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(
            name_bytes.as_ptr(),
            ifr.ifr_name.as_mut_ptr() as *mut u8,
            copy_len,
        );
    }
    
    // Get address
    let ret = unsafe { libc::ioctl(fd, libc::SIOCGIFADDR, &mut ifr) };
    if ret != 0 {
        return None;
    }
    
    let addr = unsafe {
        let sockaddr = &ifr.ifr_ifru.ifru_addr as *const _ as *const libc::sockaddr_in;
        std::net::Ipv4Addr::from(u32::from_be((*sockaddr).sin_addr.s_addr))
    };
    
    // Get netmask
    let netmask = {
        let ret = unsafe { libc::ioctl(fd, libc::SIOCGIFNETMASK, &mut ifr) };
        if ret == 0 {
            unsafe {
                let sockaddr = &ifr.ifr_ifru.ifru_netmask as *const _ as *const libc::sockaddr_in;
                Some(IpAddr::V4(std::net::Ipv4Addr::from(u32::from_be((*sockaddr).sin_addr.s_addr))))
            }
        } else {
            None
        }
    };
    
    // Get broadcast
    let broadcast = {
        let ret = unsafe { libc::ioctl(fd, libc::SIOCGIFBRDADDR, &mut ifr) };
        if ret == 0 {
            unsafe {
                let sockaddr = &ifr.ifr_ifru.ifru_broadaddr as *const _ as *const libc::sockaddr_in;
                Some(IpAddr::V4(std::net::Ipv4Addr::from(u32::from_be((*sockaddr).sin_addr.s_addr))))
            }
        } else {
            None
        }
    };
    
    Some(vec![(IpAddr::V4(addr), netmask, broadcast)])
}

#[cfg(target_os = "linux")]
fn get_ipv6_addresses_proc(name: &str) -> Option<Vec<(IpAddr, Option<IpAddr>, Option<IpAddr>)>> {
    use std::fs;
    
    let content = fs::read_to_string("/proc/net/if_inet6").ok()?;
    let mut addresses = Vec::new();
    
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 6 && parts[5] == name {
            // Parse the hex IPv6 address
            let hex_addr = parts[0];
            if hex_addr.len() == 32 {
                let mut bytes = [0u8; 16];
                for i in 0..16 {
                    if let Ok(b) = u8::from_str_radix(&hex_addr[i*2..i*2+2], 16) {
                        bytes[i] = b;
                    }
                }
                let addr = std::net::Ipv6Addr::from(bytes);
                
                // Parse prefix length
                let prefix_len: u8 = parts[2].parse().unwrap_or(128);
                let netmask = prefix_len_to_ipv6_mask(prefix_len);
                
                addresses.push((IpAddr::V6(addr), Some(IpAddr::V6(netmask)), None));
            }
        }
    }
    
    Some(addresses)
}

#[cfg(target_os = "linux")]
fn prefix_len_to_ipv6_mask(prefix_len: u8) -> std::net::Ipv6Addr {
    let mut mask = [0u8; 16];
    let full_bytes = (prefix_len / 8) as usize;
    let remaining_bits = prefix_len % 8;
    
    for byte in mask.iter_mut().take(full_bytes) {
        *byte = 0xff;
    }
    
    if full_bytes < 16 && remaining_bits > 0 {
        mask[full_bytes] = 0xff << (8 - remaining_bits);
    }
    
    std::net::Ipv6Addr::from(mask)
}

#[cfg(target_os = "linux")]
fn get_linux_interfaces_via_ioctl() -> Vec<NetworkInterface> {
    // Fallback using getifaddrs
    get_interfaces_via_getifaddrs()
}

// ============================================================================
// macOS Implementation
// ============================================================================

#[cfg(target_os = "macos")]
fn get_macos_interfaces() -> Vec<NetworkInterface> {
    get_interfaces_via_getifaddrs()
}

// ============================================================================
// Common getifaddrs implementation
// ============================================================================

#[cfg(unix)]
fn get_interfaces_via_getifaddrs() -> Vec<NetworkInterface> {
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

                #[allow(clippy::cast_ptr_alignment)]
                let (addr, netmask, broadcast) = match family {
                    libc::AF_INET => {
                        let sockaddr = ifa.ifa_addr.cast::<libc::sockaddr_in>();
                        let ip = std::net::Ipv4Addr::from(u32::from_be((*sockaddr).sin_addr.s_addr));
                        
                        let netmask = if !ifa.ifa_netmask.is_null() {
                            let mask = ifa.ifa_netmask.cast::<libc::sockaddr_in>();
                            Some(IpAddr::V4(std::net::Ipv4Addr::from(u32::from_be((*mask).sin_addr.s_addr))))
                        } else {
                            None
                        };
                        
                        // On macOS, use ifa_dstaddr for broadcast; on Linux use ifa_broadaddr
                        #[cfg(target_os = "macos")]
                        let broadcast = if !ifa.ifa_dstaddr.is_null() && (ifa.ifa_flags as i32 & libc::IFF_BROADCAST) != 0 {
                            let bcast = ifa.ifa_dstaddr.cast::<libc::sockaddr_in>();
                            Some(IpAddr::V4(std::net::Ipv4Addr::from(u32::from_be((*bcast).sin_addr.s_addr))))
                        } else {
                            None
                        };
                        
                        #[cfg(target_os = "linux")]
                        let broadcast: Option<IpAddr> = {
                            // On Linux, ifa_ifu is a union containing ifu_broadaddr
                            // Access it safely via raw pointer arithmetic
                            // The ifa_ifu field is at offset of ifa_dstaddr on macOS
                            // On Linux, we can use ifa_broadaddr if IFF_BROADCAST is set
                            if (ifa.ifa_flags as i32 & libc::IFF_BROADCAST) != 0 {
                                // ifa_ifu.ifu_broadaddr is the broadcast address
                                // Cast the union to access the broadaddr field
                                let union_ptr = std::ptr::addr_of!(ifa.ifa_ifu) as *const *mut libc::sockaddr;
                                let broadaddr = *union_ptr;
                                if !broadaddr.is_null() {
                                    let bcast = broadaddr.cast::<libc::sockaddr_in>();
                                    Some(IpAddr::V4(std::net::Ipv4Addr::from(u32::from_be((*bcast).sin_addr.s_addr))))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        };
                        
                        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
                        let broadcast: Option<IpAddr> = None;
                        
                        (Some(IpAddr::V4(ip)), netmask, broadcast)
                    }
                    libc::AF_INET6 => {
                        let sockaddr = ifa.ifa_addr.cast::<libc::sockaddr_in6>();
                        let ip = std::net::Ipv6Addr::from((*sockaddr).sin6_addr.s6_addr);
                        
                        let netmask = if !ifa.ifa_netmask.is_null() {
                            let mask = ifa.ifa_netmask.cast::<libc::sockaddr_in6>();
                            Some(IpAddr::V6(std::net::Ipv6Addr::from((*mask).sin6_addr.s6_addr)))
                        } else {
                            None
                        };
                        
                        (Some(IpAddr::V6(ip)), netmask, None)
                    }
                    _ => (None, None, None),
                };

                if let Some(address) = addr {
                    let is_up = (ifa.ifa_flags as i32 & libc::IFF_UP) != 0;
                    let is_running = (ifa.ifa_flags as i32 & libc::IFF_RUNNING) != 0;
                    let is_loopback = (ifa.ifa_flags as i32 & libc::IFF_LOOPBACK) != 0;
                    
                    // Get interface index
                    let index = super::if_nametoindex(&name).unwrap_or(0);

                    interfaces.push(NetworkInterface {
                        name: name.clone(),
                        index,
                        address,
                        netmask,
                        broadcast,
                        is_up,
                        is_running,
                        is_loopback,
                        mtu: 1500, // Would need ioctl to get actual MTU
                        interface_type: guess_interface_type(&name),
                        mac_address: None, // Would need separate query
                    });
                }
            }

            current = ifa.ifa_next;
        }

        libc::freeifaddrs(ifaddrs);
    }

    interfaces
}

/// Parse a MAC address from a string like "aa:bb:cc:dd:ee:ff".
#[allow(dead_code)]
fn parse_mac_address(s: &str) -> Option<[u8; 6]> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 6 {
        return None;
    }
    
    let mut mac = [0u8; 6];
    for (i, part) in parts.iter().enumerate() {
        mac[i] = u8::from_str_radix(part, 16).ok()?;
    }
    Some(mac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_network_interfaces() {
        let interfaces = get_network_interfaces();
        // Should have at least loopback
        assert!(!interfaces.is_empty() || cfg!(not(unix)));
        
        // Check loopback exists on Unix
        #[cfg(unix)]
        {
            let has_loopback = interfaces.iter().any(|i| i.is_loopback);
            assert!(has_loopback, "Should have loopback interface");
        }
    }

    #[test]
    fn test_guess_interface_type() {
        assert_eq!(guess_interface_type("lo"), crate::types::InterfaceType::Loopback);
        assert_eq!(guess_interface_type("eth0"), crate::types::InterfaceType::Ethernet);
        assert_eq!(guess_interface_type("enp0s3"), crate::types::InterfaceType::Ethernet);
        assert_eq!(guess_interface_type("wlan0"), crate::types::InterfaceType::Wifi);
        assert_eq!(guess_interface_type("wlp2s0"), crate::types::InterfaceType::Wifi);
        assert_eq!(guess_interface_type("tun0"), crate::types::InterfaceType::Tunnel);
        assert_eq!(guess_interface_type("wg0"), crate::types::InterfaceType::Tunnel);
    }

    #[test]
    fn test_parse_mac_address() {
        assert_eq!(
            parse_mac_address("aa:bb:cc:dd:ee:ff"),
            Some([0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff])
        );
        assert_eq!(parse_mac_address("invalid"), None);
        assert_eq!(parse_mac_address("aa:bb:cc"), None);
    }
}
