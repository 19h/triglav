//! Cross-platform TUN device implementation.
//!
//! Provides platform-specific TUN device creation and I/O.

use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::error::{Error, Result};

/// TUN device configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunConfig {
    /// Device name (e.g., "tun0", "utun3", "Triglav").
    /// On some platforms this is a hint and the actual name may differ.
    #[serde(default = "default_device_name")]
    pub name: String,

    /// MTU (Maximum Transmission Unit).
    #[serde(default = "default_mtu")]
    pub mtu: u16,

    /// IPv4 address to assign to the interface.
    #[serde(default)]
    pub ipv4_addr: Option<Ipv4Addr>,

    /// IPv4 netmask (e.g., 24 for /24).
    #[serde(default = "default_netmask_v4")]
    pub ipv4_netmask: u8,

    /// IPv6 address to assign to the interface.
    #[serde(default)]
    pub ipv6_addr: Option<Ipv6Addr>,

    /// IPv6 prefix length.
    #[serde(default = "default_netmask_v6")]
    pub ipv6_prefix: u8,

    /// Whether to set this as the default route.
    #[serde(default)]
    pub set_default_route: bool,

    /// Packet queue size for async operations.
    #[serde(default = "default_queue_size")]
    pub queue_size: usize,
}

fn default_device_name() -> String {
    super::recommended_device_name().to_string()
}

fn default_mtu() -> u16 {
    super::DEFAULT_TUN_MTU
}

fn default_netmask_v4() -> u8 {
    24
}

fn default_netmask_v6() -> u8 {
    64
}

fn default_queue_size() -> usize {
    512
}

impl Default for TunConfig {
    fn default() -> Self {
        Self {
            name: default_device_name(),
            mtu: default_mtu(),
            ipv4_addr: Some("10.0.85.1".parse().unwrap()),
            ipv4_netmask: default_netmask_v4(),
            ipv6_addr: None,
            ipv6_prefix: default_netmask_v6(),
            set_default_route: false,
            queue_size: default_queue_size(),
        }
    }
}

/// Handle to a TUN device for reading and writing IP packets.
#[derive(Debug)]
pub struct TunHandle {
    #[cfg(target_os = "linux")]
    inner: LinuxTunHandle,

    #[cfg(target_os = "macos")]
    inner: MacOsTunHandle,

    #[cfg(target_os = "windows")]
    inner: WindowsTunHandle,

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    inner: StubTunHandle,
}

/// Platform-specific TUN device.
pub struct TunDevice {
    /// Configuration used to create this device.
    config: TunConfig,

    /// Actual device name (may differ from requested).
    name: String,

    /// File descriptor or handle for I/O.
    handle: Arc<TunHandle>,

    /// Whether the device is currently up.
    is_up: bool,
}

impl TunDevice {
    /// Create a new TUN device with the given configuration.
    ///
    /// # Privileges
    ///
    /// This operation requires elevated privileges:
    /// - Linux: `CAP_NET_ADMIN` capability or root
    /// - macOS: root privileges
    /// - Windows: Administrator or WinTUN driver access
    pub fn create(config: TunConfig) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::create_linux(config)
        }

        #[cfg(target_os = "macos")]
        {
            Self::create_macos(config)
        }

        #[cfg(target_os = "windows")]
        {
            Self::create_windows(config)
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            Err(Error::Config("TUN devices not supported on this platform".into()))
        }
    }

    /// Get the actual device name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the MTU.
    pub fn mtu(&self) -> u16 {
        self.config.mtu
    }

    /// Get the configuration.
    pub fn config(&self) -> &TunConfig {
        &self.config
    }

    /// Check if the device is up.
    pub fn is_up(&self) -> bool {
        self.is_up
    }

    /// Get a clone of the handle for I/O operations.
    pub fn handle(&self) -> Arc<TunHandle> {
        Arc::clone(&self.handle)
    }

    /// Read an IP packet from the TUN device.
    ///
    /// Returns the number of bytes read into the buffer.
    /// The buffer should be at least MTU bytes.
    pub async fn read(&self, buf: &mut [u8]) -> Result<usize> {
        self.handle.read(buf).await
    }

    /// Write an IP packet to the TUN device.
    ///
    /// Returns the number of bytes written.
    pub async fn write(&self, buf: &[u8]) -> Result<usize> {
        self.handle.write(buf).await
    }

    /// Bring the interface up.
    pub fn up(&mut self) -> Result<()> {
        self.set_interface_up(true)?;
        self.is_up = true;
        Ok(())
    }

    /// Bring the interface down.
    pub fn down(&mut self) -> Result<()> {
        self.set_interface_up(false)?;
        self.is_up = false;
        Ok(())
    }

    /// Configure IP addresses on the interface.
    pub fn configure_addresses(&self) -> Result<()> {
        if let Some(ipv4) = self.config.ipv4_addr {
            self.add_ipv4_address(ipv4, self.config.ipv4_netmask)?;
        }
        if let Some(ipv6) = self.config.ipv6_addr {
            self.add_ipv6_address(ipv6, self.config.ipv6_prefix)?;
        }
        Ok(())
    }

    // Platform-specific implementations

    #[cfg(target_os = "linux")]
    fn create_linux(config: TunConfig) -> Result<Self> {
        use std::os::unix::io::AsRawFd;
        use nix::sys::socket::{socket, AddressFamily, SockType, SockFlag};
        use std::fs::OpenOptions;
        use std::os::unix::fs::OpenOptionsExt;

        // Open /dev/net/tun
        let tun_fd = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open("/dev/net/tun")
            .map_err(|e| Error::Io(e))?;

        // Set up the interface using ioctl
        let mut ifr: libc::ifreq = unsafe { std::mem::zeroed() };
        
        // Copy name (max 15 chars + null terminator)
        let name_bytes = config.name.as_bytes();
        let name_len = name_bytes.len().min(15);
        unsafe {
            std::ptr::copy_nonoverlapping(
                name_bytes.as_ptr(),
                ifr.ifr_name.as_mut_ptr() as *mut u8,
                name_len,
            );
        }

        // IFF_TUN = 0x0001, IFF_NO_PI = 0x1000 (no packet info header)
        ifr.ifr_ifru.ifru_flags = (libc::IFF_TUN | libc::IFF_NO_PI) as i16;

        // TUNSETIFF ioctl
        const TUNSETIFF: libc::c_ulong = 0x400454ca;
        let ret = unsafe {
            libc::ioctl(tun_fd.as_raw_fd(), TUNSETIFF, &mut ifr)
        };

        if ret < 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }

        // Get the actual interface name
        let actual_name = unsafe {
            std::ffi::CStr::from_ptr(ifr.ifr_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        tracing::info!(
            requested = %config.name,
            actual = %actual_name,
            mtu = config.mtu,
            "Created TUN device"
        );

        let handle = Arc::new(TunHandle {
            inner: LinuxTunHandle {
                fd: tun_fd.as_raw_fd(),
                _file: tun_fd,
            },
        });

        let mut device = Self {
            config,
            name: actual_name,
            handle,
            is_up: false,
        };

        // Set MTU
        device.set_mtu(device.config.mtu)?;

        Ok(device)
    }

    #[cfg(target_os = "macos")]
    fn create_macos(config: TunConfig) -> Result<Self> {
        use std::os::unix::io::{FromRawFd, RawFd};

        // Create utun socket
        // PF_SYSTEM = 32, SYSPROTO_CONTROL = 2
        let fd = unsafe {
            libc::socket(32, libc::SOCK_DGRAM, 2)
        };

        if fd < 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }

        // Get the control ID for utun
        #[repr(C)]
        struct CtlInfo {
            ctl_id: u32,
            ctl_name: [u8; 96],
        }

        let mut info: CtlInfo = unsafe { std::mem::zeroed() };
        let utun_control = b"com.apple.net.utun_control\0";
        info.ctl_name[..utun_control.len()].copy_from_slice(utun_control);

        // CTLIOCGINFO
        const CTLIOCGINFO: libc::c_ulong = 0xc0644e03;
        let ret = unsafe { libc::ioctl(fd, CTLIOCGINFO, &mut info) };
        if ret < 0 {
            unsafe { libc::close(fd) };
            return Err(Error::Io(io::Error::last_os_error()));
        }

        // Connect to the control
        #[repr(C)]
        struct SockaddrCtl {
            sc_len: u8,
            sc_family: u8,
            ss_sysaddr: u16,
            sc_id: u32,
            sc_unit: u32,
            sc_reserved: [u32; 5],
        }

        // Try to get the requested utun unit or let system assign
        // sc_unit = 0 means auto-assign, non-zero requests specific unit (1-indexed)
        let sc_unit = if config.name.starts_with("utun") && config.name.len() > 4 {
            // User specified a number like "utun5" -> sc_unit = 6 (1-indexed)
            config.name[4..].parse::<u32>().map(|n| n + 1).unwrap_or(0)
        } else {
            0 // Auto-assign next available
        };

        let mut addr: SockaddrCtl = unsafe { std::mem::zeroed() };
        addr.sc_len = std::mem::size_of::<SockaddrCtl>() as u8;
        addr.sc_family = 32; // AF_SYSTEM
        addr.ss_sysaddr = 2; // AF_SYS_CONTROL
        addr.sc_id = info.ctl_id;
        addr.sc_unit = sc_unit;

        let ret = unsafe {
            libc::connect(
                fd,
                &addr as *const _ as *const libc::sockaddr,
                std::mem::size_of::<SockaddrCtl>() as u32,
            )
        };

        if ret < 0 {
            unsafe { libc::close(fd) };
            return Err(Error::Io(io::Error::last_os_error()));
        }

        // Get the actual interface name
        let mut name_buf = [0u8; 256];
        let mut name_len: libc::socklen_t = 256;
        
        // UTUN_OPT_IFNAME = 2
        let ret = unsafe {
            libc::getsockopt(
                fd,
                2, // SYSPROTO_CONTROL
                2, // UTUN_OPT_IFNAME
                name_buf.as_mut_ptr() as *mut _,
                &mut name_len,
            )
        };

        let actual_name = if ret >= 0 && name_len > 1 {
            String::from_utf8_lossy(&name_buf[..name_len as usize - 1]).into_owned()
        } else {
            // Fallback - shouldn't happen if connect succeeded
            "utun?".to_string()
        };

        // Set non-blocking
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };

        tracing::info!(
            requested = %config.name,
            actual = %actual_name,
            mtu = config.mtu,
            "Created TUN device"
        );

        let handle = Arc::new(TunHandle {
            inner: MacOsTunHandle { fd },
        });

        let mut device = Self {
            config,
            name: actual_name,
            handle,
            is_up: false,
        };

        // Set MTU
        device.set_mtu(device.config.mtu)?;

        Ok(device)
    }

    #[cfg(target_os = "windows")]
    fn create_windows(config: TunConfig) -> Result<Self> {
        // Windows implementation using WinTUN
        // This is a placeholder - full implementation requires wintun crate
        Err(Error::Config("Windows TUN support requires WinTUN driver".into()))
    }

    fn set_mtu(&self, mtu: u16) -> Result<()> {
        #[cfg(unix)]
        {
            use std::process::Command;
            
            let output = Command::new("ifconfig")
                .args([&self.name, "mtu", &mtu.to_string()])
                .output()
                .map_err(|e| Error::Io(e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(Error::Config(format!("Failed to set MTU: {}", stderr)));
            }
        }
        Ok(())
    }

    fn set_interface_up(&self, up: bool) -> Result<()> {
        #[cfg(unix)]
        {
            use std::process::Command;
            
            let flag = if up { "up" } else { "down" };
            let output = Command::new("ifconfig")
                .args([&self.name, flag])
                .output()
                .map_err(|e| Error::Io(e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(Error::Config(format!("Failed to set interface {}: {}", flag, stderr)));
            }
        }
        Ok(())
    }

    fn add_ipv4_address(&self, addr: Ipv4Addr, netmask: u8) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            use std::process::Command;
            
            let cidr = format!("{}/{}", addr, netmask);
            let output = Command::new("ip")
                .args(["addr", "add", &cidr, "dev", &self.name])
                .output()
                .map_err(|e| Error::Io(e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                // Ignore "already exists" errors
                if !stderr.contains("File exists") {
                    return Err(Error::Config(format!("Failed to add IPv4 address: {}", stderr)));
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            
            // macOS uses different syntax: ifconfig utun0 inet 10.0.85.1 10.0.85.1 netmask 255.255.255.0
            let netmask_addr = Ipv4Addr::from(
                !((1u32 << (32 - netmask)) - 1)
            );
            
            let output = Command::new("ifconfig")
                .args([
                    &self.name,
                    "inet",
                    &addr.to_string(),
                    &addr.to_string(), // peer address (point-to-point)
                    "netmask",
                    &netmask_addr.to_string(),
                ])
                .output()
                .map_err(|e| Error::Io(e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(Error::Config(format!("Failed to add IPv4 address: {}", stderr)));
            }
        }

        Ok(())
    }

    fn add_ipv6_address(&self, addr: Ipv6Addr, prefix: u8) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            use std::process::Command;
            
            let cidr = format!("{}/{}", addr, prefix);
            let output = Command::new("ip")
                .args(["-6", "addr", "add", &cidr, "dev", &self.name])
                .output()
                .map_err(|e| Error::Io(e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.contains("File exists") {
                    return Err(Error::Config(format!("Failed to add IPv6 address: {}", stderr)));
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            
            let output = Command::new("ifconfig")
                .args([
                    &self.name,
                    "inet6",
                    &format!("{}/{}", addr, prefix),
                ])
                .output()
                .map_err(|e| Error::Io(e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(Error::Config(format!("Failed to add IPv6 address: {}", stderr)));
            }
        }

        Ok(())
    }
}

impl Drop for TunDevice {
    fn drop(&mut self) {
        // Interface will be automatically removed when fd is closed
        tracing::debug!(name = %self.name, "Closing TUN device");
    }
}

impl TunHandle {
    /// Read an IP packet from the TUN device.
    pub async fn read(&self, buf: &mut [u8]) -> Result<usize> {
        self.inner.read(buf).await
    }

    /// Write an IP packet to the TUN device.
    pub async fn write(&self, buf: &[u8]) -> Result<usize> {
        self.inner.write(buf).await
    }
}

// Platform-specific handle implementations

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct LinuxTunHandle {
    fd: std::os::unix::io::RawFd,
    _file: std::fs::File,
}

#[cfg(target_os = "linux")]
impl LinuxTunHandle {
    async fn read(&self, buf: &mut [u8]) -> Result<usize> {
        use tokio::io::unix::AsyncFd;
        
        let async_fd = AsyncFd::new(self.fd)
            .map_err(|e| Error::Io(e))?;
        
        loop {
            let mut guard = async_fd.readable().await
                .map_err(|e| Error::Io(e))?;
            
            match guard.try_io(|_| {
                let ret = unsafe {
                    libc::read(self.fd, buf.as_mut_ptr() as *mut _, buf.len())
                };
                if ret < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(ret as usize)
                }
            }) {
                Ok(result) => return result.map_err(Error::Io),
                Err(_would_block) => continue,
            }
        }
    }

    async fn write(&self, buf: &[u8]) -> Result<usize> {
        use tokio::io::unix::AsyncFd;
        
        let async_fd = AsyncFd::new(self.fd)
            .map_err(|e| Error::Io(e))?;
        
        loop {
            let mut guard = async_fd.writable().await
                .map_err(|e| Error::Io(e))?;
            
            match guard.try_io(|_| {
                let ret = unsafe {
                    libc::write(self.fd, buf.as_ptr() as *const _, buf.len())
                };
                if ret < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(ret as usize)
                }
            }) {
                Ok(result) => return result.map_err(Error::Io),
                Err(_would_block) => continue,
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct MacOsTunHandle {
    fd: std::os::unix::io::RawFd,
}

#[cfg(target_os = "macos")]
impl Drop for MacOsTunHandle {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
        tracing::debug!(fd = self.fd, "Closed macOS utun socket");
    }
}

#[cfg(target_os = "macos")]
impl MacOsTunHandle {
    async fn read(&self, buf: &mut [u8]) -> Result<usize> {
        use tokio::io::unix::AsyncFd;
        
        // macOS utun prepends a 4-byte header (AF_INET/AF_INET6)
        // We need to strip this when reading
        let mut full_buf = vec![0u8; buf.len() + 4];
        
        let async_fd = unsafe { AsyncFd::new(self.fd) }
            .map_err(|e| Error::Io(e))?;
        
        loop {
            let mut guard = async_fd.readable().await
                .map_err(|e| Error::Io(e))?;
            
            match guard.try_io(|_| {
                let ret = unsafe {
                    libc::read(self.fd, full_buf.as_mut_ptr() as *mut _, full_buf.len())
                };
                if ret < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(ret as usize)
                }
            }) {
                Ok(Ok(len)) if len > 4 => {
                    // Copy without the 4-byte header
                    let data_len = len - 4;
                    buf[..data_len].copy_from_slice(&full_buf[4..len]);
                    return Ok(data_len);
                }
                Ok(Ok(len)) => return Ok(len),
                Ok(Err(e)) => return Err(Error::Io(e)),
                Err(_would_block) => continue,
            }
        }
    }

    async fn write(&self, buf: &[u8]) -> Result<usize> {
        use tokio::io::unix::AsyncFd;
        
        // macOS utun requires a 4-byte header
        // Determine AF from IP version
        let af: u32 = if !buf.is_empty() {
            match buf[0] >> 4 {
                4 => 2,  // AF_INET
                6 => 30, // AF_INET6
                _ => 2,  // Default to AF_INET
            }
        } else {
            2
        };
        
        let mut full_buf = vec![0u8; buf.len() + 4];
        full_buf[..4].copy_from_slice(&af.to_be_bytes());
        full_buf[4..].copy_from_slice(buf);
        
        let async_fd = unsafe { AsyncFd::new(self.fd) }
            .map_err(|e| Error::Io(e))?;
        
        loop {
            let mut guard = async_fd.writable().await
                .map_err(|e| Error::Io(e))?;
            
            match guard.try_io(|_| {
                let ret = unsafe {
                    libc::write(self.fd, full_buf.as_ptr() as *const _, full_buf.len())
                };
                if ret < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(ret as usize)
                }
            }) {
                Ok(Ok(len)) if len > 4 => return Ok(len - 4),
                Ok(Ok(len)) => return Ok(len),
                Ok(Err(e)) => return Err(Error::Io(e)),
                Err(_would_block) => continue,
            }
        }
    }
}

#[cfg(target_os = "windows")]
#[derive(Debug)]
struct WindowsTunHandle {
    // WinTUN session handle
}

#[cfg(target_os = "windows")]
impl WindowsTunHandle {
    async fn read(&self, buf: &mut [u8]) -> Result<usize> {
        Err(Error::Config("Windows TUN not yet implemented".into()))
    }

    async fn write(&self, buf: &[u8]) -> Result<usize> {
        Err(Error::Config("Windows TUN not yet implemented".into()))
    }
}

// Stub for unsupported platforms
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
#[derive(Debug)]
struct StubTunHandle;

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
impl StubTunHandle {
    async fn read(&self, _buf: &mut [u8]) -> Result<usize> {
        Err(Error::Config("TUN not supported on this platform".into()))
    }

    async fn write(&self, _buf: &[u8]) -> Result<usize> {
        Err(Error::Config("TUN not supported on this platform".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TunConfig::default();
        assert_eq!(config.mtu, 1420);
        assert!(config.ipv4_addr.is_some());
    }
}
