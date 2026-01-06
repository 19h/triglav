//! Internet connectivity probing.
//!
//! Tests whether network interfaces have actual internet connectivity
//! using multiple methods: STUN, DNS, and HTTP.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tokio::net::UdpSocket;
use tokio::sync::broadcast;
use tracing::debug;

/// STUN servers for connectivity testing and external address discovery.
pub const DEFAULT_STUN_SERVERS: &[&str] = &[
    "stun.l.google.com:19302",
    "stun1.l.google.com:19302",
    "stun.cloudflare.com:3478",
    "stun.stunprotocol.org:3478",
];

/// DNS servers for connectivity testing.
pub const DEFAULT_DNS_SERVERS: &[&str] = &["8.8.8.8:53", "1.1.1.1:53", "9.9.9.9:53"];

/// Connectivity probe configuration.
#[derive(Debug, Clone)]
pub struct ConnectivityConfig {
    /// STUN servers to use.
    pub stun_servers: Vec<String>,
    /// DNS servers to use.
    pub dns_servers: Vec<String>,
    /// Probe timeout.
    pub timeout: Duration,
    /// Probe interval.
    pub probe_interval: Duration,
    /// Number of retries per probe.
    pub retries: u32,
}

impl Default for ConnectivityConfig {
    fn default() -> Self {
        Self {
            stun_servers: DEFAULT_STUN_SERVERS.iter().map(|s| s.to_string()).collect(),
            dns_servers: DEFAULT_DNS_SERVERS.iter().map(|s| s.to_string()).collect(),
            timeout: Duration::from_secs(3),
            probe_interval: Duration::from_secs(30),
            retries: 2,
        }
    }
}

/// Result of a connectivity probe.
#[derive(Debug, Clone)]
pub struct ConnectivityResult {
    /// Interface name.
    pub interface: String,
    /// Local address used.
    pub local_addr: SocketAddr,
    /// Whether the interface has internet connectivity.
    pub has_internet: bool,
    /// External IP address (if discovered via STUN).
    pub external_addr: Option<SocketAddr>,
    /// NAT type detected.
    pub nat_type: NatType,
    /// Round-trip time to probe server.
    pub rtt: Option<Duration>,
    /// Time of probe.
    pub timestamp: Instant,
    /// Error message if probe failed.
    pub error: Option<String>,
}

/// NAT type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatType {
    /// No NAT (direct public IP).
    None,
    /// Full cone NAT (endpoint-independent mapping and filtering).
    FullCone,
    /// Restricted cone NAT (endpoint-independent mapping, address-dependent filtering).
    RestrictedCone,
    /// Port-restricted cone NAT (endpoint-independent mapping, address+port-dependent filtering).
    PortRestrictedCone,
    /// Symmetric NAT (endpoint-dependent mapping).
    Symmetric,
    /// Unknown or could not be determined.
    Unknown,
}

impl std::fmt::Display for NatType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NatType::None => write!(f, "No NAT"),
            NatType::FullCone => write!(f, "Full Cone"),
            NatType::RestrictedCone => write!(f, "Restricted Cone"),
            NatType::PortRestrictedCone => write!(f, "Port Restricted Cone"),
            NatType::Symmetric => write!(f, "Symmetric"),
            NatType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Connectivity prober for network interfaces.
pub struct ConnectivityProber {
    config: ConnectivityConfig,
    /// Cached results by interface.
    results: Arc<RwLock<HashMap<String, ConnectivityResult>>>,
    /// Event channel.
    event_tx: broadcast::Sender<ConnectivityResult>,
    /// Shutdown channel.
    shutdown_tx: broadcast::Sender<()>,
}

impl ConnectivityProber {
    /// Create a new connectivity prober.
    pub fn new(config: ConnectivityConfig) -> Self {
        let (event_tx, _) = broadcast::channel(64);
        let (shutdown_tx, _) = broadcast::channel(1);

        Self {
            config,
            results: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            shutdown_tx,
        }
    }

    /// Subscribe to connectivity events.
    pub fn subscribe(&self) -> broadcast::Receiver<ConnectivityResult> {
        self.event_tx.subscribe()
    }

    /// Get cached result for an interface.
    pub fn get_result(&self, interface: &str) -> Option<ConnectivityResult> {
        self.results.read().get(interface).cloned()
    }

    /// Get all cached results.
    pub fn all_results(&self) -> HashMap<String, ConnectivityResult> {
        self.results.read().clone()
    }

    /// Probe connectivity for a specific interface and local address.
    pub async fn probe(&self, interface: &str, local_addr: IpAddr) -> ConnectivityResult {
        let _start = Instant::now();

        // Try STUN first
        let stun_result = self.probe_stun(interface, local_addr).await;

        if let Some(result) = stun_result {
            // Update cache
            self.results
                .write()
                .insert(interface.to_string(), result.clone());
            let _ = self.event_tx.send(result.clone());
            return result;
        }

        // Fallback to DNS probe
        let dns_result = self.probe_dns(interface, local_addr).await;

        if let Some(result) = dns_result {
            self.results
                .write()
                .insert(interface.to_string(), result.clone());
            let _ = self.event_tx.send(result.clone());
            return result;
        }

        // Failed
        let result = ConnectivityResult {
            interface: interface.to_string(),
            local_addr: SocketAddr::new(local_addr, 0),
            has_internet: false,
            external_addr: None,
            nat_type: NatType::Unknown,
            rtt: None,
            timestamp: Instant::now(),
            error: Some("All connectivity probes failed".to_string()),
        };

        self.results
            .write()
            .insert(interface.to_string(), result.clone());
        let _ = self.event_tx.send(result.clone());
        result
    }

    /// Probe using STUN.
    async fn probe_stun(&self, interface: &str, local_addr: IpAddr) -> Option<ConnectivityResult> {
        for server in &self.config.stun_servers {
            for retry in 0..self.config.retries {
                match self.do_stun_probe(interface, local_addr, server).await {
                    Ok(result) => return Some(result),
                    Err(e) => {
                        debug!(
                            "STUN probe to {} failed (attempt {}): {}",
                            server,
                            retry + 1,
                            e
                        );
                    }
                }
            }
        }
        None
    }

    /// Perform a single STUN probe.
    async fn do_stun_probe(
        &self,
        interface: &str,
        local_addr: IpAddr,
        server: &str,
    ) -> Result<ConnectivityResult, String> {
        let start = Instant::now();

        // Parse server address
        let server_addr: SocketAddr = server
            .parse()
            .map_err(|e| format!("Invalid STUN server address: {}", e))?;

        // Create socket bound to local address
        let bind_addr = SocketAddr::new(local_addr, 0);

        // Bind with interface if possible
        let socket = self
            .create_bound_socket(bind_addr, interface)
            .await
            .map_err(|e| format!("Failed to create socket: {}", e))?;

        // Build STUN binding request
        let txn_id: [u8; 12] = rand::random();
        let request = build_stun_binding_request(&txn_id);

        // Send request
        socket
            .send_to(&request, server_addr)
            .await
            .map_err(|e| format!("Failed to send STUN request: {}", e))?;

        // Wait for response
        let mut buf = [0u8; 1024];
        let timeout_result =
            tokio::time::timeout(self.config.timeout, socket.recv_from(&mut buf)).await;

        let rtt = start.elapsed();

        let (len, _from) = timeout_result
            .map_err(|_| "STUN response timeout".to_string())?
            .map_err(|e| format!("Failed to receive STUN response: {}", e))?;

        // Parse STUN response
        let external_addr = parse_stun_response(&buf[..len], &txn_id)
            .ok_or_else(|| "Failed to parse STUN response".to_string())?;

        // Determine NAT type
        let local = socket.local_addr().map_err(|e| e.to_string())?;
        let nat_type = if external_addr.ip() == local.ip() && external_addr.port() == local.port() {
            NatType::None
        } else if external_addr.ip() == local.ip() {
            // Same IP, different port
            NatType::PortRestrictedCone
        } else {
            // Different IP
            // Would need multiple STUN servers to determine full NAT type
            NatType::Unknown
        };

        Ok(ConnectivityResult {
            interface: interface.to_string(),
            local_addr: local,
            has_internet: true,
            external_addr: Some(external_addr),
            nat_type,
            rtt: Some(rtt),
            timestamp: Instant::now(),
            error: None,
        })
    }

    /// Probe using DNS.
    async fn probe_dns(&self, interface: &str, local_addr: IpAddr) -> Option<ConnectivityResult> {
        for server in &self.config.dns_servers {
            for retry in 0..self.config.retries {
                match self.do_dns_probe(interface, local_addr, server).await {
                    Ok(result) => return Some(result),
                    Err(e) => {
                        debug!(
                            "DNS probe to {} failed (attempt {}): {}",
                            server,
                            retry + 1,
                            e
                        );
                    }
                }
            }
        }
        None
    }

    /// Perform a single DNS probe.
    async fn do_dns_probe(
        &self,
        interface: &str,
        local_addr: IpAddr,
        server: &str,
    ) -> Result<ConnectivityResult, String> {
        let start = Instant::now();

        let server_addr: SocketAddr = server
            .parse()
            .map_err(|e| format!("Invalid DNS server address: {}", e))?;

        let bind_addr = SocketAddr::new(local_addr, 0);
        let socket = self
            .create_bound_socket(bind_addr, interface)
            .await
            .map_err(|e| format!("Failed to create socket: {}", e))?;

        // Build a simple DNS query for "." (root)
        let query = build_dns_query();

        socket
            .send_to(&query, server_addr)
            .await
            .map_err(|e| format!("Failed to send DNS query: {}", e))?;

        let mut buf = [0u8; 512];
        let timeout_result =
            tokio::time::timeout(self.config.timeout, socket.recv_from(&mut buf)).await;

        let rtt = start.elapsed();

        let (len, _) = timeout_result
            .map_err(|_| "DNS response timeout".to_string())?
            .map_err(|e| format!("Failed to receive DNS response: {}", e))?;

        // Basic validation - check it looks like a DNS response
        if len < 12 {
            return Err("DNS response too short".to_string());
        }

        let local = socket.local_addr().map_err(|e| e.to_string())?;

        Ok(ConnectivityResult {
            interface: interface.to_string(),
            local_addr: local,
            has_internet: true,
            external_addr: None, // DNS doesn't tell us external address
            nat_type: NatType::Unknown,
            rtt: Some(rtt),
            timestamp: Instant::now(),
            error: None,
        })
    }

    /// Create a socket bound to the specified address and optionally the interface.
    async fn create_bound_socket(
        &self,
        bind_addr: SocketAddr,
        interface: &str,
    ) -> std::io::Result<UdpSocket> {
        use socket2::{Domain, Protocol, Socket, Type};

        let domain = if bind_addr.is_ipv6() {
            Domain::IPV6
        } else {
            Domain::IPV4
        };
        let socket = Socket::new(domain, Type::DGRAM, Some(Protocol::UDP))?;

        // Bind to interface if supported
        #[cfg(target_os = "linux")]
        {
            use std::ffi::CString;
            use std::os::unix::io::AsRawFd;

            if let Ok(cname) = CString::new(interface) {
                // SO_BINDTODEVICE requires CAP_NET_RAW or root
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
                    debug!(
                        "SO_BINDTODEVICE failed for {}: {}, falling back to address binding",
                        interface,
                        std::io::Error::last_os_error()
                    );
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Some(idx) = super::if_nametoindex(interface) {
                use std::os::unix::io::AsRawFd;

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
                    debug!(
                        "IP_BOUND_IF failed for {}: {}, falling back to address binding",
                        interface,
                        std::io::Error::last_os_error()
                    );
                }
            }
        }

        socket.set_nonblocking(true)?;
        socket.bind(&bind_addr.into())?;

        UdpSocket::from_std(socket.into())
    }

    /// Stop the prober.
    pub fn stop(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

/// Build a STUN Binding Request.
fn build_stun_binding_request(txn_id: &[u8; 12]) -> Vec<u8> {
    let mut request = Vec::with_capacity(20);

    // Message Type: Binding Request (0x0001)
    request.extend_from_slice(&[0x00, 0x01]);
    // Message Length: 0 (no attributes)
    request.extend_from_slice(&[0x00, 0x00]);
    // Magic Cookie
    request.extend_from_slice(&[0x21, 0x12, 0xa4, 0x42]);
    // Transaction ID
    request.extend_from_slice(txn_id);

    request
}

/// Parse a STUN Binding Response and extract the mapped address.
fn parse_stun_response(data: &[u8], expected_txn_id: &[u8; 12]) -> Option<SocketAddr> {
    if data.len() < 20 {
        return None;
    }

    // Check message type (Binding Response: 0x0101)
    if data[0] != 0x01 || data[1] != 0x01 {
        return None;
    }

    // Check magic cookie
    if &data[4..8] != &[0x21, 0x12, 0xa4, 0x42] {
        return None;
    }

    // Check transaction ID
    if &data[8..20] != expected_txn_id {
        return None;
    }

    // Parse attributes
    let msg_len = u16::from_be_bytes([data[2], data[3]]) as usize;
    let attrs_end = 20 + msg_len.min(data.len() - 20);
    let mut pos = 20;

    while pos + 4 <= attrs_end {
        let attr_type = u16::from_be_bytes([data[pos], data[pos + 1]]);
        let attr_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + attr_len > attrs_end {
            break;
        }

        // XOR-MAPPED-ADDRESS (0x0020) or MAPPED-ADDRESS (0x0001)
        if attr_type == 0x0020 || attr_type == 0x0001 {
            if attr_len >= 8 {
                let family = data[pos + 1];
                let port_bytes = [data[pos + 2], data[pos + 3]];
                let port = if attr_type == 0x0020 {
                    // XOR'd port
                    u16::from_be_bytes(port_bytes) ^ 0x2112
                } else {
                    u16::from_be_bytes(port_bytes)
                };

                if family == 0x01 && attr_len >= 8 {
                    // IPv4
                    let addr_bytes = [data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]];
                    let addr = if attr_type == 0x0020 {
                        // XOR'd address
                        let xor_key = [0x21, 0x12, 0xa4, 0x42];
                        [
                            addr_bytes[0] ^ xor_key[0],
                            addr_bytes[1] ^ xor_key[1],
                            addr_bytes[2] ^ xor_key[2],
                            addr_bytes[3] ^ xor_key[3],
                        ]
                    } else {
                        addr_bytes
                    };
                    let ip = std::net::Ipv4Addr::from(addr);
                    return Some(SocketAddr::new(IpAddr::V4(ip), port));
                } else if family == 0x02 && attr_len >= 20 {
                    // IPv6
                    let mut addr_bytes = [0u8; 16];
                    addr_bytes.copy_from_slice(&data[pos + 4..pos + 20]);
                    if attr_type == 0x0020 {
                        // XOR with magic cookie + transaction ID
                        let mut xor_key = [0u8; 16];
                        xor_key[0..4].copy_from_slice(&[0x21, 0x12, 0xa4, 0x42]);
                        xor_key[4..16].copy_from_slice(expected_txn_id);
                        for i in 0..16 {
                            addr_bytes[i] ^= xor_key[i];
                        }
                    }
                    let ip = std::net::Ipv6Addr::from(addr_bytes);
                    return Some(SocketAddr::new(IpAddr::V6(ip), port));
                }
            }
        }

        // Align to 4 bytes
        pos += (attr_len + 3) & !3;
    }

    None
}

/// Build a minimal DNS query.
fn build_dns_query() -> Vec<u8> {
    let mut query = Vec::with_capacity(17);

    // Transaction ID
    let txn_id: u16 = rand::random();
    query.extend_from_slice(&txn_id.to_be_bytes());

    // Flags: standard query, recursion desired
    query.extend_from_slice(&[0x01, 0x00]);

    // Questions: 1
    query.extend_from_slice(&[0x00, 0x01]);
    // Answer RRs: 0
    query.extend_from_slice(&[0x00, 0x00]);
    // Authority RRs: 0
    query.extend_from_slice(&[0x00, 0x00]);
    // Additional RRs: 0
    query.extend_from_slice(&[0x00, 0x00]);

    // Query: root domain "."
    query.push(0x00); // Empty label = root

    // Type: A (1)
    query.extend_from_slice(&[0x00, 0x01]);
    // Class: IN (1)
    query.extend_from_slice(&[0x00, 0x01]);

    query
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_stun_request() {
        let txn_id = [0u8; 12];
        let request = build_stun_binding_request(&txn_id);

        assert_eq!(request.len(), 20);
        assert_eq!(&request[0..2], &[0x00, 0x01]); // Binding Request
        assert_eq!(&request[4..8], &[0x21, 0x12, 0xa4, 0x42]); // Magic Cookie
    }

    #[test]
    fn test_build_dns_query() {
        let query = build_dns_query();

        assert!(query.len() >= 17);
        // Check it's a standard query
        assert_eq!(query[2] & 0x80, 0x00);
    }

    #[test]
    fn test_nat_type_display() {
        assert_eq!(format!("{}", NatType::None), "No NAT");
        assert_eq!(format!("{}", NatType::Symmetric), "Symmetric");
    }
}
