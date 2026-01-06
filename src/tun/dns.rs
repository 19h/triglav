//! DNS interception and forwarding for TUN tunnel.
//!
//! Handles DNS queries to ensure they go through the tunnel and
//! optionally provides a local DNS resolver.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::net::UdpSocket;

use crate::error::{Error, Result};

/// DNS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnsConfig {
    /// Listen address for local DNS resolver (optional).
    #[serde(default)]
    pub listen_addr: Option<SocketAddr>,

    /// Upstream DNS servers to forward queries to.
    #[serde(default = "default_upstream_dns")]
    pub upstream_servers: Vec<SocketAddr>,

    /// Enable DNS caching.
    #[serde(default = "default_cache_enabled")]
    pub cache_enabled: bool,

    /// Maximum cache entries.
    #[serde(default = "default_cache_size")]
    pub cache_size: usize,

    /// Default TTL for cached entries.
    #[serde(default = "default_cache_ttl", with = "humantime_serde")]
    pub cache_ttl: Duration,

    /// Timeout for upstream DNS queries.
    #[serde(default = "default_query_timeout", with = "humantime_serde")]
    pub query_timeout: Duration,

    /// Block these domains (simple blocklist).
    #[serde(default)]
    pub blocked_domains: Vec<String>,

    /// Force these domains to resolve to specific IPs.
    #[serde(default)]
    pub overrides: HashMap<String, IpAddr>,
}

fn default_upstream_dns() -> Vec<SocketAddr> {
    vec!["1.1.1.1:53".parse().unwrap(), "8.8.8.8:53".parse().unwrap()]
}

fn default_cache_enabled() -> bool {
    true
}

fn default_cache_size() -> usize {
    10000
}

fn default_cache_ttl() -> Duration {
    Duration::from_secs(300)
}

fn default_query_timeout() -> Duration {
    Duration::from_secs(5)
}

impl Default for DnsConfig {
    fn default() -> Self {
        Self {
            listen_addr: None,
            upstream_servers: default_upstream_dns(),
            cache_enabled: default_cache_enabled(),
            cache_size: default_cache_size(),
            cache_ttl: default_cache_ttl(),
            query_timeout: default_query_timeout(),
            blocked_domains: Vec::new(),
            overrides: HashMap::new(),
        }
    }
}

/// Cached DNS entry.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Response data.
    response: Vec<u8>,
    /// When this entry was cached.
    cached_at: Instant,
    /// TTL from the response.
    ttl: Duration,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        self.cached_at.elapsed() > self.ttl
    }
}

/// DNS interceptor and resolver.
pub struct DnsInterceptor {
    config: DnsConfig,

    /// DNS cache.
    cache: RwLock<HashMap<String, CacheEntry>>,

    /// Statistics.
    stats: RwLock<DnsStats>,
}

/// DNS statistics.
#[derive(Debug, Clone, Default)]
pub struct DnsStats {
    pub queries_total: u64,
    pub queries_cached: u64,
    pub queries_forwarded: u64,
    pub queries_blocked: u64,
    pub queries_failed: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl DnsInterceptor {
    /// Create a new DNS interceptor.
    pub fn new(config: DnsConfig) -> Self {
        Self {
            config,
            cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(DnsStats::default()),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> DnsStats {
        self.stats.read().clone()
    }

    /// Process a DNS query packet.
    ///
    /// Returns the response packet to send back.
    pub async fn process_query(&self, query: &[u8], from: SocketAddr) -> Result<Vec<u8>> {
        self.stats.write().queries_total += 1;

        // Parse query to get the domain name
        let domain = match self.parse_query_domain(query) {
            Some(d) => d,
            None => {
                self.stats.write().queries_failed += 1;
                return Err(Error::Protocol(
                    crate::error::ProtocolError::MalformedPacket("Invalid DNS query".into()),
                ));
            }
        };

        tracing::trace!(domain = %domain, from = %from, "DNS query");

        // Check blocklist
        if self.is_blocked(&domain) {
            self.stats.write().queries_blocked += 1;
            return self.create_nxdomain_response(query);
        }

        // Check overrides
        if let Some(&ip) = self.config.overrides.get(&domain) {
            return self.create_override_response(query, ip);
        }

        // Check cache
        if self.config.cache_enabled {
            let cache_key = self.make_cache_key(query);
            if let Some(response) = self.check_cache(&cache_key) {
                self.stats.write().cache_hits += 1;
                self.stats.write().queries_cached += 1;
                return Ok(response);
            }
            self.stats.write().cache_misses += 1;
        }

        // Forward to upstream
        let response = self.forward_query(query).await?;

        // Cache the response
        if self.config.cache_enabled {
            let cache_key = self.make_cache_key(query);
            self.cache_response(&cache_key, &response);
        }

        self.stats.write().queries_forwarded += 1;
        Ok(response)
    }

    /// Forward a DNS query to upstream servers.
    async fn forward_query(&self, query: &[u8]) -> Result<Vec<u8>> {
        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| Error::Io(e))?;

        // Try each upstream server
        for server in &self.config.upstream_servers {
            socket
                .send_to(query, server)
                .await
                .map_err(|e| Error::Io(e))?;

            let mut buf = vec![0u8; 4096];

            match tokio::time::timeout(self.config.query_timeout, socket.recv_from(&mut buf)).await
            {
                Ok(Ok((len, _))) => {
                    buf.truncate(len);
                    return Ok(buf);
                }
                Ok(Err(e)) => {
                    tracing::debug!(server = %server, error = %e, "DNS query error");
                }
                Err(_) => {
                    tracing::debug!(server = %server, "DNS query timeout");
                }
            }
        }

        self.stats.write().queries_failed += 1;
        Err(Error::Config("All DNS servers failed".into()))
    }

    /// Run the DNS server (if listen address is configured).
    pub async fn run(&self) -> Result<()> {
        let listen_addr = match self.config.listen_addr {
            Some(addr) => addr,
            None => return Ok(()),
        };

        let socket = UdpSocket::bind(listen_addr)
            .await
            .map_err(|e| Error::Io(e))?;

        tracing::info!(addr = %listen_addr, "DNS interceptor listening");

        let mut buf = vec![0u8; 4096];

        loop {
            let (len, from) = socket.recv_from(&mut buf).await.map_err(|e| Error::Io(e))?;

            let query = buf[..len].to_vec();

            // Process query
            match self.process_query(&query, from).await {
                Ok(response) => {
                    if let Err(e) = socket.send_to(&response, from).await {
                        tracing::debug!(error = %e, to = %from, "Failed to send DNS response");
                    }
                }
                Err(e) => {
                    tracing::debug!(error = %e, from = %from, "Failed to process DNS query");
                }
            }
        }
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Get cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.read().len()
    }

    /// Remove expired cache entries.
    pub fn cleanup_cache(&self) {
        let mut cache = self.cache.write();
        cache.retain(|_, entry| !entry.is_expired());
    }

    // Helper methods

    fn parse_query_domain(&self, query: &[u8]) -> Option<String> {
        // DNS header is 12 bytes
        if query.len() < 12 {
            return None;
        }

        // Start of question section
        let mut pos = 12;
        let mut domain_parts = Vec::new();

        // Parse domain name labels
        while pos < query.len() {
            let len = query[pos] as usize;
            if len == 0 {
                break;
            }
            if len > 63 || pos + 1 + len > query.len() {
                return None;
            }

            if let Ok(label) = std::str::from_utf8(&query[pos + 1..pos + 1 + len]) {
                domain_parts.push(label.to_lowercase());
            }
            pos += 1 + len;
        }

        if domain_parts.is_empty() {
            None
        } else {
            Some(domain_parts.join("."))
        }
    }

    fn is_blocked(&self, domain: &str) -> bool {
        let domain_lower = domain.to_lowercase();

        for blocked in &self.config.blocked_domains {
            if domain_lower == *blocked || domain_lower.ends_with(&format!(".{}", blocked)) {
                return true;
            }
        }

        false
    }

    fn make_cache_key(&self, query: &[u8]) -> String {
        // Use first 12 bytes (header) + question section as cache key
        // Skip the ID field (first 2 bytes) for better cache hits
        if query.len() < 12 {
            return String::new();
        }

        // Find end of question section
        let mut pos = 12;
        while pos < query.len() && query[pos] != 0 {
            let len = query[pos] as usize;
            if len > 63 {
                break;
            }
            pos += 1 + len;
        }
        pos += 1 + 4; // null byte + qtype + qclass

        if pos > query.len() {
            pos = query.len();
        }

        // Hash the question part (skip ID)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query[2..pos].hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    fn check_cache(&self, key: &str) -> Option<Vec<u8>> {
        let cache = self.cache.read();
        if let Some(entry) = cache.get(key) {
            if !entry.is_expired() {
                return Some(entry.response.clone());
            }
        }
        None
    }

    fn cache_response(&self, key: &str, response: &[u8]) {
        // Extract TTL from response
        let ttl = self.extract_ttl(response).unwrap_or(self.config.cache_ttl);

        let mut cache = self.cache.write();

        // Evict if at capacity
        if cache.len() >= self.config.cache_size {
            // Simple eviction: remove oldest expired, or first entry
            let to_remove: Vec<_> = cache
                .iter()
                .filter(|(_, e)| e.is_expired())
                .map(|(k, _)| k.clone())
                .take(self.config.cache_size / 10)
                .collect();

            for key in to_remove {
                cache.remove(&key);
            }
        }

        cache.insert(
            key.to_string(),
            CacheEntry {
                response: response.to_vec(),
                cached_at: Instant::now(),
                ttl,
            },
        );
    }

    fn extract_ttl(&self, response: &[u8]) -> Option<Duration> {
        // Simple TTL extraction from first answer record
        if response.len() < 12 {
            return None;
        }

        // Skip header (12 bytes) and question section
        let mut pos = 12;

        // Skip question
        while pos < response.len() && response[pos] != 0 {
            let len = response[pos] as usize;
            if len > 63 {
                break;
            }
            pos += 1 + len;
        }
        pos += 1 + 4; // null + qtype + qclass

        // Check ANCOUNT (answer count) in header
        let ancount = u16::from_be_bytes([response[6], response[7]]);
        if ancount == 0 {
            return None;
        }

        // Parse first answer to get TTL
        // Skip name (may be compressed)
        if pos >= response.len() {
            return None;
        }

        if response[pos] & 0xc0 == 0xc0 {
            pos += 2; // Compressed name
        } else {
            while pos < response.len() && response[pos] != 0 {
                let len = response[pos] as usize;
                pos += 1 + len;
            }
            pos += 1;
        }

        // Now at TYPE (2) + CLASS (2) + TTL (4)
        if pos + 8 > response.len() {
            return None;
        }

        pos += 4; // Skip type + class
        let ttl = u32::from_be_bytes([
            response[pos],
            response[pos + 1],
            response[pos + 2],
            response[pos + 3],
        ]);

        Some(Duration::from_secs(ttl as u64))
    }

    fn create_nxdomain_response(&self, query: &[u8]) -> Result<Vec<u8>> {
        // Create a minimal NXDOMAIN response
        if query.len() < 12 {
            return Err(Error::Protocol(
                crate::error::ProtocolError::MalformedPacket("Query too short".into()),
            ));
        }

        let mut response = query.to_vec();

        // Set QR bit (response) and RCODE = 3 (NXDOMAIN)
        response[2] = 0x81; // QR=1, RD=1
        response[3] = 0x03; // RCODE=3 (NXDOMAIN)

        // Clear answer, authority, additional counts
        response[6] = 0;
        response[7] = 0;
        response[8] = 0;
        response[9] = 0;
        response[10] = 0;
        response[11] = 0;

        Ok(response)
    }

    fn create_override_response(&self, query: &[u8], ip: IpAddr) -> Result<Vec<u8>> {
        if query.len() < 12 {
            return Err(Error::Protocol(
                crate::error::ProtocolError::MalformedPacket("Query too short".into()),
            ));
        }

        let mut response = query.to_vec();

        // Set QR bit (response), RCODE = 0
        response[2] = 0x81; // QR=1, RD=1
        response[3] = 0x80; // RA=1

        // Set ANCOUNT = 1
        response[6] = 0;
        response[7] = 1;

        // Clear authority, additional
        response[8] = 0;
        response[9] = 0;
        response[10] = 0;
        response[11] = 0;

        // Add answer record
        // Name pointer to question
        response.extend_from_slice(&[0xc0, 0x0c]);

        match ip {
            IpAddr::V4(ipv4) => {
                // TYPE A
                response.extend_from_slice(&[0x00, 0x01]);
                // CLASS IN
                response.extend_from_slice(&[0x00, 0x01]);
                // TTL (300 seconds)
                response.extend_from_slice(&[0x00, 0x00, 0x01, 0x2c]);
                // RDLENGTH
                response.extend_from_slice(&[0x00, 0x04]);
                // RDATA (IP address)
                response.extend_from_slice(&ipv4.octets());
            }
            IpAddr::V6(ipv6) => {
                // TYPE AAAA
                response.extend_from_slice(&[0x00, 0x1c]);
                // CLASS IN
                response.extend_from_slice(&[0x00, 0x01]);
                // TTL (300 seconds)
                response.extend_from_slice(&[0x00, 0x00, 0x01, 0x2c]);
                // RDLENGTH
                response.extend_from_slice(&[0x00, 0x10]);
                // RDATA (IP address)
                response.extend_from_slice(&ipv6.octets());
            }
        }

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dns_config_default() {
        let config = DnsConfig::default();
        assert!(config.cache_enabled);
        assert!(!config.upstream_servers.is_empty());
    }

    #[test]
    fn test_is_blocked() {
        let mut config = DnsConfig::default();
        config.blocked_domains = vec!["ads.example.com".to_string()];

        let interceptor = DnsInterceptor::new(config);

        assert!(interceptor.is_blocked("ads.example.com"));
        assert!(interceptor.is_blocked("sub.ads.example.com"));
        assert!(!interceptor.is_blocked("example.com"));
    }

    #[test]
    fn test_cache_key() {
        let interceptor = DnsInterceptor::new(DnsConfig::default());

        // Sample DNS query for example.com A record
        let query = vec![
            0x00, 0x01, // ID
            0x01, 0x00, // Flags
            0x00, 0x01, // QDCOUNT
            0x00, 0x00, // ANCOUNT
            0x00, 0x00, // NSCOUNT
            0x00, 0x00, // ARCOUNT
            0x07, b'e', b'x', b'a', b'm', b'p', b'l', b'e', // example
            0x03, b'c', b'o', b'm', // com
            0x00, // null
            0x00, 0x01, // QTYPE A
            0x00, 0x01, // QCLASS IN
        ];

        let key = interceptor.make_cache_key(&query);
        assert!(!key.is_empty());

        // Same query with different ID should produce same cache key
        let mut query2 = query.clone();
        query2[0] = 0xff;
        query2[1] = 0xff;

        let key2 = interceptor.make_cache_key(&query2);
        assert_eq!(key, key2);
    }
}
