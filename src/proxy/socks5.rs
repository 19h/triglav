//! SOCKS5 proxy server implementation.
//!
//! Implements RFC 1928 SOCKS5 protocol with support for:
//! - CONNECT command
//! - No authentication (configurable)
//! - Username/password authentication (configurable)
//! - IPv4 and IPv6 addresses
//! - Domain name resolution

use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::{debug, error, info, warn};

use crate::error::{Error, Result};
use crate::multipath::MultipathManager;

// SOCKS5 constants
const SOCKS_VERSION: u8 = 0x05;
const AUTH_NONE: u8 = 0x00;
const AUTH_PASSWORD: u8 = 0x02;
const AUTH_NO_ACCEPTABLE: u8 = 0xFF;
const CMD_CONNECT: u8 = 0x01;
const ADDR_IPV4: u8 = 0x01;
const ADDR_DOMAIN: u8 = 0x03;
const ADDR_IPV6: u8 = 0x04;
const REPLY_SUCCEEDED: u8 = 0x00;
const REPLY_GENERAL_FAILURE: u8 = 0x01;
const REPLY_NETWORK_UNREACHABLE: u8 = 0x03;
const REPLY_CONNECTION_REFUSED: u8 = 0x05;
const REPLY_TTL_EXPIRED: u8 = 0x06;

/// SOCKS5 server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Socks5Config {
    /// Listen address.
    pub listen_addr: SocketAddr,
    /// Allow no authentication.
    pub allow_no_auth: bool,
    /// Username for authentication (if any).
    pub username: Option<String>,
    /// Password for authentication (if any).
    pub password: Option<String>,
    /// Connection timeout in seconds.
    pub connect_timeout_secs: u64,
    /// Maximum concurrent connections.
    pub max_connections: usize,
}

impl Default for Socks5Config {
    fn default() -> Self {
        Self {
            listen_addr: SocketAddr::from(([127, 0, 0, 1], 1080)),
            allow_no_auth: true,
            username: None,
            password: None,
            connect_timeout_secs: 30,
            max_connections: 1000,
        }
    }
}

/// SOCKS5 target address.
#[derive(Debug, Clone)]
pub enum SocksAddr {
    Ipv4(Ipv4Addr, u16),
    Ipv6(Ipv6Addr, u16),
    Domain(String, u16),
}

impl SocksAddr {
    /// Encode for SOCKS5 response.
    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            Self::Ipv4(ip, port) => {
                buf.push(ADDR_IPV4);
                buf.extend_from_slice(&ip.octets());
                buf.extend_from_slice(&port.to_be_bytes());
            }
            Self::Ipv6(ip, port) => {
                buf.push(ADDR_IPV6);
                buf.extend_from_slice(&ip.octets());
                buf.extend_from_slice(&port.to_be_bytes());
            }
            Self::Domain(domain, port) => {
                buf.push(ADDR_DOMAIN);
                buf.push(domain.len() as u8);
                buf.extend_from_slice(domain.as_bytes());
                buf.extend_from_slice(&port.to_be_bytes());
            }
        }
        buf
    }
}

impl std::fmt::Display for SocksAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ipv4(ip, port) => write!(f, "{ip}:{port}"),
            Self::Ipv6(ip, port) => write!(f, "[{ip}]:{port}"),
            Self::Domain(domain, port) => write!(f, "{domain}:{port}"),
        }
    }
}

/// SOCKS5 proxy server.
pub struct Socks5Server {
    config: Socks5Config,
    manager: Arc<MultipathManager>,
    active_connections: AtomicU64,
}

impl Socks5Server {
    /// Create a new SOCKS5 server.
    pub fn new(config: Socks5Config, manager: Arc<MultipathManager>) -> Self {
        Self {
            config,
            manager,
            active_connections: AtomicU64::new(0),
        }
    }

    /// Run the SOCKS5 server.
    pub async fn run(&self) -> Result<()> {
        let listener = TcpListener::bind(self.config.listen_addr)
            .await
            .map_err(|e| {
                Error::Transport(crate::error::TransportError::BindFailed {
                    addr: self.config.listen_addr,
                    reason: e.to_string(),
                })
            })?;

        info!("SOCKS5 proxy listening on {}", self.config.listen_addr);

        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    let active = self.active_connections.load(Ordering::Relaxed);
                    if active >= self.config.max_connections as u64 {
                        warn!("Max connections reached, rejecting {}", addr);
                        continue;
                    }

                    self.active_connections.fetch_add(1, Ordering::Relaxed);
                    let config = self.config.clone();
                    let manager = Arc::clone(&self.manager);

                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_client(stream, addr, &config, &manager).await {
                            debug!("Client {} error: {}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    error!("Accept error: {}", e);
                }
            }
        }
    }

    /// Handle a client connection.
    async fn handle_client(
        mut stream: TcpStream,
        addr: SocketAddr,
        config: &Socks5Config,
        manager: &MultipathManager,
    ) -> Result<()> {
        debug!("New SOCKS5 connection from {}", addr);

        // Authentication negotiation
        Self::handle_auth(&mut stream, config).await?;

        // Read request
        let target = Self::read_request(&mut stream).await?;
        debug!("SOCKS5 CONNECT request to {}", target);

        // Connect through multipath
        match Self::connect_through_multipath(&target, manager).await {
            Ok(()) => {
                // Send success reply
                Self::send_reply(&mut stream, REPLY_SUCCEEDED, &target).await?;
                debug!("Connected to {} via multipath", target);

                // Relay data
                Self::relay_data(&mut stream, manager).await?;
            }
            Err(e) => {
                warn!("Failed to connect to {}: {}", target, e);
                let reply_code = Self::error_to_reply(&e);
                Self::send_reply(&mut stream, reply_code, &target).await?;
            }
        }

        Ok(())
    }

    /// Handle SOCKS5 authentication.
    async fn handle_auth(stream: &mut TcpStream, config: &Socks5Config) -> Result<()> {
        // Read version and number of methods
        let mut buf = [0u8; 2];
        stream.read_exact(&mut buf).await.map_err(Error::Io)?;

        if buf[0] != SOCKS_VERSION {
            return Err(Error::Protocol(
                crate::error::ProtocolError::InvalidVersion {
                    expected: SOCKS_VERSION,
                    got: buf[0],
                },
            ));
        }

        let nmethods = buf[1] as usize;
        let mut methods = vec![0u8; nmethods];
        stream.read_exact(&mut methods).await.map_err(Error::Io)?;

        // Select authentication method
        let selected_method = if config.allow_no_auth && methods.contains(&AUTH_NONE) {
            AUTH_NONE
        } else if config.username.is_some()
            && config.password.is_some()
            && methods.contains(&AUTH_PASSWORD)
        {
            AUTH_PASSWORD
        } else {
            AUTH_NO_ACCEPTABLE
        };

        // Send selected method
        stream
            .write_all(&[SOCKS_VERSION, selected_method])
            .await
            .map_err(Error::Io)?;

        if selected_method == AUTH_NO_ACCEPTABLE {
            return Err(Error::Authentication(
                "No acceptable authentication method".into(),
            ));
        }

        // Handle password authentication if selected
        if selected_method == AUTH_PASSWORD {
            Self::handle_password_auth(stream, config).await?;
        }

        Ok(())
    }

    /// Handle username/password authentication.
    async fn handle_password_auth(stream: &mut TcpStream, config: &Socks5Config) -> Result<()> {
        // Read auth version
        let mut buf = [0u8; 1];
        stream.read_exact(&mut buf).await.map_err(Error::Io)?;

        if buf[0] != 0x01 {
            return Err(Error::Authentication("Invalid auth version".into()));
        }

        // Read username
        stream.read_exact(&mut buf).await.map_err(Error::Io)?;
        let ulen = buf[0] as usize;
        let mut username = vec![0u8; ulen];
        stream.read_exact(&mut username).await.map_err(Error::Io)?;

        // Read password
        stream.read_exact(&mut buf).await.map_err(Error::Io)?;
        let plen = buf[0] as usize;
        let mut password = vec![0u8; plen];
        stream.read_exact(&mut password).await.map_err(Error::Io)?;

        // Verify credentials
        let username_str = String::from_utf8_lossy(&username);
        let password_str = String::from_utf8_lossy(&password);

        let valid = config
            .username
            .as_ref()
            .is_some_and(|u| u == &*username_str)
            && config
                .password
                .as_ref()
                .is_some_and(|p| p == &*password_str);

        // Send auth result
        let status = u8::from(!valid);
        stream.write_all(&[0x01, status]).await.map_err(Error::Io)?;

        if !valid {
            return Err(Error::Authentication("Invalid credentials".into()));
        }

        Ok(())
    }

    /// Read SOCKS5 request.
    async fn read_request(stream: &mut TcpStream) -> Result<SocksAddr> {
        // Read header: VER, CMD, RSV, ATYP
        let mut header = [0u8; 4];
        stream.read_exact(&mut header).await.map_err(Error::Io)?;

        if header[0] != SOCKS_VERSION {
            return Err(Error::Protocol(
                crate::error::ProtocolError::InvalidVersion {
                    expected: SOCKS_VERSION,
                    got: header[0],
                },
            ));
        }

        let cmd = header[1];
        let atyp = header[3];

        if cmd != CMD_CONNECT {
            return Err(Error::Protocol(
                crate::error::ProtocolError::InvalidMessageType(cmd),
            ));
        }

        // Read address based on type
        let addr = match atyp {
            ADDR_IPV4 => {
                let mut ip_buf = [0u8; 4];
                stream.read_exact(&mut ip_buf).await.map_err(Error::Io)?;
                let mut port_buf = [0u8; 2];
                stream.read_exact(&mut port_buf).await.map_err(Error::Io)?;
                let port = u16::from_be_bytes(port_buf);
                SocksAddr::Ipv4(Ipv4Addr::from(ip_buf), port)
            }
            ADDR_DOMAIN => {
                let mut len_buf = [0u8; 1];
                stream.read_exact(&mut len_buf).await.map_err(Error::Io)?;
                let len = len_buf[0] as usize;
                let mut domain_buf = vec![0u8; len];
                stream
                    .read_exact(&mut domain_buf)
                    .await
                    .map_err(Error::Io)?;
                let mut port_buf = [0u8; 2];
                stream.read_exact(&mut port_buf).await.map_err(Error::Io)?;
                let port = u16::from_be_bytes(port_buf);
                let domain = String::from_utf8_lossy(&domain_buf).to_string();
                SocksAddr::Domain(domain, port)
            }
            ADDR_IPV6 => {
                let mut ip_buf = [0u8; 16];
                stream.read_exact(&mut ip_buf).await.map_err(Error::Io)?;
                let mut port_buf = [0u8; 2];
                stream.read_exact(&mut port_buf).await.map_err(Error::Io)?;
                let port = u16::from_be_bytes(port_buf);
                SocksAddr::Ipv6(Ipv6Addr::from(ip_buf), port)
            }
            _ => {
                return Err(Error::Protocol(
                    crate::error::ProtocolError::InvalidMessageType(atyp),
                ));
            }
        };

        Ok(addr)
    }

    /// Send SOCKS5 reply.
    async fn send_reply(stream: &mut TcpStream, reply: u8, addr: &SocksAddr) -> Result<()> {
        let mut response = vec![SOCKS_VERSION, reply, 0x00];
        response.extend(addr.encode());
        stream.write_all(&response).await.map_err(Error::Io)?;
        Ok(())
    }

    /// Connect to target through multipath.
    async fn connect_through_multipath(
        target: &SocksAddr,
        manager: &MultipathManager,
    ) -> Result<()> {
        // Encode the connect request
        let connect_req = format!("CONNECT {target}\r\n");

        // Send connect request through multipath
        manager.send(connect_req.as_bytes()).await?;

        // For now, we assume immediate success
        // A full implementation would wait for a response from the server
        Ok(())
    }

    /// Relay data between client and multipath connection.
    async fn relay_data(stream: &mut TcpStream, manager: &MultipathManager) -> Result<()> {
        let mut buf = vec![0u8; 65536];

        loop {
            tokio::select! {
                // Read from client, send through multipath
                result = stream.read(&mut buf) => {
                    match result {
                        Ok(0) => break, // EOF
                        Ok(n) => {
                            if let Err(e) = manager.send(&buf[..n]).await {
                                debug!("Send error: {}", e);
                                break;
                            }
                        }
                        Err(e) => {
                            debug!("Read error: {}", e);
                            break;
                        }
                    }
                }
                // Receive from multipath, write to client
                result = manager.recv() => {
                    match result {
                        Ok((data, _uplink_id)) => {
                            if let Err(e) = stream.write_all(&data).await {
                                debug!("Write error: {}", e);
                                break;
                            }
                        }
                        Err(e) => {
                            if !e.is_recoverable() {
                                debug!("Recv error: {}", e);
                                break;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert error to SOCKS5 reply code.
    fn error_to_reply(error: &Error) -> u8 {
        match error {
            Error::ConnectionTimeout => REPLY_TTL_EXPIRED,
            Error::NoAvailableUplinks => REPLY_NETWORK_UNREACHABLE,
            Error::ConnectionFailed { .. } => REPLY_CONNECTION_REFUSED,
            _ => REPLY_GENERAL_FAILURE,
        }
    }
}
