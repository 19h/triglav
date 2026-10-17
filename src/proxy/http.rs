//! HTTP CONNECT proxy server implementation.
//!
//! Implements HTTP CONNECT tunneling for routing HTTPS traffic
//! through the multipath connection.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tracing::{debug, error, info, warn};

use crate::error::{Error, Result};
use crate::multipath::MultipathManager;

/// HTTP proxy server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpProxyConfig {
    /// Listen address.
    pub listen_addr: SocketAddr,
    /// Connection timeout in seconds.
    pub connect_timeout_secs: u64,
    /// Maximum concurrent connections.
    pub max_connections: usize,
}

impl Default for HttpProxyConfig {
    fn default() -> Self {
        Self {
            listen_addr: SocketAddr::from(([127, 0, 0, 1], 8080)),
            connect_timeout_secs: 30,
            max_connections: 1000,
        }
    }
}

/// HTTP CONNECT proxy server.
pub struct HttpProxyServer {
    config: HttpProxyConfig,
    manager: Arc<MultipathManager>,
    active_connections: Arc<AtomicU64>,
}

impl HttpProxyServer {
    /// Create a new HTTP proxy server.
    pub fn new(config: HttpProxyConfig, manager: Arc<MultipathManager>) -> Self {
        Self {
            config,
            manager,
            active_connections: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Run the HTTP proxy server.
    pub async fn run(&self) -> Result<()> {
        let listener = TcpListener::bind(self.config.listen_addr)
            .await
            .map_err(|e| {
                Error::Transport(crate::error::TransportError::BindFailed {
                    addr: self.config.listen_addr,
                    reason: e.to_string(),
                })
            })?;

        info!(
            "HTTP CONNECT proxy listening on {}",
            self.config.listen_addr
        );

        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    let active = self.active_connections.load(Ordering::Relaxed);
                    if active >= self.config.max_connections as u64 {
                        warn!("Max connections reached, rejecting {}", addr);
                        continue;
                    }

                    self.active_connections.fetch_add(1, Ordering::Relaxed);
                    let manager = Arc::clone(&self.manager);

                    let active = Arc::clone(&self.active_connections);
                    let timeout = std::time::Duration::from_secs(self.config.connect_timeout_secs);
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_client(stream, addr, &manager, timeout).await {
                            debug!("Client {} error: {}", addr, e);
                        }
                        active.fetch_sub(1, Ordering::Relaxed);
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
        stream: TcpStream,
        addr: SocketAddr,
        manager: &MultipathManager,
        timeout: std::time::Duration,
    ) -> Result<()> {
        debug!("New HTTP proxy connection from {}", addr);

        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);

        // Read the first line (request line)
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .await
            .map_err(Error::Io)?;

        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 3 {
            Self::send_error(&mut writer, 400, "Bad Request").await?;
            return Err(Error::Protocol(
                crate::error::ProtocolError::MalformedHeader,
            ));
        }

        let method = parts[0];
        let target = parts[1];

        // Only handle CONNECT method
        if method != "CONNECT" {
            Self::send_error(&mut writer, 405, "Method Not Allowed").await?;
            return Err(Error::Protocol(
                crate::error::ProtocolError::InvalidMessageType(0),
            ));
        }

        let target = target.to_string();
        let (host, port) = target
            .rsplit_once(':')
            .ok_or_else(|| Error::Config("invalid CONNECT target".into()))?;
        let port: u16 = port
            .parse()
            .map_err(|_| Error::Config("invalid CONNECT port".into()))?;

        // Read and discard remaining headers until empty line
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).await.map_err(Error::Io)?;
            if line.trim().is_empty() {
                break;
            }
        }

        debug!("HTTP CONNECT request to {}:{}", host, port);

        if manager.assigned_ipv4().is_some() {
            match super::exit_stream::open(manager, &target, timeout).await {
                Ok(mut tunnel) => {
                    writer
                        .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                        .await?;
                    let (mut tunnel_read, mut tunnel_write) = tokio::io::split(&mut tunnel);
                    let upstream = async {
                        tokio::io::copy(&mut reader, &mut tunnel_write).await?;
                        tunnel_write.shutdown().await
                    };
                    let downstream = async {
                        tokio::io::copy(&mut tunnel_read, &mut writer).await?;
                        writer.shutdown().await
                    };
                    tokio::try_join!(upstream, downstream)?;
                }
                Err(_) => Self::send_error(&mut writer, 502, "Bad Gateway").await?,
            }
            return Ok(());
        }

        // Send connect request through multipath
        let connect_req = format!("CONNECT {host}:{port}\r\n");
        match manager.send(connect_req.as_bytes()).await {
            Ok(_) => {
                // Send success response
                writer
                    .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    .await
                    .map_err(Error::Io)?;
                writer.flush().await.map_err(Error::Io)?;

                debug!("CONNECT tunnel established to {}:{}", host, port);

                // Relay data between client and multipath
                Self::relay_data(reader, writer, manager).await?;
            }
            Err(e) => {
                warn!("Failed to connect to {}:{}: {}", host, port, e);
                Self::send_error(&mut writer, 502, "Bad Gateway").await?;
            }
        }

        Ok(())
    }

    /// Relay data between client and multipath connection.
    async fn relay_data<R, W>(
        mut reader: R,
        mut writer: W,
        manager: &MultipathManager,
    ) -> Result<()>
    where
        R: AsyncReadExt + Unpin,
        W: AsyncWriteExt + Unpin,
    {
        let mut buf = vec![0u8; 65536];

        loop {
            tokio::select! {
                // Read from client, send through multipath
                result = reader.read(&mut buf) => {
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
                            if let Err(e) = writer.write_all(&data).await {
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

    /// Send an HTTP error response.
    async fn send_error<W: AsyncWriteExt + Unpin>(
        writer: &mut W,
        code: u16,
        message: &str,
    ) -> Result<()> {
        let response =
            format!("HTTP/1.1 {code} {message}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        writer
            .write_all(response.as_bytes())
            .await
            .map_err(Error::Io)?;
        writer.flush().await.map_err(Error::Io)?;
        Ok(())
    }
}
