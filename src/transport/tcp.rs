//! TCP transport implementation for fallback connectivity.
//!
//! Provides reliable transport through restrictive networks where UDP may be blocked.

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream as TokioTcpStream};
use tokio::sync::Mutex;
use tokio::time::timeout;

use super::{SocketConfig, Transport, TransportConfig};
use crate::error::{Result, TransportError};

/// TCP transport for reliable communication.
pub struct TcpTransport {
    /// Active stream (if connected).
    stream: RwLock<Option<Arc<Mutex<TokioTcpStream>>>>,
    /// Listener for server mode.
    listener: Option<TcpListener>,
    /// Configuration.
    config: TransportConfig,
    /// Local address.
    local_addr: SocketAddr,
    /// Remote address (if connected).
    remote_addr: RwLock<Option<SocketAddr>>,
}

impl TcpTransport {
    /// Bind to a local address (for listening).
    pub fn bind(addr: SocketAddr, config: &TransportConfig) -> Result<Self> {
        let socket_config = SocketConfig::from_transport_config(config);
        let std_socket = super::socket::create_tcp_socket(addr, &socket_config)?;

        // Set to non-blocking and convert to tokio
        std_socket
            .set_nonblocking(true)
            .map_err(|e| TransportError::BindFailed {
                addr,
                reason: e.to_string(),
            })?;

        std_socket
            .listen(1024)
            .map_err(|e| TransportError::BindFailed {
                addr,
                reason: e.to_string(),
            })?;

        // Convert socket2::Socket to std::net::TcpListener then to tokio
        let std_listener: std::net::TcpListener = std_socket.into();
        let listener =
            TcpListener::from_std(std_listener).map_err(|e| TransportError::BindFailed {
                addr,
                reason: e.to_string(),
            })?;

        let local_addr = listener
            .local_addr()
            .map_err(|e| TransportError::SocketError(e.to_string()))?;

        Ok(Self {
            stream: RwLock::new(None),
            listener: Some(listener),
            config: config.clone(),
            local_addr,
            remote_addr: RwLock::new(None),
        })
    }

    /// Connect to a remote address.
    pub async fn connect(
        remote_addr: SocketAddr,
        bind_addr: Option<SocketAddr>,
        config: &TransportConfig,
    ) -> Result<Self> {
        let bind = bind_addr.unwrap_or_else(|| {
            if remote_addr.is_ipv6() {
                SocketAddr::from(([0u8; 16], 0))
            } else {
                SocketAddr::from(([0u8; 4], 0))
            }
        });

        // Create tokio TCP socket directly
        let tokio_socket = if remote_addr.is_ipv6() {
            tokio::net::TcpSocket::new_v6()
        } else {
            tokio::net::TcpSocket::new_v4()
        }
        .map_err(|e| TransportError::Tcp(e.to_string()))?;

        // Apply socket options
        tokio_socket
            .set_reuseaddr(config.reuse_addr)
            .map_err(|e| TransportError::Tcp(e.to_string()))?;

        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "freebsd"))]
        if config.reuse_port {
            tokio_socket
                .set_reuseport(true)
                .map_err(|e| TransportError::Tcp(e.to_string()))?;
        }

        // Bind
        tokio_socket
            .bind(bind)
            .map_err(|e| TransportError::BindFailed {
                addr: bind,
                reason: e.to_string(),
            })?;

        // Connect with timeout
        let stream = timeout(config.connect_timeout, tokio_socket.connect(remote_addr))
            .await
            .map_err(|_| crate::Error::ConnectionTimeout)?
            .map_err(|e| crate::Error::ConnectionFailed {
                addr: remote_addr,
                reason: e.to_string(),
            })?;

        // Apply TCP options
        if config.tcp_nodelay {
            stream
                .set_nodelay(true)
                .map_err(|e| TransportError::Tcp(e.to_string()))?;
        }

        let local_addr = stream
            .local_addr()
            .map_err(|e| TransportError::SocketError(e.to_string()))?;

        Ok(Self {
            stream: RwLock::new(Some(Arc::new(Mutex::new(stream)))),
            listener: None,
            config: config.clone(),
            local_addr,
            remote_addr: RwLock::new(Some(remote_addr)),
        })
    }

    /// Accept a new connection (server mode).
    pub async fn accept(&self) -> Result<(TcpStream, SocketAddr)> {
        let listener = self
            .listener
            .as_ref()
            .ok_or_else(|| TransportError::Tcp("not in listen mode".into()))?;

        let (stream, addr) = listener
            .accept()
            .await
            .map_err(|e| TransportError::Tcp(e.to_string()))?;

        // Apply TCP options
        if self.config.tcp_nodelay {
            stream
                .set_nodelay(true)
                .map_err(|e| TransportError::Tcp(e.to_string()))?;
        }

        Ok((TcpStream::new(stream), addr))
    }

    /// Get the underlying stream for advanced operations.
    pub fn stream(&self) -> Option<Arc<Mutex<TokioTcpStream>>> {
        self.stream.read().clone()
    }
}

#[async_trait]
impl Transport for TcpTransport {
    fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.local_addr)
    }

    fn remote_addr(&self) -> Option<SocketAddr> {
        *self.remote_addr.read()
    }

    async fn send_to(&self, data: &[u8], _addr: SocketAddr) -> Result<usize> {
        // TCP is connection-oriented, so we just send
        self.send(data).await
    }

    async fn send(&self, data: &[u8]) -> Result<usize> {
        let stream = self
            .stream
            .read()
            .clone()
            .ok_or_else(|| TransportError::SendFailed("not connected".into()))?;

        let mut guard = stream.lock().await;

        // Write length prefix (4 bytes, big-endian) then data
        let len = data.len() as u32;
        guard
            .write_all(&len.to_be_bytes())
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;

        guard
            .write_all(data)
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;

        guard
            .flush()
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;

        Ok(data.len())
    }

    async fn recv_from(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        let len = self.recv(buf).await?;
        let addr = self.remote_addr().unwrap_or(self.local_addr);
        Ok((len, addr))
    }

    async fn recv(&self, buf: &mut [u8]) -> Result<usize> {
        let stream = self
            .stream
            .read()
            .clone()
            .ok_or_else(|| TransportError::ReceiveFailed("not connected".into()))?;

        let mut guard = stream.lock().await;

        // Read length prefix
        let mut len_buf = [0u8; 4];
        guard
            .read_exact(&mut len_buf)
            .await
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()))?;

        let len = u32::from_be_bytes(len_buf) as usize;

        if len > buf.len() {
            return Err(TransportError::ReceiveFailed(format!(
                "message too large: {} > {}",
                len,
                buf.len()
            ))
            .into());
        }

        guard
            .read_exact(&mut buf[..len])
            .await
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()))?;

        Ok(len)
    }

    async fn close(&self) -> Result<()> {
        // Take stream outside of async context to avoid holding lock across await
        let stream = { self.stream.write().take() };
        if let Some(stream) = stream {
            if let Ok(mut guard) = stream.try_lock() {
                let _ = guard.shutdown().await;
            }
        }
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.stream.read().is_some()
    }

    fn transport_type(&self) -> &'static str {
        "tcp"
    }
}

/// Wrapper for TCP stream with length-prefixed framing.
pub struct TcpStream {
    inner: TokioTcpStream,
}

impl TcpStream {
    /// Create from a tokio stream.
    pub fn new(stream: TokioTcpStream) -> Self {
        Self { inner: stream }
    }

    /// Get the inner stream.
    pub fn into_inner(self) -> TokioTcpStream {
        self.inner
    }

    /// Get local address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.inner
            .local_addr()
            .map_err(|e| TransportError::SocketError(e.to_string()).into())
    }

    /// Get peer address.
    pub fn peer_addr(&self) -> Result<SocketAddr> {
        self.inner
            .peer_addr()
            .map_err(|e| TransportError::SocketError(e.to_string()).into())
    }

    /// Send a length-prefixed message.
    pub async fn send(&mut self, data: &[u8]) -> Result<usize> {
        let len = data.len() as u32;
        self.inner
            .write_all(&len.to_be_bytes())
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;

        self.inner
            .write_all(data)
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;

        self.inner
            .flush()
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;

        Ok(data.len())
    }

    /// Receive a length-prefixed message.
    pub async fn recv(&mut self, buf: &mut [u8]) -> Result<usize> {
        let mut len_buf = [0u8; 4];
        self.inner
            .read_exact(&mut len_buf)
            .await
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()))?;

        let len = u32::from_be_bytes(len_buf) as usize;

        if len > buf.len() {
            return Err(TransportError::ReceiveFailed(format!(
                "message too large: {} > {}",
                len,
                buf.len()
            ))
            .into());
        }

        self.inner
            .read_exact(&mut buf[..len])
            .await
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()))?;

        Ok(len)
    }

    /// Shutdown the stream.
    pub async fn shutdown(&mut self) -> Result<()> {
        self.inner
            .shutdown()
            .await
            .map_err(|e| TransportError::Tcp(e.to_string()).into())
    }
}

impl std::ops::Deref for TcpStream {
    type Target = TokioTcpStream;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for TcpStream {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
