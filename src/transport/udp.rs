//! High-performance UDP transport implementation.
//!
//! Optimized for low-latency packet transmission with:
//! - Zero-copy where possible
//! - Dual-stack IPv4/IPv6 support
//! - Platform-specific optimizations (GSO, GRO on Linux)

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use tokio::net::UdpSocket as TokioUdpSocket;

use super::{Transport, TransportConfig, SocketConfig};
use crate::error::{Result, TransportError};

/// High-performance UDP transport.
pub struct UdpTransport {
    socket: Arc<TokioUdpSocket>,
    remote_addr: RwLock<Option<SocketAddr>>,
    is_connected: std::sync::atomic::AtomicBool,
}

impl UdpTransport {
    /// Bind to a local address.
    pub fn bind(addr: SocketAddr, config: &TransportConfig) -> Result<Self> {
        let socket_config = SocketConfig::from_transport_config(config);
        let std_socket = super::socket::create_udp_socket(addr, &socket_config)?;

        let socket = TokioUdpSocket::from_std(std_socket.into())
            .map_err(|e| TransportError::BindFailed {
                addr,
                reason: e.to_string(),
            })?;

        Ok(Self {
            socket: Arc::new(socket),
            remote_addr: RwLock::new(None),
            is_connected: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Create a connected UDP socket.
    pub async fn connect(
        remote_addr: SocketAddr,
        bind_addr: Option<SocketAddr>,
        config: &TransportConfig,
    ) -> Result<Self> {
        // Determine bind address based on remote address family
        let bind = bind_addr.unwrap_or_else(|| {
            if remote_addr.is_ipv6() {
                SocketAddr::from(([0u8; 16], 0))
            } else {
                SocketAddr::from(([0u8; 4], 0))
            }
        });

        let socket_config = SocketConfig::from_transport_config(config);
        let std_socket = super::socket::create_udp_socket(bind, &socket_config)?;

        let socket = TokioUdpSocket::from_std(std_socket.into())
            .map_err(|e| TransportError::BindFailed {
                addr: bind,
                reason: e.to_string(),
            })?;

        // Connect the socket
        socket
            .connect(remote_addr)
            .await
            .map_err(|e| crate::Error::ConnectionFailed {
                addr: remote_addr,
                reason: e.to_string(),
            })?;

        Ok(Self {
            socket: Arc::new(socket),
            remote_addr: RwLock::new(Some(remote_addr)),
            is_connected: std::sync::atomic::AtomicBool::new(true),
        })
    }

    /// Get a clone of the underlying socket for sharing.
    pub fn socket(&self) -> Arc<TokioUdpSocket> {
        Arc::clone(&self.socket)
    }

    /// Set the remote address for send operations.
    pub fn set_remote(&self, addr: SocketAddr) {
        *self.remote_addr.write() = Some(addr);
    }
}

#[async_trait]
impl Transport for UdpTransport {
    fn local_addr(&self) -> Result<SocketAddr> {
        self.socket
            .local_addr()
            .map_err(|e| TransportError::SocketError(e.to_string()).into())
    }

    fn remote_addr(&self) -> Option<SocketAddr> {
        *self.remote_addr.read()
    }

    async fn send_to(&self, data: &[u8], addr: SocketAddr) -> Result<usize> {
        self.socket
            .send_to(data, addr)
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()).into())
    }

    async fn send(&self, data: &[u8]) -> Result<usize> {
        if self.is_connected.load(std::sync::atomic::Ordering::Acquire) {
            self.socket
                .send(data)
                .await
                .map_err(|e| TransportError::SendFailed(e.to_string()).into())
        } else {
            // Copy the address before await to avoid holding lock across await point
            let addr = { *self.remote_addr.read() };
            if let Some(addr) = addr {
                self.send_to(data, addr).await
            } else {
                Err(TransportError::SendFailed("not connected and no remote address".into()).into())
            }
        }
    }

    async fn recv_from(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        self.socket
            .recv_from(buf)
            .await
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()).into())
    }

    async fn recv(&self, buf: &mut [u8]) -> Result<usize> {
        if self.is_connected.load(std::sync::atomic::Ordering::Acquire) {
            self.socket
                .recv(buf)
                .await
                .map_err(|e| TransportError::ReceiveFailed(e.to_string()).into())
        } else {
            let (len, _) = self.recv_from(buf).await?;
            Ok(len)
        }
    }

    async fn close(&self) -> Result<()> {
        // UDP sockets don't need explicit closing
        self.is_connected
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.is_connected.load(std::sync::atomic::Ordering::Acquire)
    }

    fn transport_type(&self) -> &'static str {
        "udp"
    }
}

/// Wrapper around tokio's UdpSocket with additional utilities.
pub struct UdpSocket {
    inner: TokioUdpSocket,
}

impl UdpSocket {
    /// Create from a tokio socket.
    pub fn from_tokio(socket: TokioUdpSocket) -> Self {
        Self { inner: socket }
    }

    /// Get the inner tokio socket.
    pub fn into_inner(self) -> TokioUdpSocket {
        self.inner
    }

    /// Get a reference to the inner socket.
    pub fn inner(&self) -> &TokioUdpSocket {
        &self.inner
    }

    /// Bind to an address.
    pub async fn bind(addr: SocketAddr) -> Result<Self> {
        let socket = TokioUdpSocket::bind(addr)
            .await
            .map_err(|e| TransportError::BindFailed {
                addr,
                reason: e.to_string(),
            })?;
        Ok(Self { inner: socket })
    }

    /// Get local address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.inner
            .local_addr()
            .map_err(|e| TransportError::SocketError(e.to_string()).into())
    }

    /// Send to address.
    pub async fn send_to(&self, data: &[u8], addr: SocketAddr) -> Result<usize> {
        self.inner
            .send_to(data, addr)
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()).into())
    }

    /// Receive from.
    pub async fn recv_from(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        self.inner
            .recv_from(buf)
            .await
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()).into())
    }
}

impl std::ops::Deref for UdpSocket {
    type Target = TokioUdpSocket;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
