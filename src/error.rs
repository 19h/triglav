//! Error types for Triglav.

use std::fmt;
use std::io;
use std::net::SocketAddr;

use thiserror::Error;

/// Result type alias for Triglav operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for Triglav.
#[derive(Error, Debug)]
pub enum Error {
    // Cryptographic errors
    #[error("cryptographic error: {0}")]
    Crypto(#[from] CryptoError),

    #[error("authentication failed: {0}")]
    Authentication(String),

    #[error("invalid key: {0}")]
    InvalidKey(String),

    #[error("handshake failed: {0}")]
    HandshakeFailed(String),

    // Transport errors
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),

    #[error("connection failed to {addr}: {reason}")]
    ConnectionFailed { addr: SocketAddr, reason: String },

    #[error("connection closed")]
    ConnectionClosed,

    #[error("connection timeout")]
    ConnectionTimeout,

    // Protocol errors
    #[error("protocol error: {0}")]
    Protocol(#[from] ProtocolError),

    #[error("invalid packet: {0}")]
    InvalidPacket(String),

    #[error("sequence error: expected {expected}, got {got}")]
    SequenceError { expected: u64, got: u64 },

    // Multipath errors
    #[error("no available uplinks")]
    NoAvailableUplinks,

    #[error("uplink {0} not found")]
    UplinkNotFound(String),

    #[error("all uplinks failed")]
    AllUplinksFailed,

    // Configuration errors
    #[error("configuration error: {0}")]
    Config(String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    // IO errors
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    // General errors
    #[error("internal error: {0}")]
    Internal(String),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Cryptographic operation errors.
#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("encryption failed: {0}")]
    EncryptionFailed(String),

    #[error("decryption failed: {0}")]
    DecryptionFailed(String),

    #[error("key derivation failed: {0}")]
    KeyDerivationFailed(String),

    #[error("signature verification failed")]
    SignatureVerificationFailed,

    #[error("invalid nonce")]
    InvalidNonce,

    #[error("invalid ciphertext length")]
    InvalidCiphertextLength,

    #[error("noise protocol error: {0}")]
    NoiseProtocol(String),
}

/// Transport layer errors.
#[derive(Error, Debug)]
pub enum TransportError {
    #[error("bind failed on {addr}: {reason}")]
    BindFailed { addr: SocketAddr, reason: String },

    #[error("send failed: {0}")]
    SendFailed(String),

    #[error("receive failed: {0}")]
    ReceiveFailed(String),

    #[error("socket error: {0}")]
    SocketError(String),

    #[error("address not available: {0}")]
    AddressNotAvailable(SocketAddr),

    #[error("UDP error: {0}")]
    Udp(String),

    #[error("TCP error: {0}")]
    Tcp(String),

    #[error("QUIC error: {0}")]
    Quic(String),

    #[error("MTU exceeded: packet size {size}, max {max}")]
    MtuExceeded { size: usize, max: usize },
}

/// Protocol parsing and handling errors.
#[derive(Error, Debug)]
pub enum ProtocolError {
    #[error("invalid message type: {0}")]
    InvalidMessageType(u8),

    #[error("invalid version: expected {expected}, got {got}")]
    InvalidVersion { expected: u8, got: u8 },

    #[error("malformed header")]
    MalformedHeader,

    #[error("malformed packet: {0}")]
    MalformedPacket(String),

    #[error("checksum mismatch")]
    ChecksumMismatch,

    #[error("payload too large: {size} bytes (max {max})")]
    PayloadTooLarge { size: usize, max: usize },

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("deserialization error: {0}")]
    Deserialization(String),

    #[error("unexpected message: expected {expected}, got {got}")]
    UnexpectedMessage { expected: String, got: String },

    #[error("session not found: {0}")]
    SessionNotFound(String),
}

impl Error {
    /// Check if error is recoverable (should retry).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Error::ConnectionTimeout
                | Error::Transport(
                    TransportError::SendFailed(_) | TransportError::ReceiveFailed(_)
                )
                | Error::Io(_)
        )
    }

    /// Check if error indicates connection should be reset.
    pub fn should_reconnect(&self) -> bool {
        matches!(
            self,
            Error::ConnectionClosed
                | Error::ConnectionFailed { .. }
                | Error::HandshakeFailed(_)
                | Error::Protocol(ProtocolError::InvalidVersion { .. })
        )
    }
}

/// Error context for debugging.
#[derive(Debug)]
pub struct ErrorContext {
    pub uplink_id: Option<String>,
    pub peer_addr: Option<SocketAddr>,
    pub operation: String,
    pub timestamp: std::time::Instant,
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "operation={}", self.operation)?;
        if let Some(ref uplink) = self.uplink_id {
            write!(f, ", uplink={uplink}")?;
        }
        if let Some(addr) = self.peer_addr {
            write!(f, ", peer={addr}")?;
        }
        Ok(())
    }
}
