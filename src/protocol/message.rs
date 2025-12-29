//! Higher-level message types.

use serde::{Deserialize, Serialize};

use crate::types::{SessionId, TrafficStats};

/// Message type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MessageType {
    // Handshake messages
    ClientHello = 0,
    ServerHello = 1,
    ClientAuth = 2,
    ServerAuthOk = 3,
    ServerAuthFail = 4,

    // Data messages
    Data = 10,
    DataAck = 11,
    DataNack = 12,

    // Control messages
    Ping = 20,
    Pong = 21,
    PathProbe = 22,
    PathProbeAck = 23,

    // Session management
    SessionUpdate = 30,
    UplinkAdd = 31,
    UplinkRemove = 32,
    UplinkStatus = 33,

    // Connection management
    Close = 40,
    CloseAck = 41,
    Error = 42,

    // Quality metrics
    QualityReport = 50,
    BandwidthEstimate = 51,
}

impl MessageType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::ClientHello),
            1 => Some(Self::ServerHello),
            2 => Some(Self::ClientAuth),
            3 => Some(Self::ServerAuthOk),
            4 => Some(Self::ServerAuthFail),
            10 => Some(Self::Data),
            11 => Some(Self::DataAck),
            12 => Some(Self::DataNack),
            20 => Some(Self::Ping),
            21 => Some(Self::Pong),
            22 => Some(Self::PathProbe),
            23 => Some(Self::PathProbeAck),
            30 => Some(Self::SessionUpdate),
            31 => Some(Self::UplinkAdd),
            32 => Some(Self::UplinkRemove),
            33 => Some(Self::UplinkStatus),
            40 => Some(Self::Close),
            41 => Some(Self::CloseAck),
            42 => Some(Self::Error),
            50 => Some(Self::QualityReport),
            51 => Some(Self::BandwidthEstimate),
            _ => None,
        }
    }
}

/// High-level message container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    // Handshake
    ClientHello(ClientHelloMessage),
    ServerHello(ServerHelloMessage),
    ClientAuth(ClientAuthMessage),
    ServerAuthOk(ServerAuthOkMessage),
    ServerAuthFail(ServerAuthFailMessage),

    // Data
    Data(DataMessage),
    DataAck(DataAckMessage),
    DataNack(DataNackMessage),

    // Control
    Control(ControlMessage),

    // Session
    UplinkStatus(UplinkStatusMessage),

    // Close
    Close(CloseMessage),
    Error(ErrorMessage),
}

impl Message {
    /// Get the message type.
    pub fn message_type(&self) -> MessageType {
        match self {
            Self::ClientHello(_) => MessageType::ClientHello,
            Self::ServerHello(_) => MessageType::ServerHello,
            Self::ClientAuth(_) => MessageType::ClientAuth,
            Self::ServerAuthOk(_) => MessageType::ServerAuthOk,
            Self::ServerAuthFail(_) => MessageType::ServerAuthFail,
            Self::Data(_) => MessageType::Data,
            Self::DataAck(_) => MessageType::DataAck,
            Self::DataNack(_) => MessageType::DataNack,
            Self::Control(c) => match c {
                ControlMessage::Ping { .. } => MessageType::Ping,
                ControlMessage::Pong { .. } => MessageType::Pong,
                ControlMessage::PathProbe { .. } => MessageType::PathProbe,
                ControlMessage::PathProbeAck { .. } => MessageType::PathProbeAck,
                ControlMessage::QualityReport(_) => MessageType::QualityReport,
            },
            Self::UplinkStatus(_) => MessageType::UplinkStatus,
            Self::Close(_) => MessageType::Close,
            Self::Error(_) => MessageType::Error,
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, crate::error::ProtocolError> {
        bincode::serialize(self)
            .map_err(|e| crate::error::ProtocolError::Serialization(e.to_string()))
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, crate::error::ProtocolError> {
        bincode::deserialize(data)
            .map_err(|e| crate::error::ProtocolError::Deserialization(e.to_string()))
    }
}

/// Client hello message (initiates handshake).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientHelloMessage {
    /// Client's ephemeral public key (for Noise).
    pub ephemeral_public: [u8; 32],
    /// Requested session ID.
    pub session_id: SessionId,
    /// Client capabilities.
    pub capabilities: ClientCapabilities,
    /// Uplinks the client wants to use.
    pub uplink_count: u8,
}

/// Server hello message (responds to client hello).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerHelloMessage {
    /// Server's ephemeral public key.
    pub ephemeral_public: [u8; 32],
    /// Assigned session ID.
    pub session_id: SessionId,
    /// Server capabilities.
    pub capabilities: ServerCapabilities,
}

/// Client authentication message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientAuthMessage {
    /// Authentication token (derived from key).
    pub auth_token: [u8; 32],
    /// Noise handshake payload (encrypted).
    pub handshake_payload: Vec<u8>,
}

/// Server auth success response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerAuthOkMessage {
    /// Confirmation token.
    pub confirmation: [u8; 32],
    /// Server's address for additional uplinks.
    pub uplink_endpoints: Vec<String>,
}

/// Server auth failure response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerAuthFailMessage {
    /// Error code.
    pub error_code: u32,
    /// Error message.
    pub message: String,
}

/// Data message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMessage {
    /// Stream ID (for multiplexing).
    pub stream_id: u32,
    /// Offset in stream.
    pub offset: u64,
    /// Payload data.
    pub payload: Vec<u8>,
    /// End of stream flag.
    pub fin: bool,
}

/// Data acknowledgment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAckMessage {
    /// Acknowledged sequence numbers.
    pub acked: Vec<u64>,
    /// Selective ACK ranges.
    pub sack_ranges: Vec<(u64, u64)>,
    /// Receive window size.
    pub window: u32,
}

/// Negative acknowledgment (request retransmit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataNackMessage {
    /// Missing sequence numbers.
    pub missing: Vec<u64>,
    /// Reason for NACK.
    pub reason: NackReason,
}

/// Reason for negative acknowledgment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NackReason {
    /// Packet was lost.
    PacketLoss,
    /// Timeout waiting for packet.
    Timeout,
    /// Checksum/integrity failure.
    IntegrityFailure,
    /// Decryption failure.
    DecryptionFailure,
}

/// Control messages for connection management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    Ping {
        timestamp: u64,
        uplink_id: u16,
    },
    Pong {
        ping_timestamp: u64,
        uplink_id: u16,
    },
    PathProbe {
        probe_id: u64,
        uplink_id: u16,
        size: u16,
    },
    PathProbeAck {
        probe_id: u64,
        uplink_id: u16,
        received_size: u16,
    },
    QualityReport(QualityReportMessage),
}

/// Quality report from client or server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReportMessage {
    /// Per-uplink statistics.
    pub uplink_stats: Vec<UplinkQualityStats>,
    /// Aggregate statistics.
    pub aggregate: TrafficStats,
    /// Timestamp of report.
    pub timestamp: u64,
}

/// Per-uplink quality statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UplinkQualityStats {
    /// Uplink identifier.
    pub uplink_id: u16,
    /// Round-trip time in microseconds.
    pub rtt_us: u32,
    /// RTT variance in microseconds.
    pub rtt_var_us: u32,
    /// Packet loss ratio (0.0 - 1.0).
    pub loss_ratio: f32,
    /// Estimated bandwidth in bytes/sec.
    pub bandwidth_bps: u64,
    /// Bytes sent since last report.
    pub bytes_sent: u64,
    /// Bytes received since last report.
    pub bytes_received: u64,
    /// Packets sent.
    pub packets_sent: u64,
    /// Packets dropped.
    pub packets_dropped: u64,
}

/// Uplink status update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UplinkStatusMessage {
    /// Uplink identifier.
    pub uplink_id: u16,
    /// Status.
    pub status: UplinkStatusType,
    /// Optional details.
    pub details: Option<String>,
}

/// Uplink status type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UplinkStatusType {
    /// Uplink is active and healthy.
    Active,
    /// Uplink is degraded.
    Degraded,
    /// Uplink is down.
    Down,
    /// Uplink is being added.
    Adding,
    /// Uplink is being removed.
    Removing,
}

/// Connection close message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloseMessage {
    /// Reason code.
    pub code: u32,
    /// Reason string.
    pub reason: String,
}

/// Error message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessage {
    /// Error code.
    pub code: u32,
    /// Error message.
    pub message: String,
    /// Severity.
    pub severity: ErrorSeverity,
}

/// Error severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Informational (can continue).
    Info,
    /// Warning (degraded operation).
    Warning,
    /// Error (operation failed).
    Error,
    /// Fatal (connection will close).
    Fatal,
}

/// Client capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Maximum number of uplinks supported.
    pub max_uplinks: u8,
    /// Supports compression.
    pub compression: bool,
    /// Supports fast path (UDP).
    pub fast_path: bool,
    /// Supports TCP fallback.
    pub tcp_fallback: bool,
    /// Supports IPv6.
    pub ipv6: bool,
    /// Protocol extensions supported.
    pub extensions: Vec<String>,
}

impl Default for ClientCapabilities {
    fn default() -> Self {
        Self {
            max_uplinks: 8,
            compression: true,
            fast_path: true,
            tcp_fallback: true,
            ipv6: true,
            extensions: vec![],
        }
    }
}

/// Server capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Maximum number of uplinks allowed.
    pub max_uplinks: u8,
    /// Supports compression.
    pub compression: bool,
    /// Supports fast path (UDP).
    pub fast_path: bool,
    /// Supports TCP fallback.
    pub tcp_fallback: bool,
    /// Server version.
    pub version: String,
    /// Protocol extensions supported.
    pub extensions: Vec<String>,
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            max_uplinks: 16,
            compression: true,
            fast_path: true,
            tcp_fallback: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            extensions: vec![],
        }
    }
}
