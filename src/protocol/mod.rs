//! Wire protocol for Triglav.
//!
//! Defines the packet format, message types, and serialization.
//!
//! ## Packet Format
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Version (1) │ Type (1) │ Flags (2) │ Sequence (8) │ Timestamp (8) │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                        Session ID (32)                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Uplink ID (2) │ Payload Length (2) │ Checksum (4) │ Payload ... │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

mod packet;
mod message;
mod codec;

pub use packet::{Packet, PacketHeader, PacketFlags, PacketType};
pub use message::{Message, MessageType, ControlMessage, DataMessage};
pub use codec::{encode_packet, decode_packet, PacketCodec};

use crate::PROTOCOL_VERSION;

/// Header size in bytes.
pub const HEADER_SIZE: usize = 60;

/// Maximum payload size.
pub const MAX_PAYLOAD_SIZE: usize = 1400;

/// Minimum packet size (header only).
pub const MIN_PACKET_SIZE: usize = HEADER_SIZE;

/// Calculate CRC32 checksum.
pub fn checksum(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

/// Verify protocol version compatibility.
pub fn is_compatible_version(version: u8) -> bool {
    version == PROTOCOL_VERSION
}
