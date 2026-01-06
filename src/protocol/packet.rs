//! Packet structure and handling.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use byteorder::{BigEndian, ByteOrder};
use serde::{Deserialize, Serialize};

use crate::error::{ProtocolError, Result};
use crate::types::{SequenceNumber, SessionId};
use crate::PROTOCOL_VERSION;

use super::{HEADER_SIZE, MAX_PAYLOAD_SIZE};

/// Packet type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PacketType {
    /// Data packet.
    Data = 0,
    /// Control/signaling packet.
    Control = 1,
    /// Acknowledgment packet.
    Ack = 2,
    /// Negative acknowledgment (request retransmit).
    Nack = 3,
    /// Ping (keep-alive / latency probe).
    Ping = 4,
    /// Pong (ping response).
    Pong = 5,
    /// Handshake packet.
    Handshake = 6,
    /// Close connection.
    Close = 7,
    /// Error notification.
    Error = 8,
}

impl PacketType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Data),
            1 => Some(Self::Control),
            2 => Some(Self::Ack),
            3 => Some(Self::Nack),
            4 => Some(Self::Ping),
            5 => Some(Self::Pong),
            6 => Some(Self::Handshake),
            7 => Some(Self::Close),
            8 => Some(Self::Error),
            _ => None,
        }
    }

    /// Check if this packet type requires reliable delivery.
    pub fn is_reliable(self) -> bool {
        matches!(
            self,
            Self::Data | Self::Control | Self::Handshake | Self::Close
        )
    }

    /// Check if this is a control packet.
    pub fn is_control(self) -> bool {
        !matches!(self, Self::Data)
    }
}

/// Packet flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PacketFlags(u16);

impl PacketFlags {
    /// No flags set.
    pub const NONE: Self = Self(0);

    /// Packet requires acknowledgment.
    pub const NEED_ACK: u16 = 1 << 0;

    /// Packet is a retransmission.
    pub const RETRANSMIT: u16 = 1 << 1;

    /// Packet is fragmented (more fragments follow).
    pub const FRAGMENT: u16 = 1 << 2;

    /// Last fragment of a fragmented packet.
    pub const LAST_FRAGMENT: u16 = 1 << 3;

    /// Packet is encrypted.
    pub const ENCRYPTED: u16 = 1 << 4;

    /// Packet is compressed.
    pub const COMPRESSED: u16 = 1 << 5;

    /// Priority packet (fast path).
    pub const PRIORITY: u16 = 1 << 6;

    /// Probe packet for path quality measurement.
    pub const PROBE: u16 = 1 << 7;

    /// Create new flags.
    pub fn new(bits: u16) -> Self {
        Self(bits)
    }

    /// Check if a flag is set.
    pub fn has(self, flag: u16) -> bool {
        self.0 & flag != 0
    }

    /// Set a flag.
    pub fn set(&mut self, flag: u16) {
        self.0 |= flag;
    }

    /// Clear a flag.
    pub fn clear(&mut self, flag: u16) {
        self.0 &= !flag;
    }

    /// Get raw bits.
    pub fn bits(self) -> u16 {
        self.0
    }
}

impl Serialize for PacketFlags {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u16(self.0)
    }
}

impl<'de> Deserialize<'de> for PacketFlags {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self(u16::deserialize(deserializer)?))
    }
}

/// Packet header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketHeader {
    /// Protocol version.
    pub version: u8,
    /// Packet type.
    pub packet_type: PacketType,
    /// Flags.
    pub flags: PacketFlags,
    /// Sequence number.
    pub sequence: SequenceNumber,
    /// Timestamp (microseconds since epoch).
    pub timestamp: u64,
    /// Session ID.
    pub session_id: SessionId,
    /// Uplink ID (numeric, for routing).
    pub uplink_id: u16,
    /// Payload length.
    pub payload_len: u16,
    /// Header checksum.
    pub checksum: u32,
}

impl PacketHeader {
    /// Create a new header.
    pub fn new(
        packet_type: PacketType,
        sequence: SequenceNumber,
        session_id: SessionId,
        uplink_id: u16,
        payload_len: usize,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64;

        let mut header = Self {
            version: PROTOCOL_VERSION,
            packet_type,
            flags: PacketFlags::NONE,
            sequence,
            timestamp,
            session_id,
            uplink_id,
            payload_len: payload_len as u16,
            checksum: 0,
        };

        header.checksum = header.compute_checksum();
        header
    }

    /// Compute header checksum (excludes the checksum field itself).
    fn compute_checksum(&self) -> u32 {
        let mut buf = [0u8; HEADER_SIZE - 4]; // Exclude checksum field
        self.encode_without_checksum(&mut buf);
        super::checksum(&buf)
    }

    /// Encode header without checksum field.
    fn encode_without_checksum(&self, buf: &mut [u8]) {
        buf[0] = self.version;
        buf[1] = self.packet_type as u8;
        BigEndian::write_u16(&mut buf[2..4], self.flags.bits());
        BigEndian::write_u64(&mut buf[4..12], self.sequence.0);
        BigEndian::write_u64(&mut buf[12..20], self.timestamp);
        buf[20..52].copy_from_slice(self.session_id.as_bytes());
        BigEndian::write_u16(&mut buf[52..54], self.uplink_id);
        BigEndian::write_u16(&mut buf[54..56], self.payload_len);
    }

    /// Encode header to bytes.
    pub fn encode(&self, buf: &mut [u8]) -> Result<()> {
        if buf.len() < HEADER_SIZE {
            return Err(ProtocolError::MalformedHeader.into());
        }

        self.encode_without_checksum(&mut buf[..HEADER_SIZE - 4]);
        BigEndian::write_u32(&mut buf[56..60], self.checksum);
        Ok(())
    }

    /// Decode header from bytes.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(ProtocolError::MalformedHeader.into());
        }

        let version = buf[0];
        if version != PROTOCOL_VERSION {
            return Err(ProtocolError::InvalidVersion {
                expected: PROTOCOL_VERSION,
                got: version,
            }
            .into());
        }

        let packet_type =
            PacketType::from_u8(buf[1]).ok_or(ProtocolError::InvalidMessageType(buf[1]))?;

        let flags = PacketFlags::new(BigEndian::read_u16(&buf[2..4]));
        let sequence = SequenceNumber(BigEndian::read_u64(&buf[4..12]));
        let timestamp = BigEndian::read_u64(&buf[12..20]);

        let mut session_bytes = [0u8; 32];
        session_bytes.copy_from_slice(&buf[20..52]);
        let session_id = SessionId::new(session_bytes);

        let uplink_id = BigEndian::read_u16(&buf[52..54]);
        let payload_len = BigEndian::read_u16(&buf[54..56]);
        let checksum = BigEndian::read_u32(&buf[56..60]);

        let header = Self {
            version,
            packet_type,
            flags,
            sequence,
            timestamp,
            session_id,
            uplink_id,
            payload_len,
            checksum,
        };

        // Verify checksum
        let computed = header.compute_checksum();
        if computed != checksum {
            return Err(ProtocolError::ChecksumMismatch.into());
        }

        Ok(header)
    }

    /// Get the timestamp as Duration since UNIX epoch.
    pub fn timestamp_duration(&self) -> Duration {
        Duration::from_micros(self.timestamp)
    }

    /// Calculate one-way delay from send timestamp to now.
    pub fn one_way_delay(&self) -> Option<Duration> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_micros() as u64;

        if now >= self.timestamp {
            Some(Duration::from_micros(now - self.timestamp))
        } else {
            None // Clock skew
        }
    }
}

/// Complete packet with header and payload.
#[derive(Debug, Clone)]
pub struct Packet {
    /// Packet header.
    pub header: PacketHeader,
    /// Packet payload.
    pub payload: Vec<u8>,
}

impl Packet {
    /// Create a new packet.
    pub fn new(
        packet_type: PacketType,
        sequence: SequenceNumber,
        session_id: SessionId,
        uplink_id: u16,
        payload: Vec<u8>,
    ) -> Result<Self> {
        if payload.len() > MAX_PAYLOAD_SIZE {
            return Err(ProtocolError::PayloadTooLarge {
                size: payload.len(),
                max: MAX_PAYLOAD_SIZE,
            }
            .into());
        }

        Ok(Self {
            header: PacketHeader::new(packet_type, sequence, session_id, uplink_id, payload.len()),
            payload,
        })
    }

    /// Create a data packet.
    pub fn data(
        sequence: SequenceNumber,
        session_id: SessionId,
        uplink_id: u16,
        payload: Vec<u8>,
    ) -> Result<Self> {
        Self::new(PacketType::Data, sequence, session_id, uplink_id, payload)
    }

    /// Create an ACK packet.
    pub fn ack(
        sequence: SequenceNumber,
        session_id: SessionId,
        uplink_id: u16,
        acked_sequences: &[u64],
    ) -> Result<Self> {
        let payload = bincode::serialize(acked_sequences)
            .map_err(|e| ProtocolError::Serialization(e.to_string()))?;
        Self::new(PacketType::Ack, sequence, session_id, uplink_id, payload)
    }

    /// Create a ping packet.
    pub fn ping(sequence: SequenceNumber, session_id: SessionId, uplink_id: u16) -> Result<Self> {
        Self::new(PacketType::Ping, sequence, session_id, uplink_id, vec![])
    }

    /// Create a pong packet.
    pub fn pong(
        sequence: SequenceNumber,
        session_id: SessionId,
        uplink_id: u16,
        ping_timestamp: u64,
    ) -> Result<Self> {
        let payload = ping_timestamp.to_be_bytes().to_vec();
        Self::new(PacketType::Pong, sequence, session_id, uplink_id, payload)
    }

    /// Encode packet to bytes.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; HEADER_SIZE + self.payload.len()];
        self.header.encode(&mut buf)?;
        buf[HEADER_SIZE..].copy_from_slice(&self.payload);
        Ok(buf)
    }

    /// Decode packet from bytes.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(ProtocolError::MalformedHeader.into());
        }

        let header = PacketHeader::decode(buf)?;

        let expected_len = HEADER_SIZE + header.payload_len as usize;
        if buf.len() < expected_len {
            return Err(ProtocolError::MalformedHeader.into());
        }

        let payload = buf[HEADER_SIZE..expected_len].to_vec();

        Ok(Self { header, payload })
    }

    /// Get total packet size.
    pub fn size(&self) -> usize {
        HEADER_SIZE + self.payload.len()
    }

    /// Set a flag on the packet.
    pub fn set_flag(&mut self, flag: u16) {
        self.header.flags.set(flag);
        self.header.checksum = self.header.compute_checksum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_encode_decode() {
        let session_id = SessionId::generate();
        let header = PacketHeader::new(PacketType::Data, SequenceNumber(42), session_id, 1, 100);

        let mut buf = [0u8; HEADER_SIZE];
        header.encode(&mut buf).unwrap();

        let decoded = PacketHeader::decode(&buf).unwrap();
        assert_eq!(decoded.version, header.version);
        assert_eq!(decoded.packet_type, header.packet_type);
        assert_eq!(decoded.sequence.0, header.sequence.0);
        assert_eq!(decoded.uplink_id, header.uplink_id);
        assert_eq!(decoded.payload_len, header.payload_len);
    }

    #[test]
    fn test_packet_encode_decode() {
        let session_id = SessionId::generate();
        let payload = b"hello world".to_vec();

        let packet = Packet::data(SequenceNumber(1), session_id, 0, payload.clone()).unwrap();

        let encoded = packet.encode().unwrap();
        let decoded = Packet::decode(&encoded).unwrap();

        assert_eq!(decoded.payload, payload);
        assert_eq!(decoded.header.sequence.0, 1);
    }

    #[test]
    fn test_checksum_validation() {
        let session_id = SessionId::generate();
        let packet = Packet::data(SequenceNumber(1), session_id, 0, b"test".to_vec()).unwrap();

        let mut encoded = packet.encode().unwrap();

        // Corrupt a byte
        encoded[10] ^= 0xff;

        assert!(Packet::decode(&encoded).is_err());
    }
}
