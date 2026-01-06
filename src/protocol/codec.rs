//! Packet encoding and decoding.

use bytes::{BufMut, BytesMut};
use tokio_util::codec::{Decoder, Encoder};

use super::{Packet, HEADER_SIZE, MAX_PAYLOAD_SIZE};
use crate::error::{ProtocolError, Result};

/// Encode a packet to bytes.
pub fn encode_packet(packet: &Packet) -> Result<Vec<u8>> {
    packet.encode()
}

/// Decode a packet from bytes.
pub fn decode_packet(data: &[u8]) -> Result<Packet> {
    Packet::decode(data)
}

/// Tokio codec for packet framing.
pub struct PacketCodec {
    max_payload_size: usize,
}

impl PacketCodec {
    /// Create a new codec with default settings.
    pub fn new() -> Self {
        Self {
            max_payload_size: MAX_PAYLOAD_SIZE,
        }
    }

    /// Create a codec with custom max payload size.
    pub fn with_max_payload(max_payload_size: usize) -> Self {
        Self { max_payload_size }
    }
}

impl Default for PacketCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for PacketCodec {
    type Item = Packet;
    type Error = crate::Error;

    fn decode(
        &mut self,
        src: &mut BytesMut,
    ) -> std::result::Result<Option<Self::Item>, Self::Error> {
        // Need at least header to determine packet size
        if src.len() < HEADER_SIZE {
            return Ok(None);
        }

        // Peek at payload length from header
        let payload_len = u16::from_be_bytes([src[54], src[55]]) as usize;

        if payload_len > self.max_payload_size {
            return Err(ProtocolError::PayloadTooLarge {
                size: payload_len,
                max: self.max_payload_size,
            }
            .into());
        }

        let total_len = HEADER_SIZE + payload_len;

        // Wait for complete packet
        if src.len() < total_len {
            src.reserve(total_len - src.len());
            return Ok(None);
        }

        // Extract and decode packet
        let packet_data = src.split_to(total_len);
        let packet = Packet::decode(&packet_data)?;

        Ok(Some(packet))
    }
}

impl Encoder<Packet> for PacketCodec {
    type Error = crate::Error;

    fn encode(&mut self, item: Packet, dst: &mut BytesMut) -> std::result::Result<(), Self::Error> {
        if item.payload.len() > self.max_payload_size {
            return Err(ProtocolError::PayloadTooLarge {
                size: item.payload.len(),
                max: self.max_payload_size,
            }
            .into());
        }

        let encoded = item.encode()?;
        dst.reserve(encoded.len());
        dst.put_slice(&encoded);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ProtocolError;
    use crate::types::{SequenceNumber, SessionId};
    use bytes::{Buf, BufMut};

    /// Framing codec for length-prefixed messages (TCP).
    struct LengthPrefixedCodec {
        max_message_size: usize,
    }

    impl LengthPrefixedCodec {
        fn new(max_message_size: usize) -> Self {
            Self { max_message_size }
        }
    }

    impl Decoder for LengthPrefixedCodec {
        type Item = Vec<u8>;
        type Error = crate::Error;

        fn decode(
            &mut self,
            src: &mut BytesMut,
        ) -> std::result::Result<Option<Self::Item>, Self::Error> {
            if src.len() < 4 {
                return Ok(None);
            }

            let length = u32::from_be_bytes([src[0], src[1], src[2], src[3]]) as usize;

            if length > self.max_message_size {
                return Err(ProtocolError::PayloadTooLarge {
                    size: length,
                    max: self.max_message_size,
                }
                .into());
            }

            let total_len = 4 + length;

            if src.len() < total_len {
                src.reserve(total_len - src.len());
                return Ok(None);
            }

            src.advance(4); // Skip length prefix
            let data = src.split_to(length).to_vec();

            Ok(Some(data))
        }
    }

    impl Encoder<Vec<u8>> for LengthPrefixedCodec {
        type Error = crate::Error;

        fn encode(
            &mut self,
            item: Vec<u8>,
            dst: &mut BytesMut,
        ) -> std::result::Result<(), Self::Error> {
            if item.len() > self.max_message_size {
                return Err(ProtocolError::PayloadTooLarge {
                    size: item.len(),
                    max: self.max_message_size,
                }
                .into());
            }

            dst.reserve(4 + item.len());
            dst.put_u32(item.len() as u32);
            dst.put_slice(&item);

            Ok(())
        }
    }

    #[test]
    fn test_packet_codec() {
        let mut codec = PacketCodec::new();
        let session_id = SessionId::generate();

        let packet = Packet::data(SequenceNumber(1), session_id, 0, b"hello".to_vec()).unwrap();

        let mut buf = BytesMut::new();
        codec.encode(packet.clone(), &mut buf).unwrap();

        let decoded = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(decoded.header.sequence.0, packet.header.sequence.0);
        assert_eq!(decoded.payload, packet.payload);
    }

    #[test]
    fn test_length_prefixed_codec() {
        let mut codec = LengthPrefixedCodec::new(1024);

        let data = b"test message".to_vec();
        let mut buf = BytesMut::new();

        codec.encode(data.clone(), &mut buf).unwrap();
        let decoded = codec.decode(&mut buf).unwrap().unwrap();

        assert_eq!(decoded, data);
    }
}
