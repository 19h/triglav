//! Boundary and edge case tests.
//!
//! Tests for boundary conditions including:
//! - Empty payloads (0-byte)
//! - Maximum payload size
//! - Malformed packets
//! - Invalid checksums
//! - Session boundaries
//! - Flow limits

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use triglav::crypto::{KeyPair, NoiseSession};
use triglav::protocol::{Packet, PacketFlags, PacketType, HEADER_SIZE, MAX_PAYLOAD_SIZE};
use triglav::types::{SequenceNumber, SessionId, UplinkId};

// ============================================================================
// Empty Payload Tests
// ============================================================================

#[test]
fn test_empty_payload_packet() {
    // Create packet with empty payload
    let session_id = SessionId::generate();
    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        session_id,
        1,
        vec![], // Empty payload
    );

    assert!(packet.is_ok(), "Empty payload should be valid");
    let packet = packet.unwrap();
    assert_eq!(packet.payload.len(), 0);
    assert_eq!(packet.header.payload_len, 0);

    // Encode and decode
    let encoded = packet.encode().unwrap();
    assert_eq!(encoded.len(), HEADER_SIZE); // Just header, no payload

    let decoded = Packet::decode(&encoded).unwrap();
    assert!(decoded.payload.is_empty());
}

#[test]
fn test_empty_encryption() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let mut client = NoiseSession::new_initiator(&client_kp.secret, &server_kp.public).unwrap();
    let mut server = NoiseSession::new_responder(&server_kp.secret).unwrap();

    // Complete handshake
    let msg1 = client.write_handshake(&[]).unwrap();
    let _ = server.read_handshake(&msg1).unwrap();
    let msg2 = server.write_handshake(&[]).unwrap();
    let _ = client.read_handshake(&msg2).unwrap();

    // Encrypt empty data
    let ciphertext = client.encrypt(&[]).unwrap();
    assert!(
        !ciphertext.is_empty(),
        "Ciphertext should have auth tag even for empty plaintext"
    );
    assert_eq!(
        ciphertext.len(),
        16,
        "Empty plaintext should only have auth tag"
    );

    // Decrypt back to empty
    let decrypted = server.decrypt(&ciphertext).unwrap();
    assert!(decrypted.is_empty(), "Decrypted should be empty");
}

// ============================================================================
// Maximum Payload Tests
// ============================================================================

#[test]
fn test_max_payload_packet() {
    let session_id = SessionId::generate();
    let max_payload = vec![0xAB; MAX_PAYLOAD_SIZE];

    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        session_id,
        1,
        max_payload.clone(),
    );

    assert!(packet.is_ok(), "Max payload should be valid");
    let packet = packet.unwrap();
    assert_eq!(packet.payload.len(), MAX_PAYLOAD_SIZE);

    // Encode and decode
    let encoded = packet.encode().unwrap();
    assert_eq!(encoded.len(), HEADER_SIZE + MAX_PAYLOAD_SIZE);

    let decoded = Packet::decode(&encoded).unwrap();
    assert_eq!(decoded.payload.len(), MAX_PAYLOAD_SIZE);
    assert_eq!(decoded.payload, max_payload);
}

#[test]
fn test_oversized_payload_rejected() {
    let session_id = SessionId::generate();
    let oversized = vec![0xFF; MAX_PAYLOAD_SIZE + 1];

    let result = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        session_id,
        1,
        oversized,
    );

    assert!(result.is_err(), "Oversized payload should be rejected");
}

#[test]
fn test_max_encrypted_payload() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let mut client = NoiseSession::new_initiator(&client_kp.secret, &server_kp.public).unwrap();
    let mut server = NoiseSession::new_responder(&server_kp.secret).unwrap();

    // Complete handshake
    let msg1 = client.write_handshake(&[]).unwrap();
    let _ = server.read_handshake(&msg1).unwrap();
    let msg2 = server.write_handshake(&[]).unwrap();
    let _ = client.read_handshake(&msg2).unwrap();

    // Encrypt large data (but not larger than Noise max)
    // Note: Noise has its own max message size
    let large_data = vec![0xCD; 8192]; // 8KB
    let ciphertext = client.encrypt(&large_data).unwrap();

    // Decrypt
    let decrypted = server.decrypt(&ciphertext).unwrap();
    assert_eq!(decrypted, large_data);
}

// ============================================================================
// Malformed Packet Tests
// ============================================================================

#[test]
fn test_truncated_header() {
    // Header is 60 bytes, try decoding less
    let short = vec![0u8; HEADER_SIZE - 1];
    let result = Packet::decode(&short);
    assert!(result.is_err(), "Should reject truncated header");
}

#[test]
fn test_empty_packet() {
    let result = Packet::decode(&[]);
    assert!(result.is_err(), "Should reject empty packet");
}

#[test]
fn test_single_byte_packet() {
    let result = Packet::decode(&[0x01]);
    assert!(result.is_err(), "Should reject single byte packet");
}

#[test]
fn test_header_only_with_nonzero_payload_len() {
    // Create valid header claiming payload length > 0
    let session_id = SessionId::generate();
    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        session_id,
        1,
        vec![1, 2, 3, 4, 5], // 5 bytes payload
    )
    .unwrap();

    let mut encoded = packet.encode().unwrap();

    // Truncate to just header (remove payload)
    encoded.truncate(HEADER_SIZE);

    // Decode should fail because payload_len doesn't match
    let result = Packet::decode(&encoded);
    assert!(result.is_err(), "Should reject when payload missing");
}

#[test]
fn test_invalid_packet_type() {
    let session_id = SessionId::generate();
    let packet = Packet::new(PacketType::Data, SequenceNumber(1), session_id, 1, vec![]).unwrap();

    let mut encoded = packet.encode().unwrap();

    // Set invalid packet type (byte 1)
    encoded[1] = 0xFF; // Invalid type

    // Recalculate would normally be needed, but we're testing rejection
    // The checksum will fail first, or the type will be rejected
    let result = Packet::decode(&encoded);
    assert!(result.is_err(), "Should reject invalid packet type");
}

#[test]
fn test_corrupted_checksum() {
    let session_id = SessionId::generate();
    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        session_id,
        1,
        b"test payload".to_vec(),
    )
    .unwrap();

    let mut encoded = packet.encode().unwrap();

    // Corrupt the checksum (last 4 bytes of header at offset 56-60)
    encoded[56] ^= 0xFF;
    encoded[57] ^= 0xFF;

    let result = Packet::decode(&encoded);
    assert!(result.is_err(), "Should detect checksum corruption");
}

#[test]
fn test_payload_corruption_detected() {
    let session_id = SessionId::generate();
    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        session_id,
        1,
        b"original payload".to_vec(),
    )
    .unwrap();

    let mut encoded = packet.encode().unwrap();

    // Corrupt payload (but not header checksum - only covers header)
    let payload_start = HEADER_SIZE;
    encoded[payload_start] ^= 0xFF;

    // Packet will decode (header checksum doesn't cover payload)
    // This is intentional - payload integrity is handled by encryption layer
    let decoded = Packet::decode(&encoded);
    assert!(
        decoded.is_ok(),
        "Payload corruption not detected by header checksum (expected)"
    );

    // The payload will be different
    let decoded = decoded.unwrap();
    assert_ne!(decoded.payload, b"original payload".to_vec());
}

// ============================================================================
// Sequence Number Boundary Tests
// ============================================================================

#[test]
fn test_sequence_number_zero() {
    let session_id = SessionId::generate();
    let packet = Packet::new(PacketType::Data, SequenceNumber(0), session_id, 1, vec![]);

    assert!(packet.is_ok(), "Sequence 0 should be valid");

    let encoded = packet.unwrap().encode().unwrap();
    let decoded = Packet::decode(&encoded).unwrap();
    assert_eq!(decoded.header.sequence.0, 0);
}

#[test]
fn test_sequence_number_max() {
    let session_id = SessionId::generate();
    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(u64::MAX),
        session_id,
        1,
        vec![],
    );

    assert!(packet.is_ok(), "Max sequence should be valid");

    let encoded = packet.unwrap().encode().unwrap();
    let decoded = Packet::decode(&encoded).unwrap();
    assert_eq!(decoded.header.sequence.0, u64::MAX);
}

#[test]
fn test_sequence_number_wraparound() {
    let seq = SequenceNumber(u64::MAX);
    let next = seq.next();
    assert_eq!(next.0, 0, "Sequence should wrap around");
}

// ============================================================================
// Session ID Tests
// ============================================================================

#[test]
fn test_zero_session_id() {
    let zero_session = SessionId::new([0u8; 32]);
    let packet = Packet::new(PacketType::Data, SequenceNumber(1), zero_session, 1, vec![]);

    assert!(
        packet.is_ok(),
        "Zero session ID should be valid (application decides semantics)"
    );
}

#[test]
fn test_max_session_id() {
    let max_session = SessionId::new([0xFF; 32]);
    let packet = Packet::new(PacketType::Data, SequenceNumber(1), max_session, 1, vec![]);

    assert!(packet.is_ok(), "Max session ID should be valid");
}

#[test]
fn test_session_id_uniqueness() {
    let s1 = SessionId::generate();
    let s2 = SessionId::generate();

    // With 256 bits of randomness, collision is astronomically unlikely
    assert_ne!(
        s1.as_bytes(),
        s2.as_bytes(),
        "Random session IDs should be unique"
    );
}

// ============================================================================
// Uplink ID Tests
// ============================================================================

#[test]
fn test_uplink_id_zero() {
    let session_id = SessionId::generate();
    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        session_id,
        0, // Uplink ID 0
        vec![],
    );

    assert!(packet.is_ok(), "Uplink ID 0 should be valid");
}

#[test]
fn test_uplink_id_max() {
    let session_id = SessionId::generate();
    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        session_id,
        u16::MAX, // Max uplink ID
        vec![],
    );

    assert!(packet.is_ok(), "Max uplink ID should be valid");

    let encoded = packet.unwrap().encode().unwrap();
    let decoded = Packet::decode(&encoded).unwrap();
    assert_eq!(decoded.header.uplink_id, u16::MAX);
}

// ============================================================================
// Timestamp Tests
// ============================================================================

#[test]
fn test_timestamp_in_packet() {
    let session_id = SessionId::generate();
    let packet = Packet::new(PacketType::Data, SequenceNumber(1), session_id, 1, vec![]).unwrap();

    // Timestamp should be set to roughly current time
    assert!(packet.header.timestamp > 0, "Timestamp should be set");

    // Verify one-way delay calculation
    let delay = packet.header.one_way_delay();
    // Should be very small (just created)
    if let Some(d) = delay {
        assert!(
            d < Duration::from_secs(1),
            "One-way delay should be small for just-created packet"
        );
    }
}

// ============================================================================
// Packet Flag Tests
// ============================================================================

#[test]
fn test_all_flags_combined() {
    let mut flags = PacketFlags::new(0);

    flags.set(PacketFlags::NEED_ACK);
    flags.set(PacketFlags::RETRANSMIT);
    flags.set(PacketFlags::FRAGMENT);
    flags.set(PacketFlags::LAST_FRAGMENT);
    flags.set(PacketFlags::ENCRYPTED);
    flags.set(PacketFlags::COMPRESSED);
    flags.set(PacketFlags::PRIORITY);
    flags.set(PacketFlags::PROBE);

    assert!(flags.has(PacketFlags::NEED_ACK));
    assert!(flags.has(PacketFlags::RETRANSMIT));
    assert!(flags.has(PacketFlags::FRAGMENT));
    assert!(flags.has(PacketFlags::LAST_FRAGMENT));
    assert!(flags.has(PacketFlags::ENCRYPTED));
    assert!(flags.has(PacketFlags::COMPRESSED));
    assert!(flags.has(PacketFlags::PRIORITY));
    assert!(flags.has(PacketFlags::PROBE));
}

#[test]
fn test_flag_clear() {
    let mut flags = PacketFlags::new(0);

    flags.set(PacketFlags::ENCRYPTED);
    assert!(flags.has(PacketFlags::ENCRYPTED));

    flags.clear(PacketFlags::ENCRYPTED);
    assert!(!flags.has(PacketFlags::ENCRYPTED));
}

#[test]
fn test_flags_preserved_in_packet() {
    let session_id = SessionId::generate();
    let mut packet =
        Packet::new(PacketType::Data, SequenceNumber(1), session_id, 1, vec![]).unwrap();

    packet.set_flag(PacketFlags::ENCRYPTED);
    packet.set_flag(PacketFlags::PRIORITY);

    let encoded = packet.encode().unwrap();
    let decoded = Packet::decode(&encoded).unwrap();

    assert!(decoded.header.flags.has(PacketFlags::ENCRYPTED));
    assert!(decoded.header.flags.has(PacketFlags::PRIORITY));
}

// ============================================================================
// Packet Type Specific Tests
// ============================================================================

#[test]
fn test_all_packet_types() {
    let session_id = SessionId::generate();

    let types = vec![
        PacketType::Data,
        PacketType::Control,
        PacketType::Ack,
        PacketType::Nack,
        PacketType::Ping,
        PacketType::Pong,
        PacketType::Handshake,
        PacketType::Close,
        PacketType::Error,
    ];

    for ptype in types {
        let packet = Packet::new(ptype, SequenceNumber(1), session_id, 1, vec![]).unwrap();

        let encoded = packet.encode().unwrap();
        let decoded = Packet::decode(&encoded).unwrap();

        assert_eq!(
            decoded.header.packet_type, ptype,
            "Packet type should be preserved"
        );
    }
}

#[test]
fn test_ping_packet() {
    let session_id = SessionId::generate();
    let packet = Packet::ping(SequenceNumber(1), session_id, 1).unwrap();

    assert_eq!(packet.header.packet_type, PacketType::Ping);
    assert!(packet.payload.is_empty());
}

#[test]
fn test_pong_packet() {
    let session_id = SessionId::generate();
    let timestamp = 1234567890u64;

    let packet = Packet::pong(SequenceNumber(1), session_id, 1, timestamp).unwrap();

    assert_eq!(packet.header.packet_type, PacketType::Pong);
    assert_eq!(packet.payload.len(), 8); // u64 timestamp

    // Verify timestamp is encoded correctly
    let stored_ts = u64::from_be_bytes(packet.payload[..8].try_into().unwrap());
    assert_eq!(stored_ts, timestamp);
}

#[test]
fn test_ack_packet() {
    let session_id = SessionId::generate();
    let acked = vec![1u64, 2, 3, 5, 8, 13];

    let packet = Packet::ack(SequenceNumber(1), session_id, 1, &acked).unwrap();

    assert_eq!(packet.header.packet_type, PacketType::Ack);
    assert!(!packet.payload.is_empty());

    // Deserialize and verify
    let decoded_acked: Vec<u64> = bincode::deserialize(&packet.payload).unwrap();
    assert_eq!(decoded_acked, acked);
}

// ============================================================================
// Binary Data Tests
// ============================================================================

#[test]
fn test_binary_payload() {
    let session_id = SessionId::generate();

    // Payload with all possible byte values
    let payload: Vec<u8> = (0..=255u8).collect();

    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        session_id,
        1,
        payload.clone(),
    )
    .unwrap();

    let encoded = packet.encode().unwrap();
    let decoded = Packet::decode(&encoded).unwrap();

    assert_eq!(
        decoded.payload, payload,
        "Binary payload should be preserved exactly"
    );
}

#[test]
fn test_null_bytes_in_payload() {
    let session_id = SessionId::generate();

    // Payload with embedded nulls
    let payload = vec![0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x03];

    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        session_id,
        1,
        payload.clone(),
    )
    .unwrap();

    let encoded = packet.encode().unwrap();
    let decoded = Packet::decode(&encoded).unwrap();

    assert_eq!(decoded.payload, payload, "Null bytes should be preserved");
}

// ============================================================================
// Protocol Version Tests
// ============================================================================

#[test]
fn test_invalid_protocol_version() {
    let session_id = SessionId::generate();
    let packet = Packet::new(PacketType::Data, SequenceNumber(1), session_id, 1, vec![]).unwrap();

    let mut encoded = packet.encode().unwrap();

    // Change version byte (first byte)
    encoded[0] = 0xFF;

    let result = Packet::decode(&encoded);
    assert!(result.is_err(), "Should reject invalid protocol version");
}

// ============================================================================
// Handshake Payload Tests
// ============================================================================

#[test]
fn test_handshake_with_payload() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let mut client = NoiseSession::new_initiator(&client_kp.secret, &server_kp.public).unwrap();

    // Handshake can carry optional payload
    let custom_payload = b"client hello metadata";
    let msg = client.write_handshake(custom_payload);

    assert!(msg.is_ok(), "Handshake with payload should work");
    let msg = msg.unwrap();

    // Responder should receive the payload
    let mut server = NoiseSession::new_responder(&server_kp.secret).unwrap();
    let received_payload = server.read_handshake(&msg).unwrap();

    assert_eq!(
        received_payload, custom_payload,
        "Handshake payload should be delivered"
    );
}

#[test]
fn test_handshake_empty_payload() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let mut client = NoiseSession::new_initiator(&client_kp.secret, &server_kp.public).unwrap();
    let msg = client.write_handshake(&[]).unwrap();

    let mut server = NoiseSession::new_responder(&server_kp.secret).unwrap();
    let received = server.read_handshake(&msg).unwrap();

    assert!(received.is_empty(), "Empty handshake payload should work");
}
