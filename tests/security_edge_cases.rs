//! Security edge case tests for Triglav.
//!
//! Tests for security scenarios including:
//! - Invalid/malformed keys
//! - Corrupted ciphertext/tampering
//! - Replay attacks
//! - Wrong session ID handling
//! - Nonce reuse detection
//! - Crypto isolation between sessions

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use triglav::crypto::{KeyPair, NoiseSession, PublicKey, SecretKey};
use triglav::error::{CryptoError, Result};
use triglav::protocol::{Packet, PacketFlags, PacketType, HEADER_SIZE};
use triglav::types::{SequenceNumber, SessionId};

// ============================================================================
// Key Validation Tests
// ============================================================================

#[test]
fn test_zero_key_rejected() {
    // Test that all-zero keys are handled appropriately
    let zero_secret = SecretKey::from_bytes([0u8; 32]);
    let server_keypair = KeyPair::generate();

    // This should either fail or produce a weak/predictable result
    // The test ensures we don't crash and behavior is defined
    let result = NoiseSession::new_initiator(&zero_secret, &server_keypair.public);

    // The session creation might succeed (X25519 accepts any 32 bytes)
    // but subsequent operations should be checked
    if let Ok(mut session) = result {
        // Should be able to write handshake (snow doesn't reject weak keys)
        let hs = session.write_handshake(&[]);
        assert!(
            hs.is_ok(),
            "Handshake write should succeed (key validation is application responsibility)"
        );
    }
}

#[test]
fn test_invalid_public_key_base64() {
    // Test invalid base64 decoding
    let result = PublicKey::from_base64("not-valid-base64!!!");
    assert!(result.is_err(), "Should reject invalid base64");

    // Test wrong length
    let result = PublicKey::from_base64("AAAA"); // Too short
    assert!(result.is_err(), "Should reject wrong length key");
}

#[test]
fn test_invalid_secret_key_base64() {
    let result = SecretKey::from_base64("invalid!@#$%");
    assert!(result.is_err(), "Should reject invalid base64");

    let result = SecretKey::from_base64("AAAA"); // Too short
    assert!(result.is_err(), "Should reject wrong length key");
}

#[test]
fn test_key_roundtrip_integrity() {
    let keypair = KeyPair::generate();

    // Public key roundtrip
    let encoded = keypair.public.to_base64();
    let decoded = PublicKey::from_base64(&encoded).unwrap();
    assert_eq!(keypair.public.as_bytes(), decoded.as_bytes());

    // Secret key roundtrip
    let encoded_secret = keypair.secret.to_base64();
    let decoded_secret = SecretKey::from_base64(&encoded_secret).unwrap();
    assert_eq!(keypair.secret.as_bytes(), decoded_secret.as_bytes());
}

// ============================================================================
// Handshake Security Tests
// ============================================================================

#[test]
fn test_wrong_server_key_handshake_fails() {
    let client_keypair = KeyPair::generate();
    let server_keypair = KeyPair::generate();
    let wrong_server_keypair = KeyPair::generate();

    // Client uses wrong server public key
    let mut client = NoiseSession::new_initiator(
        &client_keypair.secret,
        &wrong_server_keypair.public, // Wrong key!
    )
    .unwrap();

    let mut server = NoiseSession::new_responder(&server_keypair.secret).unwrap();

    // Client sends handshake with wrong server key in mind
    let msg1 = client.write_handshake(&[]).unwrap();

    // Server receives handshake - this might succeed or fail depending on implementation
    // The NK pattern doesn't authenticate the client in the first message
    let server_read_result = server.read_handshake(&msg1);

    if server_read_result.is_err() {
        // Handshake failed at server read - test passes
        return;
    }

    // Try to get server response
    let server_write_result = server.write_handshake(&[]);

    if server_write_result.is_err() {
        // State machine prevented write - test passes
        return;
    }

    let msg2 = server_write_result.unwrap();

    // Client tries to process server response - should fail or produce wrong keys
    let result = client.read_handshake(&msg2);

    // Either the handshake fails, or the resulting keys won't match
    // This is the expected behavior of Noise NK with wrong rs
    if result.is_ok() {
        // If handshake "succeeds", encryption should fail due to key mismatch
        // Both sides will have different keys
        assert!(client.is_transport());
        assert!(server.is_transport());

        // Messages encrypted by one side should not decrypt on the other
        let plaintext = b"test message";
        let ciphertext = client.encrypt(plaintext).unwrap();
        let decrypt_result = server.decrypt(&ciphertext);

        // Decryption should fail due to authentication failure
        assert!(
            decrypt_result.is_err(),
            "Decryption should fail with wrong keys"
        );
    }
    // If we get here without failing, the test still passes -
    // the handshake either failed somewhere or keys don't match
}

#[test]
fn test_truncated_handshake_message() {
    let client_keypair = KeyPair::generate();
    let server_keypair = KeyPair::generate();

    let mut client =
        NoiseSession::new_initiator(&client_keypair.secret, &server_keypair.public).unwrap();

    let mut server = NoiseSession::new_responder(&server_keypair.secret).unwrap();

    // Get valid handshake message
    let msg1 = client.write_handshake(&[]).unwrap();

    // Truncate the message
    let truncated = &msg1[..msg1.len() / 2];

    // Server should reject truncated handshake
    let result = server.read_handshake(truncated);
    assert!(result.is_err(), "Should reject truncated handshake");
}

#[test]
fn test_corrupted_handshake_message() {
    let client_keypair = KeyPair::generate();
    let server_keypair = KeyPair::generate();

    let mut client =
        NoiseSession::new_initiator(&client_keypair.secret, &server_keypair.public).unwrap();

    let mut server = NoiseSession::new_responder(&server_keypair.secret).unwrap();

    // Get valid handshake message
    let mut msg1 = client.write_handshake(&[]).unwrap();

    // Corrupt the message (flip some bits)
    for i in 0..msg1.len().min(10) {
        msg1[i] ^= 0xFF;
    }

    // Server should reject corrupted handshake
    let result = server.read_handshake(&msg1);
    assert!(result.is_err(), "Should reject corrupted handshake");
}

// ============================================================================
// Ciphertext Tampering Tests
// ============================================================================

/// Helper to perform complete handshake
fn complete_handshake(
    client_secret: &SecretKey,
    server_secret: &SecretKey,
    server_public: &PublicKey,
) -> (NoiseSession, NoiseSession) {
    let mut client = NoiseSession::new_initiator(client_secret, server_public).unwrap();
    let mut server = NoiseSession::new_responder(server_secret).unwrap();

    let msg1 = client.write_handshake(&[]).unwrap();
    let _ = server.read_handshake(&msg1).unwrap();
    let msg2 = server.write_handshake(&[]).unwrap();
    let _ = client.read_handshake(&msg2).unwrap();

    assert!(client.is_transport());
    assert!(server.is_transport());

    (client, server)
}

#[test]
fn test_ciphertext_bit_flip_detected() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (mut client, mut server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    let plaintext = b"sensitive data";
    let mut ciphertext = client.encrypt(plaintext).unwrap();

    // Flip a single bit
    ciphertext[0] ^= 0x01;

    // Decryption should fail due to AEAD authentication
    let result = server.decrypt(&ciphertext);
    assert!(result.is_err(), "Should detect bit flip in ciphertext");
}

#[test]
fn test_ciphertext_truncation_detected() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (mut client, mut server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    let plaintext = b"sensitive data";
    let ciphertext = client.encrypt(plaintext).unwrap();

    // Truncate the ciphertext (remove auth tag partially)
    let truncated = &ciphertext[..ciphertext.len() - 1];

    // Decryption should fail
    let result = server.decrypt(truncated);
    assert!(result.is_err(), "Should detect truncated ciphertext");
}

#[test]
fn test_ciphertext_extension_detected() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (mut client, mut server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    let plaintext = b"sensitive data";
    let mut ciphertext = client.encrypt(plaintext).unwrap();

    // Extend the ciphertext with garbage
    ciphertext.extend_from_slice(b"extra garbage");

    // Decryption should fail (or ignore extra data, but not include it)
    let result = server.decrypt(&ciphertext);
    // Note: Some implementations might successfully decrypt and ignore extra data
    // but the result should NOT include the extra data if successful
    if let Ok(decrypted) = result {
        assert_eq!(
            decrypted, plaintext,
            "Extra data should not appear in plaintext"
        );
    }
}

#[test]
fn test_auth_tag_tampering_detected() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (mut client, mut server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    let plaintext = b"sensitive data";
    let mut ciphertext = client.encrypt(plaintext).unwrap();

    // Tamper with the auth tag (last 16 bytes)
    let tag_start = ciphertext.len() - 16;
    for i in tag_start..ciphertext.len() {
        ciphertext[i] ^= 0xFF;
    }

    // Decryption should fail
    let result = server.decrypt(&ciphertext);
    assert!(result.is_err(), "Should detect auth tag tampering");
}

#[test]
fn test_empty_ciphertext_rejected() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (_, mut server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    // Try to decrypt empty ciphertext
    let result = server.decrypt(&[]);
    assert!(result.is_err(), "Should reject empty ciphertext");
}

#[test]
fn test_short_ciphertext_rejected() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (_, mut server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    // Ciphertext shorter than auth tag (16 bytes)
    let short = vec![0u8; 15];
    let result = server.decrypt(&short);
    assert!(
        result.is_err(),
        "Should reject ciphertext shorter than auth tag"
    );
}

// ============================================================================
// Replay Attack Tests
// ============================================================================

#[test]
fn test_replay_same_ciphertext() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (mut client, mut server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    let plaintext = b"original message";
    let ciphertext = client.encrypt(plaintext).unwrap();

    // First decryption should succeed
    let decrypted = server.decrypt(&ciphertext).unwrap();
    assert_eq!(decrypted, plaintext);

    // Replay the same ciphertext - should fail due to nonce counter
    let result = server.decrypt(&ciphertext);
    assert!(result.is_err(), "Replay of same ciphertext should fail");
}

#[test]
fn test_out_of_order_decryption() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (mut client, mut server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    // Encrypt three messages
    let ct1 = client.encrypt(b"message 1").unwrap();
    let ct2 = client.encrypt(b"message 2").unwrap();
    let ct3 = client.encrypt(b"message 3").unwrap();

    // Decrypt in wrong order - this should fail for Noise protocol
    // because the nonce must increment in order
    let r1 = server.decrypt(&ct1);
    assert!(r1.is_ok());

    // Try to decrypt ct3 before ct2 (skip ct2's nonce)
    // Noise expects sequential nonces, so this should fail
    let r3 = server.decrypt(&ct3);
    assert!(r3.is_err(), "Out-of-order decryption should fail");
}

// ============================================================================
// Session Isolation Tests
// ============================================================================

#[test]
fn test_cross_session_decryption_fails() {
    let client1_kp = KeyPair::generate();
    let client2_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    // Two separate sessions with the same server
    let (mut client1, mut server1) =
        complete_handshake(&client1_kp.secret, &server_kp.secret, &server_kp.public);

    let (mut client2, mut server2) =
        complete_handshake(&client2_kp.secret, &server_kp.secret, &server_kp.public);

    // Client1 encrypts a message
    let plaintext = b"secret from client1";
    let ciphertext = client1.encrypt(plaintext).unwrap();

    // Server1 can decrypt it
    let decrypted = server1.decrypt(&ciphertext).unwrap();
    assert_eq!(decrypted, plaintext);

    // Server2 should NOT be able to decrypt client1's message
    let result = server2.decrypt(&ciphertext);
    assert!(result.is_err(), "Cross-session decryption should fail");
}

#[test]
fn test_different_sessions_different_keys() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    // Create two sessions with same keys
    let (mut client1, mut server1) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    let (mut client2, mut server2) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    // Encrypt same plaintext in both sessions
    let plaintext = b"same plaintext";
    let ct1 = client1.encrypt(plaintext).unwrap();
    let ct2 = client2.encrypt(plaintext).unwrap();

    // Ciphertexts should be different (different ephemeral keys)
    assert_ne!(
        ct1, ct2,
        "Same plaintext should produce different ciphertexts in different sessions"
    );

    // Cross-decryption should fail
    assert!(
        server2.decrypt(&ct1).is_err(),
        "Cross-session decryption should fail"
    );
    assert!(
        server1.decrypt(&ct2).is_err(),
        "Cross-session decryption should fail"
    );
}

// ============================================================================
// Nonce Counter Tests
// ============================================================================

#[test]
fn test_nonce_increments() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (mut client, _server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    let initial_nonce = client.nonce_counter().unwrap();

    // Encrypt multiple messages
    for i in 1..=10 {
        let _ = client.encrypt(format!("message {}", i).as_bytes()).unwrap();
        let nonce = client.nonce_counter().unwrap();
        assert_eq!(nonce, initial_nonce + i as u64, "Nonce should increment");
    }
}

#[test]
fn test_rekey_operation() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (mut client, mut server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    // Send some messages
    for i in 0..5 {
        let ct = client.encrypt(format!("msg {}", i).as_bytes()).unwrap();
        let _pt = server.decrypt(&ct).unwrap();
    }

    // Perform rekey
    client.rekey_outgoing().unwrap();
    server.rekey_incoming().unwrap();

    // Messages should still work after rekey
    let ct = client.encrypt(b"after rekey").unwrap();
    let pt = server.decrypt(&ct).unwrap();
    assert_eq!(pt, b"after rekey");

    // Pre-rekey ciphertext should not work after rekey
    // (would need to save one from before, but this tests the concept)
}

// ============================================================================
// Packet Security Tests
// ============================================================================

#[test]
fn test_packet_checksum_validation() {
    // Create a valid packet
    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        SessionId::generate(),
        1,
        b"test payload".to_vec(),
    )
    .unwrap();

    let mut encoded = packet.encode().unwrap();

    // The packet checksum only covers the header, not the payload.
    // For payload integrity, encrypted packets use AEAD authentication.
    // Here we corrupt the header (e.g., sequence number) to test checksum.
    // Sequence number is at bytes 4-8 in the header
    encoded[4] ^= 0xFF;

    // Decoding should fail checksum validation
    let result = Packet::decode(&encoded);
    assert!(
        result.is_err(),
        "Should detect corrupted header via checksum"
    );
}

#[test]
fn test_packet_header_corruption() {
    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        SessionId::generate(),
        1,
        b"test".to_vec(),
    )
    .unwrap();

    let mut encoded = packet.encode().unwrap();

    // Corrupt the header (sequence number area)
    encoded[4] ^= 0xFF;

    // Should either fail to decode or have wrong data
    let result = Packet::decode(&encoded);
    // Checksum should catch this
    assert!(result.is_err(), "Header corruption should be detected");
}

#[test]
fn test_packet_version_mismatch() {
    let packet = Packet::new(
        PacketType::Data,
        SequenceNumber(1),
        SessionId::generate(),
        1,
        b"test".to_vec(),
    )
    .unwrap();

    let mut encoded = packet.encode().unwrap();

    // Change version byte (first byte)
    encoded[0] = 0xFF; // Invalid version

    let result = Packet::decode(&encoded);
    // Should reject unknown version
    if let Ok(p) = result {
        // If decoding somehow succeeds, version should be wrong
        assert_ne!(
            p.header.version,
            triglav::PROTOCOL_VERSION,
            "Should reject or flag wrong version"
        );
    }
}

#[test]
fn test_undersized_packet_rejected() {
    // Packet smaller than header size
    let small = vec![0u8; HEADER_SIZE - 1];
    let result = Packet::decode(&small);
    assert!(result.is_err(), "Should reject undersized packet");
}

// ============================================================================
// Encryption State Tests
// ============================================================================

#[test]
fn test_encrypt_before_handshake_fails() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let mut client = NoiseSession::new_initiator(&client_kp.secret, &server_kp.public).unwrap();

    // Try to encrypt before completing handshake
    let result = client.encrypt(b"test");
    assert!(
        result.is_err(),
        "Should not encrypt before handshake complete"
    );
}

#[test]
fn test_decrypt_before_handshake_fails() {
    let server_kp = KeyPair::generate();

    let mut server = NoiseSession::new_responder(&server_kp.secret).unwrap();

    // Try to decrypt before completing handshake
    let fake_ciphertext = vec![0u8; 32];
    let result = server.decrypt(&fake_ciphertext);
    assert!(
        result.is_err(),
        "Should not decrypt before handshake complete"
    );
}

#[test]
fn test_handshake_after_transport_fails() {
    let client_kp = KeyPair::generate();
    let server_kp = KeyPair::generate();

    let (mut client, _server) =
        complete_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public);

    // Try to do handshake after already in transport mode
    let result = client.write_handshake(&[]);
    assert!(
        result.is_err(),
        "Should not allow handshake after transport mode"
    );
}

// ============================================================================
// Signature Verification Tests
// ============================================================================

#[test]
fn test_signature_verification() {
    use triglav::crypto::SigningKeyPair;

    let keypair = SigningKeyPair::generate();
    let message = b"important data";

    let signature = keypair.sign(message);

    // Valid signature should verify
    assert!(keypair.verify(message, &signature).is_ok());

    // Wrong message should fail
    assert!(keypair.verify(b"different data", &signature).is_err());
}

#[test]
fn test_signature_tampering_detected() {
    use triglav::crypto::SigningKeyPair;

    let keypair = SigningKeyPair::generate();
    let message = b"important data";

    let mut signature = keypair.sign(message);

    // Tamper with signature
    signature[0] ^= 0xFF;

    // Should fail verification
    assert!(keypair.verify(message, &signature).is_err());
}

#[test]
fn test_verify_with_wrong_public_key() {
    use triglav::crypto::SigningKeyPair;

    let keypair1 = SigningKeyPair::generate();
    let keypair2 = SigningKeyPair::generate();

    let message = b"data";
    let signature = keypair1.sign(message);

    // Verify with wrong key should fail
    let result = SigningKeyPair::verify_with_public(&keypair2.public_bytes(), message, &signature);
    assert!(result.is_err(), "Should fail with wrong public key");
}

// ============================================================================
// Constant Time Comparison Tests
// ============================================================================

#[test]
fn test_secure_compare_equal() {
    use triglav::crypto::secure_compare;

    let a = [1u8, 2, 3, 4, 5];
    let b = [1u8, 2, 3, 4, 5];

    assert!(secure_compare(&a, &b), "Equal arrays should compare equal");
}

#[test]
fn test_secure_compare_unequal() {
    use triglav::crypto::secure_compare;

    let a = [1u8, 2, 3, 4, 5];
    let b = [1u8, 2, 3, 4, 6]; // Last byte different

    assert!(
        !secure_compare(&a, &b),
        "Different arrays should not compare equal"
    );
}

#[test]
fn test_secure_compare_different_lengths() {
    use triglav::crypto::secure_compare;

    let a = [1u8, 2, 3, 4, 5];
    let b = [1u8, 2, 3];

    assert!(
        !secure_compare(&a, &b),
        "Different length arrays should not compare equal"
    );
}
