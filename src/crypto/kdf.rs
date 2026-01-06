//! Key derivation functions for Triglav.
//!
//! Uses HKDF-SHA256 for all key material derivation.

use hkdf::Hkdf;
use sha2::Sha256;

/// Key schedule for deriving all cryptographic keys.
pub struct KeySchedule;

impl KeySchedule {
    /// Domain separation prefix for all Triglav keys.
    const DOMAIN: &'static [u8] = b"triglav/v1/";

    /// Derive a key using HKDF-SHA256.
    ///
    /// # Arguments
    /// * `ikm` - Input key material
    /// * `salt` - Optional salt (can be empty)
    /// * `info` - Context/label for the key
    /// * `length` - Output key length (max 255 * 32 = 8160 bytes)
    pub fn derive(ikm: &[u8], salt: Option<&[u8]>, info: &[u8], length: usize) -> Vec<u8> {
        let hk = Hkdf::<Sha256>::new(salt, ikm);
        let mut okm = vec![0u8; length];
        hk.expand(info, &mut okm)
            .expect("HKDF output length should be valid");
        okm
    }

    /// Derive a 32-byte key.
    pub fn derive_key(ikm: &[u8], salt: Option<&[u8]>, info: &[u8]) -> [u8; 32] {
        let hk = Hkdf::<Sha256>::new(salt, ikm);
        let mut okm = [0u8; 32];
        hk.expand(info, &mut okm)
            .expect("32-byte HKDF output should be valid");
        okm
    }

    /// Derive a 24-byte nonce.
    pub fn derive_nonce(ikm: &[u8], salt: Option<&[u8]>, info: &[u8]) -> [u8; 24] {
        let hk = Hkdf::<Sha256>::new(salt, ikm);
        let mut okm = [0u8; 24];
        hk.expand(info, &mut okm)
            .expect("24-byte HKDF output should be valid");
        okm
    }

    /// Derive session encryption key from shared secret.
    pub fn session_key(shared_secret: &[u8; 32], session_id: &[u8; 32]) -> [u8; 32] {
        let mut info = Vec::with_capacity(Self::DOMAIN.len() + b"session-key".len());
        info.extend_from_slice(Self::DOMAIN);
        info.extend_from_slice(b"session-key");
        Self::derive_key(shared_secret, Some(session_id), &info)
    }

    /// Derive packet encryption key from session key.
    pub fn packet_key(session_key: &[u8; 32], direction: Direction) -> [u8; 32] {
        let mut info = Vec::with_capacity(Self::DOMAIN.len() + b"packet-key-".len() + 1);
        info.extend_from_slice(Self::DOMAIN);
        info.extend_from_slice(b"packet-key-");
        info.push(direction as u8);
        Self::derive_key(session_key, None, &info)
    }

    /// Derive MAC key for packet authentication.
    pub fn mac_key(session_key: &[u8; 32]) -> [u8; 32] {
        let mut info = Vec::with_capacity(Self::DOMAIN.len() + b"mac-key".len());
        info.extend_from_slice(Self::DOMAIN);
        info.extend_from_slice(b"mac-key");
        Self::derive_key(session_key, None, &info)
    }

    /// Derive handshake binding key.
    pub fn handshake_binding(
        initiator_public: &[u8; 32],
        responder_public: &[u8; 32],
        transcript_hash: &[u8; 32],
    ) -> [u8; 32] {
        let mut ikm = Vec::with_capacity(96);
        ikm.extend_from_slice(initiator_public);
        ikm.extend_from_slice(responder_public);
        ikm.extend_from_slice(transcript_hash);

        let mut info = Vec::with_capacity(Self::DOMAIN.len() + b"handshake-binding".len());
        info.extend_from_slice(Self::DOMAIN);
        info.extend_from_slice(b"handshake-binding");

        Self::derive_key(&ikm, None, &info)
    }

    /// Derive uplink-specific key for per-path encryption (optional layer).
    pub fn uplink_key(session_key: &[u8; 32], uplink_id: &str) -> [u8; 32] {
        let mut info =
            Vec::with_capacity(Self::DOMAIN.len() + b"uplink-key/".len() + uplink_id.len());
        info.extend_from_slice(Self::DOMAIN);
        info.extend_from_slice(b"uplink-key/");
        info.extend_from_slice(uplink_id.as_bytes());
        Self::derive_key(session_key, None, &info)
    }

    /// Derive rekey material for session key rotation.
    pub fn rekey(current_key: &[u8; 32], rekey_counter: u64) -> [u8; 32] {
        let mut info = Vec::with_capacity(Self::DOMAIN.len() + b"rekey".len());
        info.extend_from_slice(Self::DOMAIN);
        info.extend_from_slice(b"rekey");

        let salt = rekey_counter.to_le_bytes();
        Self::derive_key(current_key, Some(&salt), &info)
    }

    /// Derive auth token from server public key and secret.
    pub fn auth_token(server_pubkey: &[u8; 32], client_secret: &[u8; 32]) -> [u8; 32] {
        let mut ikm = Vec::with_capacity(64);
        ikm.extend_from_slice(server_pubkey);
        ikm.extend_from_slice(client_secret);

        let mut info = Vec::with_capacity(Self::DOMAIN.len() + b"auth-token".len());
        info.extend_from_slice(Self::DOMAIN);
        info.extend_from_slice(b"auth-token");

        Self::derive_key(&ikm, None, &info)
    }
}

/// Direction of data flow for key derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
    /// Client to server.
    ClientToServer = 0,
    /// Server to client.
    ServerToClient = 1,
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmac::{Hmac, Mac};
    type HmacSha256 = Hmac<sha2::Sha256>;

    /// HMAC-SHA256 for message authentication (test-only).
    fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
        let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
        mac.update(data);
        mac.finalize().into_bytes().into()
    }

    /// Verify HMAC-SHA256 in constant time (test-only).
    fn hmac_sha256_verify(key: &[u8], data: &[u8], expected: &[u8; 32]) -> bool {
        let computed = hmac_sha256(key, data);
        crate::crypto::secure_compare(&computed, expected)
    }

    #[test]
    fn test_derive_key() {
        let ikm = b"input key material";
        let salt = b"salt";
        let info = b"context";

        let key1 = KeySchedule::derive_key(ikm, Some(salt), info);
        let key2 = KeySchedule::derive_key(ikm, Some(salt), info);
        assert_eq!(key1, key2);

        // Different info should produce different key
        let key3 = KeySchedule::derive_key(ikm, Some(salt), b"different");
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_session_key() {
        let shared_secret: [u8; 32] = crate::crypto::random_bytes();
        let session_id: [u8; 32] = crate::crypto::random_bytes();

        let key1 = KeySchedule::session_key(&shared_secret, &session_id);
        let key2 = KeySchedule::session_key(&shared_secret, &session_id);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_packet_key_directions() {
        let session_key: [u8; 32] = crate::crypto::random_bytes();

        let c2s = KeySchedule::packet_key(&session_key, Direction::ClientToServer);
        let s2c = KeySchedule::packet_key(&session_key, Direction::ServerToClient);

        // Different directions should produce different keys
        assert_ne!(c2s, s2c);
    }

    #[test]
    fn test_rekey() {
        let key: [u8; 32] = crate::crypto::random_bytes();

        let rekey1 = KeySchedule::rekey(&key, 1);
        let rekey2 = KeySchedule::rekey(&key, 2);

        assert_ne!(key, rekey1);
        assert_ne!(rekey1, rekey2);
    }

    #[test]
    fn test_hmac() {
        let key = b"secret key";
        let data = b"data to authenticate";

        let mac = hmac_sha256(key, data);
        assert!(hmac_sha256_verify(key, data, &mac));

        // Wrong data should fail
        assert!(!hmac_sha256_verify(key, b"wrong data", &mac));
    }
}
