//! Cryptographic primitives for Triglav.
//!
//! This module provides:
//! - Key generation and management (X25519, Ed25519)
//! - Noise NK protocol implementation
//! - Symmetric encryption (XChaCha20-Poly1305)
//! - Key derivation (HKDF-SHA256)
//! - Secure hashing (BLAKE3)

mod kdf;
mod keys;
mod noise;
mod symmetric;

pub use kdf::KeySchedule;
pub use keys::{KeyPair, PublicKey, SecretKey, SigningKeyPair};
pub use noise::{HandshakeRole, NoiseSession, NoiseState};
pub use symmetric::{decrypt, encrypt, EncryptedPacket};

/// Hash data using BLAKE3.
pub fn hash(data: &[u8]) -> [u8; 32] {
    blake3::hash(data).into()
}

/// Hash multiple pieces of data using BLAKE3.
pub fn hash_many(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// Generate cryptographically secure random bytes.
pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0u8; N];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut bytes);
    bytes
}

/// Constant-time comparison of byte slices.
pub fn secure_compare(a: &[u8], b: &[u8]) -> bool {
    constant_time_eq::constant_time_eq(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash() {
        let data = b"hello world";
        let h1 = hash(data);
        let h2 = hash(data);
        assert_eq!(h1, h2);

        let h3 = hash(b"different data");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_random_bytes() {
        let r1: [u8; 32] = random_bytes();
        let r2: [u8; 32] = random_bytes();
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_secure_compare() {
        let a = [1u8, 2, 3, 4];
        let b = [1u8, 2, 3, 4];
        let c = [1u8, 2, 3, 5];

        assert!(secure_compare(&a, &b));
        assert!(!secure_compare(&a, &c));
    }
}
