//! Symmetric encryption using XChaCha20-Poly1305.

use chacha20poly1305::{
    aead::{Aead, KeyInit},
    XChaCha20Poly1305, XNonce,
};
use serde::{Deserialize, Serialize};

use crate::error::CryptoError;

/// Tag size for XChaCha20-Poly1305.
pub const TAG_SIZE: usize = 16;

/// Nonce size for XChaCha20-Poly1305.
pub const NONCE_SIZE: usize = 24;

/// Key size for XChaCha20-Poly1305.
pub const KEY_SIZE: usize = 32;

/// Encrypted packet structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedPacket {
    /// Random nonce (24 bytes for XChaCha20).
    pub nonce: [u8; NONCE_SIZE],
    /// Ciphertext with authentication tag.
    pub ciphertext: Vec<u8>,
}

impl EncryptedPacket {
    /// Total overhead (nonce + tag).
    pub const OVERHEAD: usize = NONCE_SIZE + TAG_SIZE;

    /// Serialize to bytes (nonce || ciphertext).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(NONCE_SIZE + self.ciphertext.len());
        buf.extend_from_slice(&self.nonce);
        buf.extend_from_slice(&self.ciphertext);
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, CryptoError> {
        if data.len() < NONCE_SIZE + TAG_SIZE {
            return Err(CryptoError::InvalidCiphertextLength);
        }

        let mut nonce = [0u8; NONCE_SIZE];
        nonce.copy_from_slice(&data[..NONCE_SIZE]);
        let ciphertext = data[NONCE_SIZE..].to_vec();

        Ok(Self { nonce, ciphertext })
    }
}

/// Encrypt data with XChaCha20-Poly1305.
///
/// # Arguments
/// * `key` - 256-bit key
/// * `plaintext` - Data to encrypt
/// * `aad` - Additional authenticated data (optional)
///
/// # Returns
/// Encrypted packet containing nonce and ciphertext with auth tag.
pub fn encrypt(key: &[u8; KEY_SIZE], plaintext: &[u8], aad: Option<&[u8]>) -> Result<EncryptedPacket, CryptoError> {
    let cipher = XChaCha20Poly1305::new_from_slice(key)
        .map_err(|e| CryptoError::EncryptionFailed(format!("cipher init: {e}")))?;

    // Generate random nonce
    let nonce: [u8; NONCE_SIZE] = crate::crypto::random_bytes();
    let xnonce = XNonce::from_slice(&nonce);

    let ciphertext = if let Some(aad_data) = aad {
        cipher
            .encrypt(xnonce, chacha20poly1305::aead::Payload {
                msg: plaintext,
                aad: aad_data,
            })
            .map_err(|e| CryptoError::EncryptionFailed(format!("encrypt: {e}")))?
    } else {
        cipher
            .encrypt(xnonce, plaintext)
            .map_err(|e| CryptoError::EncryptionFailed(format!("encrypt: {e}")))?
    };

    Ok(EncryptedPacket { nonce, ciphertext })
}

/// Decrypt data with XChaCha20-Poly1305.
///
/// # Arguments
/// * `key` - 256-bit key
/// * `packet` - Encrypted packet
/// * `aad` - Additional authenticated data (must match encryption)
///
/// # Returns
/// Decrypted plaintext.
pub fn decrypt(key: &[u8; KEY_SIZE], packet: &EncryptedPacket, aad: Option<&[u8]>) -> Result<Vec<u8>, CryptoError> {
    let cipher = XChaCha20Poly1305::new_from_slice(key)
        .map_err(|e| CryptoError::DecryptionFailed(format!("cipher init: {e}")))?;

    let xnonce = XNonce::from_slice(&packet.nonce);

    let plaintext = if let Some(aad_data) = aad {
        cipher
            .decrypt(xnonce, chacha20poly1305::aead::Payload {
                msg: packet.ciphertext.as_slice(),
                aad: aad_data,
            })
            .map_err(|_| CryptoError::DecryptionFailed("authentication failed".into()))?
    } else {
        cipher
            .decrypt(xnonce, packet.ciphertext.as_slice())
            .map_err(|_| CryptoError::DecryptionFailed("authentication failed".into()))?
    };

    Ok(plaintext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt() {
        let key: [u8; 32] = crate::crypto::random_bytes();
        let plaintext = b"hello world";

        let packet = encrypt(&key, plaintext, None).unwrap();
        let decrypted = decrypt(&key, &packet, None).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_encrypt_decrypt_with_aad() {
        let key: [u8; 32] = crate::crypto::random_bytes();
        let plaintext = b"secret message";
        let aad = b"associated data";

        let packet = encrypt(&key, plaintext, Some(aad)).unwrap();
        let decrypted = decrypt(&key, &packet, Some(aad)).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());

        // Wrong AAD should fail
        assert!(decrypt(&key, &packet, Some(b"wrong aad")).is_err());
        assert!(decrypt(&key, &packet, None).is_err());
    }

    #[test]
    fn test_wrong_key_fails() {
        let key1: [u8; 32] = crate::crypto::random_bytes();
        let key2: [u8; 32] = crate::crypto::random_bytes();
        let plaintext = b"hello world";

        let packet = encrypt(&key1, plaintext, None).unwrap();
        assert!(decrypt(&key2, &packet, None).is_err());
    }

    #[test]
    fn test_tampered_ciphertext_fails() {
        let key: [u8; 32] = crate::crypto::random_bytes();
        let plaintext = b"hello world";

        let mut packet = encrypt(&key, plaintext, None).unwrap();

        // Tamper with ciphertext
        if !packet.ciphertext.is_empty() {
            packet.ciphertext[0] ^= 0xff;
        }

        assert!(decrypt(&key, &packet, None).is_err());
    }

    #[test]
    fn test_packet_serialization() {
        let key: [u8; 32] = crate::crypto::random_bytes();
        let plaintext = b"hello world";

        let packet = encrypt(&key, plaintext, None).unwrap();
        let bytes = packet.to_bytes();
        let packet2 = EncryptedPacket::from_bytes(&bytes).unwrap();

        assert_eq!(packet.nonce, packet2.nonce);
        assert_eq!(packet.ciphertext, packet2.ciphertext);

        let decrypted = decrypt(&key, &packet2, None).unwrap();
        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }
}
