//! Key management for Triglav.

use std::fmt;

use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use x25519_dalek::{PublicKey as X25519Public, StaticSecret};
use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::CryptoError;

/// X25519 public key for key exchange.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PublicKey(pub [u8; 32]);

impl PublicKey {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the raw bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to X25519 public key.
    pub fn to_x25519(&self) -> X25519Public {
        X25519Public::from(self.0)
    }

    /// Encode as base64.
    pub fn to_base64(&self) -> String {
        use base64::Engine;
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(self.0)
    }

    /// Decode from base64.
    pub fn from_base64(s: &str) -> Result<Self, CryptoError> {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(s)
            .map_err(|e| CryptoError::KeyDerivationFailed(format!("invalid base64: {e}")))?;
        if bytes.len() != 32 {
            return Err(CryptoError::KeyDerivationFailed("invalid key length".into()));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Self(arr))
    }
}

impl fmt::Debug for PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PublicKey({})", &self.to_base64()[..8])
    }
}

impl fmt::Display for PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_base64())
    }
}

impl Serialize for PublicKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_base64())
    }
}

impl<'de> Deserialize<'de> for PublicKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_base64(&s).map_err(serde::de::Error::custom)
    }
}

/// X25519 secret key for key exchange.
#[derive(Clone, ZeroizeOnDrop)]
pub struct SecretKey {
    inner: StaticSecret,
}

impl SecretKey {
    /// Generate a new random secret key.
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut OsRng, &mut bytes);
        let inner = StaticSecret::from(bytes);
        // Zeroize the temporary bytes
        bytes.zeroize();
        Self { inner }
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        let inner = StaticSecret::from(bytes);
        Self { inner }
    }

    /// Get the raw bytes (use with caution).
    pub fn as_bytes(&self) -> [u8; 32] {
        self.inner.to_bytes()
    }

    /// Get the corresponding public key.
    pub fn public_key(&self) -> PublicKey {
        let x25519_pub = X25519Public::from(&self.inner);
        PublicKey(*x25519_pub.as_bytes())
    }

    /// Perform Diffie-Hellman key exchange.
    pub fn diffie_hellman(&self, their_public: &PublicKey) -> [u8; 32] {
        let shared = self.inner.diffie_hellman(&their_public.to_x25519());
        *shared.as_bytes()
    }

    /// Encode as base64.
    pub fn to_base64(&self) -> String {
        use base64::Engine;
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(self.inner.to_bytes())
    }

    /// Decode from base64.
    pub fn from_base64(s: &str) -> Result<Self, CryptoError> {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(s)
            .map_err(|e| CryptoError::KeyDerivationFailed(format!("invalid base64: {e}")))?;
        if bytes.len() != 32 {
            return Err(CryptoError::KeyDerivationFailed("invalid key length".into()));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Self::from_bytes(arr))
    }
}

impl fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretKey([REDACTED])")
    }
}

/// Key pair for key exchange (X25519).
#[derive(Clone)]
pub struct KeyPair {
    pub secret: SecretKey,
    pub public: PublicKey,
}

impl KeyPair {
    /// Generate a new random key pair.
    pub fn generate() -> Self {
        let secret = SecretKey::generate();
        let public = secret.public_key();
        Self { secret, public }
    }

    /// Create from a secret key.
    pub fn from_secret(secret: SecretKey) -> Self {
        let public = secret.public_key();
        Self { secret, public }
    }

    /// Create from raw secret bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self::from_secret(SecretKey::from_bytes(bytes))
    }
}

impl fmt::Debug for KeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyPair")
            .field("public", &self.public)
            .finish_non_exhaustive()
    }
}

/// Key pair for digital signatures (Ed25519).
#[derive(Clone)]
pub struct SigningKeyPair {
    signing: SigningKey,
    verifying: VerifyingKey,
}

impl SigningKeyPair {
    /// Generate a new random signing key pair.
    pub fn generate() -> Self {
        let signing = SigningKey::generate(&mut OsRng);
        let verifying = signing.verifying_key();
        Self { signing, verifying }
    }

    /// Create from raw secret bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Result<Self, CryptoError> {
        let signing = SigningKey::from_bytes(&bytes);
        let verifying = signing.verifying_key();
        Ok(Self { signing, verifying })
    }

    /// Get the signing key bytes.
    pub fn secret_bytes(&self) -> [u8; 32] {
        self.signing.to_bytes()
    }

    /// Get the verifying (public) key bytes.
    pub fn public_bytes(&self) -> [u8; 32] {
        self.verifying.to_bytes()
    }

    /// Sign a message.
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        let sig: Signature = self.signing.sign(message);
        sig.to_bytes()
    }

    /// Verify a signature.
    pub fn verify(&self, message: &[u8], signature: &[u8; 64]) -> Result<(), CryptoError> {
        let sig = Signature::from_bytes(signature);
        self.verifying
            .verify(message, &sig)
            .map_err(|_| CryptoError::SignatureVerificationFailed)
    }

    /// Verify using only the public key bytes.
    pub fn verify_with_public(
        public_bytes: &[u8; 32],
        message: &[u8],
        signature: &[u8; 64],
    ) -> Result<(), CryptoError> {
        let verifying = VerifyingKey::from_bytes(public_bytes)
            .map_err(|e| CryptoError::KeyDerivationFailed(format!("invalid public key: {e}")))?;
        let sig = Signature::from_bytes(signature);
        verifying
            .verify(message, &sig)
            .map_err(|_| CryptoError::SignatureVerificationFailed)
    }
}

impl fmt::Debug for SigningKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SigningKeyPair")
            .field("public", &hex::encode(&self.public_bytes()[..8]))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let kp1 = KeyPair::generate();
        let kp2 = KeyPair::generate();
        assert_ne!(kp1.public.0, kp2.public.0);
    }

    #[test]
    fn test_diffie_hellman() {
        let alice = KeyPair::generate();
        let bob = KeyPair::generate();

        let alice_shared = alice.secret.diffie_hellman(&bob.public);
        let bob_shared = bob.secret.diffie_hellman(&alice.public);

        assert_eq!(alice_shared, bob_shared);
    }

    #[test]
    fn test_base64_roundtrip() {
        let kp = KeyPair::generate();
        let encoded = kp.public.to_base64();
        let decoded = PublicKey::from_base64(&encoded).unwrap();
        assert_eq!(kp.public.0, decoded.0);
    }

    #[test]
    fn test_signing() {
        let kp = SigningKeyPair::generate();
        let message = b"hello world";
        let signature = kp.sign(message);

        kp.verify(message, &signature).unwrap();

        // Verify with wrong message should fail
        assert!(kp.verify(b"wrong message", &signature).is_err());
    }
}
