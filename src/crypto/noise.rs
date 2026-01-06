//! Noise NK protocol implementation for Triglav.
//!
//! Uses Noise_NK_25519_ChaChaPoly_BLAKE2s pattern:
//! - NK: No initiator static key, Known responder static key
//! - 25519: X25519 for key exchange
//! - ChaChaPoly: ChaCha20-Poly1305 for AEAD
//! - BLAKE2s: BLAKE3 (via custom resolver, maps BLAKE2s to BLAKE3)

use std::fmt;

use snow::params::NoiseParams;
use snow::resolvers::{CryptoResolver, DefaultResolver};
use snow::{Builder, HandshakeState, TransportState};

use crate::crypto::{PublicKey, SecretKey};
use crate::error::CryptoError;

/// Maximum Noise message size.
pub const MAX_NOISE_MSG_SIZE: usize = 65535;

/// Noise protocol pattern string.
/// Note: We specify BLAKE2s but use a custom resolver that maps it to BLAKE3.
const NOISE_PATTERN: &str = "Noise_NK_25519_ChaChaPoly_BLAKE2s";

/// Role in the Noise handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeRole {
    /// Initiator (client) - starts the handshake.
    Initiator,
    /// Responder (server) - responds to handshake.
    Responder,
}

/// Noise session state.
pub enum NoiseState {
    /// Handshake in progress.
    Handshake(Box<HandshakeState>),
    /// Transport mode (handshake complete).
    Transport(Box<TransportState>),
    /// Failed state.
    Failed,
}

impl fmt::Debug for NoiseState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handshake(_) => write!(f, "NoiseState::Handshake"),
            Self::Transport(_) => write!(f, "NoiseState::Transport"),
            Self::Failed => write!(f, "NoiseState::Failed"),
        }
    }
}

/// Custom crypto resolver that uses BLAKE3 instead of BLAKE2s.
struct TriglavResolver {
    default: DefaultResolver,
}

impl TriglavResolver {
    fn new() -> Self {
        Self {
            default: DefaultResolver,
        }
    }
}

impl CryptoResolver for TriglavResolver {
    fn resolve_rng(&self) -> Option<Box<dyn snow::types::Random>> {
        self.default.resolve_rng()
    }

    fn resolve_dh(&self, choice: &snow::params::DHChoice) -> Option<Box<dyn snow::types::Dh>> {
        self.default.resolve_dh(choice)
    }

    fn resolve_hash(
        &self,
        choice: &snow::params::HashChoice,
    ) -> Option<Box<dyn snow::types::Hash>> {
        // Map BLAKE2s to our BLAKE3 implementation
        match choice {
            snow::params::HashChoice::Blake2s => Some(Box::new(Blake3Hash::default())),
            _ => self.default.resolve_hash(choice),
        }
    }

    fn resolve_cipher(
        &self,
        choice: &snow::params::CipherChoice,
    ) -> Option<Box<dyn snow::types::Cipher>> {
        self.default.resolve_cipher(choice)
    }
}

/// BLAKE3 hash implementation for snow.
#[derive(Default)]
struct Blake3Hash {
    hasher: blake3::Hasher,
}

impl snow::types::Hash for Blake3Hash {
    fn name(&self) -> &'static str {
        "BLAKE3"
    }

    fn block_len(&self) -> usize {
        64 // BLAKE3 block size
    }

    fn hash_len(&self) -> usize {
        32 // Output 256 bits
    }

    fn reset(&mut self) {
        self.hasher = blake3::Hasher::new();
    }

    fn input(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    fn result(&mut self, out: &mut [u8]) {
        let hash = self.hasher.finalize();
        let hash_bytes = hash.as_bytes();
        // Only copy hash_len() bytes, even if out buffer is larger
        let len = self.hash_len().min(out.len());
        out[..len].copy_from_slice(&hash_bytes[..len]);
    }
}

/// Noise session for encrypted communication.
pub struct NoiseSession {
    state: NoiseState,
    role: HandshakeRole,
    remote_public: Option<PublicKey>,
    handshake_complete: bool,
}

impl NoiseSession {
    /// Create a new initiator (client) session.
    ///
    /// The initiator needs to know the responder's public key (from the auth key).
    pub fn new_initiator(
        local_secret: &SecretKey,
        remote_public: &PublicKey,
    ) -> Result<Self, CryptoError> {
        let params: NoiseParams = NOISE_PATTERN
            .parse()
            .map_err(|e| CryptoError::NoiseProtocol(format!("invalid pattern: {e}")))?;

        let secret_bytes = local_secret.as_bytes();
        let builder = Builder::with_resolver(params, Box::new(TriglavResolver::new()))
            .local_private_key(&secret_bytes)
            .remote_public_key(remote_public.as_bytes());

        let handshake = builder
            .build_initiator()
            .map_err(|e| CryptoError::NoiseProtocol(format!("build initiator failed: {e}")))?;

        Ok(Self {
            state: NoiseState::Handshake(Box::new(handshake)),
            role: HandshakeRole::Initiator,
            remote_public: Some(*remote_public),
            handshake_complete: false,
        })
    }

    /// Create a new responder (server) session.
    ///
    /// The responder uses its static key pair.
    pub fn new_responder(local_secret: &SecretKey) -> Result<Self, CryptoError> {
        let params: NoiseParams = NOISE_PATTERN
            .parse()
            .map_err(|e| CryptoError::NoiseProtocol(format!("invalid pattern: {e}")))?;

        let secret_bytes = local_secret.as_bytes();
        let builder = Builder::with_resolver(params, Box::new(TriglavResolver::new()))
            .local_private_key(&secret_bytes);

        let handshake = builder
            .build_responder()
            .map_err(|e| CryptoError::NoiseProtocol(format!("build responder failed: {e}")))?;

        Ok(Self {
            state: NoiseState::Handshake(Box::new(handshake)),
            role: HandshakeRole::Responder,
            remote_public: None,
            handshake_complete: false,
        })
    }

    /// Check if handshake is complete.
    pub fn is_handshake_complete(&self) -> bool {
        self.handshake_complete
    }

    /// Check if in transport mode (ready for encrypted messages).
    pub fn is_transport(&self) -> bool {
        matches!(self.state, NoiseState::Transport(_))
    }

    /// Get the role in this session.
    pub fn role(&self) -> HandshakeRole {
        self.role
    }

    /// Get the remote public key (if known).
    pub fn remote_public(&self) -> Option<&PublicKey> {
        self.remote_public.as_ref()
    }

    /// Write a handshake message (for initiator: first message, for responder: second message).
    pub fn write_handshake(&mut self, payload: &[u8]) -> Result<Vec<u8>, CryptoError> {
        match &mut self.state {
            NoiseState::Handshake(hs) => {
                let mut buf = vec![0u8; MAX_NOISE_MSG_SIZE];
                let len = hs
                    .write_message(payload, &mut buf)
                    .map_err(|e| CryptoError::NoiseProtocol(format!("write handshake: {e}")))?;
                buf.truncate(len);

                // Check if handshake is now complete
                if hs.is_handshake_finished() {
                    self.complete_handshake()?;
                }

                Ok(buf)
            }
            NoiseState::Transport(_) => Err(CryptoError::NoiseProtocol(
                "already in transport mode".into(),
            )),
            NoiseState::Failed => Err(CryptoError::NoiseProtocol("session failed".into())),
        }
    }

    /// Read a handshake message.
    pub fn read_handshake(&mut self, message: &[u8]) -> Result<Vec<u8>, CryptoError> {
        match &mut self.state {
            NoiseState::Handshake(hs) => {
                let mut buf = vec![0u8; MAX_NOISE_MSG_SIZE];
                let len = hs
                    .read_message(message, &mut buf)
                    .map_err(|e| CryptoError::NoiseProtocol(format!("read handshake: {e}")))?;
                buf.truncate(len);

                // Check if handshake is now complete
                if hs.is_handshake_finished() {
                    self.complete_handshake()?;
                }

                Ok(buf)
            }
            NoiseState::Transport(_) => Err(CryptoError::NoiseProtocol(
                "already in transport mode".into(),
            )),
            NoiseState::Failed => Err(CryptoError::NoiseProtocol("session failed".into())),
        }
    }

    /// Complete the handshake and transition to transport mode.
    fn complete_handshake(&mut self) -> Result<(), CryptoError> {
        let state = std::mem::replace(&mut self.state, NoiseState::Failed);

        match state {
            NoiseState::Handshake(hs) => {
                // Get remote public key before transitioning
                if self.remote_public.is_none() {
                    if let Some(rs) = hs.get_remote_static() {
                        let mut key = [0u8; 32];
                        key.copy_from_slice(rs);
                        self.remote_public = Some(PublicKey(key));
                    }
                }

                let transport = hs
                    .into_transport_mode()
                    .map_err(|e| CryptoError::NoiseProtocol(format!("transport mode: {e}")))?;
                self.state = NoiseState::Transport(Box::new(transport));
                self.handshake_complete = true;
                Ok(())
            }
            _ => Err(CryptoError::NoiseProtocol("not in handshake mode".into())),
        }
    }

    /// Encrypt a message (transport mode only).
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        match &mut self.state {
            NoiseState::Transport(ts) => {
                // Reserve space for ciphertext + auth tag (16 bytes)
                let mut buf = vec![0u8; plaintext.len() + 16];
                let len = ts
                    .write_message(plaintext, &mut buf)
                    .map_err(|e| CryptoError::EncryptionFailed(format!("noise encrypt: {e}")))?;
                buf.truncate(len);
                Ok(buf)
            }
            NoiseState::Handshake(_) => Err(CryptoError::EncryptionFailed(
                "handshake not complete".into(),
            )),
            NoiseState::Failed => Err(CryptoError::EncryptionFailed("session failed".into())),
        }
    }

    /// Decrypt a message (transport mode only).
    pub fn decrypt(&mut self, ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        match &mut self.state {
            NoiseState::Transport(ts) => {
                if ciphertext.len() < 16 {
                    return Err(CryptoError::InvalidCiphertextLength);
                }
                let mut buf = vec![0u8; ciphertext.len() - 16];
                let len = ts
                    .read_message(ciphertext, &mut buf)
                    .map_err(|e| CryptoError::DecryptionFailed(format!("noise decrypt: {e}")))?;
                buf.truncate(len);
                Ok(buf)
            }
            NoiseState::Handshake(_) => Err(CryptoError::DecryptionFailed(
                "handshake not complete".into(),
            )),
            NoiseState::Failed => Err(CryptoError::DecryptionFailed("session failed".into())),
        }
    }

    /// Get the current nonce counter (for debugging/monitoring).
    pub fn nonce_counter(&self) -> Option<u64> {
        match &self.state {
            NoiseState::Transport(ts) => Some(ts.sending_nonce()),
            _ => None,
        }
    }

    /// Rekey the session (for long-lived connections).
    pub fn rekey_outgoing(&mut self) -> Result<(), CryptoError> {
        match &mut self.state {
            NoiseState::Transport(ts) => {
                ts.rekey_outgoing();
                Ok(())
            }
            _ => Err(CryptoError::NoiseProtocol("not in transport mode".into())),
        }
    }

    /// Rekey incoming (for long-lived connections).
    pub fn rekey_incoming(&mut self) -> Result<(), CryptoError> {
        match &mut self.state {
            NoiseState::Transport(ts) => {
                ts.rekey_incoming();
                Ok(())
            }
            _ => Err(CryptoError::NoiseProtocol("not in transport mode".into())),
        }
    }
}

impl fmt::Debug for NoiseSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NoiseSession")
            .field("role", &self.role)
            .field("state", &self.state)
            .field("handshake_complete", &self.handshake_complete)
            .field("remote_public", &self.remote_public)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::KeyPair;

    /// Perform a full Noise NK handshake between initiator and responder.
    /// Returns the encrypted sessions for both sides (test-only helper).
    fn perform_handshake(
        initiator_secret: &SecretKey,
        responder_secret: &SecretKey,
        responder_public: &PublicKey,
    ) -> Result<(NoiseSession, NoiseSession), CryptoError> {
        let mut initiator = NoiseSession::new_initiator(initiator_secret, responder_public)?;
        let mut responder = NoiseSession::new_responder(responder_secret)?;

        // Initiator -> Responder (message 1: e, es)
        let msg1 = initiator.write_handshake(&[])?;

        // Responder processes message 1
        let _ = responder.read_handshake(&msg1)?;

        // Responder -> Initiator (message 2: e, ee)
        let msg2 = responder.write_handshake(&[])?;

        // Initiator processes message 2
        let _ = initiator.read_handshake(&msg2)?;

        // Both should now be in transport mode
        assert!(initiator.is_transport());
        assert!(responder.is_transport());

        Ok((initiator, responder))
    }

    #[test]
    fn test_noise_handshake() {
        let client_kp = KeyPair::generate();
        let server_kp = KeyPair::generate();

        let (mut client, mut server) =
            perform_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public).unwrap();

        // Test encryption/decryption
        let plaintext = b"hello from client";
        let ciphertext = client.encrypt(plaintext).unwrap();
        let decrypted = server.decrypt(&ciphertext).unwrap();
        assert_eq!(plaintext.as_slice(), decrypted.as_slice());

        // Test other direction
        let plaintext2 = b"hello from server";
        let ciphertext2 = server.encrypt(plaintext2).unwrap();
        let decrypted2 = client.decrypt(&ciphertext2).unwrap();
        assert_eq!(plaintext2.as_slice(), decrypted2.as_slice());
    }

    #[test]
    fn test_multiple_messages() {
        let client_kp = KeyPair::generate();
        let server_kp = KeyPair::generate();

        let (mut client, mut server) =
            perform_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public).unwrap();

        for i in 0..100 {
            let msg = format!("message {i}");
            let ciphertext = client.encrypt(msg.as_bytes()).unwrap();
            let decrypted = server.decrypt(&ciphertext).unwrap();
            assert_eq!(msg.as_bytes(), decrypted.as_slice());
        }
    }

    #[test]
    fn test_large_message() {
        let client_kp = KeyPair::generate();
        let server_kp = KeyPair::generate();

        let (mut client, mut server) =
            perform_handshake(&client_kp.secret, &server_kp.secret, &server_kp.public).unwrap();

        // Test with a larger message
        let plaintext = vec![0x42u8; 8192];
        let ciphertext = client.encrypt(&plaintext).unwrap();
        let decrypted = server.decrypt(&ciphertext).unwrap();
        assert_eq!(plaintext, decrypted);
    }
}
