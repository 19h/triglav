//! Key-based identity management.
//!
//! Users are identified solely by their public key. No passwords.
//! A "user" is simply an authorized key with associated metadata and limits.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::crypto::{KeyPair, PublicKey};
use crate::error::{Error, Result};

/// An authorized key (identity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizedKey {
    /// The public key in base64 (this IS the identity).
    pub public_key: String,
    /// Optional label/name for the key.
    pub label: Option<String>,
    /// Whether the key is enabled.
    pub enabled: bool,
    /// Created timestamp (Unix epoch seconds).
    pub created_at: u64,
    /// Last seen timestamp.
    pub last_seen: Option<u64>,
    /// Maximum allowed concurrent connections.
    pub max_connections: u32,
    /// Rate limit (bytes per second, 0 = unlimited).
    pub rate_limit_bps: u64,
    /// Expiry timestamp (0 = never).
    pub expires_at: u64,
    /// Total bytes sent.
    pub total_bytes_tx: u64,
    /// Total bytes received.
    pub total_bytes_rx: u64,
    /// Total connections made.
    pub total_connections: u64,
}

impl AuthorizedKey {
    /// Create a new authorized key from a public key.
    pub fn new(public_key: impl Into<String>) -> Self {
        Self {
            public_key: public_key.into(),
            label: None,
            enabled: true,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            last_seen: None,
            max_connections: 100,
            rate_limit_bps: 0,
            expires_at: 0,
            total_bytes_tx: 0,
            total_bytes_rx: 0,
            total_connections: 0,
        }
    }

    /// Create from a PublicKey.
    pub fn from_public_key(key: &PublicKey) -> Self {
        Self::new(key.to_base64())
    }

    /// Set a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set max connections.
    pub fn with_max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    /// Set rate limit.
    pub fn with_rate_limit(mut self, bps: u64) -> Self {
        self.rate_limit_bps = bps;
        self
    }

    /// Set expiry.
    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = expires_at;
        self
    }

    /// Check if the key is valid (enabled and not expired).
    pub fn is_valid(&self) -> bool {
        if !self.enabled {
            return false;
        }
        if self.expires_at > 0 {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            if now > self.expires_at {
                return false;
            }
        }
        true
    }

    /// Get the short ID (first 8 chars of public key).
    pub fn short_id(&self) -> &str {
        &self.public_key[..self.public_key.len().min(8)]
    }
}

/// Key store - manages authorized keys.
pub struct KeyStore {
    db: std::sync::Mutex<Connection>,
    /// In-memory cache for fast lookups.
    cache: RwLock<HashMap<String, AuthorizedKey>>,
    /// Last cache refresh.
    last_refresh: RwLock<Instant>,
}

impl KeyStore {
    /// Create a new key store with SQLite backing.
    pub fn new(db_path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(db_path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to open key database: {}", e)))?;
        
        Self::init_schema(&conn)?;
        
        let store = Self {
            db: std::sync::Mutex::new(conn),
            cache: RwLock::new(HashMap::new()),
            last_refresh: RwLock::new(Instant::now()),
        };
        
        store.refresh_cache()?;
        
        Ok(store)
    }

    /// Create an in-memory key store (for testing).
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| Error::Config(format!("Failed to create in-memory database: {}", e)))?;
        
        Self::init_schema(&conn)?;
        
        Ok(Self {
            db: std::sync::Mutex::new(conn),
            cache: RwLock::new(HashMap::new()),
            last_refresh: RwLock::new(Instant::now()),
        })
    }

    /// Initialize database schema.
    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS authorized_keys (
                public_key TEXT PRIMARY KEY,
                label TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL,
                last_seen INTEGER,
                max_connections INTEGER NOT NULL DEFAULT 100,
                rate_limit_bps INTEGER NOT NULL DEFAULT 0,
                expires_at INTEGER NOT NULL DEFAULT 0,
                total_bytes_tx INTEGER NOT NULL DEFAULT 0,
                total_bytes_rx INTEGER NOT NULL DEFAULT 0,
                total_connections INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_keys_enabled ON authorized_keys(enabled);
            CREATE INDEX IF NOT EXISTS idx_keys_expires ON authorized_keys(expires_at);
            "#
        ).map_err(|e| Error::Config(format!("Failed to initialize schema: {}", e)))?;
        
        Ok(())
    }

    /// Refresh the in-memory cache.
    pub fn refresh_cache(&self) -> Result<()> {
        let db = self.db.lock().unwrap();
        
        let mut stmt = db.prepare(
            "SELECT public_key, label, enabled, created_at, last_seen, 
                    max_connections, rate_limit_bps, expires_at,
                    total_bytes_tx, total_bytes_rx, total_connections
             FROM authorized_keys"
        ).map_err(|e| Error::Config(format!("Failed to prepare query: {}", e)))?;
        
        let keys: Vec<AuthorizedKey> = stmt.query_map([], |row| {
            Ok(AuthorizedKey {
                public_key: row.get(0)?,
                label: row.get(1)?,
                enabled: row.get::<_, i32>(2)? != 0,
                created_at: row.get(3)?,
                last_seen: row.get(4)?,
                max_connections: row.get(5)?,
                rate_limit_bps: row.get(6)?,
                expires_at: row.get(7)?,
                total_bytes_tx: row.get(8)?,
                total_bytes_rx: row.get(9)?,
                total_connections: row.get(10)?,
            })
        }).map_err(|e| Error::Config(format!("Failed to query keys: {}", e)))?
        .filter_map(|r| r.ok())
        .collect();
        
        let mut cache = self.cache.write();
        cache.clear();
        for key in keys {
            cache.insert(key.public_key.clone(), key);
        }
        
        *self.last_refresh.write() = Instant::now();
        
        info!("Loaded {} authorized keys", cache.len());
        
        Ok(())
    }

    /// Authorize a new key.
    pub fn authorize(&self, key: AuthorizedKey) -> Result<()> {
        let db = self.db.lock().unwrap();
        db.execute(
            "INSERT OR REPLACE INTO authorized_keys 
             (public_key, label, enabled, created_at, last_seen, max_connections, 
              rate_limit_bps, expires_at, total_bytes_tx, total_bytes_rx, total_connections)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                key.public_key,
                key.label,
                if key.enabled { 1 } else { 0 },
                key.created_at,
                key.last_seen,
                key.max_connections,
                key.rate_limit_bps,
                key.expires_at,
                key.total_bytes_tx,
                key.total_bytes_rx,
                key.total_connections,
            ]
        ).map_err(|e| Error::Config(format!("Failed to authorize key: {}", e)))?;
        
        drop(db);
        
        // Update cache
        self.cache.write().insert(key.public_key.clone(), key);
        
        Ok(())
    }

    /// Authorize a key from a PublicKey.
    pub fn authorize_public_key(&self, public_key: &PublicKey, label: Option<&str>) -> Result<AuthorizedKey> {
        let mut key = AuthorizedKey::from_public_key(public_key);
        if let Some(l) = label {
            key.label = Some(l.to_string());
        }
        self.authorize(key.clone())?;
        Ok(key)
    }

    /// Generate and authorize a new keypair.
    pub fn generate_authorized_key(&self, label: Option<&str>) -> Result<(AuthorizedKey, KeyPair)> {
        let keypair = KeyPair::generate();
        let key = self.authorize_public_key(&keypair.public, label)?;
        Ok((key, keypair))
    }

    /// Check if a public key is authorized.
    pub fn is_authorized(&self, public_key: &str) -> bool {
        self.cache.read()
            .get(public_key)
            .map(|k| k.is_valid())
            .unwrap_or(false)
    }

    /// Get an authorized key.
    pub fn get(&self, public_key: &str) -> Option<AuthorizedKey> {
        self.cache.read().get(public_key).cloned()
    }

    /// Authenticate a connection attempt. Returns the key if authorized.
    pub fn authenticate(&self, public_key: &str) -> Option<AuthorizedKey> {
        let key = self.cache.read().get(public_key).cloned()?;
        
        if !key.is_valid() {
            debug!("Key {} is not valid", key.short_id());
            return None;
        }
        
        // Update last seen
        let _ = self.update_last_seen(public_key);
        let _ = self.increment_connections(public_key);
        
        Some(key)
    }

    /// Revoke a key.
    pub fn revoke(&self, public_key: &str) -> Result<()> {
        let db = self.db.lock().unwrap();
        db.execute(
            "UPDATE authorized_keys SET enabled = 0 WHERE public_key = ?1",
            params![public_key]
        ).map_err(|e| Error::Config(format!("Failed to revoke key: {}", e)))?;
        
        drop(db);
        
        // Update cache
        if let Some(key) = self.cache.write().get_mut(public_key) {
            key.enabled = false;
        }
        
        info!("Revoked key {}", &public_key[..public_key.len().min(8)]);
        Ok(())
    }

    /// Delete a key entirely.
    pub fn delete(&self, public_key: &str) -> Result<()> {
        let db = self.db.lock().unwrap();
        db.execute(
            "DELETE FROM authorized_keys WHERE public_key = ?1",
            params![public_key]
        ).map_err(|e| Error::Config(format!("Failed to delete key: {}", e)))?;
        
        drop(db);
        
        self.cache.write().remove(public_key);
        
        info!("Deleted key {}", &public_key[..public_key.len().min(8)]);
        Ok(())
    }

    /// List all authorized keys.
    pub fn list(&self) -> Vec<AuthorizedKey> {
        self.cache.read().values().cloned().collect()
    }

    /// List only valid (enabled, not expired) keys.
    pub fn list_valid(&self) -> Vec<AuthorizedKey> {
        self.cache.read()
            .values()
            .filter(|k| k.is_valid())
            .cloned()
            .collect()
    }

    /// Get key count.
    pub fn count(&self) -> usize {
        self.cache.read().len()
    }

    /// Update last seen timestamp.
    fn update_last_seen(&self, public_key: &str) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let db = self.db.lock().unwrap();
        db.execute(
            "UPDATE authorized_keys SET last_seen = ?1 WHERE public_key = ?2",
            params![now, public_key]
        ).map_err(|e| Error::Config(format!("Failed to update last seen: {}", e)))?;
        
        drop(db);
        
        if let Some(key) = self.cache.write().get_mut(public_key) {
            key.last_seen = Some(now);
        }
        
        Ok(())
    }

    /// Increment connection count.
    fn increment_connections(&self, public_key: &str) -> Result<()> {
        let db = self.db.lock().unwrap();
        db.execute(
            "UPDATE authorized_keys SET total_connections = total_connections + 1 WHERE public_key = ?1",
            params![public_key]
        ).map_err(|e| Error::Config(format!("Failed to increment connections: {}", e)))?;
        
        drop(db);
        
        if let Some(key) = self.cache.write().get_mut(public_key) {
            key.total_connections += 1;
        }
        
        Ok(())
    }

    /// Record traffic for a key.
    pub fn record_traffic(&self, public_key: &str, bytes_tx: u64, bytes_rx: u64) -> Result<()> {
        let db = self.db.lock().unwrap();
        db.execute(
            "UPDATE authorized_keys SET 
                total_bytes_tx = total_bytes_tx + ?1,
                total_bytes_rx = total_bytes_rx + ?2
             WHERE public_key = ?3",
            params![bytes_tx, bytes_rx, public_key]
        ).map_err(|e| Error::Config(format!("Failed to record traffic: {}", e)))?;
        
        drop(db);
        
        if let Some(key) = self.cache.write().get_mut(public_key) {
            key.total_bytes_tx += bytes_tx;
            key.total_bytes_rx += bytes_rx;
        }
        
        Ok(())
    }
}

// Re-export old types for compatibility but mark as deprecated
#[deprecated(note = "Use AuthorizedKey instead")]
pub type User = AuthorizedKey;

#[deprecated(note = "Use KeyStore instead")]
pub type UserManager = KeyStore;

/// User role - simplified, mostly for display/grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    #[default]
    User,
    Admin,
    Api,
    ReadOnly,
}

impl std::fmt::Display for UserRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UserRole::User => write!(f, "user"),
            UserRole::Admin => write!(f, "admin"),
            UserRole::Api => write!(f, "api"),
            UserRole::ReadOnly => write!(f, "readonly"),
        }
    }
}

/// Registration request (for compatibility).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRegistration {
    pub username: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<UserRole>,
}

/// User key (for compatibility).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserKey {
    pub id: String,
    pub user_id: String,
    pub public_key: String,
    pub name: String,
    pub created_at: u64,
    pub expires_at: u64,
    pub revoked: bool,
    pub revocation_reason: Option<String>,
    pub last_used: Option<u64>,
}

impl UserKey {
    pub fn is_valid(&self) -> bool {
        if self.revoked {
            return false;
        }
        if self.expires_at > 0 {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            if now > self.expires_at {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_authorize_key() {
        let store = KeyStore::in_memory().unwrap();
        
        let keypair = KeyPair::generate();
        let key = store.authorize_public_key(&keypair.public, Some("test-key")).unwrap();
        
        assert!(key.enabled);
        assert_eq!(key.label, Some("test-key".to_string()));
        assert!(store.is_authorized(&key.public_key));
    }

    #[test]
    fn test_authenticate() {
        let store = KeyStore::in_memory().unwrap();
        
        let keypair = KeyPair::generate();
        store.authorize_public_key(&keypair.public, None).unwrap();
        
        let pubkey = keypair.public.to_base64();
        let result = store.authenticate(&pubkey);
        assert!(result.is_some());
        
        // Check that last_seen was updated
        let key = store.get(&pubkey).unwrap();
        assert!(key.last_seen.is_some());
        assert_eq!(key.total_connections, 1);
    }

    #[test]
    fn test_revoke() {
        let store = KeyStore::in_memory().unwrap();
        
        let keypair = KeyPair::generate();
        let key = store.authorize_public_key(&keypair.public, None).unwrap();
        
        assert!(store.is_authorized(&key.public_key));
        
        store.revoke(&key.public_key).unwrap();
        
        assert!(!store.is_authorized(&key.public_key));
        assert!(store.authenticate(&key.public_key).is_none());
    }

    #[test]
    fn test_expiry() {
        let store = KeyStore::in_memory().unwrap();
        
        let keypair = KeyPair::generate();
        let mut key = AuthorizedKey::from_public_key(&keypair.public);
        // Set expiry in the past
        key.expires_at = 1;
        store.authorize(key.clone()).unwrap();
        
        assert!(!store.is_authorized(&key.public_key));
        assert!(store.authenticate(&key.public_key).is_none());
    }

    #[test]
    fn test_generate_authorized_key() {
        let store = KeyStore::in_memory().unwrap();
        
        let (key, keypair) = store.generate_authorized_key(Some("generated")).unwrap();
        
        assert_eq!(key.public_key, keypair.public.to_base64());
        assert_eq!(key.label, Some("generated".to_string()));
        assert!(store.is_authorized(&key.public_key));
    }

    #[test]
    fn test_traffic_recording() {
        let store = KeyStore::in_memory().unwrap();
        
        let keypair = KeyPair::generate();
        let key = store.authorize_public_key(&keypair.public, None).unwrap();
        
        store.record_traffic(&key.public_key, 1000, 2000).unwrap();
        store.record_traffic(&key.public_key, 500, 500).unwrap();
        
        let updated = store.get(&key.public_key).unwrap();
        assert_eq!(updated.total_bytes_tx, 1500);
        assert_eq!(updated.total_bytes_rx, 2500);
    }
}
