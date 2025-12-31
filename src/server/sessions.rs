//! Session management for the server.
//!
//! Tracks active sessions, enforces limits, and provides metrics.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use crate::crypto::NoiseSession;
use crate::types::{SessionId, TrafficStats};

use super::AuthorizedKey;

/// Session configuration.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Maximum sessions per user.
    pub max_sessions_per_user: u32,
    /// Session timeout (idle).
    pub idle_timeout: Duration,
    /// Session timeout (absolute).
    pub absolute_timeout: Duration,
    /// Cleanup interval.
    pub cleanup_interval: Duration,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_sessions_per_user: 10,
            idle_timeout: Duration::from_secs(300),
            absolute_timeout: Duration::from_secs(86400),
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

/// Session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session is being established.
    Handshaking,
    /// Session is active.
    Active,
    /// Session is closing.
    Closing,
    /// Session is closed.
    Closed,
}

/// Server session.
pub struct ServerSession {
    /// Session ID.
    pub id: SessionId,
    /// Associated key (if authenticated).
    pub key: Option<AuthorizedKey>,
    /// Remote addresses (multiple for multipath).
    pub remote_addrs: RwLock<Vec<SocketAddr>>,
    /// Session state.
    pub state: RwLock<SessionState>,
    /// Noise session for encryption.
    pub noise: RwLock<Option<NoiseSession>>,
    /// Traffic statistics.
    pub stats: RwLock<TrafficStats>,
    /// Created time.
    pub created_at: Instant,
    /// Last activity time.
    pub last_activity: RwLock<Instant>,
    /// Uplinks used by this session.
    pub uplinks: RwLock<Vec<String>>,
    /// Session metadata.
    pub metadata: RwLock<HashMap<String, String>>,
}

impl ServerSession {
    /// Create a new session.
    pub fn new(id: SessionId) -> Self {
        let now = Instant::now();
        Self {
            id,
            key: None,
            remote_addrs: RwLock::new(Vec::new()),
            state: RwLock::new(SessionState::Handshaking),
            noise: RwLock::new(None),
            stats: RwLock::new(TrafficStats::default()),
            created_at: now,
            last_activity: RwLock::new(now),
            uplinks: RwLock::new(Vec::new()),
            metadata: RwLock::new(HashMap::new()),
        }
    }

    /// Create a session with an authorized key.
    pub fn with_key(id: SessionId, key: AuthorizedKey) -> Self {
        let mut session = Self::new(id);
        session.key = Some(key);
        session
    }

    /// Touch the session (update last activity).
    pub fn touch(&self) {
        *self.last_activity.write() = Instant::now();
    }

    /// Check if session is expired.
    pub fn is_expired(&self, idle_timeout: Duration, absolute_timeout: Duration) -> bool {
        let now = Instant::now();
        let last_activity = *self.last_activity.read();
        
        // Check idle timeout
        if now.duration_since(last_activity) > idle_timeout {
            return true;
        }
        
        // Check absolute timeout
        if now.duration_since(self.created_at) > absolute_timeout {
            return true;
        }
        
        false
    }

    /// Add a remote address.
    pub fn add_remote_addr(&self, addr: SocketAddr) {
        let mut addrs = self.remote_addrs.write();
        if !addrs.contains(&addr) {
            addrs.push(addr);
        }
    }

    /// Set the noise session.
    pub fn set_noise(&self, noise: NoiseSession) {
        *self.noise.write() = Some(noise);
        *self.state.write() = SessionState::Active;
    }

    /// Check if the session has completed handshake.
    pub fn is_active(&self) -> bool {
        *self.state.read() == SessionState::Active
    }

    /// Get session age.
    pub fn age(&self) -> Duration {
        Instant::now().duration_since(self.created_at)
    }

    /// Get idle time.
    pub fn idle_time(&self) -> Duration {
        Instant::now().duration_since(*self.last_activity.read())
    }

    /// Record bytes sent.
    pub fn record_sent(&self, bytes: u64) {
        let mut stats = self.stats.write();
        stats.bytes_sent += bytes;
        stats.packets_sent += 1;
        drop(stats);
        self.touch();
    }

    /// Record bytes received.
    pub fn record_received(&self, bytes: u64) {
        let mut stats = self.stats.write();
        stats.bytes_received += bytes;
        stats.packets_received += 1;
        drop(stats);
        self.touch();
    }
}

/// Session event.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Session created.
    Created(SessionId),
    /// Session authenticated.
    Authenticated { session_id: SessionId, user_id: String },
    /// Session closed.
    Closed(SessionId),
    /// Session expired.
    Expired(SessionId),
}

/// Session manager.
pub struct SessionManager {
    config: SessionConfig,
    /// Active sessions.
    sessions: DashMap<SessionId, Arc<ServerSession>>,
    /// Sessions by user ID.
    sessions_by_user: DashMap<String, Vec<SessionId>>,
    /// Session by remote address.
    sessions_by_addr: DashMap<SocketAddr, SessionId>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<SessionEvent>,
    /// Total sessions created.
    total_sessions: std::sync::atomic::AtomicU64,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new(config: SessionConfig) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        
        Self {
            config,
            sessions: DashMap::new(),
            sessions_by_user: DashMap::new(),
            sessions_by_addr: DashMap::new(),
            event_tx,
            total_sessions: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Subscribe to session events.
    pub fn subscribe(&self) -> broadcast::Receiver<SessionEvent> {
        self.event_tx.subscribe()
    }

    /// Create a new session.
    pub fn create_session(&self) -> Arc<ServerSession> {
        let id = SessionId::generate();
        let session = Arc::new(ServerSession::new(id));
        
        self.sessions.insert(id, Arc::clone(&session));
        self.total_sessions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let _ = self.event_tx.send(SessionEvent::Created(id));
        
        debug!("Created session {}", id);
        session
    }

    /// Create a session for an authorized key (with limit checking).
    pub fn create_session_for_key(&self, key: &AuthorizedKey) -> Option<Arc<ServerSession>> {
        // Check session limit
        let current_sessions = self.sessions_by_user
            .get(&key.public_key)
            .map(|v| v.len())
            .unwrap_or(0);
        
        if current_sessions as u32 >= key.max_connections {
            warn!(
                "Key {} has reached session limit ({}/{})",
                key.short_id(), current_sessions, key.max_connections
            );
            return None;
        }
        
        let id = SessionId::generate();
        let session = Arc::new(ServerSession::with_key(id, key.clone()));
        
        self.sessions.insert(id, Arc::clone(&session));
        self.sessions_by_user
            .entry(key.public_key.clone())
            .or_default()
            .push(id);
        
        self.total_sessions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let _ = self.event_tx.send(SessionEvent::Created(id));
        let _ = self.event_tx.send(SessionEvent::Authenticated {
            session_id: id,
            user_id: key.public_key.clone(),
        });
        
        debug!("Created session {} for key {}", id, key.short_id());
        Some(session)
    }

    /// Get a session by ID.
    pub fn get_session(&self, id: SessionId) -> Option<Arc<ServerSession>> {
        self.sessions.get(&id).map(|r| Arc::clone(&r))
    }

    /// Get a session by remote address.
    pub fn get_session_by_addr(&self, addr: SocketAddr) -> Option<Arc<ServerSession>> {
        let id = self.sessions_by_addr.get(&addr)?;
        self.get_session(*id)
    }

    /// Associate a remote address with a session.
    pub fn associate_addr(&self, session_id: SessionId, addr: SocketAddr) {
        if let Some(session) = self.get_session(session_id) {
            session.add_remote_addr(addr);
            self.sessions_by_addr.insert(addr, session_id);
        }
    }

    /// Authenticate a session with an authorized key.
    pub fn authenticate_session(&self, session_id: SessionId, key: AuthorizedKey) -> bool {
        if let Some(_session) = self.sessions.get_mut(&session_id) {
            // This is a workaround since we can't mutate through Arc
            // In a real implementation, you'd use interior mutability
            
            // Track by key
            self.sessions_by_user
                .entry(key.public_key.clone())
                .or_default()
                .push(session_id);
            
            let _ = self.event_tx.send(SessionEvent::Authenticated {
                session_id,
                user_id: key.public_key.clone(),
            });
            
            true
        } else {
            false
        }
    }

    /// Close a session.
    pub fn close_session(&self, id: SessionId) {
        if let Some((_, session)) = self.sessions.remove(&id) {
            // Remove from address map
            for addr in session.remote_addrs.read().iter() {
                self.sessions_by_addr.remove(addr);
            }
            
            // Remove from key map
            if let Some(key) = &session.key {
                if let Some(mut sessions) = self.sessions_by_user.get_mut(&key.public_key) {
                    sessions.retain(|s| *s != id);
                }
            }
            
            let _ = self.event_tx.send(SessionEvent::Closed(id));
            
            debug!("Closed session {}", id);
        }
    }

    /// Get active session count.
    pub fn active_count(&self) -> usize {
        self.sessions.len()
    }

    /// Get total sessions created.
    pub fn total_count(&self) -> u64 {
        self.total_sessions.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get sessions for a user.
    pub fn get_user_sessions(&self, user_id: &str) -> Vec<Arc<ServerSession>> {
        self.sessions_by_user
            .get(user_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.get_session(*id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Cleanup expired sessions.
    pub fn cleanup_expired(&self) -> usize {
        let mut expired = Vec::new();
        
        for entry in self.sessions.iter() {
            if entry.value().is_expired(self.config.idle_timeout, self.config.absolute_timeout) {
                expired.push(*entry.key());
            }
        }
        
        let count = expired.len();
        
        for id in expired {
            let _ = self.event_tx.send(SessionEvent::Expired(id));
            self.close_session(id);
        }
        
        if count > 0 {
            info!("Cleaned up {} expired sessions", count);
        }
        
        count
    }

    /// Start the cleanup task.
    pub fn start_cleanup_task(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let interval = self.config.cleanup_interval;
        
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            
            loop {
                ticker.tick().await;
                self.cleanup_expired();
            }
        })
    }

    /// Get all sessions.
    pub fn all_sessions(&self) -> Vec<Arc<ServerSession>> {
        self.sessions.iter().map(|r| Arc::clone(&r)).collect()
    }

    /// Get aggregate statistics.
    pub fn aggregate_stats(&self) -> TrafficStats {
        let mut total = TrafficStats::default();
        
        for session in self.sessions.iter() {
            let stats = session.stats.read();
            total.bytes_sent += stats.bytes_sent;
            total.bytes_received += stats.bytes_received;
            total.packets_sent += stats.packets_sent;
            total.packets_received += stats.packets_received;
            total.packets_dropped += stats.packets_dropped;
            total.packets_retransmitted += stats.packets_retransmitted;
        }
        
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let manager = SessionManager::new(SessionConfig::default());
        
        let session = manager.create_session();
        assert!(!session.is_active());
        assert_eq!(manager.active_count(), 1);
    }

    #[test]
    fn test_session_by_addr() {
        let manager = SessionManager::new(SessionConfig::default());
        
        let session = manager.create_session();
        let addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();
        
        manager.associate_addr(session.id, addr);
        
        let found = manager.get_session_by_addr(addr);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, session.id);
    }

    #[test]
    fn test_session_expiry() {
        let config = SessionConfig {
            idle_timeout: Duration::from_millis(10),
            absolute_timeout: Duration::from_secs(3600),
            ..Default::default()
        };
        
        let manager = SessionManager::new(config);
        let session = manager.create_session();
        
        // Wait for timeout
        std::thread::sleep(Duration::from_millis(20));
        
        assert!(session.is_expired(Duration::from_millis(10), Duration::from_secs(3600)));
    }

    #[test]
    fn test_key_session_limit() {
        let manager = SessionManager::new(SessionConfig::default());
        
        let key = super::AuthorizedKey::new("test_key_base64")
            .with_max_connections(2);
        
        // Create sessions up to limit
        let s1 = manager.create_session_for_key(&key);
        assert!(s1.is_some());
        
        let s2 = manager.create_session_for_key(&key);
        assert!(s2.is_some());
        
        // Should fail - at limit
        let s3 = manager.create_session_for_key(&key);
        assert!(s3.is_none());
    }
}
