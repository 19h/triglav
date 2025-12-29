//! Multi-path connection manager.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU16, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};

use super::{Scheduler, SchedulerConfig, Uplink, UplinkConfig};
use crate::crypto::{NoiseSession, KeyPair, PublicKey};
use crate::error::{Error, Result};
use crate::protocol::{Packet, PacketType, PacketFlags};
use crate::transport::{Transport, TransportConfig, connect};
use crate::types::{
    ConnectionId, ConnectionState, SequenceNumber, SessionId,
    TrafficStats, UplinkHealth, UplinkId,
};

/// Multi-path manager configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipathConfig {
    /// Scheduler configuration.
    #[serde(default)]
    pub scheduler: SchedulerConfig,

    /// Transport configuration.
    #[serde(default)]
    pub transport: TransportConfig,

    /// Maximum number of uplinks.
    #[serde(default = "default_max_uplinks")]
    pub max_uplinks: usize,

    /// Retry count for failed sends.
    #[serde(default = "default_retries")]
    pub send_retries: u32,

    /// Retry delay base.
    #[serde(default = "default_retry_delay", with = "humantime_serde")]
    pub retry_delay: Duration,

    /// Enable automatic uplink discovery.
    #[serde(default)]
    pub auto_discover: bool,

    /// Uplink health check interval.
    #[serde(default = "default_health_interval", with = "humantime_serde")]
    pub health_check_interval: Duration,

    /// Enable packet deduplication.
    #[serde(default = "default_dedup")]
    pub deduplication: bool,

    /// Deduplication window size.
    #[serde(default = "default_dedup_window")]
    pub dedup_window_size: usize,
}

fn default_max_uplinks() -> usize { 16 }
fn default_retries() -> u32 { 3 }
fn default_retry_delay() -> Duration { Duration::from_millis(100) }
fn default_health_interval() -> Duration { Duration::from_secs(1) }
fn default_dedup() -> bool { true }
fn default_dedup_window() -> usize { 1000 }

impl Default for MultipathConfig {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            transport: TransportConfig::default(),
            max_uplinks: default_max_uplinks(),
            send_retries: default_retries(),
            retry_delay: default_retry_delay(),
            auto_discover: false,
            health_check_interval: default_health_interval(),
            deduplication: default_dedup(),
            dedup_window_size: default_dedup_window(),
        }
    }
}

/// Event types for multipath manager.
#[derive(Debug, Clone)]
pub enum MultipathEvent {
    /// Uplink connected.
    UplinkConnected(UplinkId),
    /// Uplink disconnected.
    UplinkDisconnected(UplinkId),
    /// Uplink health changed.
    UplinkHealthChanged(UplinkId, UplinkHealth),
    /// Packet received.
    PacketReceived(Packet),
    /// Send failed on all uplinks.
    SendFailed(u64), // sequence number
    /// All uplinks down.
    AllUplinksDown,
    /// Connection established.
    Connected,
    /// Connection closed.
    Disconnected,
}

/// Pending packet for retry.
#[derive(Debug)]
struct PendingPacket {
    packet: Packet,
    sent_at: Instant,
    retries: u32,
    uplink_id: u16,
}

/// Sliding window for packet deduplication with O(1) operations.
#[derive(Debug)]
struct DeduplicationWindow {
    /// Set for O(1) duplicate detection.
    seen: HashSet<u64>,
    /// Queue for maintaining insertion order (for cleanup).
    order: VecDeque<u64>,
    /// Maximum window size.
    max_size: usize,
}

impl DeduplicationWindow {
    /// Create a new deduplication window.
    fn new(max_size: usize) -> Self {
        Self {
            seen: HashSet::with_capacity(max_size),
            order: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Check if sequence number is a duplicate. Returns true if duplicate, false if new.
    /// If new, adds the sequence number to the window.
    fn is_duplicate(&mut self, seq: u64) -> bool {
        if self.seen.contains(&seq) {
            return true; // Duplicate
        }

        // New sequence number - add to window
        self.seen.insert(seq);
        self.order.push_back(seq);

        // Cleanup: remove oldest entries if window is full
        while self.order.len() > self.max_size {
            if let Some(old_seq) = self.order.pop_front() {
                self.seen.remove(&old_seq);
            }
        }

        false // Not a duplicate
    }
}

/// Multi-path connection manager.
pub struct MultipathManager {
    /// Configuration.
    config: MultipathConfig,
    /// Session ID.
    session_id: SessionId,
    /// Connection ID.
    connection_id: ConnectionId,
    /// Active uplinks.
    uplinks: DashMap<u16, Arc<Uplink>>,
    /// Uplink ID to numeric ID mapping.
    uplink_ids: DashMap<UplinkId, u16>,
    /// Scheduler.
    scheduler: Scheduler,
    /// Next sequence number.
    next_seq: AtomicU64,
    /// Next uplink numeric ID.
    next_uplink_id: AtomicU16,
    /// Pending packets (for retry).
    pending: DashMap<u64, PendingPacket>,
    /// Deduplication window for received packets.
    dedup_window: RwLock<DeduplicationWindow>,
    /// Aggregate stats.
    stats: RwLock<TrafficStats>,
    /// Event sender.
    event_tx: broadcast::Sender<MultipathEvent>,
    /// State.
    state: RwLock<ConnectionState>,
    /// Local keypair.
    local_keypair: KeyPair,
}

impl MultipathManager {
    /// Create a new multipath manager.
    pub fn new(config: MultipathConfig, local_keypair: KeyPair) -> Self {
        let (event_tx, _) = broadcast::channel(256);

        Self {
            config: config.clone(),
            session_id: SessionId::generate(),
            connection_id: ConnectionId::new(),
            uplinks: DashMap::new(),
            uplink_ids: DashMap::new(),
            scheduler: Scheduler::new(config.scheduler),
            next_seq: AtomicU64::new(1),
            next_uplink_id: AtomicU16::new(0),
            pending: DashMap::new(),
            dedup_window: RwLock::new(DeduplicationWindow::new(config.dedup_window_size)),
            stats: RwLock::new(TrafficStats::default()),
            event_tx,
            state: RwLock::new(ConnectionState::Disconnected),
            local_keypair,
        }
    }

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Get the connection ID.
    pub fn connection_id(&self) -> ConnectionId {
        self.connection_id
    }

    /// Get current state.
    pub fn state(&self) -> ConnectionState {
        *self.state.read()
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> TrafficStats {
        *self.stats.read()
    }

    /// Subscribe to events.
    pub fn subscribe(&self) -> broadcast::Receiver<MultipathEvent> {
        self.event_tx.subscribe()
    }

    /// Get uplink count.
    pub fn uplink_count(&self) -> usize {
        self.uplinks.len()
    }

    /// Get all uplinks.
    pub fn uplinks(&self) -> Vec<Arc<Uplink>> {
        self.uplinks.iter().map(|r| r.value().clone()).collect()
    }

    /// Get uplink by numeric ID.
    pub fn get_uplink(&self, id: u16) -> Option<Arc<Uplink>> {
        self.uplinks.get(&id).map(|r| r.value().clone())
    }

    /// Get uplink by name.
    pub fn get_uplink_by_name(&self, name: &UplinkId) -> Option<Arc<Uplink>> {
        self.uplink_ids.get(name)
            .and_then(|id| self.uplinks.get(&*id).map(|r| r.value().clone()))
    }

    /// Add an uplink.
    pub fn add_uplink(&self, config: UplinkConfig) -> Result<u16> {
        if self.uplinks.len() >= self.config.max_uplinks {
            return Err(Error::Config(format!(
                "maximum uplinks ({}) reached",
                self.config.max_uplinks
            )));
        }

        let numeric_id = self.next_uplink_id.fetch_add(1, Ordering::SeqCst);
        let uplink = Arc::new(Uplink::new(config.clone(), numeric_id));

        self.uplinks.insert(numeric_id, uplink);
        self.uplink_ids.insert(config.id, numeric_id);

        Ok(numeric_id)
    }

    /// Remove an uplink.
    pub fn remove_uplink(&self, id: u16) -> Option<Arc<Uplink>> {
        if let Some((_, uplink)) = self.uplinks.remove(&id) {
            self.uplink_ids.remove(uplink.id());
            let _ = self.event_tx.send(MultipathEvent::UplinkDisconnected(uplink.id().clone()));
            Some(uplink)
        } else {
            None
        }
    }

    /// Connect all uplinks to the remote server.
    pub async fn connect(&self, remote_public: PublicKey) -> Result<()> {
        *self.state.write() = ConnectionState::Connecting;

        let uplinks: Vec<_> = self.uplinks.iter().map(|r| r.value().clone()).collect();

        for uplink in uplinks {
            if let Err(e) = self.connect_uplink(&uplink, &remote_public).await {
                tracing::warn!(
                    uplink = %uplink.id(),
                    error = %e,
                    "Failed to connect uplink"
                );
                uplink.record_failure(&e.to_string());
            }
        }

        // Check if at least one uplink connected
        let connected = self.uplinks.iter()
            .any(|r| r.value().state().connection_state == ConnectionState::Connected);

        if connected {
            *self.state.write() = ConnectionState::Connected;
            let _ = self.event_tx.send(MultipathEvent::Connected);
            Ok(())
        } else {
            *self.state.write() = ConnectionState::Failed;
            Err(Error::NoAvailableUplinks)
        }
    }

    /// Connect a single uplink.
    async fn connect_uplink(&self, uplink: &Uplink, remote_public: &PublicKey) -> Result<()> {
        uplink.set_connection_state(ConnectionState::Connecting);

        let config = uplink.config();

        // Create transport
        let transport = connect(
            config.protocol,
            config.remote_addr,
            config.local_addr,
            &self.config.transport,
        ).await?;

        // Wrap in Arc for sharing across async operations
        let transport: Arc<dyn Transport> = Arc::from(transport);
        uplink.set_transport(transport.clone());

        // Perform Noise handshake
        uplink.set_connection_state(ConnectionState::Handshaking);

        let mut noise = NoiseSession::new_initiator(&self.local_keypair.secret, remote_public)?;

        // Write handshake message 1 (e, es)
        let msg1 = noise.write_handshake(&[])?;

        // Create handshake packet
        let handshake_packet = Packet::new(
            PacketType::Handshake,
            SequenceNumber(0),
            self.session_id,
            uplink.numeric_id(),
            msg1,
        )?;

        // Send handshake message
        let encoded = handshake_packet.encode()?;
        transport.send(&encoded).await?;

        tracing::debug!(
            uplink = %uplink.id(),
            "Sent handshake message 1"
        );

        // Receive handshake response
        let mut buf = vec![0u8; 65536];
        let len = tokio::time::timeout(
            self.config.transport.connect_timeout,
            transport.recv(&mut buf)
        ).await
            .map_err(|_| Error::ConnectionTimeout)?
            ?;

        // Decode response packet
        let response_packet = Packet::decode(&buf[..len])?;

        if response_packet.header.packet_type != PacketType::Handshake {
            return Err(Error::Protocol(crate::error::ProtocolError::UnexpectedMessage {
                expected: "Handshake".into(),
                got: format!("{:?}", response_packet.header.packet_type),
            }));
        }

        // Process handshake message 2
        let _ = noise.read_handshake(&response_packet.payload)?;

        if !noise.is_transport() {
            return Err(Error::Crypto(crate::error::CryptoError::NoiseProtocol(
                "Handshake did not complete".into()
            )));
        }

        tracing::debug!(
            uplink = %uplink.id(),
            "Handshake complete, noise session ready"
        );

        uplink.set_noise_session(noise);
        uplink.set_connection_state(ConnectionState::Connected);

        let _ = self.event_tx.send(MultipathEvent::UplinkConnected(uplink.id().clone()));

        Ok(())
    }

    /// Send data through the multi-path connection.
    pub async fn send(&self, data: &[u8]) -> Result<u64> {
        let seq = SequenceNumber(self.next_seq.fetch_add(1, Ordering::SeqCst));

        // Create packet
        let uplinks = self.uplinks();
        let selected = self.scheduler.select(&uplinks, None);

        if selected.is_empty() {
            return Err(Error::NoAvailableUplinks);
        }

        // Send on selected uplinks
        let mut last_error = None;
        let mut sent = false;

        for uplink_id in &selected {
            if let Some(uplink) = self.get_uplink(*uplink_id) {
                match self.send_on_uplink(&uplink, seq, data).await {
                    Ok(_) => {
                        sent = true;
                        // For non-redundant strategies, stop after first success
                        if self.config.scheduler.strategy != super::SchedulingStrategy::Redundant {
                            break;
                        }
                    }
                    Err(e) => {
                        uplink.record_failure(&e.to_string());
                        last_error = Some(e);
                    }
                }
            }
        }

        if sent {
            let mut stats = self.stats.write();
            stats.bytes_sent += data.len() as u64;
            stats.packets_sent += 1;
            Ok(seq.0)
        } else if let Some(e) = last_error {
            Err(e)
        } else {
            Err(Error::NoAvailableUplinks)
        }
    }

    /// Send packet on a specific uplink.
    async fn send_on_uplink(&self, uplink: &Uplink, seq: SequenceNumber, data: &[u8]) -> Result<()> {
        // Encrypt the payload using the uplink's noise session
        let encrypted_payload = uplink.encrypt(data)?;

        // Create packet with encrypted payload
        let mut packet = Packet::data(
            seq,
            self.session_id,
            uplink.numeric_id(),
            encrypted_payload,
        )?;

        // Mark as encrypted
        packet.set_flag(PacketFlags::ENCRYPTED);

        let encoded = packet.encode()?;

        // Get transport and send
        let transport = uplink.get_transport()
            .ok_or_else(|| Error::Transport(crate::error::TransportError::SendFailed(
                "no transport".into()
            )))?;

        transport.send(&encoded).await?;
        uplink.record_send(encoded.len());

        tracing::trace!(
            uplink = %uplink.id(),
            seq = seq.0,
            len = data.len(),
            "Sent encrypted packet"
        );

        // Track pending packet for retry (store original data for potential retransmission)
        self.pending.insert(seq.0, PendingPacket {
            packet: Packet::data(seq, self.session_id, uplink.numeric_id(), data.to_vec())?,
            sent_at: Instant::now(),
            retries: 0,
            uplink_id: uplink.numeric_id(),
        });

        Ok(())
    }

    /// Handle incoming packet.
    pub fn handle_packet(&self, data: &[u8], from_uplink: u16) -> Result<Option<Vec<u8>>> {
        let packet = Packet::decode(data)?;

        // Verify session
        if packet.header.session_id != self.session_id {
            return Err(Error::Protocol(crate::error::ProtocolError::InvalidVersion {
                expected: 0,
                got: 1,
            }));
        }

        // Deduplication using O(1) sliding window
        if self.config.deduplication {
            let mut dedup = self.dedup_window.write();
            if dedup.is_duplicate(packet.header.sequence.0) {
                return Ok(None); // Duplicate
            }
        }

        // Update uplink stats
        if let Some(uplink) = self.get_uplink(from_uplink) {
            uplink.record_recv(data.len());
            uplink.record_success();
        }

        // Update aggregate stats
        {
            let mut stats = self.stats.write();
            stats.bytes_received += data.len() as u64;
            stats.packets_received += 1;
        }

        // Handle by packet type
        match packet.header.packet_type {
            PacketType::Data => {
                // Decrypt if encrypted
                let payload = if packet.header.flags.has(PacketFlags::ENCRYPTED) {
                    if let Some(uplink) = self.get_uplink(from_uplink) {
                        uplink.decrypt(&packet.payload)?
                    } else {
                        packet.payload.clone()
                    }
                } else {
                    packet.payload.clone()
                };

                let decrypted_packet = Packet::data(
                    packet.header.sequence,
                    packet.header.session_id,
                    packet.header.uplink_id,
                    payload.clone(),
                )?;

                let _ = self.event_tx.send(MultipathEvent::PacketReceived(decrypted_packet));
                Ok(Some(payload))
            }
            PacketType::Ack => {
                self.handle_ack(&packet)?;
                Ok(None)
            }
            PacketType::Pong => {
                self.handle_pong(&packet, from_uplink);
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Receive data from any uplink.
    /// Returns the decrypted payload and the uplink ID it came from.
    pub async fn recv(&self) -> Result<(Vec<u8>, u16)> {
        let mut buf = vec![0u8; 65536];

        // Try to receive from any connected uplink
        let uplinks: Vec<_> = self.uplinks.iter()
            .filter(|r| r.value().is_usable())
            .map(|r| r.value().clone())
            .collect();

        if uplinks.is_empty() {
            return Err(Error::NoAvailableUplinks);
        }

        // Simple approach: try each uplink in sequence with a short timeout
        // A more sophisticated approach would use select! to wait on all simultaneously
        for uplink in &uplinks {
            if let Some(transport) = uplink.get_transport() {
                match tokio::time::timeout(
                    Duration::from_millis(100),
                    transport.recv(&mut buf)
                ).await {
                    Ok(Ok(len)) => {
                        // Decode and handle the packet
                        if let Some(payload) = self.handle_packet(&buf[..len], uplink.numeric_id())? {
                            return Ok((payload, uplink.numeric_id()));
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::debug!(uplink = %uplink.id(), error = %e, "Receive error");
                    }
                    Err(_) => {
                        // Timeout, try next uplink
                    }
                }
            }
        }

        Err(Error::NoAvailableUplinks)
    }

    /// Start the receive loop in a background task.
    /// Returns a channel to receive decrypted data from.
    pub fn start_recv_loop(self: &Arc<Self>) -> mpsc::Receiver<(Vec<u8>, u16)> {
        let (tx, rx) = mpsc::channel(256);
        let manager = Arc::clone(self);

        tokio::spawn(async move {
            let mut buf = vec![0u8; 65536];

            loop {
                if *manager.state.read() != ConnectionState::Connected {
                    break;
                }

                // Get usable uplinks
                let uplinks: Vec<_> = manager.uplinks.iter()
                    .filter(|r| r.value().is_usable())
                    .map(|r| r.value().clone())
                    .collect();

                if uplinks.is_empty() {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }

                // Try to receive from any uplink
                for uplink in &uplinks {
                    if let Some(transport) = uplink.get_transport() {
                        match tokio::time::timeout(
                            Duration::from_millis(50),
                            transport.recv(&mut buf)
                        ).await {
                            Ok(Ok(len)) => {
                                match manager.handle_packet(&buf[..len], uplink.numeric_id()) {
                                    Ok(Some(payload)) => {
                                        if tx.send((payload, uplink.numeric_id())).await.is_err() {
                                            return; // Channel closed
                                        }
                                    }
                                    Ok(None) => {} // Control packet, no data
                                    Err(e) => {
                                        tracing::debug!(
                                            uplink = %uplink.id(),
                                            error = %e,
                                            "Packet handling error"
                                        );
                                    }
                                }
                            }
                            Ok(Err(e)) => {
                                tracing::debug!(uplink = %uplink.id(), error = %e, "Receive error");
                                uplink.record_failure(&e.to_string());
                            }
                            Err(_) => {} // Timeout, try next
                        }
                    }
                }
            }
        });

        rx
    }

    /// Handle ACK packet.
    fn handle_ack(&self, packet: &Packet) -> Result<()> {
        let acked: Vec<u64> = bincode::deserialize(&packet.payload)
            .map_err(|e| crate::error::ProtocolError::Deserialization(e.to_string()))?;

        for seq in acked {
            if let Some((_, pending)) = self.pending.remove(&seq) {
                // Calculate RTT
                let rtt = pending.sent_at.elapsed();
                if let Some(uplink) = self.get_uplink(pending.uplink_id) {
                    uplink.record_rtt(rtt);
                }
            }
        }

        Ok(())
    }

    /// Handle pong packet.
    fn handle_pong(&self, packet: &Packet, from_uplink: u16) {
        if packet.payload.len() >= 8 {
            let ping_ts = u64::from_be_bytes(packet.payload[..8].try_into().unwrap());
            let now_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;

            if now_us > ping_ts {
                let rtt = Duration::from_micros(now_us - ping_ts);
                if let Some(uplink) = self.get_uplink(from_uplink) {
                    uplink.record_rtt(rtt);
                }
            }
        }
    }

    /// Retry pending packets.
    pub async fn retry_pending(&self) {
        let now = Instant::now();
        let mut to_retry: Vec<(u64, Vec<u8>, u16)> = Vec::new();
        let mut to_fail = Vec::new();

        // Find packets that need retry
        for entry in &self.pending {
            let pending = entry.value();
            let uplink = self.get_uplink(pending.uplink_id);

            let rto = uplink.as_ref()
                .map_or(Duration::from_millis(500), |u| Duration::from_millis((u.rtt().as_millis() as u64 * 2).max(100)));

            if now.duration_since(pending.sent_at) > rto {
                if pending.retries < self.config.send_retries {
                    // Clone data for retry
                    to_retry.push((*entry.key(), pending.packet.payload.clone(), pending.uplink_id));
                } else {
                    to_fail.push(*entry.key());
                }
            }
        }

        // Retry packets on different uplinks
        for (seq, data, old_uplink_id) in to_retry {
            // Record loss on previous uplink
            if let Some(old_uplink) = self.get_uplink(old_uplink_id) {
                old_uplink.record_loss();
            }

            // Select new uplink (prefer different one from the failed one)
            let uplinks = self.uplinks();
            let selected = self.scheduler.select(&uplinks, Some(seq));

            // Try to find an uplink that's different from the one that failed
            let new_uplink_id = selected.iter()
                .find(|&&id| id != old_uplink_id)
                .copied()
                .or_else(|| selected.first().copied());

            if let Some(new_uplink_id) = new_uplink_id {
                if let Some(uplink) = self.get_uplink(new_uplink_id) {
                    // Actually send the packet through the new uplink
                    match self.send_on_uplink(&uplink, SequenceNumber(seq), &data).await {
                        Ok(_) => {
                            tracing::debug!(
                                seq = seq,
                                old_uplink = old_uplink_id,
                                new_uplink = new_uplink_id,
                                "Retransmitted packet on different uplink"
                            );

                            // Update pending entry
                            if let Some(mut entry) = self.pending.get_mut(&seq) {
                                entry.retries += 1;
                                entry.uplink_id = new_uplink_id;
                                entry.sent_at = Instant::now();
                            }

                            // Record as retransmission
                            let mut stats = self.stats.write();
                            stats.packets_retransmitted += 1;
                        }
                        Err(e) => {
                            tracing::warn!(
                                seq = seq,
                                uplink = %uplink.id(),
                                error = %e,
                                "Retry failed"
                            );
                            uplink.record_failure(&e.to_string());
                        }
                    }
                }
            }
        }

        // Fail packets that exceeded retries
        for seq in to_fail {
            if let Some((_, pending)) = self.pending.remove(&seq) {
                let mut stats = self.stats.write();
                stats.packets_dropped += 1;

                if let Some(uplink) = self.get_uplink(pending.uplink_id) {
                    uplink.record_loss();
                }

                tracing::warn!(seq = seq, "Packet failed after {} retries", self.config.send_retries);
                let _ = self.event_tx.send(MultipathEvent::SendFailed(seq));
            }
        }
    }

    /// Periodic maintenance.
    pub fn periodic_maintenance(&self) {
        // Update all uplinks
        for entry in &self.uplinks {
            entry.value().periodic_update();
        }

        // Check for all uplinks down
        let any_usable = self.uplinks.iter().any(|r| r.value().is_usable());
        if !any_usable && *self.state.read() == ConnectionState::Connected {
            let _ = self.event_tx.send(MultipathEvent::AllUplinksDown);
        }

        // Cleanup scheduler state
        self.scheduler.cleanup();
    }

    /// Start the maintenance loop in a background task.
    /// This handles retries, keepalives, and uplink health monitoring.
    pub fn start_maintenance_loop(self: &Arc<Self>) {
        let manager = Arc::clone(self);

        tokio::spawn(async move {
            let mut retry_interval = tokio::time::interval(Duration::from_millis(100));
            let mut maintenance_interval = tokio::time::interval(manager.config.health_check_interval);
            let mut ping_interval = tokio::time::interval(Duration::from_secs(5));

            loop {
                if *manager.state.read() != ConnectionState::Connected {
                    break;
                }

                tokio::select! {
                    _ = retry_interval.tick() => {
                        manager.retry_pending().await;
                    }
                    _ = maintenance_interval.tick() => {
                        manager.periodic_maintenance();
                    }
                    _ = ping_interval.tick() => {
                        // Send ping on all uplinks for quality measurement
                        for uplink_ref in &manager.uplinks {
                            let uplink = uplink_ref.value();
                            if uplink.is_usable() {
                                if let Some(transport) = uplink.get_transport() {
                                    let seq = SequenceNumber(manager.next_seq.fetch_add(1, Ordering::SeqCst));
                                    if let Ok(ping) = Packet::ping(seq, manager.session_id, uplink.numeric_id()) {
                                        if let Ok(encoded) = ping.encode() {
                                            let _ = transport.send(&encoded).await;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /// Close the connection.
    pub fn close(&self) -> Result<()> {
        *self.state.write() = ConnectionState::Disconnecting;

        // Close all uplinks
        for entry in &self.uplinks {
            entry.value().set_connection_state(ConnectionState::Disconnecting);
        }

        self.pending.clear();
        *self.state.write() = ConnectionState::Disconnected;
        let _ = self.event_tx.send(MultipathEvent::Disconnected);

        Ok(())
    }

    /// Get quality summary.
    pub fn quality_summary(&self) -> QualitySummary {
        let uplinks: Vec<_> = self.uplinks.iter()
            .map(|r| r.value().clone())
            .collect();

        let usable = uplinks.iter().filter(|u| u.is_usable()).count();
        let total = uplinks.len();

        let avg_rtt = if uplinks.is_empty() {
            Duration::ZERO
        } else {
            let sum: Duration = uplinks.iter().map(|u| u.rtt()).sum();
            sum / uplinks.len() as u32
        };

        let avg_loss = if uplinks.is_empty() {
            0.0
        } else {
            let sum: f64 = uplinks.iter().map(|u| u.loss_ratio()).sum();
            sum / uplinks.len() as f64
        };

        let total_bandwidth = uplinks.iter()
            .filter(|u| u.is_usable())
            .map(|u| u.bandwidth())
            .fold(crate::types::Bandwidth::ZERO, |acc, b| {
                crate::types::Bandwidth::from_bps(acc.bytes_per_sec + b.bytes_per_sec)
            });

        QualitySummary {
            usable_uplinks: usable,
            total_uplinks: total,
            avg_rtt,
            avg_loss,
            total_bandwidth,
            stats: *self.stats.read(),
        }
    }
}

/// Quality summary.
#[derive(Debug, Clone)]
pub struct QualitySummary {
    pub usable_uplinks: usize,
    pub total_uplinks: usize,
    pub avg_rtt: Duration,
    pub avg_loss: f64,
    pub total_bandwidth: crate::types::Bandwidth,
    pub stats: TrafficStats,
}

// Intentionally abbreviated Debug output - internal state not useful for debugging
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for MultipathManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultipathManager")
            .field("session_id", &self.session_id)
            .field("connection_id", &self.connection_id)
            .field("state", &*self.state.read())
            .field("uplinks", &self.uplinks.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_new_sequences() {
        let mut window = DeduplicationWindow::new(10);
        assert!(!window.is_duplicate(1));
        assert!(!window.is_duplicate(2));
        assert!(!window.is_duplicate(3));
    }

    #[test]
    fn test_dedup_detects_duplicates() {
        let mut window = DeduplicationWindow::new(10);
        assert!(!window.is_duplicate(1));
        assert!(window.is_duplicate(1)); // Duplicate
        assert!(!window.is_duplicate(2));
        assert!(window.is_duplicate(2)); // Duplicate
    }

    #[test]
    fn test_dedup_window_cleanup() {
        let mut window = DeduplicationWindow::new(3);
        assert!(!window.is_duplicate(1));
        assert!(!window.is_duplicate(2));
        assert!(!window.is_duplicate(3));
        // Window full: [1, 2, 3]
        assert!(!window.is_duplicate(4)); // Pushes out 1
        // Window now: [2, 3, 4]
        assert!(!window.is_duplicate(1)); // 1 is no longer in window, so not duplicate
        assert!(window.is_duplicate(3));  // Still in window
        assert!(window.is_duplicate(4));  // Still in window
    }

    #[test]
    fn test_dedup_out_of_order() {
        let mut window = DeduplicationWindow::new(10);
        assert!(!window.is_duplicate(5));
        assert!(!window.is_duplicate(3));
        assert!(!window.is_duplicate(7));
        assert!(!window.is_duplicate(1));
        // All unique, should detect duplicates regardless of order
        assert!(window.is_duplicate(5));
        assert!(window.is_duplicate(3));
        assert!(window.is_duplicate(7));
        assert!(window.is_duplicate(1));
    }
}
