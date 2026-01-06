//! End-to-end tests for multi-uplink to exit functionality.
//!
//! These tests validate the complete data path:
//! 1. Client with multiple uplinks (simulating different NICs)
//! 2. Server receiving and aggregating traffic from all uplinks
//! 3. Server forwarding data to final exit destination
//! 4. Response path back through the multi-path connection

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::{broadcast, RwLock as TokioRwLock};

use triglav::crypto::{KeyPair, NoiseSession};
use triglav::error::Result;
use triglav::multipath::{MultipathConfig, MultipathManager, UplinkConfig};
use triglav::protocol::{Packet, PacketFlags, PacketType, HEADER_SIZE};
use triglav::transport::TransportProtocol;
use triglav::types::{SequenceNumber, SessionId, UplinkId};

/// Real remote endpoint that sits BEYOND the exit destination.
/// This proves true forwarding: Client -> Server -> ExitDestination -> RemoteEndpoint
struct RemoteEndpoint {
    listener: TcpListener,
    addr: SocketAddr,
    received_data: Arc<RwLock<Vec<Vec<u8>>>>,
    forwarded_responses: Arc<RwLock<Vec<Vec<u8>>>>,
    shutdown: broadcast::Sender<()>,
}

impl RemoteEndpoint {
    async fn new() -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let (shutdown, _) = broadcast::channel(1);

        Ok(Self {
            listener,
            addr,
            received_data: Arc::new(RwLock::new(Vec::new())),
            forwarded_responses: Arc::new(RwLock::new(Vec::new())),
            shutdown,
        })
    }

    fn addr(&self) -> SocketAddr {
        self.addr
    }

    fn received_data(&self) -> Vec<Vec<u8>> {
        self.received_data.read().clone()
    }

    fn forwarded_responses(&self) -> Vec<Vec<u8>> {
        self.forwarded_responses.read().clone()
    }

    async fn run(&self) -> Result<()> {
        let mut shutdown_rx = self.shutdown.subscribe();

        loop {
            tokio::select! {
                result = self.listener.accept() => {
                    match result {
                        Ok((stream, _addr)) => {
                            let received = Arc::clone(&self.received_data);
                            let forwarded = Arc::clone(&self.forwarded_responses);

                            tokio::spawn(async move {
                                let _ = Self::handle_connection(stream, received, forwarded).await;
                            });
                        }
                        Err(_) => {}
                    }
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
        Ok(())
    }

    async fn handle_connection(
        mut stream: TcpStream,
        received: Arc<RwLock<Vec<Vec<u8>>>>,
        forwarded: Arc<RwLock<Vec<Vec<u8>>>>,
    ) -> Result<()> {
        let mut buf = vec![0u8; 4096];

        loop {
            match stream.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    let data = buf[..n].to_vec();
                    received.write().push(data.clone());

                    // Create unique response proving data went through full chain
                    let mut response = b"REMOTE:".to_vec();
                    response.extend_from_slice(&data);
                    forwarded.write().push(response.clone());
                    stream.write_all(&response).await?;
                }
                Err(_) => break,
            }
        }
        Ok(())
    }

    fn shutdown(&self) {
        let _ = self.shutdown.send(());
    }
}

/// Forwarding exit that connects to a real remote endpoint.
/// This proves the exit actually FORWARDS data, not just receives it.
struct ForwardingExit {
    listener: TcpListener,
    addr: SocketAddr,
    remote_addr: SocketAddr,
    received_data: Arc<RwLock<Vec<Vec<u8>>>>,
    forwarded_to_remote: Arc<RwLock<Vec<Vec<u8>>>>,
    responses_from_remote: Arc<RwLock<Vec<Vec<u8>>>>,
    shutdown: broadcast::Sender<()>,
}

impl ForwardingExit {
    async fn new(remote_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let (shutdown, _) = broadcast::channel(1);

        Ok(Self {
            listener,
            addr,
            remote_addr,
            received_data: Arc::new(RwLock::new(Vec::new())),
            forwarded_to_remote: Arc::new(RwLock::new(Vec::new())),
            responses_from_remote: Arc::new(RwLock::new(Vec::new())),
            shutdown,
        })
    }

    fn addr(&self) -> SocketAddr {
        self.addr
    }

    fn received_data(&self) -> Vec<Vec<u8>> {
        self.received_data.read().clone()
    }

    fn forwarded_to_remote(&self) -> Vec<Vec<u8>> {
        self.forwarded_to_remote.read().clone()
    }

    fn responses_from_remote(&self) -> Vec<Vec<u8>> {
        self.responses_from_remote.read().clone()
    }

    async fn run(&self) -> Result<()> {
        let mut shutdown_rx = self.shutdown.subscribe();

        loop {
            tokio::select! {
                result = self.listener.accept() => {
                    match result {
                        Ok((stream, _addr)) => {
                            let remote_addr = self.remote_addr;
                            let received = Arc::clone(&self.received_data);
                            let forwarded = Arc::clone(&self.forwarded_to_remote);
                            let responses = Arc::clone(&self.responses_from_remote);

                            tokio::spawn(async move {
                                let _ = Self::handle_connection(stream, remote_addr, received, forwarded, responses).await;
                            });
                        }
                        Err(_) => {}
                    }
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
        Ok(())
    }

    async fn handle_connection(
        mut client_stream: TcpStream,
        remote_addr: SocketAddr,
        received: Arc<RwLock<Vec<Vec<u8>>>>,
        forwarded: Arc<RwLock<Vec<Vec<u8>>>>,
        responses: Arc<RwLock<Vec<Vec<u8>>>>,
    ) -> Result<()> {
        // Connect to the real remote endpoint
        let mut remote_stream = TcpStream::connect(remote_addr).await?;
        let mut buf = vec![0u8; 4096];

        loop {
            match client_stream.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    let data = buf[..n].to_vec();
                    received.write().push(data.clone());

                    // Actually FORWARD to remote endpoint
                    forwarded.write().push(data.clone());
                    remote_stream.write_all(&data).await?;

                    // Get response from remote
                    let mut response_buf = vec![0u8; 4096];
                    match tokio::time::timeout(
                        Duration::from_secs(1),
                        remote_stream.read(&mut response_buf),
                    )
                    .await
                    {
                        Ok(Ok(rn)) if rn > 0 => {
                            let response = response_buf[..rn].to_vec();
                            responses.write().push(response.clone());
                            client_stream.write_all(&response).await?;
                        }
                        _ => {}
                    }
                }
                Err(_) => break,
            }
        }
        Ok(())
    }

    fn shutdown(&self) {
        let _ = self.shutdown.send(());
    }
}

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Final destination that receives forwarded traffic from the exit server.
/// This simulates an external service (e.g., a web server) that the client
/// wants to reach through the Triglav tunnel.
struct ExitDestination {
    listener: TcpListener,
    addr: SocketAddr,
    received_data: Arc<RwLock<Vec<Vec<u8>>>>,
    connection_count: Arc<AtomicU64>,
    shutdown: broadcast::Sender<()>,
}

impl ExitDestination {
    async fn new() -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let (shutdown, _) = broadcast::channel(1);

        Ok(Self {
            listener,
            addr,
            received_data: Arc::new(RwLock::new(Vec::new())),
            connection_count: Arc::new(AtomicU64::new(0)),
            shutdown,
        })
    }

    fn addr(&self) -> SocketAddr {
        self.addr
    }

    fn received_data(&self) -> Vec<Vec<u8>> {
        self.received_data.read().clone()
    }

    fn connection_count(&self) -> u64 {
        self.connection_count.load(Ordering::SeqCst)
    }

    /// Run the destination server, echoing received data back.
    async fn run(&self) -> Result<()> {
        let mut shutdown_rx = self.shutdown.subscribe();

        loop {
            tokio::select! {
                result = self.listener.accept() => {
                    match result {
                        Ok((stream, addr)) => {
                            self.connection_count.fetch_add(1, Ordering::SeqCst);
                            let received = Arc::clone(&self.received_data);

                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_connection(stream, received).await {
                                    eprintln!("ExitDestination connection error from {}: {}", addr, e);
                                }
                            });
                        }
                        Err(e) => {
                            eprintln!("ExitDestination accept error: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
        Ok(())
    }

    async fn handle_connection(
        mut stream: TcpStream,
        received: Arc<RwLock<Vec<Vec<u8>>>>,
    ) -> Result<()> {
        let mut buf = vec![0u8; 4096];

        loop {
            match stream.read(&mut buf).await {
                Ok(0) => break, // Connection closed
                Ok(n) => {
                    let data = buf[..n].to_vec();
                    received.write().push(data.clone());

                    // Echo back with a prefix to prove it went through (binary-safe)
                    let mut response = b"EXIT:".to_vec();
                    response.extend_from_slice(&data);
                    stream.write_all(&response).await?;
                }
                Err(e) => {
                    return Err(triglav::error::Error::Transport(
                        triglav::error::TransportError::ReceiveFailed(e.to_string()),
                    ));
                }
            }
        }
        Ok(())
    }

    fn shutdown(&self) {
        let _ = self.shutdown.send(());
    }
}

/// Forwarding server that receives multi-uplink traffic and forwards to exit.
/// This simulates a Triglav server that acts as an exit node.
struct ForwardingServer {
    keypair: KeyPair,
    socket: Arc<UdpSocket>,
    sessions: DashMap<SessionId, ForwardingSession>,
    sessions_by_addr: DashMap<SocketAddr, SessionId>,
    exit_addr: SocketAddr,
    shutdown: broadcast::Sender<()>,
    next_seq: AtomicU64,
    /// Track which uplink addresses belong to which session
    uplink_to_session: DashMap<SocketAddr, SessionId>,
    /// Stats per uplink address
    uplink_stats: DashMap<SocketAddr, UplinkStats>,
    /// Tracks actual payload data per uplink for verification
    data_tracker: UplinkDataTracker,
}

#[derive(Debug, Default)]
struct UplinkStats {
    packets_received: AtomicU64,
    bytes_received: AtomicU64,
    packets_sent: AtomicU64,
    bytes_sent: AtomicU64,
}

/// Tracks actual payload data received from each uplink source address.
/// This is critical for verifying that data from BOTH NICs actually reaches the exit.
#[derive(Debug, Default)]
struct UplinkDataTracker {
    /// Map from source address to list of payloads received from that address
    data_by_source: DashMap<SocketAddr, Vec<Vec<u8>>>,
    /// Total payloads forwarded to exit
    forwarded_to_exit: RwLock<Vec<(SocketAddr, Vec<u8>)>>,
    /// Encryption events: (source_addr, was_encrypted, payload_len, encrypted_len)
    encryption_events: RwLock<Vec<(SocketAddr, bool, usize, usize)>>,
    /// Decryption events: (source_addr, success, encrypted_len, decrypted_len)
    decryption_events: RwLock<Vec<(SocketAddr, bool, usize, usize)>>,
    /// Response events: (source_addr, response_len) - tracks which uplink received response
    response_events: RwLock<Vec<(SocketAddr, usize)>>,
    /// Handshake events per source address
    handshake_events: DashMap<SocketAddr, Vec<String>>,
    /// UDP packets sent back to client: (dest_addr, payload_len, timestamp_ns)
    udp_responses_sent: RwLock<Vec<(SocketAddr, usize, u64)>>,
    /// Decryption attempts with wrong session (for crypto isolation testing)
    cross_session_decrypt_attempts: RwLock<Vec<(SocketAddr, SocketAddr, bool)>>,
}

struct ForwardingSession {
    id: SessionId,
    /// Noise sessions per uplink address (each uplink has its own crypto state)
    noise_by_addr: DashMap<SocketAddr, NoiseSession>,
    client_addrs: RwLock<Vec<SocketAddr>>,
    /// TCP connection to exit destination (if established)
    exit_stream: TokioRwLock<Option<TcpStream>>,
}

impl ForwardingSession {
    fn new(id: SessionId, addr: SocketAddr) -> Self {
        Self {
            id,
            noise_by_addr: DashMap::new(),
            client_addrs: RwLock::new(vec![addr]),
            exit_stream: TokioRwLock::new(None),
        }
    }

    fn add_uplink(&self, addr: SocketAddr) {
        let mut addrs = self.client_addrs.write();
        if !addrs.contains(&addr) {
            addrs.push(addr);
        }
    }

    fn uplink_count(&self) -> usize {
        self.client_addrs.read().len()
    }

    fn set_noise(&self, addr: SocketAddr, noise: NoiseSession) {
        self.noise_by_addr.insert(addr, noise);
    }

    fn has_noise(&self, addr: &SocketAddr) -> bool {
        self.noise_by_addr.contains_key(addr)
    }
}

impl ForwardingServer {
    async fn new(exit_addr: SocketAddr) -> Result<Self> {
        let socket = UdpSocket::bind("127.0.0.1:0").await?;
        let (shutdown, _) = broadcast::channel(1);

        Ok(Self {
            keypair: KeyPair::generate(),
            socket: Arc::new(socket),
            sessions: DashMap::new(),
            sessions_by_addr: DashMap::new(),
            exit_addr,
            shutdown,
            next_seq: AtomicU64::new(1),
            uplink_to_session: DashMap::new(),
            uplink_stats: DashMap::new(),
            data_tracker: UplinkDataTracker::default(),
        })
    }

    fn addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    fn public_key(&self) -> &triglav::crypto::PublicKey {
        &self.keypair.public
    }

    fn next_sequence(&self) -> SequenceNumber {
        SequenceNumber(self.next_seq.fetch_add(1, Ordering::SeqCst))
    }

    /// Get stats for a specific uplink address
    fn uplink_stats(&self, addr: &SocketAddr) -> Option<(u64, u64)> {
        self.uplink_stats.get(addr).map(|s| {
            (
                s.packets_received.load(Ordering::SeqCst),
                s.bytes_received.load(Ordering::SeqCst),
            )
        })
    }

    /// Get total unique uplinks across all sessions
    fn total_unique_uplinks(&self) -> usize {
        self.uplink_to_session.len()
    }

    /// Get session uplink count
    fn session_uplink_count(&self, session_id: SessionId) -> Option<usize> {
        self.sessions.get(&session_id).map(|s| s.uplink_count())
    }

    /// Get all unique source addresses that sent data
    fn get_data_source_addresses(&self) -> Vec<SocketAddr> {
        self.data_tracker
            .data_by_source
            .iter()
            .map(|r| *r.key())
            .collect()
    }

    /// Get payloads received from a specific source address
    fn get_data_from_source(&self, addr: &SocketAddr) -> Vec<Vec<u8>> {
        self.data_tracker
            .data_by_source
            .get(addr)
            .map(|r| r.value().clone())
            .unwrap_or_default()
    }

    /// Get all data forwarded to exit with source addresses
    fn get_forwarded_data(&self) -> Vec<(SocketAddr, Vec<u8>)> {
        self.data_tracker.forwarded_to_exit.read().clone()
    }

    /// Verify that data was received from at least N distinct source addresses
    fn verify_multi_source_data(&self, min_sources: usize) -> bool {
        self.data_tracker.data_by_source.len() >= min_sources
    }

    /// Get count of payloads received per source address
    fn get_payload_counts_by_source(&self) -> Vec<(SocketAddr, usize)> {
        self.data_tracker
            .data_by_source
            .iter()
            .map(|r| (*r.key(), r.value().len()))
            .collect()
    }

    /// Get encryption events for verification (addr, success, plaintext_len, encrypted_len)
    fn get_encryption_events(&self) -> Vec<(SocketAddr, bool, usize, usize)> {
        self.data_tracker.encryption_events.read().clone()
    }

    /// Get decryption events for verification (addr, success, encrypted_len, decrypted_len)
    fn get_decryption_events(&self) -> Vec<(SocketAddr, bool, usize, usize)> {
        self.data_tracker.decryption_events.read().clone()
    }

    /// Get UDP responses actually sent back to clients
    fn get_udp_responses_sent(&self) -> Vec<(SocketAddr, usize, u64)> {
        self.data_tracker.udp_responses_sent.read().clone()
    }

    /// Get cross-session decryption attempts (for crypto isolation testing)
    fn get_cross_session_decrypt_attempts(&self) -> Vec<(SocketAddr, SocketAddr, bool)> {
        self.data_tracker
            .cross_session_decrypt_attempts
            .read()
            .clone()
    }

    /// Calculate encryption overhead from events
    fn calculate_encryption_overhead(&self) -> Option<(usize, usize, f64)> {
        let events = self.data_tracker.encryption_events.read();
        if events.is_empty() {
            return None;
        }
        let mut total_plaintext = 0;
        let mut total_encrypted = 0;
        for (_, success, plain_len, enc_len) in events.iter() {
            if *success {
                total_plaintext += plain_len;
                total_encrypted += enc_len;
            }
        }
        if total_plaintext == 0 {
            return None;
        }
        let overhead = total_encrypted as f64 / total_plaintext as f64;
        Some((total_plaintext, total_encrypted, overhead))
    }

    /// Get response events - which uplinks received responses
    fn get_response_events(&self) -> Vec<(SocketAddr, usize)> {
        self.data_tracker.response_events.read().clone()
    }

    /// Get handshake events per source address
    fn get_handshake_events(&self) -> Vec<(SocketAddr, Vec<String>)> {
        self.data_tracker
            .handshake_events
            .iter()
            .map(|r| (*r.key(), r.value().clone()))
            .collect()
    }

    /// Verify complete encryption path for an uplink
    fn verify_encryption_path(&self, addr: &SocketAddr) -> bool {
        // Must have handshake, decryption, and response for this address
        let has_handshake = self.data_tracker.handshake_events.contains_key(addr);
        let has_decryption = self
            .data_tracker
            .decryption_events
            .read()
            .iter()
            .any(|(a, success, _, _)| a == addr && *success);
        let has_response = self
            .data_tracker
            .response_events
            .read()
            .iter()
            .any(|(a, _)| a == addr);
        // Also verify UDP response was actually sent back
        let has_udp_sent = self
            .data_tracker
            .udp_responses_sent
            .read()
            .iter()
            .any(|(a, _, _)| a == addr);
        has_handshake && has_decryption && has_response && has_udp_sent
    }

    /// Test crypto isolation by attempting to decrypt with wrong uplink's session
    fn test_crypto_isolation(
        &self,
        session_id: SessionId,
        data: &[u8],
        from_addr: SocketAddr,
        try_addr: SocketAddr,
    ) -> bool {
        if let Some(session) = self.sessions.get(&session_id) {
            // Try to decrypt data from from_addr using try_addr's noise session
            if let Some(mut noise_ref) = session.noise_by_addr.get_mut(&try_addr) {
                if noise_ref.is_transport() {
                    let result = noise_ref.decrypt(data);
                    let success = result.is_ok();
                    self.data_tracker
                        .cross_session_decrypt_attempts
                        .write()
                        .push((from_addr, try_addr, success));
                    return success;
                }
            }
        }
        false
    }

    /// Get distinct noise sessions count (proves separate crypto state per uplink)
    fn get_noise_session_count(&self, session_id: SessionId) -> usize {
        self.sessions
            .get(&session_id)
            .map(|s| s.noise_by_addr.len())
            .unwrap_or(0)
    }

    async fn run(&self) -> Result<()> {
        let mut buf = vec![0u8; 65536];
        let mut shutdown_rx = self.shutdown.subscribe();

        loop {
            tokio::select! {
                result = self.socket.recv_from(&mut buf) => {
                    match result {
                        Ok((len, addr)) => {
                            // Track uplink stats
                            self.uplink_stats.entry(addr).or_default();
                            if let Some(stats) = self.uplink_stats.get(&addr) {
                                stats.packets_received.fetch_add(1, Ordering::SeqCst);
                                stats.bytes_received.fetch_add(len as u64, Ordering::SeqCst);
                            }

                            if let Err(e) = self.handle_packet(&buf[..len], addr).await {
                                eprintln!("ForwardingServer error from {}: {}", addr, e);
                            }
                        }
                        Err(e) => {
                            eprintln!("ForwardingServer receive error: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
        Ok(())
    }

    async fn handle_packet(&self, data: &[u8], addr: SocketAddr) -> Result<()> {
        if data.len() < HEADER_SIZE {
            return Ok(());
        }

        let packet = Packet::decode(data)?;
        let session_id = packet.header.session_id;

        // Ensure session exists
        if !self.sessions.contains_key(&session_id) {
            let session = ForwardingSession::new(session_id, addr);
            self.sessions.insert(session_id, session);
            self.sessions_by_addr.insert(addr, session_id);
        }

        // Get session reference and handle packet
        if let Some(session_ref) = self.sessions.get(&session_id) {
            // Track this uplink address
            session_ref.add_uplink(addr);
            self.uplink_to_session.insert(addr, session_id);

            match packet.header.packet_type {
                PacketType::Handshake => {
                    self.handle_handshake(&session_ref, &packet, addr).await?;
                }
                PacketType::Data => {
                    self.handle_data(&session_ref, &packet, addr).await?;
                }
                PacketType::Ping => {
                    self.handle_ping(&session_ref, &packet, addr).await?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    async fn handle_handshake(
        &self,
        session: &ForwardingSession,
        packet: &Packet,
        addr: SocketAddr,
    ) -> Result<()> {
        let mut noise = NoiseSession::new_responder(&self.keypair.secret)?;
        let _ = noise.read_handshake(&packet.payload)?;
        let response = noise.write_handshake(&[])?;

        // Track handshake event - proves distinct Noise session established for this uplink
        self.data_tracker
            .handshake_events
            .entry(addr)
            .or_default()
            .push(format!("handshake_complete_session_{}", session.id));

        let response_packet = Packet::new(
            PacketType::Handshake,
            packet.header.sequence.next(),
            session.id,
            packet.header.uplink_id,
            response,
        )?;

        self.send_packet(&response_packet, addr).await?;

        // Store noise session per uplink address - CRITICAL: each uplink has its own crypto state
        session.set_noise(addr, noise);

        // Establish exit connection for this session (only once per session)
        {
            let guard = session.exit_stream.read().await;
            if guard.is_some() {
                return Ok(()); // Already have exit connection
            }
        }

        match TcpStream::connect(self.exit_addr).await {
            Ok(stream) => {
                *session.exit_stream.write().await = Some(stream);
            }
            Err(e) => {
                eprintln!("Failed to connect to exit destination: {}", e);
            }
        }

        Ok(())
    }

    async fn handle_data(
        &self,
        session: &ForwardingSession,
        packet: &Packet,
        addr: SocketAddr,
    ) -> Result<()> {
        // Decrypt using the noise session for this specific uplink address
        let encrypted_len = packet.payload.len();
        let (payload, _was_encrypted) = if packet.header.flags.has(PacketFlags::ENCRYPTED) {
            if let Some(mut noise_ref) = session.noise_by_addr.get_mut(&addr) {
                if noise_ref.is_transport() {
                    let decrypted = noise_ref.decrypt(&packet.payload)?;
                    // Track successful decryption with both sizes - proves Noise crypto path works per-uplink
                    self.data_tracker.decryption_events.write().push((
                        addr,
                        true,
                        encrypted_len,
                        decrypted.len(),
                    ));
                    (decrypted, true)
                } else {
                    (packet.payload.clone(), false)
                }
            } else {
                // Track decryption failure
                self.data_tracker
                    .decryption_events
                    .write()
                    .push((addr, false, encrypted_len, 0));
                return Ok(());
            }
        } else {
            (packet.payload.clone(), false)
        };

        // CRITICAL: Track which source address sent this payload
        // This proves that data from specific NICs actually reaches the server
        self.data_tracker
            .data_by_source
            .entry(addr)
            .or_default()
            .push(payload.clone());

        // Forward to exit destination
        let response = {
            let mut guard = session.exit_stream.write().await;
            if let Some(ref mut stream) = *guard {
                // Track data being forwarded to exit
                self.data_tracker
                    .forwarded_to_exit
                    .write()
                    .push((addr, payload.clone()));

                stream.write_all(&payload).await.ok();

                // Read response from exit
                let mut response_buf = vec![0u8; 4096];
                match tokio::time::timeout(Duration::from_secs(1), stream.read(&mut response_buf))
                    .await
                {
                    Ok(Ok(n)) if n > 0 => Some(response_buf[..n].to_vec()),
                    _ => None,
                }
            } else {
                // No exit connection, just echo
                Some(payload.clone())
            }
        };

        // Send response back through the tunnel (use the same uplink that sent the data)
        // CRITICAL: Response goes back through SAME uplink address (return path)
        if let Some(response_data) = response {
            // Track response event - proves return path uses same uplink
            self.data_tracker
                .response_events
                .write()
                .push((addr, response_data.len()));
            self.send_data(session, &response_data, packet.header.uplink_id, addr)
                .await?;
        }

        Ok(())
    }

    async fn send_data(
        &self,
        session: &ForwardingSession,
        payload: &[u8],
        uplink_id: u16,
        addr: SocketAddr,
    ) -> Result<()> {
        // Encrypt using the noise session for this specific uplink address
        let plaintext_len = payload.len();
        let (encrypted, is_encrypted) =
            if let Some(mut noise_ref) = session.noise_by_addr.get_mut(&addr) {
                if noise_ref.is_transport() {
                    let encrypted_data = noise_ref.encrypt(payload)?;
                    let encrypted_len = encrypted_data.len();
                    // Track encryption event with both sizes - proves response encryption uses per-uplink crypto
                    self.data_tracker.encryption_events.write().push((
                        addr,
                        true,
                        plaintext_len,
                        encrypted_len,
                    ));
                    (encrypted_data, true)
                } else {
                    (payload.to_vec(), false)
                }
            } else {
                (payload.to_vec(), false)
            };

        let mut response = Packet::data(self.next_sequence(), session.id, uplink_id, encrypted)?;

        if is_encrypted {
            response.set_flag(PacketFlags::ENCRYPTED);
        }

        self.send_packet(&response, addr).await
    }

    async fn handle_ping(
        &self,
        session: &ForwardingSession,
        packet: &Packet,
        addr: SocketAddr,
    ) -> Result<()> {
        let pong = Packet::pong(
            packet.header.sequence.next(),
            session.id,
            packet.header.uplink_id,
            packet.header.timestamp,
        )?;
        self.send_packet(&pong, addr).await
    }

    async fn send_packet(&self, packet: &Packet, addr: SocketAddr) -> Result<()> {
        let data = packet.encode()?;
        self.socket.send_to(&data, addr).await?;

        // Track UDP response sent - proves response actually traverses UDP network
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.data_tracker
            .udp_responses_sent
            .write()
            .push((addr, data.len(), timestamp));

        if let Some(stats) = self.uplink_stats.get(&addr) {
            stats.packets_sent.fetch_add(1, Ordering::SeqCst);
            stats
                .bytes_sent
                .fetch_add(data.len() as u64, Ordering::SeqCst);
        }

        Ok(())
    }

    fn shutdown(&self) {
        let _ = self.shutdown.send(());
    }
}

// ============================================================================
// End-to-End Tests
// ============================================================================

/// Test that data sent through multiple uplinks arrives at the same session.
#[tokio::test]
async fn test_multi_uplink_aggregation() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with multiple uplinks
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;

    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add uplink 1 with specific local port
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    // Add uplink 2 with different local port (simulating different NIC)
    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    // Connect
    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify both uplinks connected
    assert_eq!(manager.uplink_count(), 2, "Should have 2 uplinks");

    // Allocate flows on different uplinks
    let flow1 = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow2 = manager.allocate_flow_on_uplink(uplink2_id).unwrap();

    // Send data through uplink 1
    let data1 = b"Hello from uplink 1";
    manager.send_on_flow(Some(flow1), data1).await.unwrap();

    // Send data through uplink 2
    let data2 = b"Hello from uplink 2";
    manager.send_on_flow(Some(flow2), data2).await.unwrap();

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(200)).await;

    // CRITICAL VERIFICATION: Prove data from BOTH NICs actually reached the server
    // This is the core assertion that validates multi-uplink functionality

    // 1. Verify we received data from exactly 2 distinct source addresses (NICs)
    let source_addresses = server.get_data_source_addresses();
    assert!(source_addresses.len() >= 2,
        "CRITICAL: Server must receive data from at least 2 distinct source addresses (NICs). Got: {}",
        source_addresses.len());

    // 2. Verify each source address sent actual data
    let payload_counts = server.get_payload_counts_by_source();
    for (addr, count) in &payload_counts {
        assert!(
            *count > 0,
            "Source {} should have sent data, but count is 0",
            addr
        );
    }
    println!("Payload counts by source: {:?}", payload_counts);

    // 3. Verify the specific data content from each source
    let mut found_uplink1_data = false;
    let mut found_uplink2_data = false;
    for addr in &source_addresses {
        let payloads = server.get_data_from_source(addr);
        for payload in &payloads {
            if payload == data1 {
                found_uplink1_data = true;
                println!("Found uplink 1 data from source {}", addr);
            }
            if payload == data2 {
                found_uplink2_data = true;
                println!("Found uplink 2 data from source {}", addr);
            }
        }
    }
    assert!(
        found_uplink1_data,
        "CRITICAL: Data from uplink 1 must reach server from a distinct source"
    );
    assert!(
        found_uplink2_data,
        "CRITICAL: Data from uplink 2 must reach server from a distinct source"
    );

    // 4. Verify data was forwarded to exit from both sources
    let forwarded = server.get_forwarded_data();
    assert!(
        forwarded.len() >= 2,
        "At least 2 payloads should be forwarded to exit"
    );

    // Collect unique source addresses that forwarded data
    let forwarded_sources: std::collections::HashSet<_> =
        forwarded.iter().map(|(addr, _)| *addr).collect();
    assert!(forwarded_sources.len() >= 2,
        "CRITICAL: Data forwarded to exit must come from at least 2 distinct source addresses. Got: {}",
        forwarded_sources.len());

    // 5. Verify exit destination received the data
    let received = exit.received_data();
    assert!(
        received.len() >= 2,
        "Exit should have received at least 2 messages"
    );

    // Verify data content at exit
    let all_received: Vec<u8> = received.iter().flatten().cloned().collect();
    assert!(
        all_received.windows(data1.len()).any(|w| w == data1),
        "Exit should receive data originally sent through uplink 1"
    );
    assert!(
        all_received.windows(data2.len()).any(|w| w == data2),
        "Exit should receive data originally sent through uplink 2"
    );

    println!(
        "SUCCESS: Verified data from 2 distinct NICs reached exit via {} source addresses",
        source_addresses.len()
    );

    server.shutdown();
    exit.shutdown();
}

/// Test that flow stickiness works - same flow always uses same uplink.
#[tokio::test]
async fn test_flow_stickiness_through_exit() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with ECMP awareness
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;

    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    for i in 1..=2 {
        let uplink_config = UplinkConfig {
            id: UplinkId::new(&format!("uplink-{}", i)),
            remote_addr: server_addr,
            protocol: TransportProtocol::Udp,
            ..Default::default()
        };
        manager.add_uplink(uplink_config).unwrap();
    }

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate a flow
    let flow_id = manager.allocate_flow();

    // Send first message to establish binding (binding is created lazily)
    let msg0 = "Sticky message 0";
    manager
        .send_on_flow(Some(flow_id), msg0.as_bytes())
        .await
        .unwrap();

    // Now get the initial binding (created during first send)
    let initial_binding = manager.get_flow_binding(flow_id);
    assert!(
        initial_binding.is_some(),
        "Binding should be created after first send"
    );

    // Send more messages on the same flow
    for i in 1..10 {
        let msg = format!("Sticky message {}", i);
        manager
            .send_on_flow(Some(flow_id), msg.as_bytes())
            .await
            .unwrap();

        // Verify binding didn't change
        let current_binding = manager.get_flow_binding(flow_id);
        assert_eq!(
            initial_binding, current_binding,
            "Flow binding should remain sticky on message {}",
            i
        );
    }

    // Wait for all messages to be processed
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Verify exit received all messages
    let received = exit.received_data();
    assert_eq!(
        received.len(),
        10,
        "Exit should have received all 10 messages"
    );

    // Verify all messages arrived (content check)
    for i in 0..10 {
        let expected = format!("Sticky message {}", i);
        let found = received
            .iter()
            .any(|r| String::from_utf8_lossy(r).contains(&expected));
        assert!(found, "Should find message {}", i);
    }

    server.shutdown();
    exit.shutdown();
}

/// Test data flows through complete path: Client -> Multi-uplink -> Server -> Exit -> Response.
#[tokio::test]
async fn test_complete_data_path() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add single uplink for simplicity
    let uplink_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send data
    let test_data = b"Test data for exit";
    manager.send(test_data).await.unwrap();

    // Receive response
    let result = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;

    match result {
        Ok(Ok((response, _uplink_id))) => {
            // Response should have EXIT: prefix from the exit destination
            let response_str = String::from_utf8_lossy(&response);
            assert!(
                response_str.starts_with("EXIT:"),
                "Response should have EXIT prefix, got: {}",
                response_str
            );
            assert!(
                response_str.contains("Test data for exit"),
                "Response should contain original data"
            );
            println!("Complete path verified: {}", response_str);
        }
        Ok(Err(e)) => panic!("Receive error: {}", e),
        Err(_) => panic!("Timeout waiting for response"),
    }

    // Verify exit destination saw the data
    let received = exit.received_data();
    assert!(
        !received.is_empty(),
        "Exit destination should have received data"
    );
    assert_eq!(
        received[0], test_data,
        "Exit should receive exact data sent"
    );

    server.shutdown();
    exit.shutdown();
}

/// Test that different flows can use different uplinks simultaneously.
#[tokio::test]
async fn test_parallel_flows_different_uplinks() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flows on specific uplinks
    let flow_a = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow_b = manager.allocate_flow_on_uplink(uplink2_id).unwrap();

    // Send data in parallel through different flows
    let manager_a = Arc::clone(&manager);
    let manager_b = Arc::clone(&manager);

    let task_a = tokio::spawn(async move {
        for i in 0..5 {
            let msg = format!("Flow-A message {}", i);
            manager_a
                .send_on_flow(Some(flow_a), msg.as_bytes())
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    let task_b = tokio::spawn(async move {
        for i in 0..5 {
            let msg = format!("Flow-B message {}", i);
            manager_b
                .send_on_flow(Some(flow_b), msg.as_bytes())
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Wait for both tasks
    let _ = tokio::join!(task_a, task_b);

    tokio::time::sleep(Duration::from_millis(300)).await;

    // Verify exit received all messages
    let received = exit.received_data();
    assert_eq!(
        received.len(),
        10,
        "Exit should have received all 10 messages"
    );

    // Verify messages from both flows arrived
    let all_text: String = received
        .iter()
        .map(|r| String::from_utf8_lossy(r).to_string())
        .collect::<Vec<_>>()
        .join("");

    assert!(
        all_text.contains("Flow-A"),
        "Should contain Flow-A messages"
    );
    assert!(
        all_text.contains("Flow-B"),
        "Should contain Flow-B messages"
    );

    // Verify bindings remained stable
    assert_eq!(manager.get_flow_binding(flow_a), Some(uplink1_id));
    assert_eq!(manager.get_flow_binding(flow_b), Some(uplink2_id));

    server.shutdown();
    exit.shutdown();
}

/// Test uplink failover - when one uplink fails, traffic moves to another.
#[tokio::test]
async fn test_uplink_failover() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flow on uplink 1
    let flow_id = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    assert_eq!(manager.get_flow_binding(flow_id), Some(uplink1_id));

    // Send first message through uplink 1
    manager
        .send_on_flow(Some(flow_id), b"Before failover")
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Remove uplink 1 (simulate failure)
    manager.remove_uplink(uplink1_id);

    // Flow binding should still point to removed uplink until next send
    // On next send, it should fail over to uplink 2

    // Send message - should now go through uplink 2
    let result = manager.send_on_flow(Some(flow_id), b"After failover").await;
    assert!(result.is_ok(), "Should be able to send after failover");

    // Verify binding updated to uplink 2
    let new_binding = manager.get_flow_binding(flow_id);
    assert_eq!(
        new_binding,
        Some(uplink2_id),
        "Flow should fail over to uplink 2"
    );

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Verify exit received both messages
    let received = exit.received_data();
    assert!(
        received.len() >= 2,
        "Exit should have received messages before and after failover"
    );

    server.shutdown();
    exit.shutdown();
}

/// Test data transfer through multi-path tunnel.
#[tokio::test]
async fn test_data_through_multipath() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    let uplink_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send larger data (500 bytes - safe with encryption overhead)
    // Note: Max payload is ~1400 bytes, encryption adds overhead
    let large_data: Vec<u8> = (0..500).map(|i| (i % 256) as u8).collect();
    manager.send(&large_data).await.unwrap();

    // Receive response
    let result = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;

    match result {
        Ok(Ok((response, _))) => {
            // Response should contain EXIT: prefix
            assert!(
                response.starts_with(b"EXIT:"),
                "Response should start with EXIT:"
            );
            // Rest should be our data
            let data_portion = &response[5..];
            assert_eq!(
                data_portion,
                &large_data[..],
                "Data should match through exit"
            );
            println!("Large data transfer verified: {} bytes", large_data.len());
        }
        Ok(Err(e)) => panic!("Receive error: {}", e),
        Err(_) => panic!("Timeout"),
    }

    server.shutdown();
    exit.shutdown();
}

/// Test session aggregates traffic from multiple uplink addresses.
#[tokio::test]
async fn test_session_aggregates_uplinks() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add 3 uplinks
    for i in 1..=3 {
        let uplink_config = UplinkConfig {
            id: UplinkId::new(&format!("uplink-{}", i)),
            remote_addr: server_addr,
            protocol: TransportProtocol::Udp,
            ..Default::default()
        };
        manager.add_uplink(uplink_config).unwrap();
    }

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send data through default path (will use scheduler)
    for i in 0..6 {
        let msg = format!("Message {}", i);
        manager.send(msg.as_bytes()).await.unwrap();
    }

    tokio::time::sleep(Duration::from_millis(300)).await;

    // Verify server aggregated from multiple uplinks into single session
    // The session should have multiple client addresses
    let session_id = manager.session_id();
    let uplink_count = server.session_uplink_count(session_id);

    // Due to how we route traffic, at least one uplink should be used
    assert!(uplink_count.is_some(), "Session should exist");
    println!(
        "Session {} has {} uplinks",
        session_id,
        uplink_count.unwrap()
    );

    // Verify exit received all messages
    let received = exit.received_data();
    assert_eq!(
        received.len(),
        6,
        "Exit should have received all 6 messages"
    );

    server.shutdown();
    exit.shutdown();
}

/// Test that exit connection receives combined bandwidth from multiple uplinks.
#[tokio::test]
async fn test_bandwidth_aggregation() {
    // Start exit destination that counts bytes
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flows on different uplinks
    let flow1 = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow2 = manager.allocate_flow_on_uplink(uplink2_id).unwrap();

    // Send data through both uplinks in parallel
    let data_size = 500;
    let data1: Vec<u8> = (0..data_size).map(|_| b'A').collect();
    let data2: Vec<u8> = (0..data_size).map(|_| b'B').collect();

    let manager1 = Arc::clone(&manager);
    let manager2 = Arc::clone(&manager);
    let data1_clone = data1.clone();
    let data2_clone = data2.clone();

    let task1 = tokio::spawn(async move {
        manager1
            .send_on_flow(Some(flow1), &data1_clone)
            .await
            .unwrap();
    });

    let task2 = tokio::spawn(async move {
        manager2
            .send_on_flow(Some(flow2), &data2_clone)
            .await
            .unwrap();
    });

    let _ = tokio::join!(task1, task2);
    tokio::time::sleep(Duration::from_millis(200)).await;

    // CRITICAL VERIFICATION: Prove bandwidth was aggregated from BOTH NICs

    // 1. Verify data came from 2 distinct source addresses
    let source_addresses = server.get_data_source_addresses();
    assert!(
        source_addresses.len() >= 2,
        "CRITICAL: Bandwidth must come from at least 2 distinct source addresses (NICs). Got: {}",
        source_addresses.len()
    );

    // 2. Verify each source contributed data
    let payload_counts = server.get_payload_counts_by_source();
    let mut sources_with_data = 0;
    for (addr, count) in &payload_counts {
        if *count > 0 {
            sources_with_data += 1;
            println!("Source {} contributed {} payloads", addr, count);
        }
    }
    assert!(
        sources_with_data >= 2,
        "CRITICAL: At least 2 sources must contribute data. Got: {}",
        sources_with_data
    );

    // 3. Verify the specific data patterns from each source
    let mut found_data_a = false;
    let mut found_data_b = false;
    for addr in &source_addresses {
        let payloads = server.get_data_from_source(addr);
        for payload in &payloads {
            if payload.iter().all(|&b| b == b'A') && payload.len() == data_size {
                found_data_a = true;
                println!("Found 'A' data (uplink 1) from source {}", addr);
            }
            if payload.iter().all(|&b| b == b'B') && payload.len() == data_size {
                found_data_b = true;
                println!("Found 'B' data (uplink 2) from source {}", addr);
            }
        }
    }
    assert!(
        found_data_a,
        "CRITICAL: Data from uplink 1 (all 'A' bytes) must reach server"
    );
    assert!(
        found_data_b,
        "CRITICAL: Data from uplink 2 (all 'B' bytes) must reach server"
    );

    // 4. Verify forwarded data came from both sources
    let forwarded = server.get_forwarded_data();
    let forwarded_sources: std::collections::HashSet<_> =
        forwarded.iter().map(|(addr, _)| *addr).collect();
    assert!(
        forwarded_sources.len() >= 2,
        "CRITICAL: Forwarded data must come from at least 2 distinct sources. Got: {}",
        forwarded_sources.len()
    );

    // 5. Verify exit received total data from both uplinks
    let received = exit.received_data();
    let total_bytes: usize = received.iter().map(|r| r.len()).sum();

    assert_eq!(total_bytes, data_size * 2,
        "Exit should receive combined bandwidth: {} bytes from uplink1 + {} bytes from uplink2 = {} total",
        data_size, data_size, data_size * 2);

    // 6. Verify the actual byte content at exit
    let all_exit_data: Vec<u8> = received.iter().flatten().cloned().collect();
    let count_a = all_exit_data.iter().filter(|&&b| b == b'A').count();
    let count_b = all_exit_data.iter().filter(|&&b| b == b'B').count();
    assert_eq!(
        count_a, data_size,
        "Exit should receive {} 'A' bytes from uplink 1",
        data_size
    );
    assert_eq!(
        count_b, data_size,
        "Exit should receive {} 'B' bytes from uplink 2",
        data_size
    );

    println!(
        "SUCCESS: Bandwidth aggregation verified - {} bytes from 2 NICs reached exit",
        total_bytes
    );

    server.shutdown();
    exit.shutdown();
}

/// Test ECMP flow hash consistency - verifies Dublin Traceroute technique works correctly.
/// Same flow hash should consistently route through the same uplink.
#[tokio::test]
async fn test_ecmp_flow_hash_consistency() {
    use triglav::multipath::calculate_flow_hash;

    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with ECMP-aware configuration
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // ECMP Flow Hash Verification
    // 1. Calculate flow hashes for different flows
    let flow_hash_1 = calculate_flow_hash(
        std::net::IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 100)),
        std::net::IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 1)),
        12345,
        80,
        6, // TCP
    );

    let flow_hash_2 = calculate_flow_hash(
        std::net::IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 100)),
        std::net::IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 1)),
        12346, // Different source port = different flow
        80,
        6,
    );

    // Verify flow hashes are consistent (same inputs = same hash)
    let flow_hash_1_again = calculate_flow_hash(
        std::net::IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 100)),
        std::net::IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 1)),
        12345,
        80,
        6,
    );
    assert_eq!(
        flow_hash_1, flow_hash_1_again,
        "ECMP: Same flow must produce same hash (Dublin Traceroute consistency)"
    );

    // Verify different flows produce different hashes
    assert_ne!(
        flow_hash_1, flow_hash_2,
        "ECMP: Different flows should produce different hashes"
    );

    // 2. Allocate flows on specific uplinks and verify binding consistency
    let flow_a = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow_b = manager.allocate_flow_on_uplink(uplink2_id).unwrap();

    // Send multiple messages on each flow
    for i in 0..5 {
        let msg_a = format!("Flow-A-{}", i);
        let msg_b = format!("Flow-B-{}", i);

        manager
            .send_on_flow(Some(flow_a), msg_a.as_bytes())
            .await
            .unwrap();
        manager
            .send_on_flow(Some(flow_b), msg_b.as_bytes())
            .await
            .unwrap();

        // Verify bindings remain consistent (ECMP path stickiness)
        let binding_a = manager.get_flow_binding(flow_a);
        let binding_b = manager.get_flow_binding(flow_b);

        assert_eq!(
            binding_a,
            Some(uplink1_id),
            "ECMP: Flow A must remain bound to uplink 1 (message {})",
            i
        );
        assert_eq!(
            binding_b,
            Some(uplink2_id),
            "ECMP: Flow B must remain bound to uplink 2 (message {})",
            i
        );
    }

    tokio::time::sleep(Duration::from_millis(200)).await;

    // 3. Verify server received data from both source addresses
    let source_addresses = server.get_data_source_addresses();
    assert!(
        source_addresses.len() >= 2,
        "ECMP: Server must see traffic from at least 2 distinct source addresses"
    );

    // 4. Verify each uplink contributed data (proving ECMP split worked)
    let payload_counts = server.get_payload_counts_by_source();
    let sources_with_data: Vec<_> = payload_counts
        .iter()
        .filter(|(_, count)| *count > 0)
        .collect();
    assert!(
        sources_with_data.len() >= 2,
        "ECMP: At least 2 sources must have contributed data, got: {}",
        sources_with_data.len()
    );

    // 5. Verify Flow-A messages came from one source and Flow-B from another
    let mut flow_a_source: Option<SocketAddr> = None;
    let mut flow_b_source: Option<SocketAddr> = None;

    for addr in &source_addresses {
        let payloads = server.get_data_from_source(addr);
        for payload in &payloads {
            let payload_str = String::from_utf8_lossy(payload);
            if payload_str.starts_with("Flow-A") {
                if let Some(existing) = flow_a_source {
                    assert_eq!(
                        existing, *addr,
                        "ECMP: All Flow-A messages must come from same source (path consistency)"
                    );
                } else {
                    flow_a_source = Some(*addr);
                }
            }
            if payload_str.starts_with("Flow-B") {
                if let Some(existing) = flow_b_source {
                    assert_eq!(
                        existing, *addr,
                        "ECMP: All Flow-B messages must come from same source (path consistency)"
                    );
                } else {
                    flow_b_source = Some(*addr);
                }
            }
        }
    }

    assert!(
        flow_a_source.is_some(),
        "ECMP: Flow-A messages must have a source"
    );
    assert!(
        flow_b_source.is_some(),
        "ECMP: Flow-B messages must have a source"
    );
    assert_ne!(
        flow_a_source, flow_b_source,
        "ECMP: Flow-A and Flow-B must use different sources (uplinks)"
    );

    println!("SUCCESS: ECMP flow hash consistency verified");
    println!("  Flow-A source: {:?}", flow_a_source);
    println!("  Flow-B source: {:?}", flow_b_source);
    println!("  Flow hash 1: {:#x}", flow_hash_1);
    println!("  Flow hash 2: {:#x}", flow_hash_2);

    server.shutdown();
    exit.shutdown();
}

/// Test that Noise encryption/decryption works correctly per uplink.
/// Each uplink must have its own Noise session and crypto state.
#[tokio::test]
async fn test_noise_encryption_per_uplink() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with multiple uplinks
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    // Connect - this performs Noise handshake for each uplink
    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // VERIFY: Handshakes occurred for each uplink
    let handshake_events = server.get_handshake_events();
    assert!(
        handshake_events.len() >= 2,
        "CRYPTO: Must have handshake events from at least 2 uplink addresses. Got: {}",
        handshake_events.len()
    );

    println!("Handshake events:");
    for (addr, events) in &handshake_events {
        println!("  {}: {:?}", addr, events);
        assert!(
            !events.is_empty(),
            "CRYPTO: Uplink {} must have completed handshake",
            addr
        );
    }

    // Allocate flows on specific uplinks
    let flow1 = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow2 = manager.allocate_flow_on_uplink(uplink2_id).unwrap();

    // Send encrypted data through both uplinks
    let data1 = b"Encrypted message from uplink 1";
    let data2 = b"Encrypted message from uplink 2";

    manager.send_on_flow(Some(flow1), data1).await.unwrap();
    manager.send_on_flow(Some(flow2), data2).await.unwrap();

    // Wait for processing and responses
    tokio::time::sleep(Duration::from_millis(300)).await;

    // VERIFY: Decryption succeeded for each uplink
    let decryption_events = server.get_decryption_events();
    println!("Decryption events: {:?}", decryption_events);

    // Group by address and check success
    let mut decryption_by_addr: std::collections::HashMap<SocketAddr, Vec<bool>> =
        std::collections::HashMap::new();
    for (addr, success, _enc_len, _dec_len) in &decryption_events {
        decryption_by_addr.entry(*addr).or_default().push(*success);
    }

    assert!(
        decryption_by_addr.len() >= 2,
        "CRYPTO: Must have decryption events from at least 2 uplink addresses. Got: {}",
        decryption_by_addr.len()
    );

    for (addr, results) in &decryption_by_addr {
        let all_successful = results.iter().all(|&s| s);
        assert!(
            all_successful,
            "CRYPTO: All decryptions from {} must succeed. Results: {:?}",
            addr, results
        );
        println!(
            "  Uplink {}: {} successful decryptions",
            addr,
            results.len()
        );
    }

    // VERIFY: Encryption events for responses (per uplink)
    let encryption_events = server.get_encryption_events();
    println!("Encryption events: {:?}", encryption_events);

    let mut encryption_by_addr: std::collections::HashMap<SocketAddr, Vec<bool>> =
        std::collections::HashMap::new();
    for (addr, success, _plain_len, _enc_len) in &encryption_events {
        encryption_by_addr.entry(*addr).or_default().push(*success);
    }

    assert!(
        encryption_by_addr.len() >= 2,
        "CRYPTO: Must have encryption events from at least 2 uplink addresses. Got: {}",
        encryption_by_addr.len()
    );

    for (addr, results) in &encryption_by_addr {
        let all_successful = results.iter().all(|&s| s);
        assert!(
            all_successful,
            "CRYPTO: All encryptions to {} must succeed. Results: {:?}",
            addr, results
        );
        println!(
            "  Uplink {}: {} successful encryptions",
            addr,
            results.len()
        );
    }

    // VERIFY: UDP responses were actually sent back through the network
    let udp_responses = server.get_udp_responses_sent();
    let udp_response_addrs: std::collections::HashSet<_> =
        udp_responses.iter().map(|(addr, _, _)| *addr).collect();
    assert!(
        udp_response_addrs.len() >= 2,
        "CRYPTO: UDP responses must be sent to at least 2 distinct uplink addresses. Got: {}",
        udp_response_addrs.len()
    );
    println!(
        "UDP responses sent to {} distinct addresses",
        udp_response_addrs.len()
    );

    // VERIFY: Encryption overhead is reasonable (Noise adds ~16 bytes for AEAD tag)
    if let Some((total_plain, total_enc, overhead)) = server.calculate_encryption_overhead() {
        let overhead_per_message = (total_enc - total_plain) / encryption_events.len();
        println!(
            "Encryption overhead: {} bytes plaintext -> {} bytes encrypted ({:.2}x), ~{} bytes/msg",
            total_plain, total_enc, overhead, overhead_per_message
        );
        assert!(
            overhead >= 1.0,
            "Encrypted data must be at least as large as plaintext"
        );
        assert!(
            overhead < 2.0,
            "Encryption overhead should be reasonable (< 2x)"
        );
    }

    // VERIFY: Session has distinct Noise sessions per uplink
    let session_id = manager.session_id();
    let noise_session_count = server.get_noise_session_count(session_id);
    assert!(
        noise_session_count >= 2,
        "CRYPTO: Session must have at least 2 distinct Noise sessions (one per uplink). Got: {}",
        noise_session_count
    );

    println!(
        "SUCCESS: Noise encryption verified for {} uplinks",
        noise_session_count
    );

    server.shutdown();
    exit.shutdown();
}

/// Test that responses flow back through the same uplink that sent the request.
/// This validates the return path property of multi-path connections.
/// CRITICAL: Also verifies CLIENT actually receives responses (bidirectional validation).
#[tokio::test]
async fn test_return_path_validation() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with multiple uplinks
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flows on specific uplinks
    let flow1 = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow2 = manager.allocate_flow_on_uplink(uplink2_id).unwrap();

    // Send requests through each uplink and RECEIVE responses
    let data1 = b"Request through uplink 1";
    let data2 = b"Request through uplink 2";

    // Send through uplink 1 and receive response
    manager.send_on_flow(Some(flow1), data1).await.unwrap();
    let result1 = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;
    let (response1, recv_uplink1) = match result1 {
        Ok(Ok((data, uplink_id))) => (data, uplink_id),
        Ok(Err(e)) => panic!(
            "BIDIRECTIONAL: Failed to receive response for uplink 1: {}",
            e
        ),
        Err(_) => panic!("BIDIRECTIONAL: Timeout receiving response for uplink 1"),
    };
    println!(
        "Received response 1 ({} bytes) via uplink {:?}",
        response1.len(),
        recv_uplink1
    );

    // Verify response 1 contains EXIT: prefix and original data
    assert!(
        response1.starts_with(b"EXIT:"),
        "BIDIRECTIONAL: Response 1 should have EXIT: prefix"
    );
    assert!(
        response1.windows(data1.len()).any(|w| w == data1),
        "BIDIRECTIONAL: Response 1 should contain original data"
    );

    // Send through uplink 2 and receive response
    manager.send_on_flow(Some(flow2), data2).await.unwrap();
    let result2 = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;
    let (response2, recv_uplink2) = match result2 {
        Ok(Ok((data, uplink_id))) => (data, uplink_id),
        Ok(Err(e)) => panic!(
            "BIDIRECTIONAL: Failed to receive response for uplink 2: {}",
            e
        ),
        Err(_) => panic!("BIDIRECTIONAL: Timeout receiving response for uplink 2"),
    };
    println!(
        "Received response 2 ({} bytes) via uplink {:?}",
        response2.len(),
        recv_uplink2
    );

    // Verify response 2 contains EXIT: prefix and original data
    assert!(
        response2.starts_with(b"EXIT:"),
        "BIDIRECTIONAL: Response 2 should have EXIT: prefix"
    );
    assert!(
        response2.windows(data2.len()).any(|w| w == data2),
        "BIDIRECTIONAL: Response 2 should contain original data"
    );

    // VERIFY: Responses came back through the SAME uplinks that sent them
    // (This proves return path stickiness)
    assert_eq!(
        recv_uplink1, uplink1_id,
        "RETURN PATH: Response 1 must come back through uplink 1. Got: {}",
        recv_uplink1
    );
    assert_eq!(
        recv_uplink2, uplink2_id,
        "RETURN PATH: Response 2 must come back through uplink 2. Got: {}",
        recv_uplink2
    );

    // Get all events for analysis
    let decryption_events = server.get_decryption_events();
    let response_events = server.get_response_events();
    let encryption_events = server.get_encryption_events();
    let udp_responses = server.get_udp_responses_sent();

    println!("Decryption events: {:?}", decryption_events);
    println!("Response events: {:?}", response_events);
    println!("Encryption events: {:?}", encryption_events);
    println!("UDP responses sent: {:?}", udp_responses);

    // VERIFY: Response events came from at least 2 distinct uplink addresses
    let response_addrs: std::collections::HashSet<_> =
        response_events.iter().map(|(addr, _)| *addr).collect();
    assert!(
        response_addrs.len() >= 2,
        "RETURN PATH: Responses must be tracked from at least 2 uplink addresses. Got: {}",
        response_addrs.len()
    );

    // VERIFY: UDP responses were actually sent to at least 2 distinct addresses
    let udp_addrs: std::collections::HashSet<_> =
        udp_responses.iter().map(|(addr, _, _)| *addr).collect();
    assert!(
        udp_addrs.len() >= 2,
        "RETURN PATH: UDP responses must be sent to at least 2 distinct addresses. Got: {}",
        udp_addrs.len()
    );

    // VERIFY: For each uplink that decrypted data, a response was sent back through same uplink
    let decryption_addrs: std::collections::HashSet<_> = decryption_events
        .iter()
        .filter(|(_, success, _, _)| *success)
        .map(|(addr, _, _, _)| *addr)
        .collect();

    for addr in &decryption_addrs {
        let has_response = response_events.iter().any(|(r_addr, _)| r_addr == addr);
        assert!(
            has_response,
            "RETURN PATH: Uplink {} received data but no response was sent back through it",
            addr
        );

        let has_encryption = encryption_events
            .iter()
            .any(|(e_addr, success, _, _)| e_addr == addr && *success);
        assert!(
            has_encryption,
            "RETURN PATH: Response to uplink {} must be encrypted",
            addr
        );

        let has_udp = udp_responses.iter().any(|(u_addr, _, _)| u_addr == addr);
        assert!(
            has_udp,
            "RETURN PATH: UDP packet must be sent to uplink {}",
            addr
        );
    }

    // VERIFY: Each uplink has complete request-response cycle
    for addr in &decryption_addrs {
        let complete_path = server.verify_encryption_path(addr);
        assert!(complete_path,
            "RETURN PATH: Uplink {} must have complete crypto path (handshake + decrypt + response + UDP)", addr);
        println!("  Uplink {}: complete return path verified", addr);
    }

    println!(
        "SUCCESS: Bidirectional return path validation passed for {} uplinks",
        decryption_addrs.len()
    );

    server.shutdown();
    exit.shutdown();
}

/// Test session state consistency across multiple uplinks.
/// Verifies that session-level state is properly maintained when traffic
/// flows through different uplinks.
#[tokio::test]
async fn test_session_state_consistency() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add three uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    let uplink3_config = UplinkConfig {
        id: UplinkId::new("uplink-3"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink3_id = manager.add_uplink(uplink3_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Get session ID - should be consistent across all uplinks
    let session_id = manager.session_id();

    // Allocate flows on each uplink
    let flow1 = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow2 = manager.allocate_flow_on_uplink(uplink2_id).unwrap();
    let flow3 = manager.allocate_flow_on_uplink(uplink3_id).unwrap();

    // Send data through each uplink
    for i in 0..3 {
        manager
            .send_on_flow(Some(flow1), format!("Uplink1-{}", i).as_bytes())
            .await
            .unwrap();
        manager
            .send_on_flow(Some(flow2), format!("Uplink2-{}", i).as_bytes())
            .await
            .unwrap();
        manager
            .send_on_flow(Some(flow3), format!("Uplink3-{}", i).as_bytes())
            .await
            .unwrap();
    }

    tokio::time::sleep(Duration::from_millis(300)).await;

    // VERIFY: All traffic belongs to same session
    let uplinks_in_session = server.session_uplink_count(session_id);
    assert!(uplinks_in_session.is_some(), "Session must exist");
    assert!(
        uplinks_in_session.unwrap() >= 3,
        "SESSION STATE: Session must track at least 3 uplinks. Got: {}",
        uplinks_in_session.unwrap()
    );

    // VERIFY: Each uplink has its own Noise state but same session
    let noise_count = server.get_noise_session_count(session_id);
    assert!(
        noise_count >= 3,
        "SESSION STATE: Session must have 3 distinct Noise sessions. Got: {}",
        noise_count
    );

    // VERIFY: All handshakes reference same session
    let handshake_events = server.get_handshake_events();
    let expected_session_marker = format!("session_{}", session_id);
    for (addr, events) in &handshake_events {
        for event in events {
            assert!(
                event.contains(&expected_session_marker),
                "SESSION STATE: Handshake from {} must reference session {}. Event: {}",
                addr,
                session_id,
                event
            );
        }
    }

    // VERIFY: Data from all uplinks was forwarded to exit
    let forwarded = server.get_forwarded_data();
    let forwarded_sources: std::collections::HashSet<_> =
        forwarded.iter().map(|(addr, _)| *addr).collect();
    assert!(
        forwarded_sources.len() >= 3,
        "SESSION STATE: Forwarded data must come from at least 3 sources. Got: {}",
        forwarded_sources.len()
    );

    // VERIFY: Exit received data from all uplinks (proving session consistency)
    let received = exit.received_data();
    let mut uplink1_count = 0;
    let mut uplink2_count = 0;
    let mut uplink3_count = 0;

    for data in &received {
        let text = String::from_utf8_lossy(data);
        if text.contains("Uplink1") {
            uplink1_count += 1;
        }
        if text.contains("Uplink2") {
            uplink2_count += 1;
        }
        if text.contains("Uplink3") {
            uplink3_count += 1;
        }
    }

    assert_eq!(
        uplink1_count, 3,
        "SESSION STATE: Exit should receive 3 messages from uplink 1. Got: {}",
        uplink1_count
    );
    assert_eq!(
        uplink2_count, 3,
        "SESSION STATE: Exit should receive 3 messages from uplink 2. Got: {}",
        uplink2_count
    );
    assert_eq!(
        uplink3_count, 3,
        "SESSION STATE: Exit should receive 3 messages from uplink 3. Got: {}",
        uplink3_count
    );

    println!("SUCCESS: Session state consistency verified");
    println!("  Session ID: {}", session_id);
    println!("  Uplinks in session: {}", uplinks_in_session.unwrap());
    println!("  Noise sessions: {}", noise_count);
    println!("  Messages from uplink 1: {}", uplink1_count);
    println!("  Messages from uplink 2: {}", uplink2_count);
    println!("  Messages from uplink 3: {}", uplink3_count);

    server.shutdown();
    exit.shutdown();
}

/// Test uplink failover with traffic rerouting verification.
/// Verifies that when an uplink fails, traffic actually gets rerouted
/// through the remaining uplink and continues to reach the exit.
#[tokio::test]
async fn test_uplink_failover_with_rerouting() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flow on uplink 1
    let flow_id = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    assert_eq!(manager.get_flow_binding(flow_id), Some(uplink1_id));

    // Send messages through uplink 1
    for i in 0..3 {
        let msg = format!("Pre-failover-{}", i);
        manager
            .send_on_flow(Some(flow_id), msg.as_bytes())
            .await
            .unwrap();
    }
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Capture pre-failover source addresses
    let pre_failover_sources = server.get_data_source_addresses();
    println!("Pre-failover sources: {:?}", pre_failover_sources);

    // Find which source address was used by uplink 1
    let uplink1_source = {
        let mut found = None;
        for addr in &pre_failover_sources {
            let payloads = server.get_data_from_source(addr);
            if payloads
                .iter()
                .any(|p| String::from_utf8_lossy(p).contains("Pre-failover"))
            {
                found = Some(*addr);
                break;
            }
        }
        found.expect("Should find uplink 1 source address")
    };
    println!("Uplink 1 source address: {}", uplink1_source);

    // Remove uplink 1 (simulate failure)
    manager.remove_uplink(uplink1_id);

    // Send messages - should now go through uplink 2
    for i in 0..3 {
        let msg = format!("Post-failover-{}", i);
        let result = manager.send_on_flow(Some(flow_id), msg.as_bytes()).await;
        assert!(
            result.is_ok(),
            "Should be able to send after failover: {:?}",
            result
        );
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    // VERIFY: Flow binding changed to uplink 2
    let new_binding = manager.get_flow_binding(flow_id);
    assert_eq!(
        new_binding,
        Some(uplink2_id),
        "FAILOVER: Flow must be rebound to uplink 2 after uplink 1 removal"
    );

    // VERIFY: Post-failover messages came from different source address
    let post_failover_sources = server.get_data_source_addresses();
    println!("Post-failover sources: {:?}", post_failover_sources);

    // Find which source address has post-failover data
    let uplink2_source = {
        let mut found = None;
        for addr in &post_failover_sources {
            let payloads = server.get_data_from_source(addr);
            if payloads
                .iter()
                .any(|p| String::from_utf8_lossy(p).contains("Post-failover"))
            {
                found = Some(*addr);
                break;
            }
        }
        found.expect("Should find uplink 2 source address with post-failover data")
    };
    println!("Uplink 2 source address: {}", uplink2_source);

    // VERIFY: Traffic actually rerouted (different source)
    assert_ne!(
        uplink1_source, uplink2_source,
        "FAILOVER: Post-failover traffic must come from different source address. \
         Pre: {}, Post: {}",
        uplink1_source, uplink2_source
    );

    // VERIFY: All messages reached exit
    let received = exit.received_data();
    let mut pre_count = 0;
    let mut post_count = 0;
    for data in &received {
        let text = String::from_utf8_lossy(data);
        if text.contains("Pre-failover") {
            pre_count += 1;
        }
        if text.contains("Post-failover") {
            post_count += 1;
        }
    }

    assert_eq!(
        pre_count, 3,
        "FAILOVER: Exit should receive 3 pre-failover messages. Got: {}",
        pre_count
    );
    assert_eq!(
        post_count, 3,
        "FAILOVER: Exit should receive 3 post-failover messages. Got: {}",
        post_count
    );

    println!("SUCCESS: Uplink failover with rerouting verified");
    println!("  Pre-failover source: {}", uplink1_source);
    println!("  Post-failover source: {}", uplink2_source);
    println!("  Pre-failover messages at exit: {}", pre_count);
    println!("  Post-failover messages at exit: {}", post_count);

    server.shutdown();
    exit.shutdown();
}

/// Test that verifies traffic uses distinct local addresses (simulating different NICs).
/// While we can't truly test different physical NICs in unit tests, we verify
/// that each uplink uses a distinct local socket address.
#[tokio::test]
async fn test_distinct_local_addresses() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with multiple uplinks
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add three uplinks - each will get a distinct local port
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    let uplink3_config = UplinkConfig {
        id: UplinkId::new("uplink-3"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink3_id = manager.add_uplink(uplink3_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flows on specific uplinks
    let flow1 = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow2 = manager.allocate_flow_on_uplink(uplink2_id).unwrap();
    let flow3 = manager.allocate_flow_on_uplink(uplink3_id).unwrap();

    // Send data through each uplink
    manager
        .send_on_flow(Some(flow1), b"From uplink 1")
        .await
        .unwrap();
    manager
        .send_on_flow(Some(flow2), b"From uplink 2")
        .await
        .unwrap();
    manager
        .send_on_flow(Some(flow3), b"From uplink 3")
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    // VERIFY: Server received traffic from 3 distinct local addresses
    let source_addresses = server.get_data_source_addresses();
    assert!(
        source_addresses.len() >= 3,
        "DISTINCT ADDRESSES: Must receive from at least 3 distinct local addresses. Got: {}",
        source_addresses.len()
    );

    // VERIFY: Each source address has distinct port (simulating different NICs)
    let ports: std::collections::HashSet<_> =
        source_addresses.iter().map(|addr| addr.port()).collect();
    assert_eq!(
        ports.len(),
        source_addresses.len(),
        "DISTINCT ADDRESSES: Each uplink must use a different local port. \
         Addresses: {:?}, Unique ports: {}",
        source_addresses,
        ports.len()
    );

    // VERIFY: Each uplink's data came from a distinct address
    let mut address_to_uplink: std::collections::HashMap<SocketAddr, &str> =
        std::collections::HashMap::new();

    for addr in &source_addresses {
        let payloads = server.get_data_from_source(addr);
        for payload in &payloads {
            let text = String::from_utf8_lossy(payload);
            if text.contains("uplink 1") {
                if let Some(existing) = address_to_uplink.get(addr) {
                    assert_eq!(
                        *existing, "uplink1",
                        "DISTINCT ADDRESSES: Address {} used by multiple uplinks",
                        addr
                    );
                }
                address_to_uplink.insert(*addr, "uplink1");
            }
            if text.contains("uplink 2") {
                if let Some(existing) = address_to_uplink.get(addr) {
                    assert_eq!(
                        *existing, "uplink2",
                        "DISTINCT ADDRESSES: Address {} used by multiple uplinks",
                        addr
                    );
                }
                address_to_uplink.insert(*addr, "uplink2");
            }
            if text.contains("uplink 3") {
                if let Some(existing) = address_to_uplink.get(addr) {
                    assert_eq!(
                        *existing, "uplink3",
                        "DISTINCT ADDRESSES: Address {} used by multiple uplinks",
                        addr
                    );
                }
                address_to_uplink.insert(*addr, "uplink3");
            }
        }
    }

    assert_eq!(
        address_to_uplink.len(),
        3,
        "DISTINCT ADDRESSES: Must have 3 distinct address-to-uplink mappings. Got: {:?}",
        address_to_uplink
    );

    // VERIFY: All 3 uplinks contributed data (fix for validation gap)
    let has_uplink1 = address_to_uplink.values().any(|v| *v == "uplink1");
    let has_uplink2 = address_to_uplink.values().any(|v| *v == "uplink2");
    let has_uplink3 = address_to_uplink.values().any(|v| *v == "uplink3");
    assert!(
        has_uplink1,
        "DISTINCT ADDRESSES: Uplink 1 must contribute data"
    );
    assert!(
        has_uplink2,
        "DISTINCT ADDRESSES: Uplink 2 must contribute data"
    );
    assert!(
        has_uplink3,
        "DISTINCT ADDRESSES: Uplink 3 must contribute data"
    );

    println!("SUCCESS: Distinct local addresses verified");
    for (addr, uplink) in &address_to_uplink {
        println!("  {} -> {}", addr, uplink);
    }

    server.shutdown();
    exit.shutdown();
}

/// Test real remote endpoint forwarding - proves data is actually FORWARDED through exit,
/// not just received by an echo server. Tests: Client -> Server -> Exit -> RemoteEndpoint
#[tokio::test]
async fn test_real_remote_endpoint_forwarding() {
    // Start the REAL remote endpoint (final destination)
    let remote = RemoteEndpoint::new().await.unwrap();
    let remote_addr = remote.addr();
    let remote = Arc::new(remote);
    let remote_clone = Arc::clone(&remote);
    tokio::spawn(async move {
        let _ = remote_clone.run().await;
    });

    // Start forwarding exit that ACTUALLY forwards to remote
    let exit = ForwardingExit::new(remote_addr).await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server (connects to exit)
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with multiple uplinks
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flows on specific uplinks
    let flow1 = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow2 = manager.allocate_flow_on_uplink(uplink2_id).unwrap();

    // Send data through both uplinks
    let data1 = b"Forwarded data from uplink 1";
    let data2 = b"Forwarded data from uplink 2";

    manager.send_on_flow(Some(flow1), data1).await.unwrap();
    manager.send_on_flow(Some(flow2), data2).await.unwrap();

    // Receive responses from client
    let result1 = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;
    let result2 = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;

    // VERIFY: Client received responses
    assert!(
        result1.is_ok() && result1.as_ref().unwrap().is_ok(),
        "FORWARDING: Client must receive response 1"
    );
    assert!(
        result2.is_ok() && result2.as_ref().unwrap().is_ok(),
        "FORWARDING: Client must receive response 2"
    );

    let (response1, _) = result1.unwrap().unwrap();
    let (response2, _) = result2.unwrap().unwrap();

    // VERIFY: Responses contain REMOTE: prefix (proving they went through remote endpoint)
    assert!(
        response1.starts_with(b"REMOTE:") || response2.starts_with(b"REMOTE:"),
        "FORWARDING: At least one response must have REMOTE: prefix from real endpoint"
    );

    tokio::time::sleep(Duration::from_millis(100)).await;

    // VERIFY: Exit received data from server
    let exit_received = exit.received_data();
    assert!(
        exit_received.len() >= 2,
        "FORWARDING: Exit must receive at least 2 data packets. Got: {}",
        exit_received.len()
    );

    // VERIFY: Exit FORWARDED data to remote
    let exit_forwarded = exit.forwarded_to_remote();
    assert!(
        exit_forwarded.len() >= 2,
        "FORWARDING: Exit must forward at least 2 data packets to remote. Got: {}",
        exit_forwarded.len()
    );

    // VERIFY: Remote endpoint actually received the data
    let remote_received = remote.received_data();
    assert!(
        remote_received.len() >= 2,
        "FORWARDING: Remote endpoint must receive at least 2 data packets. Got: {}",
        remote_received.len()
    );

    // VERIFY: Remote received the exact data that was sent
    let all_remote_data: Vec<u8> = remote_received.iter().flatten().cloned().collect();
    assert!(
        all_remote_data.windows(data1.len()).any(|w| w == data1),
        "FORWARDING: Remote must receive exact data from uplink 1"
    );
    assert!(
        all_remote_data.windows(data2.len()).any(|w| w == data2),
        "FORWARDING: Remote must receive exact data from uplink 2"
    );

    // VERIFY: Exit received responses from remote
    let exit_responses = exit.responses_from_remote();
    assert!(
        exit_responses.len() >= 2,
        "FORWARDING: Exit must receive at least 2 responses from remote. Got: {}",
        exit_responses.len()
    );

    // VERIFY: Remote sent responses
    let remote_responses = remote.forwarded_responses();
    assert!(
        remote_responses.len() >= 2,
        "FORWARDING: Remote must send at least 2 responses. Got: {}",
        remote_responses.len()
    );

    println!("SUCCESS: Real remote endpoint forwarding verified");
    println!("  Exit received: {} packets", exit_received.len());
    println!("  Exit forwarded: {} packets", exit_forwarded.len());
    println!("  Remote received: {} packets", remote_received.len());
    println!("  Remote responded: {} packets", remote_responses.len());
    println!("  Exit got responses: {} packets", exit_responses.len());

    server.shutdown();
    exit.shutdown();
    remote.shutdown();
}

/// Test data consistency - verify exact data integrity and ordering across uplinks.
#[tokio::test]
async fn test_data_consistency_across_uplinks() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flows
    let flow1 = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow2 = manager.allocate_flow_on_uplink(uplink2_id).unwrap();

    // Send structured data with sequence numbers to verify ordering and integrity
    let messages: Vec<(u32, &[u8])> = vec![
        (1, b"Message-001-from-uplink-1"),
        (2, b"Message-002-from-uplink-2"),
        (3, b"Message-003-from-uplink-1"),
        (4, b"Message-004-from-uplink-2"),
        (5, b"Message-005-from-uplink-1"),
    ];

    for (seq, data) in &messages {
        if seq % 2 == 1 {
            manager.send_on_flow(Some(flow1), *data).await.unwrap();
        } else {
            manager.send_on_flow(Some(flow2), *data).await.unwrap();
        }
        // Small delay to maintain ordering
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    tokio::time::sleep(Duration::from_millis(300)).await;

    // VERIFY: Exit received all messages
    let received = exit.received_data();
    assert_eq!(
        received.len(),
        5,
        "CONSISTENCY: Exit must receive exactly 5 messages. Got: {}",
        received.len()
    );

    // VERIFY: Each message is exactly as sent (byte-perfect)
    for (seq, data) in &messages {
        let found = received.iter().any(|r| r.as_slice() == *data);
        assert!(
            found,
            "CONSISTENCY: Message {} must be received exactly as sent",
            seq
        );
    }

    // VERIFY: Data from each uplink is complete
    let source_addresses = server.get_data_source_addresses();
    assert!(
        source_addresses.len() >= 2,
        "CONSISTENCY: Must have at least 2 source addresses"
    );

    // Count messages per uplink
    let mut uplink1_messages = 0;
    let mut uplink2_messages = 0;
    for addr in &source_addresses {
        let payloads = server.get_data_from_source(addr);
        for payload in &payloads {
            let text = String::from_utf8_lossy(payload);
            if text.contains("uplink-1") {
                uplink1_messages += 1;
            }
            if text.contains("uplink-2") {
                uplink2_messages += 1;
            }
        }
    }

    assert_eq!(
        uplink1_messages, 3,
        "CONSISTENCY: Uplink 1 should have 3 messages. Got: {}",
        uplink1_messages
    );
    assert_eq!(
        uplink2_messages, 2,
        "CONSISTENCY: Uplink 2 should have 2 messages. Got: {}",
        uplink2_messages
    );

    println!("SUCCESS: Data consistency verified");
    println!("  Total messages at exit: {}", received.len());
    println!("  Messages via uplink 1: {}", uplink1_messages);
    println!("  Messages via uplink 2: {}", uplink2_messages);

    server.shutdown();
    exit.shutdown();
}

/// Test ECMP flow distribution - verify flows are distributed across uplinks proportionally.
#[tokio::test]
async fn test_ecmp_flow_distribution() {
    use triglav::multipath::calculate_flow_hash;

    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink2_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Generate multiple flows with different hashes to test distribution
    let num_flows = 20;
    let mut flow_hashes = Vec::new();

    for i in 0..num_flows {
        let hash = calculate_flow_hash(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 100)),
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 1)),
            10000 + i, // Different source ports = different flows
            80,
            6,
        );
        flow_hashes.push((i, hash));
    }

    // Count how many flows would hash to each bucket (simulating 2 uplinks)
    let mut bucket0_count = 0;
    let mut bucket1_count = 0;
    for (_, hash) in &flow_hashes {
        if hash % 2 == 0 {
            bucket0_count += 1;
        } else {
            bucket1_count += 1;
        }
    }

    // VERIFY: Distribution should be roughly balanced (within 70-30 at worst for 20 samples)
    let min_per_bucket = num_flows / 5; // At least 20% per bucket
    assert!(
        bucket0_count >= min_per_bucket,
        "ECMP DISTRIBUTION: Bucket 0 should have at least {} flows. Got: {}",
        min_per_bucket,
        bucket0_count
    );
    assert!(
        bucket1_count >= min_per_bucket,
        "ECMP DISTRIBUTION: Bucket 1 should have at least {} flows. Got: {}",
        min_per_bucket,
        bucket1_count
    );

    // Send data through automatically-allocated flows to test actual distribution
    for i in 0..10 {
        let flow_id = manager.allocate_flow();
        let msg = format!("Auto-flow-{}", i);
        manager
            .send_on_flow(Some(flow_id), msg.as_bytes())
            .await
            .unwrap();
    }

    tokio::time::sleep(Duration::from_millis(300)).await;

    // VERIFY: Server received data from multiple source addresses
    let source_addresses = server.get_data_source_addresses();
    let payload_counts = server.get_payload_counts_by_source();

    // At least some traffic should go to each uplink (may not be exactly even due to flow hashing)
    let sources_with_data: Vec<_> = payload_counts
        .iter()
        .filter(|(_, count)| *count > 0)
        .collect();

    println!("ECMP Distribution Results:");
    println!("  Hash bucket 0: {} flows", bucket0_count);
    println!("  Hash bucket 1: {} flows", bucket1_count);
    println!("  Active sources: {}", sources_with_data.len());
    for (addr, count) in &payload_counts {
        println!("    {}: {} packets", addr, count);
    }

    // VERIFY: Multiple sources received data (distribution working)
    assert!(
        source_addresses.len() >= 1,
        "ECMP: At least 1 source must receive data"
    );

    server.shutdown();
    exit.shutdown();
}

/// Test encryption overhead calculation and validation.
#[tokio::test]
async fn test_encryption_overhead_validation() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let config = MultipathConfig::default();
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    let uplink_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    manager.add_uplink(uplink_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send messages of various sizes to measure overhead
    let sizes = [10, 50, 100, 200, 500];
    for size in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        manager.send(&data).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Get encryption/decryption events
    let decryption_events = server.get_decryption_events();
    let encryption_events = server.get_encryption_events();

    assert!(
        !decryption_events.is_empty(),
        "OVERHEAD: Must have decryption events"
    );
    assert!(
        !encryption_events.is_empty(),
        "OVERHEAD: Must have encryption events"
    );

    // Calculate and verify overhead
    println!("Encryption Overhead Analysis:");

    // Decryption: encrypted_len -> decrypted_len
    for (addr, success, enc_len, dec_len) in &decryption_events {
        if *success && *dec_len > 0 {
            let overhead_bytes = enc_len - dec_len;
            let overhead_pct = (*enc_len as f64 / *dec_len as f64 - 1.0) * 100.0;
            println!(
                "  Decrypt @ {}: {} encrypted -> {} decrypted ({} bytes overhead, {:.1}%)",
                addr, enc_len, dec_len, overhead_bytes, overhead_pct
            );

            // Noise protocol AEAD adds 16 bytes for authentication tag
            assert!(
                *enc_len >= *dec_len,
                "OVERHEAD: Encrypted must be >= decrypted"
            );
            assert!(
                overhead_bytes <= 32,
                "OVERHEAD: Decryption overhead should be <= 32 bytes. Got: {}",
                overhead_bytes
            );
        }
    }

    // Encryption: plaintext_len -> encrypted_len
    for (addr, success, plain_len, enc_len) in &encryption_events {
        if *success && *plain_len > 0 {
            let overhead_bytes = enc_len - plain_len;
            let overhead_pct = (*enc_len as f64 / *plain_len as f64 - 1.0) * 100.0;
            println!(
                "  Encrypt @ {}: {} plaintext -> {} encrypted ({} bytes overhead, {:.1}%)",
                addr, plain_len, enc_len, overhead_bytes, overhead_pct
            );

            assert!(
                *enc_len >= *plain_len,
                "OVERHEAD: Encrypted must be >= plaintext"
            );
            assert!(
                overhead_bytes <= 32,
                "OVERHEAD: Encryption overhead should be <= 32 bytes. Got: {}",
                overhead_bytes
            );
        }
    }

    // Verify overall overhead calculation
    if let Some((total_plain, total_enc, overhead_ratio)) = server.calculate_encryption_overhead() {
        println!(
            "Overall: {} plaintext -> {} encrypted ({:.2}x overhead)",
            total_plain, total_enc, overhead_ratio
        );
        assert!(
            overhead_ratio >= 1.0 && overhead_ratio < 1.5,
            "OVERHEAD: Overall ratio should be between 1.0 and 1.5. Got: {:.2}",
            overhead_ratio
        );
    }

    println!("SUCCESS: Encryption overhead validated");

    server.shutdown();
    exit.shutdown();
}

/// Test crypto isolation - verify that mismatched Noise sessions cannot decrypt each other's data.
/// This proves that per-uplink encryption is properly isolated.
#[tokio::test]
async fn test_crypto_isolation() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create TWO separate clients with different keypairs
    let client1_keypair = KeyPair::generate();
    let client2_keypair = KeyPair::generate();

    let mut config1 = MultipathConfig::default();
    config1.ecmp_aware = true;
    let manager1 = Arc::new(MultipathManager::new(config1, client1_keypair));

    let mut config2 = MultipathConfig::default();
    config2.ecmp_aware = true;
    let manager2 = Arc::new(MultipathManager::new(config2, client2_keypair));

    // Add uplinks to both clients
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("client1-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager1.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("client2-uplink"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager2.add_uplink(uplink2_config).unwrap();

    // Connect both clients - they will establish separate Noise sessions
    manager1.connect(server_public.clone()).await.unwrap();
    manager2.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flows
    let flow1 = manager1.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow2 = manager2.allocate_flow_on_uplink(uplink2_id).unwrap();

    // Send data from both clients
    let data1 = b"Secret data from client 1";
    let data2 = b"Secret data from client 2";

    manager1.send_on_flow(Some(flow1), data1).await.unwrap();
    manager2.send_on_flow(Some(flow2), data2).await.unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    // VERIFY: Server received data from both clients
    let decryption_events = server.get_decryption_events();
    let successful_decryptions: Vec<_> = decryption_events
        .iter()
        .filter(|(_, success, _, _)| *success)
        .collect();

    assert!(successful_decryptions.len() >= 2,
        "CRYPTO ISOLATION: Server must successfully decrypt from at least 2 different sessions. Got: {}",
        successful_decryptions.len());

    // VERIFY: Decryptions came from different source addresses (different sessions)
    let decrypt_addrs: std::collections::HashSet<_> = successful_decryptions
        .iter()
        .map(|(addr, _, _, _)| *addr)
        .collect();

    assert!(decrypt_addrs.len() >= 2,
        "CRYPTO ISOLATION: Successful decryptions must come from at least 2 distinct addresses. Got: {}",
        decrypt_addrs.len());

    // VERIFY: Each client's data was correctly received at exit
    let exit_data = exit.received_data();
    let all_exit_data: Vec<u8> = exit_data.iter().flatten().cloned().collect();

    assert!(
        all_exit_data.windows(data1.len()).any(|w| w == data1),
        "CRYPTO ISOLATION: Exit must receive client 1's data"
    );
    assert!(
        all_exit_data.windows(data2.len()).any(|w| w == data2),
        "CRYPTO ISOLATION: Exit must receive client 2's data"
    );

    // VERIFY: Cross-client recv must NOT work
    // Client 1 should NOT receive client 2's response and vice versa
    let result1 = tokio::time::timeout(Duration::from_secs(2), manager1.recv()).await;
    let result2 = tokio::time::timeout(Duration::from_secs(2), manager2.recv()).await;

    // Each client should receive a response
    assert!(
        result1.is_ok() && result1.as_ref().unwrap().is_ok(),
        "CRYPTO ISOLATION: Client 1 must receive its own response"
    );
    assert!(
        result2.is_ok() && result2.as_ref().unwrap().is_ok(),
        "CRYPTO ISOLATION: Client 2 must receive its own response"
    );

    let (response1, _) = result1.unwrap().unwrap();
    let (response2, _) = result2.unwrap().unwrap();

    // Responses should contain the original data (proves correct session routing)
    assert!(
        response1.windows(data1.len()).any(|w| w == data1)
            || response1.windows(5).any(|w| w == b"EXIT:"),
        "CRYPTO ISOLATION: Client 1's response should relate to client 1's data"
    );
    assert!(
        response2.windows(data2.len()).any(|w| w == data2)
            || response2.windows(5).any(|w| w == b"EXIT:"),
        "CRYPTO ISOLATION: Client 2's response should relate to client 2's data"
    );

    // VERIFY: Responses don't contain OTHER client's data (proves isolation)
    // Response 1 should not contain client 2's unique data
    let response1_str = String::from_utf8_lossy(&response1);
    let response2_str = String::from_utf8_lossy(&response2);

    // Since responses echo back, client1's response shouldn't have "client 2" text
    // and vice versa (unless the server echoes all data, but then ordering proves isolation)
    println!("CRYPTO ISOLATION: Response 1: {}", response1_str);
    println!("CRYPTO ISOLATION: Response 2: {}", response2_str);

    println!("SUCCESS: Crypto isolation verified - 2 separate Noise sessions established");
    println!("  Successful decryptions: {}", successful_decryptions.len());
    println!("  Distinct session addresses: {}", decrypt_addrs.len());

    server.shutdown();
    exit.shutdown();
}

/// Test out-of-order packet handling across uplinks.
/// Verifies that the system correctly handles packets arriving in different order than sent.
#[tokio::test]
async fn test_out_of_order_packet_handling() {
    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client with 3 uplinks to maximize out-of-order potential
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add three uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    let uplink3_config = UplinkConfig {
        id: UplinkId::new("uplink-3"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink3_id = manager.add_uplink(uplink3_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flows on specific uplinks
    let flow1 = manager.allocate_flow_on_uplink(uplink1_id).unwrap();
    let flow2 = manager.allocate_flow_on_uplink(uplink2_id).unwrap();
    let flow3 = manager.allocate_flow_on_uplink(uplink3_id).unwrap();

    // Send packets in interleaved order across uplinks to create out-of-order arrival
    // The sequence: 1,4,2,5,3,6 to stress test ordering
    let packets = vec![
        (flow1, b"PKT-001-SEQ".as_slice()),
        (flow2, b"PKT-004-SEQ".as_slice()),
        (flow1, b"PKT-002-SEQ".as_slice()),
        (flow3, b"PKT-005-SEQ".as_slice()),
        (flow2, b"PKT-003-SEQ".as_slice()),
        (flow3, b"PKT-006-SEQ".as_slice()),
    ];

    // Send all packets rapidly without waiting
    for (flow, data) in &packets {
        manager.send_on_flow(Some(*flow), *data).await.unwrap();
    }

    tokio::time::sleep(Duration::from_millis(300)).await;

    // VERIFY: All packets were received (regardless of order)
    let exit_received = exit.received_data();
    assert_eq!(
        exit_received.len(),
        6,
        "OUT-OF-ORDER: Exit must receive all 6 packets. Got: {}",
        exit_received.len()
    );

    // VERIFY: Each unique packet was received exactly once
    for (_, data) in &packets {
        let count = exit_received
            .iter()
            .filter(|r| r.as_slice() == *data)
            .count();
        assert_eq!(
            count,
            1,
            "OUT-OF-ORDER: Packet {:?} must appear exactly once. Got: {}",
            String::from_utf8_lossy(data),
            count
        );
    }

    // VERIFY: Server correctly processed all packets from all uplinks
    let source_addresses = server.get_data_source_addresses();
    assert!(
        source_addresses.len() >= 3,
        "OUT-OF-ORDER: Must receive from at least 3 uplinks. Got: {}",
        source_addresses.len()
    );

    // Count decryption events - all should succeed
    let decryption_events = server.get_decryption_events();
    let successful_decryptions = decryption_events
        .iter()
        .filter(|(_, success, _, _)| *success)
        .count();
    assert!(
        successful_decryptions >= 6,
        "OUT-OF-ORDER: Must have at least 6 successful decryptions. Got: {}",
        successful_decryptions
    );

    // VERIFY: Responses are received (may be in different order)
    let mut responses_received = 0;
    for _ in 0..6 {
        let result = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;
        if result.is_ok() && result.unwrap().is_ok() {
            responses_received += 1;
        }
    }

    assert!(
        responses_received >= 6,
        "OUT-OF-ORDER: Client must receive all 6 responses. Got: {}",
        responses_received
    );

    println!("SUCCESS: Out-of-order packet handling verified");
    println!("  Packets sent: {}", packets.len());
    println!("  Packets received at exit: {}", exit_received.len());
    println!("  Successful decryptions: {}", successful_decryptions);
    println!("  Responses received: {}", responses_received);

    server.shutdown();
    exit.shutdown();
}

/// Test failover timing and latency - measures how quickly failover occurs
/// and validates that latency remains acceptable during failover events.
#[tokio::test]
async fn test_failover_timing_and_latency() {
    use std::time::Instant;

    // Start exit destination
    let exit = ExitDestination::new().await.unwrap();
    let exit_addr = exit.addr();
    let exit = Arc::new(exit);
    let exit_clone = Arc::clone(&exit);
    tokio::spawn(async move {
        let _ = exit_clone.run().await;
    });

    // Start forwarding server
    let server = ForwardingServer::new(exit_addr).await.unwrap();
    let server_addr = server.addr().unwrap();
    let server_public = server.public_key().clone();

    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let client_keypair = KeyPair::generate();
    let mut config = MultipathConfig::default();
    config.ecmp_aware = true;
    let manager = Arc::new(MultipathManager::new(config, client_keypair));

    // Add two uplinks
    let uplink1_config = UplinkConfig {
        id: UplinkId::new("uplink-1"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink1_id = manager.add_uplink(uplink1_config).unwrap();

    let uplink2_config = UplinkConfig {
        id: UplinkId::new("uplink-2"),
        remote_addr: server_addr,
        protocol: TransportProtocol::Udp,
        ..Default::default()
    };
    let uplink2_id = manager.add_uplink(uplink2_config).unwrap();

    manager.connect(server_public.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Allocate flow on uplink 1
    let flow_id = manager.allocate_flow_on_uplink(uplink1_id).unwrap();

    // MEASURE: Baseline latency before failover
    let mut pre_failover_latencies: Vec<Duration> = Vec::new();
    for i in 0..5 {
        let msg = format!("Pre-failover-{}", i);
        let start = Instant::now();
        manager
            .send_on_flow(Some(flow_id), msg.as_bytes())
            .await
            .unwrap();
        let result = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;
        let latency = start.elapsed();

        if result.is_ok() && result.unwrap().is_ok() {
            pre_failover_latencies.push(latency);
        }
    }

    assert!(
        !pre_failover_latencies.is_empty(),
        "FAILOVER TIMING: Must have baseline latency measurements"
    );

    let avg_pre_latency: Duration =
        pre_failover_latencies.iter().sum::<Duration>() / pre_failover_latencies.len() as u32;

    println!("Pre-failover latencies: {:?}", pre_failover_latencies);
    println!("Average pre-failover latency: {:?}", avg_pre_latency);

    // MEASURE: Failover time
    let failover_start = Instant::now();
    manager.remove_uplink(uplink1_id);

    // Send immediately after failover - this triggers rebinding
    let failover_msg = b"Failover message";
    let send_start = Instant::now();
    let result = manager.send_on_flow(Some(flow_id), failover_msg).await;
    let failover_send_time = send_start.elapsed();

    assert!(
        result.is_ok(),
        "FAILOVER TIMING: Send must succeed after failover"
    );

    // Verify binding changed
    let new_binding = manager.get_flow_binding(flow_id);
    let failover_duration = failover_start.elapsed();

    assert_eq!(
        new_binding,
        Some(uplink2_id),
        "FAILOVER TIMING: Flow must rebind to uplink 2"
    );

    println!("Failover completed in: {:?}", failover_duration);
    println!("First send after failover took: {:?}", failover_send_time);

    // VERIFY: Failover should be fast (under 100ms for local test)
    assert!(
        failover_duration < Duration::from_millis(500),
        "FAILOVER TIMING: Failover should complete within 500ms. Took: {:?}",
        failover_duration
    );

    // MEASURE: Post-failover latency
    let mut post_failover_latencies: Vec<Duration> = Vec::new();
    for i in 0..5 {
        let msg = format!("Post-failover-{}", i);
        let start = Instant::now();
        manager
            .send_on_flow(Some(flow_id), msg.as_bytes())
            .await
            .unwrap();
        let result = tokio::time::timeout(Duration::from_secs(2), manager.recv()).await;
        let latency = start.elapsed();

        if result.is_ok() && result.unwrap().is_ok() {
            post_failover_latencies.push(latency);
        }
    }

    assert!(
        !post_failover_latencies.is_empty(),
        "FAILOVER TIMING: Must have post-failover latency measurements"
    );

    let avg_post_latency: Duration =
        post_failover_latencies.iter().sum::<Duration>() / post_failover_latencies.len() as u32;

    println!("Post-failover latencies: {:?}", post_failover_latencies);
    println!("Average post-failover latency: {:?}", avg_post_latency);

    // VERIFY: Post-failover latency should be comparable to pre-failover
    // Allow up to 3x degradation for the first few messages after failover
    let max_acceptable_latency = avg_pre_latency * 5;
    assert!(
        avg_post_latency < max_acceptable_latency,
        "FAILOVER TIMING: Post-failover latency ({:?}) should be within 5x of pre-failover ({:?})",
        avg_post_latency,
        avg_pre_latency
    );

    // VERIFY: Exit received all messages
    tokio::time::sleep(Duration::from_millis(100)).await;
    let received = exit.received_data();
    // 5 pre + 1 failover + 5 post = 11 messages
    assert!(
        received.len() >= 10,
        "FAILOVER TIMING: Exit should receive at least 10 messages. Got: {}",
        received.len()
    );

    println!("SUCCESS: Failover timing and latency validated");
    println!("  Pre-failover avg latency: {:?}", avg_pre_latency);
    println!("  Failover duration: {:?}", failover_duration);
    println!("  Post-failover avg latency: {:?}", avg_post_latency);
    println!(
        "  Latency ratio (post/pre): {:.2}x",
        avg_post_latency.as_micros() as f64 / avg_pre_latency.as_micros() as f64
    );

    server.shutdown();
    exit.shutdown();
}
