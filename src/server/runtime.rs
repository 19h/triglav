//! UDP server runtime shared by `triglav server` and `triglav-server`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::net::UdpSocket;
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, error, info};

use crate::config::ServerConfig;
use crate::crypto::{KeyPair, NoiseSession};
use crate::error::{Error, Result};
use crate::metrics::{
    init_metrics, DefaultHealthChecker, HttpServerConfig, MetricsHttpServer, PrometheusMetrics,
    SessionStatus, StatusProvider, StatusResponse,
};
use crate::protocol::{Packet, PacketFlags, PacketType, HEADER_SIZE};
use crate::server::{Signal, SignalHandler};
use crate::types::{SequenceNumber, SessionId, TrafficStats};

/// Options required to run a Triglav UDP server.
pub struct ServerRuntimeOptions {
    /// Server bind and session configuration.
    pub config: ServerConfig,
    /// Server static keypair.
    pub keypair: KeyPair,
    /// Metrics/status HTTP bind address.
    pub metrics_addr: SocketAddr,
}

/// Server statistics.
#[derive(Debug, Default, Clone)]
pub struct ServerStats {
    pub total_connections: u64,
    pub active_connections: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub packets_dropped: u64,
}

/// Client transport session.
struct TransportSession {
    id: SessionId,
    client_addrs: RwLock<Vec<SocketAddr>>,
    noise: RwLock<Option<NoiseSession>>,
    last_activity: RwLock<Instant>,
    stats: RwLock<TrafficStats>,
    user_id: RwLock<Option<String>>,
}

impl TransportSession {
    fn new(id: SessionId, client_addr: SocketAddr) -> Self {
        Self {
            id,
            client_addrs: RwLock::new(vec![client_addr]),
            noise: RwLock::new(None),
            last_activity: RwLock::new(Instant::now()),
            stats: RwLock::new(TrafficStats::default()),
            user_id: RwLock::new(None),
        }
    }

    fn touch(&self) {
        *self.last_activity.write() = Instant::now();
    }

    fn is_expired(&self, timeout: Duration) -> bool {
        self.last_activity.read().elapsed() > timeout
    }

    fn add_address(&self, addr: SocketAddr) {
        let mut addrs = self.client_addrs.write();
        if !addrs.contains(&addr) {
            addrs.push(addr);
        }
    }
}

/// UDP server runtime.
pub struct TriglavServer {
    config: ServerConfig,
    keypair: KeyPair,
    transport_sessions: DashMap<SessionId, Arc<TransportSession>>,
    sessions_by_addr: DashMap<SocketAddr, SessionId>,
    sockets: Vec<Arc<UdpSocket>>,
    metrics: Arc<PrometheusMetrics>,
    stats: Arc<RwLock<ServerStats>>,
    start_time: Instant,
    shutdown_tx: broadcast::Sender<()>,
}

impl TriglavServer {
    /// Create a new server runtime.
    pub async fn new(
        config: ServerConfig,
        keypair: KeyPair,
        metrics: Arc<PrometheusMetrics>,
    ) -> Result<Self> {
        if config.listen_addrs.is_empty() {
            return Err(Error::Config("No listen address specified".into()));
        }

        let mut sockets = Vec::with_capacity(config.listen_addrs.len());
        for addr in &config.listen_addrs {
            let socket = UdpSocket::bind(addr).await.map_err(|e| {
                Error::Transport(crate::error::TransportError::BindFailed {
                    addr: *addr,
                    reason: e.to_string(),
                })
            })?;

            info!("Server bound to {}", socket.local_addr().unwrap_or(*addr));
            sockets.push(Arc::new(socket));
        }

        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            config,
            keypair,
            transport_sessions: DashMap::new(),
            sessions_by_addr: DashMap::new(),
            sockets,
            metrics,
            stats: Arc::new(RwLock::new(ServerStats::default())),
            start_time: Instant::now(),
            shutdown_tx,
        })
    }

    /// Run the server until shutdown is requested.
    pub async fn run(&self) -> Result<()> {
        let (packet_tx, mut packet_rx) =
            mpsc::channel::<(Arc<UdpSocket>, Vec<u8>, SocketAddr)>(1024);
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        self.start_background_tasks();
        self.start_receive_tasks(packet_tx);

        info!("Server running, waiting for connections...");

        loop {
            tokio::select! {
                packet = packet_rx.recv() => {
                    if let Some((socket, data, addr)) = packet {
                        self.metrics.packets_received_total.with_label_values(&["server"]).inc();
                        self.metrics.bytes_received_total.with_label_values(&["server"]).inc_by(data.len() as u64);
                        {
                            let mut stats = self.stats.write();
                            stats.packets_received += 1;
                            stats.bytes_received += data.len() as u64;
                        }

                        if let Err(e) = self.handle_packet(&socket, &data, addr).await {
                            debug!("Error handling packet from {}: {}", addr, e);
                            self.stats.write().packets_dropped += 1;
                            self.metrics.packets_dropped_total.with_label_values(&["server", "error"]).inc();
                        }
                    } else {
                        break;
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Trigger shutdown.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    fn start_background_tasks(&self) {
        let transport_sessions = self.transport_sessions.clone();
        let timeout = self.config.idle_timeout;
        let metrics = Arc::clone(&self.metrics);
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                let before = transport_sessions.len();
                transport_sessions.retain(|_, session| !session.is_expired(timeout));
                let removed = before - transport_sessions.len();
                if removed > 0 {
                    info!("Cleaned up {} expired sessions", removed);
                    metrics.sessions_active.set(transport_sessions.len() as i64);
                    metrics
                        .connections_active
                        .set(transport_sessions.len() as i64);
                    let mut stats = stats.write();
                    stats.active_connections =
                        stats.active_connections.saturating_sub(removed as u64);
                }
            }
        });

        let metrics = Arc::clone(&self.metrics);
        let start_time = self.start_time;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                metrics
                    .server_uptime_seconds
                    .set(start_time.elapsed().as_secs_f64());
            }
        });
    }

    fn start_receive_tasks(&self, packet_tx: mpsc::Sender<(Arc<UdpSocket>, Vec<u8>, SocketAddr)>) {
        for socket in &self.sockets {
            let socket = Arc::clone(socket);
            let packet_tx = packet_tx.clone();
            let metrics = Arc::clone(&self.metrics);
            let mut shutdown_rx = self.shutdown_tx.subscribe();

            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];

                loop {
                    tokio::select! {
                        result = socket.recv_from(&mut buf) => {
                            match result {
                                Ok((len, addr)) => {
                                    if packet_tx.send((Arc::clone(&socket), buf[..len].to_vec(), addr)).await.is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    error!("Receive error on {}: {}", socket.local_addr().map_or_else(|_| "<unknown>".to_string(), |addr| addr.to_string()), e);
                                    metrics.record_error("receive");
                                }
                            }
                        }
                        _ = shutdown_rx.recv() => {
                            break;
                        }
                    }
                }
            });
        }
    }

    async fn handle_packet(&self, socket: &UdpSocket, data: &[u8], addr: SocketAddr) -> Result<()> {
        if data.len() < HEADER_SIZE {
            return Err(Error::InvalidPacket("Packet too short".into()));
        }

        let packet = Packet::decode(data)?;
        let session_id = packet.header.session_id;

        let session = if let Some(session) = self.transport_sessions.get(&session_id) {
            session.clone()
        } else {
            let session = Arc::new(TransportSession::new(session_id, addr));
            self.transport_sessions.insert(session_id, session.clone());
            self.sessions_by_addr.insert(addr, session_id);

            self.stats.write().total_connections += 1;
            self.stats.write().active_connections += 1;

            self.metrics.connections_total.inc();
            self.metrics.connections_active.inc();
            self.metrics.sessions_total.inc();
            self.metrics.sessions_active.inc();

            info!("New session {} from {}", session_id, addr);
            session
        };

        session.touch();
        session.add_address(addr);

        match packet.header.packet_type {
            PacketType::Handshake => {
                self.metrics.handshakes_total.inc();
                self.handle_handshake(socket, &session, &packet, addr)
                    .await?;
            }
            PacketType::Data => {
                self.handle_data(socket, &session, &packet, addr).await?;
            }
            PacketType::Ping => {
                self.handle_ping(socket, &session, &packet, addr).await?;
            }
            PacketType::Close => {
                self.handle_close(&session, addr).await?;
            }
            _ => {
                debug!("Unhandled packet type: {:?}", packet.header.packet_type);
            }
        }

        Ok(())
    }

    async fn handle_handshake(
        &self,
        socket: &UdpSocket,
        session: &TransportSession,
        packet: &Packet,
        addr: SocketAddr,
    ) -> Result<()> {
        debug!("Handshake from {} (session {})", addr, session.id);

        let mut noise = NoiseSession::new_responder(&self.keypair.secret)?;
        let _payload = noise.read_handshake(&packet.payload)?;
        let response = noise.write_handshake(&[])?;

        let response_packet = Packet::new(
            PacketType::Handshake,
            packet.header.sequence.next(),
            session.id,
            packet.header.uplink_id,
            response,
        )?;

        self.send_packet(socket, &response_packet, addr).await?;
        *session.noise.write() = Some(noise);

        info!("Handshake complete with {} (session {})", addr, session.id);
        Ok(())
    }

    async fn handle_data(
        &self,
        socket: &UdpSocket,
        session: &TransportSession,
        packet: &Packet,
        addr: SocketAddr,
    ) -> Result<()> {
        let payload = if packet.header.flags.has(PacketFlags::ENCRYPTED) {
            if let Some(ref mut noise) = *session.noise.write() {
                if noise.is_transport() {
                    self.metrics.decrypt_operations.inc();
                    noise.decrypt(&packet.payload)?
                } else {
                    packet.payload.clone()
                }
            } else {
                return Err(Error::Protocol(
                    crate::error::ProtocolError::UnexpectedMessage {
                        expected: "unencrypted or established session".into(),
                        got: "encrypted without session".into(),
                    },
                ));
            }
        } else {
            packet.payload.clone()
        };

        {
            let mut stats = session.stats.write();
            stats.bytes_received += payload.len() as u64;
            stats.packets_received += 1;
        }

        debug!(
            "Received {} bytes of data from {} (session {})",
            payload.len(),
            addr,
            session.id
        );

        self.send_data(socket, session, &payload, packet.header.uplink_id, addr)
            .await
    }

    async fn send_data(
        &self,
        socket: &UdpSocket,
        session: &TransportSession,
        payload: &[u8],
        uplink_id: u16,
        addr: SocketAddr,
    ) -> Result<()> {
        let (encrypted_payload, is_encrypted) = if let Some(ref mut noise) = *session.noise.write()
        {
            if noise.is_transport() {
                self.metrics.encrypt_operations.inc();
                (noise.encrypt(payload)?, true)
            } else {
                (payload.to_vec(), false)
            }
        } else {
            (payload.to_vec(), false)
        };

        let mut response =
            Packet::data(SequenceNumber(1), session.id, uplink_id, encrypted_payload)?;

        if is_encrypted {
            response.set_flag(PacketFlags::ENCRYPTED);
        }

        self.send_packet(socket, &response, addr).await?;

        {
            let mut stats = session.stats.write();
            stats.bytes_sent += payload.len() as u64;
            stats.packets_sent += 1;
        }

        Ok(())
    }

    async fn handle_ping(
        &self,
        socket: &UdpSocket,
        session: &TransportSession,
        packet: &Packet,
        addr: SocketAddr,
    ) -> Result<()> {
        let pong = Packet::pong(
            packet.header.sequence.next(),
            session.id,
            packet.header.uplink_id,
            packet.header.timestamp,
        )?;

        self.send_packet(socket, &pong, addr).await
    }

    async fn handle_close(&self, session: &TransportSession, addr: SocketAddr) -> Result<()> {
        info!("Session {} closed by {}", session.id, addr);

        let duration = session.last_activity.read().elapsed();
        self.metrics
            .session_duration_seconds
            .observe(duration.as_secs_f64());

        self.transport_sessions.remove(&session.id);
        self.sessions_by_addr.remove(&addr);
        {
            let mut stats = self.stats.write();
            stats.active_connections = stats.active_connections.saturating_sub(1);
        }

        self.metrics.connections_active.dec();
        self.metrics.sessions_active.dec();

        Ok(())
    }

    async fn send_packet(
        &self,
        socket: &UdpSocket,
        packet: &Packet,
        addr: SocketAddr,
    ) -> Result<()> {
        let data = packet.encode()?;

        socket
            .send_to(&data, addr)
            .await
            .map_err(|e| crate::error::TransportError::SendFailed(e.to_string()))?;

        self.stats.write().packets_sent += 1;
        self.stats.write().bytes_sent += data.len() as u64;

        self.metrics
            .packets_sent_total
            .with_label_values(&["server"])
            .inc();
        self.metrics
            .bytes_sent_total
            .with_label_values(&["server"])
            .inc_by(data.len() as u64);

        Ok(())
    }
}

struct ServerStatusProvider {
    start_time: Instant,
    transport_sessions: DashMap<SessionId, Arc<TransportSession>>,
    stats: Arc<RwLock<ServerStats>>,
}

impl StatusProvider for ServerStatusProvider {
    fn get_status(&self) -> StatusResponse {
        let stats = self.stats.read();

        let sessions: Vec<SessionStatus> = self
            .transport_sessions
            .iter()
            .take(100)
            .map(|entry| {
                let session = entry.value();
                let addrs = session.client_addrs.read();
                let session_stats = session.stats.read();

                SessionStatus {
                    id: session.id.to_string(),
                    user_id: session.user_id.read().clone(),
                    remote_addrs: addrs.iter().map(|a| a.to_string()).collect(),
                    connected_at: String::new(),
                    bytes_sent: session_stats.bytes_sent,
                    bytes_received: session_stats.bytes_received,
                    uplinks_used: vec![],
                }
            })
            .collect();

        StatusResponse {
            version: crate::VERSION.to_string(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            state: "running".to_string(),
            role: Some("server".to_string()),
            mode: Some("server".to_string()),
            process_id: Some(std::process::id()),
            session_id: None,
            connection_id: None,
            quality: None,
            tunnel: None,
            uplinks: vec![],
            sessions,
            total_bytes_sent: stats.bytes_sent,
            total_bytes_received: stats.bytes_received,
            total_connections: stats.total_connections,
        }
    }
}

/// Run a UDP Triglav server with metrics and signal handling.
pub async fn run_server(options: ServerRuntimeOptions) -> Result<()> {
    let metrics = init_metrics();
    let server =
        Arc::new(TriglavServer::new(options.config, options.keypair, Arc::clone(&metrics)).await?);

    let status_provider = Arc::new(ServerStatusProvider {
        start_time: server.start_time,
        transport_sessions: server.transport_sessions.clone(),
        stats: Arc::clone(&server.stats),
    });

    let health_checker = Arc::new(DefaultHealthChecker::new());
    health_checker.set_ready(true);

    let http_config = HttpServerConfig {
        bind_addr: options.metrics_addr,
        enable_cors: true,
        shutdown_timeout: Duration::from_secs(5),
    };
    let http_server = MetricsHttpServer::new(http_config, metrics, status_provider, health_checker);

    tokio::spawn(async move {
        if let Err(e) = http_server.start().await {
            error!("Metrics HTTP server error: {}", e);
        }
    });

    let signal_handler = SignalHandler::new();
    signal_handler.set_reload_callback(|| {
        info!("Received reload signal - reloading configuration is not implemented yet");
    });

    let mut signal_rx = signal_handler.subscribe();
    let server_for_shutdown = Arc::clone(&server);
    tokio::spawn(async move {
        while let Ok(signal) = signal_rx.recv().await {
            match signal {
                Signal::Terminate | Signal::Interrupt => {
                    info!("Received shutdown signal");
                    server_for_shutdown.shutdown();
                    break;
                }
                Signal::Hangup => {
                    info!("Received HUP signal");
                }
                Signal::User1 => {
                    info!("Received USR1 - dumping stats");
                    let stats = server_for_shutdown.stats.read();
                    info!("Stats: {:?}", *stats);
                }
                Signal::User2 => {
                    info!("Received USR2");
                }
                Signal::Child => {}
            }
        }
    });

    tokio::spawn(async move {
        signal_handler.listen().await;
    });

    server.run().await
}
