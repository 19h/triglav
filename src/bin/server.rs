//! Triglav Server Binary
//!
//! Dedicated server process for handling client connections with:
//! - User authentication and management
//! - Session tracking
//! - Prometheus metrics export
//! - Signal handling for graceful shutdown
//! - Optional daemon mode

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::net::UdpSocket;
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use triglav::config::{ServerConfig, init_logging};
use triglav::crypto::{KeyPair, NoiseSession};
use triglav::error::{Error, Result};
use triglav::metrics::{
    MetricsHttpServer, HttpServerConfig, DefaultHealthChecker,
    SessionStatus, StatusResponse, StatusProvider,
    init_metrics, PrometheusMetrics,
};
use triglav::protocol::{Packet, PacketType, PacketFlags, HEADER_SIZE};
use triglav::server::{
    DaemonConfig, daemonize, PidFileGuard,
    SignalHandler, Signal,
};
use triglav::types::{SessionId, SequenceNumber, TrafficStats};

/// Server state.
struct Server {
    /// Server configuration.
    config: ServerConfig,
    /// Server keypair.
    keypair: KeyPair,
    /// Active sessions (transport-level).
    transport_sessions: DashMap<SessionId, Arc<TransportSession>>,
    /// Session by client address.
    sessions_by_addr: DashMap<SocketAddr, SessionId>,
    /// UDP socket.
    socket: Arc<UdpSocket>,
    /// Prometheus metrics.
    metrics: Arc<PrometheusMetrics>,
    /// Statistics.
    stats: RwLock<ServerStats>,
    /// Start time.
    start_time: Instant,
    /// Shutdown signal.
    shutdown_tx: broadcast::Sender<()>,
}

/// Server statistics.
#[derive(Debug, Default, Clone)]
struct ServerStats {
    total_connections: u64,
    active_connections: u64,
    bytes_sent: u64,
    bytes_received: u64,
    packets_sent: u64,
    packets_received: u64,
    packets_dropped: u64,
}

/// Client transport session.
struct TransportSession {
    /// Session ID.
    id: SessionId,
    /// Client addresses (multiple uplinks).
    client_addrs: RwLock<Vec<SocketAddr>>,
    /// Noise session.
    noise: RwLock<Option<NoiseSession>>,
    /// Last activity.
    last_activity: RwLock<Instant>,
    /// Traffic stats.
    stats: RwLock<TrafficStats>,
    /// Authenticated user ID.
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

impl Server {
    /// Create a new server.
    async fn new(
        config: ServerConfig,
        keypair: KeyPair,
        metrics: Arc<PrometheusMetrics>,
    ) -> Result<Self> {
        let addr = config.listen_addrs.first()
            .ok_or_else(|| Error::Config("No listen address specified".into()))?;

        let socket = UdpSocket::bind(addr)
            .await
            .map_err(|e| Error::Transport(triglav::error::TransportError::BindFailed {
                addr: *addr,
                reason: e.to_string(),
            }))?;

        info!("Server bound to {}", addr);

        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            config,
            keypair,
            transport_sessions: DashMap::new(),
            sessions_by_addr: DashMap::new(),
            socket: Arc::new(socket),
            metrics,
            stats: RwLock::new(ServerStats::default()),
            start_time: Instant::now(),
            shutdown_tx,
        })
    }

    /// Run the server.
    async fn run(&self) -> Result<()> {
        let mut buf = vec![0u8; 65536];
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        // Spawn cleanup task
        let transport_sessions = self.transport_sessions.clone();
        let timeout = self.config.idle_timeout;
        let metrics = Arc::clone(&self.metrics);
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
                }
            }
        });

        // Spawn metrics update task
        let metrics = Arc::clone(&self.metrics);
        let start_time = self.start_time;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                metrics.server_uptime_seconds.set(start_time.elapsed().as_secs_f64());
            }
        });

        info!("Server running, waiting for connections...");

        loop {
            tokio::select! {
                result = self.socket.recv_from(&mut buf) => {
                    match result {
                        Ok((len, addr)) => {
                            self.metrics.packets_received_total.with_label_values(&["server"]).inc();
                            self.metrics.bytes_received_total.with_label_values(&["server"]).inc_by(len as u64);
                            self.stats.write().packets_received += 1;
                            self.stats.write().bytes_received += len as u64;

                            if let Err(e) = self.handle_packet(&buf[..len], addr).await {
                                debug!("Error handling packet from {}: {}", addr, e);
                                self.stats.write().packets_dropped += 1;
                                self.metrics.packets_dropped_total.with_label_values(&["server", "error"]).inc();
                            }
                        }
                        Err(e) => {
                            error!("Receive error: {}", e);
                            self.metrics.record_error("receive");
                        }
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

    /// Handle an incoming packet.
    async fn handle_packet(&self, data: &[u8], addr: SocketAddr) -> Result<()> {
        if data.len() < HEADER_SIZE {
            return Err(Error::InvalidPacket("Packet too short".into()));
        }

        let packet = Packet::decode(data)?;
        let session_id = packet.header.session_id;

        // Get or create session
        let session = if let Some(s) = self.transport_sessions.get(&session_id) {
            s.clone()
        } else {
            // New session
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

        // Update session
        session.touch();
        session.add_address(addr);

        // Handle packet type
        match packet.header.packet_type {
            PacketType::Handshake => {
                self.metrics.handshakes_total.inc();
                self.handle_handshake(&session, &packet, addr).await?;
            }
            PacketType::Data => {
                self.handle_data(&session, &packet, addr).await?;
            }
            PacketType::Ping => {
                self.handle_ping(&session, &packet, addr).await?;
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

    /// Handle handshake packet.
    async fn handle_handshake(&self, session: &TransportSession, packet: &Packet, addr: SocketAddr) -> Result<()> {
        debug!("Handshake from {} (session {})", addr, session.id);

        // Create responder noise session
        let mut noise = NoiseSession::new_responder(&self.keypair.secret)?;

        // Process handshake message
        let _payload = noise.read_handshake(&packet.payload)?;

        // Send response
        let response = noise.write_handshake(&[])?;

        let response_packet = Packet::new(
            PacketType::Handshake,
            packet.header.sequence.next(),
            session.id,
            packet.header.uplink_id,
            response,
        )?;

        self.send_packet(&response_packet, addr).await?;

        // Store noise session
        *session.noise.write() = Some(noise);

        info!("Handshake complete with {} (session {})", addr, session.id);

        Ok(())
    }

    /// Handle data packet.
    async fn handle_data(&self, session: &TransportSession, packet: &Packet, addr: SocketAddr) -> Result<()> {
        // Decrypt if we have a noise session
        let payload = if packet.header.flags.has(PacketFlags::ENCRYPTED) {
            if let Some(ref mut noise) = *session.noise.write() {
                if noise.is_transport() {
                    self.metrics.decrypt_operations.inc();
                    noise.decrypt(&packet.payload)?
                } else {
                    packet.payload.clone()
                }
            } else {
                return Err(Error::Protocol(triglav::error::ProtocolError::UnexpectedMessage {
                    expected: "unencrypted or established session".into(),
                    got: "encrypted without session".into(),
                }));
            }
        } else {
            packet.payload.clone()
        };

        // Update stats
        {
            let mut stats = session.stats.write();
            stats.bytes_received += payload.len() as u64;
            stats.packets_received += 1;
        }

        debug!("Received {} bytes of data from {} (session {})", payload.len(), addr, session.id);

        // Echo back the data (encrypted if we have a noise session)
        self.send_data(session, &payload, packet.header.uplink_id, addr).await?;

        Ok(())
    }

    /// Send encrypted data to a client.
    async fn send_data(&self, session: &TransportSession, payload: &[u8], uplink_id: u16, addr: SocketAddr) -> Result<()> {
        // Encrypt if we have a noise session
        let (encrypted_payload, is_encrypted) = if let Some(ref mut noise) = *session.noise.write() {
            if noise.is_transport() {
                self.metrics.encrypt_operations.inc();
                (noise.encrypt(payload)?, true)
            } else {
                (payload.to_vec(), false)
            }
        } else {
            (payload.to_vec(), false)
        };

        let mut response = Packet::data(
            SequenceNumber(1), // TODO: proper sequence tracking
            session.id,
            uplink_id,
            encrypted_payload,
        )?;

        if is_encrypted {
            response.set_flag(PacketFlags::ENCRYPTED);
        }

        self.send_packet(&response, addr).await?;

        // Update stats
        {
            let mut stats = session.stats.write();
            stats.bytes_sent += payload.len() as u64;
            stats.packets_sent += 1;
        }

        Ok(())
    }

    /// Handle ping packet.
    async fn handle_ping(&self, session: &TransportSession, packet: &Packet, addr: SocketAddr) -> Result<()> {
        let pong = Packet::pong(
            packet.header.sequence.next(),
            session.id,
            packet.header.uplink_id,
            packet.header.timestamp,
        )?;

        self.send_packet(&pong, addr).await?;

        Ok(())
    }

    /// Handle close packet.
    async fn handle_close(&self, session: &TransportSession, addr: SocketAddr) -> Result<()> {
        info!("Session {} closed by {}", session.id, addr);

        // Record session duration
        let duration = session.last_activity.read().elapsed();
        self.metrics.session_duration_seconds.observe(duration.as_secs_f64());

        self.transport_sessions.remove(&session.id);
        self.sessions_by_addr.remove(&addr);
        self.stats.write().active_connections -= 1;
        
        self.metrics.connections_active.dec();
        self.metrics.sessions_active.dec();

        Ok(())
    }

    /// Send a packet.
    async fn send_packet(&self, packet: &Packet, addr: SocketAddr) -> Result<()> {
        let data = packet.encode()?;

        self.socket.send_to(&data, addr)
            .await
            .map_err(|e| triglav::error::TransportError::SendFailed(e.to_string()))?;

        self.stats.write().packets_sent += 1;
        self.stats.write().bytes_sent += data.len() as u64;
        
        self.metrics.packets_sent_total.with_label_values(&["server"]).inc();
        self.metrics.bytes_sent_total.with_label_values(&["server"]).inc_by(data.len() as u64);

        Ok(())
    }
    
    /// Trigger shutdown.
    fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

/// Status provider implementation for the server.
struct ServerStatusProvider {
    start_time: Instant,
    transport_sessions: DashMap<SessionId, Arc<TransportSession>>,
    stats: Arc<RwLock<ServerStats>>,
}

impl StatusProvider for ServerStatusProvider {
    fn get_status(&self) -> StatusResponse {
        let stats = self.stats.read();
        
        let sessions: Vec<SessionStatus> = self.transport_sessions.iter()
            .take(100)
            .map(|entry| {
                let session = entry.value();
                let addrs = session.client_addrs.read();
                let session_stats = session.stats.read();
                
                SessionStatus {
                    id: session.id.to_string(),
                    user_id: session.user_id.read().clone(),
                    remote_addrs: addrs.iter().map(|a| a.to_string()).collect(),
                    connected_at: "".to_string(),
                    bytes_sent: session_stats.bytes_sent,
                    bytes_received: session_stats.bytes_received,
                    uplinks_used: vec![],
                }
            })
            .collect();
        
        StatusResponse {
            version: triglav::VERSION.to_string(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            state: "running".to_string(),
            uplinks: vec![],
            sessions,
            total_bytes_sent: stats.bytes_sent,
            total_bytes_received: stats.bytes_received,
            total_connections: stats.total_connections,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line args
    let args: Vec<String> = std::env::args().collect();

    // Simple argument parsing for standalone server
    let mut listen_addr: SocketAddr = "0.0.0.0:7443".parse().unwrap();
    let mut metrics_addr: SocketAddr = "0.0.0.0:9090".parse().unwrap();
    let mut key_path: Option<PathBuf> = None;
    let mut generate_key = false;
    let mut daemon_mode = false;
    let mut pid_file: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-l" | "--listen" => {
                if i + 1 < args.len() {
                    listen_addr = args[i + 1].parse().unwrap();
                    i += 1;
                }
            }
            "-m" | "--metrics" => {
                if i + 1 < args.len() {
                    metrics_addr = args[i + 1].parse().unwrap();
                    i += 1;
                }
            }
            "-k" | "--key" => {
                if i + 1 < args.len() {
                    key_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--generate-key" => {
                generate_key = true;
            }
            "-d" | "--daemon" => {
                daemon_mode = true;
            }
            "--pid-file" => {
                if i + 1 < args.len() {
                    pid_file = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "-h" | "--help" => {
                println!("Triglav Server");
                println!();
                println!("Usage: triglav-server [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -l, --listen <ADDR>    Listen address (default: 0.0.0.0:7443)");
                println!("  -m, --metrics <ADDR>   Metrics HTTP address (default: 0.0.0.0:9090)");
                println!("  -k, --key <PATH>       Path to key file");
                println!("      --generate-key     Generate new key if not exists");
                println!("  -d, --daemon           Run as daemon");
                println!("      --pid-file <PATH>  PID file path (for daemon mode)");
                println!("  -h, --help             Show this help");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    // Daemonize if requested
    let _pid_guard: Option<PidFileGuard>;
    if daemon_mode {
        let daemon_config = DaemonConfig {
            pid_file: pid_file.clone(),
            work_dir: PathBuf::from("/"),
            user: None,
            group: None,
            umask: Some(0o027),
            close_fds: true,
        };
        
        daemonize(&daemon_config)?;
        _pid_guard = pid_file.map(|p| PidFileGuard::new(p).expect("Failed to create PID file"));
    } else {
        _pid_guard = None;
    }

    // Initialize logging
    let log_config = triglav::config::LoggingConfig::default();
    init_logging(&log_config)?;

    // Load or generate keypair
    let keypair = if let Some(ref path) = key_path {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let secret = triglav::crypto::SecretKey::from_base64(content.trim())?;
            KeyPair::from_secret(secret)
        } else if generate_key {
            let kp = KeyPair::generate();
            std::fs::write(path, kp.secret.to_base64())?;
            info!("Generated new keypair at {}", path.display());
            kp
        } else {
            return Err(Error::Config(format!("Key file not found: {}", path.display())));
        }
    } else if generate_key {
        let kp = KeyPair::generate();
        warn!("Using ephemeral keypair (not saved)");
        kp
    } else {
        return Err(Error::Config("No key specified".into()));
    };

    // Initialize Prometheus metrics
    let metrics = init_metrics();

    // Create server config
    let config = ServerConfig {
        enabled: true,
        listen_addrs: vec![listen_addr],
        ..Default::default()
    };

    // Print connection key
    let auth_key = triglav::types::AuthKey::new(*keypair.public.as_bytes(), vec![listen_addr]);
    if !daemon_mode {
        println!();
        println!("╔══════════════════════════════════════════╗");
        println!("║     TRIGLAV SERVER                       ║");
        println!("║     Version {}                         ║", triglav::VERSION);
        println!("╚══════════════════════════════════════════╝");
        println!();
        println!("Listening on: {}", listen_addr);
        println!("Metrics at:   http://{}", metrics_addr);
        println!();
        println!("Client Connection Key:");
        println!("{}", auth_key);
        println!();
    }
    
    info!("Triglav server starting");
    info!("Listen: {}, Metrics: {}", listen_addr, metrics_addr);

    // Create server
    let server = Arc::new(
        Server::new(config, keypair, Arc::clone(&metrics)).await?
    );
    
    // Create a shared stats reference for the status provider
    let shared_stats: Arc<RwLock<ServerStats>> = Arc::new(RwLock::new(ServerStats::default()));
    
    // Create status provider that shares data with server
    let status_provider = Arc::new(ServerStatusProvider {
        start_time: server.start_time,
        transport_sessions: server.transport_sessions.clone(),
        stats: shared_stats,
    });
    
    // Create health checker
    let health_checker = Arc::new(DefaultHealthChecker::new());
    health_checker.set_ready(true);
    
    // Start metrics HTTP server
    let http_config = HttpServerConfig {
        bind_addr: metrics_addr,
        enable_cors: true,
        shutdown_timeout: Duration::from_secs(5),
    };
    let http_server = MetricsHttpServer::new(
        http_config,
        metrics,
        status_provider,
        health_checker,
    );
    
    tokio::spawn(async move {
        if let Err(e) = http_server.start().await {
            error!("Metrics HTTP server error: {}", e);
        }
    });

    // Setup signal handler
    let signal_handler = SignalHandler::new();
    signal_handler.set_reload_callback(|| {
        info!("Received reload signal - reloading configuration");
        // TODO: Implement config reload
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
                Signal::Child => {
                    // Child process exited
                }
            }
        }
    });
    
    // Start signal handler
    tokio::spawn(async move {
        signal_handler.listen().await;
    });

    // Run server
    server.run().await?;
    
    info!("Triglav server stopped");

    Ok(())
}
