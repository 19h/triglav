//! Triglav Server Binary
//!
//! Dedicated server process for handling client connections.

use std::net::SocketAddr;
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
use triglav::protocol::{Packet, PacketType, PacketFlags, HEADER_SIZE};
use triglav::types::{SessionId, SequenceNumber, TrafficStats};

/// Server state.
struct Server {
    /// Server configuration.
    config: ServerConfig,
    /// Server keypair.
    keypair: KeyPair,
    /// Active sessions.
    sessions: DashMap<SessionId, Arc<Session>>,
    /// Session by client address.
    sessions_by_addr: DashMap<SocketAddr, SessionId>,
    /// UDP socket.
    socket: Arc<UdpSocket>,
    /// Statistics.
    stats: RwLock<ServerStats>,
    /// Shutdown signal.
    _shutdown: broadcast::Sender<()>,
}

/// Server statistics.
#[derive(Debug, Default)]
struct ServerStats {
    total_connections: u64,
    active_connections: u64,
    bytes_sent: u64,
    bytes_received: u64,
    packets_sent: u64,
    packets_received: u64,
    packets_dropped: u64,
}

/// Client session.
struct Session {
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
}

impl Session {
    fn new(id: SessionId, client_addr: SocketAddr) -> Self {
        Self {
            id,
            client_addrs: RwLock::new(vec![client_addr]),
            noise: RwLock::new(None),
            last_activity: RwLock::new(Instant::now()),
            stats: RwLock::new(TrafficStats::default()),
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
    async fn new(config: ServerConfig, keypair: KeyPair) -> Result<Self> {
        let addr = config.listen_addrs.first()
            .ok_or_else(|| Error::Config("No listen address specified".into()))?;

        let socket = UdpSocket::bind(addr)
            .await
            .map_err(|e| Error::Transport(triglav::error::TransportError::BindFailed {
                addr: *addr,
                reason: e.to_string(),
            }))?;

        info!("Server bound to {}", addr);

        let (shutdown, _) = broadcast::channel(1);

        Ok(Self {
            config,
            keypair,
            sessions: DashMap::new(),
            sessions_by_addr: DashMap::new(),
            socket: Arc::new(socket),
            stats: RwLock::new(ServerStats::default()),
            _shutdown: shutdown,
        })
    }

    /// Run the server.
    async fn run(&self) -> Result<()> {
        let mut buf = vec![0u8; 65536];
        let mut shutdown_rx = self._shutdown.subscribe();

        // Spawn cleanup task
        let sessions = self.sessions.clone();
        let timeout = self.config.idle_timeout;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                sessions.retain(|_, session| !session.is_expired(timeout));
            }
        });

        loop {
            tokio::select! {
                result = self.socket.recv_from(&mut buf) => {
                    match result {
                        Ok((len, addr)) => {
                            self.stats.write().packets_received += 1;
                            self.stats.write().bytes_received += len as u64;

                            if let Err(e) = self.handle_packet(&buf[..len], addr).await {
                                debug!("Error handling packet from {}: {}", addr, e);
                                self.stats.write().packets_dropped += 1;
                            }
                        }
                        Err(e) => {
                            error!("Receive error: {}", e);
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
        let session = if let Some(s) = self.sessions.get(&session_id) {
            s.clone()
        } else {
            // New session
            let session = Arc::new(Session::new(session_id, addr));
            self.sessions.insert(session_id, session.clone());
            self.sessions_by_addr.insert(addr, session_id);

            self.stats.write().total_connections += 1;
            self.stats.write().active_connections += 1;

            info!("New session {} from {}", session_id, addr);
            session
        };

        // Update session
        session.touch();
        session.add_address(addr);

        // Handle packet type
        match packet.header.packet_type {
            PacketType::Handshake => {
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
    async fn handle_handshake(&self, session: &Session, packet: &Packet, addr: SocketAddr) -> Result<()> {
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
    async fn handle_data(&self, session: &Session, packet: &Packet, addr: SocketAddr) -> Result<()> {
        // Decrypt if we have a noise session
        let payload = if packet.header.flags.has(PacketFlags::ENCRYPTED) {
            if let Some(ref mut noise) = *session.noise.write() {
                if noise.is_transport() {
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
    async fn send_data(&self, session: &Session, payload: &[u8], uplink_id: u16, addr: SocketAddr) -> Result<()> {
        // Encrypt if we have a noise session
        let (encrypted_payload, is_encrypted) = if let Some(ref mut noise) = *session.noise.write() {
            if noise.is_transport() {
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
    async fn handle_ping(&self, session: &Session, packet: &Packet, addr: SocketAddr) -> Result<()> {
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
    async fn handle_close(&self, session: &Session, addr: SocketAddr) -> Result<()> {
        info!("Session {} closed by {}", session.id, addr);

        self.sessions.remove(&session.id);
        self.sessions_by_addr.remove(&addr);
        self.stats.write().active_connections -= 1;

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

        Ok(())
    }

}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line args using the same CLI as main binary
    let args: Vec<String> = std::env::args().collect();

    // Simple argument parsing for standalone server
    let mut listen_addr: SocketAddr = "0.0.0.0:7443".parse().unwrap();
    let mut key_path: Option<std::path::PathBuf> = None;
    let mut generate_key = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-l" | "--listen" => {
                if i + 1 < args.len() {
                    listen_addr = args[i + 1].parse().unwrap();
                    i += 1;
                }
            }
            "-k" | "--key" => {
                if i + 1 < args.len() {
                    key_path = Some(std::path::PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--generate-key" => {
                generate_key = true;
            }
            "-h" | "--help" => {
                println!("Triglav Server");
                println!();
                println!("Usage: triglav-server [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -l, --listen <ADDR>    Listen address (default: 0.0.0.0:7443)");
                println!("  -k, --key <PATH>       Path to key file");
                println!("      --generate-key     Generate new key if not exists");
                println!("  -h, --help             Show this help");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
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

    // Create server config
    let config = ServerConfig {
        enabled: true,
        listen_addrs: vec![listen_addr],
        ..Default::default()
    };

    // Print connection key
    let auth_key = triglav::types::AuthKey::new(*keypair.public.as_bytes(), vec![listen_addr]);
    println!();
    println!("=== Triglav Server ===");
    println!();
    println!("Listening on: {}", listen_addr);
    println!();
    println!("Client Connection Key:");
    println!("{}", auth_key);
    println!();

    // Create and run server
    let server = Server::new(config, keypair).await?;

    // Handle Ctrl+C
    let _server_ref = &server;
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        info!("Shutting down...");
    });

    server.run().await?;

    Ok(())
}
