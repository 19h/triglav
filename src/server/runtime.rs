//! UDP server runtime shared by `triglav server` and `triglav-server`.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::net::UdpSocket;
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio::time::timeout;
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
use crate::tun::{IpPacket, IpTransportProtocol, IpVersion};
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
    next_sequence: AtomicU64,
    udp_flows: DashMap<UdpExitKey, Arc<UdpExitFlow>>,
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
            next_sequence: AtomicU64::new(1),
            udp_flows: DashMap::new(),
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

    fn next_sequence(&self) -> SequenceNumber {
        SequenceNumber(self.next_sequence.fetch_add(1, Ordering::SeqCst))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct UdpExitKey {
    tunnel_src: IpAddr,
    tunnel_src_port: u16,
    remote_dst: IpAddr,
    remote_dst_port: u16,
}

struct UdpExitFlow {
    socket: Arc<UdpSocket>,
    read_lock: Mutex<()>,
}

#[derive(Debug)]
struct UdpExitDatagram {
    key: UdpExitKey,
    destination: SocketAddr,
    payload: Vec<u8>,
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

                        if let Err(e) = self.handle_packet(Arc::clone(&socket), &data, addr).await {
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

    async fn handle_packet(
        &self,
        socket: Arc<UdpSocket>,
        data: &[u8],
        addr: SocketAddr,
    ) -> Result<()> {
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
                self.handle_handshake(&socket, &session, &packet, addr)
                    .await?;
            }
            PacketType::Data => {
                self.handle_data(socket, session, &packet, addr).await?;
            }
            PacketType::Ping => {
                self.handle_ping(&socket, &session, &packet, addr).await?;
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
            session.next_sequence(),
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
        socket: Arc<UdpSocket>,
        session: Arc<TransportSession>,
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

        self.send_ack(
            &socket,
            &session,
            packet.header.sequence.0,
            packet.header.uplink_id,
            addr,
        )
        .await?;

        let datagram = match parse_udp_datagram(&payload) {
            Ok(Some(datagram)) => datagram,
            Ok(None) => {
                if let Some(response) = build_ipv4_protocol_unreachable(&payload)? {
                    send_data_packet(
                        &socket,
                        &session,
                        &self.metrics,
                        &self.stats,
                        &response,
                        packet.header.uplink_id,
                        addr,
                    )
                    .await?;
                }
                debug!(
                    session = %session.id,
                    "Rejected unsupported tunneled packet; UDP/IPv4 forwarding is available"
                );
                return Ok(());
            }
            Err(e) => {
                debug!(error = %e, session = %session.id, "Dropping malformed tunneled packet");
                return Ok(());
            }
        };

        let flow = self
            .get_or_create_udp_flow(&session, datagram.key.clone())
            .await?;
        let metrics = Arc::clone(&self.metrics);
        let stats = Arc::clone(&self.stats);
        let uplink_id = packet.header.uplink_id;

        tokio::spawn(async move {
            if let Err(e) = relay_udp_datagram(
                socket, metrics, stats, session, flow, datagram, uplink_id, addr,
            )
            .await
            {
                debug!(error = %e, "UDP exit relay failed");
            }
        });

        Ok(())
    }

    async fn get_or_create_udp_flow(
        &self,
        session: &TransportSession,
        key: UdpExitKey,
    ) -> Result<Arc<UdpExitFlow>> {
        if let Some(flow) = session.udp_flows.get(&key) {
            return Ok(flow.clone());
        }

        let bind_addr = match key.remote_dst {
            IpAddr::V4(_) => SocketAddr::from(([0, 0, 0, 0], 0)),
            IpAddr::V6(_) => SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], 0)),
        };
        let socket = UdpSocket::bind(bind_addr).await.map_err(|e| {
            Error::Transport(crate::error::TransportError::BindFailed {
                addr: bind_addr,
                reason: e.to_string(),
            })
        })?;

        let flow = Arc::new(UdpExitFlow {
            socket: Arc::new(socket),
            read_lock: Mutex::new(()),
        });

        match session.udp_flows.entry(key) {
            dashmap::mapref::entry::Entry::Occupied(existing) => Ok(existing.get().clone()),
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                vacant.insert(flow.clone());
                Ok(flow)
            }
        }
    }

    async fn send_ack(
        &self,
        socket: &UdpSocket,
        session: &TransportSession,
        acked_sequence: u64,
        uplink_id: u16,
        addr: SocketAddr,
    ) -> Result<()> {
        let ack = Packet::ack(
            session.next_sequence(),
            session.id,
            uplink_id,
            &[acked_sequence],
        )?;

        self.send_packet(socket, &ack, addr).await
    }

    async fn send_packet(
        &self,
        socket: &UdpSocket,
        packet: &Packet,
        addr: SocketAddr,
    ) -> Result<()> {
        send_packet_with_metrics(socket, packet, addr, &self.metrics, &self.stats).await
    }
}

async fn send_data_packet(
    socket: &UdpSocket,
    session: &TransportSession,
    metrics: &PrometheusMetrics,
    server_stats: &RwLock<ServerStats>,
    payload: &[u8],
    uplink_id: u16,
    addr: SocketAddr,
) -> Result<()> {
    let (encrypted_payload, is_encrypted) = if let Some(ref mut noise) = *session.noise.write() {
        if noise.is_transport() {
            metrics.encrypt_operations.inc();
            (noise.encrypt(payload)?, true)
        } else {
            (payload.to_vec(), false)
        }
    } else {
        (payload.to_vec(), false)
    };

    let mut response = Packet::data(
        session.next_sequence(),
        session.id,
        uplink_id,
        encrypted_payload,
    )?;

    if is_encrypted {
        response.set_flag(PacketFlags::ENCRYPTED);
    }

    send_packet_with_metrics(socket, &response, addr, metrics, server_stats).await?;

    {
        let mut stats = session.stats.write();
        stats.bytes_sent += payload.len() as u64;
        stats.packets_sent += 1;
    }

    Ok(())
}

async fn relay_udp_datagram(
    client_socket: Arc<UdpSocket>,
    metrics: Arc<PrometheusMetrics>,
    server_stats: Arc<RwLock<ServerStats>>,
    session: Arc<TransportSession>,
    flow: Arc<UdpExitFlow>,
    datagram: UdpExitDatagram,
    uplink_id: u16,
    client_addr: SocketAddr,
) -> Result<()> {
    let _read_guard = flow.read_lock.lock().await;

    flow.socket
        .send_to(&datagram.payload, datagram.destination)
        .await
        .map_err(|e| crate::error::TransportError::SendFailed(e.to_string()))?;

    let mut buf = vec![0u8; 65535];
    let deadline = Duration::from_secs(2);

    loop {
        let (len, response_addr) = match timeout(deadline, flow.socket.recv_from(&mut buf)).await {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                return Err(crate::error::TransportError::ReceiveFailed(e.to_string()).into());
            }
            Err(_) => return Ok(()),
        };

        if response_addr.ip() != datagram.key.remote_dst
            || response_addr.port() != datagram.key.remote_dst_port
        {
            debug!(
                expected = %datagram.destination,
                actual = %response_addr,
                "Ignoring UDP response from unexpected endpoint"
            );
            continue;
        }

        let packet = build_udp_packet(
            datagram.key.remote_dst,
            datagram.key.tunnel_src,
            datagram.key.remote_dst_port,
            datagram.key.tunnel_src_port,
            &buf[..len],
        )?;

        send_data_packet(
            &client_socket,
            &session,
            &metrics,
            &server_stats,
            &packet,
            uplink_id,
            client_addr,
        )
        .await?;
        return Ok(());
    }
}

impl TriglavServer {
    async fn handle_ping(
        &self,
        socket: &UdpSocket,
        session: &TransportSession,
        packet: &Packet,
        addr: SocketAddr,
    ) -> Result<()> {
        let pong = Packet::pong(
            session.next_sequence(),
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
}

async fn send_packet_with_metrics(
    socket: &UdpSocket,
    packet: &Packet,
    addr: SocketAddr,
    metrics: &PrometheusMetrics,
    stats: &RwLock<ServerStats>,
) -> Result<()> {
    let data = packet.encode()?;

    socket
        .send_to(&data, addr)
        .await
        .map_err(|e| crate::error::TransportError::SendFailed(e.to_string()))?;

    stats.write().packets_sent += 1;
    stats.write().bytes_sent += data.len() as u64;

    metrics
        .packets_sent_total
        .with_label_values(&["server"])
        .inc();
    metrics
        .bytes_sent_total
        .with_label_values(&["server"])
        .inc_by(data.len() as u64);

    Ok(())
}

fn parse_udp_datagram(packet: &[u8]) -> Result<Option<UdpExitDatagram>> {
    let parsed = IpPacket::parse(packet)?;
    if parsed.protocol != IpTransportProtocol::Udp {
        return Ok(None);
    }

    let src_port = match parsed.src_port {
        Some(port) => port,
        None => return Ok(None),
    };
    let dst_port = match parsed.dst_port {
        Some(port) => port,
        None => return Ok(None),
    };

    if parsed.total_len > packet.len() || parsed.header_len + 8 > parsed.total_len {
        return Err(Error::InvalidPacket("Malformed UDP packet".into()));
    }

    let udp_len_offset = parsed.header_len + 4;
    let udp_len = u16::from_be_bytes([packet[udp_len_offset], packet[udp_len_offset + 1]]) as usize;
    if udp_len < 8 || parsed.header_len + udp_len > parsed.total_len {
        return Err(Error::InvalidPacket("Malformed UDP length".into()));
    }

    Ok(Some(UdpExitDatagram {
        key: UdpExitKey {
            tunnel_src: parsed.src_addr,
            tunnel_src_port: src_port,
            remote_dst: parsed.dst_addr,
            remote_dst_port: dst_port,
        },
        destination: SocketAddr::from((parsed.dst_addr, dst_port)),
        payload: packet[parsed.header_len + 8..parsed.header_len + udp_len].to_vec(),
    }))
}

fn build_udp_packet(
    src_addr: IpAddr,
    dst_addr: IpAddr,
    src_port: u16,
    dst_port: u16,
    payload: &[u8],
) -> Result<Vec<u8>> {
    match (src_addr, dst_addr) {
        (IpAddr::V4(src), IpAddr::V4(dst)) => {
            build_ipv4_udp_packet(src, dst, src_port, dst_port, payload)
        }
        (IpAddr::V6(src), IpAddr::V6(dst)) => {
            build_ipv6_udp_packet(src, dst, src_port, dst_port, payload)
        }
        _ => Err(Error::InvalidPacket(
            "UDP response address version mismatch".into(),
        )),
    }
}

fn build_ipv4_udp_packet(
    src_addr: Ipv4Addr,
    dst_addr: Ipv4Addr,
    src_port: u16,
    dst_port: u16,
    payload: &[u8],
) -> Result<Vec<u8>> {
    let udp_len = 8usize
        .checked_add(payload.len())
        .ok_or_else(|| Error::InvalidPacket("UDP payload too large".into()))?;
    let total_len = 20usize
        .checked_add(udp_len)
        .ok_or_else(|| Error::InvalidPacket("IPv4 packet too large".into()))?;

    if udp_len > u16::MAX as usize || total_len > u16::MAX as usize {
        return Err(Error::InvalidPacket("IPv4 UDP packet too large".into()));
    }

    let mut packet = vec![0u8; total_len];
    packet[0] = 0x45;
    packet[2..4].copy_from_slice(&(total_len as u16).to_be_bytes());
    packet[6..8].copy_from_slice(&0x4000u16.to_be_bytes());
    packet[8] = 64;
    packet[9] = IpTransportProtocol::Udp.protocol_number();
    packet[12..16].copy_from_slice(&src_addr.octets());
    packet[16..20].copy_from_slice(&dst_addr.octets());

    let header_checksum = internet_checksum(&packet[..20]);
    packet[10..12].copy_from_slice(&header_checksum.to_be_bytes());

    packet[20..22].copy_from_slice(&src_port.to_be_bytes());
    packet[22..24].copy_from_slice(&dst_port.to_be_bytes());
    packet[24..26].copy_from_slice(&(udp_len as u16).to_be_bytes());
    packet[28..].copy_from_slice(payload);

    let udp_checksum = udp_ipv4_checksum(src_addr, dst_addr, &packet[20..]);
    packet[26..28].copy_from_slice(&udp_checksum.to_be_bytes());

    Ok(packet)
}

fn build_ipv6_udp_packet(
    src_addr: Ipv6Addr,
    dst_addr: Ipv6Addr,
    src_port: u16,
    dst_port: u16,
    payload: &[u8],
) -> Result<Vec<u8>> {
    let udp_len = 8usize
        .checked_add(payload.len())
        .ok_or_else(|| Error::InvalidPacket("UDP payload too large".into()))?;
    let total_len = 40usize
        .checked_add(udp_len)
        .ok_or_else(|| Error::InvalidPacket("IPv6 packet too large".into()))?;

    if udp_len > u16::MAX as usize {
        return Err(Error::InvalidPacket("IPv6 UDP packet too large".into()));
    }

    let mut packet = vec![0u8; total_len];
    packet[0] = 0x60;
    packet[4..6].copy_from_slice(&(udp_len as u16).to_be_bytes());
    packet[6] = IpTransportProtocol::Udp.protocol_number();
    packet[7] = 64;
    packet[8..24].copy_from_slice(&src_addr.octets());
    packet[24..40].copy_from_slice(&dst_addr.octets());

    packet[40..42].copy_from_slice(&src_port.to_be_bytes());
    packet[42..44].copy_from_slice(&dst_port.to_be_bytes());
    packet[44..46].copy_from_slice(&(udp_len as u16).to_be_bytes());
    packet[48..].copy_from_slice(payload);

    let udp_checksum = udp_ipv6_checksum(src_addr, dst_addr, &packet[40..]);
    packet[46..48].copy_from_slice(&udp_checksum.to_be_bytes());

    Ok(packet)
}

fn build_ipv4_protocol_unreachable(original_packet: &[u8]) -> Result<Option<Vec<u8>>> {
    let parsed = match IpPacket::parse(original_packet) {
        Ok(packet) => packet,
        Err(_) => return Ok(None),
    };

    if parsed.version != IpVersion::V4 {
        return Ok(None);
    }

    let (src_addr, dst_addr) = match (parsed.src_addr, parsed.dst_addr) {
        (IpAddr::V4(src), IpAddr::V4(dst)) => (src, dst),
        _ => return Ok(None),
    };

    if parsed.total_len > original_packet.len() || parsed.header_len > parsed.total_len {
        return Ok(None);
    }

    let quote_len = (parsed.header_len + 8).min(parsed.total_len);
    let icmp_len = 8usize
        .checked_add(quote_len)
        .ok_or_else(|| Error::InvalidPacket("ICMP payload too large".into()))?;
    let total_len = 20usize
        .checked_add(icmp_len)
        .ok_or_else(|| Error::InvalidPacket("IPv4 ICMP packet too large".into()))?;

    if total_len > u16::MAX as usize {
        return Err(Error::InvalidPacket("IPv4 ICMP packet too large".into()));
    }

    let mut packet = vec![0u8; total_len];
    packet[0] = 0x45;
    packet[2..4].copy_from_slice(&(total_len as u16).to_be_bytes());
    packet[6..8].copy_from_slice(&0x4000u16.to_be_bytes());
    packet[8] = 64;
    packet[9] = IpTransportProtocol::Icmp.protocol_number();
    packet[12..16].copy_from_slice(&dst_addr.octets());
    packet[16..20].copy_from_slice(&src_addr.octets());

    let header_checksum = internet_checksum(&packet[..20]);
    packet[10..12].copy_from_slice(&header_checksum.to_be_bytes());

    let icmp_offset = 20;
    packet[icmp_offset] = 3;
    packet[icmp_offset + 1] = 2;
    packet[icmp_offset + 8..icmp_offset + 8 + quote_len]
        .copy_from_slice(&original_packet[..quote_len]);

    let icmp_checksum = internet_checksum(&packet[icmp_offset..]);
    packet[icmp_offset + 2..icmp_offset + 4].copy_from_slice(&icmp_checksum.to_be_bytes());

    Ok(Some(packet))
}

fn udp_ipv4_checksum(src_addr: Ipv4Addr, dst_addr: Ipv4Addr, udp_segment: &[u8]) -> u16 {
    let mut data = Vec::with_capacity(12 + udp_segment.len() + (udp_segment.len() % 2));
    data.extend_from_slice(&src_addr.octets());
    data.extend_from_slice(&dst_addr.octets());
    data.push(0);
    data.push(IpTransportProtocol::Udp.protocol_number());
    data.extend_from_slice(&(udp_segment.len() as u16).to_be_bytes());
    data.extend_from_slice(udp_segment);
    if data.len() % 2 != 0 {
        data.push(0);
    }

    let checksum = internet_checksum(&data);
    if checksum == 0 {
        0xffff
    } else {
        checksum
    }
}

fn udp_ipv6_checksum(src_addr: Ipv6Addr, dst_addr: Ipv6Addr, udp_segment: &[u8]) -> u16 {
    let mut data = Vec::with_capacity(40 + udp_segment.len() + (udp_segment.len() % 2));
    data.extend_from_slice(&src_addr.octets());
    data.extend_from_slice(&dst_addr.octets());
    data.extend_from_slice(&(udp_segment.len() as u32).to_be_bytes());
    data.extend_from_slice(&[0, 0, 0]);
    data.push(IpTransportProtocol::Udp.protocol_number());
    data.extend_from_slice(udp_segment);
    if data.len() % 2 != 0 {
        data.push(0);
    }

    let checksum = internet_checksum(&data);
    if checksum == 0 {
        0xffff
    } else {
        checksum
    }
}

fn internet_checksum(data: &[u8]) -> u16 {
    let mut sum = 0u32;
    let mut chunks = data.chunks_exact(2);
    for chunk in &mut chunks {
        sum += u16::from_be_bytes([chunk[0], chunk[1]]) as u32;
    }
    if let Some(&byte) = chunks.remainder().first() {
        sum += u16::from_be_bytes([byte, 0]) as u32;
    }

    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }

    !(sum as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn udp_ipv4_parser_extracts_exit_datagram() {
        let packet = build_ipv4_udp_packet(
            Ipv4Addr::new(10, 77, 0, 2),
            Ipv4Addr::new(203, 0, 113, 10),
            49152,
            53,
            b"hello",
        )
        .unwrap();

        let datagram = parse_udp_datagram(&packet).unwrap().unwrap();

        assert_eq!(
            datagram.key.tunnel_src,
            IpAddr::V4(Ipv4Addr::new(10, 77, 0, 2))
        );
        assert_eq!(datagram.key.tunnel_src_port, 49152);
        assert_eq!(
            datagram.key.remote_dst,
            IpAddr::V4(Ipv4Addr::new(203, 0, 113, 10))
        );
        assert_eq!(datagram.key.remote_dst_port, 53);
        assert_eq!(
            datagram.destination,
            SocketAddr::from((Ipv4Addr::new(203, 0, 113, 10), 53))
        );
        assert_eq!(datagram.payload, b"hello");
    }

    #[test]
    fn udp_ipv4_builder_creates_valid_reverse_packet() {
        let packet = build_ipv4_udp_packet(
            Ipv4Addr::new(8, 8, 8, 8),
            Ipv4Addr::new(10, 77, 0, 2),
            53,
            49152,
            b"world",
        )
        .unwrap();

        assert_eq!(internet_checksum(&packet[..20]), 0);
        assert_eq!(
            udp_ipv4_checksum(
                Ipv4Addr::new(8, 8, 8, 8),
                Ipv4Addr::new(10, 77, 0, 2),
                &packet[20..]
            ),
            0xffff
        );

        let parsed = IpPacket::parse(&packet).unwrap();
        assert_eq!(parsed.version, IpVersion::V4);
        assert_eq!(parsed.protocol, IpTransportProtocol::Udp);
        assert_eq!(parsed.src_addr, IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)));
        assert_eq!(parsed.dst_addr, IpAddr::V4(Ipv4Addr::new(10, 77, 0, 2)));
        assert_eq!(parsed.src_port, Some(53));
        assert_eq!(parsed.dst_port, Some(49152));
        assert_eq!(parsed.payload(), b"world");
    }

    #[test]
    fn udp_ipv6_parser_and_builder_round_trip() {
        let src = Ipv6Addr::new(0x2001, 0xdb8, 0, 1, 0, 0, 0, 10);
        let dst = Ipv6Addr::new(0x2001, 0xdb8, 0, 2, 0, 0, 0, 20);
        let packet = build_ipv6_udp_packet(src, dst, 53000, 53, b"hello6").unwrap();

        let datagram = parse_udp_datagram(&packet).unwrap().unwrap();
        assert_eq!(datagram.key.tunnel_src, IpAddr::V6(src));
        assert_eq!(datagram.key.tunnel_src_port, 53000);
        assert_eq!(datagram.key.remote_dst, IpAddr::V6(dst));
        assert_eq!(datagram.key.remote_dst_port, 53);
        assert_eq!(datagram.destination, SocketAddr::from((dst, 53)));
        assert_eq!(datagram.payload, b"hello6");

        let parsed = IpPacket::parse(&packet).unwrap();
        assert_eq!(parsed.version, IpVersion::V6);
        assert_eq!(parsed.protocol, IpTransportProtocol::Udp);
        assert_eq!(parsed.src_addr, IpAddr::V6(src));
        assert_eq!(parsed.dst_addr, IpAddr::V6(dst));
        assert_eq!(parsed.src_port, Some(53000));
        assert_eq!(parsed.dst_port, Some(53));
        assert_eq!(parsed.payload(), b"hello6");
        assert_eq!(udp_ipv6_checksum(src, dst, &packet[40..]), 0xffff);
    }

    #[test]
    fn unsupported_ipv4_packet_builds_protocol_unreachable() {
        let tcp_packet = [
            0x45, 0x00, 0x00, 0x28, 0x1c, 0x46, 0x40, 0x00, 0x40, 0x06, 0x00, 0x00, 0x0a, 0x4d,
            0x00, 0x02, 0xcb, 0x00, 0x71, 0x0a, 0xc0, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x50, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        let response = build_ipv4_protocol_unreachable(&tcp_packet)
            .unwrap()
            .unwrap();

        assert_eq!(internet_checksum(&response[..20]), 0);

        let parsed = IpPacket::parse(&response).unwrap();
        assert_eq!(parsed.version, IpVersion::V4);
        assert_eq!(parsed.protocol, IpTransportProtocol::Icmp);
        assert_eq!(parsed.src_addr, IpAddr::V4(Ipv4Addr::new(203, 0, 113, 10)));
        assert_eq!(parsed.dst_addr, IpAddr::V4(Ipv4Addr::new(10, 77, 0, 2)));

        let icmp = parsed.payload();
        assert_eq!(icmp[0], 3);
        assert_eq!(icmp[1], 2);
        assert_eq!(internet_checksum(icmp), 0);
        assert_eq!(&icmp[8..28], &tcp_packet[..20]);
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
