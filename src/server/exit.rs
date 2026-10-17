//! Production single-exit IP forwarding service.
//!
//! The host kernel owns routing, conntrack and egress NAT. This module transfers
//! complete IP packets between authenticated per-uplink records and one TUN.
use std::collections::{BTreeSet, HashMap};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::Serialize;
use tokio::net::{TcpListener, UdpSocket};
use tokio::sync::watch;

use crate::crypto::{secure_compare, KeyPair, NoiseSession};
use crate::protocol::{Packet, PacketFlags, PacketType};
use crate::tun::{IpPacket, TunConfig, TunDevice};
use crate::types::{SequenceNumber, SessionId};
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct ExitConfig {
    pub listen: SocketAddr,
    pub tun: TunConfig,
    pub max_sessions: usize,
    pub idle_timeout: Duration,
    pub access_token: Option<[u8; 32]>,
}

impl Default for ExitConfig {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:7443".parse().unwrap(),
            tun: TunConfig {
                name: "tg-exit".into(),
                ipv4_addr: Some(Ipv4Addr::new(10, 0, 85, 254)),
                ipv6_addr: Some("fd00:7472:6967::ffff".parse().unwrap()),
                ..Default::default()
            },
            max_sessions: 252,
            idle_timeout: Duration::from_secs(120),
            access_token: None,
        }
    }
}

struct Path {
    address: SocketAddr,
    noise: NoiseSession,
    hello: Vec<u8>,
    reply: Vec<u8>,
    last_seen: Instant,
    received: u64,
    sent: u64,
}

struct Session {
    token: [u8; 32],
    ipv4: Ipv4Addr,
    ipv6: Option<Ipv6Addr>,
    paths: HashMap<u16, Path>,
    seen: BTreeSet<u64>,
    highest: u64,
    next_sequence: u64,
    last_seen: Instant,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExitPathStatus {
    pub session: String,
    pub ipv4: Ipv4Addr,
    pub ipv6: Option<Ipv6Addr>,
    pub uplink: u16,
    pub address: SocketAddr,
    pub received: u64,
    pub sent: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExitStatus {
    pub role: &'static str,
    pub state: &'static str,
    pub sessions: usize,
    pub authentication_required: bool,
    pub paths: Vec<ExitPathStatus>,
    pub forwarded_packets: u64,
    pub returned_packets: u64,
    pub rejected_packets: u64,
    pub proxy_connections_by_source: std::collections::BTreeMap<String, u64>,
}

pub struct ExitServer {
    config: ExitConfig,
    keypair: KeyPair,
    socket: UdpSocket,
    tcp: TcpListener,
    proxy_sources: Arc<Mutex<std::collections::BTreeMap<String, u64>>>,
    tun: TunDevice,
    sessions: Mutex<HashMap<SessionId, Session>>,
    forwarded: AtomicU64,
    returned: AtomicU64,
    rejected: AtomicU64,
    shutdown: watch::Sender<bool>,
}

impl ExitServer {
    pub async fn new(config: ExitConfig, keypair: KeyPair) -> Result<Self> {
        if config.tun.ipv4_netmask != 24 || config.tun.ipv4_addr.is_none() {
            return Err(Error::Config(
                "Exit requires an IPv4 /24 TUN network".into(),
            ));
        }
        if config.tun.ipv6_addr.is_some() && config.tun.ipv6_prefix != 64 {
            return Err(Error::Config("Exit IPv6 pool requires a /64 prefix".into()));
        }
        let socket = UdpSocket::bind(config.listen).await?;
        let tcp = TcpListener::bind(socket.local_addr()?).await?;
        let mut tun = TunDevice::create(config.tun.clone())?;
        tun.configure_addresses()?;
        tun.up()?;
        let (shutdown, _) = watch::channel(false);
        Ok(Self {
            config,
            keypair,
            socket,
            tcp,
            proxy_sources: Arc::new(Mutex::new(Default::default())),
            tun,
            sessions: Mutex::new(HashMap::new()),
            forwarded: AtomicU64::new(0),
            returned: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            shutdown,
        })
    }

    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }
    pub fn tun_name(&self) -> &str {
        self.tun.name()
    }
    pub fn shutdown(&self) {
        self.shutdown.send_replace(true);
    }

    pub fn status(&self) -> ExitStatus {
        let sessions = self.sessions.lock();
        ExitStatus {
            role: "exit",
            state: "running",
            sessions: sessions.len(),
            authentication_required: self.config.access_token.is_some(),
            paths: sessions
                .iter()
                .flat_map(|(id, s)| {
                    s.paths.iter().map(move |(uplink, p)| ExitPathStatus {
                        session: id.to_string(),
                        ipv4: s.ipv4,
                        ipv6: s.ipv6,
                        uplink: *uplink,
                        address: p.address,
                        received: p.received,
                        sent: p.sent,
                    })
                })
                .collect(),
            forwarded_packets: self.forwarded.load(Ordering::Relaxed),
            returned_packets: self.returned.load(Ordering::Relaxed),
            rejected_packets: self.rejected.load(Ordering::Relaxed),
            proxy_connections_by_source: self.proxy_sources.lock().clone(),
        }
    }

    pub async fn run(&self) -> Result<()> {
        let mut wire = vec![0u8; 65536];
        let mut ip = vec![0u8; 65536];
        let mut shutdown = self.shutdown.subscribe();
        let mut cleanup = tokio::time::interval(Duration::from_secs(1));
        let mut proxy_tasks = tokio::task::JoinSet::new();
        loop {
            tokio::select! {
                result = self.tcp.accept(), if proxy_tasks.len() < self.config.max_sessions => {
                    let (stream, peer) = result?;
                    let keypair = self.keypair.clone();
                    let sources = Arc::clone(&self.proxy_sources);
                    let access_token = self.config.access_token;
                    proxy_tasks.spawn(async move {
                        match crate::proxy::exit_stream::serve(stream, keypair, access_token).await {
                            Ok(()) => { *sources.lock().entry(peer.ip().to_string()).or_default() += 1; }
                            Err(e) => tracing::debug!(error = %e, "Exit proxy stream closed"),
                        }
                    });
                }
                _ = proxy_tasks.join_next(), if !proxy_tasks.is_empty() => {}
                result = self.socket.recv_from(&mut wire) => {
                    let (len, address) = result?;
                    if let Err(e) = self.incoming(&wire[..len], address).await {
                        self.rejected.fetch_add(1, Ordering::Relaxed);
                        tracing::debug!(%address, error = %e, "Exit rejected packet");
                    }
                }
                result = self.tun.read(&mut ip) => {
                    let len = result?;
                    if let Err(e) = self.outgoing(&ip[..len]).await {
                        tracing::debug!(error = %e, "Exit cannot return packet");
                    }
                }
                _ = cleanup.tick() => { self.sessions.lock().retain(|_, s| s.last_seen.elapsed() < self.config.idle_timeout); }
                _ = shutdown.changed() => { if *shutdown.borrow() { break; } }
            }
        }
        Ok(())
    }

    async fn incoming(&self, wire: &[u8], address: SocketAddr) -> Result<()> {
        let packet = Packet::decode(wire)?;
        if packet.header.packet_type == PacketType::Handshake {
            let reply = self.handshake(&packet, address)?;
            self.socket.send_to(&reply, address).await?;
            return Ok(());
        }
        let (payload, response, duplicate) = {
            let mut sessions = self.sessions.lock();
            let session = sessions
                .get_mut(&packet.header.session_id)
                .ok_or_else(|| Error::Authentication("unknown session".into()))?;
            let path = session
                .paths
                .get_mut(&packet.header.uplink_id)
                .ok_or_else(|| Error::Authentication("unknown uplink".into()))?;
            if path.address != address || !packet.header.flags.has(PacketFlags::ENCRYPTED) {
                return Err(Error::Authentication(
                    "unbound address or unencrypted record".into(),
                ));
            }
            let payload = path.noise.decrypt_packet(
                &packet.payload,
                packet.header.sequence.0,
                packet.header.packet_type as u8,
            )?;
            path.last_seen = Instant::now();
            session.last_seen = Instant::now();
            if packet.header.packet_type == PacketType::Close {
                sessions.remove(&packet.header.session_id);
                return Ok(());
            }
            let mut duplicate = false;
            let (kind, response_payload) = match packet.header.packet_type {
                PacketType::Data => {
                    let parsed = IpPacket::parse(&payload)?;
                    if (parsed.src_addr != IpAddr::V4(session.ipv4)
                        && Some(parsed.src_addr) != session.ipv6.map(IpAddr::V6))
                        || parsed.total_len != payload.len()
                    {
                        return Err(Error::InvalidPacket(
                            "invalid client source or IP length".into(),
                        ));
                    }
                    let seq = packet.header.sequence.0;
                    duplicate =
                        session.seen.contains(&seq) || seq.saturating_add(8192) < session.highest;
                    session.seen.insert(seq);
                    session.highest = session.highest.max(seq);
                    let minimum = session.highest.saturating_sub(8192);
                    while session.seen.first().is_some_and(|n| *n < minimum) {
                        session.seen.pop_first();
                    }
                    if !duplicate {
                        path.received += 1;
                    }
                    (
                        PacketType::Ack,
                        bincode::serialize(&vec![seq])
                            .map_err(|e| Error::InvalidPacket(e.to_string()))?,
                    )
                }
                PacketType::Ping => (PacketType::Pong, payload.clone()),
                _ => return Err(Error::InvalidPacket("unsupported exit record".into())),
            };
            let response = Self::record(
                session,
                packet.header.session_id,
                packet.header.uplink_id,
                kind,
                &response_payload,
            )?;
            (payload, response, duplicate)
        };
        self.socket.send_to(&response, address).await?;
        if packet.header.packet_type == PacketType::Data && !duplicate {
            self.tun.write(&payload).await?;
            self.forwarded.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn client_ipv6(&self, ipv4: Ipv4Addr) -> Option<Ipv6Addr> {
        self.config.tun.ipv6_addr.map(|gateway| {
            let p = gateway.segments();
            Ipv6Addr::new(p[0], p[1], p[2], p[3], 0, 0, 0, u16::from(ipv4.octets()[3]))
        })
    }

    fn handshake(&self, packet: &Packet, address: SocketAddr) -> Result<Vec<u8>> {
        let mut noise = NoiseSession::new_responder(&self.keypair.secret)?;
        let hello = noise.read_handshake(&packet.payload)?;
        if !matches!(hello.len(), 69 | 101)
            || &hello[..3] != b"TG2"
            || hello[35..67] != packet.header.session_id.0
            || hello[67..69] != packet.header.uplink_id.to_be_bytes()
        {
            return Err(Error::Authentication(
                "invalid exit handshake binding".into(),
            ));
        }
        if let Some(expected) = self.config.access_token {
            if hello.len() != 101 || !secure_compare(&expected, &hello[69..101]) {
                return Err(Error::Authentication("invalid access token".into()));
            }
        }
        let mut sessions = self.sessions.lock();
        let id = packet.header.session_id;
        let token: [u8; 32] = hello[3..35].try_into().unwrap();
        if let Some(s) = sessions.get(&id) {
            if !secure_compare(&s.token, &token) {
                return Err(Error::Authentication("invalid session join token".into()));
            }
            if let Some(path) = s.paths.get(&packet.header.uplink_id) {
                if path.address == address && path.hello == packet.payload {
                    return Ok(path.reply.clone());
                }
                return Err(Error::Authentication("uplink already established".into()));
            }
            if s.paths.len() >= 16 {
                return Err(Error::Config("uplink limit".into()));
            }
        } else {
            if sessions.len() >= self.config.max_sessions.min(252) {
                return Err(Error::Config("session limit".into()));
            }
            let gateway = self.config.tun.ipv4_addr.unwrap().octets();
            let address = (1..=254)
                .map(|last| Ipv4Addr::new(gateway[0], gateway[1], gateway[2], last))
                .find(|ip| {
                    *ip != self.config.tun.ipv4_addr.unwrap()
                        && self
                            .client_ipv6(*ip)
                            .map_or(true, |v6| Some(v6) != self.config.tun.ipv6_addr)
                        && !sessions.values().any(|s| s.ipv4 == *ip)
                })
                .ok_or_else(|| Error::Config("address pool exhausted".into()))?;
            sessions.insert(
                id,
                Session {
                    token,
                    ipv4: address,
                    ipv6: self.client_ipv6(address),
                    paths: HashMap::new(),
                    seen: BTreeSet::new(),
                    highest: 0,
                    next_sequence: 0,
                    last_seen: Instant::now(),
                },
            );
        }
        let session = sessions.get_mut(&id).unwrap();
        let mut response = b"TG2".to_vec();
        response.extend_from_slice(&session.ipv4.octets());
        if let Some(ipv6) = session.ipv6 {
            response.extend_from_slice(&ipv6.octets());
        }
        let response = noise.write_handshake(&response)?;
        noise.enable_datagrams();
        let reply = Packet::new(
            PacketType::Handshake,
            packet.header.sequence.next(),
            id,
            packet.header.uplink_id,
            response,
        )?
        .encode()?;
        session.paths.insert(
            packet.header.uplink_id,
            Path {
                address,
                noise,
                hello: packet.payload.clone(),
                reply: reply.clone(),
                last_seen: Instant::now(),
                received: 0,
                sent: 0,
            },
        );
        tracing::info!(%id, %address, ipv4 = %session.ipv4, uplink = packet.header.uplink_id, "Exit uplink established");
        Ok(reply)
    }

    fn record(
        session: &mut Session,
        id: SessionId,
        uplink: u16,
        kind: PacketType,
        data: &[u8],
    ) -> Result<Vec<u8>> {
        let seq = session.next_sequence;
        if kind == PacketType::Data {
            session.next_sequence = seq
                .checked_add(1)
                .ok_or_else(|| Error::InvalidPacket("sequence exhausted".into()))?;
        }
        let path = session
            .paths
            .get_mut(&uplink)
            .ok_or(Error::NoAvailableUplinks)?;
        let payload = path.noise.encrypt_packet(data, seq, kind as u8)?;
        let mut packet = Packet::new(kind, SequenceNumber(seq), id, uplink, payload)?;
        packet.set_flag(PacketFlags::ENCRYPTED);
        Ok(packet.encode()?)
    }

    async fn outgoing(&self, data: &[u8]) -> Result<()> {
        let parsed = IpPacket::parse(data)?;
        let (record, address) = {
            let mut sessions = self.sessions.lock();
            let (id, session) = sessions
                .iter_mut()
                .find(|(_, s)| {
                    parsed.dst_addr == IpAddr::V4(s.ipv4)
                        || Some(parsed.dst_addr) == s.ipv6.map(IpAddr::V6)
                })
                .ok_or_else(|| Error::InvalidPacket("no route to client".into()))?;
            let mut paths: Vec<_> = session
                .paths
                .iter()
                .filter(|(_, p)| p.last_seen.elapsed() < Duration::from_secs(10))
                .map(|(id, _)| *id)
                .collect();
            paths.sort_unstable();
            if paths.is_empty() {
                return Err(Error::NoAvailableUplinks);
            }
            let uplink = paths[(self.returned.load(Ordering::Relaxed) as usize) % paths.len()];
            let path = session.paths.get_mut(&uplink).unwrap();
            let address = path.address;
            path.sent += 1;
            (
                Self::record(session, *id, uplink, PacketType::Data, data)?,
                address,
            )
        };
        self.socket.send_to(&record, address).await?;
        self.returned.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// Serve status from the same live object as the forwarding loop.
#[cfg(feature = "server")]
pub async fn serve_exit_status(server: Arc<ExitServer>, address: SocketAddr) -> Result<()> {
    use axum::{routing::get, Json, Router};
    let status_server = Arc::clone(&server);
    let metrics_server = Arc::clone(&server);
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/metrics", get(move || {
            let server = Arc::clone(&metrics_server);
            async move {
                let status = server.status();
                let text = format!(
                    "# TYPE triglav_sessions_active gauge\ntriglav_sessions_active {}\n\
                     # TYPE triglav_uplinks_active gauge\ntriglav_uplinks_active {}\n\
                     # TYPE triglav_exit_forwarded_packets_total counter\ntriglav_exit_forwarded_packets_total {}\n\
                     # TYPE triglav_exit_returned_packets_total counter\ntriglav_exit_returned_packets_total {}\n\
                     # TYPE triglav_exit_rejected_packets_total counter\ntriglav_exit_rejected_packets_total {}\n",
                    status.sessions, status.paths.len(), status.forwarded_packets,
                    status.returned_packets, status.rejected_packets,
                );
                ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")], text)
            }
        }))
        .route(
            "/status",
            get(move || {
                let server = Arc::clone(&status_server);
                async move { Json(server.status()) }
            }),
        );
    let listener = tokio::net::TcpListener::bind(address).await?;
    let mut shutdown = server.shutdown.subscribe();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            if !*shutdown.borrow() {
                let _ = shutdown.changed().await;
            }
        })
        .await?;
    Ok(())
}

/// Parse a 256-bit deployment access token supplied out of band.
pub fn parse_access_token(value: Option<&str>) -> Result<Option<[u8; 32]>> {
    value
        .map(|value| {
            let bytes = hex::decode(value)
                .map_err(|_| Error::Config("Access token must be 64 hexadecimal digits".into()))?;
            bytes
                .try_into()
                .map_err(|_| Error::Config("Access token must contain 32 bytes".into()))
        })
        .transpose()
}
