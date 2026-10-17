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

struct PendingPath {
    noise: NoiseSession,
    hello: Vec<u8>,
    reply: Vec<u8>,
    address: SocketAddr,
    created: Instant,
    next_return_sequence: u64,
    epoch: u64,
}

struct Path {
    address: SocketAddr,
    noise: NoiseSession,
    hello: Vec<u8>,
    reply: Vec<u8>,
    last_seen: Instant,
    received: u64,
    sent: u64,
    pending: Option<PendingPath>,
    generations: u64,
    confirmed: bool,
    next_return_sequence: u64,
    epoch: u64,
}

impl Path {
    /// Only an authenticated fresh record may move an endpoint or commit new keys.
    fn decrypt(&mut self, packet: &Packet, address: SocketAddr) -> Result<Vec<u8>> {
        if !packet.header.flags.has(PacketFlags::ENCRYPTED) {
            return Err(Error::Authentication("unencrypted exit record".into()));
        }
        let result = self.noise.decrypt_packet(
            &packet.payload,
            packet.header.sequence.0,
            packet.header.packet_type as u8,
        );
        let payload = match result {
            Ok(payload) => payload,
            Err(original) => {
                let candidate = self
                    .pending
                    .as_mut()
                    .filter(|p| p.created.elapsed() < Duration::from_secs(30));
                let Some(candidate) = candidate else {
                    return Err(original.into());
                };
                let payload = candidate.noise.decrypt_packet(
                    &packet.payload,
                    packet.header.sequence.0,
                    packet.header.packet_type as u8,
                )?;
                let candidate = self.pending.take().unwrap();
                self.noise = candidate.noise;
                self.hello = candidate.hello;
                self.reply = candidate.reply;
                self.next_return_sequence = candidate.next_return_sequence;
                self.epoch = candidate.epoch;
                self.generations += 1;
                payload
            }
        };
        self.address = address;
        self.confirmed = true;
        self.last_seen = Instant::now();
        Ok(payload)
    }
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
    pub key_generations: u64,
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
    pub tun_mtu: u16,
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
        crate::protocol::mtu::validate_mtu(config.tun.mtu)?;
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
                        key_generations: p.generations,
                    })
                })
                .collect(),
            forwarded_packets: self.forwarded.load(Ordering::Relaxed),
            returned_packets: self.returned.load(Ordering::Relaxed),
            rejected_packets: self.rejected.load(Ordering::Relaxed),
            tun_mtu: self.tun.mtu(),
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
            let payload = path.decrypt(&packet, address)?;
            session.next_sequence = session.next_sequence.max(path.next_return_sequence);
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
                PacketType::Ping => {
                    if let Some(requested) = crate::protocol::mtu::requested(&payload) {
                        let mtu = requested?.min(self.tun.mtu());
                        self.tun.reduce_mtu(mtu)?;
                        (PacketType::Pong, crate::protocol::mtu::reply(&payload, mtu))
                    } else {
                        (PacketType::Pong, payload.clone())
                    }
                }
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

    fn client_ipv6(config: &ExitConfig, ipv4: Ipv4Addr) -> Option<Ipv6Addr> {
        config.tun.ipv6_addr.map(|gateway| {
            let p = gateway.segments();
            Ipv6Addr::new(p[0], p[1], p[2], p[3], 0, 0, 0, u16::from(ipv4.octets()[3]))
        })
    }

    fn handshake(&self, packet: &Packet, address: SocketAddr) -> Result<Vec<u8>> {
        Self::negotiate_handshake(
            &self.config,
            &self.keypair,
            &mut self.sessions.lock(),
            packet,
            address,
        )
    }

    fn negotiate_handshake(
        config: &ExitConfig,
        keypair: &KeyPair,
        sessions: &mut HashMap<SessionId, Session>,
        packet: &Packet,
        address: SocketAddr,
    ) -> Result<Vec<u8>> {
        use crate::protocol::recovery::{encode_exit_reply, ExitHello};
        let mut noise = NoiseSession::new_responder(&keypair.secret)?;
        let hello = ExitHello::decode(&noise.read_handshake(&packet.payload)?)?;
        if hello.session != packet.header.session_id || hello.uplink != packet.header.uplink_id {
            return Err(Error::Authentication(
                "invalid exit handshake binding".into(),
            ));
        }
        if let Some(expected) = config.access_token {
            if !hello
                .access_token
                .is_some_and(|provided| secure_compare(&expected, &provided))
            {
                return Err(Error::Authentication("invalid access token".into()));
            }
        }
        let id = hello.session;
        if let Some(session) = sessions.get(&id) {
            if !secure_compare(&session.token, &hello.token) {
                return Err(Error::Authentication("invalid session join token".into()));
            }
            if let Some(path) = session.paths.get(&hello.uplink) {
                if path.address == address && path.hello == packet.payload {
                    return Ok(path.reply.clone());
                }
                if let Some(pending) = path
                    .pending
                    .as_ref()
                    .filter(|p| p.created.elapsed() < Duration::from_secs(30))
                {
                    if pending.address == address && pending.hello == packet.payload {
                        return Ok(pending.reply.clone());
                    }
                }
                if hello.version >= 3
                    && (hello.path_epoch <= path.epoch
                        || path
                            .pending
                            .as_ref()
                            .is_some_and(|p| hello.path_epoch <= p.epoch))
                {
                    return Err(Error::Authentication("stale path handshake epoch".into()));
                }
            } else if session.paths.len() >= 16 {
                return Err(Error::Config("uplink limit".into()));
            }
            if hello
                .preferred_ipv4
                .is_some_and(|preferred| preferred != session.ipv4)
            {
                return Err(Error::Authentication(
                    "requested tunnel address differs from existing lease".into(),
                ));
            }
        } else {
            if sessions.len() >= config.max_sessions.min(252) {
                return Err(Error::Config("session limit".into()));
            }
            let gateway = config.tun.ipv4_addr.unwrap().octets();
            let available = |ip: Ipv4Addr| {
                let octets = ip.octets();
                octets[..3] == gateway[..3]
                    && (1..=254).contains(&octets[3])
                    && Some(ip) != config.tun.ipv4_addr
                    && Self::client_ipv6(config, ip)
                        .map_or(true, |v6| Some(v6) != config.tun.ipv6_addr)
                    && !sessions.values().any(|s| s.ipv4 == ip)
            };
            let ipv4 = match hello.preferred_ipv4 {
                Some(preferred) if available(preferred) => preferred,
                Some(_) => {
                    return Err(Error::Authentication(
                        "requested tunnel address unavailable".into(),
                    ))
                }
                None => (1..=254)
                    .map(|last| Ipv4Addr::new(gateway[0], gateway[1], gateway[2], last))
                    .find(|ip| available(*ip))
                    .ok_or_else(|| Error::Config("address pool exhausted".into()))?,
            };
            sessions.insert(
                id,
                Session {
                    token: hello.token,
                    ipv4,
                    ipv6: Self::client_ipv6(config, ipv4),
                    paths: HashMap::new(),
                    seen: BTreeSet::new(),
                    highest: 0,
                    next_sequence: 0,
                    last_seen: Instant::now(),
                },
            );
        }
        let session = sessions.get_mut(&id).unwrap();
        let response = encode_exit_reply(hello.version, session.ipv4, session.ipv6);
        let response = noise.write_handshake(&response)?;
        noise.enable_datagrams();
        let reply = Packet::new(
            PacketType::Handshake,
            packet.header.sequence.next(),
            id,
            hello.uplink,
            response,
        )?
        .encode()?;
        if let Some(path) = session.paths.get_mut(&hello.uplink) {
            // Do not discard live keys or advance sequence state on an unconfirmed hello.
            path.pending = Some(PendingPath {
                noise,
                hello: packet.payload.clone(),
                reply: reply.clone(),
                address,
                created: Instant::now(),
                next_return_sequence: hello.next_return_sequence,
                epoch: hello.path_epoch,
            });
        } else {
            session.paths.insert(
                hello.uplink,
                Path {
                    address,
                    noise,
                    hello: packet.payload.clone(),
                    reply: reply.clone(),
                    last_seen: Instant::now(),
                    received: 0,
                    sent: 0,
                    pending: None,
                    generations: 1,
                    confirmed: false,
                    next_return_sequence: hello.next_return_sequence,
                    epoch: hello.path_epoch,
                },
            );
        }
        tracing::info!(%id, %address, ipv4 = %session.ipv4, uplink = hello.uplink, "Exit handshake prepared");
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
                .filter(|(_, p)| p.confirmed && p.last_seen.elapsed() < Duration::from_secs(10))
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

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use crate::protocol::recovery::ExitHello;

    fn establish(
        config: &ExitConfig,
        key: &KeyPair,
        sessions: &mut HashMap<SessionId, Session>,
        hello: &ExitHello,
        address: SocketAddr,
    ) -> (NoiseSession, Packet) {
        let client = KeyPair::generate();
        let mut noise = NoiseSession::new_initiator(&client.secret, &key.public).unwrap();
        let packet = Packet::new(
            PacketType::Handshake,
            SequenceNumber(0),
            hello.session,
            hello.uplink,
            noise.write_handshake(&hello.encode()).unwrap(),
        )
        .unwrap();
        let wire =
            ExitServer::negotiate_handshake(config, key, sessions, &packet, address).unwrap();
        let response = Packet::decode(&wire).unwrap();
        assert!(noise
            .read_handshake(&response.payload)
            .unwrap()
            .starts_with(b"TG3"));
        noise.enable_datagrams();
        (noise, packet)
    }

    fn hello() -> ExitHello {
        ExitHello {
            token: [12; 32],
            session: SessionId::generate(),
            uplink: 4,
            preferred_ipv4: None,
            next_return_sequence: 0,
            path_epoch: 1,
            access_token: None,
            version: 3,
        }
    }

    fn ping(noise: &mut NoiseSession, hello: &ExitHello, seq: u64) -> Packet {
        let payload = noise
            .encrypt_packet(b"probe", seq, PacketType::Ping as u8)
            .unwrap();
        let mut packet = Packet::new(
            PacketType::Ping,
            SequenceNumber(seq),
            hello.session,
            hello.uplink,
            payload,
        )
        .unwrap();
        packet.set_flag(PacketFlags::ENCRYPTED);
        packet
    }

    #[test]
    fn replayed_handshake_cannot_replace_live_keys_or_move_peer() {
        let config = ExitConfig::default();
        let key = KeyPair::generate();
        let mut sessions = HashMap::new();
        let hello = hello();
        let peer = "127.0.0.1:1001".parse().unwrap();
        let other = "127.0.0.1:1002".parse().unwrap();
        let (mut active, old_hello) = establish(&config, &key, &mut sessions, &hello, peer);
        let path = sessions
            .get_mut(&hello.session)
            .unwrap()
            .paths
            .get_mut(&hello.uplink)
            .unwrap();
        let old_record = ping(&mut active, &hello, 1);
        assert_eq!(path.decrypt(&old_record, peer).unwrap(), b"probe");
        assert!(
            ExitServer::negotiate_handshake(&config, &key, &mut sessions, &old_hello, other)
                .is_err()
        );
        let path = sessions
            .get_mut(&hello.session)
            .unwrap()
            .paths
            .get_mut(&hello.uplink)
            .unwrap();
        assert!(path.decrypt(&old_record, other).is_err());
        assert_eq!(path.address, peer);
        assert_eq!(path.generations, 1);
        assert_eq!(
            path.decrypt(&ping(&mut active, &hello, 2), peer).unwrap(),
            b"probe"
        );
        assert_eq!(path.generations, 1);
    }

    #[test]
    fn replacement_requires_new_key_confirmation_and_rejects_old_records() {
        let config = ExitConfig::default();
        let key = KeyPair::generate();
        let mut sessions = HashMap::new();
        let hello = hello();
        let old_peer = "127.0.0.1:1001".parse().unwrap();
        let new_peer = "127.0.0.1:1002".parse().unwrap();
        let moved_peer = "127.0.0.1:1003".parse().unwrap();
        let (mut old, _) = establish(&config, &key, &mut sessions, &hello, old_peer);
        let mut new_hello = hello.clone();
        new_hello.path_epoch += 1;
        let (mut new, _) = establish(&config, &key, &mut sessions, &new_hello, new_peer);
        let path = sessions
            .get_mut(&hello.session)
            .unwrap()
            .paths
            .get_mut(&hello.uplink)
            .unwrap();
        assert_eq!(
            path.decrypt(&ping(&mut old, &hello, 1), old_peer).unwrap(),
            b"probe"
        );
        assert_eq!(path.generations, 1);
        assert_eq!(
            path.decrypt(&ping(&mut new, &hello, 2), new_peer).unwrap(),
            b"probe"
        );
        assert_eq!(path.generations, 2);
        assert_eq!(path.address, new_peer);
        assert!(path.decrypt(&ping(&mut old, &hello, 3), old_peer).is_err());
        assert_eq!(path.address, new_peer);
        assert_eq!(
            path.decrypt(&ping(&mut new, &hello, 4), moved_peer)
                .unwrap(),
            b"probe"
        );
        assert_eq!(path.address, moved_peer);
        assert_eq!(path.generations, 2);
    }

    #[test]
    fn requested_lease_survives_new_registry_and_collisions_are_rejected() {
        let config = ExitConfig::default();
        let key = KeyPair::generate();
        let mut sessions = HashMap::new();
        let mut hello = hello();
        hello.preferred_ipv4 = Some("10.0.85.37".parse().unwrap());
        let peer = "127.0.0.1:1001".parse().unwrap();
        establish(&config, &key, &mut sessions, &hello, peer);
        assert_eq!(sessions[&hello.session].ipv4, hello.preferred_ipv4.unwrap());
        let mut conflicting = hello.clone();
        conflicting.session = SessionId::generate();
        let client = KeyPair::generate();
        let mut noise = NoiseSession::new_initiator(&client.secret, &key.public).unwrap();
        let packet = Packet::new(
            PacketType::Handshake,
            SequenceNumber(0),
            conflicting.session,
            conflicting.uplink,
            noise.write_handshake(&conflicting.encode()).unwrap(),
        )
        .unwrap();
        assert!(
            ExitServer::negotiate_handshake(&config, &key, &mut sessions, &packet, peer).is_err()
        );
        assert_eq!(sessions.len(), 1);
        sessions.clear();
        establish(&config, &key, &mut sessions, &hello, peer);
        assert_eq!(sessions[&hello.session].ipv4, hello.preferred_ipv4.unwrap());
    }
}
