use super::*;
use crate::transport::TransportProtocol;

fn manager(config: MultipathConfig) -> Arc<MultipathManager> {
    Arc::new(MultipathManager::new(config, KeyPair::generate()))
}

fn path(manager: &MultipathManager, name: &str) -> Arc<Uplink> {
    let id = manager
        .add_uplink(UplinkConfig {
            id: UplinkId::new(name),
            remote_addr: "127.0.0.1:7443".parse().unwrap(),
            protocol: TransportProtocol::Udp,
            ..Default::default()
        })
        .unwrap();
    let uplink = manager.get_uplink(id).unwrap();
    uplink.set_connection_state(ConnectionState::Connected);
    uplink
}

fn ack(manager: &MultipathManager, seq: u64, path: u16) {
    let packet = Packet::new(
        PacketType::Ack,
        SequenceNumber(100),
        manager.session_id(),
        path,
        bincode::serialize(&vec![seq]).unwrap(),
    )
    .unwrap();
    manager
        .handle_packet(&packet.encode().unwrap(), path)
        .unwrap();
}

#[test]
fn pending_limits_are_atomic_and_released_by_ack_and_close() {
    let manager = manager(MultipathConfig {
        max_pending_packets: 2,
        max_pending_bytes: 8,
        ..Default::default()
    });
    let uplink = path(&manager, "first");
    manager
        .track_packet(SequenceNumber(1), b"12345", uplink.numeric_id())
        .unwrap();
    assert!(matches!(
        manager.track_packet(SequenceNumber(2), b"1234", uplink.numeric_id()),
        Err(Error::Backpressure)
    ));
    manager
        .track_packet(SequenceNumber(2), b"123", uplink.numeric_id())
        .unwrap();
    assert!(matches!(
        manager.track_packet(SequenceNumber(3), b"", uplink.numeric_id()),
        Err(Error::Backpressure)
    ));
    assert_eq!(
        (
            manager.pending_stats().packets,
            manager.pending_stats().bytes
        ),
        (2, 8)
    );
    ack(&manager, 1, uplink.numeric_id());
    manager
        .track_packet(SequenceNumber(3), b"12345", uplink.numeric_id())
        .unwrap();
    manager.close().unwrap();
    assert_eq!(
        (
            manager.pending_stats().packets,
            manager.pending_stats().bytes
        ),
        (0, 0)
    );
    assert_eq!(uplink.in_flight(), 0);
}

#[test]
fn first_ack_retires_all_hedges_and_late_acks_do_not_underflow() {
    let manager = manager(MultipathConfig::default());
    let first = path(&manager, "first");
    let second = path(&manager, "second");
    manager
        .track_packet(SequenceNumber(1), b"payload", first.numeric_id())
        .unwrap();
    assert!(manager.track_attempt(1, &first));
    assert!(manager.track_attempt(1, &second));
    assert_eq!((first.in_flight(), second.in_flight()), (1, 1));
    ack(&manager, 1, second.numeric_id());
    assert_eq!((first.in_flight(), second.in_flight()), (0, 0));
    ack(&manager, 1, first.numeric_id());
    first.record_loss();
    second.record_rtt(Duration::from_millis(1));
    assert_eq!((first.in_flight(), second.in_flight()), (0, 0));
    assert!(
        !manager.track_attempt(1, &first),
        "an acknowledged hedge must not recreate pending state"
    );
}

#[tokio::test]
async fn unavailable_paths_expire_pending_without_repeated_loss() {
    let manager = manager(MultipathConfig {
        send_retries: 1,
        max_pending_age: Duration::from_secs(1),
        ..Default::default()
    });
    let uplink = path(&manager, "first");
    manager
        .track_packet(SequenceNumber(1), b"payload", uplink.numeric_id())
        .unwrap();
    manager.track_attempt(1, &uplink);
    uplink.mark_unavailable();
    {
        let mut pending = manager.pending.lock();
        pending.packets.get_mut(&1).unwrap().sent_at = Instant::now() - Duration::from_secs(2);
    }
    manager.retry_pending().await;
    assert_eq!(uplink.in_flight(), 0);
    let losses = uplink.stats().packets_dropped;
    manager.retry_pending().await;
    assert_eq!(uplink.stats().packets_dropped, losses);
    {
        let mut pending = manager.pending.lock();
        let packet = pending.packets.get_mut(&1).unwrap();
        packet.created_at = Instant::now() - Duration::from_secs(2);
    }
    manager.retry_pending().await;
    assert_eq!(manager.pending_stats().packets, 0);
    assert_eq!(manager.stats().packets_dropped, 1);
}

#[test]
fn loss_accounting_saturates_even_after_late_ack() {
    let manager = manager(MultipathConfig::default());
    let uplink = path(&manager, "first");
    uplink.record_send(20);
    uplink.record_loss();
    uplink.record_rtt(Duration::from_millis(1));
    uplink.record_loss();
    assert_eq!(uplink.in_flight(), 0);
    assert!(uplink.can_send());
}

struct ImmediateAck(std::sync::Weak<MultipathManager>);

#[async_trait::async_trait]
impl Transport for ImmediateAck {
    fn local_addr(&self) -> Result<std::net::SocketAddr> {
        Ok("127.0.0.1:7000".parse().unwrap())
    }
    fn remote_addr(&self) -> Option<std::net::SocketAddr> {
        Some("127.0.0.1:7443".parse().unwrap())
    }
    async fn send(&self, bytes: &[u8]) -> Result<usize> {
        let packet = Packet::decode(bytes)?;
        let manager = self.0.upgrade().unwrap();
        assert_eq!(manager.pending_stats().packets, 1);
        assert_eq!(
            manager
                .get_uplink(packet.header.uplink_id)
                .unwrap()
                .in_flight(),
            1
        );
        ack(&manager, packet.header.sequence.0, packet.header.uplink_id);
        Ok(bytes.len())
    }
    async fn send_to(&self, bytes: &[u8], _: std::net::SocketAddr) -> Result<usize> {
        self.send(bytes).await
    }
    async fn recv(&self, _: &mut [u8]) -> Result<usize> {
        std::future::pending().await
    }
    async fn recv_from(&self, _: &mut [u8]) -> Result<(usize, std::net::SocketAddr)> {
        std::future::pending().await
    }
    async fn close(&self) -> Result<()> {
        Ok(())
    }
    fn is_connected(&self) -> bool {
        true
    }
    fn transport_type(&self) -> &'static str {
        "test"
    }
}

#[tokio::test]
async fn acknowledgement_before_send_returns_cannot_leak_pending_state() {
    let manager = manager(MultipathConfig {
        hedged_packets: false,
        throughput_optimization: false,
        ..Default::default()
    });
    let uplink = path(&manager, "first");
    let keys = KeyPair::generate();
    let mut initiator = NoiseSession::new_initiator(&keys.secret, &keys.public).unwrap();
    let mut responder = NoiseSession::new_responder(&keys.secret).unwrap();
    responder
        .read_handshake(&initiator.write_handshake(b"").unwrap())
        .unwrap();
    initiator
        .read_handshake(&responder.write_handshake(b"").unwrap())
        .unwrap();
    initiator.enable_datagrams();
    uplink.set_noise_session(initiator);
    uplink.set_transport(Arc::new(ImmediateAck(Arc::downgrade(&manager))));
    manager.send(b"payload").await.unwrap();
    assert_eq!(manager.pending_stats().packets, 0);
    assert_eq!(uplink.in_flight(), 0);
}

#[test]
fn parallel_admission_never_exceeds_packet_or_byte_budget() {
    let manager = manager(MultipathConfig {
        max_pending_packets: 8,
        max_pending_bytes: 28,
        ..Default::default()
    });
    std::thread::scope(|scope| {
        for seq in 0..32 {
            let manager = Arc::clone(&manager);
            scope.spawn(move || {
                let _ = manager.track_packet(SequenceNumber(seq), b"1234", 0);
            });
        }
    });
    assert_eq!(
        (
            manager.pending_stats().packets,
            manager.pending_stats().bytes
        ),
        (7, 28)
    );
}

#[test]
fn malformed_ack_count_is_rejected_before_allocation() {
    let manager = manager(MultipathConfig::default());
    let uplink = path(&manager, "first");
    let packet = Packet::new(
        PacketType::Ack,
        SequenceNumber(1),
        manager.session_id(),
        uplink.numeric_id(),
        u64::MAX.to_le_bytes().to_vec(),
    )
    .unwrap();
    assert!(manager
        .handle_packet(&packet.encode().unwrap(), uplink.numeric_id())
        .is_err());
}
