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
        if let Some(manager) = self.0.upgrade() {
            let _ = manager.event_tx.send(MultipathEvent::Connected);
        }
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

async fn delayed_handshake_server(
    delay: Duration,
) -> (std::net::SocketAddr, PublicKey, tokio::task::JoinHandle<()>) {
    let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let address = socket.local_addr().unwrap();
    let keys = KeyPair::generate();
    let public = keys.public;
    let task = tokio::spawn(async move {
        tokio::time::sleep(delay).await;
        let mut buffer = vec![0; 65536];
        loop {
            let (len, peer) = socket.recv_from(&mut buffer).await.unwrap();
            let packet = Packet::decode(&buffer[..len]).unwrap();
            if packet.header.packet_type != PacketType::Handshake {
                continue;
            }
            let mut noise = NoiseSession::new_responder(&keys.secret).unwrap();
            noise.read_handshake(&packet.payload).unwrap();
            let reply = Packet::new(
                PacketType::Handshake,
                SequenceNumber(1),
                packet.header.session_id,
                packet.header.uplink_id,
                noise.write_handshake(b"legacy fixture").unwrap(),
            )
            .unwrap();
            socket
                .send_to(&reply.encode().unwrap(), peer)
                .await
                .unwrap();
        }
    });
    (address, public, task)
}

#[tokio::test]
async fn startup_retries_until_an_exit_becomes_available() {
    let (address, public, server) = delayed_handshake_server(Duration::from_millis(150)).await;
    let mut config = MultipathConfig::default();
    config.transport.connect_timeout = Duration::from_millis(30);
    config.reconnect_initial_delay = Duration::from_millis(20);
    config.reconnect_max_delay = Duration::from_millis(40);
    let manager = manager(config);
    manager
        .add_uplink(UplinkConfig {
            remote_addr: address,
            ..Default::default()
        })
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), manager.connect_until_ready(public))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(manager.state(), ConnectionState::Connected);
    manager.close().unwrap();
    server.abort();
}

#[tokio::test]
async fn failed_secondary_path_is_retried_without_an_interface_event() {
    let (good, public, good_server) = delayed_handshake_server(Duration::ZERO).await;
    // An established path remains usable while the absent endpoint is retried.
    let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let bad = socket.local_addr().unwrap();
    drop(socket);
    let mut config = MultipathConfig::default();
    config.transport.connect_timeout = Duration::from_millis(30);
    config.health_check_interval = Duration::from_millis(20);
    config.reconnect_initial_delay = Duration::from_millis(20);
    config.reconnect_max_delay = Duration::from_millis(40);
    let manager = manager(config);
    manager
        .add_uplink(UplinkConfig {
            id: UplinkId::new("good"),
            remote_addr: good,
            ..Default::default()
        })
        .unwrap();
    let bad_id = manager
        .add_uplink(UplinkConfig {
            id: UplinkId::new("bad"),
            remote_addr: bad,
            ..Default::default()
        })
        .unwrap();
    manager.connect(public).await.unwrap();
    manager.start_maintenance_loop();
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if manager
                .reconnect
                .lock()
                .get(&bad_id)
                .is_some_and(|r| r.failures >= 2)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        manager.get_uplink(0).unwrap().state().connection_state,
        ConnectionState::Connected
    );
    manager.close().unwrap();
    good_server.abort();
}

#[tokio::test]
async fn shutdown_rejects_a_late_handshake_response() {
    let (address, public, server) = delayed_handshake_server(Duration::from_millis(100)).await;
    let manager = manager(MultipathConfig::default());
    let id = manager
        .add_uplink(UplinkConfig {
            remote_addr: address,
            ..Default::default()
        })
        .unwrap();
    let connecting = {
        let manager = Arc::clone(&manager);
        tokio::spawn(async move { manager.connect(public).await })
    };
    tokio::time::sleep(Duration::from_millis(30)).await;
    manager.close().unwrap();
    assert!(connecting.await.unwrap().is_err());
    assert_eq!(manager.state(), ConnectionState::Disconnected);
    assert!(!manager.get_uplink(id).unwrap().is_noise_ready());
    server.abort();
}

#[test]
fn recovery_backoff_is_bounded_under_extreme_retry_counts() {
    let config = MultipathConfig::default();
    assert!(reconnect_delay(&config, 1, 1.0) > reconnect_delay(&config, 0, 1.0));
    for failures in [0, 1, 10, 31, u32::MAX] {
        for jitter in [0.8, 1.0, 1.2] {
            let delay = reconnect_delay(&config, failures, jitter);
            assert!(delay >= Duration::from_millis(20));
            assert!(delay <= config.reconnect_max_delay);
        }
    }
}

#[test]
fn legacy_echo_cannot_confirm_a_reduced_path_mtu() {
    let manager = manager(MultipathConfig::default());
    let uplink = path(&manager, "legacy");
    let packet = Packet::new(
        PacketType::Pong,
        SequenceNumber(0),
        manager.session_id(),
        uplink.numeric_id(),
        crate::protocol::mtu::request(&[0; 16], 1280),
    )
    .unwrap();
    manager.handle_pong(&packet, uplink.numeric_id());
    assert_eq!(uplink.state().connection_state, ConnectionState::Failed);
    assert_eq!(manager.negotiated_mtu(), 1340);
}

#[tokio::test]
async fn tg2_exit_cannot_silently_accept_a_reduced_mtu() {
    let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let address = socket.local_addr().unwrap();
    let key = KeyPair::generate();
    let public = key.public;
    let server = tokio::spawn(async move {
        let mut buffer = [0; 2048];
        let (size, peer) = socket.recv_from(&mut buffer).await.unwrap();
        let packet = Packet::decode(&buffer[..size]).unwrap();
        let mut noise = NoiseSession::new_responder(&key.secret).unwrap();
        noise.read_handshake(&packet.payload).unwrap();
        let reply = Packet::new(
            PacketType::Handshake,
            SequenceNumber(1),
            packet.header.session_id,
            packet.header.uplink_id,
            noise.write_handshake(b"TG2\x0a\x00\x55\x01").unwrap(),
        )
        .unwrap();
        socket
            .send_to(&reply.encode().unwrap(), peer)
            .await
            .unwrap();
    });
    let manager = manager(MultipathConfig {
        inner_mtu: 1280,
        ..Default::default()
    });
    manager
        .add_uplink(UplinkConfig {
            remote_addr: address,
            ..Default::default()
        })
        .unwrap();
    assert!(matches!(
        manager.connect(public).await,
        Err(Error::HandshakeFailed(_))
    ));
    server.await.unwrap();
}

#[test]
fn closing_stops_pending_admission() {
    let manager = manager(MultipathConfig::default());
    manager.close().unwrap();
    assert!(matches!(
        manager.track_packet(SequenceNumber(1), b"payload", 0),
        Err(Error::ConnectionClosed)
    ));
    assert_eq!(manager.pending_stats().bytes, 0);
}

#[tokio::test]
async fn dropping_receive_consumer_cancels_a_blocked_read_promptly() {
    let manager = manager(MultipathConfig::default());
    *manager.state.write() = ConnectionState::Connected;
    let uplink = path(&manager, "blocked");
    uplink.set_transport(Arc::new(ImmediateAck(Arc::downgrade(&manager))));
    let mut entered_read = manager.subscribe();
    let receive = manager.start_recv_loop();
    manager.start_maintenance_loop();
    tokio::time::timeout(Duration::from_secs(1), entered_read.recv())
        .await
        .unwrap()
        .unwrap();
    let weak = Arc::downgrade(&manager);
    drop(manager);
    drop(receive);
    tokio::time::timeout(Duration::from_millis(250), async {
        while weak.upgrade().is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("receive task retained the connection after its consumer closed");
}

#[tokio::test]
async fn receive_owner_is_exclusive_and_can_be_reacquired_after_drop() {
    let manager = manager(MultipathConfig::default());
    *manager.state.write() = ConnectionState::Connected;
    let first = manager.try_start_recv_loop().unwrap();
    assert!(manager.try_start_recv_loop().is_err());
    drop(first);
    let second = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if let Ok(receive) = manager.try_start_recv_loop() {
                break receive;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    drop(second);
    manager.close().unwrap();
}

#[tokio::test]
async fn full_production_consumer_queue_does_not_block_control_reader() {
    let manager = manager(MultipathConfig::default());
    *manager.assigned_ipv4.write() = Some("10.0.85.1".parse().unwrap());
    let (tx, mut rx) = mpsc::channel(1);
    tx.send((vec![1], 1)).await.unwrap();
    tokio::time::timeout(Duration::from_millis(100), async {
        for _ in 0..1000 {
            assert!(manager.deliver_payload(&tx, vec![2], 1).await);
        }
    })
    .await
    .unwrap();
    assert_eq!(manager.stats().packets_dropped, 1000);
    assert_eq!(rx.recv().await.unwrap().0, vec![1]);
    drop(rx);
    assert!(!manager.deliver_payload(&tx, vec![3], 1).await);
}

#[test]
fn removing_path_releases_attempts_and_reconnect_metadata() {
    let manager = manager(MultipathConfig::default());
    let uplink = path(&manager, "removed");
    manager
        .track_packet(SequenceNumber(1), b"payload", uplink.numeric_id())
        .unwrap();
    manager.track_attempt(1, &uplink);
    assert_eq!(uplink.in_flight(), 1);
    manager.reconnect.lock().insert(
        uplink.numeric_id(),
        ReconnectState {
            failures: 1,
            next_attempt: Instant::now(),
            in_progress: false,
        },
    );
    manager.remove_uplink(uplink.numeric_id()).unwrap();
    assert_eq!(uplink.in_flight(), 0);
    assert!(!manager.reconnect.lock().contains_key(&uplink.numeric_id()));
    assert!(manager.pending.lock().packets[&1].attempts.is_empty());
}

#[test]
fn closing_releases_transport_even_with_a_retained_path_handle() {
    let manager = manager(MultipathConfig::default());
    let uplink = path(&manager, "retained");
    let transport: Arc<dyn Transport> = Arc::new(ImmediateAck(Arc::downgrade(&manager)));
    let weak = Arc::downgrade(&transport);
    uplink.set_transport(transport);
    assert!(weak.upgrade().is_some());
    manager.close().unwrap();
    assert!(uplink.get_transport().is_none());
    assert!(weak.upgrade().is_none());
    assert_eq!(
        uplink.state().connection_state,
        ConnectionState::Disconnected
    );
}

#[tokio::test]
async fn production_ip_sends_do_not_wait_on_legacy_pacing_estimates() {
    for aggregation_mode in [AggregationMode::None, AggregationMode::Full] {
        let manager = manager(MultipathConfig {
            aggregation_mode,
            ..Default::default()
        });
        let uplink = path(&manager, "slow-estimate");
        manager
            .throughput_optimizer
            .record_bandwidth(uplink.numeric_id(), 1.0);
        assert!(
            manager
                .throughput_optimizer
                .pacing_interval(uplink.numeric_id(), 64)
                >= Duration::from_secs(1)
        );
        *manager.assigned_ipv4.write() = Some("10.0.85.1".parse().unwrap());
        // No cipher is installed, so the send must fail promptly at encryption,
        // without spending seconds asleep in the legacy rate estimator first.
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            manager.send_on_flow(None, &[0; 64]),
        )
        .await;
        assert!(result
            .expect("production send was blocked by legacy pacing")
            .is_err());
        manager.close().unwrap();
    }
}
