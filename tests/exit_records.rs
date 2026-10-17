//! Record invariants for the production UDP data plane, independent of sockets.
use triglav::crypto::{KeyPair, NoiseSession};
use triglav::types::AuthKey;

fn sessions() -> (NoiseSession, NoiseSession) {
    let client = KeyPair::generate();
    let server = KeyPair::generate();
    let mut a = NoiseSession::new_initiator(&client.secret, &server.public).unwrap();
    let mut b = NoiseSession::new_responder(&server.secret).unwrap();
    let hello = a.write_handshake(b"session binding").unwrap();
    assert_eq!(b.read_handshake(&hello).unwrap(), b"session binding");
    let reply = b.write_handshake(b"allocated address").unwrap();
    assert_eq!(a.read_handshake(&reply).unwrap(), b"allocated address");
    a.enable_datagrams();
    b.enable_datagrams();
    (a, b)
}

#[test]
fn loss_reordering_replay_and_bidirectionality() {
    let (mut a, mut b) = sessions();
    let _lost = a.encrypt_packet(b"lost", 0, 2).unwrap();
    let second = a.encrypt_packet(b"second", 1, 2).unwrap();
    let third = a.encrypt_packet(b"third", 2, 2).unwrap();
    assert_eq!(b.decrypt_packet(&third, 2, 2).unwrap(), b"third");
    assert_eq!(b.decrypt_packet(&second, 1, 2).unwrap(), b"second");
    assert!(b.decrypt_packet(&second, 1, 2).is_err());
    let reply = b.encrypt_packet(b"reply", 0, 2).unwrap();
    assert_eq!(a.decrypt_packet(&reply, 0, 2).unwrap(), b"reply");
}

#[test]
fn invalid_headers_and_ciphertext_cannot_poison_receive_state() {
    let (mut a, mut b) = sessions();
    let record = a.encrypt_packet(b"payload", 17, 2).unwrap();
    assert!(b.decrypt_packet(&record, u64::MAX, 2).is_err());
    assert!(b.decrypt_packet(&record, 17, 3).is_err());
    let mut tampered = record.clone();
    tampered[8] ^= 1;
    assert!(b.decrypt_packet(&tampered, 17, 2).is_err());
    let mut nonce_tampered = record.clone();
    nonce_tampered[7] ^= 1;
    assert!(b.decrypt_packet(&nonce_tampered, 17, 2).is_err());
    assert_eq!(b.decrypt_packet(&record, 17, 2).unwrap(), b"payload");
}

#[test]
fn records_from_another_uplink_cannot_be_decrypted() {
    let (mut a, mut b) = sessions();
    let (mut c, mut d) = sessions();
    let one = a.encrypt_packet(b"first", 1, 2).unwrap();
    let two = c.encrypt_packet(b"other", 1, 2).unwrap();
    assert!(d.decrypt_packet(&one, 1, 2).is_err());
    assert!(b.decrypt_packet(&two, 1, 2).is_err());
    assert_eq!(b.decrypt_packet(&one, 1, 2).unwrap(), b"first");
    assert_eq!(d.decrypt_packet(&two, 1, 2).unwrap(), b"other");
}

#[test]
fn legacy_url_safe_key_with_underscores_and_base58_roundtrip() {
    use base64::Engine;
    let key = AuthKey::new([255; 32], vec!["127.0.0.1:7443".parse().unwrap()]);
    assert!(key.to_string().starts_with("tg1"));
    assert!(!key.to_string().contains('_'));
    let mut legacy_data = vec![255; 32];
    legacy_data.extend_from_slice(&[4, 127, 0, 0, 1]);
    legacy_data.extend_from_slice(&7443u16.to_be_bytes());
    let legacy = format!(
        "tg1_{}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(legacy_data)
    );
    assert!(legacy[4..].contains('_'));
    for encoded in [key.to_string(), legacy] {
        let parsed = AuthKey::parse(&encoded).unwrap();
        assert_eq!(parsed.server_pubkey(), key.server_pubkey());
        assert_eq!(parsed.server_addrs(), key.server_addrs());
    }
}

#[test]
fn malformed_ip_headers_do_not_panic_and_fragments_have_no_ports() {
    use triglav::tun::IpPacket;
    for size in 0..=80 {
        for ihl in 0..=15 {
            let mut bytes = vec![0u8; size];
            if size > 0 {
                bytes[0] = 0x40 | ihl;
            }
            if size >= 4 {
                bytes[2..4].copy_from_slice(&(size as u16).to_be_bytes());
            }
            if size >= 10 {
                bytes[9] = 6;
            }
            let result = std::panic::catch_unwind(|| IpPacket::parse(&bytes));
            assert!(result.is_ok(), "size={size}, ihl={ihl}");
        }
    }
    let mut fragment = vec![0xff; 40];
    fragment[0] = 0x45;
    fragment[2..4].copy_from_slice(&40u16.to_be_bytes());
    fragment[6..8].copy_from_slice(&1u16.to_be_bytes());
    fragment[9] = 17;
    let parsed = IpPacket::parse(&fragment).unwrap();
    assert_eq!(parsed.src_port, None);
    assert_eq!(parsed.dst_port, None);
}

/// Invoked explicitly by validation/netns.sh against its live production exit.
#[tokio::test]
#[ignore = "requires the production network namespace lab"]
async fn production_rejects_session_hijack() {
    use triglav::protocol::{Packet, PacketType};
    use triglav::types::{SequenceNumber, SessionId};
    let key = AuthKey::parse(&std::env::var("TRIGLAV_TEST_KEY").expect("live exit key required"))
        .unwrap();
    let id_bytes =
        hex::decode(std::env::var("TRIGLAV_TEST_SESSION").expect("existing session required"))
            .unwrap();
    let id = SessionId(id_bytes.try_into().unwrap());
    let client = KeyPair::generate();
    let server = triglav::crypto::PublicKey::from_bytes(key.server_pubkey());
    let mut noise = NoiseSession::new_initiator(&client.secret, &server).unwrap();
    let mut hello = b"TG2".to_vec();
    hello.extend_from_slice(&[0x42; 32]); // Wrong join capability for the existing session.
    hello.extend_from_slice(&id.0);
    hello.extend_from_slice(&99u16.to_be_bytes());
    hello.extend_from_slice(
        &hex::decode(std::env::var("TRIGLAV_ACCESS_TOKEN").expect("access token required"))
            .unwrap(),
    );
    let wire = Packet::new(
        PacketType::Handshake,
        SequenceNumber(0),
        id,
        99,
        noise.write_handshake(&hello).unwrap(),
    )
    .unwrap()
    .encode()
    .unwrap();
    let socket = tokio::net::UdpSocket::bind("10.71.1.2:0").await.unwrap();
    socket.connect(key.server_addrs()[0]).await.unwrap();
    socket.send(&wire).await.unwrap();
    let mut buffer = [0u8; 1500];
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(600),
            socket.recv(&mut buffer)
        )
        .await
        .is_err(),
        "exit accepted an unrelated client's uplink"
    );
    // A fresh logical session with the wrong deployment capability also fails.
    let fresh = SessionId::generate();
    let mut noise = NoiseSession::new_initiator(&client.secret, &server).unwrap();
    let mut hello = b"TG2".to_vec();
    hello.extend_from_slice(&[0x42; 32]);
    hello.extend_from_slice(&fresh.0);
    hello.extend_from_slice(&99u16.to_be_bytes());
    hello.extend_from_slice(&[0; 32]);
    let wire = Packet::new(
        PacketType::Handshake,
        SequenceNumber(0),
        fresh,
        99,
        noise.write_handshake(&hello).unwrap(),
    )
    .unwrap()
    .encode()
    .unwrap();
    socket.send(&wire).await.unwrap();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(600),
            socket.recv(&mut buffer)
        )
        .await
        .is_err(),
        "exit accepted the wrong access token"
    );
}

#[test]
fn nat_preserves_udp_checksums_and_payload_in_both_directions() {
    use triglav::tun::{NatConfig, NatTable};
    fn checksum(bytes: &[u8]) -> u16 {
        let mut sum = 0u32;
        for pair in bytes.chunks(2) {
            sum += u32::from(pair[0]) << 8 | u32::from(*pair.get(1).unwrap_or(&0));
        }
        while sum >> 16 != 0 {
            sum = (sum & 0xffff) + (sum >> 16);
        }
        !(sum as u16)
    }
    fn udp_checksum(packet: &[u8]) -> u16 {
        let mut data = packet[12..20].to_vec();
        data.extend_from_slice(&[0, 17]);
        data.extend_from_slice(&packet[24..26]);
        data.extend_from_slice(&packet[20..]);
        checksum(&data)
    }
    let payload = b"an odd binary payload\x00\xff";
    let mut packet = vec![0; 28 + payload.len()];
    packet[0] = 0x45;
    let size = packet.len() as u16;
    packet[2..4].copy_from_slice(&size.to_be_bytes());
    packet[8] = 64;
    packet[9] = 17;
    packet[12..16].copy_from_slice(&[10, 0, 85, 1]);
    packet[16..20].copy_from_slice(&[198, 18, 0, 2]);
    packet[20..22].copy_from_slice(&12345u16.to_be_bytes());
    packet[22..24].copy_from_slice(&9000u16.to_be_bytes());
    packet[24..26].copy_from_slice(&(8u16 + payload.len() as u16).to_be_bytes());
    packet[28..].copy_from_slice(payload);
    let crc = udp_checksum(&packet);
    packet[26..28].copy_from_slice(&crc.to_be_bytes());
    let nat = NatTable::new(NatConfig::default());
    nat.translate_outbound(&mut packet).unwrap().unwrap();
    assert_eq!(&packet[12..16], &[10, 0, 85, 2]);
    assert_eq!(udp_checksum(&packet), 0);
    assert_eq!(checksum(&packet[..20]), 0);
    let mapped_port = [packet[20], packet[21]];
    packet[12..16].copy_from_slice(&[198, 18, 0, 2]);
    packet[16..20].copy_from_slice(&[10, 0, 85, 2]);
    packet[20..22].copy_from_slice(&9000u16.to_be_bytes());
    packet[22..24].copy_from_slice(&mapped_port);
    packet[26..28].fill(0);
    let crc = udp_checksum(&packet);
    packet[26..28].copy_from_slice(&crc.to_be_bytes());
    nat.translate_inbound(&mut packet).unwrap().unwrap();
    assert_eq!(&packet[16..20], &[10, 0, 85, 1]);
    assert_eq!(&packet[22..24], &12345u16.to_be_bytes());
    assert_eq!(udp_checksum(&packet), 0);
    assert_eq!(checksum(&packet[..20]), 0);
    assert_eq!(&packet[28..], payload);
}

#[test]
fn access_tokens_require_exactly_256_bits() {
    use triglav::server::parse_access_token;
    assert_eq!(parse_access_token(None).unwrap(), None);
    assert_eq!(
        parse_access_token(Some(&"42".repeat(32))).unwrap(),
        Some([0x42; 32])
    );
    assert!(parse_access_token(Some(&"42".repeat(31))).is_err());
    assert!(parse_access_token(Some(&"42".repeat(33))).is_err());
    assert!(parse_access_token(Some(&"zz".repeat(32))).is_err());
}

#[tokio::test]
async fn failed_stripes_preserve_unique_logical_packet_sequences() {
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};
    use triglav::multipath::{AggregationMode, MultipathConfig, MultipathManager, UplinkConfig};
    use triglav::protocol::Packet;
    use triglav::transport::Transport;
    use triglav::{Error, Result};
    struct Sink {
        fail: bool,
        sequences: Arc<Mutex<Vec<u64>>>,
    }
    #[async_trait::async_trait]
    impl Transport for Sink {
        fn local_addr(&self) -> Result<SocketAddr> {
            Ok("127.0.0.1:1".parse().unwrap())
        }
        fn remote_addr(&self) -> Option<SocketAddr> {
            Some("127.0.0.1:2".parse().unwrap())
        }
        async fn send_to(&self, data: &[u8], _: SocketAddr) -> Result<usize> {
            self.send(data).await
        }
        async fn send(&self, data: &[u8]) -> Result<usize> {
            if self.fail {
                return Err(Error::ConnectionClosed);
            }
            let packet = Packet::decode(data)?;
            if packet.header.packet_type == triglav::protocol::PacketType::Data {
                self.sequences
                    .lock()
                    .unwrap()
                    .push(packet.header.sequence.0);
            }
            Ok(data.len())
        }
        async fn recv_from(&self, _: &mut [u8]) -> Result<(usize, SocketAddr)> {
            Err(Error::ConnectionClosed)
        }
        async fn recv(&self, _: &mut [u8]) -> Result<usize> {
            Err(Error::ConnectionClosed)
        }
        async fn close(&self) -> Result<()> {
            Ok(())
        }
        fn is_connected(&self) -> bool {
            true
        }
        fn transport_type(&self) -> &'static str {
            "controlled-send-failure"
        }
    }
    let config = MultipathConfig {
        aggregation_mode: AggregationMode::Full,
        throughput_optimization: false,
        ..Default::default()
    };
    let manager = MultipathManager::new(config, KeyPair::generate());
    let sequences = Arc::new(Mutex::new(Vec::new()));
    for index in 0..2 {
        let id = manager
            .add_uplink(UplinkConfig {
                id: format!("path-{index}").into(),
                ..Default::default()
            })
            .unwrap();
        let uplink = manager.get_uplink(id).unwrap();
        let (noise, _) = sessions();
        uplink.set_noise_session(noise);
        uplink.set_transport(Arc::new(Sink {
            fail: index == 0,
            sequences: Arc::clone(&sequences),
        }));
        uplink.set_connection_state(triglav::types::ConnectionState::Connected);
    }
    for index in 0..120u64 {
        let seq = manager.send(&index.to_be_bytes()).await.unwrap();
        assert_eq!(
            seq,
            index + 1,
            "a failed path must not consume another packet's sequence"
        );
    }
    assert_eq!(*sequences.lock().unwrap(), (1..=120).collect::<Vec<_>>());
}

#[test]
#[cfg(unix)]
fn generated_server_key_is_private_and_not_overwritten() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("server.key");
    let generate = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_triglav"))
            .args(["keygen", "--output"])
            .arg(&path)
            .output()
            .unwrap()
    };
    assert!(generate().status.success());
    let before = std::fs::read(&path).unwrap();
    assert!(!before.is_empty());
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(!generate().status.success());
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn authenticated_probe_restores_a_path_with_stale_loss_history() {
    use std::time::Duration;
    use triglav::multipath::{Uplink, UplinkConfig};
    use triglav::types::ConnectionState;
    let uplink = Uplink::new(UplinkConfig::default(), 0);
    uplink.set_connection_state(ConnectionState::Connected);
    for _ in 0..20 {
        uplink.record_send(100);
        uplink.record_loss();
        uplink.record_failure("link down");
    }
    uplink.periodic_update();
    assert!(!uplink.is_usable());
    uplink.record_probe_success(Duration::from_millis(10));
    uplink.periodic_update();
    assert!(
        uplink.is_usable(),
        "historical loss must not permanently exclude a live path"
    );
}

#[tokio::test]
async fn failed_concurrent_handshake_can_be_retried() {
    use std::{sync::Arc, time::Duration};
    use triglav::multipath::{MultipathConfig, MultipathManager, UplinkConfig};
    use triglav::protocol::{Packet, PacketType};
    use triglav::types::{ConnectionState, SequenceNumber};
    let server_key = KeyPair::generate();
    let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let mut config = MultipathConfig::default();
    config.transport.connect_timeout = Duration::from_millis(200);
    let manager = Arc::new(MultipathManager::new(config, KeyPair::generate()));
    let uplink_config = UplinkConfig {
        remote_addr: socket.local_addr().unwrap(),
        ..Default::default()
    };
    let id = manager.add_uplink(uplink_config.clone()).unwrap();
    assert!(manager.connect(server_key.public).await.is_err());
    assert_eq!(
        manager.get_uplink(id).unwrap().state().connection_state,
        ConnectionState::Failed
    );
    // Drain the deliberately unanswered first attempt before servicing a retry.
    let mut buffer = [0; 2048];
    socket.recv_from(&mut buffer).await.unwrap();
    let server = tokio::spawn(async move {
        let (len, peer) = socket.recv_from(&mut buffer).await.unwrap();
        let packet = Packet::decode(&buffer[..len]).unwrap();
        let mut noise = NoiseSession::new_responder(&server_key.secret).unwrap();
        noise.read_handshake(&packet.payload).unwrap();
        let response = Packet::new(
            PacketType::Handshake,
            SequenceNumber(0),
            packet.header.session_id,
            packet.header.uplink_id,
            noise.write_handshake(&[]).unwrap(),
        )
        .unwrap()
        .encode()
        .unwrap();
        socket.send_to(&response, peer).await.unwrap();
    });
    manager
        .add_and_connect_uplink(uplink_config, &server_key.public)
        .await
        .unwrap();
    assert_eq!(manager.state(), ConnectionState::Connected);
    assert_eq!(
        manager.get_uplink(id).unwrap().state().connection_state,
        ConnectionState::Connected
    );
    server.await.unwrap();
}
