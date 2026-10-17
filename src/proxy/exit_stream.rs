//! Authenticated, length-bounded TCP streams through the production exit.
//! Each application stream owns its cipher state and its physical uplink.
use crate::crypto::{KeyPair, NoiseSession};
use crate::multipath::MultipathManager;
use crate::{Error, Result};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

const MAX_RECORD: usize = 65535;

async fn read_record<R: AsyncRead + Unpin>(reader: &mut R) -> Result<Vec<u8>> {
    let length = reader.read_u32().await? as usize;
    if length == 0 || length > MAX_RECORD {
        return Err(Error::InvalidPacket("invalid stream record size".into()));
    }
    let mut data = vec![0; length];
    reader.read_exact(&mut data).await?;
    Ok(data)
}

async fn write_record<W: AsyncWrite + Unpin>(writer: &mut W, data: &[u8]) -> Result<()> {
    if data.is_empty() || data.len() > MAX_RECORD {
        return Err(Error::InvalidPacket("invalid stream record size".into()));
    }
    writer.write_u32(data.len() as u32).await?;
    writer.write_all(data).await?;
    Ok(())
}

/// Relay half-closes explicitly inside authenticated records. Concurrent reads
/// and writes never share an asynchronous stream lock.
async fn relay<S>(sealed: TcpStream, plain: S, noise: NoiseSession) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let noise = Arc::new(Mutex::new(noise));
    let (mut wire_read, mut wire_write) = sealed.into_split();
    let (mut plain_read, mut plain_write) = tokio::io::split(plain);
    let sending = Arc::clone(&noise);
    let upstream = async move {
        let mut buffer = [0u8; 16384];
        loop {
            let size = plain_read.read(&mut buffer[1..]).await?;
            buffer[0] = u8::from(size != 0);
            let record = sending.lock().encrypt(&buffer[..size + 1])?;
            write_record(&mut wire_write, &record).await?;
            if size == 0 {
                wire_write.shutdown().await?;
                break;
            }
        }
        Ok::<(), Error>(())
    };
    let downstream = async move {
        loop {
            let record = read_record(&mut wire_read).await?;
            let payload = noise.lock().decrypt(&record)?;
            match payload.first() {
                Some(0) if payload.len() == 1 => {
                    plain_write.shutdown().await?;
                    break;
                }
                Some(1) => plain_write.write_all(&payload[1..]).await?,
                _ => return Err(Error::InvalidPacket("invalid stream message".into())),
            }
        }
        Ok::<(), Error>(())
    };
    tokio::try_join!(upstream, downstream)?;
    Ok(())
}

pub(crate) async fn open(
    manager: &MultipathManager,
    target: &str,
    timeout: Duration,
) -> Result<tokio::io::DuplexStream> {
    if target.len() > 1024 {
        return Err(Error::Config("proxy target too long".into()));
    }
    let uplinks = manager.stream_uplinks();
    if uplinks.is_empty() {
        return Err(Error::NoAvailableUplinks);
    }
    let start = 0;
    for index in 0..uplinks.len() {
        let uplink = &uplinks[(start + index) % uplinks.len()];
        let config = uplink.config();
        let Some(public) = uplink.remote_public_key() else {
            continue;
        };
        let attempt = async {
            let local = match (config.local_addr, config.interface.as_deref()) {
                (Some(address), _) => address,
                (None, Some(name)) => {
                    crate::util::interface_bind_address(name, config.remote_addr)?
                }
                (None, None) if config.remote_addr.is_ipv4() => "0.0.0.0:0".parse().unwrap(),
                (None, None) => "[::]:0".parse().unwrap(),
            };
            let mut options = crate::transport::SocketConfig::default();
            options.bind_interface = config.interface.clone();
            let socket = crate::transport::socket::create_tcp_socket(local, &options)?;
            socket.set_nonblocking(true)?;
            let socket = tokio::net::TcpSocket::from_std_stream(socket.into());
            let mut stream = socket.connect(config.remote_addr).await?;
            stream.set_nodelay(true)?;
            let ephemeral = KeyPair::generate();
            let mut noise = NoiseSession::new_initiator(&ephemeral.secret, &public)?;
            let mut hello = if let Some(token) = manager.access_token() {
                let mut value = b"TGP3".to_vec();
                value.extend_from_slice(&token);
                value
            } else {
                b"TGP2".to_vec()
            };
            hello.extend_from_slice(target.as_bytes());
            write_record(&mut stream, &noise.write_handshake(&hello)?).await?;
            let response = noise.read_handshake(&read_record(&mut stream).await?)?;
            if response != b"OK" {
                return Err(Error::ConnectionFailed {
                    addr: config.remote_addr,
                    reason: String::from_utf8_lossy(&response).into_owned(),
                });
            }
            Ok::<_, Error>((stream, noise))
        };
        if let Ok(Ok((stream, noise))) = tokio::time::timeout(timeout, attempt).await {
            let (application, tunnel) = tokio::io::duplex(65536);
            tokio::spawn(async move {
                if let Err(e) = relay(stream, tunnel, noise).await {
                    tracing::debug!(error = %e, "Proxy stream closed");
                }
            });
            return Ok(application);
        }
    }
    Err(Error::ConnectionFailed {
        addr: uplinks[0].config().remote_addr,
        reason: "all exit stream connections failed".into(),
    })
}

pub(crate) async fn serve(
    mut stream: TcpStream,
    keypair: KeyPair,
    access_token: Option<[u8; 32]>,
) -> Result<()> {
    stream.set_nodelay(true)?;
    let (remote, noise) = tokio::time::timeout(Duration::from_secs(10), async {
        let mut noise = NoiseSession::new_responder(&keypair.secret)?;
        let hello = noise.read_handshake(&read_record(&mut stream).await?)?;
        let target = if let Some(expected) = access_token {
            if hello.len() < 36
                || &hello[..4] != b"TGP3"
                || !crate::crypto::secure_compare(&expected, &hello[4..36])
            {
                return Err(Error::Authentication("invalid proxy access token".into()));
            }
            &hello[36..]
        } else if hello.starts_with(b"TGP3") && hello.len() >= 36 {
            &hello[36..]
        } else {
            hello
                .strip_prefix(b"TGP2")
                .ok_or_else(|| Error::Authentication("invalid proxy handshake".into()))?
        };
        if target.is_empty() || target.len() > 1024 {
            return Err(Error::InvalidPacket("invalid proxy target".into()));
        }
        let target = std::str::from_utf8(target).map_err(|e| Error::Config(e.to_string()))?;
        match TcpStream::connect(target).await {
            Ok(remote) => {
                write_record(&mut stream, &noise.write_handshake(b"OK")?).await?;
                Ok::<_, Error>((remote, noise))
            }
            Err(error) => {
                write_record(&mut stream, &noise.write_handshake(b"CONNECT_FAILED")?).await?;
                Err(Error::Io(error))
            }
        }
    })
    .await
    .map_err(|_| Error::ConnectionTimeout)??;
    relay(stream, remote, noise).await
}
