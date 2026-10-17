//! Bounded, atomic packet writes. A partial write is never a delivered packet.
use std::future::Future;
use std::time::Duration;

use crate::{Error, Result};

pub(crate) const PACKET_IO_TIMEOUT: Duration = Duration::from_secs(1);

pub(crate) async fn complete_packet_write(
    write: impl Future<Output = Result<usize>>,
    expected: usize,
    deadline: Duration,
) -> Result<()> {
    let written = tokio::time::timeout(deadline, write)
        .await
        .map_err(|_| Error::ConnectionTimeout)??;
    if written != expected {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::WriteZero,
            "incomplete packet write",
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn packet_completion_requires_full_write_and_a_finite_wait() {
        assert!(
            complete_packet_write(async { Ok(80) }, 80, PACKET_IO_TIMEOUT)
                .await
                .is_ok()
        );
        assert!(matches!(
            complete_packet_write(async { Ok(79) }, 80, PACKET_IO_TIMEOUT).await,
            Err(Error::Io(_))
        ));
        assert!(matches!(
            complete_packet_write(std::future::pending(), 80, Duration::from_millis(10)).await,
            Err(Error::ConnectionTimeout)
        ));
        assert!(matches!(
            complete_packet_write(
                async { Err(Error::ConnectionClosed) },
                80,
                PACKET_IO_TIMEOUT
            )
            .await,
            Err(Error::ConnectionClosed)
        ));
    }
}
