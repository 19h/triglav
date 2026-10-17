//! Server-side components for Triglav.
//!
//! This module contains:
//! - Key-based identity management
//! - Session management
//! - Daemon mode support
//! - Signal handling

mod daemon;
#[cfg(feature = "metrics")]
mod runtime;
mod sessions;
mod signals;
mod users;

// New key-based API
pub use users::{AuthorizedKey, KeyStore};

// Legacy compatibility (deprecated)
#[allow(deprecated)]
pub use users::{User, UserKey, UserManager, UserRegistration, UserRole};

pub use daemon::*;
#[cfg(feature = "metrics")]
pub use runtime::*;
pub use sessions::*;
pub use signals::*;

mod exit;
pub use exit::*;

/// Service-manager termination and interactive interruption share cleanup.
pub async fn shutdown_signal() -> crate::Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result?,
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}
