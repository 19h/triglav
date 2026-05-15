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
