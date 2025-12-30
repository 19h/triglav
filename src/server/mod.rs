//! Server-side components for Triglav.
//!
//! This module contains:
//! - Key-based identity management
//! - Session management
//! - Daemon mode support
//! - Signal handling

mod users;
mod sessions;
mod daemon;
mod signals;

// New key-based API
pub use users::{AuthorizedKey, KeyStore};

// Legacy compatibility (deprecated)
#[allow(deprecated)]
pub use users::{User, UserManager, UserRole, UserRegistration, UserKey};

pub use sessions::*;
pub use daemon::*;
pub use signals::*;
