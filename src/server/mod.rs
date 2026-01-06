//! Server-side components for Triglav.
//!
//! This module contains:
//! - Key-based identity management
//! - Session management
//! - Daemon mode support
//! - Signal handling

mod daemon;
mod sessions;
mod signals;
mod users;

// New key-based API
pub use users::{AuthorizedKey, KeyStore};

// Legacy compatibility (deprecated)
#[allow(deprecated)]
pub use users::{User, UserKey, UserManager, UserRegistration, UserRole};

pub use daemon::*;
pub use sessions::*;
pub use signals::*;
