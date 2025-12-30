//! # Triglav
//!
//! High-performance multi-path networking tool with intelligent uplink management.
//!
//! Triglav provides encrypted, redundant connections across multiple network interfaces,
//! with predictive quality assessment, automatic failover, and bandwidth aggregation.
//!
//! ## Architecture
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Application Layer                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                     Multiplexer / Demultiplexer                 │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                   Multi-Path Connection Manager                 │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
//! │  │ Uplink 1 │  │ Uplink 2 │  │ Uplink 3 │  │ Uplink N │         │
//! │  │  (WiFi)  │  │(Cellular)│  │(Ethernet)│  │   ...    │         │
//! │  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │              Quality Metrics & Prediction Engine                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                    Noise NK Encryption Layer                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │               Transport (UDP Fast Path / TCP Fallback)          │
//! └─────────────────────────────────────────────────────────────────┘

#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
// Allow stylistic lints that don't affect correctness
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_const_for_fn)]      // Many functions can't be const due to trait bounds
#![allow(clippy::doc_markdown)]              // ASCII diagrams in docs
#![allow(clippy::unreadable_literal)]        // Numeric literals are clear
#![allow(clippy::cast_possible_truncation)]  // Intentional score calculations
#![allow(clippy::cast_sign_loss)]            // Scores are always positive
#![allow(clippy::cast_precision_loss)]       // Acceptable for stats
#![allow(clippy::cast_possible_wrap)]        // Intentional for sequence arithmetic
#![allow(clippy::suboptimal_flops)]          // Clarity over micro-optimization
#![allow(clippy::similar_names)]             // state/stats are intentionally named
#![allow(clippy::significant_drop_tightening)] // Lock ordering is intentional
#![allow(clippy::option_if_let_else)]        // More readable in context
#![allow(clippy::use_self)]                  // Explicit type names in matches
#![allow(clippy::redundant_pub_crate)]       // Explicit visibility
#![allow(clippy::cognitive_complexity)]      // Complex state machines
#![allow(clippy::too_many_lines)]            // Complete implementations
#![allow(clippy::future_not_send)]           // Async internals
#![allow(clippy::struct_excessive_bools)]    // Boolean config fields are appropriate
#![allow(clippy::match_same_arms)]           // Explicit arm per variant is clearer
#![allow(clippy::return_self_not_must_use)]  // Builder methods don't need must_use
#![allow(clippy::ignored_unit_patterns)]     // Ok(_) vs Ok(()) is stylistic

pub mod config;
pub mod crypto;
pub mod error;
pub mod metrics;
pub mod multipath;
pub mod protocol;
pub mod proxy;
pub mod server;
pub mod transport;
pub mod types;
pub mod util;

#[cfg(feature = "cli")]
pub mod cli;

pub use config::Config;
pub use error::{Error, Result};
pub use types::*;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Protocol version for wire compatibility
pub const PROTOCOL_VERSION: u8 = 1;

/// Maximum transmission unit for packets
pub const MAX_MTU: usize = 1500;

/// Maximum payload size after encryption overhead
pub const MAX_PAYLOAD: usize = MAX_MTU - 64; // Reserve space for headers + auth tag

/// Default port for Triglav server
pub const DEFAULT_PORT: u16 = 7443;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::config::Config;
    pub use crate::crypto::{KeyPair, PublicKey, SecretKey};
    pub use crate::error::{Error, Result};
    pub use crate::metrics::QualityMetrics;
    pub use crate::multipath::{MultipathManager, Uplink};
    pub use crate::types::UplinkId;
    pub use crate::protocol::{Message, Packet};
    pub use crate::proxy::{Socks5Server, Socks5Config, HttpProxyServer, HttpProxyConfig};
    pub use crate::transport::{Transport, TransportConfig};
    pub use crate::types::*;
}
