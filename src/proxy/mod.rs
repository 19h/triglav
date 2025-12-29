//! Proxy servers for Triglav client.
//!
//! Provides SOCKS5 and HTTP proxy functionality for routing traffic
//! through the multipath connection.

mod http;
mod socks5;

pub use http::{HttpProxyServer, HttpProxyConfig};
pub use socks5::{Socks5Server, Socks5Config};
