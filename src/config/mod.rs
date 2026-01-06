//! Configuration management for Triglav.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::metrics::MetricsConfig;
use crate::multipath::{MultipathConfig, UplinkConfig};
use crate::transport::TransportConfig;

/// Main configuration structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// Server configuration.
    #[serde(default)]
    pub server: ServerConfig,

    /// Client configuration.
    #[serde(default)]
    pub client: ClientConfig,

    /// Transport configuration.
    #[serde(default)]
    pub transport: TransportConfig,

    /// Multipath configuration.
    #[serde(default)]
    pub multipath: MultipathConfig,

    /// Metrics configuration.
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// Logging configuration.
    #[serde(default)]
    pub logging: LoggingConfig,
}

impl Config {
    /// Load configuration from file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to read config: {e}")))?;

        let config: Self = toml::from_str(&content)
            .map_err(|e| Error::Config(format!("Failed to parse config: {e}")))?;

        config.validate()?;
        Ok(config)
    }

    /// Save configuration to file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize config: {e}")))?;

        std::fs::write(path.as_ref(), content)
            .map_err(|e| Error::Config(format!("Failed to write config: {e}")))?;

        Ok(())
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.server.enabled && self.server.listen_addrs.is_empty() {
            return Err(Error::InvalidConfig(
                "Server enabled but no listen addresses".into(),
            ));
        }

        if self.client.enabled && self.client.uplinks.is_empty() {
            return Err(Error::InvalidConfig(
                "Client enabled but no uplinks configured".into(),
            ));
        }

        Ok(())
    }

    /// Get default config path.
    pub fn default_path() -> PathBuf {
        directories::ProjectDirs::from("com", "triglav", "triglav").map_or_else(
            || PathBuf::from("triglav.toml"),
            |dirs| dirs.config_dir().join("config.toml"),
        )
    }

    /// Create example configuration.
    pub fn example() -> Self {
        Self {
            server: ServerConfig {
                enabled: true,
                listen_addrs: vec![
                    "0.0.0.0:7443".parse().unwrap(),
                    "[::]:7443".parse().unwrap(),
                ],
                ..Default::default()
            },
            client: ClientConfig {
                enabled: true,
                uplinks: vec![
                    UplinkConfig {
                        id: "primary".into(),
                        interface: Some("en0".into()),
                        remote_addr: "server.example.com:7443".parse().unwrap(),
                        ..Default::default()
                    },
                    UplinkConfig {
                        id: "backup".into(),
                        interface: Some("en1".into()),
                        remote_addr: "server.example.com:7443".parse().unwrap(),
                        weight: 50,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Enable server mode.
    #[serde(default)]
    pub enabled: bool,

    /// Listen addresses.
    #[serde(default)]
    pub listen_addrs: Vec<SocketAddr>,

    /// Maximum concurrent connections.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Connection idle timeout.
    #[serde(default = "default_idle_timeout", with = "humantime_serde")]
    pub idle_timeout: std::time::Duration,

    /// Path to key file.
    pub key_file: Option<PathBuf>,

    /// Enable TCP fallback.
    #[serde(default = "default_tcp_fallback")]
    pub tcp_fallback: bool,

    /// Rate limit (packets per second, 0 = unlimited).
    #[serde(default)]
    pub rate_limit: u32,
}

fn default_max_connections() -> usize {
    10000
}
fn default_idle_timeout() -> std::time::Duration {
    std::time::Duration::from_secs(300)
}
fn default_tcp_fallback() -> bool {
    true
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            listen_addrs: vec![],
            max_connections: default_max_connections(),
            idle_timeout: default_idle_timeout(),
            key_file: None,
            tcp_fallback: default_tcp_fallback(),
            rate_limit: 0,
        }
    }
}

/// Client configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// Enable client mode.
    #[serde(default)]
    pub enabled: bool,

    /// Server authentication key.
    pub auth_key: Option<String>,

    /// Uplink configurations.
    #[serde(default)]
    pub uplinks: Vec<UplinkConfig>,

    /// Auto-discover network interfaces.
    #[serde(default)]
    pub auto_discover: bool,

    /// Local SOCKS5 proxy port.
    pub socks_port: Option<u16>,

    /// Local HTTP proxy port.
    pub http_proxy_port: Option<u16>,

    /// DNS server address.
    pub dns_server: Option<SocketAddr>,

    /// Reconnect delay after disconnect.
    #[serde(default = "default_reconnect_delay", with = "humantime_serde")]
    pub reconnect_delay: std::time::Duration,

    /// Maximum reconnect attempts.
    #[serde(default = "default_max_reconnects")]
    pub max_reconnects: u32,
}

fn default_reconnect_delay() -> std::time::Duration {
    std::time::Duration::from_secs(5)
}
fn default_max_reconnects() -> u32 {
    10
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            auth_key: None,
            uplinks: vec![],
            auto_discover: false,
            socks_port: None,
            http_proxy_port: None,
            dns_server: None,
            reconnect_delay: default_reconnect_delay(),
            max_reconnects: default_max_reconnects(),
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level.
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Log format (text or json).
    #[serde(default = "default_log_format")]
    pub format: String,

    /// Log file path.
    pub file: Option<PathBuf>,

    /// Enable colored output.
    #[serde(default = "default_color")]
    pub color: bool,
}

fn default_log_level() -> String {
    "info".into()
}
fn default_log_format() -> String {
    "text".into()
}
fn default_color() -> bool {
    true
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            file: None,
            color: default_color(),
        }
    }
}

/// Initialize logging.
pub fn init_logging(config: &LoggingConfig) -> Result<()> {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    let subscriber = tracing_subscriber::registry().with(filter);

    if config.format == "json" {
        subscriber
            .with(fmt::layer().json())
            .try_init()
            .map_err(|e| Error::Config(format!("Failed to init logging: {e}")))?;
    } else {
        subscriber
            .with(fmt::layer().with_ansi(config.color))
            .try_init()
            .map_err(|e| Error::Config(format!("Failed to init logging: {e}")))?;
    }

    Ok(())
}
