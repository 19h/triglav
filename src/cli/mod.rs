//! Beautiful CLI interface for Triglav.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand, Args, ValueEnum};

/// Triglav - High-performance multi-path VPN
#[derive(Parser, Debug)]
#[command(
    name = "triglav",
    author,
    version,
    about = "High-performance multi-path VPN with intelligent uplink management",
    long_about = r#"
Triglav is a sophisticated multi-path VPN that provides:

  - True virtual network interface (TUN) for transparent tunneling
  - Encrypted, redundant connections across multiple network interfaces
  - Intelligent uplink selection based on real-time quality metrics
  - Automatic failover and bandwidth aggregation
  - ECMP-aware flow routing for connection consistency

QUICK START:
  Server:  triglav server --generate-key
  Client:  triglav tun <key> --full-tunnel
  Legacy:  triglav connect <key> --socks 1080

For more information, visit https://github.com/triglav/triglav
"#
)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Configuration file path
    #[arg(short, long, global = true)]
    pub config: Option<PathBuf>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, global = true, default_value = "info")]
    pub log_level: String,

    /// Output format
    #[arg(long, global = true, default_value = "text")]
    pub format: OutputFormat,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,

    #[command(subcommand)]
    pub command: Commands,
}

/// Available commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the Triglav server
    Server(ServerArgs),

    /// Start TUN tunnel (recommended - true VPN mode)
    Tun(TunArgs),

    /// Connect to a Triglav server (legacy proxy mode)
    Connect(ConnectArgs),

    /// Generate a new key pair
    Keygen(KeygenArgs),

    /// Show status and statistics
    Status(StatusArgs),

    /// Manage uplinks
    Uplink(UplinkArgs),

    /// Run diagnostics
    Diagnose(DiagnoseArgs),

    /// Benchmark connection
    Benchmark(BenchmarkArgs),

    /// Generate shell completions
    Completions(CompletionsArgs),

    /// Show example configuration
    Config(ConfigArgs),
}

/// Server command arguments
#[derive(Args, Debug)]
pub struct ServerArgs {
    /// Listen addresses (can be specified multiple times)
    #[arg(short, long, default_value = "0.0.0.0:7443")]
    pub listen: Vec<SocketAddr>,

    /// Path to key file
    #[arg(short, long)]
    pub key: Option<PathBuf>,

    /// Generate new key if not exists
    #[arg(long)]
    pub generate_key: bool,

    /// Print client connection key and exit
    #[arg(long)]
    pub print_key: bool,

    /// Maximum concurrent connections
    #[arg(long, default_value = "10000")]
    pub max_connections: usize,

    /// Enable TCP fallback
    #[arg(long, default_value = "true")]
    pub tcp_fallback: bool,

    /// Daemonize (run in background)
    #[arg(short, long)]
    pub daemon: bool,

    /// PID file path (for daemon mode)
    #[arg(long)]
    pub pid_file: Option<PathBuf>,
}

/// TUN tunnel command arguments (recommended mode)
#[derive(Args, Debug)]
pub struct TunArgs {
    /// Server key (tg1_...)
    pub key: String,

    /// Network interfaces to use (can be specified multiple times)
    #[arg(short, long)]
    pub interface: Vec<String>,

    /// Auto-discover network interfaces
    #[arg(long, default_value = "true")]
    pub auto_discover: bool,

    /// TUN device name (e.g., tun0, utun3)
    #[arg(long, default_value = "tg0")]
    pub tun_name: String,

    /// Tunnel IPv4 address
    #[arg(long, default_value = "10.0.85.1")]
    pub ipv4: String,

    /// Tunnel IPv6 address (optional)
    #[arg(long)]
    pub ipv6: Option<String>,

    /// Route all traffic through tunnel (full VPN mode)
    #[arg(long)]
    pub full_tunnel: bool,

    /// Specific routes to tunnel (can be specified multiple times)
    #[arg(long)]
    pub route: Vec<String>,

    /// Exclude routes from tunnel (can be specified multiple times)
    #[arg(long)]
    pub exclude: Vec<String>,

    /// Use tunnel for DNS queries
    #[arg(long)]
    pub dns: bool,

    /// Upstream DNS servers (used with --dns)
    #[arg(long, default_value = "1.1.1.1:53")]
    pub dns_server: Vec<String>,

    /// Scheduling strategy
    #[arg(long, default_value = "adaptive")]
    pub strategy: SchedulingStrategy,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Stay in foreground (don't daemonize)
    #[arg(short, long)]
    pub foreground: bool,
}

/// Connect command arguments (legacy proxy mode)
#[derive(Args, Debug)]
pub struct ConnectArgs {
    /// Server key (tg1_...)
    pub key: String,

    /// Network interfaces to use (can be specified multiple times)
    #[arg(short, long)]
    pub interface: Vec<String>,

    /// Auto-discover network interfaces
    #[arg(long)]
    pub auto_discover: bool,

    /// Local SOCKS5 proxy port
    #[arg(long)]
    pub socks: Option<u16>,

    /// Local HTTP proxy port
    #[arg(long)]
    pub http_proxy: Option<u16>,

    /// Stay in foreground (don't daemonize)
    #[arg(short, long)]
    pub foreground: bool,

    /// Scheduling strategy
    #[arg(long, default_value = "adaptive")]
    pub strategy: SchedulingStrategy,

    /// Enable verbose connection info
    #[arg(short, long)]
    pub verbose: bool,
}

/// Keygen command arguments
#[derive(Args, Debug)]
pub struct KeygenArgs {
    /// Output path for the key
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Server addresses to encode in key
    #[arg(short, long)]
    pub address: Vec<SocketAddr>,

    /// Show key in QR code
    #[arg(long)]
    pub qr: bool,

    /// Key format
    #[arg(long = "key-format", default_value = "base64")]
    pub key_format: KeyFormat,
}

/// Status command arguments
#[derive(Args, Debug, Clone)]
pub struct StatusArgs {
    /// Show detailed statistics
    #[arg(short, long)]
    pub detailed: bool,

    /// Watch mode (continuous updates)
    #[arg(short, long)]
    pub watch: bool,

    /// Update interval for watch mode (seconds)
    #[arg(long, default_value = "1")]
    pub interval: u64,

    /// Show JSON output
    #[arg(long)]
    pub json: bool,
}

/// Uplink management arguments
#[derive(Args, Debug)]
pub struct UplinkArgs {
    #[command(subcommand)]
    pub command: UplinkCommands,
}

/// Uplink subcommands
#[derive(Subcommand, Debug)]
pub enum UplinkCommands {
    /// List all uplinks
    List,
    /// Add a new uplink
    Add {
        /// Interface name
        #[arg(short, long)]
        interface: String,
        /// Weight for load balancing
        #[arg(short, long, default_value = "100")]
        weight: u32,
    },
    /// Remove an uplink
    Remove {
        /// Uplink ID or interface name
        id: String,
    },
    /// Show uplink details
    Show {
        /// Uplink ID or interface name
        id: String,
    },
    /// Enable an uplink
    Enable {
        /// Uplink ID or interface name
        id: String,
    },
    /// Disable an uplink
    Disable {
        /// Uplink ID or interface name
        id: String,
    },
}

/// Diagnose command arguments
#[derive(Args, Debug)]
pub struct DiagnoseArgs {
    /// Run full diagnostics
    #[arg(short, long)]
    pub full: bool,

    /// Test specific interface
    #[arg(short, long)]
    pub interface: Option<String>,

    /// Check connectivity to server
    #[arg(long)]
    pub connectivity: bool,

    /// Measure MTU
    #[arg(long)]
    pub mtu: bool,
}

/// Benchmark command arguments
#[derive(Args, Debug)]
pub struct BenchmarkArgs {
    /// Server key
    pub key: String,

    /// Duration in seconds
    #[arg(short, long, default_value = "10")]
    pub duration: u64,

    /// Number of parallel streams
    #[arg(short, long, default_value = "4")]
    pub streams: u32,

    /// Direction (upload, download, both)
    #[arg(long, default_value = "both")]
    pub direction: BenchmarkDirection,
}

/// Completions command arguments
#[derive(Args, Debug)]
pub struct CompletionsArgs {
    /// Shell to generate completions for
    pub shell: Shell,
}

/// Config command arguments
#[derive(Args, Debug)]
pub struct ConfigArgs {
    /// Print example server config
    #[arg(long)]
    pub server: bool,

    /// Print example client config
    #[arg(long)]
    pub client: bool,

    /// Output path
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Output format
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Table,
}

/// Scheduling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SchedulingStrategy {
    /// Weighted round-robin
    Wrr,
    /// Lowest latency
    Latency,
    /// Lowest loss
    Loss,
    /// Adaptive (recommended)
    Adaptive,
    /// Redundant (send on all)
    Redundant,
    /// Primary with backup
    PrimaryBackup,
}

impl From<SchedulingStrategy> for crate::multipath::SchedulingStrategy {
    fn from(s: SchedulingStrategy) -> Self {
        match s {
            SchedulingStrategy::Wrr => Self::WeightedRoundRobin,
            SchedulingStrategy::Latency => Self::LowestLatency,
            SchedulingStrategy::Loss => Self::LowestLoss,
            SchedulingStrategy::Adaptive => Self::Adaptive,
            SchedulingStrategy::Redundant => Self::Redundant,
            SchedulingStrategy::PrimaryBackup => Self::PrimaryBackup,
        }
    }
}

/// Key format
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum KeyFormat {
    Base64,
    Hex,
}

/// Benchmark direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BenchmarkDirection {
    Upload,
    Download,
    Both,
}

/// Shell for completions
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
}
