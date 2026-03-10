//! Client status providers for local HTTP metrics and GUI integration.

use std::collections::{BTreeSet, HashMap};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::task::JoinHandle;
use tracing::warn;

use super::{
    init_metrics, CheckResult, HealthChecker, HealthDetails, HttpServerConfig, MetricsHttpServer,
    QualityStatus, SessionStatus, StatusProvider, StatusResponse, TunnelStatus, UplinkStatus,
};
use crate::multipath::MultipathManager;

/// Static runtime configuration exposed in the client status endpoint.
#[derive(Debug, Clone)]
pub struct ClientRuntimeConfig {
    /// Client mode name (`tun` or `proxy`).
    pub mode: String,
    /// TUN details when running in tunnel mode.
    pub tunnel: Option<TunnelRuntimeConfig>,
}

/// Tunnel-specific runtime details.
#[derive(Debug, Clone)]
pub struct TunnelRuntimeConfig {
    /// TUN interface name.
    pub tun_name: String,
    /// Whether the tunnel is full-tunnel.
    pub full_tunnel: bool,
    /// Included routes.
    pub include_routes: Vec<String>,
    /// Excluded routes.
    pub exclude_routes: Vec<String>,
}

/// Local client status provider backed by the live multipath manager.
pub struct ClientStatusProvider {
    manager: Arc<MultipathManager>,
    runtime: ClientRuntimeConfig,
    connected_at: DateTime<Utc>,
}

impl ClientStatusProvider {
    /// Create a new client status provider.
    pub fn new(manager: Arc<MultipathManager>, runtime: ClientRuntimeConfig) -> Self {
        Self {
            manager,
            runtime,
            connected_at: Utc::now(),
        }
    }
}

impl StatusProvider for ClientStatusProvider {
    fn get_status(&self) -> StatusResponse {
        let traffic = self.manager.stats();
        let quality = self.manager.quality_summary();
        let uplinks: Vec<UplinkStatus> = self
            .manager
            .uplinks()
            .into_iter()
            .map(|uplink| {
                let uplink_state = uplink.state();
                let uplink_stats = uplink.stats();
                let config = uplink.config();

                UplinkStatus {
                    id: uplink.id().to_string(),
                    interface: config.interface.clone(),
                    remote_addr: config.remote_addr.to_string(),
                    state: uplink_state.connection_state.to_string(),
                    health: uplink_state.health.to_string(),
                    rtt_ms: uplink.rtt().as_secs_f64() * 1000.0,
                    loss_percent: uplink.loss_ratio() * 100.0,
                    bandwidth_mbps: uplink.bandwidth().as_mbps(),
                    bytes_sent: uplink_stats.bytes_sent,
                    bytes_received: uplink_stats.bytes_received,
                    nat_type: format!("{:?}", uplink.nat_type()).to_lowercase(),
                    external_addr: uplink.external_addr().map(|addr| addr.to_string()),
                }
            })
            .collect();

        let remote_addrs: Vec<String> = uplinks
            .iter()
            .map(|uplink| uplink.remote_addr.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        let active_uplinks = uplinks
            .iter()
            .filter(|uplink| uplink.state == "connected")
            .map(|uplink| uplink.id.clone())
            .collect();

        StatusResponse {
            version: crate::VERSION.to_string(),
            uptime_seconds: 0,
            state: self.manager.state().to_string(),
            role: Some("client".to_string()),
            mode: Some(self.runtime.mode.clone()),
            process_id: Some(std::process::id()),
            session_id: Some(self.manager.session_id().to_string()),
            connection_id: Some(self.manager.connection_id().to_string()),
            quality: Some(QualityStatus {
                usable_uplinks: quality.usable_uplinks,
                total_uplinks: quality.total_uplinks,
                avg_rtt_ms: quality.avg_rtt.as_secs_f64() * 1000.0,
                avg_loss_percent: quality.avg_loss * 100.0,
                total_bandwidth_mbps: quality.total_bandwidth.as_mbps(),
                packets_sent: quality.stats.packets_sent,
                packets_received: quality.stats.packets_received,
                packets_dropped: quality.stats.packets_dropped,
            }),
            tunnel: self.runtime.tunnel.as_ref().map(|tunnel| TunnelStatus {
                tun_name: tunnel.tun_name.clone(),
                full_tunnel: tunnel.full_tunnel,
                include_routes: tunnel.include_routes.clone(),
                exclude_routes: tunnel.exclude_routes.clone(),
            }),
            uplinks,
            sessions: vec![SessionStatus {
                id: self.manager.session_id().to_string(),
                user_id: None,
                remote_addrs,
                connected_at: self.connected_at.to_rfc3339(),
                bytes_sent: traffic.bytes_sent,
                bytes_received: traffic.bytes_received,
                uplinks_used: active_uplinks,
            }],
            total_bytes_sent: traffic.bytes_sent,
            total_bytes_received: traffic.bytes_received,
            total_connections: usize::from(self.manager.state().is_established()) as u64,
        }
    }
}

/// Health checker for the local client process.
pub struct ClientHealthChecker {
    manager: Arc<MultipathManager>,
}

impl ClientHealthChecker {
    /// Create a new client health checker.
    pub fn new(manager: Arc<MultipathManager>) -> Self {
        Self { manager }
    }
}

impl HealthChecker for ClientHealthChecker {
    fn is_alive(&self) -> bool {
        true
    }

    fn is_ready(&self) -> bool {
        self.manager.state().is_established() && self.manager.quality_summary().usable_uplinks > 0
    }

    fn health_details(&self) -> HealthDetails {
        let state = self.manager.state();
        let quality = self.manager.quality_summary();
        let mut checks = HashMap::new();

        checks.insert(
            "connection".to_string(),
            CheckResult {
                status: if state.is_established() {
                    "ok".to_string()
                } else {
                    "waiting".to_string()
                },
                message: Some(format!("state={state}")),
                duration_ms: None,
            },
        );

        checks.insert(
            "uplinks".to_string(),
            CheckResult {
                status: if quality.usable_uplinks > 0 {
                    "ok".to_string()
                } else {
                    "degraded".to_string()
                },
                message: Some(format!(
                    "usable={}/{} avg_rtt_ms={:.1} loss={:.1}%",
                    quality.usable_uplinks,
                    quality.total_uplinks,
                    quality.avg_rtt.as_secs_f64() * 1000.0,
                    quality.avg_loss * 100.0
                )),
                duration_ms: None,
            },
        );

        HealthDetails {
            alive: self.is_alive(),
            ready: self.is_ready(),
            checks,
        }
    }
}

/// Start a local metrics and status server for a client process.
pub fn start_client_status_server(
    bind_addr: SocketAddr,
    manager: Arc<MultipathManager>,
    runtime: ClientRuntimeConfig,
) -> JoinHandle<()> {
    let metrics = init_metrics();
    let status_provider = Arc::new(ClientStatusProvider::new(Arc::clone(&manager), runtime));
    let health_checker = Arc::new(ClientHealthChecker::new(manager));
    let http_server = MetricsHttpServer::new(
        HttpServerConfig {
            bind_addr,
            enable_cors: true,
            shutdown_timeout: Duration::from_secs(5),
        },
        metrics,
        status_provider,
        health_checker,
    );

    tokio::spawn(async move {
        if let Err(error) = http_server.start().await {
            warn!(bind_addr = %bind_addr, error = %error, "Client status server exited");
        }
    })
}
