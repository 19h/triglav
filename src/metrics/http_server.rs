//! HTTP server for metrics, health, and status endpoints.
//!
//! Provides:
//! - `/metrics` - Prometheus metrics
//! - `/health` - Health check (liveness probe)
//! - `/ready` - Readiness check
//! - `/status` - Detailed status information

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::info;

use super::prometheus_export::PrometheusMetrics;

/// HTTP server configuration.
#[derive(Debug, Clone)]
pub struct HttpServerConfig {
    /// Bind address.
    pub bind_addr: SocketAddr,
    /// Enable CORS.
    pub enable_cors: bool,
    /// Shutdown timeout.
    pub shutdown_timeout: Duration,
}

impl Default for HttpServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:9090".parse().unwrap(),
            enable_cors: true,
            shutdown_timeout: Duration::from_secs(5),
        }
    }
}

/// Shared state for HTTP handlers.
#[derive(Clone)]
pub struct HttpServerState {
    /// Prometheus metrics.
    pub metrics: Arc<PrometheusMetrics>,
    /// Status provider.
    pub status_provider: Arc<dyn StatusProvider + Send + Sync>,
    /// Health checker.
    pub health_checker: Arc<dyn HealthChecker + Send + Sync>,
    /// Server start time.
    pub start_time: Instant,
}

/// Trait for providing status information.
pub trait StatusProvider: Send + Sync {
    /// Get current status.
    fn get_status(&self) -> StatusResponse;
}

/// Trait for health checking.
pub trait HealthChecker: Send + Sync {
    /// Check if the service is alive.
    fn is_alive(&self) -> bool;
    /// Check if the service is ready to accept traffic.
    fn is_ready(&self) -> bool;
    /// Get detailed health information.
    fn health_details(&self) -> HealthDetails;
}

/// Default status provider.
pub struct DefaultStatusProvider {
    uplinks: Arc<RwLock<Vec<UplinkStatus>>>,
    sessions: Arc<RwLock<Vec<SessionStatus>>>,
}

impl DefaultStatusProvider {
    pub fn new() -> Self {
        Self {
            uplinks: Arc::new(RwLock::new(Vec::new())),
            sessions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn update_uplinks(&self, uplinks: Vec<UplinkStatus>) {
        *self.uplinks.write() = uplinks;
    }

    pub fn update_sessions(&self, sessions: Vec<SessionStatus>) {
        *self.sessions.write() = sessions;
    }
}

impl StatusProvider for DefaultStatusProvider {
    fn get_status(&self) -> StatusResponse {
        StatusResponse {
            version: crate::VERSION.to_string(),
            uptime_seconds: 0, // Will be set by handler
            state: "running".to_string(),
            uplinks: self.uplinks.read().clone(),
            sessions: self.sessions.read().clone(),
            total_bytes_sent: 0,
            total_bytes_received: 0,
            total_connections: 0,
        }
    }
}

/// Default health checker.
pub struct DefaultHealthChecker {
    alive: Arc<RwLock<bool>>,
    ready: Arc<RwLock<bool>>,
}

impl DefaultHealthChecker {
    pub fn new() -> Self {
        Self {
            alive: Arc::new(RwLock::new(true)),
            ready: Arc::new(RwLock::new(false)),
        }
    }

    pub fn set_alive(&self, alive: bool) {
        *self.alive.write() = alive;
    }

    pub fn set_ready(&self, ready: bool) {
        *self.ready.write() = ready;
    }
}

impl HealthChecker for DefaultHealthChecker {
    fn is_alive(&self) -> bool {
        *self.alive.read()
    }

    fn is_ready(&self) -> bool {
        *self.ready.read()
    }

    fn health_details(&self) -> HealthDetails {
        HealthDetails {
            alive: self.is_alive(),
            ready: self.is_ready(),
            checks: HashMap::new(),
        }
    }
}

/// Status response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub version: String,
    pub uptime_seconds: u64,
    pub state: String,
    pub uplinks: Vec<UplinkStatus>,
    pub sessions: Vec<SessionStatus>,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub total_connections: u64,
}

/// Uplink status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UplinkStatus {
    pub id: String,
    pub interface: Option<String>,
    pub remote_addr: String,
    pub state: String,
    pub health: String,
    pub rtt_ms: f64,
    pub loss_percent: f64,
    pub bandwidth_mbps: f64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub nat_type: String,
    pub external_addr: Option<String>,
}

/// Session status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatus {
    pub id: String,
    pub user_id: Option<String>,
    pub remote_addrs: Vec<String>,
    pub connected_at: String,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub uplinks_used: Vec<String>,
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<HealthDetails>,
}

/// Detailed health information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDetails {
    pub alive: bool,
    pub ready: bool,
    pub checks: HashMap<String, CheckResult>,
}

/// Individual health check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub status: String,
    pub message: Option<String>,
    pub duration_ms: Option<u64>,
}

/// HTTP metrics server.
pub struct MetricsHttpServer {
    config: HttpServerConfig,
    state: HttpServerState,
    shutdown_tx: broadcast::Sender<()>,
}

impl MetricsHttpServer {
    /// Create a new HTTP server.
    pub fn new(
        config: HttpServerConfig,
        metrics: Arc<PrometheusMetrics>,
        status_provider: Arc<dyn StatusProvider + Send + Sync>,
        health_checker: Arc<dyn HealthChecker + Send + Sync>,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);

        Self {
            config,
            state: HttpServerState {
                metrics,
                status_provider,
                health_checker,
                start_time: Instant::now(),
            },
            shutdown_tx,
        }
    }

    /// Start the HTTP server.
    pub async fn start(&self) -> Result<(), std::io::Error> {
        let app = self.build_router();
        let addr = self.config.bind_addr;

        info!("Starting metrics HTTP server on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.recv().await;
            })
            .await
    }

    /// Build the router.
    fn build_router(&self) -> Router {
        let state = self.state.clone();

        let router = Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_handler))
            .route("/health/live", get(liveness_handler))
            .route("/health/ready", get(readiness_handler))
            .route("/ready", get(readiness_handler))
            .route("/status", get(status_handler))
            .route("/", get(root_handler))
            .with_state(state);

        #[cfg(feature = "metrics")]
        let router = {
            use tower_http::cors::{Any, CorsLayer};
            use tower_http::trace::TraceLayer;

            if self.config.enable_cors {
                router
                    .layer(CorsLayer::new().allow_origin(Any))
                    .layer(TraceLayer::new_for_http())
            } else {
                router.layer(TraceLayer::new_for_http())
            }
        };

        router
    }

    /// Stop the server.
    pub fn stop(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

/// Root handler - returns service info.
async fn root_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "triglav",
        "version": crate::VERSION,
        "endpoints": ["/metrics", "/health", "/health/live", "/health/ready", "/status"]
    }))
}

/// Metrics handler - returns Prometheus metrics.
async fn metrics_handler(State(state): State<HttpServerState>) -> impl IntoResponse {
    match state.metrics.encode() {
        Ok(metrics) => (
            StatusCode::OK,
            [("content-type", "text/plain; charset=utf-8")],
            metrics,
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [("content-type", "text/plain; charset=utf-8")],
            format!("Failed to encode metrics: {}", e),
        ),
    }
}

/// Health handler - returns overall health status.
#[derive(Debug, Deserialize)]
struct HealthQuery {
    #[serde(default)]
    verbose: bool,
}

async fn health_handler(
    State(state): State<HttpServerState>,
    Query(query): Query<HealthQuery>,
) -> impl IntoResponse {
    let alive = state.health_checker.is_alive();
    let ready = state.health_checker.is_ready();

    let status = if alive && ready {
        "healthy"
    } else if alive {
        "degraded"
    } else {
        "unhealthy"
    };

    let status_code = if alive {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let response = HealthResponse {
        status: status.to_string(),
        details: if query.verbose {
            Some(state.health_checker.health_details())
        } else {
            None
        },
    };

    (status_code, Json(response))
}

/// Liveness handler - for Kubernetes liveness probe.
async fn liveness_handler(State(state): State<HttpServerState>) -> impl IntoResponse {
    if state.health_checker.is_alive() {
        (StatusCode::OK, "OK")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "NOT OK")
    }
}

/// Readiness handler - for Kubernetes readiness probe.
async fn readiness_handler(State(state): State<HttpServerState>) -> impl IntoResponse {
    if state.health_checker.is_ready() {
        (StatusCode::OK, "READY")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "NOT READY")
    }
}

/// Status handler - returns detailed status.
async fn status_handler(State(state): State<HttpServerState>) -> impl IntoResponse {
    let mut status = state.status_provider.get_status();
    status.uptime_seconds = state.start_time.elapsed().as_secs();

    Json(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "healthy".to_string(),
            details: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("healthy"));
        assert!(!json.contains("details"));
    }

    #[test]
    fn test_default_health_checker() {
        let checker = DefaultHealthChecker::new();

        assert!(checker.is_alive());
        assert!(!checker.is_ready());

        checker.set_ready(true);
        assert!(checker.is_ready());
    }
}
