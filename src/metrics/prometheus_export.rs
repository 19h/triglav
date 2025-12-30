//! Prometheus metrics export.
//!
//! Exposes internal metrics in Prometheus format for scraping.

use std::sync::Arc;
use std::time::Duration;

use prometheus::{
    Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
    TextEncoder, Encoder,
};

/// Prometheus metrics registry and collectors.
pub struct PrometheusMetrics {
    registry: Registry,
    
    // Connection metrics
    pub connections_total: IntCounter,
    pub connections_active: IntGauge,
    pub connections_by_user: IntGaugeVec,
    
    // Session metrics
    pub sessions_total: IntCounter,
    pub sessions_active: IntGauge,
    pub session_duration_seconds: Histogram,
    
    // Uplink metrics
    pub uplinks_total: IntGauge,
    pub uplinks_active: IntGauge,
    pub uplink_state: IntGaugeVec,
    pub uplink_health: GaugeVec,
    
    // Traffic metrics
    pub bytes_sent_total: IntCounterVec,
    pub bytes_received_total: IntCounterVec,
    pub packets_sent_total: IntCounterVec,
    pub packets_received_total: IntCounterVec,
    pub packets_dropped_total: IntCounterVec,
    pub packets_retransmitted_total: IntCounterVec,
    
    // Latency metrics
    pub rtt_seconds: GaugeVec,
    pub rtt_histogram: HistogramVec,
    pub jitter_seconds: GaugeVec,
    
    // Quality metrics
    pub packet_loss_ratio: GaugeVec,
    pub quality_score: GaugeVec,
    pub bandwidth_bytes_per_sec: GaugeVec,
    
    // NAT metrics
    pub nat_type: IntGaugeVec,
    pub external_port: IntGaugeVec,
    
    // Flow metrics
    pub active_flows: IntGauge,
    pub flow_bindings: IntGaugeVec,
    
    // Scheduler metrics
    pub scheduler_selections_total: IntCounterVec,
    pub scheduler_latency_seconds: Histogram,
    
    // Crypto metrics
    pub handshakes_total: IntCounter,
    pub handshakes_failed: IntCounter,
    pub rekeys_total: IntCounter,
    pub encrypt_operations: IntCounter,
    pub decrypt_operations: IntCounter,
    
    // Error metrics
    pub errors_total: IntCounterVec,
    
    // Server metrics (if running as server)
    pub server_uptime_seconds: Gauge,
    pub users_total: IntGauge,
    pub users_active: IntGauge,
}

impl PrometheusMetrics {
    /// Create a new metrics instance with all collectors registered.
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();
        
        // Connection metrics
        let connections_total = IntCounter::new(
            "triglav_connections_total",
            "Total number of connections established"
        )?;
        let connections_active = IntGauge::new(
            "triglav_connections_active",
            "Number of currently active connections"
        )?;
        let connections_by_user = IntGaugeVec::new(
            Opts::new("triglav_connections_by_user", "Active connections per user"),
            &["user_id"]
        )?;
        
        // Session metrics
        let sessions_total = IntCounter::new(
            "triglav_sessions_total",
            "Total number of sessions created"
        )?;
        let sessions_active = IntGauge::new(
            "triglav_sessions_active",
            "Number of currently active sessions"
        )?;
        let session_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "triglav_session_duration_seconds",
                "Session duration in seconds"
            ).buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0])
        )?;
        
        // Uplink metrics
        let uplinks_total = IntGauge::new(
            "triglav_uplinks_total",
            "Total number of configured uplinks"
        )?;
        let uplinks_active = IntGauge::new(
            "triglav_uplinks_active",
            "Number of currently active/usable uplinks"
        )?;
        let uplink_state = IntGaugeVec::new(
            Opts::new("triglav_uplink_state", "Uplink connection state (0=disconnected, 1=connecting, 2=connected, 3=failed)"),
            &["uplink_id", "interface"]
        )?;
        let uplink_health = GaugeVec::new(
            Opts::new("triglav_uplink_health", "Uplink health score (0-1)"),
            &["uplink_id", "interface"]
        )?;
        
        // Traffic metrics
        let bytes_sent_total = IntCounterVec::new(
            Opts::new("triglav_bytes_sent_total", "Total bytes sent"),
            &["uplink_id"]
        )?;
        let bytes_received_total = IntCounterVec::new(
            Opts::new("triglav_bytes_received_total", "Total bytes received"),
            &["uplink_id"]
        )?;
        let packets_sent_total = IntCounterVec::new(
            Opts::new("triglav_packets_sent_total", "Total packets sent"),
            &["uplink_id"]
        )?;
        let packets_received_total = IntCounterVec::new(
            Opts::new("triglav_packets_received_total", "Total packets received"),
            &["uplink_id"]
        )?;
        let packets_dropped_total = IntCounterVec::new(
            Opts::new("triglav_packets_dropped_total", "Total packets dropped"),
            &["uplink_id", "reason"]
        )?;
        let packets_retransmitted_total = IntCounterVec::new(
            Opts::new("triglav_packets_retransmitted_total", "Total packets retransmitted"),
            &["uplink_id"]
        )?;
        
        // Latency metrics
        let rtt_seconds = GaugeVec::new(
            Opts::new("triglav_rtt_seconds", "Current smoothed RTT in seconds"),
            &["uplink_id"]
        )?;
        let rtt_histogram = HistogramVec::new(
            HistogramOpts::new(
                "triglav_rtt_histogram_seconds",
                "RTT distribution in seconds"
            ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]),
            &["uplink_id"]
        )?;
        let jitter_seconds = GaugeVec::new(
            Opts::new("triglav_jitter_seconds", "Current jitter (RTT variance) in seconds"),
            &["uplink_id"]
        )?;
        
        // Quality metrics
        let packet_loss_ratio = GaugeVec::new(
            Opts::new("triglav_packet_loss_ratio", "Packet loss ratio (0-1)"),
            &["uplink_id"]
        )?;
        let quality_score = GaugeVec::new(
            Opts::new("triglav_quality_score", "Overall quality score (0-100)"),
            &["uplink_id"]
        )?;
        let bandwidth_bytes_per_sec = GaugeVec::new(
            Opts::new("triglav_bandwidth_bytes_per_sec", "Estimated bandwidth in bytes/sec"),
            &["uplink_id", "direction"]
        )?;
        
        // NAT metrics
        let nat_type = IntGaugeVec::new(
            Opts::new("triglav_nat_type", "NAT type (0=none, 1=full_cone, 2=restricted, 3=port_restricted, 4=symmetric, 5=unknown)"),
            &["uplink_id"]
        )?;
        let external_port = IntGaugeVec::new(
            Opts::new("triglav_external_port", "External port as seen by server"),
            &["uplink_id"]
        )?;
        
        // Flow metrics
        let active_flows = IntGauge::new(
            "triglav_active_flows",
            "Number of active flows"
        )?;
        let flow_bindings = IntGaugeVec::new(
            Opts::new("triglav_flow_bindings", "Number of flows bound to each uplink"),
            &["uplink_id"]
        )?;
        
        // Scheduler metrics
        let scheduler_selections_total = IntCounterVec::new(
            Opts::new("triglav_scheduler_selections_total", "Total scheduler selections per uplink"),
            &["uplink_id", "strategy"]
        )?;
        let scheduler_latency_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "triglav_scheduler_latency_seconds",
                "Time spent in scheduler selection"
            ).buckets(vec![0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01])
        )?;
        
        // Crypto metrics
        let handshakes_total = IntCounter::new(
            "triglav_handshakes_total",
            "Total number of Noise handshakes completed"
        )?;
        let handshakes_failed = IntCounter::new(
            "triglav_handshakes_failed",
            "Total number of failed Noise handshakes"
        )?;
        let rekeys_total = IntCounter::new(
            "triglav_rekeys_total",
            "Total number of rekey operations"
        )?;
        let encrypt_operations = IntCounter::new(
            "triglav_encrypt_operations_total",
            "Total encryption operations"
        )?;
        let decrypt_operations = IntCounter::new(
            "triglav_decrypt_operations_total",
            "Total decryption operations"
        )?;
        
        // Error metrics
        let errors_total = IntCounterVec::new(
            Opts::new("triglav_errors_total", "Total errors by type"),
            &["error_type"]
        )?;
        
        // Server metrics
        let server_uptime_seconds = Gauge::new(
            "triglav_server_uptime_seconds",
            "Server uptime in seconds"
        )?;
        let users_total = IntGauge::new(
            "triglav_users_total",
            "Total registered users"
        )?;
        let users_active = IntGauge::new(
            "triglav_users_active",
            "Currently active users"
        )?;
        
        // Register all metrics
        registry.register(Box::new(connections_total.clone()))?;
        registry.register(Box::new(connections_active.clone()))?;
        registry.register(Box::new(connections_by_user.clone()))?;
        registry.register(Box::new(sessions_total.clone()))?;
        registry.register(Box::new(sessions_active.clone()))?;
        registry.register(Box::new(session_duration_seconds.clone()))?;
        registry.register(Box::new(uplinks_total.clone()))?;
        registry.register(Box::new(uplinks_active.clone()))?;
        registry.register(Box::new(uplink_state.clone()))?;
        registry.register(Box::new(uplink_health.clone()))?;
        registry.register(Box::new(bytes_sent_total.clone()))?;
        registry.register(Box::new(bytes_received_total.clone()))?;
        registry.register(Box::new(packets_sent_total.clone()))?;
        registry.register(Box::new(packets_received_total.clone()))?;
        registry.register(Box::new(packets_dropped_total.clone()))?;
        registry.register(Box::new(packets_retransmitted_total.clone()))?;
        registry.register(Box::new(rtt_seconds.clone()))?;
        registry.register(Box::new(rtt_histogram.clone()))?;
        registry.register(Box::new(jitter_seconds.clone()))?;
        registry.register(Box::new(packet_loss_ratio.clone()))?;
        registry.register(Box::new(quality_score.clone()))?;
        registry.register(Box::new(bandwidth_bytes_per_sec.clone()))?;
        registry.register(Box::new(nat_type.clone()))?;
        registry.register(Box::new(external_port.clone()))?;
        registry.register(Box::new(active_flows.clone()))?;
        registry.register(Box::new(flow_bindings.clone()))?;
        registry.register(Box::new(scheduler_selections_total.clone()))?;
        registry.register(Box::new(scheduler_latency_seconds.clone()))?;
        registry.register(Box::new(handshakes_total.clone()))?;
        registry.register(Box::new(handshakes_failed.clone()))?;
        registry.register(Box::new(rekeys_total.clone()))?;
        registry.register(Box::new(encrypt_operations.clone()))?;
        registry.register(Box::new(decrypt_operations.clone()))?;
        registry.register(Box::new(errors_total.clone()))?;
        registry.register(Box::new(server_uptime_seconds.clone()))?;
        registry.register(Box::new(users_total.clone()))?;
        registry.register(Box::new(users_active.clone()))?;
        
        Ok(Self {
            registry,
            connections_total,
            connections_active,
            connections_by_user,
            sessions_total,
            sessions_active,
            session_duration_seconds,
            uplinks_total,
            uplinks_active,
            uplink_state,
            uplink_health,
            bytes_sent_total,
            bytes_received_total,
            packets_sent_total,
            packets_received_total,
            packets_dropped_total,
            packets_retransmitted_total,
            rtt_seconds,
            rtt_histogram,
            jitter_seconds,
            packet_loss_ratio,
            quality_score,
            bandwidth_bytes_per_sec,
            nat_type,
            external_port,
            active_flows,
            flow_bindings,
            scheduler_selections_total,
            scheduler_latency_seconds,
            handshakes_total,
            handshakes_failed,
            rekeys_total,
            encrypt_operations,
            decrypt_operations,
            errors_total,
            server_uptime_seconds,
            users_total,
            users_active,
        })
    }

    /// Encode metrics to Prometheus text format.
    pub fn encode(&self) -> Result<String, prometheus::Error> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer).unwrap_or_default())
    }

    /// Record an RTT observation.
    pub fn record_rtt(&self, uplink_id: &str, rtt: Duration) {
        let seconds = rtt.as_secs_f64();
        self.rtt_seconds.with_label_values(&[uplink_id]).set(seconds);
        self.rtt_histogram.with_label_values(&[uplink_id]).observe(seconds);
    }

    /// Record packet sent.
    pub fn record_packet_sent(&self, uplink_id: &str, bytes: u64) {
        self.packets_sent_total.with_label_values(&[uplink_id]).inc();
        self.bytes_sent_total.with_label_values(&[uplink_id]).inc_by(bytes);
    }

    /// Record packet received.
    pub fn record_packet_received(&self, uplink_id: &str, bytes: u64) {
        self.packets_received_total.with_label_values(&[uplink_id]).inc();
        self.bytes_received_total.with_label_values(&[uplink_id]).inc_by(bytes);
    }

    /// Record packet dropped.
    pub fn record_packet_dropped(&self, uplink_id: &str, reason: &str) {
        self.packets_dropped_total.with_label_values(&[uplink_id, reason]).inc();
    }

    /// Record uplink state.
    pub fn set_uplink_state(&self, uplink_id: &str, interface: &str, state: i64) {
        self.uplink_state.with_label_values(&[uplink_id, interface]).set(state);
    }

    /// Record uplink health.
    pub fn set_uplink_health(&self, uplink_id: &str, interface: &str, health: f64) {
        self.uplink_health.with_label_values(&[uplink_id, interface]).set(health);
    }

    /// Record uplink quality metrics.
    pub fn set_uplink_quality(&self, uplink_id: &str, loss: f64, score: f64, jitter_secs: f64) {
        self.packet_loss_ratio.with_label_values(&[uplink_id]).set(loss);
        self.quality_score.with_label_values(&[uplink_id]).set(score);
        self.jitter_seconds.with_label_values(&[uplink_id]).set(jitter_secs);
    }

    /// Record bandwidth.
    pub fn set_bandwidth(&self, uplink_id: &str, send_bps: f64, recv_bps: f64) {
        self.bandwidth_bytes_per_sec.with_label_values(&[uplink_id, "send"]).set(send_bps);
        self.bandwidth_bytes_per_sec.with_label_values(&[uplink_id, "recv"]).set(recv_bps);
    }

    /// Record error.
    pub fn record_error(&self, error_type: &str) {
        self.errors_total.with_label_values(&[error_type]).inc();
    }

    /// Record scheduler selection.
    pub fn record_scheduler_selection(&self, uplink_id: &str, strategy: &str, latency: Duration) {
        self.scheduler_selections_total.with_label_values(&[uplink_id, strategy]).inc();
        self.scheduler_latency_seconds.observe(latency.as_secs_f64());
    }
}

impl Default for PrometheusMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create Prometheus metrics")
    }
}

/// Global metrics instance.
static METRICS: std::sync::OnceLock<Arc<PrometheusMetrics>> = std::sync::OnceLock::new();

/// Initialize the global metrics instance.
pub fn init_metrics() -> Arc<PrometheusMetrics> {
    METRICS.get_or_init(|| {
        Arc::new(PrometheusMetrics::new().expect("Failed to initialize metrics"))
    }).clone()
}

/// Get the global metrics instance.
pub fn get_metrics() -> Option<Arc<PrometheusMetrics>> {
    METRICS.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = PrometheusMetrics::new().unwrap();
        
        metrics.connections_total.inc();
        metrics.record_rtt("uplink-1", Duration::from_millis(50));
        metrics.record_packet_sent("uplink-1", 1000);
        
        let output = metrics.encode().unwrap();
        assert!(output.contains("triglav_connections_total"));
        assert!(output.contains("triglav_rtt_seconds"));
    }

    #[test]
    fn test_global_metrics() {
        let m1 = init_metrics();
        let m2 = init_metrics();
        
        // Should be the same instance
        assert!(Arc::ptr_eq(&m1, &m2));
    }
}
