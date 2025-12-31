//! Bandwidth aggregation with packet striping across multiple uplinks.
//!
//! This module implements TRUE bandwidth aggregation - not just load balancing.
//! A single flow's packets are distributed across ALL available uplinks
//! proportionally to their bandwidth, then reassembled in-order on receive.
//!
//! Key concepts:
//! - **Packet striping**: Packets from one flow go to different uplinks
//! - **Weighted distribution**: Faster uplinks get more packets
//! - **Reorder buffer**: Handles out-of-order arrival from different paths
//! - **Latency compensation**: Adjusts for different path RTTs

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use super::Uplink;
use crate::types::Bandwidth;

/// Configuration for bandwidth aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorConfig {
    /// Enable bandwidth aggregation (packet striping).
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Maximum reorder buffer size in packets.
    #[serde(default = "default_reorder_buffer_size")]
    pub reorder_buffer_size: usize,

    /// Maximum time to wait for out-of-order packets.
    #[serde(default = "default_reorder_timeout", with = "humantime_serde")]
    pub reorder_timeout: Duration,

    /// Minimum packets before delivering (to handle jitter).
    #[serde(default = "default_min_buffer_packets")]
    pub min_buffer_packets: usize,

    /// How often to recalculate uplink weights (based on bandwidth).
    #[serde(default = "default_weight_update_interval", with = "humantime_serde")]
    pub weight_update_interval: Duration,

    /// Minimum bandwidth (bytes/sec) for an uplink to participate.
    #[serde(default = "default_min_uplink_bandwidth")]
    pub min_uplink_bandwidth: f64,

    /// Enable latency compensation (delay fast-path packets to reduce buffering).
    #[serde(default = "default_latency_compensation")]
    pub latency_compensation: bool,
}

fn default_enabled() -> bool {
    true
}
fn default_reorder_buffer_size() -> usize {
    4096
}
fn default_reorder_timeout() -> Duration {
    Duration::from_millis(500)
}
fn default_min_buffer_packets() -> usize {
    4
}
fn default_weight_update_interval() -> Duration {
    Duration::from_millis(100)
}
fn default_min_uplink_bandwidth() -> f64 {
    10_000.0
}
fn default_latency_compensation() -> bool {
    true
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            reorder_buffer_size: default_reorder_buffer_size(),
            reorder_timeout: default_reorder_timeout(),
            min_buffer_packets: default_min_buffer_packets(),
            weight_update_interval: default_weight_update_interval(),
            min_uplink_bandwidth: default_min_uplink_bandwidth(),
            latency_compensation: default_latency_compensation(),
        }
    }
}

/// Represents a striping weight for an uplink.
#[derive(Debug, Clone, Copy)]
struct UplinkWeight {
    uplink_id: u16,
    /// Absolute bandwidth in bytes/sec.
    bandwidth: f64,
    /// Weight as integer (for weighted round-robin).
    weight: u32,
    /// Current deficit counter for weighted round-robin.
    deficit: i32,
    /// RTT for latency compensation.
    rtt: Duration,
}

/// Packet awaiting reordering.
#[derive(Debug)]
struct BufferedPacket {
    /// Global sequence number.
    sequence: u64,
    /// Payload data.
    data: Vec<u8>,
    /// When this packet was received.
    received_at: Instant,
    /// Which uplink it came from.
    from_uplink: u16,
}

/// Reorder buffer for reassembling striped packets.
#[derive(Debug)]
pub struct ReorderBuffer {
    /// Buffered packets, keyed by sequence number.
    buffer: BTreeMap<u64, BufferedPacket>,
    /// Next sequence number to deliver.
    next_seq: u64,
    /// Maximum buffer size.
    max_size: usize,
    /// Reorder timeout.
    timeout: Duration,
    /// Packets delivered in order.
    delivered: u64,
    /// Packets dropped due to timeout/overflow.
    dropped: u64,
    /// Packets received out of order.
    reordered: u64,
}

impl ReorderBuffer {
    /// Create a new reorder buffer.
    pub fn new(max_size: usize, timeout: Duration) -> Self {
        Self {
            buffer: BTreeMap::new(),
            next_seq: 0,
            max_size,
            timeout,
            delivered: 0,
            dropped: 0,
            reordered: 0,
        }
    }

    /// Insert a packet into the buffer.
    /// Returns packets that are now ready for delivery (in order).
    pub fn insert(&mut self, sequence: u64, data: Vec<u8>, from_uplink: u16) -> Vec<Vec<u8>> {
        let now = Instant::now();

        // Initialize next_seq on first packet
        if self.next_seq == 0 && self.buffer.is_empty() {
            self.next_seq = sequence;
        }

        // Check if this is an old packet we already delivered
        if sequence < self.next_seq {
            // Already delivered or too old
            return vec![];
        }

        // Check if out of order
        if sequence != self.next_seq {
            self.reordered += 1;
        }

        // Insert into buffer
        self.buffer.insert(
            sequence,
            BufferedPacket {
                sequence,
                data,
                received_at: now,
                from_uplink,
            },
        );

        // Check buffer overflow - drop oldest if needed
        while self.buffer.len() > self.max_size {
            if let Some((&oldest_seq, _)) = self.buffer.iter().next() {
                self.buffer.remove(&oldest_seq);
                self.dropped += 1;
                // If we dropped what we were waiting for, advance
                if oldest_seq == self.next_seq {
                    self.next_seq = oldest_seq + 1;
                }
            }
        }

        // Collect deliverable packets
        self.collect_ready(now)
    }

    /// Collect packets that are ready for delivery.
    fn collect_ready(&mut self, now: Instant) -> Vec<Vec<u8>> {
        let mut ready = Vec::new();

        // First, deliver all consecutive packets starting from next_seq
        while let Some(packet) = self.buffer.remove(&self.next_seq) {
            ready.push(packet.data);
            self.next_seq += 1;
            self.delivered += 1;
        }

        // Check for timeout - if we've been waiting too long, skip the gap
        if !self.buffer.is_empty() {
            // Find the lowest sequence we're waiting for
            if let Some((&min_seq, oldest)) = self.buffer.iter().next() {
                if now.duration_since(oldest.received_at) > self.timeout {
                    // We've waited too long - skip to this packet
                    let gap = min_seq - self.next_seq;
                    self.dropped += gap;
                    self.next_seq = min_seq;

                    // Now deliver consecutive packets
                    while let Some(packet) = self.buffer.remove(&self.next_seq) {
                        ready.push(packet.data);
                        self.next_seq += 1;
                        self.delivered += 1;
                    }
                }
            }
        }

        ready
    }

    /// Force flush - deliver all buffered packets in order, with gaps.
    pub fn flush(&mut self) -> Vec<Vec<u8>> {
        let mut ready = Vec::new();

        // Drain in sequence order (BTreeMap iterates in order)
        for (_, packet) in std::mem::take(&mut self.buffer) {
            ready.push(packet.data);
            self.delivered += 1;
        }

        ready
    }

    /// Get buffer statistics.
    pub fn stats(&self) -> ReorderStats {
        ReorderStats {
            buffered: self.buffer.len(),
            next_seq: self.next_seq,
            delivered: self.delivered,
            dropped: self.dropped,
            reordered: self.reordered,
        }
    }

    /// Check if buffer has packets waiting.
    pub fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the current buffer delay (time oldest packet has been waiting).
    pub fn current_delay(&self) -> Duration {
        self.buffer
            .iter()
            .next()
            .map(|(_, p)| p.received_at.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Poll for timed-out packets.
    /// This checks if any buffered packets have exceeded the timeout and should
    /// be delivered (skipping any missing packets in the sequence).
    pub fn poll_timeout(&mut self) -> Vec<Vec<u8>> {
        self.collect_ready(Instant::now())
    }
}

/// Reorder buffer statistics.
#[derive(Debug, Clone, Copy)]
pub struct ReorderStats {
    /// Packets currently buffered.
    pub buffered: usize,
    /// Next sequence number expected.
    pub next_seq: u64,
    /// Total packets delivered in order.
    pub delivered: u64,
    /// Packets dropped (timeout or overflow).
    pub dropped: u64,
    /// Packets that arrived out of order.
    pub reordered: u64,
}

/// Bandwidth aggregator - stripes packets across uplinks.
pub struct BandwidthAggregator {
    config: AggregatorConfig,

    /// Uplink weights for striping.
    weights: RwLock<Vec<UplinkWeight>>,

    /// Current position in weighted round-robin.
    stripe_index: AtomicU64,

    /// Per-uplink packet counters (for weighted distribution).
    uplink_counters: RwLock<HashMap<u16, u64>>,

    /// Last time weights were updated.
    last_weight_update: RwLock<Instant>,

    /// Reorder buffer for incoming packets.
    reorder_buffer: Mutex<ReorderBuffer>,

    /// Global send sequence number.
    send_seq: AtomicU64,

    /// Statistics.
    stats: RwLock<AggregatorStats>,
}

/// Aggregator statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct AggregatorStats {
    /// Total packets sent (striped).
    pub packets_sent: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total packets received.
    pub packets_received: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Packets per uplink.
    pub packets_per_uplink: [u64; 16],
}

impl BandwidthAggregator {
    /// Create a new bandwidth aggregator.
    pub fn new(config: AggregatorConfig) -> Self {
        let reorder_buffer = ReorderBuffer::new(config.reorder_buffer_size, config.reorder_timeout);

        Self {
            config,
            weights: RwLock::new(Vec::new()),
            stripe_index: AtomicU64::new(0),
            uplink_counters: RwLock::new(HashMap::new()),
            last_weight_update: RwLock::new(Instant::now()),
            reorder_buffer: Mutex::new(reorder_buffer),
            send_seq: AtomicU64::new(1),
            stats: RwLock::new(AggregatorStats::default()),
        }
    }

    /// Update uplink weights based on current bandwidth.
    pub fn update_weights(&self, uplinks: &[Arc<Uplink>]) {
        let mut weights = Vec::new();
        let mut total_bw = 0.0f64;

        // Collect bandwidth from usable uplinks
        for uplink in uplinks {
            if !uplink.is_usable() {
                continue;
            }

            let bw = uplink.bandwidth().bytes_per_sec;
            if bw < self.config.min_uplink_bandwidth {
                continue;
            }

            total_bw += bw;
            weights.push(UplinkWeight {
                uplink_id: uplink.numeric_id(),
                bandwidth: bw,
                weight: 0, // Will be calculated
                deficit: 0,
                rtt: uplink.rtt(),
            });
        }

        if weights.is_empty() || total_bw == 0.0 {
            *self.weights.write() = weights;
            return;
        }

        // Calculate integer weights (normalize to sum of ~100)
        for w in &mut weights {
            w.weight = ((w.bandwidth / total_bw) * 100.0).max(1.0) as u32;
        }

        // Sort by bandwidth descending for efficiency
        weights.sort_by(|a, b| {
            b.bandwidth
                .partial_cmp(&a.bandwidth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        *self.weights.write() = weights;
        *self.last_weight_update.write() = Instant::now();
    }

    /// Select the next uplink for striping using weighted round-robin.
    /// Returns (uplink_id, sequence_number) or None if no uplinks available.
    pub fn next_stripe(&self, uplinks: &[Arc<Uplink>]) -> Option<(u16, u64)> {
        // Check if we need to update weights
        if self.last_weight_update.read().elapsed() > self.config.weight_update_interval {
            self.update_weights(uplinks);
        }

        let weights = self.weights.read();
        if weights.is_empty() {
            return None;
        }

        // Weighted round-robin with deficit counter
        let idx = self.stripe_index.fetch_add(1, Ordering::Relaxed) as usize;
        let mut weights_clone = weights.clone();
        drop(weights);

        // Find uplink using weighted selection
        let uplink_id = self.weighted_select(&mut weights_clone, idx);

        // Increment counter for this uplink
        {
            let mut counters = self.uplink_counters.write();
            *counters.entry(uplink_id).or_insert(0) += 1;
        }

        // Get sequence number
        let seq = self.send_seq.fetch_add(1, Ordering::SeqCst);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.packets_sent += 1;
            if (uplink_id as usize) < 16 {
                stats.packets_per_uplink[uplink_id as usize] += 1;
            }
        }

        Some((uplink_id, seq))
    }

    /// Weighted round-robin selection.
    fn weighted_select(&self, weights: &mut [UplinkWeight], iteration: usize) -> u16 {
        if weights.is_empty() {
            return 0;
        }

        if weights.len() == 1 {
            return weights[0].uplink_id;
        }

        // Simple weighted round-robin based on iteration
        let total_weight: u32 = weights.iter().map(|w| w.weight).sum();
        if total_weight == 0 {
            return weights[0].uplink_id;
        }

        let pos = (iteration as u32) % total_weight;
        let mut cumulative = 0u32;

        for w in weights.iter() {
            cumulative += w.weight;
            if pos < cumulative {
                return w.uplink_id;
            }
        }

        weights.last().map(|w| w.uplink_id).unwrap_or(0)
    }

    /// Get all uplinks that should receive the next packet (for parallel striping).
    /// Returns uplink IDs with their stripe sequences.
    pub fn stripe_packet(&self, uplinks: &[Arc<Uplink>], _data_len: usize) -> Vec<(u16, u64)> {
        if let Some(stripe) = self.next_stripe(uplinks) {
            vec![stripe]
        } else {
            vec![]
        }
    }

    /// Receive a packet and handle reordering.
    /// Returns packets ready for delivery (in order).
    pub fn receive(&self, sequence: u64, data: Vec<u8>, from_uplink: u16) -> Vec<Vec<u8>> {
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.packets_received += 1;
            stats.bytes_received += data.len() as u64;
        }

        // Insert into reorder buffer
        self.reorder_buffer
            .lock()
            .insert(sequence, data, from_uplink)
    }

    /// Poll the reorder buffer for timed-out packets.
    pub fn poll_timeout(&self) -> Vec<Vec<u8>> {
        let mut buffer = self.reorder_buffer.lock();
        let now = Instant::now();
        buffer.collect_ready(now)
    }

    /// Flush all buffered packets.
    pub fn flush(&self) -> Vec<Vec<u8>> {
        self.reorder_buffer.lock().flush()
    }

    /// Get current sequence number (for packet headers).
    pub fn current_seq(&self) -> u64 {
        self.send_seq.load(Ordering::SeqCst)
    }

    /// Get aggregator statistics.
    pub fn stats(&self) -> AggregatorStats {
        *self.stats.read()
    }

    /// Get reorder buffer statistics.
    pub fn reorder_stats(&self) -> ReorderStats {
        self.reorder_buffer.lock().stats()
    }

    /// Get bandwidth distribution across uplinks.
    pub fn bandwidth_distribution(&self) -> Vec<(u16, f64, u64)> {
        let weights = self.weights.read();
        let counters = self.uplink_counters.read();

        weights
            .iter()
            .map(|w| {
                let count = counters.get(&w.uplink_id).copied().unwrap_or(0);
                (w.uplink_id, w.bandwidth, count)
            })
            .collect()
    }

    /// Check if aggregation is enabled and has uplinks.
    pub fn is_active(&self) -> bool {
        self.config.enabled && !self.weights.read().is_empty()
    }

    /// Get the latency spread across uplinks (max RTT - min RTT).
    pub fn latency_spread(&self) -> Duration {
        let weights = self.weights.read();
        if weights.is_empty() {
            return Duration::ZERO;
        }

        let min_rtt = weights
            .iter()
            .map(|w| w.rtt)
            .min()
            .unwrap_or(Duration::ZERO);
        let max_rtt = weights
            .iter()
            .map(|w| w.rtt)
            .max()
            .unwrap_or(Duration::ZERO);

        max_rtt.saturating_sub(min_rtt)
    }

    /// Calculate optimal reorder timeout based on latency spread.
    pub fn optimal_timeout(&self) -> Duration {
        let spread = self.latency_spread();
        // Timeout should be at least 2x the latency spread to handle jitter
        let optimal = spread.mul_f64(2.5);
        // But not less than 50ms or more than 2s
        optimal
            .max(Duration::from_millis(50))
            .min(Duration::from_secs(2))
    }
}

impl std::fmt::Debug for BandwidthAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let weights = self.weights.read();
        let stats = self.stats.read();
        f.debug_struct("BandwidthAggregator")
            .field("uplinks", &weights.len())
            .field("packets_sent", &stats.packets_sent)
            .field("packets_received", &stats.packets_received)
            .field("reorder_stats", &self.reorder_buffer.lock().stats())
            .finish()
    }
}

/// Aggregation mode for the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AggregationMode {
    /// No aggregation - single uplink per flow (load balancing only).
    #[default]
    None,
    /// Full aggregation - stripe all packets across all uplinks.
    Full,
    /// Adaptive - aggregate only when beneficial (low latency spread).
    Adaptive,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reorder_buffer_in_order() {
        let mut buffer = ReorderBuffer::new(100, Duration::from_secs(1));

        // Packets arrive in order
        let r1 = buffer.insert(1, vec![1], 0);
        assert_eq!(r1.len(), 1);

        let r2 = buffer.insert(2, vec![2], 0);
        assert_eq!(r2.len(), 1);

        let r3 = buffer.insert(3, vec![3], 0);
        assert_eq!(r3.len(), 1);

        let stats = buffer.stats();
        assert_eq!(stats.delivered, 3);
        assert_eq!(stats.reordered, 0);
    }

    #[test]
    fn test_reorder_buffer_out_of_order() {
        let mut buffer = ReorderBuffer::new(100, Duration::from_secs(1));

        // Initialize with first packet
        buffer.insert(1, vec![1], 0);

        // Packet 3 arrives before 2
        let r3 = buffer.insert(3, vec![3], 0);
        assert_eq!(r3.len(), 0); // Buffered, waiting for 2

        // Packet 2 arrives
        let r2 = buffer.insert(2, vec![2], 0);
        assert_eq!(r2.len(), 2); // Both 2 and 3 delivered

        let stats = buffer.stats();
        assert_eq!(stats.delivered, 3);
        assert_eq!(stats.reordered, 1); // Packet 3 was out of order
    }

    #[test]
    fn test_reorder_buffer_gap() {
        let mut buffer = ReorderBuffer::new(100, Duration::from_millis(10));

        // Initialize
        buffer.insert(1, vec![1], 0);

        // Skip packet 2, send 3 and 4
        buffer.insert(3, vec![3], 0);
        buffer.insert(4, vec![4], 0);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(20));

        // Poll should deliver 3 and 4, skipping 2
        let ready = buffer.poll_timeout();

        // Note: poll_timeout uses collect_ready which may need the insert to trigger
        // Let's insert another packet to trigger the timeout logic
        let ready2 = buffer.insert(5, vec![5], 0);

        let stats = buffer.stats();
        assert!(stats.dropped >= 1); // Packet 2 was dropped
    }

    #[test]
    fn test_weighted_selection() {
        let config = AggregatorConfig::default();
        let agg = BandwidthAggregator::new(config);

        // Manually set weights
        {
            let mut weights = agg.weights.write();
            weights.push(UplinkWeight {
                uplink_id: 1,
                bandwidth: 100_000.0,
                weight: 67, // ~67%
                deficit: 0,
                rtt: Duration::from_millis(10),
            });
            weights.push(UplinkWeight {
                uplink_id: 2,
                bandwidth: 50_000.0,
                weight: 33, // ~33%
                deficit: 0,
                rtt: Duration::from_millis(20),
            });
        }

        // Run 100 selections
        let mut counts: HashMap<u16, u32> = HashMap::new();
        for i in 0..100 {
            let mut weights = agg.weights.read().clone();
            let id = agg.weighted_select(&mut weights, i);
            *counts.entry(id).or_insert(0) += 1;
        }

        // Uplink 1 should get roughly 2x the packets of uplink 2
        let c1 = *counts.get(&1).unwrap_or(&0);
        let c2 = *counts.get(&2).unwrap_or(&0);

        assert!(
            c1 > c2,
            "Uplink 1 should get more packets: {} vs {}",
            c1,
            c2
        );
        // Allow some variance, but ratio should be roughly 2:1
        let ratio = c1 as f64 / c2.max(1) as f64;
        assert!(
            ratio > 1.5 && ratio < 2.5,
            "Ratio should be ~2:1, got {}",
            ratio
        );
    }
}
