# Triglav

A multi-path networking tool that bonds multiple network interfaces into a single encrypted tunnel. It combines WiFi, Ethernet, and cellular connections to improve reliability and throughput.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
├─────────────────────────────────────────────────────────────────┤
│                     Multiplexer / Demultiplexer                 │
├─────────────────────────────────────────────────────────────────┤
│                   Multi-Path Connection Manager                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Uplink 1 │  │ Uplink 2 │  │ Uplink 3 │  │ Uplink N │         │
│  │  (WiFi)  │  │(Cellular)│  │(Ethernet)│  │   ...    │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
├─────────────────────────────────────────────────────────────────┤
│              Quality Metrics & Prediction Engine                │
├─────────────────────────────────────────────────────────────────┤
│                    Noise NK Encryption Layer                    │
├─────────────────────────────────────────────────────────────────┤
│               Transport (UDP Fast Path / TCP Fallback)          │
└─────────────────────────────────────────────────────────────────┘
```

## The Problem

You're on a train. Your laptop has WiFi (spotty), a phone hotspot (metered), and occasionally picks up station WiFi. Traditional VPNs use one connection at a time. When your current connection degrades, your SSH session hangs, your video call freezes, and you wait for a timeout before failover kicks in.

Triglav uses all available connections simultaneously. When WiFi signal drops, traffic shifts to cellular instantly—no reconnection, no timeout, no interruption. When WiFi recovers, traffic shifts back. Your connections stay alive through tunnels, across NATs, and between networks.

## Quick Start

**Server** (on a VPS or any machine with a public IP):

```bash
triglav server --generate-key --listen 0.0.0.0:7443
# Prints: tg1_<base64-encoded-key>
```

**Client** (on your laptop):

```bash
triglav connect tg1_<key> --socks 1080 --auto-discover
```

Point your browser's SOCKS5 proxy to `localhost:1080`. All traffic now flows through Triglav, distributed across all your network interfaces.

## How It Works

### Uplink Management

Each network interface becomes an "uplink." Triglav monitors each uplink independently:

- **RTT**: Measured via ping/pong, using Jacobson/Karels algorithm for smoothing
- **Packet Loss**: Sliding window of last N packets, tracking send/ack ratio
- **Bandwidth**: Exponential moving average (α=0.2) of observed throughput
- **NAT Type**: Dublin Traceroute-style detection (Full Cone, Restricted, Symmetric, etc.)

Health state machine:

```
Unknown ──[success]──> Healthy
Healthy ──[3 failures]──> Degraded
Degraded ──[3 more failures]──> Unhealthy
Unhealthy ──[4 more failures]──> Down
Any state ──[success]──> Healthy

Additionally:
- Loss >10% OR RTT >1s → Degraded
- Loss >30% OR RTT >5s → Unhealthy
```

### Scheduling Strategies

The scheduler decides which uplink(s) to use for each packet:

| Strategy | Description |
|----------|-------------|
| `adaptive` (default) | Multi-factor scoring: RTT (35%) + Loss (35%) + Bandwidth (20%) + NAT penalty (10%) |
| `lowest_latency` | Always pick the uplink with lowest RTT |
| `lowest_loss` | Always pick the uplink with lowest packet loss |
| `weighted_round_robin` | Distribute by configured weights |
| `bandwidth_proportional` | Route proportional to available bandwidth |
| `redundant` | Send on ALL uplinks (for critical traffic) |
| `primary_backup` | Use primary; failover to secondary on failure |
| `ecmp_aware` | Flow-sticky hashing, mimics ECMP router behavior |

Path stickiness ensures packets from the same flow stay on the same uplink (configurable timeout, default 5s). This prevents out-of-order delivery for TCP streams while still allowing failover when an uplink degrades.

### Encryption

All traffic is encrypted using the Noise NK protocol:

```
Pattern: Noise_NK_25519_ChaChaPoly_BLAKE3

Client                              Server
  |                                    |
  |──── e, es ────────────────────────>|  (client ephemeral, DH with server static)
  |<─── e, ee ─────────────────────────|  (server ephemeral, DH with client ephemeral)
  |                                    |
  |<═══════ encrypted channel ════════>|
```

Each uplink maintains its own Noise session. If one uplink is compromised, others remain secure. The server's public key is embedded in the connection key (`tg1_...`), enabling trust-on-first-use without a PKI.

### The AuthKey Format

The `tg1_<base64url>` key encodes:

```
┌──────────────────────────────────────────────────────────────┐
│ Server Public Key (32 bytes)                                 │
├──────────────────────────────────────────────────────────────┤
│ Address 1: [type:1][ip:4|16][port:2]                         │
│ Address 2: [type:1][ip:4|16][port:2]                         │
│ ...                                                          │
└──────────────────────────────────────────────────────────────┘
type: 4 = IPv4, 6 = IPv6
```

This single string contains everything a client needs: the server's identity (public key) and how to reach it (addresses). Share it via QR code, messaging, or anywhere you'd share a URL.

## Installation

```bash
# From source
git clone https://github.com/triglav/triglav
cd triglav
cargo build --release
./target/release/triglav --help
```

### Feature Flags

```bash
cargo build --release                    # Default: CLI + metrics + server
cargo build --release --no-default-features --features cli  # Client only
cargo build --release --features full    # Everything
```

## Configuration

Triglav works with CLI flags, config files, or a mix of both. Config files are TOML.

### Server Configuration

```toml
[server]
listen = ["0.0.0.0:7443", "[::]:7443"]
key_file = "/etc/triglav/server.key"
max_connections = 10000
idle_timeout = "5m"
tcp_fallback = true

[transport]
send_buffer_size = 2097152  # 2 MB
recv_buffer_size = 2097152
connect_timeout = "10s"
tcp_nodelay = true
tcp_keepalive = "30s"

[metrics]
enabled = true
listen = "127.0.0.1:9090"
```

### Client Configuration

```toml
[client]
auth_key = "tg1_..."
auto_discover = true
socks_port = 1080
http_proxy_port = 8080
reconnect_delay = "1s"
max_reconnect_delay = "30s"

# Specific interfaces (if not auto-discovering)
uplinks = ["en0", "en1", "pdp_ip0"]

[multipath]
max_uplinks = 16
retry_delay = "100ms"
max_retries = 3

[multipath.scheduler]
strategy = "adaptive"
rtt_weight = 0.35
loss_weight = 0.35
bandwidth_weight = 0.2
nat_penalty_weight = 0.1
sticky_paths = true
sticky_timeout = "5s"
probe_backup_paths = true
probe_interval = "1s"
```

## CLI Reference

```bash
# Server
triglav server --listen 0.0.0.0:7443 --key /path/to/key
triglav server --generate-key --print-key    # Generate and display key
triglav server --daemon --pid-file /var/run/triglav.pid

# Client
triglav connect <key> --auto-discover --socks 1080
triglav connect <key> -i en0 -i en1 --strategy latency
triglav connect <key> --http-proxy 8080 --foreground --verbose

# Key management
triglav keygen --output server.key --address 1.2.3.4:7443

# Operations
triglav status --watch --interval 1
triglav status --detailed --json
triglav uplink list
triglav uplink show en0
triglav diagnose --full --connectivity
triglav benchmark <key> --duration 30 --streams 8

# Utilities
triglav config --server --output server.toml
triglav completions bash > /etc/bash_completion.d/triglav
```

## Proxy Modes

### SOCKS5

Standard SOCKS5 proxy, compatible with browsers, curl, SSH, etc.

```bash
triglav connect <key> --socks 1080

# Usage
curl --socks5 localhost:1080 https://example.com
ssh -o ProxyCommand="nc -x localhost:1080 %h %p" user@remote
```

### HTTP Proxy

HTTP CONNECT proxy for HTTPS traffic.

```bash
triglav connect <key> --http-proxy 8080

# Usage
curl --proxy http://localhost:8080 https://example.com
export https_proxy=http://localhost:8080
```

## Metrics & Monitoring

Triglav exposes Prometheus metrics at `/metrics`:

```
# Uplink health
triglav_uplink_health{uplink="en0"} 1
triglav_uplink_rtt_seconds{uplink="en0"} 0.023
triglav_uplink_loss_ratio{uplink="en0"} 0.001

# Traffic
triglav_bytes_sent_total{uplink="en0"} 1234567
triglav_bytes_received_total{uplink="en0"} 7654321
triglav_packets_sent_total{uplink="en0"} 12345
triglav_packets_retransmitted_total{uplink="en0"} 12

# Connection
triglav_connection_state 1
triglav_active_uplinks 3
```

The event system broadcasts state changes internally:

```rust
manager.subscribe().await;
// Events: UplinkConnected, UplinkDisconnected, UplinkHealthChanged,
//         PacketReceived, SendFailed, AllUplinksDown, Connected,
//         Disconnected, PathDiscoveryComplete
```

## Protocol Details

### Packet Format

60-byte header + payload:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|    Version    |     Type      |            Flags              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Sequence Number                         |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Timestamp                             |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                        Session ID (32 bytes)                  |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Uplink ID            |        Payload Length         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                          Checksum                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Payload...                            |
```

**Packet Types**: Data, Control, Ack, Nack, Ping, Pong, Handshake, Close, Error

**Flags**: `NEED_ACK`, `RETRANSMIT`, `FRAGMENT`, `LAST_FRAGMENT`, `ENCRYPTED`, `COMPRESSED`, `PRIORITY`, `PROBE`

Maximum payload: 65,475 bytes (can be fragmented). Default MTU: 1,500 bytes (1,436 bytes payload after headers and encryption overhead).

### Deduplication

Packets may arrive on multiple uplinks (especially with `redundant` strategy). A sliding window deduplicator (O(1) lookup via HashSet, O(1) eviction via VecDeque) tracks the last 1,000 sequence numbers:

```rust
fn is_duplicate(&mut self, seq: u64) -> bool {
    if self.seen.contains(&seq) {
        return true;
    }
    self.seen.insert(seq);
    self.order.push_back(seq);
    if self.order.len() > 1000 {
        if let Some(old) = self.order.pop_front() {
            self.seen.remove(&old);
        }
    }
    false
}
```

### Flow Binding

For ECMP-style scheduling, flows are bound to uplinks:

```rust
let flow_id = hash(src_ip, dst_ip, src_port, dst_port, protocol);
if let Some(uplink) = flow_bindings.get(&flow_id) {
    if uplink.is_usable() {
        return uplink;
    }
}
let new_uplink = scheduler.select(&usable_uplinks, Some(flow_id));
flow_bindings.insert(flow_id, new_uplink);
```

Stale bindings (uplinks that went down) are garbage-collected periodically.

## Testing & Simulation

### Unit and Integration Tests

```bash
cargo test                           # All tests
cargo test --test integration        # Integration tests only
cargo test --test scheduler_strategies
cargo test --test security_edge_cases
cargo test --test stress_tests
```

### Network Impairment Simulation

The `simulation/` directory contains a Monte Carlo simulation framework:

```bash
cd simulation
cargo run --release
```

This runs 100,000+ iterations across 50+ network scenarios:

- **High-speed rail**: ICE, TGV, Shinkansen handoffs
- **Stationary**: Home office, cafe, data center
- **Infrastructure failures**: Fiber cuts, power outages, BGP flaps
- **Mobile**: Cross-border roaming, highway driving
- **Urban**: Walking, cycling, public transit
- **Stress**: Flash crowds, DDoS, simultaneous uplink failure

Algorithms tested include classical (threshold, EWMA, Kalman filter), ML (Q-learning, UCB1 bandit), and neural (LSTM, Transformer, TCN).

### Docker Test Environment

```bash
cd docker/testnet
docker-compose up
```

Spins up:
- Triglav server
- Multiple clients with simulated network conditions
- Routers with configurable latency/loss/jitter:
  - WiFi: 20ms latency, 5ms jitter, 1% loss
  - Ethernet: 2ms latency, 0.5ms jitter, 0.01% loss
  - LTE: 50ms latency, 20ms jitter, 2-5% loss
- Chaos agent for dynamic impairment injection
- Prometheus + Grafana for monitoring

### Physical Tests

For testing on real hardware:

```bash
cd physical_tests
./run_physical_tests.sh
```

Requires actual multiple network interfaces. Injects real impairments via `tc` (traffic control).

## Design Notes

### Why Per-Uplink Noise Sessions?

Each uplink has its own cryptographic session. If an attacker compromises one path (e.g., WiFi MITM), they cannot decrypt traffic on other paths. This is defense in depth: even with a compromised uplink, traffic that went through other uplinks remains confidential.

### Why DashMap?

Uplinks are accessed concurrently from multiple async tasks (receive loops, send paths, health checks). `DashMap` provides better performance than `Mutex<HashMap>` under contention by sharding the map internally. Each shard has its own lock, so operations on different uplinks don't contend.

### Why EMA for Bandwidth?

Bandwidth estimates need to be responsive but not jumpy. EMA with α=0.2 means:
- New sample contributes 20% to the estimate
- Old estimate contributes 80%
- Effective window of ~5 samples
- Responds to changes within seconds, but smooths out single-packet variations

### Why Flow Stickiness?

TCP expects packets in order. If packets for a single TCP connection take different paths with different latencies, they arrive out of order. The receiver buffers and reorders, but this adds latency and can confuse congestion control. Flow stickiness keeps a TCP connection on one uplink unless that uplink fails. The 5-second timeout allows gradual rebalancing when uplink quality changes.

## Constants

```rust
PROTOCOL_VERSION: 1
MAX_MTU: 1500
MAX_PAYLOAD: 1436        // After headers + crypto overhead
DEFAULT_PORT: 7443
HEADER_SIZE: 60
MAX_PAYLOAD_SIZE: 65475  // For fragmented packets
EMA_ALPHA: 0.2           // Bandwidth smoothing
DEFAULT_UPLINK_TIMEOUT: 30s
DEDUP_WINDOW_SIZE: 1000
```

## Platform Support

| Platform | Interface Discovery | Notes |
|----------|-------------------|-------|
| Linux | rtnetlink | Full support, including cellular |
| macOS | system-configuration | Full support |
| Windows | - | Not yet implemented |
| BSD | - | Should work, untested |

## License

MIT
