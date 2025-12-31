# Triglav

A multi-path VPN that bonds multiple network interfaces into a single encrypted tunnel. It creates a virtual network interface (TUN) that transparently captures all IP traffic and distributes it across WiFi, Ethernet, and cellular connections to improve reliability and throughput.

## The Problem

You're on a train. Your laptop has WiFi (spotty), a phone hotspot (metered), and occasionally picks up station WiFi. Traditional VPNs use one connection at a time. When your current connection degrades, your SSH session hangs, your video call freezes, and you wait for a timeout before failover kicks in.

Triglav uses all available connections simultaneously. When WiFi signal drops, traffic shifts to cellular instantly—no reconnection, no timeout, no interruption. When WiFi recovers, traffic shifts back. Your connections stay alive through tunnels, across NATs, and between networks.

## Key Features

- **True VPN**: Virtual TUN interface captures all IP traffic transparently
- **Multi-path**: Aggregate bandwidth across WiFi, cellular, ethernet simultaneously
- **ECMP-aware**: Flow-based routing maintains TCP connection consistency
- **Encrypted**: Noise NK protocol with per-uplink cryptographic sessions
- **NAT traversal**: Works behind NATs with Dublin Traceroute-style probing
- **Cross-platform**: Linux, macOS (Windows in progress)
- **Zero config**: Auto-discovers network interfaces
- **Split tunneling**: Route specific networks through tunnel, bypass others

## Quick Start

**Server** (on a VPS or any machine with a public IP):

```bash
triglav server --generate-key --listen 0.0.0.0:7443
# Prints: tg1_<base64-encoded-key>
```

**Client** (on your laptop):

```bash
# Full VPN mode (recommended) - routes ALL traffic through tunnel
sudo triglav tun tg1_<key> --full-tunnel --auto-discover

# Split tunnel - only route specific networks
sudo triglav tun tg1_<key> --route 10.0.0.0/8 --route 192.168.0.0/16

# Legacy proxy mode (no root required)
triglav connect tg1_<key> --socks 1080 --auto-discover
```

With TUN mode, all applications automatically use the tunnel—no proxy configuration needed. Your browser, SSH, curl, games, everything just works.

## Installation

```bash
# From source
git clone https://github.com/triglav/triglav
cd triglav
cargo build --release
sudo ./target/release/triglav tun --help
```

### Privileges

TUN mode requires elevated privileges to create virtual network interfaces:

```bash
# Linux: Either run as root or grant capabilities
sudo setcap cap_net_admin+ep ./target/release/triglav

# macOS: Must run as root
sudo triglav tun ...
```

### Feature Flags

```bash
cargo build --release                    # Default: CLI + metrics + server
cargo build --release --no-default-features --features cli  # Client only
cargo build --release --features full    # Everything
```

## How It Works

### TUN Virtual Interface

Triglav creates a TUN device (e.g., `tg0` on Linux, `utun3` on macOS) that operates at Layer 3 (IP). The kernel routes packets to this interface, Triglav reads them, encrypts and sends them through one or more physical uplinks, and the server decrypts and forwards them to their destination.

```
Application → Kernel → TUN Device → Triglav → [WiFi|LTE|Ethernet] → Server → Internet
                                         ↓
                              NAT + Encrypt + Schedule
```

Unlike proxy-based VPNs, this captures **all** IP traffic: TCP, UDP, ICMP, DNS, everything. No application configuration required.

### NAT Translation

Triglav performs NAT between your local network and the tunnel:

- **Outbound**: Local IP (192.168.1.x) → Tunnel IP (10.0.85.1)
- **Inbound**: Tunnel IP → Original local IP

This allows proper routing of return traffic and supports multiple simultaneous connections with port translation.

### Flow-Based Routing

To maintain TCP connection consistency, Triglav hashes the 5-tuple (src IP, dst IP, src port, dst port, protocol) and binds each flow to a specific uplink:

```rust
let flow_id = hash(src_ip, dst_ip, src_port, dst_port, protocol);
// Packets with same flow_id always use the same uplink (unless it fails)
```

This prevents packet reordering that would confuse TCP congestion control.

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

### Protocol


```
┌─────────────────────────────────────────────────────────────────┐
│                    Applications (Any Protocol)                  │
│              (TCP, UDP, ICMP, DNS - all traffic)                │
├─────────────────────────────────────────────────────────────────┤
│                      Kernel TCP/IP Stack                        │
├─────────────────────────────────────────────────────────────────┤
│                    TUN Virtual Interface                        │
│                 (utun/tun0 - Layer 3 IP packets)                │
├─────────────────────────────────────────────────────────────────┤
│                        TunnelRunner                             │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  IP Parser   │──│    NAT      │──│  MultipathManager       │ │
│  │  (5-tuple)   │  │ Translation │  │  (encryption, routing)  │ │
│  └──────────────┘  └─────────────┘  └─────────────────────────┘ │
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

## CLI Reference

### TUN Mode (Recommended)

```bash
# Full tunnel - route all traffic
sudo triglav tun <key> --full-tunnel

# Split tunnel - specific routes only
sudo triglav tun <key> --route 10.0.0.0/8 --route 172.16.0.0/12

# Exclude specific networks (bypass tunnel)
sudo triglav tun <key> --full-tunnel --exclude 192.168.1.0/24

# Custom TUN device name and IP
sudo triglav tun <key> --tun-name tg0 --ipv4 10.0.85.1

# Use specific interfaces
sudo triglav tun <key> -i en0 -i en1 --full-tunnel

# Enable DNS through tunnel
sudo triglav tun <key> --full-tunnel --dns

# Scheduling strategy
sudo triglav tun <key> --full-tunnel --strategy latency
```

### Legacy Proxy Mode

```bash
# SOCKS5 proxy (no root required)
triglav connect <key> --socks 1080 --auto-discover

# HTTP proxy
triglav connect <key> --http-proxy 8080

# Both proxies
triglav connect <key> --socks 1080 --http-proxy 8080
```

### Server

```bash
triglav server --listen 0.0.0.0:7443 --key /path/to/key
triglav server --generate-key --print-key    # Generate and display key
triglav server --daemon --pid-file /var/run/triglav.pid
```

### Operations

```bash
triglav status --watch --interval 1
triglav status --detailed --json
triglav uplink list
triglav uplink show en0
triglav diagnose --full --connectivity
triglav benchmark <key> --duration 30 --streams 8
triglav keygen --output server.key --address 1.2.3.4:7443
triglav config --server --output server.toml
triglav completions bash > /etc/bash_completion.d/triglav
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

### Client Configuration (TUN Mode)

```toml
[client]
auth_key = "tg1_..."
auto_discover = true

[tun]
name = "tg0"
ipv4_addr = "10.0.85.1"
ipv4_netmask = 24
mtu = 1420

[routing]
full_tunnel = true
exclude_routes = ["192.168.1.0/24"]  # Local network bypass
dns_servers = ["1.1.1.1", "8.8.8.8"]

[nat]
tunnel_ipv4 = "10.0.85.1"
udp_timeout = "3m"
tcp_timeout = "2h"
port_range_start = 32768
port_range_end = 61000

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
```

### Client Configuration (Proxy Mode)

```toml
[client]
auth_key = "tg1_..."
auto_discover = true
socks_port = 1080
http_proxy_port = 8080

# Specific interfaces (if not auto-discovering)
uplinks = ["en0", "en1", "pdp_ip0"]
```

## The AuthKey Format

The `tg1_<base64url>` key encodes:

```
┌──────────────────────────────────────────────────────────────┐
│ Server Public Key (32 bytes)                                 │
├──────────────────────────────────────────────────────────────┤
│ Address 1: [type:1][ip:4|16][port:2]                         │
│ Address 2: [type:1][ip:4|16][port:2]                         │
│ ...                                                          │
└──────────────────────────────────────────────────────────────┘
type: 1 = IPv4, 2 = IPv6
```

This single string contains everything a client needs: the server's identity (public key) and how to reach it (addresses). Share it via QR code, messaging, or anywhere you'd share a URL.

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

# TUN statistics
triglav_tun_packets_read_total 123456
triglav_tun_packets_written_total 123400
triglav_nat_active_entries 42

# Connection
triglav_connection_state 1
triglav_active_uplinks 3
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

Packets may arrive on multiple uplinks (especially with `redundant` strategy). A sliding window deduplicator (O(1) lookup via HashSet, O(1) eviction via VecDeque) tracks the last 1,000 sequence numbers.

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

### Docker Test Environment

```bash
cd docker/testnet
docker-compose up
```

Spins up:
- Triglav server
- Multiple clients with simulated network conditions
- Routers with configurable latency/loss/jitter
- Chaos agent for dynamic impairment injection
- Prometheus + Grafana for monitoring

## Design Notes

### Why TUN Instead of Proxy?

Proxies (SOCKS5, HTTP) require application-level configuration. Each application must be told to use the proxy, and some applications don't support proxies at all. TUN operates at the kernel level, capturing all IP traffic transparently. Every application—browsers, games, SSH, custom protocols—automatically uses the tunnel.

### Why Per-Uplink Noise Sessions?

Each uplink has its own cryptographic session. If an attacker compromises one path (e.g., WiFi MITM), they cannot decrypt traffic on other paths. This is defense in depth: even with a compromised uplink, traffic that went through other uplinks remains confidential.

### Why Flow Stickiness?

TCP expects packets in order. If packets for a single TCP connection take different paths with different latencies, they arrive out of order. The receiver buffers and reorders, but this adds latency and can confuse congestion control. Flow stickiness keeps a TCP connection on one uplink unless that uplink fails.

### Why NAT in the Client?

The TUN interface has its own IP (e.g., 10.0.85.1). Applications send packets to real destinations (e.g., 8.8.8.8), but these packets arrive at the TUN with the local machine's IP as source. NAT translates this to the tunnel IP, ensuring proper routing of return traffic.

## Constants

```rust
PROTOCOL_VERSION: 1
MAX_MTU: 1500
MAX_PAYLOAD: 1436        // After headers + crypto overhead
DEFAULT_PORT: 7443
HEADER_SIZE: 60
MAX_PAYLOAD_SIZE: 65475  // For fragmented packets
EMA_ALPHA: 0.2           // Bandwidth smoothing
DEFAULT_TUN_MTU: 1420    // TUN interface MTU
DEFAULT_TUNNEL_IPV4: "10.0.85.1"
```

## Platform Support

| Platform | TUN Support | Interface Discovery | Notes |
|----------|-------------|-------------------|-------|
| Linux | Yes (`/dev/net/tun`) | rtnetlink | Full support |
| macOS | Yes (`utun`) | system-configuration | Full support |
| Windows | Planned (WinTUN) | - | In progress |
| BSD | Untested | - | Should work |

## Comparison

| Feature | Triglav | WireGuard | OpenVPN | Tailscale |
|---------|---------|-----------|---------|-----------|
| Multi-path | Yes | No | No | No |
| Bandwidth aggregation | Yes | No | No | No |
| Seamless failover | Yes | Manual | Manual | Partial |
| TUN interface | Yes | Yes | Yes | Yes |
| Proxy mode | Yes (fallback) | No | No | No |
| Per-uplink encryption | Yes | No | No | No |
| ECMP-aware routing | Yes | No | No | No |

## License

MIT
