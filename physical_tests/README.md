# Triglav Physical Testing Framework

This directory contains tools for **physically testing** triglav's multi-path networking capabilities using real network interfaces, actual traffic, and simulated network conditions.

## Overview

Unlike the simulated tests in `tests/`, these physical tests:

- Use **real network interfaces** (WiFi, Ethernet, cellular, etc.)
- Test **actual network conditions** (latency, packet loss, failover)
- Validate **real traffic flow** through the SOCKS5/HTTP proxy
- Measure **actual performance** (throughput, latency)

## Quick Start

```bash
# Run basic connectivity tests
./run_physical_tests.sh --quick

# Run all tests
./run_physical_tests.sh --all

# Run end-to-end validation with real traffic
./e2e_validation.sh --local
```

## Test Categories

### 1. Interface Discovery Tests
Verify that multiple network interfaces are available and can be used independently.

```bash
cargo test --test physical_multipath test_interface_discovery -- --ignored --nocapture
```

### 2. Multi-Interface Connectivity Tests
Send traffic from multiple local IPs to verify path diversity.

```bash
cargo test --test physical_multipath test_multi_interface_connectivity -- --ignored --nocapture
```

### 3. Bandwidth Measurement
Measure raw throughput capabilities per interface.

```bash
cargo test --test physical_multipath test_bandwidth_measurement -- --ignored --nocapture
```

### 4. Latency Distribution
Measure latency characteristics of each interface.

```bash
cargo test --test physical_multipath test_latency_distribution -- --ignored --nocapture
```

### 5. Concurrent Traffic
Test simultaneous traffic on multiple interfaces.

```bash
cargo test --test physical_multipath test_concurrent_traffic -- --ignored --nocapture
```

### 6. External Connectivity
Test connectivity to real external endpoints.

```bash
cargo test --test physical_multipath test_external_connectivity -- --ignored --nocapture
```

### 7. Manual Failover Test
Interactive test requiring manual interface disconnection.

```bash
cargo test --test physical_multipath test_manual_failover -- --ignored --nocapture
```

## Network Impairment Testing

Test triglav's behavior under degraded network conditions using `network_impairment.sh`.

### Requirements
- macOS with `dnctl` and `pfctl` (built-in)
- Root/sudo privileges

### Usage

```bash
# List available interfaces
sudo ./network_impairment.sh list-interfaces

# Add 100ms latency to WiFi
sudo ./network_impairment.sh setup en0 100 0

# Add 100ms latency + 5% packet loss
sudo ./network_impairment.sh setup en0 100 5

# Add bandwidth limit (1 Mbps)
sudo ./network_impairment.sh setup en0 0 0 1000

# Simulate poor mobile connection (200ms, 10% loss)
sudo ./network_impairment.sh setup en0 200 10

# Remove all impairment
sudo ./network_impairment.sh teardown

# Check current status
sudo ./network_impairment.sh status
```

### Run Tests with Impairment

```bash
# Run impairment test suite (requires sudo)
sudo ./run_physical_tests.sh --impairment
```

## End-to-End Validation

The `e2e_validation.sh` script performs comprehensive testing with real server/client instances.

### Full Local Test

Starts server, connects client, and runs validation tests:

```bash
./e2e_validation.sh --local
```

This tests:
- SOCKS5 proxy connectivity
- Data transfer integrity
- Concurrent connections
- Bandwidth measurement
- Latency measurement

### Server-Only Mode

For testing across machines, start the server on one machine:

```bash
# On server machine (e.g., VPS)
./e2e_validation.sh --server-only

# Note the auth key printed to stdout
```

### Client-Only Mode

Connect to a remote server:

```bash
# On client machine
./e2e_validation.sh --client-only "triglav://..."
```

### Custom External Target

Test against a specific endpoint:

```bash
./e2e_validation.sh --local --external example.com
```

## Test Infrastructure Requirements

### Minimum Requirements
- 1 network interface with IP address
- Rust nightly toolchain
- macOS or Linux

### Recommended for Full Testing
- 2+ network interfaces (e.g., WiFi + Ethernet)
- USB Ethernet adapter or phone tethering
- Internet connectivity for external tests

### For Network Impairment Tests
- macOS (uses `dnctl`/`pfctl`)
- Root privileges

## Physical Test Matrix

| Test | Interfaces | Network | Duration |
|------|------------|---------|----------|
| Interface Discovery | 1+ | Local | <5s |
| Multi-Interface Connectivity | 2+ | Local | ~30s |
| MultipathManager | 1+ | Local | ~10s |
| Bandwidth Measurement | 1+ | Local | ~30s |
| Latency Distribution | 1+ | Local | ~30s |
| Concurrent Traffic | 2+ | Local | ~30s |
| External Connectivity | 1+ | Internet | ~30s |
| Manual Failover | 2+ | Local | Manual |
| E2E Validation | 1+ | Internet | ~2min |
| Impairment Tests | 1+ | Local+Sudo | ~5min |

## Interpreting Results

### Successful Multi-Path Test
```
Discovered 3 interfaces:
  en0 -> 192.168.1.100
  en1 -> 192.168.2.100
  en6 -> 10.0.0.50

Test server listening on: 0.0.0.0:54321
Sending from en0 (192.168.1.100)
Sending from en1 (192.168.2.100)
Sending from en6 (10.0.0.50)

╔══════════════════════════════════════════════════════════╗
║              PHYSICAL TEST RESULTS                       ║
╠══════════════════════════════════════════════════════════╣
║ Interfaces Used: en0, en1, en6                           ║
║ Packets Sent:    30                                      ║
║ Packets Received:30                                      ║
║ Unique Paths:    3                                       ║
║ Avg Latency:     0.45 ms                                 ║
╚══════════════════════════════════════════════════════════╝

[PASS] Multi-interface connectivity verified
```

### Failover Detection
```
Sending test traffic on 2 interfaces...
..........
Interface en0 appears down, trying next...
Recovered after 127.34 ms

Failover Test Results:
  Failover detected and recovered in 127.34 ms
```

## Troubleshooting

### "Need at least 2 interfaces"
You need multiple network interfaces for multi-path tests:
- Connect USB Ethernet adapter
- Enable WiFi alongside Ethernet
- Tether phone via USB
- Use VPN (creates `utun*` interface)

### "Bind failed" errors
Another process may be using the port, or the interface IP changed:
```bash
# Check what's using ports
lsof -i :17443
lsof -i :11080

# Re-discover interfaces
./run_physical_tests.sh --list
```

### Network impairment not working
```bash
# Check if pf is enabled
sudo pfctl -s info

# Check dummynet pipes
sudo dnctl list

# Ensure cleanup ran
sudo ./network_impairment.sh teardown
```

### Client won't connect
Check server log for the auth key:
```bash
cat /tmp/triglav_server.log
```

## Files

| File | Description |
|------|-------------|
| `run_physical_tests.sh` | Main test runner script |
| `network_impairment.sh` | Network condition simulator |
| `e2e_validation.sh` | Full end-to-end test harness |
| `../tests/physical_multipath.rs` | Rust test implementations |
| `README.md` | This documentation |

## Advanced Usage

### Running Specific Tests
```bash
# Run single test
./run_physical_tests.sh bandwidth

# With verbose output
./run_physical_tests.sh --verbose latency

# Multiple test suites
./run_physical_tests.sh --quick --stress
```

### Custom Test Parameters
Edit the script variables or use environment variables:
```bash
SERVER_PORT=8443 ./e2e_validation.sh --local
TEST_DURATION=60 ./e2e_validation.sh --local
```

### Cross-Machine Testing
```bash
# Machine A (server)
./e2e_validation.sh --server-only
# Copy auth key...

# Machine B (client)
./e2e_validation.sh --client-only "triglav://..."

# Machine B - test through proxy
curl --socks5 localhost:11080 http://example.com
```
