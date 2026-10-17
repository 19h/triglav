# Triglav

Encrypted multipath networking through a single exit.

Triglav connects several client network interfaces to one exit server. In TUN
mode, it carries IPv4 and IPv6 packets through independently encrypted UDP
paths. Applications use the client's virtual interface; the exit forwards their
traffic through its own network connection.

```text
                         encrypted UDP paths
                       +------ uplink 1 ------+
applications -- TUN ---+------ uplink 2 ------+--- exit -- destinations
                       +------ uplink N ------+
```

Enable packet aggregation to distribute a single flow across multiple uplinks,
or use flow-based scheduling. Triglav also provides local SOCKS5 and HTTP CONNECT
proxies, with a separate encrypted TCP stream for each application connection.

## Capabilities

- **Dual-stack tunneling:** full-tunnel and selected-network routing for IPv4 and
  IPv6, over IPv4 or IPv6 uplinks.
- **Packet aggregation:** stripe one connection across multiple paths with
  `--aggregate`.
- **Path management:** concurrent connection establishment, interface monitoring,
  bounded reconnect backoff, authenticated NAT rebinding and confirmed rekeying.
- **Independent encryption:** separate cipher state for each uplink, authenticated
  packet sequencing and replay rejection.
- **Shared exit:** per-client tunnel addresses, session isolation and optional
  deployment access tokens.
- **Local proxies and monitoring:** SOCKS5, HTTP CONNECT, JSON status and
  Prometheus-compatible metrics.

The exit uses Linux TUN, kernel routing and conntrack/NAT. The repository also
contains a [native macOS control app](apps/macos/README.md).

## Build

From a checkout, with Rust, Cargo and a C compiler installed:

```sh
cargo build --locked --release --bin triglav --bin triglav-server
```

On Linux, install the two executables:

```sh
sudo install -m 0755 target/release/triglav target/release/triglav-server /usr/local/bin/
```

The examples below use Linux, `iproute2`, `iptables`/`ip6tables`, OpenSSL and
`sudo`. TUN creation and route changes need network administration privileges;
interface-bound sockets also need appropriate privileges. The examples run
Triglav with `sudo`.

## Start an exit

The exit needs working IPv4 and IPv6 egress. Its firewall must allow inbound
UDP port 7443 for tunnels, TCP port 7443 for proxies, and forwarding between the
`tg-exit` interface and its egress interface.

Choose an address assigned to the exit host that clients can reach, and replace
the example interface with its egress interface. `203.0.113.10` is a documentation
address, not a public service.

```sh
SERVER_IP=203.0.113.10
EGRESS_IF=eth0
export TRIGLAV_ACCESS_TOKEN="$(openssl rand -hex 32)"

sudo install -d -m 0700 /etc/triglav
sudo sysctl -w net.ipv4.ip_forward=1 net.ipv6.conf.all.forwarding=1
sudo iptables -t nat -A POSTROUTING -s 10.0.85.0/24 -o "$EGRESS_IF" -j MASQUERADE
sudo ip6tables -t nat -A POSTROUTING -s fd00:7472:6967::/64 -o "$EGRESS_IF" -j MASQUERADE

sudo --preserve-env=TRIGLAV_ACCESS_TOKEN triglav-server \
  --listen "${SERVER_IP}:7443" \
  --key /etc/triglav/server.key --generate-key
```

This example assumes the firewall already permits forwarding. Persist the
forwarding settings, firewall rules and token through the host's normal network
and service configuration if the service must survive a reboot.

The server prints a client connection key. Supply that key and the access token
to each client through your chosen provisioning mechanism.

| Credential | Purpose |
| --- | --- |
| Server key file | Persistent private identity of the exit; created with Unix mode `0600` and loaded on subsequent starts |
| Connection key, `tg1...` | Base58-encoded server public key and endpoints; legacy `tg1_...` keys remain readable |
| `TRIGLAV_ACCESS_TOKEN` | Separate 256-bit bearer capability, encoded as 64 hexadecimal digits; checked during client admission |

Keep the private key and token stable across restarts. Omitting the access token
on the exit permits public client admission; the connection key alone identifies
the server.

`triglav server` runs the same TUN exit implementation. For an IPv6 listen address,
use bracket notation, for example `--listen '[2001:db8::10]:7443'`. Processes run
in the foreground and can be supervised by the host's service manager.

## Connect a client

Each selected interface must already have an address and a route to the exit.
Triglav binds its sockets to those interfaces; it does not provision the
underlying Ethernet, Wi-Fi or cellular connections.

On the client, substitute the values received from the exit and the actual
interface names:

```sh
CONNECTION_KEY='tg1...'
export TRIGLAV_ACCESS_TOKEN='COPY_THE_64_HEX_DIGIT_TOKEN_FROM_THE_EXIT'
UPLINK1=enp3s0
UPLINK2=wlan0
```

### Full tunnel

Route IPv4 and IPv6 traffic through the exit, with packet aggregation:

```sh
sudo --preserve-env=TRIGLAV_ACCESS_TOKEN triglav tun "$CONNECTION_KEY" \
  --interface "$UPLINK1" --interface "$UPLINK2" \
  --aggregate --full-tunnel
```

The exit assigns tunnel addresses shared by the client's uplinks. Full-tunnel
startup requires an IPv6-capable exit. For an IPv4-only deployment, start the
exit with `--ipv4-only` and use explicit IPv4 routes on clients.

### Selected networks

Replace these example private networks with destinations reachable from the exit:

```sh
sudo --preserve-env=TRIGLAV_ACCESS_TOKEN triglav tun "$CONNECTION_KEY" \
  --interface "$UPLINK1" --interface "$UPLINK2" \
  --aggregate --strategy wrr \
  --route 10.20.0.0/16 --route fd20:1234::/48
```

Repeat `--route` for additional networks. Use `--exclude CIDR` for a network that
must bypass the tunnel. `--auto-discover` selects available interfaces when no
explicit interface list is supplied.

### Application proxies

Run local SOCKS5 and HTTP CONNECT listeners through the exit:

```sh
sudo --preserve-env=TRIGLAV_ACCESS_TOKEN triglav connect "$CONNECTION_KEY" \
  --interface "$UPLINK1" --interface "$UPLINK2" --strategy wrr \
  --socks 1080 --http-proxy 8080
```

In another terminal:

```sh
curl --socks5-hostname 127.0.0.1:1080 https://example.com/
curl --proxy http://127.0.0.1:8080 https://example.com/
```

Both listeners bind to loopback. Proxy mode carries application connections
without installing TUN routes.

| Mode | Distribution across uplinks |
| --- | --- |
| TUN with `--aggregate` | Packets from one flow can use multiple uplinks |
| TUN without `--aggregate` | Flow-based scheduling, with selective redundant sends for small or connection-control packets |
| SOCKS5 / HTTP CONNECT | Each connection uses one outer TCP stream; separate connections can use different uplinks |

An established proxy stream stays on its selected TCP connection. It does not
migrate to another uplink after that connection fails.

## Operate and inspect

| Service | Default status URL | Metrics URL |
| --- | --- | --- |
| Exit | `http://127.0.0.1:9090/status` | `http://127.0.0.1:9090/metrics` |
| Client | `http://127.0.0.1:9091/status` | `http://127.0.0.1:9091/metrics` |

```sh
curl --fail http://127.0.0.1:9090/status
curl --fail http://127.0.0.1:9091/status
```

Use `triglav-server --metrics ADDRESS` to change the exit's HTTP bind address,
and `--metrics-addr ADDRESS` on client commands. Use `--log-level debug` on
`triglav` for connection and routing diagnostics.

Ctrl-C and SIGTERM stop the processes and remove owned client routes. Restoring
an interface also requires restoring its underlying network configuration;
Linux may remove static routes when an interface is administratively taken down.

Clients wait for an unavailable exit and retry stale paths automatically. Recovery
negotiation preserves the tunnel addresses and receive-sequence position across
an exit restart when the same key and tunnel prefix remain configured and the
previous address is available. A conflicting lease is rejected. Restart recovery
uses the `TG3` handshake; deploy matching client and exit versions. The exit also
accepts existing `TG2` clients.

The client status response includes pending-packet count, retained payload bytes
and their configured limits. Defaults cap retransmission storage at 4096 logical
packets, 8 MiB of payload and 10 s of age; these limits do not bound all process
memory. See the [resilience acceptance contract](docs/resilience.md).

The default inner MTU is 1340 bytes. Including Triglav framing, encryption, UDP
and ordinary IP headers, a full-sized packet occupies 1461 bytes over IPv4 or
1481 bytes over IPv6. Additional encapsulation and smaller path MTUs require
separate validation. Full-tunnel routing does not configure the system DNS
resolver.

Further command options are available through `triglav tun --help`,
`triglav connect --help` and `triglav-server --help`.

## Validation

Run the production acceptance suite with Docker:

```sh
bash validation/run.sh
```

The harness builds release binaries and runs them through isolated Linux network
namespaces, including two separately NATed IPv4 paths and two IPv6-only paths.
It checks destination-observed exit identity, exact bidirectional payloads,
concurrent clients, proxies, loss and reordering, path failure and recovery,
admission rejection, replay rejection and routing cleanup.

A separate recovery lane keeps one TCP connection open through a 20 s total
outage, an exit restart, changed NAT addresses and twenty local path replacements.
It checks process/session continuity and samples pending-buffer bounds under load.

The [R1 baseline from **2026-10-17**](docs/resilience.md#r1-validation-2026-10-17)
passed 337 Rust tests on each of Linux and
macOS, with 10 tests ignored by default on each platform and a separate live
admission test. It verified 220 destination exchanges containing 17,817,030
application bytes in each direction. In three trials per mode, with each link
capped at 2 Mbit/s, aggregation achieved 1.52 times the single-uplink control's
median request goodput. These measurements describe that workload and topology.

Each run retains source and binary hashes, source and executable copies, packet
captures, destination observations and machine-readable results under
`validation/artifacts/`. Failed checks return a nonzero status. The GitHub
workflow runs the same command.

- [Validation contract, assumptions and falsification probes](validation/README.md)
- [Recorded results and evidence](validation/REPORT.md)
- [Physical-interface diagnostics](physical_tests/README.md)

## Compatibility and limits

| Area | Current status |
| --- | --- |
| Linux TUN exit and client | Validated with dual-stack payloads over IPv4 and IPv6 paths |
| macOS | Client implementation and native app available; build and nonprivileged tests pass; privileged TUN operation remains unverified |
| Windows TUN | Not implemented |
| Physical NIC, modem and ISP behavior | Outside the namespace validation results |
| NAT rebinding, changing uplink addresses and exit restart | Covered by the controlled recovery lab; restart continuity requires the same key/prefix and an available lease |
| Arbitrary MTUs and long-duration load | Outside the recorded acceptance profile |

The protocol uses a Noise NK handshake with X25519 and ChaCha20-Poly1305. Its
custom resolver maps the BLAKE2s pattern name to BLAKE3; standard Noise BLAKE2s
implementations are not wire-compatible. See the
[wire and resource invariants](validation/README.md#wire-and-resource-invariants)
for record framing, replay handling and size limits.

## License

MIT.
