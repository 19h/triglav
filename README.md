# Triglav

Triglav carries IP traffic from multiple client uplinks through one Linux exit
node. Each uplink has an independently encrypted UDP session. With aggregation
enabled, packets from one inner connection can use multiple uplinks; the exit
forwards them through its TUN interface and the host kernel's routing/NAT.

```text
client applications
       |
    client TUN
       +---- uplink 1 ----+
       +---- uplink 2 ----+---- exit TUN ---- kernel routing/NAT ---- destination
```

SOCKS5 and HTTP CONNECT are also available. Each proxy connection uses a separate
authenticated TCP stream through the exit, bound to a selected uplink. An
established proxy stream does not migrate when its outer TCP connection fails.

## Reproducible production validation

```sh
bash validation/run.sh
```

This builds and tests the actual release binaries in a disposable Linux Docker
container, then connects them through independent network namespaces to a
separate destination. It requires a working Docker daemon; privileged operations
affect the container's network namespaces, not the host's routing configuration.

The mandatory acceptance checks cover:

- Two source-NATed IPv4 uplinks and, separately, two IPv6-only outer paths.
- Full and split IPv4/IPv6 forwarding, destination-observed exit addresses, exact
  binary payload comparisons, and matching independent SHA-256 observations.
- Single-flow striping, concurrent TCP/UDP traffic, IPv4 fragmentation, packet
  loss/reordering, path blackholes, interface-down operation and path recovery.
- A rate-capped single-uplink control versus aggregation.
- Concurrent SOCKS5 and HTTP CONNECT streams, half-close and failed destinations.
- Separate clients, access-token rejection, session isolation, packet mutation,
  replay rejection, and successful traffic after invalid records.
- Graceful shutdown, route rollback, preservation of pre-existing namespaces and
  tables, and refusal of full-tunnel mode when the exit provides only IPv4.

Artifacts include the tested source archive, executed binaries, SHA-256 manifests,
compiler/image identity, packet captures, application observations and JSON
results. Failures return nonzero. The older Rust end-to-end fixtures use a test
server and are not substitutes for this production-binary acceptance suite.

See the [validation contract](validation/README.md) for exact workload, assumptions,
falsification probes and resource bounds, and the [observed report](validation/REPORT.md)
for the identified completed run. Finite test results establish behavior in the
recorded configurations; they are not formal verification of arbitrary networks.

## Build

```sh
cargo build --locked --release --bins
cargo test --locked --all-targets -- --test-threads=1
```

The client executable is `target/release/triglav`; the standalone exit executable
is `target/release/triglav-server`. `triglav server` uses the same exit implementation.

## Linux exit configuration

The exit host needs a TUN device, kernel forwarding and egress NAT or equivalent
routing. Replace `SERVER_IP` with the server's reachable local address and `eth0`
with its actual egress interface. The example uses the default address pools.

```sh
export TRIGLAV_ACCESS_TOKEN=$(openssl rand -hex 32)
sudo sysctl -w net.ipv4.ip_forward=1 net.ipv6.conf.all.forwarding=1
sudo iptables -t nat -A POSTROUTING -s 10.0.85.0/24 -o eth0 -j MASQUERADE
sudo ip6tables -t nat -A POSTROUTING -s fd00:7472:6967::/64 -o eth0 -j MASQUERADE
sudo --preserve-env=TRIGLAV_ACCESS_TOKEN target/release/triglav-server \
  --listen SERVER_IP:7443 --key /etc/triglav/server.key --generate-key
```

The server prints a base58 `tg1...` connection key (legacy `tg1_...` keys are also accepted) containing its public key and listen
address. Distribute that connection key and the deployment access token to the
client. The access token is a separate 256-bit bearer capability. A public key
identifies the exit; it does not authorize a client. Omitting the server's access
token allows public client admission.

The server key file is created with mode `0600` on Unix. An existing key file is
loaded on subsequent starts. Both UDP and TCP port 7443 must be reachable for
TUN and proxy modes respectively. The status service defaults to loopback port
9090. Full-tunnel clients require an IPv6-capable exit; an exit started with
`--ipv4-only` supports explicit IPv4 split routing.

The service runs in the foreground. Use the host's service manager for process
supervision; daemon mode and UDP-to-TCP fallback are not implemented by this exit.
The commands below show the directly tested CLI configuration path. Configuration
file and GUI workflows are outside the production acceptance contract.

## Client configuration

Set the same `TRIGLAV_ACCESS_TOKEN` in the client environment. Replace
`CONNECTION_KEY`, `IFACE1` and `IFACE2` with the actual values.

```sh
# Split routing, with single-flow packet aggregation.
sudo --preserve-env=TRIGLAV_ACCESS_TOKEN target/release/triglav tun CONNECTION_KEY \
  --interface IFACE1 --interface IFACE2 --aggregate --strategy wrr \
  --route 198.18.0.0/24 --route 2001:db8:85::/64

# Full dual-stack routing.
sudo --preserve-env=TRIGLAV_ACCESS_TOKEN target/release/triglav tun CONNECTION_KEY \
  --interface IFACE1 --interface IFACE2 --aggregate --full-tunnel

# SOCKS5 and HTTP CONNECT. The client ports listen on loopback.
target/release/triglav connect CONNECTION_KEY \
  --interface IFACE1 --interface IFACE2 --strategy wrr \
  --socks 1080 --http-proxy 8080
```

The example destination prefixes are reserved for tests/documentation; substitute
actual destination networks. Explicit interface selection binds the underlying
sockets. On Linux this uses source-address binding and `SO_BINDTODEVICE`; the
latter requires appropriate privileges. Source-only binding without that
capability requires suitable host policy routing. The acceptance suite runs with
the privileges needed for interface binding.

The exit assigns each logical client an IPv4 address and, when enabled, an IPv6
address shared by its uplinks. The default inner MTU is 1340 bytes. Its protected
UDP packet needs 1461 bytes over outer IPv4 or 1481 bytes over outer IPv6, assuming
no additional outer encapsulation. Smaller path MTUs require separate validation.

Ctrl-C and SIGTERM stop the service and remove owned client routes. Underlay
configuration is external: for example, Linux can remove static routes when an
interface is administratively brought down. Those routes must be restored before
the path can carry traffic again.

## Protocol and operational boundaries

The record layer explicitly authenticates packet type and logical sequence,
uses an explicit nonce for reordered UDP delivery, and rejects replayed records.
The implementation inherits a custom Noise hash resolver mapping the BLAKE2s
pattern name to BLAKE3. It is not wire-compatible with standard Noise
implementations using BLAKE2s. The [protocol notes](validation/README.md#wire-and-resource-invariants)
describe the handshake, size budget and bounded state.

| Environment or behavior | Evidence boundary |
| --- | --- |
| Linux TUN, IPv4/IPv6 underlay, IPv4/IPv6 payloads | Production namespace acceptance suite |
| macOS | Build and nonprivileged tests; privileged TUN operation remains unverified |
| Windows TUN | Not implemented |
| Physical NICs, radios and independent ISPs | Not established by virtual network tests |
| NAT rebinding or changing interface addresses | Outside the validated contract |
| Failure latency | Measured per run; no sub-millisecond or interruption-free guarantee |
| Long-duration load, DNS resolver policy, arbitrary MTUs | Separate validation required |
| Proxy connection migration | Not provided; each stream remains on its selected outer TCP connection |

Legacy diagnostic scripts are described in [physical_tests](physical_tests/README.md).
They do not silently extend the production claims above.

## License

MIT
