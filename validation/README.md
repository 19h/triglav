# Multi-uplink, single-exit validation

This directory tests the shipped `triglav` and `triglav-server` binaries against
an independent destination. The existing `tests/e2e_multipath.rs` implements its
own forwarding server; that fixture alone cannot validate the production exit.

## Run

On a machine with Docker:

```sh
bash validation/run.sh
```

The command builds a Linux image, compiles with `Cargo.lock`, runs the Rust
suite, builds optimized release binaries, and executes the network tests in a disposable privileged container.
It never modifies the host's interfaces, routes, forwarding settings or
firewall. The container needs privileges to create its own network namespaces,
TUN interfaces and impairment queues. Missing prerequisites or failed assertions
produce a nonzero exit status. No network gate silently skips.

Outputs are under `validation/artifacts/<UTC timestamp>/`. Retain the source
archive, executed binaries, SHA-256 manifests, compiler/platform descriptions, image identity, logs, JSON
results and PCAPs together. A success applies to that source snapshot and test
configuration. It does not establish universal correctness on arbitrary
networks or constitute a formal verification of the implementation.

For an existing Linux build environment, install `iproute2`, `iptables`,
`tcpdump`, `python3`, `curl` and `iputils-ping`, then run as root:

```sh
CARGO_TARGET_DIR=/build cargo build --locked --bins
TRIGLAV_BIN_DIR=/build/debug bash validation/netns.sh
```

`TRIGLAV_EVIDENCE_DIR`, if supplied, must be a fresh directory. Namespace names
are reserved only for the duration of the test; an existing namespace causes
failure instead of deletion. The test generates a fresh 256-bit access token.
The token is inherited through the environment and is not written to artifacts.

## Topology and assertions

```text
                         router 1 / source NAT
                     +-- 10.73.1.1 -------------+
client up1/up2 -------+                          +-- exit -- destination
                     +-- 10.74.1.1 -------------+   TUN     198.18.0.2
                         router 2 / source NAT       |
                                               198.18.0.1
                                               sole egress
```

The client has an ordinary default gateway but no working path to the destination before Triglav starts. Both uplinks
reach the same server address, through separate veth links, routers and source
NATs. The destination has neither a client-facing network nor a route back to
client uplinks. Linux forwards IP packets between the server TUN and the
single egress interface; conntrack applies exit NAT.

| Gate | Required evidence |
| --- | --- |
| Build and regressions | Locked build and successful Rust test process |
| Path separation | Server sees both distinct NAT source addresses; PCAPs contain encrypted data on both paths |
| Single exit | Each destination peer is the exit address for its family: `198.18.0.1` or `2001:db8:85::1` |
| TCP integrity (IPv4 and IPv6) | Exact bytes and SHA-256 for a single transfer and eight concurrent bidirectional transfers |
| UDP and fragmentation | Exact IPv4/IPv6 UDP echo and fragmented 4096-byte IPv4 datagrams |
| Connection continuity | One TCP socket completes 120 frames while each path is independently blackholed for 2 s |
| Impairment | Exact TCP transfer under asymmetric delay, jitter, loss and reordering |
| Negative controls | Replayed, corrupted, forged-sequence and plaintext records are rejected; valid traffic subsequently succeeds |
| Session isolation | An unrelated client cannot join an existing session; two legitimate clients receive different inner addresses and exchange distinct payloads concurrently |
| Proxies | Sixteen concurrent SOCKS5/HTTP CONNECT transfers, authenticated half-close, destination identity and refusal of unavailable targets |
| Cleanup | SIGTERM removes client TUN interfaces and owned routes; pre-existing namespaces and routing tables survive rejection |
| Full dual-stack routing | IPv4 and IPv6 transfers use the full tunnel; both policy-rule sets are restored, and IPv4-only exits are rejected for full mode |
| Aggregation benefit | Three identical-payload trials per mode, with each path capped at 2 Mbit/s; median multi-uplink goodput must exceed the single-uplink control by at least 20% |
| Physical link state | A transfer succeeds with one interface down, then the restored interface carries payload again |
| Provenance | Source hashes remain unchanged throughout testing; executed release binary hashes are retained |

A second, IPv6-only underlay topology exercises the `triglav server` entry point,
two independently routed IPv6 uplinks, full-tunnel IPv4 and IPv6 payloads, both
proxies, per-link captures, and restoration of IPv4/IPv6 routing state. The
primary topology exercises `triglav-server` and two source-NATed IPv4 uplinks.

PCAPs establish outer-path participation and encryption flags. Application byte
comparison establishes delivered content. Server status comes from the actual
forwarding object, not a separate copied metrics map. Those observations serve
different assertions and must not be substituted for each other.

## Assumption register

| ID | Assumption and dependent results | Falsification / stress probe |
| --- | --- | --- |
| A1 | Linux TUN, IPv4/IPv6 routing and conntrack implement the service's kernel forwarding layer; all network gates depend on it | Run the container preflight and negative direct-connectivity test; disable forwarding and verify transfer gates fail |
| A2 | The test restores the owned source-policy routes after Linux removes them on link-down. The two configured namespace paths model independent IP uplinks; path and failover gates depend on this topology | Inspect routes, PCAPs and distinct NAT peers; drop each router's forwarding independently |
| A3 | The independent endpoint and byte comparisons are correct; integrity and exit-identity gates depend on them | Review the small standalone Python endpoint/client, retain per-transfer hashes, and compare destination records |
| A4 | The configured shared access token remains confidential; client admission depends on it | Attempt an authenticated NK handshake with the wrong access token and attempt a session join with an unrelated join token |
| A5 | The finite transfer sizes, concurrency and impairment settings represent the tested workload only | Increase sizes/concurrency or change `tc` parameters; require the same assertions, not a reduced pass threshold |
| A6 | Namespace paths model network-layer behavior, not independent radio hardware or ISPs | Repeat against deployed server and two physical uplinks; the container result alone does not discharge this assumption |

## Wire and resource invariants

Production UDP peers negotiate `TG2` inside the encrypted handshake. A logical
client shares an unguessable join token across its uplinks, and the handshake
binds that token to the complete session identifier and numeric uplink ID.
A configured deployment access token is independently checked before allocating
session state. Each uplink has separate encryption keys and nonce state.

A data record carries an explicit 8-byte nonce. The encrypted plaintext binds
the packet type (1 byte), logical sequence (8 bytes), and payload. The AEAD tag
adds 16 bytes. Replay state changes only after successful authentication and
header verification. A replay window permits reordered records within its
bounded history; records outside that history are rejected.

For the default inner MTU of 1340 bytes:

```text
inner IP                         1340 bytes
record binding + nonce + tag       33 bytes
Triglav header                     60 bytes
outer UDP                           8 bytes
outer IPv4                         20 bytes
                                 ----------
outer IPv4 packet                1461 bytes
outer IPv6 packet                1481 bytes
```

Thus the inner payload and protected record fit the protocol's 1400-byte
payload limit: `1340 + 33 = 1373 <= 1400`. This calculation assumes an outer
path MTU of at least 1481 bytes for IPv6 transport, or 1461 bytes for IPv4
transport. Smaller outer MTUs require configuration or additional PMTU work;
the default calculation is not proof for every access network.

Authenticated record replay tracking takes amortized `O(log W)` time and `O(W)` space,
with a bounded nonce history; one eviction batch can take `O(W log W)` time. The exit bounds logical sessions by configuration,
uplinks by 16 per session, and pending proxy tasks by the session limit. Reverse
IP delivery scans `S` sessions (`O(S)`) and sorts at most 16 candidate paths.
TCP proxy records are limited to 65535 bytes and use independent per-connection
cipher state. They use kernel TCP reliability; established proxy TCP connections
are pinned to one outer uplink and do not migrate between outer TCP connections.

## Deployment profile and limits

The implemented exit is a Linux dual-stack forwarding service. Enable kernel
IPv4 and IPv6 forwarding and configure egress NAT/routing on the exit host.
Use `--ipv4-only` for an exit without IPv6; full-tunnel clients refuse that
profile to avoid silently bypassing IPv6. For example, with
an actual local server address substituted for `SERVER_IP` and the actual
external interface substituted for `eth0`:

```sh
export TRIGLAV_ACCESS_TOKEN=$(openssl rand -hex 32)
sudo sysctl -w net.ipv4.ip_forward=1 net.ipv6.conf.all.forwarding=1
sudo iptables -t nat -A POSTROUTING -s 10.0.85.0/24 -o eth0 -j MASQUERADE
sudo ip6tables -t nat -A POSTROUTING -s fd00:7472:6967::/64 -o eth0 -j MASQUERADE
sudo --preserve-env=TRIGLAV_ACCESS_TOKEN triglav-server \
  --listen SERVER_IP:7443 --key /etc/triglav/server.key --generate-key
```

Supply the printed connection key and the same access token to the client:

```sh
sudo --preserve-env=TRIGLAV_ACCESS_TOKEN triglav tun CONNECTION_KEY \
  --interface IFACE1 --interface IFACE2 --aggregate --route DESTINATION_CIDR
```

The access token is a bearer capability, not an individual-user identity or
revocation system. Omitting it on the server permits public client admission.
The original NK implementation substitutes BLAKE3 through a custom resolver;
standard Noise implementations using BLAKE2s are not wire-compatible.

Remaining validation boundaries include physical NIC/ISP deployment, macOS
TUN runtime, sustained soak/load limits, and UDP-blocked
initial connection establishment. Windows TUN support remains unimplemented.
Full and split IPv4/IPv6 routing are exercised over IPv4 and IPv6 outer paths.
Routing restoration is verified for graceful SIGTERM, not power loss or SIGKILL.
A controlled network pass must not be represented as proof of these boundaries.


## Bounded findings beyond the forwarding check

| Impact | Finding | Probe or consequence |
| --- | --- | --- |
| High | Server startup and echo fixtures can produce apparent success without production forwarding | Run shipped binaries and require destination-side payload hashes |
| High | Interface names alone did not bind the underlying sockets | Require distinct NAT peers and per-path packet captures |
| High | A server public key is not client admission control | Configure and reject incorrect 256-bit access tokens |
| High | Historical loss could permanently suppress a restored uplink | Require payload traffic on the restored interface after a real link-down event |
| Medium | Failed stripes could reuse another packet's sequence | Force transport send failures and assert unique sequence assignment |
| Medium | Full-tunnel and harness cleanup could affect pre-existing state | Compare policy rules and routing tables; seed a namespace before rejection |
| Medium | The original TUN MTU exceeded the encrypted wire payload budget | Check the explicit byte budget and fragmented UDP transfer |
| Low | URL-safe base64 keys can contain underscores | Deterministic key parser regression |

## Primary references

- [RFC 1624, equation 3](https://www.rfc-editor.org/rfc/rfc1624.html): incremental Internet checksum update.
- [RFC 1928](https://www.rfc-editor.org/rfc/rfc1928.html): SOCKS version 5 connection semantics.
- [Noise Protocol Framework](https://noiseprotocol.org/noise.html): handshake and transport concepts; the repository's custom hash mapping is described above.
