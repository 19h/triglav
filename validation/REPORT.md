# Observed production validation

## Resilience extension, 2026-10-17

The complete release run in
`validation/artifacts/20261017-resilience-r2-02/` passes. It adds automatic
startup/path retry, confirmed rekeying, authenticated NAT rebinding, stable path
IDs across address changes, and periodic interface snapshots to the earlier
forwarding profile. The [resilience contract](../docs/resilience.md) records
assumptions, bounds and remaining MTU/lifecycle work.

| Check | Recorded result |
|---|---|
| Rust tests | 348 passed on macOS and Linux; 10 ignored on each |
| Live admission test | 1 passed separately |
| Exact destination records, all three lanes | 238 |
| Exact bytes in each direction | 18,259,398 |
| Recovery lane | One TCP connection across a 20 s total outage, exit restart, two NAT address changes and 20 path replacements |
| Recovery identity | Same client process, session and two path IDs |
| Confirmed generations after restart | 11 per path |
| Recovery maximum frame round trip | 30.469 s |
| Rate-capped median goodput ratio | 1.4315, two paths versus one; 3 trials per mode |
| Source manifest SHA-256 | `b48cde09817938409b86c56d8078e2e414f0c72b022581c9b89f7e9118db1b20` |
| Client executable SHA-256 | `1af0887997e9e16f761c3369fc2ba7d4cdefb97de1c0a4c290db57e1cfe0ecd2` |
| Exit executable SHA-256 | `4cb64cbc9d871788f08941d414554ce9efb6426bf1b1882a32cfe01e8a2ff40a` |

The following sections retain the earlier rebase baseline. They describe that
recorded snapshot rather than the extension's newer counts.

## Earlier rebase baseline

The Linux release profile passes the complete acceptance run identified below.
The shipped client and both server entry points carried real traffic through
multiple uplinks to one exit node and an independent destination. The source
snapshot, executed binaries and observations are retained together.

The rebase preserves upstream base58 keys (with legacy-key parsing), endpoint
advertisement, packet hedging and dynamic uplink support. The TUN exit remains
the default server implementation; the upstream relay runtime remains available
as a library API. Failed handshake retry and authenticated path recovery have
explicit regression coverage.

This establishes functional, reproducibly testable behavior for the recorded
profile. It is finite experimental evidence, not formal verification or a claim
about every network, operating system or failure mode.

## Measured results

| Check | Observed result |
| --- | --- |
| Linux Rust suite | 330 passed; 0 failed; 10 ignored by default |
| macOS Rust suite | 330 passed; 0 failed; 10 ignored by default; no privileged TUN claim |
| Live admission probe | 1 passed separately: wrong deployment token and unrelated session join rejected |
| Independent destination records | 188 in the IPv4-underlay lane; 32 in the IPv6-underlay lane |
| Payload integrity | 17,817,030 application bytes in each direction, exact client comparisons and matching destination SHA-256 multisets |
| Single exit | Every destination peer was `198.18.0.1` or `2001:db8:85::1`, according to address family |
| IPv4-underlay path evidence | Two distinct NAT peers; 7063 and 8430 encrypted data packets captured |
| IPv6-underlay path evidence | Two distinct IPv6 source addresses, encrypted payload on both, full dual-stack forwarding |
| Continuity | One TCP socket completed 120 frames while each path was independently blackholed for 2 s |
| Interface state | Transfer completed with one interface down; restored interface subsequently carried data |
| Proxies | 16 simultaneous SOCKS5/HTTP CONNECT transfers per underlay lane; half-close and four unavailable-target cases per lane |
| Client isolation | Two clients concurrently transferred distinct payloads using distinct assigned IPv4/IPv6 addresses |
| Routing | Full/split dual-stack forwarding, SIGTERM restoration, occupied-table rollback and pre-existing namespace preservation passed |
| Invalid input | Replay, ciphertext corruption, forged sequence and plaintext rejected; valid traffic then succeeded |
| IPv4-only exit | Full-tunnel startup rejected without leaving routing changes |
| Source identity | All 91 input files matched their pre-run SHA-256 values |

The ignored Rust tests are not counted as executed. The live admission probe is
explicitly invoked by the production namespace harness. Mock-server regressions
and production-network observations are separate evidence.

## Aggregation measurement

Each router's outgoing link was limited to 2,000,000 bit/s. Three trials per mode
used the same 262,144-byte request and exact echo through the same exit and
destination. The single-uplink control used uplink 1; aggregation used both.

| Quantity | Single uplink | Two uplinks |
| --- | ---: | ---: |
| Median round-trip transfer time | 1.994897 s | 1.187369 s |
| Request bytes / complete transfer time | 1051258 bit/s | 1766217 bit/s |

`R = median(single time) / median(multi time) = 1.680`.
The predeclared acceptance threshold was `R >= 1.2`. Request goodput includes the
complete echo transaction's elapsed time; it is not raw link capacity. Three
trials establish this measurement, not a population confidence interval or a
universal speedup guarantee. [All timing samples](artifacts/20261017-rebase-04/benchmark.json).

The largest continuous-frame round trip was 9.389 s. The separate
interface-down transfer took 13.183 s. Continuity passed; instantaneous
failover is not claimed.

## Reproduce and inspect

```sh
bash validation/run.sh
```

The harness fails on a missing prerequisite, failed assertion, source change or
pre-existing reserved namespace. It retains exact source and executable copies;
it does not depend on recovering a dirty working tree from a commit hash.

- [Machine-readable acceptance result](artifacts/20261017-rebase-04/summary.json)
- [Tested source archive](artifacts/20261017-rebase-04/source.tar.gz), [input SHA-256 manifest](artifacts/20261017-rebase-04/source-sha256.txt), [post-run verification](artifacts/20261017-rebase-04/source-verification.log)
- [Executed client binary](artifacts/20261017-rebase-04/bin/triglav), [executed standalone server binary](artifacts/20261017-rebase-04/bin/triglav-server), [archive/binary copy hashes](artifacts/20261017-rebase-04/archive-sha256.txt)
- [Linux Rust results](artifacts/20261017-rebase-04/tests.log), [macOS Rust results](artifacts/20261017-rebase-04/host-tests.log), [live admission results](artifacts/20261017-rebase-04/admission.log)
- [IPv4-underlay network results](artifacts/20261017-rebase-04/network.log), [IPv6-underlay results](artifacts/20261017-rebase-04/outer-ipv6.log)
- [Destination observations](artifacts/20261017-rebase-04/destination.jsonl), [IPv6-underlay destination observations](artifacts/20261017-rebase-04/outer-ipv6/destination.jsonl)
- [Ingress 1 capture](artifacts/20261017-rebase-04/ingress1.pcap), [ingress 2 capture](artifacts/20261017-rebase-04/ingress2.pcap), [IPv6 captures summary](artifacts/20261017-rebase-04/outer-ipv6/packet-proof.json)
- [Container image identity](artifacts/20261017-rebase-04/image.json), [compiler identity](artifacts/20261017-rebase-04/rustc.txt), [kernel identity](artifacts/20261017-rebase-04/platform.txt)

Source manifest SHA-256: `61416faf97cbdfc25099c21dc5b30c56fd4976d4bbfeae82c239ba1025da6658`.

Executed binary SHA-256 values:

```text
b937329194c4dc1312ce9d18c23437a1631dae7ab9819d9732b39bb6a70b7647  /build/release/triglav
78f90ccb52df2d653604f3989b391d50e599fb5b48049fc36af68b3535a0ab41  /build/release/triglav-server
```

This is a working-tree snapshot. No published package, pushed commit or remote
CI execution is claimed. The added GitHub workflow invokes the same harness and
uploads its evidence. The run directory is ignored by Git; retain or export it
separately. Earlier directories contain development attempts, including an
interrupted run before documentation was corrected; they are not this result.

## Assumptions and falsification probes

| ID | Assumption; dependent result | Stress or falsification probe |
| --- | --- | --- |
| A1 | Recorded Linux kernel TUN/routing/conntrack behavior; all network observations | Container preflight, failed direct-connectivity control, independent exit/destination namespaces |
| A2 | Independently routed namespace links model IP-path separation; striping/failover result | Distinct source addresses, two captures, separate router blackholes, interface-down and restored-route checks |
| A3 | Standalone endpoint and client byte comparisons represent received data; integrity result | Exact byte comparison plus independent destination SHA-256 multiset including multiplicity |
| A4 | Deployment capability is confidential; admission result | Wrong access token and wrong session join token attempted against the live server |
| A5 | Finite sizes, 16 proxy streams and controlled impairment settings define the workload; performance/reliability result | Change sizes, concurrency, rates or loss while retaining the same assertions |
| A6 | Graceful SIGTERM permits cleanup; restoration result | Before/after IPv4/IPv6 rules/routes, occupied-table startup rollback; SIGKILL/power loss is outside this result |
| A7 | Physical access networks may differ; deployment extrapolation is unknown | Repeat the same destination-side checks over real NIC/modem/ISP paths before claiming that environment |

All measured results above depend on A1-A5 as applicable; cleanup also depends on
A6. No physical-deployment conclusion is inferred from A7. Detailed protocol,
MTU calculations and runnable topology are in the [validation contract](README.md).

## Implementation findings

| Impact | Finding and completed correction |
| --- | --- |
| High | Server entry points were idle/echo fixtures; both now run a shared real TUN exit |
| High | Uplink names did not bind actual sockets; transport and proxy sockets now bind their selected interfaces |
| High | Implicit UDP cipher counters broke after loss/reordering; negotiated records now carry explicit authenticated sequencing and bounded replay history |
| High | Full routing required dual-stack behavior; address negotiation, forwarding, IPv6 underlay selection and rollback now cover both families |
| High | Proxy traffic needed reliable separate streams; authenticated TCP channels isolate concurrent requests and preserve half-close |
| Medium | New paths could receive zero aggregation weight; bootstrap weighting now permits initial traffic |
| Medium | Failed stripes could reuse another packet's sequence, and released reorder packets could be discarded; regression-tested corrections preserve sequence identity and delivery |
| Medium | NAT port rewrites omitted checksum adjustment; incremental correction follows RFC 1624 equation 3 |
| Medium | MTU and parser bounds were inconsistent; default MTU is 1340 bytes and packet lengths are validated |
| Medium | Cleanup could affect existing state; owned-route tracking and namespace guards preserve it |
| Low | URL-safe keys containing underscores failed parsing; parser and regression coverage corrected |

Primary protocol references: [RFC 1624](https://www.rfc-editor.org/rfc/rfc1624.html),
[RFC 1928](https://www.rfc-editor.org/rfc/rfc1928.html), and the
[Noise Protocol Framework](https://noiseprotocol.org/noise.html). The repository's
custom BLAKE3 resolver is retained; standard BLAKE2s Noise interoperability is
not claimed.

## Quality gates and remaining boundaries

| Gate | Result for the bounded contract |
| --- | --- |
| QG1 | No normative judgment required |
| QG2 | Assumptions and falsification probes listed above |
| QG3 | Both production entry points, two address families, multiple clients and all declared acceptance gates exercised |
| QG4 | Exact byte counts and SHA-256 observations; timing calculations include units and explicit sample limitations |
| QG5 | Negative controls, rollback and post-attack valid traffic passed; external boundaries remain explicitly unknown |
| QG6 | Source/executable hashes, primary protocol references and destination-side observations retained |
| QG7 | Adjacent findings classified by impact; concrete corrections tested |

Unknown outside this contract: physical NIC/ISP deployment, NAT rebinding or
changing interface addresses, privileged macOS TUN behavior, arbitrary smaller
MTUs, DNS resolver policy, long-duration resource limits, and abrupt process/host
failure cleanup. Windows TUN is unimplemented. Established proxy connections are
pinned to their outer TCP path. None of these are silently counted as passed.
