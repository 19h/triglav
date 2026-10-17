# Resilience acceptance contract

This work extends the existing Linux network-namespace proof. A passing unit
test does not establish live recovery. Each feature group is published only
after its relevant checks pass; the overall extension remains incomplete until
the fault matrix below passes against the resulting binaries.

| Requirement | Implementation and acceptance condition | State |
|---|---|---|
| R1: bounded retransmission | Packet and byte caps; absolute expiry during a total outage; one congestion slot per outstanding path copy; ACK/send race tests; occupancy in `/status` | Validated; evidence below |
| R2: automatic recovery | Retry failed startup and stale paths with bounded backoff; authenticated rebinding and rekey; retain tunnel addresses across exit restart when the lease is available; stop retries on shutdown | Pending |
| R3: network variability | Explicit MTU configuration and negotiation; dual-stack underlay selection; correct behavior on a constrained path; preserve existing split/full routing | Pending |
| R4: lifecycle and live faults | Idempotent background tasks; cancellation on shutdown; prolonged total outage, exit restart, changed source endpoint, constrained MTU and repeated recovery in the real-binary lab | Pending |

## R1: retransmission invariants

The default plaintext retention budget is 4096 logical packets and 8,388,608
bytes (8 MiB), with an absolute maximum age of 10 s. These are configurable
through `MultipathConfig`. Metadata, transport queues, and receive buffers are
separate allocations; this is not a process-wide memory bound. Admission beyond
either cap returns the recoverable `Error::Backpressure` before transmission.
The TUN path drops a rejected datagram; inner TCP may retransmit it. UDP delivery
remains best effort.

Admission and accounting use a single short critical section. Network I/O never
holds that lock. A pending logical packet owns a map of path attempts. The first
ACK retires all attempts, records RTT on the acknowledging path, and cancels
redundant copies without recording them as losses. A retry removes and charges
each expired attempt once. No available path still consumes the retry budget;
absolute age independently bounds retention. Late ACKs cannot decrement a
congestion counter below zero. ACK element counts are checked against the actual
payload before processing, avoiding an allocation controlled by a claimed count.

Admission and removal take expected O(1) time. Each retry scan takes O(P + A)
time for P pending packets and A expired path attempts. Retained storage is
O(B + P * U), where B is the configured payload-byte cap and U the number of
uplinks. Only one retry scan may run at a time.

### R1 validation, 2026-10-17

Commands: `cargo test --locked --all-targets -- --test-threads=1`,
`cargo check --locked --no-default-features --lib`, and
`bash validation/run.sh validation/artifacts/20261017-resilience-r1`.

| Check | Recorded result |
|---|---|
| Default Rust tests, macOS and Linux | 337 passed on each; 10 explicitly ignored on each |
| New accounting and capacity regressions | 7 passed, included in those totals |
| Live production admission test | 1 passed separately |
| Destination-verified records | 188 primary plus 32 IPv6-underlay records |
| Exact application bytes | 17,817,030 bytes in each direction |
| Rate-capped median goodput ratio | 1.517390807, two paths versus one; 3 trials per mode |
| Continuous TCP failover | 120 frames on one connection; maximum frame round trip 4.857559394 s |
| Minimal library feature build | Passed |
| Source manifest SHA-256 | `3b33398739d41f1c693a9a7c61b2f44d894d7e9d056155912ba3129f83f9603e` |

The artifact directory retains the source archive, executable hashes, packet
captures, endpoint records and `summary.json`. This run preserves the existing
acceptance profile; it does not establish R2-R4 or a process-wide memory ceiling.

## Assumption register

| ID | Assumption | Falsification probe | Dependent result |
|---|---|---|---|
| A1 | Inner TCP retransmission remains enabled and its timeout exceeds an injected outage | Keep one TCP stream open through a longer all-path outage and compare destination bytes | TCP continuity; not UDP losslessness |
| A2 | A restarted exit retains its configured static key and tunnel prefix | Restart the real exit, then repeat with a different key/prefix and require rejection | Authenticated recovery and address continuity |
| A3 | The previous tunnel address remains available after restart | Restart with another client claiming the old address and require an explicit collision outcome | Reuse of an existing inner flow |
| A4 | A successful lab run applies to its recorded source, executable and topology | Hash sources and binaries; capture topology and destination records | Reported live acceptance only |

## Adjacent findings and bounded scope

| Impact | Finding | Required response |
|---|---|---|
| High | The old pending map had no retention cap and could repeatedly charge losses during a total outage | R1 bounds, expiry and regression tests |
| High | Replacing live Noise state upon an unconfirmed handshake could permit replay-triggered denial of service | R2 must confirm fresh session-key possession before replacement |
| High | A new exit can allocate a different inner address after restart | R2 must reject a conflicting lease explicitly; never silently misroute an existing flow |
| Medium | Smaller underlay MTUs invalidate the default encapsulation budget | R3 must account for framing, authentication and IP-family overhead |
| Medium | A short lab run cannot establish indefinite availability or physical-link behavior | R4 reports duration and topology; physical ISP behavior remains unmeasured |

Quality gates require source-backed claims, the assumptions above, complete
requirement coverage, reproducible byte/time budgets, and no unresolved
contradictions within each published group. Overall acceptance requires R1-R4.
