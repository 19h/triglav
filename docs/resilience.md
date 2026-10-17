# Resilience acceptance contract

This work extends the existing Linux network-namespace proof. A passing unit
test does not establish live recovery. Each feature group is published only
after its relevant checks pass; the overall extension remains incomplete until
the fault matrix below passes against the resulting binaries.

| Requirement | Implementation and acceptance condition | State |
|---|---|---|
| R1: bounded retransmission | Packet and byte caps; absolute expiry during a total outage; one congestion slot per outstanding path copy; ACK/send race tests; occupancy in `/status` | Validated; evidence below |
| R2: automatic recovery | Retry failed startup and stale paths with bounded backoff; authenticated rebinding and rekey; retain tunnel addresses across exit restart when the lease is available; stop retries on shutdown | Validated; evidence below |
| R3: network variability | Explicit MTU configuration and negotiation; dual-stack underlay selection; correct behavior on a constrained path; preserve existing split/full routing | Validated; evidence below |
| R4: lifecycle and live faults | Owned background tasks; cancellation on shutdown; bounded affinity and receive queues; forced exit crash, repeated recovery and source-port churn in the real-binary lab | Validated; evidence below |

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

## R2: recovery protocol and lifecycle

`TG3` handshakes bind the join capability, session, path ID, preferred IPv4
lease, next return sequence, and monotonically increasing handshake epoch.
An optional deployment capability follows these fields. The exit rejects a
conflicting requested lease. IPv6 is derived from the retained IPv4 lease and
the configured exit prefix; a client rejects a changed IPv6 assignment.

A new handshake prepares replacement keys. A fresh authenticated record proves
possession before the exit replaces live keys or advances return sequence state.
The client waits for an encrypted response to its random confirmation challenge
before exposing the replacement transport. Recorded hellos cannot roll a path
back to an older epoch. Fresh authenticated records can move a path's observed
NAT endpoint without allocating another path. Replacing a local source address
retains the numeric path ID. The exit still accepts legacy `TG2` clients; new
clients require an exit that understands `TG3`.

Failed startup attempts continue with bounded exponential backoff. Established
clients retry paths after 15 s without an authenticated record. The default
backoff starts at 250 ms and caps at 15 s, with bounded jitter; handshake and
maintenance scheduling add to those delays. UDP hello retransmission runs every
500 ms within the configured handshake timeout. Per-path locks serialize
handshakes, and lifecycle checks reject a response arriving after shutdown.
Periodic interface snapshots supplement native events and exclude the client's
own TUN interface. Explicit interface lists remain binding when automatic
discovery is enabled.

The recovery lab keeps one TCP connection open across a 20 s all-path blackout,
an exit restart with the same key, two observed NAT address changes, and ten
rounds replacing both local source addresses. Each round must advance both key
generations while retaining two path IDs. Destination records must exactly match
the client hashes. The blackout also samples the bounded retransmission queue.
The restart replaces the exit process within the same network namespace, retaining
kernel routing and conntrack state. Host reboot and an egress-address change are
outside this continuity result.

### R2 validation, 2026-10-17

`validation/artifacts/20261017-resilience-r2-02/summary.json` records a complete
release acceptance pass. The macOS default suite and Linux release suite each
passed 348 tests, with 10 explicitly ignored; the live admission test passed
separately. The original 220 destination records and 17,817,030 bytes in each
direction remain verified. The recovery lane adds 18 records and 442,368 bytes
in each direction: 238 records and 18,259,398 bytes in total.

The recovery connection completed 15 frames with the same process and session
through the 20 s blackout, exit restart, two NAT endpoint changes and 20 local
path replacements. Both retained path IDs reached 11 confirmed key generations
after the exit restart. The maximum frame round trip was 30.469 s. All 133
pending-buffer samples stayed within the configured bounds; the observed peak
was 249 logical packets and 264,284 payload bytes. This workload did not reach
the capacity limits; unit tests separately exercise admission at both limits.

Source manifest SHA-256:
`b48cde09817938409b86c56d8078e2e414f0c72b022581c9b89f7e9118db1b20`.
The run retains the matching release executables, source archive, packet captures
and destination observations. A prior development run exposed missing interface
rescan behavior; the complete acceptance run includes the periodic-snapshot fix.

## R3: negotiated packet-size limits

`--mtu` on the TUN client and either exit entry point sets an inner-packet ceiling
between 1280 and 1367 bytes. The default remains 1340 bytes. The lower bound
preserves the IPv6 minimum link MTU from
[RFC 8200, section 5](https://www.rfc-editor.org/rfc/rfc8200#section-5).
The upper bound is the 1400-byte record payload limit less 33 bytes for the
explicit nonce, authenticated record binding and authentication tag.

| Component | IPv4 underlay, bytes | IPv6 underlay, bytes |
|---|---|---|
| Triglav header | 60 | 60 |
| Record overhead | 33 | 33 |
| UDP header | 8 | 8 |
| Ordinary outer IP header | 20 | 40 |
| Total outside the inner packet | 121 | 141 |

The client offers the minimum of its configured ceiling, previously negotiated
ceiling and the interface MTU minus this overhead. Encrypted probes carry the
offer; an authenticated response bound to the probe confirms the exit ceiling.
The shared exit TUN takes the minimum accepted ceiling, and each client lowers
its TUN accordingly. The 5 s liveness interval also detects local interface MTU
reductions. The client TUN loop applies a negotiated reduction on its periodic
tick. Obsolete retransmission payloads larger than the new ceiling expire.

For the controlled 1430-byte IPv6 underlay, `1430 - 141 = 1289` bytes is the
negotiated inner MTU. The dedicated lane keeps one IPv4 and one IPv6 TCP socket
open across the reduction, checks both tunnel MTUs, validates destination bytes,
and rejects oversized or fragmented UDP records in the steady-state captures.
The TCP proxy lane is validated separately; its packets are not UDP records.

These controls negotiate ceilings; they do not implement the full datagram
path-MTU search and probe-size confirmation described in
[RFC 8899](https://www.rfc-editor.org/rfc/rfc8899). An unobserved smaller router
MTU still requires an appropriate configured ceiling. Headers and encapsulations
beyond those in the table require their own allowance. Automatic increases are
not performed: the ceiling is conservative for the client/exit lifetimes. A
fresh deployment is needed to recover a larger ceiling. Older exits can retain
the default 1340-byte operation; reduced ceilings require MTU negotiation support.

The scheduler combines its relative RTT cutoff with the existing absolute RTT
allowance. Sub-millisecond measurement differences no longer exclude an
otherwise healthy path. A new path starts with the measured RTT of its encrypted
confirmation exchange. Proxy acceptance waits for both required paths before
asserting that both source addresses carried connections.

### R3 validation, 2026-10-17

`validation/artifacts/20261017-resilience-r3-02/summary.json` records a complete
release pass: 356 Linux Rust tests, 10 explicitly ignored tests, and one separate
live admission test. The macOS library suite passed 165 tests. Across all four
network lanes, 349 destination records account for 23,757,638 bytes in each
direction with matching hashes. The rate-capped median goodput ratio was 1.5161.

The constrained IPv6-underlay lane carried 40 frames on each of the existing
IPv4 and IPv6 TCP connections after negotiating a 1289-byte inner MTU. Its
steady-state captures contained no oversized or fragmented UDP records. The
recovery lane retained one connection across all faults; its largest frame round
trip was 58.775 s. This is a measured delay, not an upper-bound recovery guarantee.

Source manifest SHA-256:
`2d8e432814aafaf1c181c6152738ce49464b8a5f1c3d3e3ce8facde9cb81e270`.

## R4: bounded lifetimes and finite stability testing

The multipath manager owns its connection, receive and maintenance tasks.
There is one receive owner per lifecycle; a second claimant receives an error.
Dropping the consumer cancels the coordinator and its per-path workers, including
blocked reads. Maintenance holds a weak manager reference between ticks.
Shutdown invalidates the lifecycle before aborting tasks, clearing pending
payloads and releasing transport/key references from retained path handles.
Packet admission after shutdown is rejected under the lifecycle lock.
The TUN runner also owns and cancels its NAT/DNS maintenance tasks.

Flow affinity retains at most 4096 entries by default, evicts the least recently
used binding at capacity, and expires idle entries after 120 s. These values are
configurable in `MultipathConfig`. The scheduler's separate affinity cache also
has a 4096-entry default cap, with its existing 5 s idle timeout; its capacity is
configured by `SchedulerConfig.max_sticky_flows`. An evicted flow can be assigned a new path;
inner TCP still owns ordering and retransmission. Lookup/update/eviction take
expected O(1) hash access plus O(log F) recency updates, with O(F) storage.
Expiry batches and path cleanup take O(F log F) time in the worst case. The
scheduler can purge an expired batch during lookup; F is always bounded by its
configured entry cap.
Removing a path clears its retry metadata and outstanding congestion charges.

The ready-packet queue is capped at 4096 entries. Production receive channels
remain bounded; if the TUN consumer stalls, excess payloads are counted as drops
while the receiver continues handling authenticated control records. UDP remains
best effort. The exit retains at most 1024 distinct completed-proxy source keys,
plus one `_other` aggregate; total connection counts are preserved up to u64
saturation. These component bounds do not constitute a process-wide memory cap.

Production IP sends bypass the legacy per-packet pacing sleeps. Those sleeps
blocked the TUN processing loop and delayed unrelated TCP acknowledgements under
UDP churn. Inner TCP retains the kernel's congestion control; UDP remains best
effort. The churn gate requires at least ten new TCP frames during its 60 s run.

Production client/exit packet send and TUN-write operations use a 1 s timeout. A partial TUN or datagram write
is an error. The exit marks a logical packet delivered and acknowledges it only
after the full TUN write succeeds. A failed or timed-out write remains eligible
for a fresh-nonce retry; an already delivered duplicate receives another ACK
without another TUN write. An ACK confirms local kernel acceptance, not delivery
to the remote application.

The recovery lane now kills the exit with SIGKILL and restarts it with the same
static key. One existing TCP connection crosses that crash, the 20 s total-path
outage and repeated address changes. It then remains open during 60 s of IPv4
and IPv6 UDP source-port churn using flow-affinity scheduling. Every sampled
flow/pending count must remain within its configured bound, and the existing
stream and an IPv6 transfer must preserve exact destination bytes. RSS is
recorded; the workload gate permits a 128 MiB increase from its initial sample.
That finite envelope is a regression threshold, not an indefinite memory-use
guarantee or an operating-system-wide limit.

### R4 validation, 2026-10-17 UTC

`validation/artifacts/20261017-resilience-r4-03/summary.json` records the complete
release acceptance pass. Linux and macOS each passed 368 release tests with 10
explicitly ignored tests; the live admission test passed separately. The minimal
library feature build also passed. The four network lanes verified 960 exact
destination records and 33,768,262 bytes in each direction.

The same client process, session and TCP connection crossed a 20 s total outage,
a SIGKILL/restart of the exit process, two NAT endpoint changes and twenty local
path replacements. It completed 593 frames overall, including
572 during 60.009 s of source-port churn.
The churn generated 73,216 datagrams using
41,100 distinct family/port pairs. All 572 samples
remained within the flow and pending-buffer bounds; the flow cache reached its
4096-entry cap. RSS rose from 19,103,744 to a peak of
21,422,080 bytes, a 2,318,336-byte
increase. The maximum recovery-frame round trip was
26.518 s; this is an observation, not a deadline.

Both MTU reduction streams stayed on their original sockets, completing
53 IPv4 and 59 IPv6 frames.
The steady-state captures had no oversized or fragmented UDP records. The
rate-capped median two-path/single-path goodput ratio was
1.3988. Artifact ownership returned to the caller
while the generated exit key retained mode 0600.

An earlier debug build measured 68,009 scheduler selections/s and failed the
100,000/s timing floor. The final performance acceptance uses release builds;
no timing threshold was reduced. An initial live churn attempt exposed the
blocking-pacing stall; the final gate requires TCP progress during churn.

Source manifest SHA-256: `c8b3370e5d9b70c9f4d9c57959bf713ba0c3157f4e6c8bd1ac308805e57632af`.
The evidence directory retains matching sources, executables, packet captures,
endpoint records, resource samples and macOS test logs.

## Assumption register

| ID | Assumption | Falsification probe | Dependent result |
|---|---|---|---|
| A1 | Inner TCP retransmission remains enabled and its timeout exceeds an injected outage | Keep one TCP stream open through a longer all-path outage and compare destination bytes | TCP continuity; not UDP losslessness |
| A2 | A restarted exit retains its configured static key and tunnel prefix | Restart the real exit, then repeat with a different key/prefix and require rejection | Authenticated recovery and address continuity |
| A3 | The previous tunnel address remains available after restart | Restart with another client claiming the old address and require an explicit collision outcome | Reuse of an existing inner flow |
| A4 | A successful lab run applies to its recorded source, executable and topology | Hash sources and binaries; capture topology and destination records | Reported live acceptance only |
| A5 | The configured ceiling or local interface MTU accounts for smaller links along the route | Insert a smaller remote-router MTU without changing the local interface; configure a lower ceiling when required | MTU adaptation; no full path-MTU discovery claim |
| A6 | The async executor continues scheduling tasks | Block a transport read, close its consumer and enforce the cancellation deadline; separately kill the process | Task and I/O deadlines; no cleanup guarantee after process termination |
| A7 | Exit process restart retains the kernel network namespace, egress address and conntrack state | Recreate the exit namespace or change its egress identity and repeat the same TCP flow | Existing TCP continuity across process restart only |

## Adjacent findings and bounded scope

| Impact | Finding | Required response |
|---|---|---|
| High | The old pending map had no retention cap and could repeatedly charge losses during a total outage | R1 bounds, expiry and regression tests |
| High | Replacing live Noise state upon an unconfirmed handshake could permit replay-triggered denial of service | R2 must confirm fresh session-key possession before replacement |
| High | A new exit can allocate a different inner address after restart | R2 must reject a conflicting lease explicitly; never silently misroute an existing flow |
| Medium | Smaller underlay MTUs invalidate the default encapsulation budget | R3 must account for framing, authentication and IP-family overhead |
| High | Flow affinity had two independent unbounded caches, and detached workers could outlive their consumer | R4 caps both caches and owns/cancels worker tasks |
| Medium | A short lab run cannot establish indefinite availability or physical-link behavior | R4 reports duration and topology; physical ISP behavior remains unmeasured |

Quality gates require source-backed claims, the assumptions above, complete
requirement coverage, reproducible byte/time budgets, and no unresolved
contradictions within each published group. Overall acceptance requires R1-R4.
