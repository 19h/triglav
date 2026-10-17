#!/bin/bash
# Run inside an isolated privileged Linux container; never changes host routes.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/validation/readiness.sh"
OUT=${TRIGLAV_EVIDENCE_DIR:-$(mktemp -d /tmp/triglav-evidence.XXXXXX)}
printf 'Evidence: %s\n' "$OUT"
BIN=${TRIGLAV_BIN_DIR:-/build/debug}
mkdir -p "$OUT"
if [[ -e "$OUT/server.log" ]]; then echo "Evidence directory must be fresh" >&2; exit 1; fi
export TRIGLAV_ACCESS_TOKEN
TRIGLAV_ACCESS_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
names=(tgproof-client tgproof-r1 tgproof-r2 tgproof-exit tgproof-target)
pids=()
captures=()
created_names=()
cleanup() {
  if [[ ${TRIGLAV_KEEP_NETNS:-0} == 1 ]]; then return; fi
  for pid in "${pids[@]}"; do kill -CONT "$pid" 2>/dev/null || true; kill "$pid" 2>/dev/null || true; done
  for ns in "${created_names[@]}"; do ip netns del "$ns" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 124' TERM
for ns in "${names[@]}"; do
  if ip netns list | cut -d' ' -f1 | grep -qx "$ns"; then echo "Namespace already exists: $ns" >&2; exit 1; fi
done
# Build the active admission probe outside the deliberately isolated client network.
cargo test --locked --release --test exit_records --no-run --message-format=json >"$OUT/probe-build.jsonl" 2>"$OUT/probe-build.log"
probe=$(python3 - "$OUT/probe-build.jsonl" <<'PYPROBE'
import json, sys
for line in open(sys.argv[1]):
    item = json.loads(line)
    if item.get('reason') == 'compiler-artifact' and item.get('target', {}).get('name') == 'exit_records' and item.get('executable'):
        print(item['executable'])
PYPROBE
)
[[ -x "$probe" ]]
for ns in "${names[@]}"; do ip netns add "$ns"; created_names+=("$ns"); ip -n "$ns" link set lo up; done
link() {
  local a=$1 an=$2 ai=$3 b=$4 bn=$5 bi=$6
  ip link add "$an" type veth peer name "$bn"
  ip link set "$an" netns "$a"; ip link set "$bn" netns "$b"
  ip -n "$a" address add "$ai" dev "$an"; ip -n "$b" address add "$bi" dev "$bn"
  ip -n "$a" link set "$an" up; ip -n "$b" link set "$bn" up
}
link tgproof-client up1 10.71.1.2/24 tgproof-r1 lan1 10.71.1.1/24
link tgproof-client up2 10.72.1.2/24 tgproof-r2 lan2 10.72.1.1/24
link tgproof-r1 wan1 10.73.1.1/24 tgproof-exit in1 10.73.1.2/24
link tgproof-r2 wan2 10.74.1.1/24 tgproof-exit in2 10.74.1.2/24
link tgproof-exit out0 198.18.0.1/24 tgproof-target dest0 198.18.0.2/24
ip -n tgproof-exit address add 10.99.0.1/32 dev lo
ip -n tgproof-exit -6 address add 2001:db8:85::1/64 dev out0 nodad
ip -n tgproof-target -6 address add 2001:db8:85::2/64 dev dest0 nodad
ip netns exec tgproof-exit sysctl -qw net.ipv6.conf.all.forwarding=1
ip netns exec tgproof-exit ip6tables -t nat -A POSTROUTING -s fd00:7472:6967::/64 -o out0 -j MASQUERADE
for n in 1 2; do
  r="tgproof-r$n"
  ip netns exec "$r" sysctl -qw net.ipv4.ip_forward=1
  ip -n "$r" route add 10.99.0.1/32 via "10.7$((n+2)).1.2"
  ip netns exec "$r" iptables -t nat -A POSTROUTING -o "wan$n" -j MASQUERADE
  ip -n tgproof-client rule add from "10.7$n.1.2" table "$((100+n))"
  ip -n tgproof-client route add "10.7$n.1.0/24" dev "up$n" table "$((100+n))"
  ip -n tgproof-client route add 10.99.0.1/32 via "10.7$n.1.1" dev "up$n" table "$((100+n))"
done
ip -n tgproof-client route add 10.99.0.1/32 via 10.71.1.1 dev up1
ip -n tgproof-client route add default via 10.71.1.1 dev up1
for ns in "${names[@]}"; do
  ip netns exec "$ns" sysctl -qw net.ipv4.conf.all.rp_filter=0 net.ipv4.conf.default.rp_filter=0
  ip netns exec "$ns" sh -c 'for i in /proc/sys/net/ipv4/conf/*/rp_filter; do echo 0 > "$i"; done'
done
ip netns exec tgproof-exit sysctl -qw net.ipv4.ip_forward=1
ip netns exec tgproof-exit iptables -t nat -A POSTROUTING -s 10.0.85.0/24 -o out0 -j MASQUERADE
# A destination route exists only after the actual client creates its TUN.
if ip netns exec tgproof-client ping -n -c1 -W1 198.18.0.2; then echo 'Direct bypass unexpectedly works' >&2; exit 1; fi
ip netns exec tgproof-target python3 "$ROOT/validation/endpoint.py" "$OUT/destination.jsonl" >"$OUT/endpoint.log" 2>&1 & pids+=($!)
ip netns exec tgproof-exit "$BIN/triglav-server" --listen 10.99.0.1:7443 --generate-key --metrics 127.0.0.1:9090 >"$OUT/server.log" 2>&1 & pids+=($!)
key=''
for i in {1..100}; do
  key=$(sed -n 's/^\(tg1[A-Za-z0-9_-]*\).*/\1/p' "$OUT/server.log" | head -1)
  [[ -n "$key" ]] && break
  sleep 0.1
done
[[ -n "$key" ]] || { cat "$OUT/server.log"; exit 1; }
for n in 1 2; do
  ip netns exec tgproof-exit tcpdump -U -n -i "in$n" -w "$OUT/ingress$n.pcap" 'udp dst port 7443' >"$OUT/capture$n.log" 2>&1 &
  captures+=($!); pids+=($!)
done
ip netns exec tgproof-client "$BIN/triglav" --log-level debug tun "$key" --interface up1 --interface up2 --tun-name tg0 --route 198.18.0.0/24 --route 2001:db8:85::/64 --aggregate --strategy wrr >"$OUT/client.log" 2>&1 & client_pid=$!; pids+=($client_pid)
for i in {1..100}; do
  if ip -n tgproof-client route show 198.18.0.0/24 | grep -q tg0; then break; fi
  sleep 0.1
done
ip -n tgproof-client route show >"$OUT/client-routes.txt"
wait_client_paths tgproof-client 9091 2 "$OUT/client-ready.json"
ip netns exec tgproof-client ping -n -c3 -W3 198.18.0.2 >"$OUT/ping.txt"
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/before-single-flow.json"
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 262144 1 >"$OUT/tcp.json"
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/single-flow-status.json"
python3 - "$OUT/single-flow-status.json" <<'PYSINGLE'
import json, sys
status = json.load(open(sys.argv[1]))
import pathlib
before = json.loads((pathlib.Path(sys.argv[1]).parent/'before-single-flow.json').read_text())
a = {p['uplink']: p for p in before['paths']}
assert len(status['paths']) == 2 and all(p['received'] > a[p['uplink']]['received'] and p['sent'] > a[p['uplink']]['sent'] for p in status['paths']), status
PYSINGLE
ip netns exec tgproof-client ping -6 -n -c3 -W3 2001:db8:85::2 >"$OUT/ping6.txt"
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 262144 4 tcp6 >"$OUT/tcp6.json"
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 1200 4 udp6 >"$OUT/udp6.json"

ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 524288 8 >"$OUT/concurrent.json"
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 1200 8 udp >"$OUT/udp.json"
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 4096 4 udp >"$OUT/fragmented-udp.json"
ip netns exec tgproof-client python3 "$ROOT/validation/continuous.py" "$OUT/progress.txt" >"$OUT/failover.json" 2>"$OUT/failover.err" &
flow=$!; pids+=($flow)
for i in {1..100}; do [[ -s "$OUT/progress.txt" ]] && break; sleep .1; done
ip netns exec tgproof-r1 iptables -I FORWARD -j DROP
sleep 2
ip netns exec tgproof-r1 iptables -D FORWARD -j DROP
ip netns exec tgproof-r2 iptables -I FORWARD -j DROP
sleep 2
ip netns exec tgproof-r2 iptables -D FORWARD -j DROP
wait "$flow"
# Exercise real send errors as well as silent blackholes.
ip -n tgproof-client link set up1 down
sleep .2
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/link-down-before.json"
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 131072 1 >"$OUT/link-down-transfer.json"
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/link-down-after.json"
python3 - "$OUT" <<'PYLINK'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
a, b = [json.loads((out/f'link-down-{name}.json').read_text()) for name in ['before', 'after']]
by_address = lambda status: {p['address'].split(':')[0]: p['received'] for p in status['paths']}
a, b = by_address(a), by_address(b)
assert b['10.73.1.1'] == a['10.73.1.1'], (a, b)
assert b['10.74.1.1'] > a['10.74.1.1'], (a, b)
PYLINK
ip -n tgproof-client link set up1 up
# Linux removes static routes when a device is administratively taken down.
# Restore the owned underlay configuration before asserting VPN recovery.
ip -n tgproof-client route replace 10.71.1.0/24 dev up1 table 101
ip -n tgproof-client route replace 10.99.0.1/32 via 10.71.1.1 dev up1 table 101
ip -n tgproof-client route replace 10.99.0.1/32 via 10.71.1.1 dev up1
ip -n tgproof-client route replace default via 10.71.1.1 dev up1
ip netns exec tgproof-client ping -I up1 -n -c2 -W2 10.99.0.1 >"$OUT/underlay-restored.txt"
sleep 6
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 262144 1 >"$OUT/link-restored-transfer.json"
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/link-restored-status.json"
python3 - "$OUT" <<'PYRESTORE'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
a = json.loads((out/'link-down-after.json').read_text())
b = json.loads((out/'link-restored-status.json').read_text())
count = lambda status: next(p['received'] for p in status['paths'] if p['address'].startswith('10.73.1.1:'))
assert count(b) > count(a), (a, b)
PYRESTORE
# Repeat with actual asymmetric delay, packet loss and reordering.
ip netns exec tgproof-r1 tc qdisc add dev wan1 root netem delay 20ms 5ms loss 1% reorder 20% 50%
ip netns exec tgproof-r2 tc qdisc add dev wan2 root netem delay 5ms 1ms loss 1%
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 262144 2 >"$OUT/impaired.json"
ip netns exec tgproof-r1 tc qdisc del dev wan1 root
ip netns exec tgproof-r2 tc qdisc del dev wan2 root
for pid in "${captures[@]}"; do kill -INT "$pid"; wait "$pid" || true; done
python3 "$ROOT/validation/packets.py" verify "$OUT/ingress1.pcap" "$OUT/ingress2.pcap" >"$OUT/packet-proof.json"
# Freeze the legitimate sender to isolate mutation effects from final TCP ACKs.
kill -STOP "$client_pid"
sleep .2
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/before-rejection.json"
ip netns exec tgproof-exit python3 "$ROOT/validation/packets.py" inject "$OUT/ingress1.pcap" >"$OUT/mutations.json"
sleep .2
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/after-rejection.json"
python3 - "$OUT" <<'PYSEC'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); a, b = [json.loads((out/f'{name}-rejection.json').read_text()) for name in ['before', 'after']]
assert b['rejected_packets'] >= a['rejected_packets'] + 4, (a, b)
assert b['forwarded_packets'] == a['forwarded_packets'], (a, b)
PYSEC
kill -CONT "$client_pid"
# A valid exchange must still succeed after forged sequence and nonce attempts.
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 16384 1 >"$OUT/after-attack.json"
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/status.json"
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/metrics >"$OUT/metrics.txt"
python3 - "$OUT/metrics.txt" <<'PYMETRICS'
import re, sys
text = open(sys.argv[1]).read()
assert re.search(r'^triglav_sessions_active 1$', text, re.M)
assert re.search(r'^triglav_uplinks_active 2$', text, re.M)
assert int(re.search(r'^triglav_exit_forwarded_packets_total ([0-9]+)$', text, re.M)[1]) > 0
PYMETRICS

ip netns exec tgproof-client "$BIN/triglav" connect "$key" --interface up1 --interface up2 --socks 11080 --http-proxy 18080 --metrics-addr 127.0.0.1:9093 --strategy wrr >"$OUT/proxy-client.log" 2>&1 &
proxy_pid=$!; pids+=($proxy_pid)
wait_client_paths tgproof-client 9093 2 "$OUT/proxy-ready.json"
ip netns exec tgproof-client python3 "$ROOT/validation/proxies.py" >"$OUT/proxies.json"
# Allow the authenticated EOFs to finish the server's per-stream accounting.
sleep .2
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/proxy-status.json"
python3 - "$OUT" <<'PYPROXY'
import json, pathlib, sys
status = json.loads((pathlib.Path(sys.argv[1])/'proxy-status.json').read_text())
assert set(status['proxy_connections_by_source']) == {'10.73.1.1', '10.74.1.1'}, status
assert sum(status['proxy_connections_by_source'].values()) == 16, status
PYPROXY
kill -TERM "$proxy_pid"; wait "$proxy_pid"
session=$(python3 "$ROOT/validation/packets.py" session "$OUT/ingress1.pcap")
ip netns exec tgproof-client env TRIGLAV_TEST_KEY="$key" TRIGLAV_TEST_SESSION="$session" "$probe" production_rejects_session_hijack --ignored --exact --nocapture >"$OUT/admission.log" 2>&1
ip netns exec tgproof-client "$BIN/triglav" tun "$key" --interface up1 --interface up2 --tun-name tg1 --metrics-addr 127.0.0.1:9092 --aggregate >"$OUT/client2.log" 2>&1 &
client2=$!; pids+=($client2)
for i in {1..100}; do
  address2=$(ip -n tgproof-client -o -4 address show dev tg1 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)
  [[ -n "$address2" ]] && break
  sleep .1
done
[[ -n "$address2" ]]
wait_client_paths tgproof-client 9092 2 "$OUT/client2-ready.json"
ip -n tgproof-client rule add from "$address2" table 200
ip -n tgproof-client route add 198.18.0.0/24 dev tg1 table 200
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 262144 4 >"$OUT/client1-concurrent.json" &
first=$!; pids+=($first)
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 32769 4 tcp "$address2" >"$OUT/client2-concurrent.json"
address6=$(ip -n tgproof-client -o -6 address show dev tg1 scope global | awk '{print $4}' | cut -d/ -f1)
ip -n tgproof-client -6 rule add from "$address6" table 200
ip -n tgproof-client -6 route add 2001:db8:85::/64 dev tg1 table 200
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 32769 2 tcp6 "$address6" >"$OUT/client2-concurrent6.json"
wait "$first"
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/multiclient-status.json"
python3 - "$OUT" <<'PYMULTI'
import json, pathlib, sys
status = json.loads((pathlib.Path(sys.argv[1])/'multiclient-status.json').read_text())
assert status['sessions'] == 2, status
assert len(status['paths']) == 4, status
assert len({p['ipv4'] for p in status['paths']}) == 2, status
assert len({p['ipv6'] for p in status['paths']}) == 2 and all(p['ipv6'] for p in status['paths']), status
assert all(p['received'] > 0 and p['sent'] > 0 for p in status['paths']), status
PYMULTI
kill -TERM "$client2"; wait "$client2"
# A fresh single-uplink client provides a like-for-like rate-capped control.
ip netns exec tgproof-client "$BIN/triglav" tun "$key" --interface up1 --tun-name tgbaseline --metrics-addr 127.0.0.1:9094 --aggregate >"$OUT/baseline-client.log" 2>&1 &
baseline_pid=$!; pids+=($baseline_pid)
baseline_address=''
for i in {1..100}; do
  baseline_address=$(ip -n tgproof-client -o -4 address show dev tgbaseline 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)
  [[ -n "$baseline_address" ]] && break
  sleep .1
done
[[ -n "$baseline_address" ]]
ip -n tgproof-client rule add from "$baseline_address" table 201
ip -n tgproof-client route add 198.18.0.0/24 dev tgbaseline table 201
ip netns exec tgproof-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/benchmark-status.json"
python3 "$ROOT/validation/benchmark.py" "$baseline_address" >"$OUT/benchmark.json"
kill -TERM "$baseline_pid"; wait "$baseline_pid"
kill -TERM "$client_pid"; wait "$client_pid"
if ip -n tgproof-client link show tg0 2>/dev/null; then echo 'TUN not removed after shutdown' >&2; exit 1; fi
# Full dual-stack routing must preserve the original policy rules when removed.
ip -n tgproof-client -j rule show >"$OUT/rules-before-full.json"
ip -n tgproof-client -6 -j rule show >"$OUT/rules6-before-full.json"
ip -n tgproof-client -j route show table main >"$OUT/main-routes-before-full.json"
ip netns exec tgproof-client "$BIN/triglav" tun "$key" --interface up1 --interface up2 --tun-name tgfull --full-tunnel --aggregate >"$OUT/full-client.log" 2>&1 &
full_pid=$!; pids+=($full_pid)
for i in {1..100}; do
  if ip -n tgproof-client route show table 100 2>/dev/null | grep -q tgfull; then break; fi
  sleep .1
done
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 131072 1 >"$OUT/full-tunnel.json"
ip netns exec tgproof-client python3 "$ROOT/validation/traffic.py" 131072 1 tcp6 >"$OUT/full-tunnel6.json"
kill -TERM "$full_pid"; wait "$full_pid"
ip -n tgproof-client -j rule show >"$OUT/rules-after-full.json"
cmp "$OUT/rules-before-full.json" "$OUT/rules-after-full.json"
ip -n tgproof-client -6 -j rule show >"$OUT/rules6-after-full.json"
cmp "$OUT/rules6-before-full.json" "$OUT/rules6-after-full.json"
[[ -z $(ip -n tgproof-client -6 route show table 100 2>/dev/null) ]]
ip -n tgproof-client -j route show table main >"$OUT/main-routes-after-full.json"
cmp "$OUT/main-routes-before-full.json" "$OUT/main-routes-after-full.json"
[[ -z $(ip -n tgproof-client route show table 100 2>/dev/null) ]]
# An occupied routing table must survive a rejected full-tunnel startup.
ip -n tgproof-client route add default via 10.71.1.1 dev up1 table 100
ip -n tgproof-client rule add priority 200 table 100
ip -n tgproof-client -j rule show >"$OUT/rules-before-collision.json"
ip -n tgproof-client -j route show table 100 >"$OUT/table-before-collision.json"
set +e
ip netns exec tgproof-client timeout 5 "$BIN/triglav" tun "$key" --interface up1 --interface up2 --tun-name tgcollision --full-tunnel >"$OUT/collision-client.log" 2>&1
collision_status=$?
set -e
[[ $collision_status == 1 ]]
ip -n tgproof-client -j rule show >"$OUT/rules-after-collision.json"
ip -n tgproof-client -j route show table 100 >"$OUT/table-after-collision.json"
cmp "$OUT/rules-before-collision.json" "$OUT/rules-after-collision.json"
cmp "$OUT/table-before-collision.json" "$OUT/table-after-collision.json"
ip -n tgproof-client rule del priority 200 table 100
ip -n tgproof-client route del default table 100
ip -n tgproof-client route add 203.0.113.0/24 via 10.71.1.1 dev up1 table 100
ip -n tgproof-client -j route show table 100 >"$OUT/nondefault-before-collision.json"
set +e
ip netns exec tgproof-client timeout 5 "$BIN/triglav" tun "$key" --interface up1 --interface up2 --tun-name tgnondefault --full-tunnel >"$OUT/nondefault-collision.log" 2>&1
nondefault_status=$?
set -e
[[ $nondefault_status == 1 ]]
ip -n tgproof-client -j route show table 100 >"$OUT/nondefault-after-collision.json"
cmp "$OUT/nondefault-before-collision.json" "$OUT/nondefault-after-collision.json"
ip -n tgproof-client route del 203.0.113.0/24 table 100
ip -n tgproof-client -6 route add 2001:db8:999::/64 dev up1 table 100
ip -n tgproof-client -6 -j route show table 100 >"$OUT/ipv6-before-collision.json"
set +e
ip netns exec tgproof-client timeout 5 "$BIN/triglav" tun "$key" --interface up1 --interface up2 --tun-name tg6collision --full-tunnel >"$OUT/ipv6-collision.log" 2>&1
ipv6_collision_status=$?
set -e
[[ $ipv6_collision_status == 1 ]]
ip -n tgproof-client -6 -j route show table 100 >"$OUT/ipv6-after-collision.json"
cmp "$OUT/ipv6-before-collision.json" "$OUT/ipv6-after-collision.json"
[[ -z $(ip -n tgproof-client route show table 100 2>/dev/null) ]]
ip -n tgproof-client -6 route del 2001:db8:999::/64 table 100
# Full tunneling must refuse an IPv4-only exit rather than leak IPv6 traffic.
ip netns exec tgproof-exit "$BIN/triglav-server" --listen 10.99.0.1:7444 --generate-key --ipv4-only --tun-name tg-v4-only --tun-ipv4 10.0.86.254 --metrics 127.0.0.1:9095 >"$OUT/ipv4-only-server.log" 2>&1 &
v4_server=$!; pids+=($v4_server)
v4_key=''
for i in {1..100}; do
  v4_key=$(sed -n 's/^\(tg1[A-Za-z0-9_-]*\).*/\1/p' "$OUT/ipv4-only-server.log" | head -1)
  [[ -n "$v4_key" ]] && break
  sleep .1
done
[[ -n "$v4_key" ]]
ip -n tgproof-client -j rule show >"$OUT/before-ipv4-only-rules.json"
ip -n tgproof-client -6 -j rule show >"$OUT/before-ipv4-only-rules6.json"
set +e
ip netns exec tgproof-client timeout 5 "$BIN/triglav" tun "$v4_key" --interface up1 --tun-name tg-noipv6 --full-tunnel >"$OUT/ipv4-only-client.log" 2>&1
v4_status=$?
set -e
[[ $v4_status == 1 ]]
grep -q 'Full tunnel requires an IPv6-capable exit' "$OUT/ipv4-only-client.log"
ip -n tgproof-client -j rule show >"$OUT/after-ipv4-only-rules.json"
ip -n tgproof-client -6 -j rule show >"$OUT/after-ipv4-only-rules6.json"
cmp "$OUT/before-ipv4-only-rules.json" "$OUT/after-ipv4-only-rules.json"
cmp "$OUT/before-ipv4-only-rules6.json" "$OUT/after-ipv4-only-rules6.json"
python3 - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); status = json.loads((out/'status.json').read_text())
assert status['sessions'] == 1, status
assert status['authentication_required'], status
assert len(status['paths']) == 2, status
assert {p['address'].split(':')[0] for p in status['paths']} == {'10.73.1.1', '10.74.1.1'}, status
assert all(p['received'] > 0 and p['sent'] > 0 for p in status['paths']), status
records = [json.loads(s) for s in (out/'destination.jsonl').read_text().splitlines()]
assert records and {r['peer'] for r in records} == {'198.18.0.1', '2001:db8:85::1'}, records
print(json.dumps({'result': 'PASS', 'destination_records': len(records), 'status': status}, indent=2))
PY
