#!/bin/bash
# Real exit restart, endpoint movement and all-path outage, inside a disposable lab.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${TRIGLAV_EVIDENCE_DIR:?}/recovery
BIN=${TRIGLAV_BIN_DIR:-/build/release}
mkdir -p "$OUT"
[[ ! -e "$OUT/server.log" ]]
names=(tgrecov-client tgrecov-r1 tgrecov-r2 tgrecov-exit tgrecov-target)
created=() pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for ns in "${created[@]}"; do ip netns del "$ns" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 124' TERM
for ns in "${names[@]}"; do
  if ip netns list | cut -d' ' -f1 | grep -qx "$ns"; then echo "Namespace already exists: $ns" >&2; exit 1; fi
done
for ns in "${names[@]}"; do ip netns add "$ns"; created+=("$ns"); ip -n "$ns" link set lo up; done
link() {
  ip link add "$2" type veth peer name "$5"
  ip link set "$2" netns "$1"; ip link set "$5" netns "$4"
  ip -n "$1" addr add "$3" dev "$2"; ip -n "$4" addr add "$6" dev "$5"
  ip -n "$1" link set "$2" up; ip -n "$4" link set "$5" up
}
link tgrecov-client up1 10.81.1.2/24 tgrecov-r1 lan1 10.81.1.1/24
link tgrecov-client up2 10.82.1.2/24 tgrecov-r2 lan2 10.82.1.1/24
link tgrecov-r1 wan1 10.83.1.1/24 tgrecov-exit in1 10.83.1.2/24
link tgrecov-r2 wan2 10.84.1.1/24 tgrecov-exit in2 10.84.1.2/24
link tgrecov-exit out0 198.18.0.1/24 tgrecov-target dest0 198.18.0.2/24
ip -n tgrecov-exit addr add 10.89.0.1/32 dev lo
ip -n tgrecov-exit -6 addr add 2001:db8:85::1/64 dev out0 nodad
ip -n tgrecov-target -6 addr add 2001:db8:85::2/64 dev dest0 nodad
for ns in "${names[@]}"; do
  ip netns exec "$ns" sysctl -qw net.ipv4.conf.all.rp_filter=0 net.ipv4.conf.default.rp_filter=0
  ip netns exec "$ns" sh -c 'for f in /proc/sys/net/ipv4/conf/*/rp_filter; do echo 0 > "$f"; done'
done
for n in 1 2; do
  ip netns exec "tgrecov-r$n" sysctl -qw net.ipv4.ip_forward=1
  ip -n "tgrecov-r$n" route add 10.89.0.1/32 via "10.8$((n+2)).1.2"
  ip netns exec "tgrecov-r$n" iptables -t nat -A POSTROUTING -o "wan$n" -j MASQUERADE
  ip -n tgrecov-client rule add from "10.8$n.1.2" table "$((100+n))"
  ip -n tgrecov-client rule add from "10.8$n.1.3" table "$((100+n))"
  ip -n tgrecov-client route add "10.8$n.1.0/24" dev "up$n" table "$((100+n))"
  ip -n tgrecov-client route add 10.89.0.1/32 via "10.8$n.1.1" dev "up$n" table "$((100+n))"
done
ip -n tgrecov-client route add 10.89.0.1/32 via 10.81.1.1 dev up1
ip -n tgrecov-client route add default via 10.81.1.1 dev up1
ip netns exec tgrecov-exit sysctl -qw net.ipv4.ip_forward=1 net.ipv6.conf.all.forwarding=1
ip netns exec tgrecov-exit iptables -t nat -A POSTROUTING -s 10.0.85.0/24 -o out0 -j MASQUERADE
ip netns exec tgrecov-exit ip6tables -t nat -A POSTROUTING -s fd00:7472:6967::/64 -o out0 -j MASQUERADE
export TRIGLAV_ACCESS_TOKEN
TRIGLAV_ACCESS_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
ip netns exec tgrecov-target python3 "$ROOT/validation/endpoint.py" "$OUT/destination.jsonl" >"$OUT/endpoint.log" 2>&1 & pids+=($!)
start_exit() {
  ip netns exec tgrecov-exit "$BIN/triglav-server" --listen 10.89.0.1:7443 --key "$OUT/exit.key" --generate-key --metrics 127.0.0.1:9090 >>"$OUT/server.log" 2>&1 &
  server=$!; pids+=("$server")
  for i in {1..100}; do
    if ip netns exec tgrecov-exit curl --max-time 1 --fail --silent http://127.0.0.1:9090/status >"$OUT/exit-current.json"; then return; fi
    sleep .1
  done
  cat "$OUT/server.log" >&2; exit 1
}
client_status() { ip netns exec tgrecov-client curl --max-time 2 --fail --silent http://127.0.0.1:9091/status; }
exit_status() { ip netns exec tgrecov-exit curl --max-time 2 --fail --silent http://127.0.0.1:9090/status; }
progress() { cat "$OUT/progress.txt" 2>/dev/null || echo 0; }
wait_progress() {
  local before=$1
  for i in {1..400}; do
    kill -0 "$client"; kill -0 "$flow"
    if [[ $(progress) -gt $before ]]; then return; fi
    sleep .15
  done
  cat "$OUT/client.log" "$OUT/flow.err" >&2; exit 1
}
start_exit
key=$(sed -n 's/^\(tg1[A-Za-z0-9_-]*\).*/\1/p' "$OUT/server.log" | head -1)
[[ -n "$key" ]]
kill -TERM "$server"; wait "$server"
ip netns exec tgrecov-client "$BIN/triglav" --log-level debug tun "$key" --interface up1 --interface up2 --aggregate --route 198.18.0.0/24 --route 2001:db8:85::/64 >"$OUT/client.log" 2>&1 &
client=$!; pids+=("$client")
sleep 1
kill -0 "$client" # An unavailable exit must not terminate client startup.
start_exit
for i in {1..300}; do
  if ip -n tgrecov-client route show 198.18.0.0/24 | grep -q tg0; then break; fi
  kill -0 "$client"; sleep .1
done
ip netns exec tgrecov-client python3 "$ROOT/validation/traffic.py" 65536 1 >"$OUT/warmup.json"
ip netns exec tgrecov-client python3 "$ROOT/validation/traffic.py" 65536 1 tcp6 >"$OUT/warmup6.json"
client_status >"$OUT/client-before.json"
ip netns exec tgrecov-client python3 "$ROOT/validation/recovery_flow.py" "$OUT" >"$OUT/flow.json" 2>"$OUT/flow.err" &
flow=$!; pids+=("$flow")
wait_progress 0

# Keep all paths blackholed longer than the stale-peer threshold. Sample bounded
# pending occupancy while locally generated UDP saturates the sender.
for n in 1 2; do ip netns exec "tgrecov-r$n" iptables -I FORWARD -j DROP; done
ip netns exec tgrecov-client python3 - "$OUT" <<'PY'
import json, pathlib, socket, sys, time, urllib.request
out = pathlib.Path(sys.argv[1]); samples = []; deadline = time.monotonic() + 20
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    for batch in range(40):
        for _ in range(256): sock.sendto(b'x' * 1024, ('198.18.0.2', 19999))
        with urllib.request.urlopen('http://127.0.0.1:9091/status', timeout=2) as response:
            samples.append(json.load(response)['pending'])
        time.sleep(.025)
    while time.monotonic() < deadline:
        with urllib.request.urlopen('http://127.0.0.1:9091/status', timeout=2) as response:
            samples.append(json.load(response)['pending'])
        time.sleep(.2)
assert samples and max(s['packets'] for s in samples) > 0, samples
assert all(s['packets'] <= s['max_packets'] and s['bytes'] <= s['max_bytes'] for s in samples), samples
out.joinpath('outage-pending.json').write_text(json.dumps(samples))
PY
before=$(progress)
for n in 1 2; do ip netns exec "tgrecov-r$n" iptables -D FORWARD -j DROP; done
wait_progress "$before"
client_status >"$OUT/client-after-outage.json"

# Restart the exit with the same key while the same inner TCP stream is open.
kill -TERM "$server"; wait "$server"
sleep 2
before=$(progress)
start_exit
wait_progress "$before"
client_status >"$OUT/client-after-restart.json"

# Change both observed NAT addresses without changing client sockets or keys.
for n in 1 2; do
  ip -n "tgrecov-r$n" addr add "10.8$((n+2)).1.3/24" dev "wan$n"
  ip netns exec "tgrecov-r$n" iptables -t nat -I POSTROUTING -p udp --dport 7443 -j SNAT --to-source "10.8$((n+2)).1.3"
  ip netns exec "tgrecov-r$n" conntrack -D -p udp --dport 7443 >"$OUT/rebind$n.log" 2>&1 || true
done
before=$(progress); wait_progress "$before"
exit_status >"$OUT/exit-after-rebind.json"

# Twenty path replacements exceed the former 16-path leak limit. Confirm a new
# key generation after every pair of interface-address changes.
for round in {1..10}; do
  exit_status >"$OUT/generation-before.json"
  before=$(python3 -c 'import json,sys; print(sum(p["key_generations"] for p in json.load(open(sys.argv[1]))["paths"]))' "$OUT/generation-before.json")
  old=$((2 + (round + 1) % 2)); new=$((2 + round % 2))
  for n in 1 2; do
    ip -n tgrecov-client addr del "10.8$n.1.$old/24" dev "up$n"
    ip -n tgrecov-client addr add "10.8$n.1.$new/24" dev "up$n"
    ip -n tgrecov-client route replace "10.8$n.1.0/24" dev "up$n" table "$((100+n))"
    ip -n tgrecov-client route replace 10.89.0.1/32 via "10.8$n.1.1" dev "up$n" table "$((100+n))"
  done
  ip -n tgrecov-client route replace 10.89.0.1/32 via 10.81.1.1 dev up1
  ip -n tgrecov-client route replace default via 10.81.1.1 dev up1
  for i in {1..300}; do
    exit_status >"$OUT/generation-after.json"
    after=$(python3 -c 'import json,sys; print(sum(p["key_generations"] for p in json.load(open(sys.argv[1]))["paths"]))' "$OUT/generation-after.json")
    if [[ $after -ge $((before+2)) ]]; then break; fi
    kill -0 "$client"; sleep .1
  done
  [[ $after -ge $((before+2)) ]]
  [[ $(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["paths"]))' "$OUT/generation-after.json") == 2 ]]
  before=$(progress); wait_progress "$before"
done
touch "$OUT/stop-flow"; wait "$flow"
client_status >"$OUT/client-after.json"
exit_status >"$OUT/exit-after.json"
ip netns exec tgrecov-client python3 "$ROOT/validation/traffic.py" 65536 1 tcp6 >"$OUT/after6.json"
kill -TERM "$client"; wait "$client"
[[ -z $(ip -n tgrecov-client route show 198.18.0.0/24) ]]
python3 - "$OUT" <<'PY'
import collections, hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1]); load = lambda name: json.loads((out/name).read_text())
clients = [load(name) for name in ['client-before.json', 'client-after-outage.json', 'client-after-restart.json', 'client-after.json']]
assert len({c['process_id'] for c in clients}) == 1
assert len({c['session_id'] for c in clients}) == 1
assert all(len(c['uplinks']) == 2 for c in clients)
assert len({tuple(sorted(p['id'] for p in c['uplinks'])) for c in clients}) == 1
flow = load('flow.json'); assert flow['connections'] == 1 and flow['frames'] >= 10
expected = collections.Counter((r['bytes_each_direction'], r['sha256'], 'tcp') for r in flow['results'])
for name in ['warmup.json', 'warmup6.json', 'after6.json']:
    report = load(name)
    expected.update((s['bytes_each_direction'], s['sha256'], report['protocol']) for s in report['streams'])
records = [json.loads(line) for line in (out/'destination.jsonl').read_text().splitlines()]
assert collections.Counter((r['size'], r['sha256'], r['protocol']) for r in records) == expected
paths = load('exit-after-rebind.json')['paths']
assert {p['address'].split(':')[0] for p in paths} == {'10.83.1.3', '10.84.1.3'}, paths
summary = {'result': 'PASS', 'same_client_process': True, 'same_session': True,
           'one_tcp_connection': True, 'frames': flow['frames'], 'address_replacement_rounds': 10,
           'maximum_frame_round_trip_s': max(r['elapsed_s'] for r in flow['results']),
           'destination_records': len(records), 'bytes_each_direction': sum(r['size'] for r in records),
           'outage_seconds': 20, 'exit_restarted': True, 'nat_addresses_rebound': 2,
           'final_paths': load('exit-after.json')['paths']}
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
PY
