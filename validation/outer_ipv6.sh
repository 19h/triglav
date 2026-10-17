#!/bin/bash
# Complement the NATed IPv4 lane with IPv6-only outer paths and full routing.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/validation/readiness.sh"
OUT=${TRIGLAV_EVIDENCE_DIR:?}/outer-ipv6
BIN=${TRIGLAV_BIN_DIR:-/build/debug}
mkdir "$OUT"
names=(tg6-client tg6-r1 tg6-r2 tg6-exit tg6-target)
created=(); pids=(); captures=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for ns in "${created[@]}"; do ip netns del "$ns" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 124' TERM
for ns in "${names[@]}"; do
  if ip netns list | cut -d' ' -f1 | grep -qx "$ns"; then echo "Namespace already exists: $ns" >&2; exit 1; fi
done
for ns in "${names[@]}"; do
  ip netns add "$ns"; created+=("$ns"); ip -n "$ns" link set lo up
done
link6() {
  local a=$1 an=$2 ai=$3 b=$4 bn=$5 bi=$6
  ip link add "$an" type veth peer name "$bn"
  ip link set "$an" netns "$a"; ip link set "$bn" netns "$b"
  ip -n "$a" link set "$an" up; ip -n "$b" link set "$bn" up
  ip -n "$a" -6 address add "$ai" dev "$an" nodad
  ip -n "$b" -6 address add "$bi" dev "$bn" nodad
}
link6 tg6-client up1 2001:db8:71::2/64 tg6-r1 lan1 2001:db8:71::1/64
link6 tg6-client up2 2001:db8:72::2/64 tg6-r2 lan2 2001:db8:72::1/64
link6 tg6-r1 wan1 2001:db8:73::1/64 tg6-exit in1 2001:db8:73::2/64
link6 tg6-r2 wan2 2001:db8:74::1/64 tg6-exit in2 2001:db8:74::2/64
link6 tg6-exit out0 2001:db8:85::1/64 tg6-target dest0 2001:db8:85::2/64
ip -n tg6-exit address add 198.18.0.1/24 dev out0
ip -n tg6-target address add 198.18.0.2/24 dev dest0
ip -n tg6-exit -6 address add 2001:db8:99::1/128 dev lo nodad
for n in 1 2; do
  ip netns exec "tg6-r$n" sysctl -qw net.ipv6.conf.all.forwarding=1
  ip -n "tg6-r$n" -6 route add 2001:db8:99::1/128 via "2001:db8:7$((n+2))::2"
  ip -n tg6-exit -6 route add "2001:db8:7$n::/64" via "2001:db8:7$((n+2))::1"
  ip -n tg6-client -6 rule add from "2001:db8:7$n::2" table "$((100+n))"
  ip -n tg6-client -6 route add "2001:db8:7$n::/64" dev "up$n" table "$((100+n))"
  ip -n tg6-client -6 route add 2001:db8:99::1/128 via "2001:db8:7$n::1" dev "up$n" table "$((100+n))"
done
ip -n tg6-client -6 route add default via 2001:db8:71::1 dev up1
ip netns exec tg6-exit sysctl -qw net.ipv4.ip_forward=1 net.ipv6.conf.all.forwarding=1
ip netns exec tg6-exit iptables -t nat -A POSTROUTING -s 10.0.85.0/24 -o out0 -j MASQUERADE
ip netns exec tg6-exit ip6tables -t nat -A POSTROUTING -s fd00:7472:6967::/64 -o out0 -j MASQUERADE
if ip netns exec tg6-client ping -6 -n -c1 -W1 2001:db8:85::2 >"$OUT/no-bypass.txt" 2>&1; then
  echo 'Direct IPv6 bypass unexpectedly works' >&2; exit 1
fi
ip -n tg6-client -j -6 rule show >"$OUT/rules6-before.json"
ip -n tg6-client -j -4 rule show >"$OUT/rules4-before.json"
ip -n tg6-client -j -6 route show table main >"$OUT/routes6-before.json"
export TRIGLAV_ACCESS_TOKEN
TRIGLAV_ACCESS_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
ip netns exec tg6-target python3 "$ROOT/validation/endpoint.py" "$OUT/destination.jsonl" >"$OUT/endpoint.log" 2>&1 & pids+=($!)
ip netns exec tg6-exit "$BIN/triglav" --no-color server --listen '[2001:db8:99::1]:7443' --generate-key >"$OUT/server.log" 2>&1 & pids+=($!)
key=''
for i in {1..100}; do
  key=$(sed -n 's/^\(tg1[A-Za-z0-9_-]*\).*/\1/p' "$OUT/server.log" | head -1)
  [[ -n "$key" ]] && break
  sleep .1
done
[[ -n "$key" ]]
for n in 1 2; do
  ip netns exec tg6-exit tcpdump -U -n -i "in$n" -w "$OUT/ingress$n.pcap" 'ip6 and udp dst port 7443' >"$OUT/capture$n.log" 2>&1 &
  captures+=($!); pids+=($!)
done
ip netns exec tg6-client "$BIN/triglav" --log-level debug tun "$key" --interface up1 --interface up2 --tun-name tg6 --full-tunnel --aggregate --strategy wrr >"$OUT/client.log" 2>&1 & client=$!; pids+=($client)
for i in {1..100}; do
  ip -n tg6-client -6 route show table 100 2>/dev/null | grep -q tg6 && break
  sleep .1
done
wait_client_paths tg6-client 9091 2 "$OUT/client-ready.json"
# Optional dynamic-MTU acceptance profile; the ordinary IPv6 lane is unchanged.
if [[ -n ${TRIGLAV_TEST_MTU:-} ]]; then
  [[ $TRIGLAV_TEST_MTU =~ ^[0-9]+$ ]]
  expected_mtu=$((TRIGLAV_TEST_MTU - 141))
  [[ $expected_mtu -ge 1280 && $expected_mtu -le 1340 ]]
  ip netns exec tg6-client python3 "$ROOT/validation/recovery_flow.py" "$OUT" >"$OUT/mtu-flow.json" 2>"$OUT/mtu-flow.err" &
  mtu_flow=$!; pids+=("$mtu_flow")
  mkdir "$OUT/mtu6"
  ip netns exec tg6-client python3 "$ROOT/validation/recovery_flow.py" "$OUT/mtu6" tcp6 >"$OUT/mtu6/flow.json" 2>"$OUT/mtu6/flow.err" &
  mtu_flow6=$!; pids+=("$mtu_flow6")
  for i in {1..100}; do [[ -s "$OUT/progress.txt" ]] && break; sleep .1; done
  [[ -s "$OUT/progress.txt" ]]
  for i in {1..100}; do [[ -s "$OUT/mtu6/progress.txt" ]] && break; sleep .1; done
  [[ -s "$OUT/mtu6/progress.txt" ]]
  cp "$OUT/progress.txt" "$OUT/mtu-frames4-before.txt"
  cp "$OUT/mtu6/progress.txt" "$OUT/mtu-frames6-before.txt"
  for n in 1 2; do
    ip -n tg6-client link set "up$n" mtu "$TRIGLAV_TEST_MTU"
    ip -n "tg6-r$n" link set "lan$n" mtu "$TRIGLAV_TEST_MTU"
    ip -n "tg6-r$n" link set "wan$n" mtu "$TRIGLAV_TEST_MTU"
    ip -n tg6-exit link set "in$n" mtu "$TRIGLAV_TEST_MTU"
  done
  for i in {1..200}; do
    actual=$(ip -n tg6-client -j link show dev tg6 | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["mtu"])')
    if [[ $actual == "$expected_mtu" ]]; then break; fi
    sleep .1
  done
  [[ $actual == "$expected_mtu" ]]
  ip netns exec tg6-client curl --fail --silent http://127.0.0.1:9091/status >"$OUT/mtu-client.json"
  ip netns exec tg6-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/mtu-exit.json"
  sleep 1 # Retire datagrams queued before the MTU transition.
  for n in 1 2; do
    ip netns exec tg6-exit tcpdump -U -n -i "in$n" -w "$OUT/mtu-ingress$n.pcap" 'ip6 and dst host 2001:db8:99::1' >"$OUT/mtu-capture$n.log" 2>&1 &
    pids+=($!); captures+=($!)
  done
fi
ip netns exec tg6-client python3 "$ROOT/validation/traffic.py" 262144 4 tcp >"$OUT/tcp4.json"
ip netns exec tg6-client python3 "$ROOT/validation/traffic.py" 262144 4 tcp6 >"$OUT/tcp6.json"
ip netns exec tg6-client python3 "$ROOT/validation/traffic.py" 1200 4 udp >"$OUT/udp4.json"
ip netns exec tg6-client python3 "$ROOT/validation/traffic.py" 1200 4 udp6 >"$OUT/udp6.json"
ip netns exec tg6-exit curl --fail --silent http://127.0.0.1:9090/status >"$OUT/status.json"
ip netns exec tg6-client "$BIN/triglav" connect "$key" --interface up1 --interface up2 --socks 11080 --http-proxy 18080 --metrics-addr 127.0.0.1:9093 --strategy wrr >"$OUT/proxy-client.log" 2>&1 & proxy=$!; pids+=($proxy)
wait_client_paths tg6-client 9093 2 "$OUT/proxy-ready.json"
ip netns exec tg6-client python3 "$ROOT/validation/proxies.py" >"$OUT/proxies.json"
if [[ -n ${TRIGLAV_TEST_MTU:-} ]]; then
  touch "$OUT/stop-flow" "$OUT/mtu6/stop-flow"
  wait "$mtu_flow"; wait "$mtu_flow6"
fi
kill -TERM "$proxy"; wait "$proxy"
kill -TERM "$client"; wait "$client"
for pid in "${captures[@]}"; do kill -INT "$pid"; wait "$pid" || true; done
python3 "$ROOT/validation/packets.py" verify "$OUT/ingress1.pcap" "$OUT/ingress2.pcap" >"$OUT/packet-proof.json"
if [[ -n ${TRIGLAV_TEST_MTU:-} ]]; then
  for n in 1 2; do
    tcpdump -n -r "$OUT/mtu-ingress$n.pcap" "ip6[6] == 44 or (udp dst port 7443 and ip6[4:2] > $((TRIGLAV_TEST_MTU-40)))" >"$OUT/mtu-oversized$n.txt" 2>"$OUT/mtu-audit$n.log"
    [[ ! -s "$OUT/mtu-oversized$n.txt" ]]
  done
  python3 "$ROOT/validation/packets.py" verify "$OUT/mtu-ingress1.pcap" "$OUT/mtu-ingress2.pcap" >"$OUT/mtu-packet-proof.json"
fi

ip -n tg6-client -j -6 rule show >"$OUT/rules6-after.json"
ip -n tg6-client -j -4 rule show >"$OUT/rules4-after.json"
ip -n tg6-client -j -6 route show table main >"$OUT/routes6-after.json"
python3 - "$OUT" <<'PY'
import collections, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
load = lambda name: json.loads((out/name).read_text())
status = load('status.json')
assert status['sessions'] == 1 and len(status['paths']) == 2
assert {p['address'].split(']')[0] for p in status['paths']} == {'[2001:db8:71::2', '[2001:db8:72::2'}
assert all(p['received'] > 0 and p['sent'] > 0 for p in status['paths'])
expected = collections.Counter()
for name in ['tcp4', 'tcp6', 'udp4', 'udp6']:
    result = load(name+'.json')
    expected.update((r['bytes_each_direction'], r['sha256'], result['protocol']) for r in result['streams'])
proxies = load('proxies.json')
assert len(proxies['transfers']) == 16 and proxies['failed_targets_rejected'] == 4
expected.update((r['bytes_each_direction'], r['sha256'], r['protocol']) for r in proxies['transfers'])
if (out/'mtu-flow.json').exists():
    flow = load('mtu-flow.json')
    assert flow['connections'] == 1 and flow['frames'] >= 2
    assert flow['frames'] > int((out/'mtu-frames4-before.txt').read_text())
    expected.update((r['bytes_each_direction'], r['sha256'], 'tcp') for r in flow['results'])
    flow6 = load('mtu6/flow.json')
    assert flow6['connections'] == 1 and flow6['frames'] >= 2
    assert flow6['frames'] > int((out/'mtu-frames6-before.txt').read_text())
    expected.update((r['bytes_each_direction'], r['sha256'], 'tcp6') for r in flow6['results'])
    client, server = load('mtu-client.json'), load('mtu-exit.json')
    import os
    expected_mtu = int(os.environ['TRIGLAV_TEST_MTU']) - 141
    assert client['tunnel']['negotiated_mtu'] == expected_mtu
    assert server['tun_mtu'] == expected_mtu
    assert all(p['encrypted_data_packets'] > 0 for p in load('mtu-packet-proof.json'))
records = [json.loads(line) for line in (out/'destination.jsonl').read_text().splitlines()]
assert {r['peer'] for r in records} == {'198.18.0.1', '2001:db8:85::1'}
assert expected == collections.Counter((r['size'], r['sha256'], r['protocol']) for r in records)
for name in ['rules6', 'rules4', 'routes6']:
    assert (out/(name+'-before.json')).read_bytes() == (out/(name+'-after.json')).read_bytes(), name
summary = {'result': 'PASS', 'outer_family': 'IPv6', 'inner_families': ['IPv4', 'IPv6'],
           'destination_records': len(records), 'bytes_each_direction': sum(r['size'] for r in records),
           'full_tunnel_and_restoration': True, 'paths': status['paths']}
if (out/'mtu-flow.json').exists():
    summary['dynamic_mtu'] = {'outer_mtu': int(os.environ['TRIGLAV_TEST_MTU']),
                              'inner_mtu': expected_mtu, 'outer_fragments': 0,
                              'tcp4_connections': 1, 'tcp6_connections': 1, 'frames4': flow['frames'], 'frames6': flow6['frames']}
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary))
PY
