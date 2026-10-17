#!/usr/bin/env python3
"""Fail-closed summary of a completed validation directory, without shell logs as evidence substitutes."""
import collections
import hashlib
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
load = lambda name: json.loads((out / name).read_text())
results = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', (out/'tests.log').read_text())
assert results, 'no Rust test results'
assert all(int(failed) == 0 for _, failed, _ in results)
assert 'test result: FAILED' not in (out/'tests.log').read_text()
assert 'test result: ok. 1 passed; 0 failed; 0 ignored;' in (out/'admission.log').read_text()
status = load('status.json')
assert status['sessions'] == 1 and status['authentication_required']
assert len(status['paths']) == 2
assert all(p['received'] > 0 and p['sent'] > 0 for p in status['paths'])
assert all(p['received'] > 0 and p['sent'] > 0 for p in load('single-flow-status.json')['paths'])
expected_payloads = collections.Counter()
for name in ['tcp.json', 'concurrent.json', 'udp.json', 'fragmented-udp.json', 'impaired.json', 'after-attack.json', 'client1-concurrent.json', 'client2-concurrent.json', 'full-tunnel.json', 'link-down-transfer.json', 'link-restored-transfer.json', 'tcp6.json', 'udp6.json', 'full-tunnel6.json', 'client2-concurrent6.json']:
    report = load(name)
    assert report['streams'] and all(s['bytes_each_direction'] > 0 and len(s['sha256']) == 64 for s in report['streams'])
    expected_payloads.update((s['bytes_each_direction'], s['sha256'], report['protocol']) for s in report['streams'])
failover = load('failover.json')
assert failover['frames'] == 120 and failover['connections'] == 1
expected_payloads.update((16384, r['sha256'], 'tcp') for r in failover['results'])
proxies = load('proxies.json')
assert len(proxies['transfers']) == 16 and proxies['failed_targets_rejected'] == 4
expected_payloads.update((r['bytes_each_direction'], r['sha256'], r['protocol']) for r in proxies['transfers'])
assert set(load('proxy-status.json')['proxy_connections_by_source']) == {'10.73.1.1', '10.74.1.1'}
assert load('multiclient-status.json')['sessions'] == 2
assert re.search(r'^triglav_sessions_active 1$', (out/'metrics.txt').read_text(), re.M)
assert (out/'rules6-before-full.json').read_bytes() == (out/'rules6-after-full.json').read_bytes()
assert (out/'ipv6-before-collision.json').read_bytes() == (out/'ipv6-after-collision.json').read_bytes()
assert 'Cannot reserve IPv6 routing table 100' in (out/'ipv6-collision.log').read_text()
assert (out/'rules-before-full.json').read_bytes() == (out/'rules-after-full.json').read_bytes()
assert (out/'rules-before-collision.json').read_bytes() == (out/'rules-after-collision.json').read_bytes()
assert (out/'main-routes-before-full.json').read_bytes() == (out/'main-routes-after-full.json').read_bytes()
assert (out/'nondefault-before-collision.json').read_bytes() == (out/'nondefault-after-collision.json').read_bytes()
assert 'Cannot reserve routing table 100' in (out/'nondefault-collision.log').read_text()
assert (out/'table-before-collision.json').read_bytes() == (out/'table-after-collision.json').read_bytes()
packets = load('packet-proof.json')
assert len(packets) == 2 and all(p['encrypted_data_packets'] > 0 for p in packets)
a, b = load('before-rejection.json'), load('after-rejection.json')
assert b['rejected_packets'] >= a['rejected_packets'] + 4
assert b['forwarded_packets'] == a['forwarded_packets']
benchmark = load('benchmark.json')
assert benchmark['relative_goodput'] >= 1.2
assert len(benchmark['samples']) == 6
expected_payloads.update((r['bytes_each_direction'], r['sha256'], r['protocol']) for r in benchmark['samples'])
records = [json.loads(line) for line in (out/'destination.jsonl').read_text().splitlines()]
assert {r['peer'] for r in records} == {'198.18.0.1', '2001:db8:85::1'}
assert records
assert collections.Counter((r['size'], r['sha256'], r['protocol']) for r in records) == expected_payloads, 'destination payload hashes or multiplicities differ'
assert 'Namespace already exists: tgproof-client' in (out/'preexisting.log').read_text()
assert 'Cannot reserve routing table 100' in (out/'collision-client.log').read_text()
assert 'Full tunnel requires an IPv6-capable exit' in (out/'ipv4-only-client.log').read_text()
assert '/build/release/triglav' in (out/'binary-sha256.txt').read_text()
outer6 = load('outer-ipv6/summary.json')
assert outer6['result'] == 'PASS' and outer6['full_tunnel_and_restoration']
assert len(load('outer-ipv6/packet-proof.json')) == 2
assert all(p['encrypted_data_packets'] > 0 for p in load('outer-ipv6/packet-proof.json'))
recovery = load('recovery/summary.json')
assert recovery['result'] == 'PASS'
assert recovery['same_client_process'] and recovery['same_session'] and recovery['one_tcp_connection']
assert recovery['outage_seconds'] == 20 and recovery['exit_restarted']
assert recovery['address_replacement_rounds'] == 10 and recovery['nat_addresses_rebound'] == 2
assert len(recovery['final_paths']) == 2
mtu = load('mtu/outer-ipv6/summary.json')
assert mtu['result'] == 'PASS'
assert mtu['dynamic_mtu']['outer_mtu'] == 1430 and mtu['dynamic_mtu']['inner_mtu'] == 1289
assert mtu['dynamic_mtu']['outer_fragments'] == 0
assert mtu['dynamic_mtu']['tcp4_connections'] == mtu['dynamic_mtu']['tcp6_connections'] == 1
summary = {
    'result': 'PASS',
    'binary_profile': 'release',
    'scope': 'Linux dual-stack TUN and SOCKS5/HTTP CONNECT, two NATed uplinks, one exit node',
    'rust_tests_passed': sum(int(p) for p, _, _ in results),
    'rust_tests_ignored_in_default_suite': sum(int(i) for _, _, i in results),
    'production_admission_tests_passed': 1,
    'destination_records': len(records),
    'application_upload_bytes_verified': sum(r['size'] for r in records),
    'outer_ipv6_destination_records': outer6['destination_records'],
    'outer_ipv6_application_bytes_each_direction': outer6['bytes_each_direction'],
    'outer_ipv6_full_tunnel_and_proxies': True,
    'exit_addresses': ['198.18.0.1', '2001:db8:85::1'],
    'path_counters': status['paths'],
    'pcap_encrypted_data_packets': [p['encrypted_data_packets'] for p in packets],
    'continuous_connection_frames': failover['frames'],
    'maximum_frame_round_trip_s': max(r['elapsed_s'] for r in failover['results']),
    'full_dual_stack_routing_and_restoration': True,
    'ipv4_only_full_tunnel_rejected': True,
    'physical_interface_down_and_recovery': True,
    'rate_capped_aggregation_gain': benchmark['relative_goodput'],
    'concurrent_proxy_transfers': len(proxies['transfers']),
    'source_manifest_sha256': hashlib.sha256((out/'source-sha256.txt').read_bytes()).hexdigest(),
    'recovery': recovery,
    'mtu': mtu,
    'unproven': ['physical NIC/ISP deployment', 'macOS TUN runtime', 'arbitrary MTUs and long-duration workloads'],
}
(out/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
