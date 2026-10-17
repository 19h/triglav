#!/usr/bin/env python3
"""Finite source-port churn; keep exact timing/resource observations."""
import json
import pathlib
import socket
import sys
import time
import urllib.request

out = pathlib.Path(sys.argv[1])
duration = 60.0
start = time.monotonic()
samples = []
attempts = 0
unique = set()

def tcp_progress():
    for _ in range(100):
        value = out.joinpath('progress.txt').read_text().strip()
        if value:
            return int(value)
        time.sleep(.001)  # The writer replaces the short counter in place.
    raise AssertionError('TCP progress counter remained empty')

initial_frames = tcp_progress()
latest_frames = initial_frames
while time.monotonic() - start < duration:
    sockets = []
    for index in range(128):
        family = socket.AF_INET6 if index % 2 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_DGRAM)
        sockets.append(sock)
        sock.bind(('::' if index % 2 else '0.0.0.0', 0))
        unique.add((family, sock.getsockname()[1]))
        target = ('2001:db8:85::2' if index % 2 else '198.18.0.2', 19999)
        try:
            sock.sendto(b'churn' * 16, target)
            attempts += 1
        except OSError:
            pass
    for sock in sockets:
        sock.close()
    with urllib.request.urlopen('http://127.0.0.1:9091/status', timeout=2) as response:
        status = json.load(response)
    process = pathlib.Path('/proc') / str(status['process_id']) / 'status'
    rss_kib = int(next(line.split()[1] for line in process.read_text().splitlines() if line.startswith('VmRSS:')))
    latest_frames = max(latest_frames, tcp_progress())
    samples.append({'elapsed_s': time.monotonic() - start, 'process_id': status['process_id'],
                    'rss_bytes': rss_kib * 1024, 'flows': status['quality']['active_flow_bindings'],
                    'usable_paths': status['quality']['usable_uplinks'], 'pending': status['pending'],
                    'tcp_frames': latest_frames})
    time.sleep(.1)
summary = {'duration_s': time.monotonic() - start, 'datagrams_sent': attempts,
           'distinct_source_ports': len(unique), 'samples': len(samples),
           'peak_flow_bindings': max(s['flows'] for s in samples),
           'tcp_frames_during_churn': latest_frames - initial_frames,
           'rss_initial_bytes': samples[0]['rss_bytes'],
           'rss_peak_bytes': max(s['rss_bytes'] for s in samples),
           'rss_final_bytes': samples[-1]['rss_bytes']}
out.joinpath('soak-samples.json').write_text(json.dumps(samples) + '\n')
out.joinpath('soak-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
assert summary['duration_s'] >= duration
assert summary['distinct_source_ports'] > 4096
assert summary['peak_flow_bindings'] == 4096, summary
assert summary['tcp_frames_during_churn'] >= 10, summary
assert len({s['process_id'] for s in samples}) == 1
assert all(s['flows'] <= 4096 and s['pending']['packets'] <= s['pending']['max_packets'] and s['pending']['bytes'] <= s['pending']['max_bytes'] for s in samples)
assert summary['rss_peak_bytes'] <= summary['rss_initial_bytes'] + 128 * 1024 * 1024, summary
assert samples[-1]['usable_paths'] == 2, samples[-1]
print(json.dumps(summary, indent=2))
