#!/usr/bin/env python3
"""Same payload and exit, one uplink versus two, under equal per-link rate caps."""
import json
import pathlib
import statistics
import subprocess
import sys
import time

root = pathlib.Path(__file__).resolve().parent
source = sys.argv[1]
samples = []
try:
    for iteration in range(3):
        for mode in ['single', 'multi']:
            # Reset queues between trials; retained duplicate packets must not
            # bias the following mode's timing.
            for number in [1, 2]:
                subprocess.run(['ip', 'netns', 'exec', f'tgproof-r{number}', 'tc', 'qdisc', 'replace',
                    'dev', f'wan{number}', 'root', 'netem', 'rate', '2mbit', 'limit', '2000'], check=True)
            time.sleep(.2)
            args = ['ip', 'netns', 'exec', 'tgproof-client', 'python3', str(root/'traffic.py'), '262144', '1', 'tcp']
            if mode == 'single': args.append(source)
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            stream = json.loads(result.stdout)['streams'][0]
            samples.append({'mode': mode, 'iteration': iteration, 'protocol': 'tcp', **stream})
            time.sleep(.3)
finally:
    for number in [1, 2]:
        subprocess.run(['ip', 'netns', 'exec', f'tgproof-r{number}', 'tc', 'qdisc', 'del', 'dev', f'wan{number}', 'root'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
medians = {mode: statistics.median(s['elapsed_s'] for s in samples if s['mode'] == mode) for mode in ['single', 'multi']}
ratio = medians['single'] / medians['multi']
report = {'per_link_rate_bit_s': 2_000_000, 'payload_bytes': 262144, 'samples': samples,
          'median_elapsed_s': medians, 'relative_goodput': ratio,
          'request_goodput_bit_s': {m: 262144 * 8 / t for m, t in medians.items()}}
print(json.dumps(report))
assert ratio >= 1.2, f'No measured aggregation gain: {ratio:.3f}'
