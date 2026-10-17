#!/usr/bin/env python3
"""One TCP connection spanning path failures; any reconnect or corruption fails."""
import hashlib
import json
import random
import socket
import struct
import sys
import time
from traffic import read_exact

progress = sys.argv[1]
results = []
with socket.create_connection(('198.18.0.2', 19000), timeout=40) as sock:
    for i in range(120):
        data = random.Random(1000 + i).randbytes(16384)
        start = time.monotonic()
        sock.sendall(struct.pack('!I', len(data)) + data)
        peer_len, = struct.unpack('!I', read_exact(sock, 4))
        peer = read_exact(sock, peer_len).decode()
        size, = struct.unpack('!I', read_exact(sock, 4))
        assert size == len(data)
        assert read_exact(sock, size) == data
        assert peer == '198.18.0.1'
        results.append({'chunk': i, 'sha256': hashlib.sha256(data).hexdigest(), 'elapsed_s': time.monotonic() - start})
        with open(progress, 'a') as output: output.write(f'{i}\n')
        time.sleep(.1)
print(json.dumps({'frames': len(results), 'connections': 1, 'results': results}))
