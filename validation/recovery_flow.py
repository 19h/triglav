#!/usr/bin/env python3
"""One TCP connection across recovery phases; any reconnect or mismatch fails."""
import hashlib
import json
import pathlib
import random
import socket
import struct
import sys
import time
from traffic import read_exact

out = pathlib.Path(sys.argv[1])
results = []
with socket.create_connection(('198.18.0.2', 19000), timeout=90) as sock:
    for index in range(1200):
        if (out / 'stop-flow').exists():
            break
        payload = random.Random(20000 + index).randbytes(16384)
        start = time.monotonic()
        sock.sendall(struct.pack('!I', len(payload)) + payload)
        peer_len, = struct.unpack('!I', read_exact(sock, 4))
        peer = read_exact(sock, peer_len).decode()
        size, = struct.unpack('!I', read_exact(sock, 4))
        assert size == len(payload)
        assert read_exact(sock, size) == payload
        assert peer == '198.18.0.1', peer
        results.append({'chunk': index, 'sha256': hashlib.sha256(payload).hexdigest(),
                        'bytes_each_direction': size, 'elapsed_s': time.monotonic() - start})
        (out / 'progress.txt').write_text(str(index + 1))
        time.sleep(.1)
    else:
        raise AssertionError('recovery harness did not stop the flow')
print(json.dumps({'connections': 1, 'frames': len(results), 'results': results}))
