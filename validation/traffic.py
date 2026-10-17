#!/usr/bin/env python3
"""Native kernel TCP/UDP integrity tests, independent of Triglav internals."""
import concurrent.futures
import hashlib
import json
import random
import socket
import struct
import sys
import time


def read_exact(sock, size):
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk: raise AssertionError('truncated response')
        result.extend(chunk)
    return bytes(result)

def one(seed):
    payload = random.Random(seed).randbytes(size)
    start = time.monotonic()
    ipv6 = protocol.endswith('6')
    target = '2001:db8:85::2' if ipv6 else '198.18.0.2'
    exit_address = '2001:db8:85::1' if ipv6 else '198.18.0.1'
    if protocol.startswith('tcp'):
        with socket.create_connection((target, 19000), timeout=30, source_address=(source, 0) if source else None) as sock:
            sock.sendall(struct.pack('!I', len(payload)) + payload)
            peer_len, = struct.unpack('!I', read_exact(sock, 4))
            peer = read_exact(sock, peer_len).decode()
            actual_size, = struct.unpack('!I', read_exact(sock, 4))
            assert actual_size == size, (actual_size, size)
            response = read_exact(sock, actual_size)
            assert peer == exit_address, peer
    else:
        with socket.socket(socket.AF_INET6 if ipv6 else socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(5)
            if source: sock.bind((source, 0))
            sock.sendto(payload, (target, 19001))
            response, peer_addr = sock.recvfrom(65536)
            assert peer_addr[:2] == (target, 19001)
    assert response == payload, 'payload corruption'
    return {'seed': seed, 'bytes_each_direction': size, 'sha256': hashlib.sha256(payload).hexdigest(), 'elapsed_s': time.monotonic() - start}

if __name__ == '__main__':
    size, count = map(int, sys.argv[1:3])
    protocol = sys.argv[3] if len(sys.argv) > 3 else 'tcp'
    source = sys.argv[4] if len(sys.argv) > 4 else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(one, range(count)))
    print(json.dumps({'protocol': protocol, 'streams': results}))
