#!/usr/bin/env python3
"""Concurrent SOCKS5/HTTP CONNECT transfers with server-observed exit identity."""
import concurrent.futures
import hashlib
import json
import random
import socket
import struct
from traffic import read_exact

def connect(mode, port, ipv6=False):
    sock = socket.create_connection(('127.0.0.1', 11080 if mode == 'socks' else 18080), timeout=15)
    if mode == 'socks':
        sock.sendall(b'\x05\x01\x00')
        assert read_exact(sock, 2) == b'\x05\x00'
        if ipv6:
            request = b'\x05\x01\x00\x04' + socket.inet_pton(socket.AF_INET6, '2001:db8:85::2')
        else:
            request = b'\x05\x01\x00\x01' + socket.inet_aton('198.18.0.2')
        sock.sendall(request + struct.pack('!H', port))
        reply = read_exact(sock, 4)
        if port != 19000:
            assert reply[1] != 0
            sock.close(); return None
        assert reply[1] == 0, reply
        if reply[3] == 1: read_exact(sock, 6)
        elif reply[3] == 4: read_exact(sock, 18)
        else: read_exact(sock, read_exact(sock, 1)[0] + 2)
    else:
        target = '[2001:db8:85::2]' if ipv6 else '198.18.0.2'
        sock.sendall(f'CONNECT {target}:{port} HTTP/1.1\r\nHost: endpoint\r\n\r\n'.encode())
        response = b''
        while not response.endswith(b'\r\n\r\n'): response += read_exact(sock, 1)
        if port != 19000:
            assert response.startswith(b'HTTP/1.1 502'), response
            sock.close(); return None
        assert response.startswith(b'HTTP/1.1 200'), response
    return sock

def one(index):
    mode = 'socks' if index % 2 else 'http'
    data = random.Random(index + 5000).randbytes(131072)
    ipv6 = bool(index // 2 % 2)
    with connect(mode, 19000, ipv6) as sock:
        sock.sendall(struct.pack('!I', len(data)) + data)
        sock.shutdown(socket.SHUT_WR) # Exercise half-close while reply is outstanding.
        length, = struct.unpack('!I', read_exact(sock, 4))
        assert read_exact(sock, length) == (b'2001:db8:85::1' if ipv6 else b'198.18.0.1')
        size, = struct.unpack('!I', read_exact(sock, 4))
        assert size == len(data)
        assert read_exact(sock, size) == data
        assert sock.recv(1) == b''
    return {'mode': mode, 'protocol': 'tcp6' if ipv6 else 'tcp', 'sha256': hashlib.sha256(data).hexdigest(), 'bytes_each_direction': len(data)}

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    records = list(pool.map(one, range(16)))
for mode in ['socks', 'http']:
    for ipv6 in [False, True]:
        assert connect(mode, 19999, ipv6) is None
print(json.dumps({'transfers': records, 'failed_targets_rejected': 4}))
