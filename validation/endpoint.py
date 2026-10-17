#!/usr/bin/env python3
"""Independent destination: records actual peers and returns exact binary bytes."""
import hashlib
import json
import socket
import socketserver
import struct
import threading
import sys

lock = threading.Lock()
log = sys.argv[1]

def record(protocol, peer, payload):
    with lock, open(log, 'a') as output:
        output.write(json.dumps({'protocol': protocol, 'peer': peer[0], 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}) + '\n')

class TCP(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            header = self.read_exact(4)
            if not header: return
            length, = struct.unpack('!I', header)
            if length > 32 * 1024 * 1024: return
            payload = self.read_exact(length)
            if len(payload) != length: return
            record('tcp6' if self.server.address_family == socket.AF_INET6 else 'tcp', self.client_address, payload)
            peer = self.client_address[0].encode()
            self.request.sendall(struct.pack('!I', len(peer)) + peer + header + payload)
    def read_exact(self, size):
        result = bytearray()
        while len(result) < size:
            chunk = self.request.recv(size - len(result))
            if not chunk: break
            result.extend(chunk)
        return bytes(result)

class UDP(socketserver.BaseRequestHandler):
    def handle(self):
        payload, sock = self.request
        record('udp6' if self.server.address_family == socket.AF_INET6 else 'udp', self.client_address, payload)
        sock.sendto(payload, self.client_address)

class TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

class TCP6Server(TCPServer):
    address_family = socket.AF_INET6

class UDP6Server(socketserver.ThreadingUDPServer):
    address_family = socket.AF_INET6

tcp6 = TCP6Server(('2001:db8:85::2', 19000), TCP)
threading.Thread(target=tcp6.serve_forever, daemon=True).start()
udp6 = UDP6Server(('2001:db8:85::2', 19001), UDP)
threading.Thread(target=udp6.serve_forever, daemon=True).start()
udp = socketserver.ThreadingUDPServer(('198.18.0.2', 19001), UDP)
threading.Thread(target=udp.serve_forever, daemon=True).start()
TCPServer(('198.18.0.2', 19000), TCP).serve_forever()
