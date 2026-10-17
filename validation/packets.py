#!/usr/bin/env python3
"""Parse Ethernet PCAP without dependencies; verify encrypted data or inject mutations."""
import json
import ipaddress
import socket
import struct
import sys
import zlib

def records(path):
    with open(path, 'rb') as src:
        header = src.read(24)
        endian = '<' if header[:4] == b'\xd4\xc3\xb2\xa1' else '>'
        assert struct.unpack(endian + 'I', header[20:24])[0] == 1, 'Ethernet capture required'
        while h := src.read(16):
            length = struct.unpack(endian + 'IIII', h)[2]
            eth = src.read(length)
            ip = eth[14:]
            if eth[12:14] == b'\x08\x00':
                ihl = (ip[0] & 15) * 4
                if ip[9] != 17: continue
                source, destination, udp = ip[12:16], ip[16:20], ip[ihl:]
            elif eth[12:14] == b'\x86\xdd':
                # The controlled outer IPv6 path has no extension headers.
                if len(ip) < 40 or ip[6] != 17: continue
                source, destination, udp = ip[8:24], ip[24:40], ip[40:]
            else:
                continue
            if len(udp) < 8: continue
            srcport, dstport, length, _ = struct.unpack('!HHHH', udp[:8])
            if dstport != 7443: continue
            payload = udp[8:length]
            if len(payload) < 60: continue
            yield source, destination, srcport, dstport, payload

if sys.argv[1] == 'session':
    print(next(records(sys.argv[2]))[4][20:52].hex())
elif sys.argv[1] == 'verify':
    counts = []
    for path in sys.argv[2:]:
        packets = list(records(path))
        data = [p for p in packets if p[4][1] == 0]
        assert data, f'No data on {path}'
        assert all(struct.unpack('!H', p[4][2:4])[0] & 16 for p in data), 'unencrypted data'
        assert all(len(p[4]) <= 1460 for p in data), 'oversized outer packet'
        counts.append({'capture': path, 'encrypted_data_packets': len(data), 'sources': sorted({str(ipaddress.ip_address(p[0])) for p in data})})
    print(json.dumps(counts))
else:
    # Inject inside the exit namespace with the exact observed source tuple; a fresh UDP
    # source port or upstream NAT would test address binding instead of replay.
    chosen = next(p for p in records(sys.argv[2]) if p[4][1] == 0)
    src, dst, srcport, dstport, original = chosen
    assert len(src) == len(dst) == 4, 'mutation injection uses the IPv4-underlay lane'
    variants = [('replay', original)]
    corrupt = bytearray(original); corrupt[-1] ^= 1
    corrupt[60:68] = (10**12).to_bytes(8, 'big')
    variants.append(('ciphertext', corrupt))
    forged = bytearray(original); forged[4:12] = b'\xff' * 8
    forged[56:60] = struct.pack('!I', zlib.crc32(forged[:56]))
    variants.append(('sequence', forged))
    plaintext = bytearray(original); plaintext[2:4] = b'\x00\x00'
    plaintext[56:60] = struct.pack('!I', zlib.crc32(plaintext[:56]))
    variants.append(('plaintext', plaintext))
    with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW) as sock:
        for name, payload in variants:
            udp = struct.pack('!HHHH', srcport, dstport, len(payload) + 8, 0) + payload
            ip = bytearray(struct.pack('!BBHHHBBH4s4s', 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0, src, dst))
            total = sum(struct.unpack('!10H', ip))
            while total >> 16: total = (total & 65535) + (total >> 16)
            ip[10:12] = struct.pack('!H', ~total & 65535)
            sock.sendto(ip + udp, (socket.inet_ntoa(dst), 0))
    print(json.dumps({'mutations_sent': [name for name, _ in variants]}))
