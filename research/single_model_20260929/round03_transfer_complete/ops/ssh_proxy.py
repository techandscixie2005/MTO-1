#!/usr/bin/env python3
"""Scoped HTTP CONNECT relay for existing GitHub SSH auth; no data logging."""
import os
import socket
import sys
import threading

host, port = sys.argv[1], int(sys.argv[2])
if host not in {'github.com', 'ssh.github.com'} or port not in {22,443}:
    raise SystemExit('This relay is scoped to GitHub SSH only')
s = socket.create_connection(('127.0.0.1',7890),timeout=15)
s.sendall(f'CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n'.encode())
header = bytearray()
while not header.endswith(b'\r\n\r\n'):
    part = s.recv(1)
    if not part or len(header)>65536:
        raise SystemExit('HTTP CONNECT proxy did not return complete headers')
    header.extend(part)
if header.split(b'\r\n',1)[0].split()[1] != b'200':
    raise SystemExit('HTTP CONNECT proxy rejected the connection')
s.settimeout(None)
def upstream():
    try:
        while True:
            data = os.read(0,65536)
            if not data:
                s.shutdown(socket.SHUT_WR)
                return
            s.sendall(data)
    except (OSError,ConnectionError):
        pass
threading.Thread(target=upstream,daemon=True).start()
try:
    while True:
        data=s.recv(65536)
        if not data:
            break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
finally:
    s.close()
