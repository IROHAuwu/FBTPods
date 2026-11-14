# SPDX-License-Identifier: Apache-2.0
# L2CAP socket code derived from MousePods l2cap.py (Matteo Gaudioso, starman-tech) — MODIFIED.
"""Linux back-end: BlueZ kernel sockets. The AirPods must already be paired (bluetoothctl)."""
from __future__ import annotations

import socket
import struct
import subprocess

from ..protocol import IOS_MTU, PSM_AACP
from . import Channel

SOL_BLUETOOTH, BT_SECURITY, BT_SECURITY_MEDIUM = 274, 4, 2
SOL_L2CAP, L2CAP_OPTIONS = 6, 1


class BlueZChannel(Channel):
    def __init__(self, addr: str):
        self.addr = addr
        s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET, socket.BTPROTO_L2CAP)
        try:
            s.setsockopt(SOL_BLUETOOTH, BT_SECURITY, struct.pack("BB", BT_SECURITY_MEDIUM, 0))
            opts = bytearray(s.getsockopt(SOL_L2CAP, L2CAP_OPTIONS, 12))
            struct.pack_into("<H", opts, 2, IOS_MTU)
            s.setsockopt(SOL_L2CAP, L2CAP_OPTIONS, bytes(opts))
        except OSError:
            pass
        s.settimeout(8)
        s.connect((addr, PSM_AACP))
        self.sock = s

    def send(self, pkt):
        self.sock.send(pkt)

    def recv(self, timeout):
        self.sock.settimeout(timeout)
        try:
            data = self.sock.recv(4096)
        except socket.timeout:
            return None
        except OSError as e:
            raise ConnectionError(str(e)) from e
        if not data:
            raise ConnectionError("link closed")
        return data

    def close(self):
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass


class BlueZBackend:
    def __init__(self, cfg):
        self.cfg = cfg

    def list_paired(self):
        """[(addr, name)] of paired Apple audio devices, via bluetoothctl."""
        try:
            out = subprocess.run(["bluetoothctl", "devices", "Paired"], capture_output=True,
                                 text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            return []
        res = []
        for line in out.splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3 and parts[0] == "Device" and "airpods" in parts[2].lower():
                res.append((parts[1], parts[2]))
        return res

    def ensure_connected(self, addr):
        subprocess.run(["bluetoothctl", "connect", addr], capture_output=True, timeout=20)

    def open(self, addr) -> Channel:
        self.ensure_connected(addr)
        return BlueZChannel(addr)

    def close(self):
        pass
