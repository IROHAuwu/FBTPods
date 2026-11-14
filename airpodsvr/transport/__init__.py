# SPDX-License-Identifier: Apache-2.0
"""Bluetooth back-ends. Each one opens AACP channels (L2CAP PSM 0x1001, MTU 2582) to AirPods.

A Channel has: send(bytes), recv(timeout) -> bytes | None (None on timeout), close(), closed.
recv raises ConnectionError when the link is gone.
"""
from __future__ import annotations

import sys


def errtext(e) -> str:
    """Readable text for an exception: TimeoutError() and friends have an empty str()."""
    t = str(e)
    return t if t else e.__class__.__name__


def forgot_us(e) -> bool:
    """The AirPods no longer have the key of our bond (they were reset, or paired again from their
    case): authentication with the key we kept fails with "PIN or key missing". Only pairing again
    helps, and only while they are in pairing mode."""
    return "PIN_OR_KEY_MISSING" in str(e) or getattr(e, "error_code", None) == 0x06


class Channel:
    closed = False

    def send(self, pkt: bytes) -> None:
        raise NotImplementedError

    def recv(self, timeout: float):
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True


def open_backend(cfg: dict):
    """Pick the back-end from the config: 'bumble' (Windows + dedicated USB dongle), 'bluez' (Linux),
    'auto' (bluez on Linux, bumble elsewhere)."""
    name = cfg.get("backend", "auto")
    if name == "auto":
        name = "bluez" if sys.platform.startswith("linux") else "bumble"
    if name == "bluez":
        from .bluez import BlueZBackend
        return BlueZBackend(cfg)
    if name == "bumble":
        from .bumble_usb import BumbleBackend
        return BumbleBackend(cfg)
    raise ValueError(f"unknown backend {name!r}")
