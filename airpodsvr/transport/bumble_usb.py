# SPDX-License-Identifier: Apache-2.0
"""Windows back-end: Google Bumble Bluetooth stack driving a dedicated USB Bluetooth dongle.

Windows does not let user programs open a classic L2CAP channel on an arbitrary PSM, so the
AirPods are paired to a second USB dongle that Bumble drives directly. That dongle must use the
WinUSB driver (Zadig, one-time; see README). The built-in Bluetooth keeps working for everything else.

All Bumble work runs on one asyncio loop in a background thread; channels are thread-safe.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import queue
import threading

from ..protocol import IOS_MTU, PSM_AACP
from . import Channel, errtext

log = logging.getLogger(__name__)

APPLE_COMPANY_ID = 0x004C


class _SignalingLog(logging.Filter):
    """Keep Bumble's L2CAP signaling (connect / configure / disconnect frames, channel states) in our
    log, without colours and without data packets: what a failing AACP channel needs to be explained."""
    _ansi = __import__("re").compile(r"\x1b\[[0-9;]*m")

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        if "Signaling Control Frame" not in msg and "state change" not in msg:
            return False
        record.msg, record.args = self._ansi.sub("", msg), None
        return True


def _patch_l2cap_configure():
    """Bumble rejects every L2CAP configuration option it does not implement. The AirPods' Configure
    Request for the AACP channel carries Flush Timeout (0x02, 30 ms): Bumble answers 'unknown option',
    the AirPods ask a few more times and then close the channel, which surfaces as
    'L2CAP/Connection failure'. Flush timeout, QoS and extended flow spec only describe how the peer
    sends; a basic-mode receiver can accept them (BlueZ does). Strip them before Bumble looks."""
    import dataclasses
    from bumble import l2cap
    cls = l2cap.ClassicChannel
    if getattr(cls.on_configure_request, "_fbt_patched", False):
        return
    orig = cls.on_configure_request
    ignorable = {0x02, 0x03, 0x06}            # flush timeout, QoS, extended flow specification

    def on_configure_request(self, request):
        try:
            opts = l2cap.L2CAP_Control_Frame.decode_configuration_options(request.options)
            keep = [(t, v) for t, v in opts if (int(t) & 0x7F) not in ignorable]
            if len(keep) != len(opts):
                dropped = ", ".join(f"0x{int(t) & 0x7F:02x}={bytes(v).hex()}" for t, v in opts
                                    if (int(t) & 0x7F) in ignorable)
                log.debug("L2CAP configure: accepting %s", dropped)
                request = dataclasses.replace(
                    request, options=l2cap.L2CAP_Control_Frame.encode_configuration_options(keep))
        except Exception as e:                # noqa: BLE001 - fall back to Bumble's own handling
            log.debug("L2CAP configure options not filtered: %s", e)
        return orig(self, request)
    on_configure_request._fbt_patched = True
    cls.on_configure_request = on_configure_request


def _log_l2cap_signaling():
    lg = logging.getLogger("bumble.l2cap")
    if not any(isinstance(f, _SignalingLog) for f in lg.filters):
        lg.addFilter(_SignalingLog())
    lg.setLevel(logging.DEBUG)


class BumbleChannel(Channel):
    def __init__(self, backend, addr):
        self.backend = backend
        self.addr = addr
        self.q = queue.Queue()
        self.l2 = None
        self.conn = None

    def _on_pdu(self, pdu):
        self.q.put(bytes(pdu))

    def _on_gone(self, *_):
        self.closed = True
        self.q.put(None)

    def _on_disconnected(self, *_):
        owned = getattr(self.backend, "_owned", None)
        if owned is not None and self.conn is not None:
            owned.discard(self.conn)
        self.conn = None                  # the ACL link itself is gone: nothing left to disconnect
        self._on_gone()

    def send(self, pkt):
        if self.closed or self.l2 is None:
            raise ConnectionError("channel closed")
        self.backend.loop.call_soon_threadsafe(self.l2.send_pdu, bytes(pkt))

    def recv(self, timeout):
        try:
            item = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is None:
            raise ConnectionError("link closed")
        return item

    def close(self):
        # The AACP channel can close while the ACL link stays up (the buds refuse or drop only the
        # channel). The link must still be disconnected, or the next connect to the same AirPods
        # fails with "connection already exists" until they are power-cycled.
        self.closed = True
        conn, self.conn = self.conn, None
        if conn is not None:
            async def _bye():
                try:
                    await conn.disconnect()
                except Exception:          # noqa: BLE001 - best effort
                    pass
            try:                           # the stack may already be shutting down
                self.backend.run(_bye(), timeout=5)
            except Exception:              # noqa: BLE001
                pass


def firmware_root():
    from .. import config as C
    import sys
    for base in (C.ROOT, getattr(sys, "_MEIPASS", C.ROOT), C.APP_DIR):
        p = os.path.join(base, "firmware")
        if os.path.isdir(p):
            return p
    return None


def use_bundled_firmware():
    """Point Bumble's Realtek / Intel drivers at the firmware shipped with FBT Pods (linux-firmware
    files). Bumble loads it into the adapter on every power-on, as Linux does at boot."""
    root = firmware_root()
    if root:
        os.environ.setdefault("BUMBLE_RTK_FIRMWARE_DIR", os.path.join(root, "realtek"))
        os.environ.setdefault("BUMBLE_INTEL_FIRMWARE_DIR", os.path.join(root, "intel"))


class _FirmwareLog(logging.Handler):
    """Notices which Bumble driver loaded a firmware (for the setup screen)."""

    def __init__(self, backend):
        super().__init__(logging.INFO)
        self.backend = backend

    def emit(self, record):
        name = record.name or ""
        if name.startswith("bumble.drivers.rtk") and "loaded FW" in record.getMessage():
            self.backend.firmware = "realtek"
        elif name.startswith("bumble.drivers.intel") and "firmware" in record.getMessage().lower():
            self.backend.firmware = self.backend.firmware or "intel"


class BumbleBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.transport_name = cfg.get("bumble_transport", "usb:0")
        from .. import config as C
        self.keystore_path = C.data_path(cfg.get("bumble_keystore", "data/bumble_keys.json"))
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="bumble", daemon=True)
        self.thread.start()
        self.device = None
        self.hci = None
        self.firmware = None              # "realtek" / "intel" when a firmware was loaded
        use_bundled_firmware()
        try:
            self.run(self._start(), timeout=90)   # Intel firmware upload takes a few seconds
        except BaseException:
            self.close()
            raise

    # ---------------------------------------------------------------- loop helpers
    def run(self, coro, timeout=None):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return fut.result(timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            fut.cancel()                  # do not leave it running (and connecting) behind our back
            raise

    async def _start(self):
        from bumble.device import Device
        from bumble.hci import Address
        from bumble.keys import JsonKeyStore
        from bumble.transport import open_transport

        os.makedirs(os.path.dirname(self.keystore_path), exist_ok=True)
        fw_log = self._fw_log = _FirmwareLog(self)
        for n in ("bumble.drivers.rtk", "bumble.drivers.intel"):
            logging.getLogger(n).addHandler(fw_log)
            if logging.getLogger(n).getEffectiveLevel() > logging.INFO:
                logging.getLogger(n).setLevel(logging.INFO)
        self.hci = await open_transport(self.transport_name)
        # firmware loaders: Bumble's Realtek / Intel ones plus ours (Intel legacy, MediaTek,
        # Qualcomm); the latter need the libusb handle for vendor control transfers
        import inspect
        from bumble.host import Host
        from . import btfw
        if list(inspect.signature(Host.reset).parameters)[1:] == ["driver_factory"]:
            Host.reset.__defaults__ = (btfw.driver_for_host,)
        else:                              # a Bumble release that changed Host.reset
            log.warning("this Bumble version is not supported by the firmware loaders: only adapters "
                        "that need no firmware (or Bumble's own Realtek / Intel ones) will work")
        btfw.last_driver = None
        md = getattr(self.hci.source, "metadata", None)
        if isinstance(md, dict) and getattr(self.hci, "device", None) is not None:
            md["usb_handle"] = self.hci.device
        dev = Device.with_hci("FBT Pods", Address("F0:F1:F2:F3:F4:F5"), self.hci.source, self.hci.sink)
        dev.classic_enabled = True
        dev.le_enabled = False
        dev.class_of_device = 0x0C010C           # computer / laptop
        dev.keystore = JsonKeyStore(namespace=None, filename=self.keystore_path)
        self._add_device_id_record(dev)
        await dev.power_on()
        self.firmware = btfw.last_driver or getattr(self, "firmware", None)
        self.device = dev
        self._connect_lock = asyncio.Lock()
        _patch_l2cap_configure()
        _log_l2cap_signaling()
        q = getattr(dev.host, "acl_packet_queue", None)
        log.info("Bumble dongle ready: %s (%s), ACL buffers %s x %s bytes", dev.public_address,
                 self.transport_name, getattr(q, "max_in_flight", "?"), getattr(q, "max_packet_size", "?"))

    @staticmethod
    def _add_device_id_record(dev):
        """SDP Device ID record claiming an Apple host (MousePods changed this together with the MTU
        before the first success; probably not required, kept as it is harmless)."""
        try:
            from bumble.sdp import DataElement, ServiceAttribute
            from bumble.core import UUID
            attrs = [
                ServiceAttribute(0x0001, DataElement.sequence([DataElement.uuid(UUID.from_16_bits(0x1200))])),
                ServiceAttribute(0x0200, DataElement.unsigned_integer_16(0x0103)),
                ServiceAttribute(0x0201, DataElement.unsigned_integer_16(APPLE_COMPANY_ID)),
                ServiceAttribute(0x0202, DataElement.unsigned_integer_16(0x0000)),
                ServiceAttribute(0x0203, DataElement.unsigned_integer_16(0x0000)),
                ServiceAttribute(0x0204, DataElement.boolean(True)),
                ServiceAttribute(0x0205, DataElement.unsigned_integer_16(0x0001)),
            ]
            dev.sdp_service_records = {0x00010001: attrs}
        except Exception as e:     # noqa: BLE001
            log.debug("Device ID record not added: %s", e)

    # ---------------------------------------------------------------- discovery / pairing
    def discover(self, seconds=10.0):
        """Inquiry scan -> [(addr, name)] of devices whose name contains 'AirPods'.
        Put the AirPods in pairing mode first (case open, hold the back button until white)."""
        async def _scan():
            found = {}

            def on_result(address, cod, data, rssi):
                name = ""
                try:
                    from bumble.core import AdvertisingData
                    v = data.get(AdvertisingData.COMPLETE_LOCAL_NAME) or data.get(AdvertisingData.SHORTENED_LOCAL_NAME)
                    if v:
                        name = v if isinstance(v, str) else bytes(v).decode("utf-8", "replace")
                except Exception:   # noqa: BLE001
                    pass
                key = str(address)
                found[key] = (address, name or found.get(key, (None, ""))[1])

            self.device.on("inquiry_result", on_result)
            await self.device.start_discovery(auto_restart=False)
            await asyncio.sleep(seconds)
            try:
                await self.device.stop_discovery()
            except Exception:   # noqa: BLE001
                pass
            self.device.remove_listener("inquiry_result", on_result)
            res = []
            for key, (address, name) in found.items():
                if not name:               # no name in the inquiry response: ask the device
                    try:                   # (needs the Address object, not its text)
                        name = await asyncio.wait_for(self.device.request_remote_name(address), 5)
                    except Exception as e:   # noqa: BLE001
                        log.debug("name of %s: %s", key, errtext(e))
                        name = ""
                res.append((key.split("/")[0], name))
            return res

        return self.run(_scan(), timeout=seconds + 30)

    def list_paired(self):
        """Addresses bonded in the Bumble key store."""
        async def _keys():
            try:
                items = await self.device.keystore.get_all()
            except Exception:   # noqa: BLE001
                return []
            return [(name.split("/")[0], "AirPods (bonded)") for name, _ in items]
        return self.run(_keys(), timeout=10)

    async def _secure(self, conn):
        """Pair (first time) or reuse the bond, then encrypt: the AirPods close the AACP channel on
        a link that is not encrypted."""
        await conn.authenticate()          # pairs on first use, then reuses the bonded link key
        if not conn.is_encrypted:
            try:
                await asyncio.wait_for(conn.encrypt(), 10)
            except Exception as e:         # noqa: BLE001 - the channel open will say if it matters
                log.warning("encryption not enabled: %s", errtext(e))
        log.info("link with %s: encrypted=%s", getattr(conn, "peer_address", "?"),
                 getattr(conn, "is_encrypted", None))

    async def _connect(self, addr):
        """One page at a time (adapters refuse a second Create Connection meanwhile: 'Command
        Disallowed'), and drop a stale link to the same AirPods left by an attempt that was given up
        (or the AirPods connecting to us), which otherwise fails every retry with 'Connection
        Already Exists'."""
        from bumble.core import PhysicalTransport
        from bumble.hci import Address
        lock = getattr(self, "_connect_lock", None)
        if lock is None:
            lock = self._connect_lock = asyncio.Lock()
        async with lock:
            old = None
            try:
                old = self.device.find_connection_by_bd_addr(Address(addr), transport=PhysicalTransport.BR_EDR)
            except Exception:              # noqa: BLE001
                pass
            if old is not None and old not in getattr(self, "_owned", set()):
                log.info("dropping a stale link to %s", addr)
                try:
                    await asyncio.wait_for(old.disconnect(), 5)
                except BaseException:      # noqa: BLE001
                    pass
                await asyncio.sleep(0.5)
            conn = await self.device.connect(addr, transport=PhysicalTransport.BR_EDR, timeout=20)
            try:                           # pairing / encryption inside the lock too: some adapters
                await self._secure(conn)   # refuse a page while another link authenticates
            except BaseException:
                try:
                    await asyncio.wait_for(conn.disconnect(), 5)
                except BaseException:      # noqa: BLE001
                    pass
                raise
            return conn

    def _queue_state(self):
        q = getattr(getattr(self.device, "host", None), "acl_packet_queue", None)
        return f"ACL in flight {getattr(q, '_in_flight', '?')}/{getattr(q, 'max_in_flight', '?')}, " \
               f"queued {getattr(q, '_queued', '?')}"

    def copy_key(self, src, dst):
        """Experiment: let `dst` be tried with the link key bonded for `src` (the two buds of a pair
        may share it). Returns True when a key was copied; never replaces a key `dst` already has."""
        async def _copy():
            ks = self.device.keystore
            have = dict(await ks.get_all())

            def find(addr):
                want = str(addr).upper().split("/")[0]
                return next((k for k in have if str(k).upper().split("/")[0] == want), None)
            k_src = find(src)
            if k_src is None or find(dst) is not None:
                return False
            name = str(k_src)
            suffix = name[len(name.split("/")[0]):]
            await ks.update(str(dst).upper() + suffix, have[k_src])
            return True
        return bool(self.run(_copy(), timeout=10))

    def forget_key(self, addr):
        async def _forget():
            ks = self.device.keystore
            want = str(addr).upper().split("/")[0]
            for k, _ in await ks.get_all():
                if str(k).upper().split("/")[0] == want:
                    await ks.delete(k)
        try:
            self.run(_forget(), timeout=10)
        except Exception:   # noqa: BLE001
            pass

    def pair(self, addr):
        """Pairing only (AirPods in the open case, pairing mode): bond + encryption, no AACP channel.
        In pairing mode the buds may refuse or close the motion channel; that is tried later, with
        the buds out of the case. Raises when pairing itself fails."""
        async def _pair():
            conn = await self._connect(addr)
            try:
                await asyncio.wait_for(conn.disconnect(), 5)
            except BaseException:          # noqa: BLE001 - best effort
                pass
        # Start from nothing. A key kept from an earlier pairing is tried first otherwise, and when
        # the AirPods have dropped theirs (reset, or the case button held long) that fails with "PIN
        # or key missing", every time: these AirPods could never be paired again.
        self.forget_key(addr)
        self.run(_pair(), timeout=45)
        log.info("paired with %s", addr)

    def open(self, addr) -> Channel:
        ch = BumbleChannel(self, addr)

        async def _open():
            from bumble.l2cap import ClassicChannelSpec, L2capError
            conn = await self._connect(addr)
            ch.conn = conn
            conn.on("disconnection", ch._on_disconnected)
            try:
                # the buds sometimes accept the channel and close it again during configuration
                # (L2CAP "Connection failure", code 0), typically right after pairing or while the
                # other bud is still negotiating: wait a moment and ask again on the same link
                for attempt in range(3):
                    try:
                        l2 = await asyncio.wait_for(
                            conn.create_l2cap_channel(spec=ClassicChannelSpec(psm=PSM_AACP, mtu=IOS_MTU)), 15)
                        break
                    except (L2capError, asyncio.TimeoutError) as e:
                        log.warning("AACP channel with %s refused (attempt %d): %s; %s", addr, attempt + 1,
                                    errtext(e), self._queue_state())
                        if isinstance(e, asyncio.TimeoutError):
                            raise ConnectionError("AirPods 沒有回應動作資料通道（15 秒）") from e
                        if attempt == 2 or getattr(e, "error_code", None) not in (0, None) \
                                or ch.conn is None:          # the whole link went (see _on_disconnected)
                            raise
                        await asyncio.sleep(1.5)
            except BaseException:              # pairing refused, channel refused, or timed out
                # connected but unusable: hang up, or every retry fails with "connection already
                # exists" (the caller never gets this channel, so nobody else would)
                ch.conn = None
                try:
                    await asyncio.wait_for(conn.disconnect(), 5)
                except BaseException:          # noqa: BLE001 - best effort
                    pass
                raise
            owned = self._owned = getattr(self, "_owned", set())     # links in use: never "stale"
            owned.add(conn)

            def gone(*a):
                owned.discard(conn)
                ch._on_gone(*a)
            l2.sink = ch._on_pdu
            l2.on("close", gone)
            ch.l2 = l2

        self.run(_open(), timeout=60)
        log.info("AACP channel open with %s", addr)
        return ch

    def close(self):
        async def _stop():
            if self.hci:
                await self.hci.close()
        try:
            self.run(_stop(), timeout=5)
        except Exception:   # noqa: BLE001
            pass
        fw_log = getattr(self, "_fw_log", None)
        if fw_log is not None:             # one handler per backend would pile up on the loggers
            for n in ("bumble.drivers.rtk", "bumble.drivers.intel"):
                logging.getLogger(n).removeHandler(fw_log)
            self._fw_log = None
        self.loop.call_soon_threadsafe(self.loop.stop)
