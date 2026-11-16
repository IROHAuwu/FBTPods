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
import re
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


_L2CAP_HANDLED = {0x01, 0x04, 0x05}          # MTU, mode (retransmission / flow control), FCS
_L2CAP_IGNORABLE = {0x02, 0x03, 0x06, 0x07}    # flush timeout, QoS, extended flow spec, extended window
_patch_warned = set()                          # each "could not ..." said once per process
_configure_seen = set()                        # (peer, PSM) whose Configure Request was logged


def _warn_once(key, msg, *args):
    if key not in _patch_warned:
        _patch_warned.add(key)
        log.warning(msg, *args)


def _filter_configure_options(opts):
    """The options Bumble should see: what it implements, minus what only describes how the peer
    sends. An unknown option with the hint bit (0x80) set must be skipped, not rejected (Core spec
    Vol 3 Part A 5); a hint on an option Bumble knows is passed on without the bit, so Bumble
    recognises it. Returns (kept, dropped)."""
    keep, dropped = [], []
    for t, v in opts:
        base = int(t) & 0x7F
        if base in _L2CAP_IGNORABLE or (int(t) & 0x80 and base not in _L2CAP_HANDLED):
            dropped.append((t, v))
        else:
            keep.append((base, v) if int(t) & 0x80 else (t, v))
    return keep, dropped


def _patch_l2cap_configure():
    """Bumble rejects every L2CAP configuration option it does not implement. The AirPods' Configure
    Request for the AACP channel carries Flush Timeout (0x02, 30 ms): Bumble answers 'unknown option',
    the AirPods ask a few more times and then close the channel, which surfaces as
    'L2CAP/Connection failure'. Flush timeout, QoS, extended flow spec and extended window size only
    describe how the peer sends (or matter in a mode we do not use); a basic-mode receiver can accept
    them (BlueZ does). Strip them before Bumble looks.

    Everything this relies on is checked first: a Bumble release that renamed any of it gets one
    warning and Bumble's own handling, instead of a backend that cannot start."""
    import dataclasses
    try:
        from bumble import l2cap
        cls = l2cap.ClassicChannel
        orig = cls.on_configure_request
        frame = l2cap.L2CAP_Control_Frame
        decode, encode = frame.decode_configuration_options, frame.encode_configuration_options
        if "options" not in {f.name for f in dataclasses.fields(l2cap.L2CAP_Configure_Request)}:
            raise AttributeError("L2CAP_Configure_Request has no 'options' field")
    except Exception as e:                    # noqa: BLE001
        _warn_once("patch", "L2CAP configure fix not applied (this Bumble version differs: %s); AirPods "
                   "whose Configure Request carries a flush timeout may refuse the motion channel", errtext(e))
        return False
    if getattr(orig, "_fbt_patched", False):
        return True

    def on_configure_request(self, request):
        try:
            raw = bytes(request.options)
            seen = (str(getattr(getattr(self, "connection", None), "peer_address", "?")), getattr(self, "psm", None))
            if seen not in _configure_seen and len(_configure_seen) < 64:
                # what each model asks for, once: AirPods Pro 2 / 3 / 4 reports can be explained
                _configure_seen.add(seen)
                log.info("L2CAP configure request from %s (PSM %s): options %s", seen[0], seen[1], raw.hex())
            opts = decode(raw)
            keep, dropped = _filter_configure_options(opts)
            if dropped or any(int(t) & 0x80 for t, _ in opts):         # something dropped or un-hinted
                if dropped:
                    log.debug("L2CAP configure: accepting %s",
                              ", ".join(f"0x{int(t):02x}={bytes(v).hex()}" for t, v in dropped))
                request = dataclasses.replace(request, options=encode(keep))
        except Exception as e:                # noqa: BLE001 - fall back to Bumble's own handling
            _warn_once("filter", "L2CAP configure options not filtered: %s", errtext(e))
        return orig(self, request)
    on_configure_request._fbt_patched = True
    cls.on_configure_request = on_configure_request
    return True


def _log_l2cap_signaling():
    lg = logging.getLogger("bumble.l2cap")
    if not any(isinstance(f, _SignalingLog) for f in lg.filters):
        lg.addFilter(_SignalingLog())
    lg.setLevel(logging.DEBUG)


class AdapterLost(ConnectionError):
    """The USB Bluetooth adapter went away (unplugged, reset by Windows, a USB error): nothing can
    be sent through this backend any more. Opening a new backend is the only way on."""


# shown to the user (in the pairing screen's error, or as the reason a link keeps failing)
ADAPTER_LOST_TEXT = "藍芽已中斷連線（USB藍芽被拔掉，或被 Windows 重置）：確認插好後再試一次"
# the adapter still has Windows' own Bluetooth driver (the "Set up Bluetooth" step was not done, or
# Windows put its driver back): libusb only says LIBUSB_ERROR_NOT_SUPPORTED
NEEDS_DRIVER_TEXT = "USB藍芽還在用 Windows 的驅動程式：請先在總覽按「準備藍芽」，把它設定給 AirPods 用"
KEY_NOT_SAVED_TEXT = "已配對，但配對資料無法存檔（檔案可能被防毒軟體或同步程式鎖住）：請再配對一次"


def _cancel_requested():
    """True when this task was asked to stop and still runs: on Python 3.11 asyncio.wait_for can
    swallow a cancellation that arrives just as the awaited operation completes."""
    task = asyncio.current_task()
    return bool(task is not None and getattr(task, "cancelling", lambda: 0)())


async def _hang_up(conn, drain_s=1.0):
    """Disconnect an ACL link, after the data already queued on it has left (the goodbye packets a
    link sends just before it closes: sensors off, stem presses and ear detection given back).
    Best effort: errors are ignored, a cancellation is not."""
    drain = getattr(conn, "drain", None)          # Bumble >= 0.0.236
    if drain is not None:
        try:
            await asyncio.wait_for(drain(), drain_s)
        except Exception:                          # noqa: BLE001 - disconnect anyway
            pass
    try:
        await conn.disconnect()
    except Exception:                              # noqa: BLE001
        pass


class BumbleChannel(Channel):
    abandoned = False                  # the open() that made it gave up (see BumbleBackend.open)

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
        try:
            self.backend.loop.call_soon_threadsafe(self.l2.send_pdu, bytes(pkt))
        except RuntimeError as e:          # the backend was closed: its loop is gone
            raise ConnectionError("channel closed") from e

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
            # what was sent just before (the goodbye packets) is only queued: a disconnect right
            # away can overtake it, and the AirPods would keep ear detection off
            try:                           # the stack may already be shutting down
                self.backend.run(_hang_up(conn), timeout=5)
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


def _key_store(path):
    """Bumble's JSON key store, saving the way config._atomic_write does. Its own save replaces the
    file once; on Windows a virus scanner, the indexer or a sync client that has the fresh file open
    makes that fail with a sharing violation, Bumble only logs it, and the new bond is gone although
    pairing "succeeded"."""
    import time
    from bumble.keys import JsonKeyStore

    class RetryingKeyStore(JsonKeyStore):
        async def save(self, db):
            for attempt in range(5):
                try:
                    return await super().save(db)
                except PermissionError:
                    if attempt == 4:
                        raise
                    # a blocking wait on purpose (rare, 0.5 s at most): yielding here would let a
                    # second update read the file before this one is written, and one would be lost
                    time.sleep(0.05 * (attempt + 1))

    return RetryingKeyStore(namespace=None, filename=path)


async def _bond_saved(keystore, addr, wait=2.0):
    """True once the key store holds a link key for `addr` (Bumble stores it in a task of its own,
    shortly after authentication). True as well when there is no key store to ask."""
    if keystore is None:
        return True
    want = str(addr).upper().split("/")[0]
    loop = asyncio.get_running_loop()
    end = loop.time() + wait
    while True:
        try:
            if any(str(k).upper().split("/")[0] == want for k, _ in await keystore.get_all()):
                return True
        except Exception as e:             # noqa: BLE001 - unreadable file: same as no key
            log.debug("key store: %s", errtext(e))
        if loop.time() >= end:
            return False
        await asyncio.sleep(0.1)


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
    NAME_LOOKUP_S = 20.0           # all name requests of one discover() together (its budget is +30 s)

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
        self.lost = False                 # the USB adapter went away (see _transport_lost)
        self._closing = self._closed = False
        self._inflight = set()            # run() calls other threads are waiting on
        use_bundled_firmware()
        try:
            self.run(self._start(), timeout=90)   # Intel firmware upload takes a few seconds
        except BaseException:
            self.close()
            raise

    # ---------------------------------------------------------------- loop helpers
    def _gone(self):
        """The error for a call that can no longer be made (or was ended by close / adapter loss)."""
        if getattr(self, "lost", False):
            return AdapterLost(ADAPTER_LOST_TEXT)
        return ConnectionError("Bluetooth closed")

    def run(self, coro, timeout=None):
        if getattr(self, "lost", False) or getattr(self, "_closed", False):
            getattr(coro, "close", lambda: None)()    # (never started: no "never awaited" warning)
            raise self._gone()
        return self._submit(coro, timeout, guard=True)

    def _submit(self, coro, timeout=None, guard=False):
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        except RuntimeError:              # close() in another thread closed the loop after run()'s check
            getattr(coro, "close", lambda: None)()
            raise self._gone() from None
        # set.add / discard / list() are atomic under the GIL: no lock needed between link threads
        inflight = self.__dict__.setdefault("_inflight", set())
        inflight.add(fut)
        if guard and (getattr(self, "lost", False) or getattr(self, "_closed", False)):
            # the adapter was lost / the backend closed between run()'s check and the line above:
            # the calls ended then did not include this one, which would wait out its whole timeout
            fut.cancel()
        try:
            return fut.result(timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            fut.cancel()                  # do not leave it running (and connecting) behind our back
            raise
        except (concurrent.futures.CancelledError, asyncio.CancelledError):
            if getattr(self, "lost", False) or getattr(self, "_closing", False):
                raise self._gone() from None      # ended by close() / the adapter going away
            raise
        finally:
            inflight.discard(fut)

    def _transport_lost(self, fut=None):
        """(on the loop) Bumble's USB source ended: the dongle was unplugged or reset, or a USB
        transfer failed. Bumble then drops every link, but a new open() would send its commands to
        the dead device and wait for answers until run() gives up (60 s, every retry, for ever).
        From now on every call fails at once with AdapterLost, and those already waiting are ended."""
        if fut is not None and not fut.cancelled():
            fut.exception()               # (retrieved: no "exception was never retrieved" noise)
        if getattr(self, "_closing", False) or getattr(self, "lost", False):
            return                        # our own close(), or already said
        self.lost = True
        log.warning("Bluetooth adapter lost (unplugged, reset by Windows or a USB error): "
                    "it has to be opened again")
        for f in list(getattr(self, "_inflight", ())):
            f.cancel()

    async def _open_usb(self, open_transport):
        """open_transport, and for 'usb:VID:PID' the next adapters of the same model when the first
        one cannot be used: Bumble takes the first device with that vendor:product id, which may be
        an identical dongle that Windows' own Bluetooth driver keeps (libusb cannot open it:
        LIBUSB_ERROR_NOT_SUPPORTED). Any other error is raised as it is."""
        name = self.transport_name
        try:
            return await open_transport(name)
        except Exception as first:        # noqa: BLE001
            if "NOT_SUPPORTED" not in str(first).upper():
                raise
            if not re.fullmatch(r"usb:(\[[^\]]*\])?[0-9A-Fa-f]{4}:[0-9A-Fa-f]{4}", name or ""):
                raise ConnectionError(NEEDS_DRIVER_TEXT) from first
            for i in range(1, 4):
                try:
                    hci = await open_transport(f"{name}#{i}")
                except Exception as e:    # noqa: BLE001
                    if "not found" in str(e).lower():
                        break             # no more adapters of this model
                    continue
                log.warning("%s: the first adapter of this model cannot be opened (%s), probably the one "
                            "Windows uses; using the next one (#%d)", name, errtext(first), i)
                return hci
            raise ConnectionError(NEEDS_DRIVER_TEXT) from first

    async def _start(self):
        from bumble.device import Device
        from bumble.hci import Address
        from bumble.transport import open_transport

        os.makedirs(os.path.dirname(self.keystore_path), exist_ok=True)
        fw_log = self._fw_log = _FirmwareLog(self)
        for n in ("bumble.drivers.rtk", "bumble.drivers.intel"):
            logging.getLogger(n).addHandler(fw_log)
            if logging.getLogger(n).getEffectiveLevel() > logging.INFO:
                logging.getLogger(n).setLevel(logging.INFO)
        self.hci = await self._open_usb(open_transport)
        terminated = getattr(getattr(self.hci, "source", None), "terminated", None)
        if isinstance(terminated, asyncio.Future):     # set by Bumble when the USB device goes away
            terminated.add_done_callback(self._transport_lost)
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
        dev.keystore = _key_store(self.keystore_path)
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
            audio = set()

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
                try:
                    if (int(cod) >> 8) & 0x1F == 0x04:      # major class Audio/Video: AirPods are
                        audio.add(key)
                except (TypeError, ValueError):
                    pass

            self.device.on("inquiry_result", on_result)
            try:
                await self.device.start_discovery(auto_restart=False)
                await asyncio.sleep(seconds)
            finally:                       # also when the inquiry could not start
                try:
                    await self.device.stop_discovery()
                except Exception:   # noqa: BLE001
                    pass
                self.device.remove_listener("inquiry_result", on_result)
            # No name in the inquiry response: ask the device (needs the Address object, not its
            # text), one after the other, up to 5 s each. Audio devices first and all of it within
            # NAME_LOOKUP_S: in a crowded place a few silent devices used to run past run()'s budget,
            # and everything found (the AirPods too) was lost with the timeout.
            loop = asyncio.get_running_loop()
            end = loop.time() + self.NAME_LOOKUP_S
            names = {}
            for key in sorted((k for k, (_, n) in found.items() if not n), key=lambda k: k not in audio):
                left = end - loop.time()
                if left < 0.5:
                    log.debug("no time left to ask %s for its name", key)
                    continue
                try:
                    names[key] = await asyncio.wait_for(self.device.request_remote_name(found[key][0]), min(5.0, left))
                except Exception as e:   # noqa: BLE001
                    log.debug("name of %s: %s", key, errtext(e))
            return [(key.split("/")[0], name or names.get(key) or "") for key, (address, name) in found.items()]

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
                except Exception:          # noqa: BLE001 - never a cancellation: when run() has given
                    pass                   # up, going on would leave a link nobody owns or closes
                await asyncio.sleep(0.5)
            if _cancel_requested():
                raise asyncio.CancelledError()
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
            except Exception:              # noqa: BLE001 - best effort
                pass
            # Authentication succeeding is not enough: the old key was deleted above, and a new one
            # that could not be written to the file leaves no bond at all. Say so now, while the
            # AirPods are still in pairing mode and pairing again can work.
            if not await _bond_saved(getattr(self.device, "keystore", None), addr):
                log.warning("pairing with %s: the link key was not saved to %s", addr,
                            getattr(self, "keystore_path", "?"))
                raise RuntimeError(KEY_NOT_SAVED_TEXT)
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
                # Looked up once, without waiting: the key was stored right after authentication,
                # long before the channel got here, and waiting would leave the open channel with
                # nobody listening for its data or its close (wired up just below) for 2 s
                if not await _bond_saved(getattr(self.device, "keystore", None), addr, wait=0):
                    # the channel works now, but the next start will need pairing again
                    log.warning("%s: no link key in %s after connecting: the bond was not saved", addr,
                                getattr(self, "keystore_path", "?"))
                if ch.abandoned or _cancel_requested():
                    raise asyncio.CancelledError()   # run() gave up meanwhile: nobody would use it
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

        try:
            self.run(_open(), timeout=60)
        except BaseException:
            self._abandon(ch)
            raise
        log.info("AACP channel open with %s", addr)
        return ch

    def _abandon(self, ch):
        """open() gave up on `ch` (timeout, close, adapter lost). The coroutine may still be going
        on, or have finished just as the time ran out: then it holds a link marked as ours and an
        open channel that nobody will read or close, and every retry to these AirPods fails with
        "Connection Already Exists". Settled on the loop, where _open's own steps run, so there is no
        race: either _open has finished (hang up here) or it sees the flag / its cancellation."""
        def settle():
            ch.abandoned = True
            if ch.l2 is None:
                return                         # _open still runs: it hangs up itself
            conn, ch.conn = ch.conn, None
            ch.closed = True
            if conn is not None:
                getattr(self, "_owned", set()).discard(conn)
                log.info("%s: hanging up a link that was opened after open() gave up", ch.addr)
                self.loop.create_task(_hang_up(conn, 0.2))
        try:
            self.loop.call_soon_threadsafe(settle)
        except RuntimeError:                   # loop closed: the backend is gone, and its links
            pass

    async def _hang_up_all(self):
        """Before the adapter is let go: disconnect every link still up (a link thread that did not
        get to close its channel, or one still inside open()), and reset the controller for anything
        else (a page in progress). Releasing the USB device does neither, and the dongle stays
        powered: the AirPods would keep a connection to a host that is gone."""
        dev = self.device
        conns = list((getattr(dev, "connections", None) or {}).values())
        if conns:
            try:
                await asyncio.wait_for(asyncio.gather(*(_hang_up(c, 0.5) for c in conns)), 1.5)
            except Exception as e:             # noqa: BLE001
                log.debug("hang up at close: %s", errtext(e))
        try:
            from bumble import hci
            await asyncio.wait_for(dev.host.send_command(hci.HCI_Reset_Command()), 1)
        except Exception as e:                 # noqa: BLE001
            log.debug("controller reset at close: %s", errtext(e))

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closing = True

        async def _stop():
            if self.device is not None and not getattr(self, "lost", False):
                await self._hang_up_all()
            if self.hci:
                await self.hci.close()
        try:
            self._submit(_stop(), timeout=5)       # (also after the adapter was lost: libusb's handles)
        except Exception:   # noqa: BLE001
            pass
        fw_log = getattr(self, "_fw_log", None)
        if fw_log is not None:             # one handler per backend would pile up on the loggers
            for n in ("bumble.drivers.rtk", "bumble.drivers.intel"):
                logging.getLogger(n).removeHandler(fw_log)
            self._fw_log = None
        self._closed = True
        # calls other threads still wait on (a link inside open()) would wait out their whole
        # timeout on a loop that no longer runs: end them now (they raise ConnectionError)
        for fut in list(getattr(self, "_inflight", ())):
            fut.cancel()
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError:               # a close() in another thread got there first
            return
        th = getattr(self, "thread", None)
        if th is not None and th is not threading.current_thread():
            th.join(2)
            if not th.is_alive():
                self.loop.close()          # (the Proactor loop's handles and sockets)
