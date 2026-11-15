"""Windows Bluetooth back-end (Bumble + USB dongle) and its setup: regressions from the review.
Simulated AirPods run on Bumble's virtual controllers; nothing here touches real hardware."""
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import os
import shutil
import sys
import threading
import time
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
pytest.importorskip("bumble")

import bumble.transport                                    # noqa: E402
from bumble.controller import Controller                   # noqa: E402
from bumble.device import Device                           # noqa: E402
from bumble.hci import Address                             # noqa: E402
from bumble.l2cap import ClassicChannelSpec                # noqa: E402
from bumble.link import LocalLink                          # noqa: E402

import airpodsvr.transport.bumble_usb as bu                # noqa: E402
from airpodsvr import winsetup as W                        # noqa: E402

PODS = "AA:BB:CC:DD:EE:FF"


def wait(cond, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


# ------------------------------------------------------------------ helpers
def _backend(device=None):
    """A backend whose loop runs, without a dongle (device is a stand-in)."""
    b = bu.BumbleBackend.__new__(bu.BumbleBackend)
    b.loop = asyncio.new_event_loop()
    b.thread = threading.Thread(target=b.loop.run_forever, daemon=True)
    b.thread.start()
    b.hci = None
    b.device = device
    return b


class _Conn:
    is_encrypted = True

    def __init__(self, disconnect_s=0.0):
        self.disconnected, self.handlers, self.disconnect_s = 0, {}, disconnect_s

    def on(self, name, fn):
        self.handlers[name] = fn

    async def authenticate(self):
        pass

    async def create_l2cap_channel(self, spec):
        self.l2 = types.SimpleNamespace(on=lambda n, f: self.handlers.__setitem__("l2." + n, f), sink=None,
                                        send_pdu=lambda p: None)
        return self.l2

    async def disconnect(self):
        if self.disconnect_s:
            await asyncio.sleep(self.disconnect_s)
        self.disconnected += 1


class FakeTransport:
    def __init__(self, c):
        self.source = c
        self.sink = c

    async def close(self):
        pass


def _sim_backend(tmp_path, monkeypatch, link, watch_lost=False):
    """BumbleBackend._start against a virtual controller (as tests/test_bumble_sim.py)."""
    async def fake_open_transport(name):
        c = Controller('host', link=link, public_address='11:22:33:44:55:66')
        if watch_lost:                    # what Bumble's USB source has: resolved when the USB device goes
            c.terminated = asyncio.get_running_loop().create_future()
        return FakeTransport(c)
    monkeypatch.setattr(bumble.transport, "open_transport", fake_open_transport)

    async def secure(self, conn):         # the virtual controller cannot encrypt: skip its 10 s wait
        await conn.authenticate()
    monkeypatch.setattr(bu.BumbleBackend, "_secure", secure)
    be = bu.BumbleBackend.__new__(bu.BumbleBackend)
    be.cfg = {}
    be.transport_name = "sim"
    be.keystore_path = str(tmp_path / "keys.json")
    be.loop = asyncio.new_event_loop()
    be.thread = threading.Thread(target=be.loop.run_forever, daemon=True)
    be.thread.start()
    be.device = None
    be.hci = None
    be.run(be._start(), timeout=20)
    return be


async def _make_airpods(link, rx):
    c = Controller('pods', link=link, public_address=PODS)
    dev = Device.with_hci('AirPods', Address(PODS), c, c)
    dev.classic_enabled = True
    await dev.power_on()
    await dev.set_discoverable(True)
    await dev.set_connectable(True)

    def on_ch(ch):
        ch.sink = lambda pdu: rx.append(bytes(pdu))
    dev.create_l2cap_server(spec=ClassicChannelSpec(psm=0x1001, mtu=2582), handler=on_ch)
    return dev


# ================================================================== G2-2: a given-up open()
def test_open_given_up_while_dropping_a_stale_link_does_not_connect_afterwards():
    """run()'s timeout cancelled _connect inside the stale-link disconnect; `except BaseException`
    swallowed it and the coroutine went on to page, pair and own a link nobody would ever close."""
    stale = _Conn(disconnect_s=0.8)
    calls = []

    async def connect(addr, transport=None, timeout=None):
        calls.append(addr)
        return _Conn()

    def find(addr, transport=None):
        return stale if not calls else None
    b = _backend(types.SimpleNamespace(connect=connect, find_connection_by_bd_addr=find))
    real = b.run
    b.run = lambda coro, timeout=None: real(coro, 0.3 if timeout == 60 else timeout)   # open()'s 60 s, scaled
    try:
        with pytest.raises(TimeoutError):
            b.open(PODS)
        time.sleep(1.8)                   # the stale disconnect would be over, and the page started
        assert calls == []
        assert not getattr(b, "_owned", set())
    finally:
        b.close()


def test_open_that_finished_just_after_run_gave_up_hangs_its_link_up():
    """The coroutine can complete between run()'s timeout and the cancellation reaching it: the link
    was marked as ours with an open channel nobody reads, and every retry hit 'Connection Already
    Exists'."""
    conn = _Conn()

    async def connect(addr, transport=None, timeout=None):
        return conn
    b = _backend(types.SimpleNamespace(connect=connect))
    real = b.run

    def late(coro, timeout=None):
        real(coro, 5)                     # it does get through ...
        raise TimeoutError()              # ... just as the caller gives up
    b.run = late
    try:
        with pytest.raises(TimeoutError):
            b.open(PODS)
        assert wait(lambda: conn.disconnected == 1)
        assert conn not in getattr(b, "_owned", set())
    finally:
        b.close()


def test_open_whose_cancellation_was_lost_hangs_up_instead_of_keeping_the_link():
    """On Python 3.11 asyncio.wait_for can swallow a cancellation: _open then carries on. Once the
    caller has given up it must hang up instead of registering the link."""
    conn = _Conn()
    started = threading.Event()

    async def slow_channel(spec):
        started.set()
        await asyncio.sleep(0.4)
        return await _Conn.create_l2cap_channel(conn, spec)
    conn.create_l2cap_channel = slow_channel

    async def connect(addr, transport=None, timeout=None):
        return conn
    b = _backend(types.SimpleNamespace(connect=connect))

    def lost_cancel(coro, timeout=None):
        asyncio.run_coroutine_threadsafe(coro, b.loop)   # never cancelled: as if the cancel was swallowed
        started.wait(2)
        raise TimeoutError()
    b.run = lost_cancel
    try:
        with pytest.raises(TimeoutError):
            b.open(PODS)
        assert wait(lambda: conn.disconnected == 1)
        assert conn not in getattr(b, "_owned", set())
    finally:
        b.close()


# ================================================================== G2-3: the dongle goes away
def test_lost_adapter_fails_every_call_at_once_and_ends_waiting_ones():
    async def connect(addr, transport=None, timeout=None):
        await asyncio.sleep(30)           # a page sent to a dongle that is gone: never answered
    b = _backend(types.SimpleNamespace(connect=connect))
    res = {}

    def opener():
        t0 = time.monotonic()
        try:
            b.open(PODS)
        except BaseException as e:        # noqa: BLE001
            res["e"], res["t"] = e, time.monotonic() - t0
    th = threading.Thread(target=opener)
    th.start()
    try:
        time.sleep(0.3)
        b.loop.call_soon_threadsafe(b._transport_lost, None)
        th.join(3)
        assert isinstance(res.get("e"), bu.AdapterLost) and res["t"] < 2       # not after 60 s
        assert isinstance(res["e"], ConnectionError)          # AirPodsLink's "link lost" path
        assert b.lost
        for call in (lambda: b.open(PODS), lambda: b.pair(PODS), lambda: b.discover(1)):
            t0 = time.monotonic()
            with pytest.raises(bu.AdapterLost) as e:
                call()
            assert time.monotonic() - t0 < 0.5
            assert str(e.value) == bu.ADAPTER_LOST_TEXT
    finally:
        b.close()


def test_unplugged_dongle_is_noticed_through_bumbles_transport(tmp_path, monkeypatch):
    link, rx = LocalLink(), []
    be = _sim_backend(tmp_path, monkeypatch, link, watch_lost=True)
    try:
        be.run(_make_airpods(link, rx), timeout=20)
        ch = be.open(PODS)
        assert not getattr(be, "lost", False)

        class DeadUsb:
            def on_packet(self, pkt):
                pass

        async def unplug():                # what bumble/transport/usb.py does when the IN transfer fails
            be.device.host.hci_sink = DeadUsb()
            be.device.host.on_transport_lost()
            be.hci.source.terminated.set_result(None)
        asyncio.run_coroutine_threadsafe(unplug(), be.loop).result(5)   # (its own run() would be ended too)
        with pytest.raises(ConnectionError):
            ch.recv(2)
        assert wait(lambda: getattr(be, "lost", False))
        t0 = time.monotonic()
        with pytest.raises(bu.AdapterLost):
            be.open(PODS)
        assert time.monotonic() - t0 < 0.5
        ch.close()                         # nothing to send to: no wait, no error
    finally:
        be.close()
    assert be.loop.is_closed()


# ================================================================== G2-4: the bond must be on disk
def test_pairing_survives_a_sharing_violation_on_the_key_file(tmp_path, monkeypatch):
    link, rx = LocalLink(), []
    be = _sim_backend(tmp_path, monkeypatch, link)
    real = os.replace
    hits = {"n": 0}

    def flaky(src, dst):                   # Defender / the indexer / a sync client has the file open
        if str(dst).endswith("keys.json") and hits["n"] == 0:
            hits["n"] += 1
            raise PermissionError(32, "The process cannot access the file", str(dst))
        return real(src, dst)
    try:
        be.run(_make_airpods(link, rx), timeout=20)
        monkeypatch.setattr(os, "replace", flaky)
        be.pair(PODS)
        assert hits["n"] == 1
        assert [a for a, _ in be.list_paired()] == [PODS]
    finally:
        be.close()


def test_pairing_says_so_when_the_key_cannot_be_saved(tmp_path, monkeypatch):
    link, rx = LocalLink(), []
    be = _sim_backend(tmp_path, monkeypatch, link)
    real = os.replace

    def locked(src, dst):
        if str(dst).endswith("keys.json"):
            raise PermissionError(32, "The process cannot access the file", str(dst))
        return real(src, dst)
    try:
        be.run(_make_airpods(link, rx), timeout=20)
        monkeypatch.setattr(os, "replace", locked)
        with pytest.raises(RuntimeError) as e:
            be.pair(PODS)                  # used to log "paired" with no bond left at all
        assert str(e.value) == bu.KEY_NOT_SAVED_TEXT
    finally:
        be.close()


def test_open_without_a_saved_key_warns_without_holding_the_channel_unwired(caplog):
    """The bond check after the channel opened waited up to 2 s for a key that was not there: the
    channel was open meanwhile with nothing listening for its data or its close."""
    conn = _Conn()

    async def connect(addr, transport=None, timeout=None):
        return conn

    class NoKeys:
        async def get_all(self):
            return []
    b = _backend(types.SimpleNamespace(connect=connect, keystore=NoKeys()))
    try:
        with caplog.at_level(logging.WARNING, logger=bu.log.name):
            t0 = time.monotonic()
            ch = b.open(PODS)
            assert time.monotonic() - t0 < 1.0
        assert "the bond was not saved" in caplog.text
        assert ch.l2 is conn.l2 and "l2.close" in conn.handlers
    finally:
        b.close()


# ================================================================== G2-8: closing the backend
def test_close_hangs_up_links_resets_the_controller_and_ends_the_loop(tmp_path, monkeypatch):
    link, rx = LocalLink(), []
    be = _sim_backend(tmp_path, monkeypatch, link)
    pods = be.run(_make_airpods(link, rx), timeout=20)
    ch = be.open(PODS)                     # a link whose thread did not get to close it
    sent = []
    orig = be.device.host.send_hci_packet
    be.device.host.send_hci_packet = lambda p: (sent.append(type(p).__name__), orig(p))
    assert pods.connections
    be.close()
    assert "HCI_Disconnect_Command" in sent and "HCI_Reset_Command" in sent
    assert not pods.connections            # the AirPods know the host is gone
    assert not be.thread.is_alive() and be.loop.is_closed()
    with pytest.raises(ConnectionError):
        ch.recv(1)
    with pytest.raises(ConnectionError):
        ch.send(b"\x00")
    t0 = time.monotonic()
    with pytest.raises(ConnectionError):
        be.open(PODS)
    assert time.monotonic() - t0 < 0.5
    be.close()                             # a second close is harmless


def test_close_ends_an_open_still_waiting_in_another_thread():
    async def connect(addr, transport=None, timeout=None):
        await asyncio.sleep(30)
    b = _backend(types.SimpleNamespace(connect=connect))
    res = {}

    def opener():
        t0 = time.monotonic()
        try:
            b.open(PODS)
        except BaseException as e:        # noqa: BLE001
            res["e"], res["t"] = e, time.monotonic() - t0
    th = threading.Thread(target=opener)
    th.start()
    time.sleep(0.3)
    b.close()
    th.join(3)
    assert isinstance(res.get("e"), ConnectionError) and not isinstance(res["e"], bu.AdapterLost)
    assert res["t"] < 3                    # not open()'s 60 s


def test_channel_close_lets_queued_packets_leave_before_hanging_up():
    """The goodbye (sensors off, ear detection back on) is sent right before close(): a disconnect
    that overtakes it leaves the AirPods with ear detection off."""
    order = []

    class Conn:
        async def drain(self):
            order.append("drain")
            await asyncio.sleep(0.05)
            order.append("drained")

        async def disconnect(self):
            order.append("disconnect")

    class Stuck(Conn):
        async def drain(self):
            await asyncio.sleep(30)        # a controller that never reports the packets done
    b = _backend()
    try:
        ch = bu.BumbleChannel(b, PODS)
        ch.conn = Conn()
        ch.close()
        assert order == ["drain", "drained", "disconnect"]
        ch = bu.BumbleChannel(b, PODS)
        ch.conn = Stuck()
        t0 = time.monotonic()
        ch.close()
        assert order[-1] == "disconnect" and len(order) == 4 and time.monotonic() - t0 < 3
    finally:
        b.close()


# ================================================================== G2-9: discovery in a crowd
def test_discovery_returns_what_it_found_when_name_requests_take_too_long():
    pods = Address("AA:BB:CC:DD:EE:01")
    phones = [Address(f"10:00:00:00:00:{i:02X}") for i in range(8)]
    asked = []

    class Dev:
        def on(self, name, fn):
            self.cb = fn

        def remove_listener(self, name, fn):
            pass

        async def start_discovery(self, auto_restart=False):
            for a in phones:
                self.cb(a, 0x5A020C, {}, -60)     # major class Phone, no name in the response
            self.cb(pods, 0x240418, {}, -40)      # major class Audio/Video, no name either

        async def stop_discovery(self):
            pass

        async def request_remote_name(self, remote):
            asked.append(str(remote))
            if str(remote).startswith("AA:BB:CC:DD:EE:01"):
                return "AirPods Pro"
            await asyncio.sleep(5)                 # out of reach: answers only at the 5 s timeout
    b = _backend(Dev())
    b.NAME_LOOKUP_S = 1.0                          # (20 s in the app, within run()'s seconds + 30)
    try:
        t0 = time.monotonic()
        found = dict(b.discover(0.05))             # 8 x 5 s used to overrun the budget: nothing at all
        assert time.monotonic() - t0 < 3
        assert found["AA:BB:CC:DD:EE:01"] == "AirPods Pro"
        assert len(found) == 9
        assert asked[0].startswith("AA:BB:CC:DD:EE:01")   # audio devices are asked first
    finally:
        b.close()


def test_discovery_that_cannot_start_removes_its_listener():
    state = {}

    class Dev:
        def on(self, name, fn):
            state["on"] = fn

        def remove_listener(self, name, fn):
            state["removed"] = fn

        async def start_discovery(self, auto_restart=False):
            raise RuntimeError("Command Disallowed")

        async def stop_discovery(self):
            state["stopped"] = True
    b = _backend(Dev())
    try:
        with pytest.raises(RuntimeError):
            b.discover(0.05)
        assert state.get("removed") is state["on"] and state.get("stopped")
    finally:
        b.close()


# ================================================================== G2-M1: two identical dongles
def test_second_identical_dongle_is_tried_when_the_first_is_windows_own():
    b = _backend()
    tried = []

    async def fake_open(name):
        tried.append(name)
        if name.endswith("#1"):
            return "transport #1"
        raise Exception("LIBUSB_ERROR_NOT_SUPPORTED [-12]")   # libusb on a device of Windows' driver

    async def only_one(name):
        tried.append(name)
        if "#" in name:
            raise Exception("device not found")
        raise Exception("LIBUSB_ERROR_NOT_SUPPORTED [-12]")

    async def busy(name):
        tried.append(name)
        raise Exception("LIBUSB_ERROR_ACCESS [-3]")
    try:
        b.transport_name = "usb:[driver=rtk]0BDA:8771"
        assert b.run(b._open_usb(fake_open), 5) == "transport #1"
        assert tried == ["usb:[driver=rtk]0BDA:8771", "usb:[driver=rtk]0BDA:8771#1"]
        tried.clear()
        b.transport_name = "usb:0BDA:8771"
        with pytest.raises(ConnectionError) as e:                 # no second one: what to do about it
            b.run(b._open_usb(only_one), 5)
        assert "NOT_SUPPORTED" in str(e.value.__cause__)
        assert tried == ["usb:0BDA:8771", "usb:0BDA:8771#1"]
        tried.clear()
        with pytest.raises(Exception, match="ACCESS"):            # anything else: as it is
            b.run(b._open_usb(busy), 5)
        assert tried == ["usb:0BDA:8771"]
        tried.clear()
        b.transport_name = "usb:0"
        with pytest.raises(ConnectionError) as e:                 # not a vendor:product id: no #1 to try
            b.run(b._open_usb(fake_open), 5)
        assert "NOT_SUPPORTED" in str(e.value.__cause__)
        assert tried == ["usb:0"]
    finally:
        b.close()


# ================================================================== L2CAP configure options
def test_configure_filter_skips_hints_and_send_side_options():
    opts = [(0x01, b"\x16\x0a"), (0x02, b"\x1e\x00"), (0x07, b"\x20\x00"), (0x82, b"\x1e\x00"),
            (0x99, b"\xab\xcd"), (0x81, b"\x00\x04"), (0x04, bytes(9)), (0x05, b"\x01")]
    keep, dropped = bu._filter_configure_options(opts)
    assert [(int(t), v) for t, v in keep] == [(0x01, b"\x16\x0a"), (0x01, b"\x00\x04"), (0x04, bytes(9)),
                                              (0x05, b"\x01")]
    assert [int(t) for t, _ in dropped] == [0x02, 0x07, 0x82, 0x99]


def test_configure_patch_degrades_to_one_warning_on_another_bumble(monkeypatch, caplog):
    import bumble.l2cap as l2cap
    monkeypatch.setattr(bu, "_patch_warned", set())
    monkeypatch.setattr(l2cap, "ClassicChannel", type("ClassicChannel", (), {}))   # renamed method
    with caplog.at_level(logging.WARNING, logger=bu.log.name):
        assert bu._patch_l2cap_configure() is False
        assert bu._patch_l2cap_configure() is False
    assert caplog.text.count("L2CAP configure fix not applied") == 1
    monkeypatch.undo()
    assert bu._patch_l2cap_configure() is True


def test_airpods_configure_request_with_hints_and_window_size_is_accepted(tmp_path, monkeypatch):
    """Newer firmware (Pro 2 / Pro 3) may add more options: unknown hints and Extended Window Size
    were answered 'unknown option', and the buds closed the motion channel."""
    import bumble.l2cap as l2cap
    link, rx = LocalLink(), []
    options = bytes.fromhex("0102160a" "02021e00" "07022000" "9902abcd" "0409000000000000000000")
    orig_send = l2cap.ClassicChannel.send_configure_request

    def airpods_send(self):
        if str(self.connection.self_address).startswith(PODS):
            self.send_control_frame(l2cap.L2CAP_Configure_Request(
                identifier=self.manager.next_identifier(self.connection),
                destination_cid=self.destination_cid, flags=0, options=options))
            return
        return orig_send(self)
    monkeypatch.setattr(l2cap.ClassicChannel, "send_configure_request", airpods_send)
    answers = []
    orig_rsp = l2cap.ClassicChannel.on_configure_response

    def watch(self, response):
        answers.append(response.result)
        return orig_rsp(self, response)
    monkeypatch.setattr(l2cap.ClassicChannel, "on_configure_response", watch)
    be = _sim_backend(tmp_path, monkeypatch, link)
    try:
        be.run(_make_airpods(link, rx), timeout=20)
        ch = be.open(PODS)
        assert l2cap.L2CAP_Configure_Response.Result.FAILURE_UNKNOWN_OPTIONS not in answers
        ch.send(b"\x04\x00\x04\x00hello")
        assert wait(lambda: rx)
        ch.close()
    finally:
        be.close()


# ================================================================== G2-5: PowerShell output encoding
def test_device_list_is_read_as_utf8(monkeypatch):
    seen = {}
    devs = [{"id": "USB\\VID_0BDA&PID_8771\\00E04C239987", "name": "Bluetooth-Gerät für USB",
             "service": "WinUSB", "inf": "oem7.inf", "status": "OK"},
            {"id": "USB\\VID_0A12&PID_0001\\5&1", "name": "藍牙 USB 介面卡", "service": "BTHUSB", "inf": "bth.inf",
             "status": "OK"}]

    def fake_run(args, **kw):
        seen.update(kw, script=args[-1])
        return types.SimpleNamespace(stdout="\ufeff" + json.dumps(devs, ensure_ascii=False) + "\r\n")
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    monkeypatch.setattr(W, "IS_WIN", True)
    assert [d.name for d in W.list_bt_usb()] == ["Bluetooth-Gerät für USB", "藍牙 USB 介面卡"]
    assert seen["encoding"] == "utf-8" and seen["errors"] == "replace" and "text" not in seen
    assert seen["script"].startswith("[Console]::OutputEncoding")
    # a reader thread that failed leaves stdout None: "nothing found", not AttributeError
    monkeypatch.setattr(W.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=None))
    assert W.list_bt_usb() == []


@pytest.mark.skipif(sys.platform != "win32" or not shutil.which("powershell"), reason="Windows PowerShell")
def test_real_powershell_output_keeps_accents_and_cjk():
    out = W._powershell("Write-Output ('G' + [char]0x00E9 + 'n' + [char]0x00FC + ' ' + [char]0x85CD + [char]0x7259)")
    assert out == "Génü 藍牙"


def test_an_adapter_still_on_the_windows_driver_says_what_to_do():
    """Pair pressed before "Set up Bluetooth": the user saw only LIBUSB_ERROR_NOT_SUPPORTED [-12]."""
    import asyncio
    from airpodsvr.transport import bumble_usb

    async def refuse(name):
        raise Exception("LIBUSB_ERROR_NOT_SUPPORTED [-12]")
    for name in ("usb:0", "usb:2357:0604"):
        b = bumble_usb.BumbleBackend.__new__(bumble_usb.BumbleBackend)
        b.transport_name = name
        with pytest.raises(ConnectionError) as e:
            asyncio.run(b._open_usb(refuse))
        assert str(e.value) == bumble_usb.NEEDS_DRIVER_TEXT
