"""Validate BumbleBackend against a simulated AirPods (bumble virtual controllers, no hardware)."""
# SPDX-License-Identifier: Apache-2.0
import asyncio, os, sys, time, threading
import pytest
pytest.importorskip('bumble')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bumble.controller import Controller
from bumble.link import LocalLink
from bumble.device import Device
from bumble.hci import Address
from bumble.l2cap import ClassicChannelSpec
import bumble.transport
import airpodsvr.transport.bumble_usb as bu
from airpodsvr.protocol import rhid_frame, rhid_sub

link = LocalLink()
holder = {}

class FakeTransport:
    def __init__(self, c):
        self.source = c; self.sink = c
    async def close(self): pass

async def fake_open_transport(name):
    c = Controller('host', link=link, public_address='11:22:33:44:55:66')
    return FakeTransport(c)

async def make_airpods(loop_ready):
    c = Controller('pods', link=link, public_address='AA:BB:CC:DD:EE:FF')
    dev = Device.with_hci('AirPods', Address('AA:BB:CC:DD:EE:FF'), c, c)
    dev.classic_enabled = True
    await dev.power_on()
    await dev.set_discoverable(True); await dev.set_connectable(True)
    def on_ch(ch):
        def on_pdu(pdu):
            holder.setdefault('rx', []).append(bytes(pdu))
            # answer the first HostConnect with a raw accel+gyro input report pair
            if b'\x00\x02\x00\x00' in pdu and 'sent' not in holder:
                holder['sent'] = True
                pl = bytes.fromhex('00011f0901008a11000099ddfeff850a000000bb7967f87f1a00000000')
                pg = bytes.fromhex('0001545ce9ff4122010021731e00850a000000a27940fd7f1a00000000')
                ch.send_pdu(rhid_frame(1, [rhid_sub(4, 2, pl), rhid_sub(5, 2, pg)]))
        ch.sink = on_pdu
    dev.create_l2cap_server(spec=ClassicChannelSpec(psm=0x1001, mtu=2582), handler=on_ch)
    holder['pods'] = dev

from airpodsvr.protocol import HANDSHAKE_SEQUENCE, host_connect, ProtocolState


def test_bumble_backend_against_simulated_airpods(tmp_path, monkeypatch):
    monkeypatch.setattr(bumble.transport, "open_transport", fake_open_transport)
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
    be.run(make_airpods(None), timeout=20)
    ch = be.open("AA:BB:CC:DD:EE:FF")          # connect + pair + L2CAP PSM 0x1001, MTU 2582
    st = ProtocolState()
    for p, _ in HANDSHAKE_SEQUENCE:
        ch.send(p)
    ch.send(host_connect(1, 0))
    evs = []
    t0 = time.time()
    while time.time() - t0 < 3 and not any(e[0] == "imu" for e in evs):
        pkt = ch.recv(0.5)
        if pkt:
            evs += st.feed(pkt, time.monotonic())
    assert any(e[0] == "imu" for e in evs)
    assert be.list_paired()[0][0] == "AA:BB:CC:DD:EE:FF"
    ch.close()
    be.close()


def test_pair_only_and_channel_closed_during_config(tmp_path, monkeypatch):
    """Real-hardware report: 'L2capError(L2CAP/Connection failure [0x0])' while pairing. That error is
    the buds accepting PSM 0x1001 and then sending an L2CAP Disconnection Request during
    configuration. Pairing must not need the channel, and open() must ask again."""
    import bumble.l2cap as l2cap
    global link
    link = LocalLink()
    holder.clear()
    monkeypatch.setattr(bumble.transport, "open_transport", fake_open_transport)
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
    be.run(make_airpods(None), timeout=20)

    be.pair("AA:BB:CC:DD:EE:FF")                # bond only, no AACP channel
    assert be.list_paired()[0][0] == "AA:BB:CC:DD:EE:FF"
    assert not holder.get("rx")

    orig = l2cap.ClassicChannel.on_configure_request
    closes = {"n": 0}

    def flaky(self, request):
        if str(self.connection.self_address).startswith("AA:BB:CC:DD:EE:FF") and closes["n"] < 1:
            closes["n"] += 1
            self._disconnect_sync()            # the bud closes the channel it just accepted
            return
        return orig(self, request)
    monkeypatch.setattr(l2cap.ClassicChannel, "on_configure_request", flaky)
    ch = be.open("AA:BB:CC:DD:EE:FF")
    assert closes["n"] == 1
    ch.send(host_connect(1, 0))
    t0 = time.time()
    while time.time() - t0 < 3 and "sent" not in holder:
        time.sleep(0.05)
    assert "sent" in holder
    ch.close()
    be.close()


def test_airpods_configure_request_with_flush_timeout_is_accepted(tmp_path, monkeypatch):
    """Real-hardware log (AirPods, v0.3.2): the buds' Configure Request for PSM 0x1001 is
    0102160a 02021e00 0409000000000000000000 (MTU 2582, flush timeout 30 ms, basic mode). Bumble
    answered FAILURE_UNKNOWN_OPTIONS for the flush timeout and the buds closed the channel."""
    import struct as _st
    import bumble.l2cap as l2cap
    global link
    link = LocalLink()
    holder.clear()
    monkeypatch.setattr(bumble.transport, "open_transport", fake_open_transport)
    orig_send = l2cap.ClassicChannel.send_configure_request
    answers = []

    def airpods_send(self):
        if str(self.connection.self_address).startswith("AA:BB:CC:DD:EE:FF"):
            self.send_control_frame(l2cap.L2CAP_Configure_Request(
                identifier=self.manager.next_identifier(self.connection),
                destination_cid=self.destination_cid, flags=0,
                options=bytes.fromhex("0102160a02021e000409000000000000000000")))
            return
        return orig_send(self)
    monkeypatch.setattr(l2cap.ClassicChannel, "send_configure_request", airpods_send)
    orig_rsp = l2cap.ClassicChannel.on_configure_response

    def watch(self, response):
        answers.append(response.result)
        return orig_rsp(self, response)
    monkeypatch.setattr(l2cap.ClassicChannel, "on_configure_response", watch)

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
    be.run(make_airpods(None), timeout=20)
    ch = be.open("AA:BB:CC:DD:EE:FF")
    assert l2cap.L2CAP_Configure_Response.Result.FAILURE_UNKNOWN_OPTIONS not in answers
    ch.send(host_connect(1, 0))
    t0 = time.time()
    while time.time() - t0 < 3 and "sent" not in holder:
        time.sleep(0.05)
    assert "sent" in holder
    ch.close()
    be.close()
