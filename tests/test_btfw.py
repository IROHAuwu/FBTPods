# SPDX-License-Identifier: Apache-2.0
"""Firmware loaders against simulated controllers that follow the vendor protocols (as described by
the Linux drivers): Intel legacy bootloader, MediaTek 79xx WMT over USB, Qualcomm ROME USB DFU."""
import asyncio
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bumble import hci                                    # noqa: E402
from bumble.host import Host                              # noqa: E402

from airpodsvr.transport import btfw                      # noqa: E402

FW = os.path.join(os.path.dirname(__file__), "..", "firmware")


def cc(opcode, params, ncmd=1):
    body = bytes([ncmd]) + struct.pack("<H", opcode) + bytes(params)
    return bytes([0x04, 0x0E, len(body)]) + body


def vendor_event(params):
    return bytes([0x04, 0xFF, len(params)]) + bytes(params)


class FakeLink:
    """Transport source + sink pair around a controller model (`handle(opcode, params)`)."""

    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata
        self.sink = None
        self.loop = asyncio.get_event_loop()

    # source side
    def set_packet_sink(self, sink):
        self.sink = sink

    def deliver(self, pkt):
        self.loop.call_soon(self.sink.on_packet, pkt)

    # sink side (host -> controller)
    def on_packet(self, pkt):
        assert pkt[0] == 0x01, pkt
        opcode = struct.unpack_from("<H", pkt, 1)[0]
        for out in self.model.handle(opcode, pkt[4:]) or []:
            self.deliver(out)


def make_host(model, metadata):
    link = FakeLink(model, metadata)
    host = Host(link, link)
    return host, link


# ------------------------------------------------------------------ Intel legacy (AX200 = ThP)
class IntelLegacyModel:
    def __init__(self):
        self.mode = "boot"
        self.sent = {0: b"", 1: b"", 2: b"", 3: b""}
        self.ddc = 0
        self.boot_address = None

    def handle(self, op, p):
        if op == 0x0C03:
            return [cc(op, [0x00 if self.mode == "op" else 0x01])]
        if op == 0xFC05:      # legacy reply: status, platform 0x37, ThP variant 0x12, hw rev 16, fw var, fw rev 1
            fw_variant = 0x23 if self.mode == "op" else 0x06
            return [cc(op, [0, 0x37, 0x12, 16, fw_variant, 1, 0, 0, 0, 0])]
        if op == 0xFC0D:
            params = [0, 0, 0, 0] + list(struct.pack("<H", 0)) + [1, 0, 0, 0, 0, 0] + [1, 2, 3, 4, 5, 6] + [0, 0, 0, 0, 0]
            return [cc(op, params)]
        if op == 0xFC09:
            self.sent[p[0]] += bytes(p[1:])
            out = [cc(op, [0])]
            if p[0] == 0x01 and len(self.sent[1]) >= self.payload_len:
                out.append(vendor_event([0x06, 0x00]))
            return out
        if op == 0xFC01:
            self.boot_address = struct.unpack_from("<I", p, 4)[0]
            self.mode = "op"
            return [vendor_event([0x02, 0x00])]
        if op == 0xFC8B:
            self.ddc += 1
            return [cc(op, [0, 0, 0])]
        raise AssertionError(f"unexpected opcode {op:#x}")


@pytest.mark.skipif(not os.path.exists(os.path.join(FW, "intel", "ibt-18-16-1.sfi")), reason="firmware")
def test_intel_legacy_bootloader_download():
    image = open(os.path.join(FW, "intel", "ibt-18-16-1.sfi"), "rb").read()

    async def run():
        model = IntelLegacyModel()
        model.payload_len = len(image) - 644 - 3      # all whole 4-byte-aligned fragments
        host, _ = make_host(model, {"vendor_id": 0x8087, "product_id": 0x0029})
        drv = await btfw.driver_for_host(host)
        assert isinstance(drv, btfw.IntelLegacyDriver)
        await asyncio.wait_for(drv.init_controller(), 30)
        return model

    model = asyncio.run(run())
    assert model.sent[0] == image[0:128] and model.sent[3] == image[128:384] and model.sent[2] == image[388:644]
    assert image[644:].startswith(model.sent[1]) and len(model.sent[1]) > len(image) - 700
    assert model.mode == "op" and model.boot_address is not None and model.ddc > 0
    assert btfw.last_driver == "intel-legacy"


def test_intel_legacy_already_running_or_not_intel():
    async def run(mode, status_ok=True):
        model = IntelLegacyModel()
        model.mode = mode
        if not status_ok:
            model.handle_orig = model.handle
            model.handle = lambda op, p: [cc(op, [0x01])] if op == 0xFC05 else model.handle_orig(op, p)
        host, _ = make_host(model, {"vendor_id": 0x8087, "product_id": 0x07DA})
        drv = await btfw.driver_for_host(host)
        await asyncio.wait_for(drv.init_controller(), 10)
        return model
    assert asyncio.run(run("op")).sent[1] == b""                  # operational: nothing downloaded
    assert asyncio.run(run("op", status_ok=False)).sent[1] == b""  # CSR chip on an Intel id


# ------------------------------------------------------------------ MediaTek MT7961 (MT7921)
class FakeUsb:
    def __init__(self):
        self.reads = []
        self.writes = []
        self.bulk = b""
        self.wmt_queue = []
        self.regs = {}
        self.qca_version = None
        self.qca_status = 0

    def controlRead(self, rtype, req, value, index, length, timeout=0):
        if rtype == 0xC0 and req == 0x63:                   # MediaTek register read
            return struct.pack("<I", self.regs.get((value << 16) | index, 0))
        if rtype == 0xC0 and req == 0x01 and value == 0x30:  # WMT event poll
            return self.wmt_queue.pop(0) if self.wmt_queue else b""
        if rtype == 0xC0 and req == 0x09:
            return self.qca_version
        if rtype == 0xC0 and req == 0x05:
            return bytes([self.qca_status])
        raise AssertionError((rtype, req, value, index))

    def controlWrite(self, rtype, req, value, index, data, timeout=0):
        self.writes.append((rtype, req, value, index, bytes(data)))
        return len(data)

    def bulkWrite(self, ep, data, timeout=0):
        assert ep == 0x02
        self.bulk += bytes(data)
        return len(data)


class MtkModel:
    def __init__(self, usb):
        self.usb = usb
        self.patch = b""
        self.func_on = False

    def handle(self, op, p):
        if op == 0xFC6F:                                    # WMT: answer on the control pipe
            wop, flag = p[1], p[4]
            data = p[5:]
            if wop == 0x01:
                if flag == 0 and len(data) == 53:           # section header: "not yet downloaded"
                    ans_flag = 0
                else:
                    self.patch += data
                    ans_flag = 0
                self.usb.wmt_queue.append(bytes([0xE4, 5, 0x02, wop, 1, 0, ans_flag]))
            elif wop == 0x06:
                self.func_on = True
                self.usb.wmt_queue.append(bytes([0xE4, 7, 0x02, wop, 3, 0, 0, 0x04, 0x04]))
            return []
        if op == 0x0C03:
            return [cc(op, [0])]
        raise AssertionError(f"unexpected opcode {op:#x}")


@pytest.mark.skipif(not os.path.exists(os.path.join(FW, "mediatek", "BT_RAM_CODE_MT7961_1_2_hdr.bin")),
                    reason="firmware")
def test_mediatek_79xx_download():
    fw = open(os.path.join(FW, "mediatek", "BT_RAM_CODE_MT7961_1_2_hdr.bin"), "rb").read()
    usb = FakeUsb()
    usb.regs = {0x70010200: 0x7961, 0x80021004: 0x01, 0x70010020: 0}
    model = MtkModel(usb)

    async def run():
        host, _ = make_host(model, {"vendor_id": 0x0E8D, "product_id": 0x7961, "usb_handle": usb})
        drv = await btfw.driver_for_host(host)
        assert isinstance(drv, btfw.MediatekDriver)
        await asyncio.wait_for(drv.init_controller(), 60)
    asyncio.run(run())
    nsec = struct.unpack_from("<I", fw, 44)[0]
    expect = b""
    for i in range(nsec):
        m = 96 + 64 * i
        off, size = struct.unpack_from("<I", fw, m + 4)[0], struct.unpack_from("<I", fw, m + 16)[0]
        expect += fw[off:off + size]
    assert model.patch == expect and len(expect) > 100000
    assert model.func_on
    assert (0x5E, 0x02, 0x7401, 0x1890, struct.pack("<I", 0x00010001)) in usb.writes


# ------------------------------------------------------------------ Qualcomm ROME 3.2
class QcaModel:
    def handle(self, op, p):
        if op == 0x0C03:
            return [cc(op, [0])]
        raise AssertionError(f"unexpected opcode {op:#x}")


@pytest.mark.skipif(not os.path.exists(os.path.join(FW, "qca", "rampatch_usb_00000302.bin")), reason="firmware")
def test_qualcomm_rome_dfu():
    rp = open(os.path.join(FW, "qca", "rampatch_usb_00000302.bin"), "rb").read()
    nvm = open(os.path.join(FW, "qca", "nvm_usb_00000302.bin"), "rb").read()
    usb = FakeUsb()
    usb.qca_version = struct.pack("<IIIBBH4x", 0x00000302, 0, 0, 0, 0, 0)

    async def run():
        host, _ = make_host(QcaModel(), {"vendor_id": 0x0CF3, "product_id": 0xE300, "usb_handle": usb})
        drv = await btfw.driver_for_host(host)
        assert isinstance(drv, btfw.QualcommDriver)
        await asyncio.wait_for(drv.init_controller(), 30)
    asyncio.run(run())
    heads = [w for w in usb.writes if w[1] == 0x01]
    assert heads[0][4] == rp[:28] and heads[1][4] == nvm[:4]
    assert usb.bulk == rp[28:] + nvm[4:]


def test_chip_families():
    assert btfw.chip_family(0x8087, 0x0029) == "intel-legacy"      # AX200
    assert btfw.chip_family(0x8087, 0x0032) == "intel"              # AX210 (Bumble)
    assert btfw.chip_family(0x13D3, 0x3563) == "mediatek"
    assert btfw.chip_family(0x0CF3, 0xE300) == "qualcomm"
    assert btfw.chip_family(0x2357, 0x0604) == "realtek"
    assert btfw.chip_family(0x0A12, 0x0001) is None
