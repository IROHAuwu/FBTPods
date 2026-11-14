# SPDX-License-Identifier: Apache-2.0
"""Firmware loaders for USB Bluetooth chips that Bumble does not handle itself.

Bumble already loads firmware for Realtek and for Intel AX210 / AX211 / BE200. This module adds:

  * Intel "legacy bootloader" chips: 8260 / 8265 / 9260 / 9560 / AX200 / AX201 (and the older
    7260 / 7265 / 3160 ROM chips, which need nothing);
  * MediaTek MT7921 / MT7922 / MT7925 / MT7961 (and MT7663 / MT7668);
  * Qualcomm Atheros ROME (QCA61x4 / QCA9377) and WCN6855 / WCN785x over USB.

How each family is brought up (command opcodes, packet layouts, firmware file names, the order of
steps) follows the Linux kernel drivers btintel / btusb (GPL) and btmtk (ISC, Copyright 2021
MediaTek Inc.) — this is an independent Python implementation of those protocols, not a copy.
Firmware files are the unmodified linux-firmware ones shipped in firmware/ (see firmware/README.txt).

MediaTek and Qualcomm need USB control transfers (and Qualcomm a bulk write) besides HCI; the
backend puts the libusb device handle into the HCI metadata as "usb_handle".
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import struct
import time

from bumble import hci
from bumble.drivers import intel as bintel
from bumble.drivers import rtk as brtk

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ firmware files
def firmware_dir():
    from .bumble_usb import firmware_root
    return firmware_root()


def fw_file(*parts):
    root = firmware_dir()
    if not root:
        return None
    p = os.path.join(root, *parts)
    return p if os.path.isfile(p) else None


# ------------------------------------------------------------------ vendor HCI commands
HCI_FBT_INTEL_READ_BOOT_PARAMS_COMMAND = hci.hci_vendor_command_op_code(0x000D)
HCI_FBT_MTK_WMT_COMMAND = hci.hci_vendor_command_op_code(0x006F)
HCI_FBT_MTK_TCI_SLEEP_COMMAND = hci.hci_vendor_command_op_code(0x007A)
hci.HCI_Command.register_commands(globals())


@dataclasses.dataclass
class HCI_Fbt_Raw_ReturnParameters(hci.HCI_StatusReturnParameters):
    data: bytes = hci.field(metadata=hci.metadata('*'))


@hci.HCI_SyncCommand.sync_command(HCI_Fbt_Raw_ReturnParameters)
@dataclasses.dataclass
class HCI_Fbt_Intel_Read_Boot_Params_Command(hci.HCI_SyncCommand[HCI_Fbt_Raw_ReturnParameters]):
    pass


@hci.HCI_SyncCommand.sync_command(HCI_Fbt_Raw_ReturnParameters)
@dataclasses.dataclass
class HCI_Fbt_Mtk_Wmt_Command(hci.HCI_SyncCommand[HCI_Fbt_Raw_ReturnParameters]):
    data: bytes = dataclasses.field(metadata=hci.metadata('*'))


@hci.HCI_SyncCommand.sync_command(HCI_Fbt_Raw_ReturnParameters)
@dataclasses.dataclass
class HCI_Fbt_Mtk_Tci_Sleep_Command(hci.HCI_SyncCommand[HCI_Fbt_Raw_ReturnParameters]):
    data: bytes = dataclasses.field(metadata=hci.metadata('*'))


class _IntelReadVersionLegacy(bintel.HCI_Intel_Read_Version_Command):
    """HCI_Intel_Read_Version without its parameter: the legacy reply format."""

    @property
    def parameters(self):
        return b""

    @parameters.setter
    def parameters(self, value):
        pass


class FirmwareError(RuntimeError):
    pass


async def _reset(host):
    host.ready = True
    await host.send_sync_command(hci.HCI_Reset_Command())


# ================================================================== Intel legacy bootloader
INTEL_LEGACY_ROM = (0x07, 0x08)                      # WP / StP: ROM firmware, works as is
INTEL_LEGACY_BOOT = (0x0B, 0x0C, 0x11, 0x12, 0x13, 0x14)   # SfP WsP JfP ThP HrP CcP
INTEL_RSA_HEADER = 644


@dataclasses.dataclass
class IntelVersion:
    hw_platform: int
    hw_variant: int
    hw_revision: int
    fw_variant: int
    fw_revision: int
    fw_build_num: int
    fw_build_ww: int
    fw_build_yy: int
    fw_patch_num: int

    @classmethod
    def parse(cls, raw):          # raw = status + 9 bytes
        return cls(*raw[1:10])


def intel_legacy_name(ver: IntelVersion, dev_revid: int):
    if ver.hw_variant in (0x0B, 0x0C):
        return f"ibt-{ver.hw_variant}-{dev_revid}"
    if ver.hw_variant in (0x11, 0x12, 0x13, 0x14):
        return f"ibt-{ver.hw_variant}-{ver.hw_revision}-{ver.fw_revision}"
    return None


class IntelLegacyDriver(bintel.Driver):
    """Intel chips before AX210: same secure-send download as Bumble's Intel driver, but legacy
    version / boot-parameter commands, RSA-only images and ibt-<variant>-... file names."""

    name = "intel-legacy"

    @staticmethod
    def check(host):
        md = host.hci_metadata or {}
        if md.get("driver") == "intel-legacy":
            return True
        vid, pid = md.get("vendor_id"), md.get("product_id")
        return vid == 0x8087 and (vid, pid) not in bintel.INTEL_USB_PRODUCTS

    @classmethod
    async def for_host(cls, host, force=False):
        return cls(host) if force or cls.check(host) else None

    async def init_controller(self):
        host = self.host
        host.ready = True
        await host.send_sync_command_raw(hci.HCI_Reset_Command())   # bootloader: "unknown command"
        r = await host.send_sync_command_raw(bintel.HCI_Intel_Read_Version_Command(param0=0xFF))
        rp = r.return_parameters
        raw = bytes([rp.status]) + bytes(getattr(rp, "tlv", b""))
        if rp.status != 0:
            # not an Intel bootloader after all (e.g. 8087:07DA is a CSR chip): nothing to load
            log.info("Intel read version not supported (%#x): no firmware needed", rp.status)
            await _reset(host)
            return
        if not (len(raw) == 10 and raw[1] == 0x37):
            # TLV reply: JfP..CcP running operational firmware answer this way; ask the legacy way
            tlvs = dict(bintel._parse_tlv(getattr(rp, "tlv", b"")))
            hw = tlvs.get(bintel.ValueType.HARDWARE_INFO)
            if hw is not None and int(hw.variant) >= 0x17:
                raise FirmwareError(f"Intel variant {int(hw.variant):#x} is handled by Bumble")
            r = await host.send_sync_command_raw(_IntelReadVersionLegacy(param0=0))
            rp = r.return_parameters
            raw = bytes([rp.status]) + bytes(getattr(rp, "tlv", b""))
            if rp.status != 0 or len(raw) < 10:
                raise FirmwareError("Intel legacy read version failed")
        ver = IntelVersion.parse(raw)
        log.info("Intel legacy: variant %#x revision %d fw_variant %#x", ver.hw_variant, ver.hw_revision,
                 ver.fw_variant)
        if ver.hw_variant in INTEL_LEGACY_ROM:
            await _reset(host)
            return
        if ver.hw_variant not in INTEL_LEGACY_BOOT:
            raise FirmwareError(f"unsupported Intel hardware variant {ver.hw_variant:#x}")
        if ver.fw_variant == 0x23:                    # operational firmware already running
            log.info("Intel: firmware already loaded")
            await _reset(host)
            return
        if ver.fw_variant != 0x06:
            raise FirmwareError(f"Intel: unexpected firmware variant {ver.fw_variant:#x}")

        bp = await host.send_sync_command_raw(HCI_Fbt_Intel_Read_Boot_Params_Command())
        params = bytes([bp.return_parameters.status]) + bytes(getattr(bp.return_parameters, "data", b""))
        if params[0] != 0 or len(params) < 23:
            raise FirmwareError("Intel read boot params failed")
        dev_revid = struct.unpack_from("<H", params, 4)[0]
        if params[21] != 0:                           # limited_cce: fragments not acknowledged
            raise FirmwareError("Intel bootloader without per-fragment acknowledgements")

        base = intel_legacy_name(ver, dev_revid)
        path = fw_file("intel", base + ".sfi")
        if path is None:
            raise FirmwareError(f"firmware intel/{base}.sfi not bundled")
        image = open(path, "rb").read()
        if len(image) < INTEL_RSA_HEADER:
            raise FirmwareError("firmware image too small")
        log.info("Intel: loading %s", os.path.basename(path))
        boot_address = await self._download(image)
        await self._boot(boot_address)
        ddc = fw_file("intel", base + ".ddc")
        if ddc:
            await self.load_device_config(open(ddc, "rb").read())
        await _reset(host)
        log.info("Intel: firmware %s running", base)

    async def _download(self, image):
        host = self.host

        def on_vendor_event(event):
            if event.parameters and event.parameters[0] == 0x06:
                self.firmware_load_complete.set()
            elif event.parameters and event.parameters[0] == 0x02:
                self.reset_complete.set()
        host.on("vendor_event", on_vendor_event)
        self._vendor_handler = on_vendor_event
        saved = host.on_packet
        host.on_packet = self.on_packet               # acks arrive on the ACL pipe in bootloader mode
        self.firmware_load_complete.clear()
        boot_address = 0
        try:
            await self.send_firmware_data(0x00, image[0:128])       # CSS header
            await self.send_firmware_data(0x03, image[128:384])     # public key
            await self.send_firmware_data(0x02, image[388:644])     # signature
            offset, frag = INTEL_RSA_HEADER, 0
            while offset + frag + 3 <= len(image):
                opcode = struct.unpack_from("<H", image, offset + frag)[0]
                plen = image[offset + frag + 2]
                if opcode == bintel.HCI_INTEL_WRITE_BOOT_PARAMS_COMMAND:
                    boot_address = struct.unpack_from("<I", image, offset + frag + 3)[0]
                frag += 3 + plen
                if frag % 4 == 0:
                    await self.send_firmware_data(0x01, image[offset:offset + frag])
                    offset += frag
                    frag = 0
            await asyncio.wait_for(self.firmware_load_complete.wait(), 10)
        finally:
            host.on_packet = saved
        return boot_address

    def _drop_vendor_handler(self):
        h = getattr(self, "_vendor_handler", None)
        if h is not None:
            try:
                self.host.remove_listener("vendor_event", h)
            except Exception:   # noqa: BLE001
                pass

    async def _boot(self, boot_address):
        self.reset_complete.clear()
        self.host.send_hci_packet(bintel.HCI_Intel_Reset_Command(
            reset_type=0x00, patch_enable=0x01, ddc_reload=0x00, boot_option=0x01,
            boot_address=boot_address))
        try:
            await asyncio.wait_for(self.reset_complete.wait(), 5)
        finally:
            self._drop_vendor_handler()
        await asyncio.sleep(0.05)


# ================================================================== MediaTek
MTK_WMT_PATCH_DWNLD, MTK_WMT_FUNC_CTRL, MTK_WMT_RST, MTK_WMT_SEMAPHORE = 0x01, 0x06, 0x07, 0x17
(MTK_INVALID, MTK_PATCH_UNDONE, MTK_PATCH_PROGRESS, MTK_PATCH_DONE,
 MTK_ON_UNDONE, MTK_ON_DONE, MTK_ON_PROGRESS) = range(7)
MTK_PKT_START, MTK_PKT_CONTINUE, MTK_PKT_END = 1, 2, 3
MTK_EP_RST_OPT, MTK_EP_RST_IN_OUT_OPT = 0x74011890, 0x00010001
MTK_FW = {   # dev_id -> firmware path (flavor / version picked at runtime)
    0x7922: ("mediatek", "BT_RAM_CODE_MT7922_1_{v}_hdr.bin"),
    0x7961: ("mediatek", "BT_RAM_CODE_MT7961_1{f}_{v}_hdr.bin"),
    0x7925: ("mediatek/mt7925", "BT_RAM_CODE_MT7925_1_{v}_hdr.bin"),
    0x7902: ("mediatek", "BT_RAM_CODE_MT7902_1_{v}_hdr.bin"),
}
MTK_LEGACY_FW = {0x7663: "mt7663pr2h.bin", 0x7668: "mt7668pr2h.bin"}


class MediatekDriver:
    name = "mediatek"

    def __init__(self, host):
        self.host = host
        self.usb = host.hci_metadata.get("usb_handle")

    @staticmethod
    def check(host):
        md = host.hci_metadata or {}
        if md.get("driver") == "mtk":
            return True
        vid, pid = md.get("vendor_id"), md.get("product_id")
        return vid == 0x0E8D or (vid, pid) in MTK_USB_PRODUCTS

    @classmethod
    async def for_host(cls, host, force=False):
        return cls(host) if force or cls.check(host) else None

    # -- USB register access (vendor control requests)
    async def _ctrl_read(self, rtype, req, value, index, length):
        if self.usb is None:
            raise FirmwareError("MediaTek needs USB control transfers (no USB handle)")
        return bytes(await asyncio.to_thread(self.usb.controlRead, rtype, req, value, index, length, 1000))

    async def _ctrl_write(self, rtype, req, value, index, data):
        if self.usb is None:
            raise FirmwareError("MediaTek needs USB control transfers (no USB handle)")
        return await asyncio.to_thread(self.usb.controlWrite, rtype, req, value, index, data, 1000)

    async def reg_read(self, reg):
        d = await self._ctrl_read(0xC0, 0x63, reg >> 16, reg & 0xFFFF, 4)
        return struct.unpack("<I", d[:4])[0] if len(d) >= 4 else 0

    async def uhw_reg_write(self, reg, val):
        await self._ctrl_write(0x5E, 0x02, reg >> 16, reg & 0xFFFF, struct.pack("<I", val))

    # -- WMT: an HCI vendor command whose answer (event 0xE4) is polled on the control pipe
    def _wmt_poll(self, timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            try:
                d = bytes(self.usb.controlRead(0xC0, 0x01, 0x30, 0x00, 64, 200))
            except Exception as e:   # noqa: BLE001 - usb1 timeouts while nothing is pending
                if "TIMEOUT" not in type(e).__name__.upper() and "timeout" not in str(e).lower():
                    raise
                d = b""
            if d:
                return d
            time.sleep(0.001)
        return None

    async def wmt(self, op, flag, data=b"", timeout=10.0):
        if self.usb is None:
            raise FirmwareError("MediaTek needs USB control transfers (no USB handle)")
        hdr = bytes([0x01, op]) + struct.pack("<H", len(data) + 1) + bytes([flag])
        self.host.send_hci_packet(HCI_Fbt_Mtk_Wmt_Command(data=hdr + bytes(data)))
        evt = await asyncio.to_thread(self._wmt_poll, timeout)
        if evt is None:
            raise FirmwareError(f"MediaTek WMT op {op:#x}: no answer")
        if len(evt) < 7 or evt[0] != 0xE4:
            raise FirmwareError(f"MediaTek WMT op {op:#x}: bad event {evt[:8].hex()}")
        e_op, e_flag = evt[3], evt[6]
        if e_op != op:
            raise FirmwareError(f"MediaTek WMT: answer for op {e_op:#x}, expected {op:#x}")
        if op == MTK_WMT_SEMAPHORE:
            return MTK_PATCH_UNDONE if e_flag == 2 else MTK_PATCH_DONE
        if op == MTK_WMT_FUNC_CTRL:
            if len(evt) < 9:
                return MTK_ON_UNDONE if e_flag else MTK_ON_DONE
            st = struct.unpack(">H", evt[7:9])[0]
            return MTK_ON_DONE if st == 0x404 else MTK_ON_PROGRESS if st == 0x420 else MTK_ON_UNDONE
        if op == MTK_WMT_PATCH_DWNLD:
            return {2: MTK_PATCH_DONE, 1: MTK_PATCH_PROGRESS}.get(e_flag, MTK_PATCH_UNDONE)
        return MTK_INVALID

    async def init_controller(self):
        dev_id = await self.reg_read(0x80000008)
        fw_version = fw_flavor = 0
        if dev_id != 0x7663:
            dev_id = await self.reg_read(0x70010200)
            fw_version = await self.reg_read(0x80021004)
            fw_flavor = (await self.reg_read(0x70010020) & 0x80) >> 7
        dev_id &= 0xFFFF
        log.info("MediaTek: chip MT%04x, fw version %d, flavor %d", dev_id, fw_version & 0xFF, fw_flavor)
        if dev_id in MTK_FW:
            sub, pattern = MTK_FW[dev_id]
            name = pattern.format(v=(fw_version & 0xFF) + 1, f="a" if (dev_id == 0x7961 and fw_flavor) else "")
            path = fw_file(*sub.split("/"), name)
            if path is None:
                raise FirmwareError(f"firmware {sub}/{name} not bundled")
            await self._setup_79xx(open(path, "rb").read())
            await self.uhw_reg_write(MTK_EP_RST_OPT, MTK_EP_RST_IN_OUT_OPT)
            await self.wmt(MTK_WMT_FUNC_CTRL, 0, b"\x01")
        elif dev_id in MTK_LEGACY_FW:
            path = fw_file("mediatek", MTK_LEGACY_FW[dev_id])
            if path is None:
                raise FirmwareError(f"firmware mediatek/{MTK_LEGACY_FW[dev_id]} not bundled")
            if await self.wmt(MTK_WMT_SEMAPHORE, 1) != MTK_PATCH_DONE:
                await self._setup_legacy(open(path, "rb").read())
            t0 = time.monotonic()
            while True:
                st = await self.wmt(MTK_WMT_FUNC_CTRL, 4, b"\x00")
                if st != MTK_ON_PROGRESS or time.monotonic() - t0 > 5:
                    break
                await asyncio.sleep(0.002)
            if st != MTK_ON_DONE:
                await self.wmt(MTK_WMT_FUNC_CTRL, 0, b"\x01")
            self.host.ready = True
            await self.host.send_sync_command(HCI_Fbt_Mtk_Tci_Sleep_Command(
                data=bytes([0x05]) + struct.pack("<HH", 0x640, 0x640) + bytes([0, 0])))
        else:
            raise FirmwareError(f"unsupported MediaTek chip MT{dev_id:04x}")
        await _reset(self.host)
        log.info("MediaTek: MT%04x ready", dev_id)

    async def _setup_79xx(self, fw):
        if len(fw) < 96:
            raise FirmwareError("MediaTek firmware too small")
        sections = struct.unpack_from("<I", fw, 32 + 12)[0]
        if not sections or 96 + 64 * sections > len(fw):
            raise FirmwareError("MediaTek firmware layout invalid")
        for i in range(sections):
            m = 32 + 64 + 64 * i
            sec_offset = struct.unpack_from("<I", fw, m + 4)[0]
            dl_size = struct.unpack_from("<I", fw, m + 12 + 4)[0]       # bin_info_spec.dlsize
            if dl_size == 0:
                continue
            if sec_offset > len(fw) or dl_size > len(fw) - sec_offset:
                raise FirmwareError(f"MediaTek firmware section {i} out of bounds")
            for _ in range(20):
                st = await self.wmt(MTK_WMT_PATCH_DWNLD, 0, b"\x00" + fw[m + 12:m + 12 + 52])
                if st != MTK_PATCH_PROGRESS:
                    break
                await asyncio.sleep(0.1)
            else:
                raise FirmwareError("MediaTek: firmware download stays busy")
            if st == MTK_PATCH_DONE:
                continue
            if st != MTK_PATCH_UNDONE:
                raise FirmwareError(f"MediaTek: unexpected download status {st}")
            pos, left, first = sec_offset, dl_size, True
            while left > 0:
                n = min(250, left)
                flag = MTK_PKT_START if first else (MTK_PKT_END if left - n <= 0 else MTK_PKT_CONTINUE)
                first = False
                st = await self.wmt(MTK_WMT_PATCH_DWNLD, flag, fw[pos:pos + n])
                if st == MTK_PATCH_PROGRESS:
                    raise FirmwareError("MediaTek: firmware download interrupted")
                pos += n
                left -= n
        await asyncio.sleep(0.11)

    async def _setup_legacy(self, fw):
        await self.wmt(MTK_WMT_FUNC_CTRL, 3, b"\x01")       # power on the data RAM
        if len(fw) < 30:
            raise FirmwareError("MediaTek firmware too small")
        body, total = fw[30:], len(fw) - 30
        pos, flag = 0, MTK_PKT_START
        while pos < total:
            n = min(250, total - pos)
            if total - pos - n <= 0:
                flag = MTK_PKT_END
            elif pos > 0:
                flag = MTK_PKT_CONTINUE
            await self.wmt(MTK_WMT_PATCH_DWNLD, flag, body[pos:pos + n])
            pos += n
        await self.wmt(MTK_WMT_RST, 4)
        await asyncio.sleep(0.012)


# ================================================================== Qualcomm (USB DFU)
QCA_GET_TARGET_VERSION, QCA_CHECK_STATUS, QCA_DFU_DOWNLOAD = 0x09, 0x05, 0x01
QCA_SYSCFG_UPDATED, QCA_PATCH_UPDATED, QCA_FLAG_MULTI_NVM = 0x40, 0x80, 0x80
QCA_DEVICES = {   # rom_version -> (rampatch header length, nvm header length, version offset)
    0x00000100: (20, 4, 8), 0x00000101: (20, 4, 8),
    0x00000200: (28, 4, 16), 0x00000201: (28, 4, 16), 0x00000300: (28, 4, 16), 0x00000302: (28, 4, 16),
    0x00130100: (40, 4, 16), 0x00130200: (40, 4, 16), 0x00130201: (40, 4, 16), 0x00190200: (40, 4, 16),
}
QCA_CUSTOM_DIR = {(0x00130201, 0x030A): "QCA2066", (0x00130201, 0x030B): "QCA2066"}
QCA_RAM_GF = (0x400C1200, 0x400C1211)


@dataclasses.dataclass
class QcaVersion:
    rom: int
    patch: int
    ram: int
    chip_id: int
    platform_id: int
    flag: int

    @classmethod
    def parse(cls, d):
        rom, patch, ram, chip, plat, flag = struct.unpack_from("<IIIBBH", d, 0)
        return cls(rom, patch, ram, chip, plat, flag)

    @property
    def board_id(self):
        b = (self.chip_id << 8) + self.platform_id if ((self.flag >> 8) & 0xFF) == QCA_FLAG_MULTI_NVM else 0
        return 0 if b == 0xFFFF else b


def qca_names(ver: QcaVersion):
    sub = QCA_CUSTOM_DIR.get((ver.rom, ver.board_id)) if ver.board_id else None
    pre = [sub] if sub else []
    ram = pre + [f"rampatch_usb_{ver.rom:08x}.bin"]
    nvm = f"nvm_usb_{ver.rom:08x}" + ("_gf" if ver.ram in QCA_RAM_GF else "")
    nvm += (f"_{ver.board_id:04x}" if ver.board_id else "") + ".bin"
    return ram, pre + [nvm]


class QualcommDriver:
    name = "qualcomm"

    def __init__(self, host):
        self.host = host
        self.usb = host.hci_metadata.get("usb_handle")

    @staticmethod
    def check(host):
        md = host.hci_metadata or {}
        if md.get("driver") == "qca":
            return True
        return (md.get("vendor_id"), md.get("product_id")) in QCA_USB_PRODUCTS

    @classmethod
    async def for_host(cls, host, force=False):
        return cls(host) if force or cls.check(host) else None

    async def _req(self, request, length):
        if self.usb is None:
            raise FirmwareError("Qualcomm needs USB control transfers (no USB handle)")
        return bytes(await asyncio.to_thread(self.usb.controlRead, 0xC0, request, 0, 0, length, 1000))

    async def _dfu(self, data, hdr_len):
        head = data[:hdr_len]
        await asyncio.to_thread(self.usb.controlWrite, 0x40, QCA_DFU_DOWNLOAD, 0, 0, head, 1000)
        await asyncio.sleep(0.02)                     # ep2 switches from ACL to DFU
        pos = len(head)
        while pos < len(data):
            chunk = data[pos:pos + 4096]
            n = await asyncio.to_thread(self.usb.bulkWrite, 0x02, chunk, 3000)
            if n != len(chunk):
                raise FirmwareError("Qualcomm: short bulk write")
            pos += len(chunk)

    async def init_controller(self):
        ver = QcaVersion.parse(await self._req(QCA_GET_TARGET_VERSION, 18))
        info = QCA_DEVICES.get(ver.rom)
        log.info("Qualcomm: rom %#010x patch %#x", ver.rom, ver.patch)
        if info is None:
            if ver.rom & ~0xFFFF:
                await _reset(self.host)               # newer ROM: no patch needed
                return
            raise FirmwareError(f"unsupported Qualcomm ROM {ver.rom:#x}")
        rp_hdr, nvm_hdr, ver_off = info
        status = (await self._req(QCA_CHECK_STATUS, 1))[0]
        ram, nvm = qca_names(ver)
        if not status & QCA_PATCH_UPDATED:
            path = fw_file("qca", *ram)
            if path is None:
                raise FirmwareError(f"firmware qca/{'/'.join(ram)} not bundled")
            fw = open(path, "rb").read()
            hi, lo, rpatch = struct.unpack_from("<HHH", fw, ver_off)
            rrom = (hi << 16 | lo) if ver.rom & ~0xFFFF else lo
            if rrom != ver.rom or rpatch < ver.patch:
                raise FirmwareError("Qualcomm rampatch does not match the chip")
            await self._dfu(fw, rp_hdr)
        ver = QcaVersion.parse(await self._req(QCA_GET_TARGET_VERSION, 18))
        ram, nvm = qca_names(ver)
        if not status & QCA_SYSCFG_UPDATED:
            path = fw_file("qca", *nvm)
            if path is None:
                raise FirmwareError(f"firmware qca/{'/'.join(nvm)} not bundled")
            await self._dfu(open(path, "rb").read(), nvm_hdr)
            if info and ver.rom >= 0x00130201:
                await asyncio.sleep(0.1)
        await _reset(self.host)
        log.info("Qualcomm: firmware loaded")


# ================================================================== selection
last_driver = None
DRIVERS = [("rtk", brtk.Driver), ("intel", bintel.Driver), ("intel-legacy", IntelLegacyDriver),
           ("mtk", MediatekDriver), ("qca", QualcommDriver)]
_FORCED = {"rtk": brtk.Driver, "intel": bintel.Driver, "intel-legacy": IntelLegacyDriver,
           "mtk": MediatekDriver, "qca": QualcommDriver}


async def driver_for_host(host):
    """Bumble driver factory: Bumble's own Realtek / Intel drivers first, then ours."""
    forced = (host.hci_metadata or {}).get("driver", "")
    forced = forced.split("/")[0] if forced else ""
    candidates = [(forced, _FORCED[forced])] if forced in _FORCED else DRIVERS
    for name, cls in candidates:
        try:
            drv = await cls.for_host(host)
        except Exception as e:   # noqa: BLE001
            log.warning("driver %s: %s", name, e)
            continue
        if drv is not None:
            global last_driver
            last_driver = {"rtk": "realtek"}.get(name, name)
            log.info("Bluetooth firmware driver: %s", name)
            return drv
    return None


def chip_family(vid, pid):
    """'realtek' | 'intel' | 'intel-legacy' | 'mediatek' | 'qualcomm' | None (by USB id only)."""
    if (vid, pid) in brtk.RTK_USB_PRODUCTS or vid == 0x0BDA:
        return "realtek"
    if (vid, pid) in bintel.INTEL_USB_PRODUCTS:
        return "intel"
    if vid == 0x8087:
        return "intel-legacy"
    if vid == 0x0E8D or (vid, pid) in MTK_USB_PRODUCTS:
        return "mediatek"
    if (vid, pid) in QCA_USB_PRODUCTS:
        return "qualcomm"
    return None


from .btfw_ids import ATH3012_USB_PRODUCTS, MTK_USB_PRODUCTS, QCA_USB_PRODUCTS  # noqa: E402,F401
