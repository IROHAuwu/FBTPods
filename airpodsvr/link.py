# SPDX-License-Identifier: Apache-2.0
"""One AACP link to one pair of AirPods: handshake, sensor start, re-arming, reconnection.

Emits ImuSample objects through `on_sample(source_id, sample)` where source_id = "<addr>#<side>".
Runs in its own thread.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

from .transport import errtext, forgot_us
from .sides import BudSides
from .protocol import (HANDSHAKE_SEQUENCE, HDR, RHID_CONNECT, SVC_ACCEL, SVC_DEVMOTION6, SVC_GYRO, ProtocolState,
                       bud_shape, parse_device_info,
                       ear_detection, host_connect, rhid_frame, rhid_sub, set_report_interval, stem_enable, wx_raw_start,
                       wx_start_sequence, wx_stream)


_seq_lock = threading.Lock()


def next_wx_seq(n=1):
    """SensorDataWX request numbers, kept increasing across sessions like AirPointer does
    (8001..15999, stored in the data folder). Returns the first of n consecutive numbers."""
    from . import config as C
    path = C.data_path("wx_seq.txt")
    with _seq_lock:
        try:
            with open(path, encoding="utf-8") as f:
                cur = int(f.read().strip() or 0)
        except (OSError, ValueError):
            cur = 0
        first = cur + 1 if 8000 <= cur < 15990 else 8001
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(first + n - 1))
        except OSError:
            pass
        return first

log = logging.getLogger(__name__)


class AirPodsLink(threading.Thread):
    def __init__(self, backend, addr, on_sample, interval_us=10000, sides=(0, 1),
                 on_event=None, record_path=None, stem=False, bud_info=None, probe=False,
                 ear_off=False):
        """bud_info: what was kept about this pair at pairing (config 'buds'): with it the side
        labels are corrected back to the ones of pairing time (see sides.py), and `sides` names
        the corrected labels wanted. Without it labels are passed on as received."""
        super().__init__(name=f"link-{addr}", daemon=True)
        self.backend = backend
        self.addr = addr
        self.on_sample = on_sample
        self.on_event = on_event or (lambda *a: None)
        self.stem = bool(stem)
        self.ear_off = bool(ear_off)     # switch the buds' automatic ear detection off while we use them
        self.ear_changed = False
        self.interval_us = int(interval_us)
        self.buds = BudSides(bud_info)
        self.swap, self.lr_certain = self.buds.resolve()
        self.want = tuple(sides)        # labels passed to on_sample
        # with corrected labels either bud may be the wanted one: talk to both
        self.sides = (0, 1) if bud_info else tuple(sides)
        self.stop_flag = threading.Event()
        self.chan = None
        self.state = None
        self.status = "idle"
        self.forgot = False                  # the AirPods no longer know us: pairing again is needed
        self.stalls = 0                      # how often the stream stopped (diagnostics)
        self.ear = None
        self.battery = {}
        self.device_name = self.model = None    # from the device information packet, e.g. "A2084"
        self.model_kind = None          # "raw" (AirPods 3) | "wx-raw" (Pro 2 / Pro 3 / 4 raw IMU)
                                        # | "fused" (SensorDataWX attitude, ~25 Hz)
        self.last_sample = {}           # side -> monotonic time
        self.no_motion = False          # connected, but the buds sent no motion data (see _pump)
        self.record = open(record_path, "a", encoding="utf-8") if record_path else None
        self.t_start = time.monotonic()
        # probe: find out whether the second bud of a pair can be made to send motion data at all
        # (used while identifying buds; everything it learns goes to the log and the recording)
        self.probe = bool(probe)
        self.recorded = 0               # sensor frames written to the recording (capped)
        self.seen_ops = {}              # AACP opcode -> number of packets
        self.probe_step = 0
        self.probe_t = 0.0
        self.other_addrs = set()        # addresses of the same maker the buds mention (0x0c / 0x2b)
        self.record_all = False         # relay probe: keep every frame from then on
        self.hello_tries = {}           # side -> HostConnects sent to a side that never announced

    # ------------------------------------------------------------------ helpers
    RECORD_SENSOR_FRAMES = 1500          # of the ~100 per second; every other frame is always kept

    def _log_frame(self, direction, pkt):
        if self.record:
            if direction == "rx" and pkt[4:6] == b"\x17\x00":
                self.recorded += 1
                if self.recorded > self.RECORD_SENSOR_FRAMES and not self.record_all:
                    return
            self.record.write(json.dumps({"t": round(time.monotonic() - self.t_start, 5),
                                          "dir": direction, "hex": pkt.hex()}) + "\n")

    def _remember_lost(self):
        """Kept in the config, so the window says "pair again" also after a restart (it used to go
        on showing these AirPods as paired)."""
        try:
            from . import config as C
            lost = list(C.load_config().get("lost_pairing") or [])
            if self.addr not in lost:
                C.save_config({"lost_pairing": lost + [self.addr]})
        except Exception as e:   # noqa: BLE001
            log.debug("lost pairing not saved: %s", e)

    def _send(self, pkt):
        self._log_frame("tx", pkt)
        self.chan.send(pkt)

    def _arm(self, side):
        for p in self.state.start_raw_imu(side, self.interval_us):
            self._send(p)

    # ------------------------------------------------------------------ main loop
    def run(self):
        backoff = 1.0
        while not self.stop_flag.is_set():
            try:
                self.status = "connecting"
                self.chan = self.backend.open(self.addr)
                self.state = ProtocolState()
                for pkt, delay in HANDSHAKE_SEQUENCE:
                    self._send(pkt)
                    time.sleep(delay)
                for side in self.sides:
                    self._send(host_connect(self.state.next_gen(), side))
                # double / triple / long press come to us (single press stays play/pause). Always
                # sent: with the option off, 0 also repairs buds left changed by a crash.
                self._send(stem_enable(0x0E if self.stem else 0x00))
                self.stem_changed = self.stem
                if self.ear_off:
                    self._send(ear_detection(False))
                    if not self.ear_changed:
                        log.info("%s: automatic ear detection switched off (given back when tracking stops)", self.addr)
                    self.ear_changed = True
                self.status = "waiting for sensors"
                self.forgot = False
                backoff = 1.0
                self._pump()
            except ConnectionError as e:
                log.warning("%s: link lost (%s)", self.addr, e)
            except Exception as e:   # noqa: BLE001 - keep retrying
                if forgot_us(e) and not self.forgot:
                    log.warning("%s: the AirPods no longer accept our pairing (%s): pair them again",
                                self.addr, errtext(e))
                elif not forgot_us(e):
                    log.warning("%s: %s", self.addr, errtext(e))
                if forgot_us(e):
                    if not self.forgot:
                        self._remember_lost()
                    self.forgot = True
                    backoff = 15.0           # retrying cannot help; do not hammer the buds
            finally:
                if self.chan:
                    self.chan.close()
            if self.stop_flag.is_set():
                break
            self.status = "needs pairing" if self.forgot else f"reconnecting in {backoff:.0f}s"
            self.stop_flag.wait(backoff)
            backoff = min(backoff * 2, 15.0)
        self.status = "stopped"

    def _start_wx(self, fused=False):
        """SensorDataWX buds (AirPods Pro 2 / Pro 3 / 4): raw accelerometer + gyroscope first (services
        11 / 12, as AirPointer does), the fused attitude stream (service 16) if no raw data comes.
        Harmless for AirPods 3, which ignore it (MousePods DISCOVERIES §2)."""
        if fused:
            seq = next_wx_seq(3)
            seqs = wx_start_sequence(seq, self.interval_us)
        else:
            seq = next_wx_seq(2)
            seqs = wx_raw_start(seq, self.interval_us)
        self.wx_started = getattr(self, "wx_started", set()) | ({"fused"} if fused else {"raw"})
        for pkt, delay in seqs:
            self._send(pkt)
            if delay:
                time.sleep(delay)

    def _pump(self):
        announced_at = {}
        t_open = last_rehello = time.monotonic()
        wx_tried = 0.0
        wx_attempts = 0
        self.last_sample = {}
        self.wx_started = set()
        self.hello_tries = {}                  # per connection
        stalled = None                         # when the stream stopped, while it is stopped
        while not self.stop_flag.is_set():
            pkt = self.chan.recv(0.5)
            now = time.monotonic()
            if self.probe:
                self._probe_second_bud(now, t_open)
            if pkt is not None:
                self._log_frame("rx", pkt)
                if self.probe:
                    self._probe_note(pkt)
                info = parse_device_info(pkt)
                if info and info != (self.device_name, self.model):
                    self.device_name, self.model = info
                    log.info("%s: model %s", self.addr, self.model or "?")
                for ev in self.state.feed(pkt, now):
                    kind = ev[0]
                    if self.probe and kind == "reply":
                        log.info("%s: probe: reply from side %d, kind %d, device %d: %s", self.addr, ev[1],
                                 ev[2], ev[3], bytes(ev[4][:16]).hex())
                    elif self.probe and kind == "announce":
                        log.info("%s: probe: side %d announces %s as device %d", self.addr, ev[1], ev[2], ev[3])
                    if kind == "imu":
                        s = ev[1]
                        self.last_sample[s.side] = now
                        self.no_motion = False
                        self.status = "streaming"
                        label = s.side ^ 1 if self.swap else s.side
                        if label in self.want:
                            self.on_sample(f"{self.addr}#{label}", s)
                    elif kind == "primary":
                        if self.buds.observe_primary(ev[1], ev[2]):
                            log.info("%s: primary bud is the %s one%s", self.addr, ev[1],
                                     "" if ev[2] else " (from the battery report)")
                            self._sides_changed()
                    elif kind == "location":
                        if self.buds.observe_location(ev[1], ev[2]):
                            log.info("%s: side %d announces LocationID %d", self.addr, ev[1], ev[2])
                            self._sides_changed()
                    elif kind == "announce":
                        _, side, name, dev = ev
                        if name in ("accel", "gyro"):
                            announced_at[side] = now
                    elif kind == "ear":
                        if ev[1] != self.ear:
                            place = {0: "in ear", 1: "out of ear", 2: "in case", 3: "in closed case"}
                            log.info("%s: primary bud %s, secondary bud %s", self.addr,
                                     place.get(ev[1][0], ev[1][0]), place.get(ev[1][1], ev[1][1]))
                        self.ear = ev[1]
                        self.on_event("ear", self.addr, ev[1])
                    elif kind == "stem":
                        self.on_event("stem", self.addr, ev[1])
                    elif kind == "battery":
                        self.battery = ev[1]
                        self.on_event("battery", self.addr, ev[1])
            # start (or re-start, after a re-announce) once both raw sensors of a side are known
            for side, t in list(announced_at.items()):
                if now - t > 0.2:
                    self._arm(side)
                    del announced_at[side]
            streaming = any(now - t < 2.0 for t in self.last_sample.values())
            # A stream that ran and then stops (seen with three and four pairs on one adapter: a bud
            # freezes, its tracker stays at the same angle for good). It used to be re-armed every
            # 5 s for ever, without a word, while the status went on saying "streaming". Now it is
            # said, and after 12 s without data the link is dropped and made again.
            if self.last_sample:
                gap = now - max(self.last_sample.values())
                if gap > 2.0 and stalled is None:
                    stalled = now
                    self.status = "stalled"
                    self.stalls += 1
                    log.warning("%s: the motion stream stopped (stall no. %d of this session)", self.addr, self.stalls)
                elif gap <= 2.0 and stalled is not None:
                    log.info("%s: the motion stream is back after %.0f s", self.addr, now - stalled)
                    stalled = None
                if stalled is not None and gap > 12.0:
                    raise ConnectionError("the motion stream stopped for 12 s")
            # no RemoteHID sensors announced after 3 s: SensorDataWX buds (Pro 2 / Pro 3 / 4)
            raw_known = any(self.state.table.get(sd, "accel") is not None for sd in self.sides)
            if not streaming and not raw_known and now - t_open > 3.0 and now - wx_tried > 4.0:
                wx_tried = now
                # raw IMU first; if it does not come, the fused stream (and keep alternating)
                self._start_wx(fused="raw" in getattr(self, "wx_started", set()) and not self.state.wx_raw_seen
                               and wx_attempts % 2 == 1)
                wx_attempts += 1
                self.model_kind = self.model_kind or "fused"
            if raw_known:
                self.model_kind = "raw"
            elif self.state.wx_raw_seen:
                self.model_kind = "wx-raw"
            elif self.state.wx_seen:
                self.model_kind = "fused"
            # AirPointer: "the motion channel only answers just after a connection". Nothing at all
            # 25 s after opening: drop the link and connect again (backoff in run()).
            if not self.last_sample and now - t_open > 25.0:
                self.no_motion = True     # typical when an iPhone / iPad / Mac holds the AirPods
                raise ConnectionError("no motion data after connecting")
            # a bud silent for a while (back from the case, primary changed, out of range): say
            # hello / re-arm that side again, independently of the other bud
            if now - last_rehello > 5.0:
                silent = [sd for sd in self.sides if now - self.last_sample.get(sd, 0.0) > 2.0]
                if silent and not (self.model_kind in ("fused", "wx-raw") and streaming):
                    last_rehello = now
                    for side in silent:
                        if self.state.table.get(side, "accel") is not None:
                            self._arm(side)
                        elif not streaming or self.hello_tries.get(side, 0) < 3:
                            # Real AirPods 3 and Pro: the second bud never announces, and every
                            # HostConnect makes the first one announce everything again and restart
                            # its streams. So while another bud streams, only a few times; with
                            # nothing streaming there is nothing to disturb, and the greeting is
                            # what wakes a bud that just left the case.
                            self.hello_tries[side] = self.hello_tries.get(side, 0) + 1
                            self._send(host_connect(self.state.next_gen(), side))

    # ------------------------------------------------------------------ second-bud probe
    def _probe_note(self, pkt):
        """Log each kind of AACP packet the first time it comes (their content is in the recording)."""
        if pkt[:4] != HDR or len(pkt) < 6:
            return
        op = pkt[4] | (pkt[5] << 8)
        n = self.seen_ops[op] = self.seen_ops.get(op, 0) + 1
        if op in (0x0C, 0x2B):
            self._note_addresses(pkt[6:])
        if n == 1 and op != 0x17:
            log.info("%s: probe: first packet with opcode 0x%02x: %s", self.addr, op, pkt[:28].hex())

    def _note_addresses(self, body):
        """Packets 0x0c and 0x2b carry Bluetooth addresses (byte-reversed). One that has our pair's
        maker prefix but is not the address we connected to is very likely the other bud's own."""
        try:
            mine = bytes(int(x, 16) for x in self.addr.split("/")[0].split(":"))[::-1]
        except ValueError:
            return
        for i in range(len(body) - 5):
            w = bytes(body[i:i + 6])
            if w[3:] == mine[3:] and w != mine:
                a = ":".join(f"{b:02X}" for b in w[::-1])
                if a not in self.other_addrs:
                    self.other_addrs.add(a)
                    log.info("%s: probe: the buds mention another address of theirs: %s", self.addr, a)

    def _probe_second_bud(self, now, t_open):
        """Only side 0 has ever been seen announcing sensors. Once it streams, try the ways the
        other bud could be addressed, one every 6 s, and say what each was; whether data with
        side 1 follows shows in the per-side counts logged at the end."""
        table = self.state.table
        if table.get(1, "accel") is not None or not self.last_sample or now - self.probe_t < 6.0:
            return
        if now - t_open < 8.0 or self.probe_step >= 4:
            return
        self.probe_t = now
        self.probe_step += 1
        acc, gyr = table.get(0, "accel"), table.get(0, "gyro")
        step = self.probe_step
        try:
            if step == 1:
                what = "HostConnect with the side flag set"
                self._send(host_connect(self.state.next_gen(), 1))
            elif step == 2 and acc is not None:
                what = f"SetReport for side 0's devices ({acc}, {gyr}) with the side flag set"
                for dev in (acc, gyr):
                    if dev is not None:
                        self._send(set_report_interval(self.state.next_gen(), dev, self.interval_us, 1))
            elif step == 3 and acc is not None:
                what = "SetReport with bit 6 set in the device id, side flag clear"
                for dev in (acc, gyr):
                    if dev is not None:
                        self._send(set_report_interval(self.state.next_gen(), dev | 0x40, self.interval_us, 0))
            elif step == 4:
                what = "HostConnect with bit 6 set in the device id"
                self._send(rhid_frame(self.state.next_gen(), [rhid_sub(0x40, RHID_CONNECT)], 0))
            else:
                return
            log.info("%s: probe %d: %s", self.addr, step, what)
        except Exception as e:   # noqa: BLE001
            log.info("%s: probe %d failed: %s", self.addr, step, errtext(e))

    # ------------------------------------------------------------------ relay probe
    RELAY_SERVICES = ("iedrelay", "neorelay", "mando", "kadabra", "cma", "devmotion6")
    WX_RELAY_SERVICES = (6, 7, 8, 9, 15)      # MANDO, IED, NEO, NEORELAY, CMA (MousePods journal)

    def mark(self, label):
        """A labelled moment in the recording (what the user was asked to do from here on)."""
        if self.record:
            self.record.write(json.dumps({"t": round(time.monotonic() - self.t_start, 5), "mark": label}) + "\n")
            self.record.flush()
        log.info("%s: probe mark: %s", self.addr, label)

    def start_relay_probe(self):
        """Ask for every sensor service the buds offer besides the accelerometer and gyroscope.
        Two of them are called "relay" and nobody knows what they carry: if it is the other bud's
        data, the recording (taken while the user moves one bud, then the other) will show it."""
        self.record_all = True
        asked = []
        if self.state is None or self.chan is None:
            return asked
        try:
            for name in self.RELAY_SERVICES:
                dev = self.state.table.get(0, name)
                if dev is not None:
                    self._send(set_report_interval(self.state.next_gen(), dev, 10000, 0))
                    asked.append(f"{name}({dev})")
            if not asked:                          # SensorDataWX buds (Pro 2 / Pro 3 / 4)
                seq = next_wx_seq(len(self.WX_RELAY_SERVICES))
                for i, svc in enumerate(self.WX_RELAY_SERVICES):
                    self._send(wx_stream(seq + i, svc, 20000))
                    asked.append(f"wx{svc}")
        except Exception as e:   # noqa: BLE001
            log.info("%s: relay probe: %s", self.addr, errtext(e))
        log.info("%s: relay probe: asked for %s", self.addr, ", ".join(asked) or "nothing")
        return asked

    def probe_summary(self):
        counts = dict(getattr(self.state, "counts", {})) if self.state else {}
        per_side = {}
        for (side, dev), n in counts.items():
            per_side.setdefault(side, {})[dev] = n
        return {"reports_per_side": per_side, "opcodes": {f"0x{k:02x}": v for k, v in sorted(self.seen_ops.items())},
                "kind": self.model_kind, "ear": list(self.ear) if self.ear else None,
                "probe_steps": self.probe_step}

    @property
    def shape(self):
        """"pro" or "a3": which bud to draw."""
        return bud_shape(self.model, self.device_name, self.model_kind)

    def lr_info(self):
        """For the status display and diagnostics."""
        return {"primary": self.buds.primary, "loc": {str(k): v for k, v in sorted(self.buds.loc.items())},
                "swap": self.swap, "certain": self.lr_certain, "known": self.buds.has_base,
                "ruled_out": sorted(self.buds.ruled_out)}

    def _sides_changed(self):
        swap, certain = self.buds.resolve()
        if (swap, certain) != (self.swap, self.lr_certain):
            self.swap, self.lr_certain = swap, certain
            if not certain:
                log.warning("%s: cannot tell whether left and right are swapped since pairing (%s)",
                            self.addr, self.buds.predictions())
            else:
                log.info("%s: left/right labels %s", self.addr, "swapped back" if swap else "as paired")
        self.on_event("sides", self.addr, self.lr_info())

    def restore_stem(self):
        """Give stem presses back to the AirPods (double press = next track etc. on the phone again)."""
        if getattr(self, "stem_changed", False) and self.chan:
            try:
                self.chan.send(stem_enable(0x00))
                time.sleep(0.05)            # let it leave before the channel closes
                self.stem_changed = False
            except Exception as e:   # noqa: BLE001
                log.warning("%s: could not restore stem presses (%s)", self.addr, e)

    def out_batteries(self):
        """[(side, percent)] of the buds that are out of the case: the ones being used. The battery
        report covers both buds and the case; what matters on a leg is the bud that is on it."""
        bat = {k: v for k, v in (self.battery or {}).items() if isinstance(v, (int, float))}
        if "single" in bat:
            return [("single", bat["single"])]
        sides = [k for k in ("left", "right") if k in bat]
        if not sides:
            return []
        primary = getattr(self.buds, "primary", None)
        if primary in sides and self.ear:
            other = "right" if primary == "left" else "left"
            out = [side for side, st in ((primary, self.ear[0]), (other, self.ear[1]))
                   if st in (0, 1) and side in bat]             # 0 in ear, 1 out of ear; 2 / 3: in the case
            if out:
                return [(side, bat[side]) for side in out]
        if primary in sides:                                    # where the buds are is not known: the one that streams
            return [(primary, bat[primary])]
        live = [k for k in sides if bat[k] > 0]                 # (a bud in a closed case reports 0)
        return [(k, bat[k]) for k in (live or sides)[:1]] if len(live) != 2 else [(k, bat[k]) for k in live]

    def restore_ear(self):
        """Ear detection back on, as the AirPods had it for their normal use."""
        if self.ear_changed and self.chan:
            try:
                self.chan.send(ear_detection(True))
                time.sleep(0.05)
                self.ear_changed = False
            except Exception as e:   # noqa: BLE001
                log.warning("%s: could not switch ear detection back on (%s)", self.addr, e)

    def set_stem(self, on):
        """Settings changed while connected."""
        self.stem = bool(on)
        if not on:
            self.restore_stem()
        elif self.chan:
            try:
                self.chan.send(stem_enable(0x0E))
                self.stem_changed = True
            except Exception:   # noqa: BLE001
                pass

    def stop(self):
        self.stop_flag.set()
        if self.chan:
            try:
                for side in self.sides:
                    for p in self.state.start_raw_imu(side, 0):
                        self.chan.send(p)
                started = getattr(self, "wx_started", set())
                if started:
                    seq = next_wx_seq(3)
                    if "raw" in started:
                        self.chan.send(wx_stream(seq, SVC_ACCEL, 0))
                        self.chan.send(wx_stream(seq + 1, SVC_GYRO, 0))
                    if "fused" in started:
                        self.chan.send(wx_stream(seq + 2, SVC_DEVMOTION6, 0))
            except Exception:   # noqa: BLE001
                pass
            self.restore_stem()
            self.restore_ear()
        if self.probe and self.state is not None:
            log.info("%s: probe result: %s", self.addr, self.probe_summary())
        if self.record:
            self.record.close()
            self.record = None
