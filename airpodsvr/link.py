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
                       bud_shape, model_name, parse_device_info,
                       ear_detection, host_connect, rhid_frame, rhid_sub, set_report_interval, stem_enable, wx_raw_start,
                       wx_start_sequence, wx_stream)


_seq_lock = threading.Lock()
_cfg_lock = threading.Lock()        # links of several pairs update the same config lists at once

# config: addresses whose automatic ear detection we switched off and have not switched on again
# (the stop came while reconnecting, the app crashed or was killed). Put right at the next connect.
EAR_OFF_KEY = "ear_detection_off_by_us"


def _config_list_set(key, addr, present):
    """Add (present=True) or remove `addr` in the config list `key`, writing only on a change.
    -> whether it was in the list before."""
    from . import config as C
    with _cfg_lock:
        cur = list(C.load_config().get(key) or [])
        was = addr in cur
        if present and not was:
            C.save_config({key: cur + [addr]})
        elif not present and was:
            C.save_config({key: [a for a in cur if a != addr]})
        return was


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
        self.ear_noted = None            # EAR_OFF_KEY holds this address: True / False / None (not read)
        self.stem_changed = False        # stem presses taken over and not given back yet
        self.interval_us = int(interval_us)
        self.buds = BudSides(bud_info)
        self.swap, self.lr_certain = self.buds.resolve()
        self.want = tuple(sides)        # labels passed to on_sample
        # with corrected labels either bud may be the wanted one: talk to both
        self.sides = (0, 1) if bud_info else tuple(sides)
        self.stop_flag = threading.Event()
        self.bye_said = threading.Event()    # the link thread said goodbye (or ended): stop() waits for it
        self.chan = None
        self.state = None
        self.status = "idle"
        self.forgot = False                  # the AirPods no longer know us: pairing again is needed
        self.lost_checked = False            # config 'lost_pairing' looked at since connecting
        self.stalls = 0                      # how often the stream stopped (diagnostics)
        self.ear = None
        self.battery = {}
        self.device_name = self.model = None    # from the device information packet, e.g. "A2084"
        self.model_kind = None          # "raw" (AirPods 3) | "wx-raw" (Pro 2 / Pro 3 / 4 raw IMU)
                                        # | "fused" (SensorDataWX attitude, ~100 Hz)
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
        # switched on on the current channel, so the goodbye switches off exactly that
        self.wx_started = set()         # "raw" / "fused" SensorDataWX streams asked for
        self.armed = set()              # sides whose RemoteHID accel + gyro were started
        self.contained = {}             # what failed inside _pump without dropping the link -> count
        self.said = set()               # one-time diagnostics already logged

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
            # under the same lock as _clear_lost: another pair's link may be taking itself off the
            # list at the same moment, and one of the two changes would be lost
            _config_list_set("lost_pairing", self.addr, True)
        except Exception as e:   # noqa: BLE001
            log.debug("lost pairing not saved: %s", e)

    def _send(self, pkt):
        self._log_frame("tx", pkt)
        self.chan.send(pkt)

    def _arm(self, side):
        for p in self.state.start_raw_imu(side, self.interval_us):
            self._send(p)
        self.armed.add(side)

    def _contained(self, what, e, pkt=None):
        """Something failed inside _pump that concerns one packet or one sample: a packet the parser
        chokes on (untested models, a damaged PDU) or a bug in whoever receives the data. It used to
        drop the Bluetooth link, and a reconnect pages the buds again (seconds of frozen tracker, and
        the other pairs on the adapter wait meanwhile). Said once per kind, then only counted."""
        n = self.contained[what] = self.contained.get(what, 0) + 1
        if n == 1:
            log.warning("%s: %s failed (%s)%s; the link goes on, further failures are only counted",
                        self.addr, what, errtext(e), f", packet {bytes(pkt[:32]).hex()}" if pkt else "",
                        exc_info=True)
        elif n in (10, 100, 1000) or n % 10000 == 0:
            log.warning("%s: %s failed %d times so far (last: %s)", self.addr, what, n, errtext(e))

    def _consume(self, what, fn, *args):
        """Hand data to on_sample / on_event. Their failures stay theirs (see _contained); a
        ConnectionError still ends the connection, as it would anywhere else in the link."""
        try:
            fn(*args)
        except ConnectionError:
            raise
        except Exception as e:   # noqa: BLE001
            self._contained(what, e)

    def _note_ear_off(self, on):
        """Keep in the config whether these AirPods have ear detection switched off by us (see
        EAR_OFF_KEY). Written only when it changes; a failure costs only the self-repair."""
        if self.ear_noted is on:
            return
        try:
            _config_list_set(EAR_OFF_KEY, self.addr, on)
            self.ear_noted = on
        except Exception as e:   # noqa: BLE001
            log.debug("%s: ear detection note not saved: %s", self.addr, e)

    def _ear_on_connect(self):
        """Ear detection off while we use the buds (option on). With the option off, switch it back
        on if an earlier session switched it off and never could give it back."""
        if self.ear_off:
            self._note_ear_off(True)          # before sending: a crash right after must still be repaired
            self._send(ear_detection(False))
            if not self.ear_changed:
                log.info("%s: automatic ear detection switched off (given back when tracking stops)", self.addr)
            self.ear_changed = True
            return
        if self.ear_noted is None:
            try:
                from . import config as C
                self.ear_noted = self.addr in (C.load_config().get(EAR_OFF_KEY) or [])
            except Exception as e:   # noqa: BLE001
                log.debug("%s: config not read: %s", self.addr, e)
        if self.ear_noted:
            self._send(ear_detection(True))
            log.info("%s: automatic ear detection switched back on (left off by an earlier session)", self.addr)
            self._note_ear_off(False)

    def _clear_lost(self):
        """Connected with the key we have: the AirPods know us. Take them off the config's "pair
        again" list, which otherwise only the pair flow rewrites (a pair put there by mistake went
        on being shown as needing pairing)."""
        self.lost_checked = True
        try:
            if _config_list_set("lost_pairing", self.addr, False):
                log.info("%s: connected with our pairing: no longer listed as needing pairing", self.addr)
        except Exception as e:   # noqa: BLE001
            log.debug("%s: lost pairing not updated: %s", self.addr, e)

    def _close_record(self):
        rec, self.record = self.record, None
        if rec:
            try:
                rec.close()
            except OSError:
                pass

    # ------------------------------------------------------------------ main loop
    def run(self):
        try:
            self._run()
        finally:
            self.bye_said.set()
            self._close_record()          # here, not in stop(): the thread may still be writing it

    def _run(self):
        backoff = 1.0
        while not self.stop_flag.is_set():
            try:
                self.status = "connecting"
                self.chan = None                 # never the closed channel of an earlier connection
                self.chan = self.backend.open(self.addr)
                self.state = ProtocolState()
                self.wx_started, self.armed = set(), set()
                # stop() came while the buds were being paged: change nothing on them. What an
                # earlier, lost connection changed is still given back by the goodbye below, after
                # the handshake: before it the buds do not take commands, and the ear setting
                # would count as given back although it never was.
                if self.stop_flag.is_set() and not (self.stem_changed or self.ear_changed):
                    break
                for pkt, delay in HANDSHAKE_SEQUENCE:
                    self._send(pkt)
                    time.sleep(delay)
                if self.stop_flag.is_set():
                    break
                for side in self.sides:
                    self._send(host_connect(self.state.next_gen(), side))
                # double / triple / long press come to us (single press stays play/pause). Always
                # sent: with the option off, 0 also repairs buds left changed by a crash.
                self._send(stem_enable(0x0E if self.stem else 0x00))
                self.stem_changed = self.stem
                self._ear_on_connect()
                self.status = "waiting for sensors"
                if self.forgot or not self.lost_checked:
                    self._clear_lost()
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
                    if self.stop_flag.is_set():
                        # the goodbye, said here on the link's own thread and channel before it
                        # closes: said from stop() it raced with this close and was mostly lost.
                        # Only on a real stop: after a lost link the next connection follows.
                        self._goodbye()
                        self.bye_said.set()
                    self.chan.close()
            if self.stop_flag.is_set():
                break
            if self.forgot:
                self.status = "needs pairing"
            elif getattr(self.backend, "lost", False):
                # the USB adapter itself is gone: retrying through this backend cannot help, a new
                # tracking session (stop, start) opens the adapter again once it is back
                self.status = "adapter lost"
            else:
                self.status = f"reconnecting in {backoff:.0f}s"
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
        self.armed = set()
        self.hello_tries = {}                  # per connection
        # what this connection streams, from what arrives only (it was set to "fused" as soon as
        # SensorDataWX was asked for, and the window showed that for pairs that sent nothing)
        self.model_kind = None
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
                try:
                    info = parse_device_info(pkt)
                    events = self.state.feed(pkt, now)
                except ConnectionError:
                    raise
                except Exception as e:   # noqa: BLE001 - one packet, not the link (see _contained)
                    self._contained("parsing a packet", e, pkt)
                    info, events = None, []
                if info and info != (self.device_name, self.model):
                    self.device_name, self.model = info
                    name = self.model_name
                    log.info("%s: model %s%s", self.addr, self.model or "?", f" ({name})" if name else "")
                for ev in events:
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
                            self._consume("passing on a sample", self.on_sample, f"{self.addr}#{label}", s)
                        elif not self.buds.has_base and ("unwanted", label) not in self.said:
                            # never seen so far (a pair sends its primary bud's motion as side 0),
                            # and the tracker of this pair would stay still: say so once
                            self.said.add(("unwanted", label))
                            log.info("%s: motion data labelled side %d, not used by this link (it uses %s)",
                                     self.addr, label, ", ".join(str(w) for w in self.want) or "none")
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
                        self._consume("passing on an event", self.on_event, "ear", self.addr, ev[1])
                    elif kind == "stem":
                        self._consume("passing on an event", self.on_event, "stem", self.addr, ev[1])
                    elif kind == "battery":
                        self.battery = ev[1]
                        self._consume("passing on an event", self.on_event, "battery", self.addr, ev[1])
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
                # raw IMU first; if it does not come, the fused stream (and keep alternating). Once
                # fused samples came on this connection and raw never paired, a stall asks for the
                # fused stream again at once: asking for raw first kept the legs frozen 4 s longer.
                self._start_wx(fused="raw" in getattr(self, "wx_started", set()) and not self.state.wx_raw_seen
                               and (self.state.wx_seen or wx_attempts % 2 == 1))
                wx_attempts += 1
            if raw_known:
                self.model_kind = "raw"
            elif self.state.wx_raw_seen:
                self.model_kind = "wx-raw"
            elif self.state.wx_seen:
                self.model_kind = "fused"
                if "raw" in self.wx_started and "fused" not in self.said:
                    # raw IMU was asked for and the fused stream is what came: worth knowing when a
                    # Pro 3 (untested here) tracks worse than expected
                    self.said.add("fused")
                    log.warning("%s: SensorDataWX raw accelerometer + gyroscope (services 11 / 12) asked for "
                                "but no samples from it (%s): using the fused attitude stream instead "
                                "(service 16, about 100 Hz)", self.addr, self._wx_raw_text())
            if "raw" in self.wx_started and not self.state.wx_raw_seen and "unpaired" not in self.said:
                n = sum(getattr(self.state, "wx_raw_n", {}).values())
                n += sum(getattr(self.state, "wx_raw_odd", {}).values())
                if n >= 50:
                    self.said.add("unpaired")
                    log.warning("%s: SensorDataWX raw reports arrive but no accelerometer + gyroscope pair "
                                "forms (%s)", self.addr, self._wx_raw_text())
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
        rec = self.record
        if rec:
            # called from the probe flow's thread: the link thread closes the recording when it ends
            # (a cancelled flow stops the links meanwhile), and a mark then has nowhere to go
            try:
                rec.write(json.dumps({"t": round(time.monotonic() - self.t_start, 5), "mark": label}) + "\n")
                rec.flush()
            except (OSError, ValueError):
                pass
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

    @property
    def model_name(self):
        """What the buds are, e.g. "AirPods Pro 3" (from their model number), or None if not known."""
        return model_name(self.model)

    def _wx_raw_text(self):
        """What came of the SensorDataWX raw streams on this connection, for the log."""
        n = getattr(self.state, "wx_raw_n", None) or {}
        odd = getattr(self.state, "wx_raw_odd", None) or {}
        text = f"{n.get(SVC_ACCEL, 0)} accel and {n.get(SVC_GYRO, 0)} gyro reports"
        if odd:
            text += "; other layouts (service, bytes, first byte): " + ", ".join(
                f"{k} x{v}" for k, v in sorted(odd.items(), key=str)[:6])
        return text + (f"; model {self.model}" if self.model else "")

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
        self._consume("passing on an event", self.on_event, "sides", self.addr, self.lr_info())

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
                log.warning("%s: could not switch ear detection back on (%s); a later connection does it",
                            self.addr, e)
                return
            self._note_ear_off(False)

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

    def _goodbye(self):
        """Sensors off, stem presses and ear detection given back, on the current channel while it is
        still open. Only what was switched on on this channel is switched off; the stem and ear
        settings stay changed on the buds across a lost link, so they are given back here too.
        Never raises."""
        chan = self.chan
        if chan is None or getattr(chan, "closed", False):
            return                      # nothing to say it on (the ear setting is repaired later)
        try:
            if self.state is not None:
                for side in sorted(self.armed):
                    for p in self.state.start_raw_imu(side, 0):
                        self._send(p)
            started = self.wx_started
            if started:
                seq = next_wx_seq(3)
                if "raw" in started:
                    self._send(wx_stream(seq, SVC_ACCEL, 0))
                    self._send(wx_stream(seq + 1, SVC_GYRO, 0))
                if "fused" in started:
                    self._send(wx_stream(seq + 2, SVC_DEVMOTION6, 0))
        except Exception as e:   # noqa: BLE001
            log.debug("%s: sensors not switched off (%s)", self.addr, errtext(e))
        self.restore_stem()
        self.restore_ear()

    GOODBYE_WAIT = 1.5      # s stop() waits for the link thread's goodbye

    def stop(self):
        """Ask the link to end. A running link says its goodbye itself (see _goodbye), on its own
        thread, right before it closes its channel; said from here it raced with that close and the
        ear detection was hardly ever switched back on. This waits a moment for it, only while a
        channel is open (not while the buds are being paged), and never long."""
        self.stop_flag.set()
        running = self.is_alive()
        if running:
            chan = self.chan
            if self is not threading.current_thread() and chan is not None and not getattr(chan, "closed", False):
                self.bye_said.wait(self.GOODBYE_WAIT)
        else:
            self._goodbye()             # not running (never started, or ended): nobody else will
        if self.probe and self.state is not None:
            log.info("%s: probe result: %s", self.addr, self.probe_summary())
        rec = self.record
        if not running:
            self._close_record()        # a running link closes it when its thread ends
        elif rec:
            try:
                rec.flush()             # on disk already, should the thread still be paging a while
            except (OSError, ValueError):
                pass
