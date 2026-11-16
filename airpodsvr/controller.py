# SPDX-License-Identifier: Apache-2.0
"""UI-independent controller: everything the window can show or do, as plain data and methods.

Long interactive procedures (dongle preparation, pairing, calibration) run as *flows* in a worker
thread; a flow exposes its current text / buttons / choices and waits for the UI's answer.
"""
from __future__ import annotations

import collections
import logging
import math
import os
import subprocess
import sys
import threading
import time
import uuid

from . import config as C
from . import i18n, winsetup
from .tracker import ROLE_ZH, config_roles
from .transport import errtext

log = logging.getLogger(__name__)

SETTINGS_SCHEMA = [
    # key, label, type, extra ("help": one short line shown under the label)
    ("language", "Language", "choice", {"options": [[i, n] for i, n in enumerate(i18n.NAMES)]}),
    ("auto_yaw", "自動飄移校正", "bool", {"help": "站著、走直線或靜止時，慢慢把腿的方向對齊，並在使用中學習每顆耳機的飄移速度。"}),
    ("auto_yaw_gain", "校正強度", "float", {"min": 0.05, "max": 1.0, "step": 0.05}),
    ("auto_yaw_max_deg_s", "校正速度上限", "float", {"min": 0.1, "max": 2.0, "step": 0.1, "unit": "°/s"}),
    ("use_pose_model", "使用姿勢模型", "bool", {"help": "只在雙小腿（沒有大腿）時使用；需先在「進階」下載。"}),
    ("model_disagree_deg", "與實測相差多少就不採用", "float", {"min": 10, "max": 60, "step": 5, "unit": "°"}),
    ("height_override_m", "身高", "float", {"min": 0, "max": 2.2, "step": 0.01, "unit": "m"}),
    ("hip_offset_m", "臀部高度微調", "float", {"min": -0.15, "max": 0.15, "step": 0.01, "unit": "m"}),
    ("foot_tracker_offset_m", "腳追蹤點前移", "float", {"min": -0.1, "max": 0.2, "step": 0.01, "unit": "m"}),
    ("imu_interval_us", "耳機取樣率", "choice",
     {"options": [[0, "自動（建議）"], [10000, "100 Hz"], [20000, "50 Hz"], [5000, "200 Hz"]],
      "help": "自動：1～2 副 100 Hz，3 副以上 50 Hz。耳機一多，取樣率太高會斷斷續續。"}),
    ("output", "身體演算與輸出", "choice",
     {"options": [[0, "FBT Pods（預設）"], [1, "SlimeVR Server"], [2, "兩邊都送"]],
      "help": "FBT Pods：自己計算，經 VMT 送進 SteamVR，2～4 顆適用。SlimeVR Server：需 5 顆以上"
              "（雙小腿＋雙大腿＋腰或胸），要另外安裝。重新開始追蹤後生效。"}),
    ("loop_hz", "輸出頻率", "choice", {"options": [[60, "60 Hz"], [90, "90 Hz"], [120, "120 Hz"]]}),
    ("vmt_mode", "追蹤器類型", "choice", {"options": [[1, "一般追蹤器"], [7, "VIVE Tracker 相容"]]}),
    ("auto_launch_with_steamvr", "隨 SteamVR 自動啟動", "bool", {}),
    ("auto_tracker_roles", "自動指定追蹤器角色", "bool", {"help": "在 SteamVR 裡自動設為腰、左右腳、左右膝。"}),
    ("leg_model", "腿部演算", "choice",
     {"options": [[0, "FBT Pods（預設）"], [1, "SlimeVR 式（備案）"]],
      "help": "備案是移植自 SlimeVR Server 的鎖腳、關節限制與濾波；走路時腳可能提早被鎖住。追蹤中可直接切換。"}),
    ("skating_correction", "防滑步", "bool", {"help": "腳著地時鎖住，不跟著身體晃動滑走。"}),
    ("ear_detection_off", "使用期間關閉自動入耳偵測", "bool",
     {"help": "試驗功能。AirPods Pro 放在桌上或綁上後過幾秒就停止傳送時開啟；停止追蹤時會改回。"}),
    ("stem_actions", "耳機柄按鍵", "bool", {"help": "按兩下快速重置、按三下完整重置、長按暫停。"}),
    ("demo_sensors", "試用模式的耳機數", "choice",
     {"options": [[2, "2 顆（雙小腿）"], [3, "3 顆（加腰）"], [4, "4 顆（加雙大腿）"]]}),
    ("check_updates", "有新版 VMT 時提醒", "bool", {"help": "每天向 GitHub 查詢一次，不會自動安裝。"}),
]


ROLE_ORDER = ("left_shin", "right_shin", "left_thigh", "right_thigh", "pelvis")


def layout_text(roles):
    return "、".join(ROLE_ZH[r] for r in ROLE_ORDER if r in roles)


# ------------------------------------------------------------------ beeps
def beep(freq=880, ms=120):
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:   # noqa: BLE001
        time.sleep(ms / 1000)


# ------------------------------------------------------------------ flows
class Flow:
    def __init__(self, kind, title):
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.title = kind, title
        self.text, self.buttons, self.items, self.step = "", [], None, ""
        self.panel = None             # a live picture in the dialog, e.g. {"type": "buds", ...}
        self.picture = None           # a drawing of the pose asked for (web/poses.js), e.g. "ski"
        self.answer = None
        self.choice = None
        self.ev = threading.Event()
        self.done = False
        self.cancelled = False
        self.progress = False
        self.shown = 0                # counts the showings: an answer names the one it answers

    def show(self, text, buttons=(), step="", progress=False, panel=None, picture=None):
        self.ev.clear()
        self.answer = None
        self.shown += 1
        self.panel = panel
        self.picture = picture
        self.text, self.buttons, self.items, self.step, self.progress = text, list(buttons), None, step, progress

    def wait(self, timeout=None):
        self.ev.wait(timeout)
        return self.answer

    def choose(self, prompt, items):
        """Indices of the ticked items, or None (cancelled, or nothing ticked)."""
        self.show(prompt, ["確定", "取消"])
        self.items = list(items)
        a = self.wait()
        if a == "取消" or self.cancelled:
            return None
        if self.choice is None:              # no list came with the answer at all: the first item
            return [0]
        # Nothing ticked means "none of these", never "the first one": that paired whatever device
        # was listed first (maybe someone else's AirPods) and replaced the user's own pairing.
        picked = []
        for i in self.choice if isinstance(self.choice, (list, tuple)) else ():
            if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(self.items) and i not in picked:
                picked.append(i)
        return picked or None

    def reply(self, value, choice=None):
        if value == "取消":
            self.cancelled = True
        self.answer = value
        self.choice = choice
        self.ev.set()

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "title": self.title, "text": self.text,
                "buttons": self.buttons, "items": self.items, "step": self.step, "progress": self.progress,
                "panel": self.panel, "picture": self.picture, "done": self.done, "shown": self.shown}


class LogBuffer(logging.Handler):
    def __init__(self, n=300):
        super().__init__()
        self.lines = collections.deque(maxlen=n)
        self.seq = 0

    def emit(self, record):           # called with the handler lock held (logging.Handler.handle)
        try:
            msg = record.getMessage()
        except Exception:   # noqa: BLE001 - a bad format string must not break the caller
            msg = str(record.msg)
        self.seq += 1
        self.lines.append((self.seq, time.strftime("%H:%M:%S"), record.levelname, msg))

    def since(self, seq):
        """Lines newer than seq. Taken under the handler lock: other threads log while the window
        polls, and iterating the deque meanwhile raises "deque mutated during iteration"."""
        self.acquire()
        try:
            return [list(x) for x in self.lines if x[0] > seq]
        finally:
            self.release()


# ------------------------------------------------------------------ controller
class Controller:
    slime_up = None                    # a SlimeVR Server is holding its tracker port (None: not checked)
    vmt_new = None                     # tag of a VMT release newer than the one we install
    ui_seen = 0.0                      # last time the window asked for the state
    _stopping = None                   # a session stop() gave up waiting for: it still holds the dongle
    STOP_WAIT = 15.0                   # seconds stop() waits for a session to let go of the dongle
    # setup steps that open the Bluetooth adapter or change its driver: no tracking meanwhile
    DONGLE_FLOWS = ("dongle", "restore", "pair")

    def __init__(self):
        self.app = None
        self.flow = None
        self.logs = LogBuffer()
        logging.getLogger().addHandler(self.logs)
        self.cfg = C.load_config()
        self.dongle = ("unknown", None)
        self.vmt_ok = None
        self.steamvr_installed = None
        self.lock = threading.RLock()      # start / stop of tracking, one at a time (see _life)
        self._slow_stop = threading.Event()
        from .monitor import SensorMonitor
        self.monitor = SensorMonitor(self)
        threading.Thread(target=self._slow_loop, daemon=True).start()

    # ---------------------------------------------------------------- slow checks
    def _slow_loop(self):
        # the device check starts PowerShell: every 8 s while setting up, only once a minute while
        # tracking (so it does not cost frame time in VR)
        while not self._slow_stop.is_set():
            try:
                self.refresh()
                self._check_updates()
            except Exception:   # noqa: BLE001 - this thread must keep the status rows alive
                log.exception("status check")
            self._slow_stop.wait(self._check_interval())

    def _check_interval(self):
        """Seconds until the next device check (it starts PowerShell and walks the USB devices):
        often only while the user is looking at a setup that is not finished."""
        if self.running:
            return 60
        setting_up = self.dongle[0] != "ready" or (self.output_mode() != 1 and not self.vmt_ok)
        watching = time.monotonic() - self.ui_seen < 15
        return 8 if (setting_up and watching) else 30

    def output_mode(self):
        """0: our own body solver through VMT (default), 1: a SlimeVR Server does the body, 2: both."""
        cfg = getattr(self, "cfg", None)
        v = (C.load_config() if cfg is None else cfg).get("output", 0)
        return v if v in (0, 1, 2) else 0

    @staticmethod
    def slimevr_listening(port=6969):
        """Is something (a SlimeVR Server) listening on the tracker port of this PC?

        Asked the way a tracker would: one heartbeat packet (which the server ignores) to the
        loopback address. A closed port answers at once with "unreachable"; silence means somebody
        has it open. (Trying to bind the port instead does not work on Windows, where a socket on
        127.0.0.1 and one on all addresses can share a port; and binding all addresses could make
        the firewall ask.)"""
        import socket
        import struct
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.settimeout(0.2)
            s.connect(("127.0.0.1", int(port)))
            s.send(struct.pack(">iq", 0, 0))
            s.recv(64)
            return True                      # it even answered
        except socket.timeout:
            return True
        except OSError:
            return False                     # refused / reset: nobody there
        finally:
            s.close()

    def _check_updates(self):
        if not self.cfg.get("check_updates", True):
            self.vmt_new = None
            return
        self.vmt_new = winsetup.vmt_update_available()

    def refresh(self):
        self.cfg = C.load_config()
        if sys.platform != "win32" or self.cfg.get("backend") == "bluez":
            self.dongle = ("ready", None)
        else:
            try:
                self.dongle = winsetup.dongle_state(self.cfg)
            except Exception as e:   # noqa: BLE001
                log.debug("dongle check: %s", e)
        if sys.platform == "win32":
            self.vmt_ok = winsetup.vmt_registered()
        if self.output_mode() in (1, 2) and not self.running:
            self.slime_up = self.slimevr_listening(self.cfg.get("slimevr_port", 6969))
            self.steamvr_installed = winsetup.steamvr_dir() is not None

    def shutdown(self):
        self._slow_stop.set()
        self.cancel_flow()
        self.monitor.stop()
        self.stop()

    def busy(self):
        """What must not be interrupted by closing the window: "tracking", "working" (a setup step
        that is running: installing, downloading, measuring), or None. A dialog that only waits
        for an answer is not work: closing the window answers it."""
        if self.running:
            return "tracking"
        fl = self.flow
        if fl is not None and not fl.done and fl.progress:
            return "working"
        return None

    def cancel_flow(self):
        fl = self.flow
        if fl is not None and not fl.done:
            fl.reply("取消")

    # ---------------------------------------------------------------- running
    @property
    def running(self):
        """A session is running, or is still stopping (until it has ended it holds the dongle)."""
        return self._alive(self.app) or self._alive(self._stopping)

    @staticmethod
    def _alive(app):
        th = getattr(app, "thread", None)
        return th is not None and th.is_alive()

    def _life(self):
        """The lock that makes starting and stopping tracking one step each. Made here when missing
        (a Controller built without __init__, as the tests do)."""
        lock = self.__dict__.get("lock")
        if lock is None:
            lock = self.__dict__.setdefault("lock", threading.RLock())
        return lock

    def setup_holds_dongle(self):
        """A setup step that opens the Bluetooth adapter or changes its driver (preparing it, giving
        it back, pairing) is open: tracking would open the adapter a second time and fail, or
        disturb the step."""
        fl = getattr(self, "flow", None)
        return fl is not None and not fl.done and fl.kind in self.DONGLE_FLOWS

    def ready_to_run(self):
        roles = config_roles(self.cfg)
        if self.output_mode() == 1:          # a SlimeVR Server does the body: any paired pair will do
            return self.dongle[0] == "ready" and bool(roles)
        return self.dongle[0] == "ready" and bool(roles.get("left_shin")) and bool(roles.get("right_shin"))

    def start(self, demo=False):
        # One start or stop at a time. Building a session takes a while (calibration, the pose
        # model), and two requests in that time (a double click, the tray and the window, SteamVR's
        # autostart) made two sessions: one out of reach of stop(), holding the dongle and the VMT
        # trackers. A start that comes while stop() waits for the old session runs after it.
        with self._life():
            if self.running:
                return
            if self.setup_holds_dongle():
                log.info("start tracking: not now, the %s step is using the Bluetooth adapter", self.flow.kind)
                return
            self._stopping = None            # (one that hung in its shutdown has ended by now: let it go)
            from .app import App
            app = App(demo=demo)
            # the sensor page was showing a session that ended by itself: show this one instead
            retap = self.monitor.mode == "tap"
            if self.monitor.mode in ("live", "sim", "tap"):   # one connection to the dongle at a time
                self.monitor.stop()
            self.app = app
            app.start()
            if retap and self._alive(app):
                self.monitor.start()

    def stop(self):
        """Stop tracking. Returns once the session has let go of the dongle, so whatever opens it next
        (a new session, pairing, the sensor page, the diagnosis) finds it free; until then `running`
        stays true. A session that hangs gets STOP_WAIT seconds, and counts as running until it ends."""
        with self._life():
            app = self.app if self.app is not None else self._stopping
            if app is not None:
                t0 = time.monotonic()
                app.stop()                   # tells it to end; waits a few seconds itself
                th = getattr(app, "thread", None)
                if th is not None and th is not threading.current_thread():
                    th.join(max(0.0, self.STOP_WAIT - (time.monotonic() - t0)))
                # kept as "stopping" before it stops being the session: in between, `running` read
                # False for a session that still holds the dongle
                self._stopping = app if self._alive(app) else None
                if self.app is app:
                    self.app = None
                if self._stopping is not None:
                    log.warning("tracking has not ended %.0f s after stop: the Bluetooth adapter is still in use",
                                time.monotonic() - t0)
            # (the sensor page stays on the session while it ends: its own start would find the dongle busy)
            if self.monitor.mode == "tap":
                self.monitor.stop()

    def _dongle_free(self, fl):
        """After stop(): False, and says so, while the last session still holds the adapter."""
        if not self._alive(self._stopping):
            return True
        fl.show("上一次的追蹤還在結束中，藍芽還沒有釋放。請過幾秒後再試一次。", ["關閉"])
        fl.wait()
        return False

    def action(self, name, hand="left_hand", delay=0.0):
        """A reset from the window. With `delay` it runs after a countdown with beeps: the person
        who clicked has to get into the pose first (stand straight, or hold the controller on the
        shin). Asking again while one counts cancels it."""
        if not self.app:
            return
        pending = getattr(self, "pending", None)
        if pending is not None:
            pending["cancel"].set()
            self.pending = None
            if (pending["name"], pending["hand"]) == (name, hand):
                return                                   # the same button again: just cancel
        try:
            delay = max(0.0, min(float(delay or 0.0), 15.0))
        except (TypeError, ValueError):
            delay = 0.0
        if not delay:
            self.app.request(name, hand)
            return
        job = {"name": name, "hand": hand, "at": time.monotonic() + delay, "cancel": threading.Event()}
        self.pending = job

        def run():
            left = int(round(delay))
            while left > 0:
                beep(660, 90)
                if job["cancel"].wait(max(0.0, job["at"] - left + 1 - time.monotonic())):
                    return
                left -= 1
            if job["cancel"].is_set() or self.pending is not job:
                return
            self.pending = None
            beep(1320, 200)
            app = self.app
            if app:
                app.request(name, hand)
        threading.Thread(target=run, daemon=True).start()

    def countdown(self):
        """{"name", "hand", "left"} of the reset that is counting down, or None."""
        job = getattr(self, "pending", None)
        if job is None:
            return None
        return {"name": job["name"], "hand": job["hand"], "left": round(max(0.0, job["at"] - time.monotonic()), 2)}

    # ---------------------------------------------------------------- settings
    def settings(self):
        cfg = C.load_config()
        return [{"key": k, "label": lbl, "type": t, "value": cfg.get(k), **x} for k, lbl, t, x in SETTINGS_SCHEMA]

    def update_settings(self, values: dict):
        schema = {k: (t, x) for k, _, t, x in SETTINGS_SCHEMA}
        clean = {}
        for k, v in (values or {}).items():
            if k not in schema:
                continue
            kind, extra = schema[k]
            try:                           # a value that makes no sense is dropped, never stored
                if kind == "bool":
                    v = bool(v)
                elif kind == "float":
                    v = float(v)
                    if not math.isfinite(v):
                        continue
                    v = min(max(v, extra.get("min", v)), extra.get("max", v))
                elif kind == "choice":
                    v = int(v)
                    if v not in [o[0] for o in extra.get("options", [])]:
                        continue
            except (TypeError, ValueError):
                continue
            clean[k] = v
        if clean:
            C.save_config(clean)
        self.cfg = C.load_config()
        # only into a session that runs (the lock: not one that stop() is ending). One that has ended
        # has closed its pose model, and the "use pose model" setting would load it again into a
        # worker nobody closes (~100 MB). The next session reads the settings anyway.
        with self._life():
            app = self.app
            if self._alive(app):
                app.apply_settings(self.cfg)
        return self.settings()

    # ---------------------------------------------------------------- state for the UI
    def state(self, since_log=0):
        self.ui_seen = time.monotonic()
        app = self.app                     # read once: stop() lets go of it from another thread
        st = app.status() if app else None
        running = self.running
        roles = config_roles(self.cfg)
        dstate, dev = self.dongle
        rows = []
        # dongle
        if dstate == "ready":
            rows.append(("dongle", "ok", "已就緒" + (f" · {dev.name}" if dev and dev.name else ""), "準備藍芽"))
        elif dstate == "needs_driver":
            rows.append(("dongle", "warn", "已插上，需要安裝驅動", "安裝驅動"))
        elif dstate == "missing":
            rows.append(("dongle", "bad", "未設定給 AirPods 用的藍芽", "準備藍芽"))
        else:
            rows.append(("dongle", "idle", "檢查中…", None))
        # airpods
        if st and st["demo"]:
            n = len(st["trackers"])
            rows.append(("airpods", "idle", f"試用模式中（模擬 {n} 顆：{layout_text(st['trackers'])}）", None))
        elif st and st["links"]:
            zh = {"streaming": "串流中", "connecting": "連線中", "waiting for sensors": "等待感測器", "stopped": "已停止",
                  "idle": "準備中", "needs pairing": "耳機已不認得這台電腦", "stalled": "資料中斷，重試中",
                  "adapter lost": "USB藍芽已中斷：插回後請停止再開始追蹤"}
            kinds = {"raw": "原始 IMU 約 100 Hz", "wx-raw": "原始 IMU（Pro／4，實驗性）",
                     "fused": "融合姿態約 100 Hz（實驗性）"}

            def link_text(l):
                s = l["status"]
                s = "重新連線中" if s.startswith("reconnecting") else zh.get(s, s)
                return s + (f"（{kinds[l['kind']]}）" if l.get("kind") in kinds else "")
            ok = all(t["ok"] for t in st["trackers"].values())
            text = "、".join(link_text(l) for l in st["links"]) + f" · {layout_text(st['trackers'])}"
            if any((l.get("lr") or {}).get("certain") is False for l in st["links"]):
                # the AirPods report a different primary bud than at pairing and it is not yet known
                # whether that swaps the two data streams: say so instead of maybe tracking crossed legs
                ok = False
                text = ("左右可能和配對時相反（耳機的主耳換邊了）：請停止追蹤，做一次「配對 → 只分左右」；"
                        "做過一次後就會自動判斷 · ") + text
            if any(l["status"] == "needs pairing" for l in st["links"]):
                ok = False
                text = ("有耳機已不認得這台電腦（被重設或重新配對過）：請停止追蹤，到「配對 → 加入耳機」"
                        "把它重新配對一次 · ") + text
            if any(l.get("no_motion") for l in st["links"]):
                # connected but silent: almost always an Apple device holding the AirPods
                ok = False
                text = "已連線但沒有動作資料：請關閉 iPhone／iPad／Mac 的藍牙（自動重試中） · " + text
            rows.append(("airpods", "ok" if ok else "warn", text, None))
        elif [a for a in (self.cfg.get("lost_pairing") or []) if a in (self.cfg.get("airpods") or [])]:
            # seen at the last connection: these no longer accept the pairing we have
            lost = [a for a in self.cfg["lost_pairing"] if a in self.cfg["airpods"]]
            rows.append(("airpods", "bad", "需要重新配對：" + "、".join("…" + a[-5:] for a in lost)
                         + "（耳機已不認得這台電腦）", "配對"))
        elif self.output_mode() == 1 and roles:
            rows.append(("airpods", "ok", f"已配對 {len(roles)} 副：每副一顆，身體部位在 SlimeVR 裡分配", "重新配對"))
        elif roles.get("left_shin") and roles.get("right_shin"):
            rows.append(("airpods", "ok", f"已配對 {len(roles)} 顆：{layout_text(roles)}", "重新配對"))
        elif self.cfg.get("airpods"):
            rows.append(("airpods", "warn", "已配對，尚未分左右", "分左右"))
        else:
            rows.append(("airpods", "bad", "尚未配對", "配對"))
        # the body: a SlimeVR Server (default), our own solver through VMT, or both
        mode = self.output_mode()
        if mode in (1, 2):
            sl = st.get("slimevr") if st else None
            if running and sl and sl[1]:
                if sl[0] == sl[1]:
                    rows.append(("slimevr", "ok", f"已接上 {sl[0]} 顆耳機：身體比例與安裝角度校正請在 SlimeVR 裡做", None))
                elif sl[0]:
                    rows.append(("slimevr", "warn", f"只接受了 {sl[0]}/{sl[1]} 顆：請在 SlimeVR 視窗按「連接追蹤器」", "說明"))
                else:
                    rows.append(("slimevr", "warn", "沒有回應：請開啟 SlimeVR Server；第一次要在它的視窗接受新追蹤器", "說明"))
            elif self.slime_up:
                rows.append(("slimevr", "ok", "SlimeVR Server 執行中", None))
            else:
                rows.append(("slimevr", "bad", "SlimeVR Server 沒有開啟（需另外安裝）", "說明"))
        if mode == 1:
            rows.append(("vmt", "idle", "不使用（身體交給 SlimeVR Server）", None))
        elif self.vmt_ok:
            newer = f" · VMT 有新版 {self.vmt_new}（目前驗證過的是 {winsetup.VMT_TAG}，不會自動更新）" if self.vmt_new else ""
            if running and st and st.get("vmt_alive") is False:
                rows.append(("vmt", "warn", "已安裝，但 SteamVR 裡的 VMT 沒有回應：請重新啟動 SteamVR" + newer, None))
            elif running and st and st.get("vmt_alive"):
                rows.append(("vmt", "ok", "已安裝，運作中" + newer, None))
            else:
                rows.append(("vmt", "ok", "已安裝" + newer, None))
        elif self.vmt_ok is False:
            rows.append(("vmt", "bad", "未安裝", "安裝"))
        else:
            rows.append(("vmt", "idle", "—", None))
        # steamvr
        if st:
            rows.append(("steamvr", "ok" if st["steamvr"] == "已連線" else "warn", st["steamvr"], None))
        elif self.steamvr_installed is False:
            rows.append(("steamvr", "bad", "找不到 SteamVR（請安裝並執行一次）", None))
        else:
            rows.append(("steamvr", "idle", "開始追蹤時連線", None))

        # one entry per bud that is out of its case (the ones in use), named after where it is worn
        battery = []
        if st:
            place = {src.partition("#")[0]: ROLE_ZH.get(role, role) for role, src in roles.items()}
            ear = {"left": "左耳", "right": "右耳", "single": "耳機"}
            for l in st["links"]:
                for side, level in l.get("out") or []:
                    name = place.get(l["addr"]) or f"…{l['addr'][-5:]}"
                    battery.append({"label": f"{name}・{ear.get(side, side)}", "level": level})
        logs = self.logs.since(since_log)
        return {
            "rows": [{"key": k, "level": lv, "text": t, "fix": f} for k, lv, t, f in rows],
            "running": running, "demo": bool(st and st["demo"]), "ready": self.ready_to_run(),
            "paused": bool(st and st["paused"]), "calibrated": bool(st and st["calibrated"]),
            # who does the body (while tracking: what this session was started with)
            "output": (st or {}).get("output") or ("vmt", "slimevr", "both")[self.output_mode()],
            "message": st["message"] if st else "", "countdown": self.countdown(),
            "trackers": st["trackers"] if st else {}, "model": st["model"] if st else None,
            "battery": battery, "pose": self.pose_snapshot(),
            "flow": self.flow.to_dict() if self.flow and not self.flow.done else None,
            "logs": logs, "platform": sys.platform, "version": self.version(),
        }

    @staticmethod
    def version():
        from .version import text
        return text()

    def pose_snapshot(self):
        app = self.app
        if not app or app.core.last_pose is None or app.frame is None or app.frame.hmd is None:
            return None
        p = app.core.last_pose
        R_h, p_h = app.frame.hmd

        def v(x):
            return [round(float(c), 3) for c in x]
        out = {"head": v(p_h), "head_fwd": v(-R_h[:, 2]), "hip": v(p.hip), "hip_right": v(p.R_hip[:, 0])}
        for leg, r in p.legs.items():
            out[leg] = {"knee": v(r.knee), "ankle": v(r.ankle), "toe": v(r.ankle - 0.15 * r.R_foot[:, 2]),
                        "planted": round(float(r.planted), 2)}
        return out

    # ---------------------------------------------------------------- flows
    def flow_buds(self):
        """Live buds of the pairing screen (None when no such screen is open)."""
        ident = getattr(self, "flow_ident", None)
        if ident is None:
            return None
        try:
            snap = dict(ident.snapshot())
        except Exception as e:   # noqa: BLE001 - the identifier is being closed
            log.debug("flow buds: %s", e)
            return None
        # the places to hand out follow the buds that stream right now: a pair that connects late
        # joins in, one that drops out leaves
        from .tracker import layout_for
        live = [b["src"] for b in snap.get("buds", []) if b.get("here", b.get("streaming", True))]
        snap["roles"] = [[r, ROLE_ZH[r]] for r in layout_for(len(live))]
        snap["expected"] = getattr(self, "flow_expected", len(live))
        return snap

    def _track_then_calibrate(self, fl):
        """After pairing: start tracking and open the calibration wizard, as one step."""
        t0 = time.time()
        while not fl.done and time.time() - t0 < 5:
            time.sleep(0.05)
        try:
            self.start(False)
            t0 = time.time()
            while time.time() - t0 < 40:
                app = self.app
                if app is None:
                    return
                if not self.running and time.time() - t0 > 3:
                    return                       # it stopped again: the overview says why
                try:
                    tr = app.status()["trackers"]
                    if tr and all(v["ok"] for v in tr.values()):
                        break
                except Exception:   # noqa: BLE001 - still starting
                    pass
                time.sleep(0.3)
            if self.running:
                # (a SlimeVR Server has its own resets and mounting calibration: ours do nothing there)
                self.start_flow("slimevr" if self.output_mode() == 1 else "wizard")
        except Exception:   # noqa: BLE001
            log.exception("start after pairing")

    def answer_flow(self, flow_id, value, choice=None, shown=None):
        """`shown`: which showing of the step the window answered. A late click (or a Cancel from a
        dialog shown again unchanged) used to land on whatever step came next."""
        fl = self.flow
        if fl and fl.id == flow_id:
            if shown is not None and shown != getattr(fl, "shown", shown):
                log.info("flow answer %r for an earlier step ignored", value)
                return
            fl.reply(value, choice)

    def _run_flow(self, kind, title, fn):
        if self.flow and not self.flow.done:
            return self.flow.to_dict()
        fl = Flow(kind, title)
        self.flow = fl

        def run():
            try:
                fn(fl)
            except Exception as e:   # noqa: BLE001
                log.exception(title)
                fl.show(f"發生錯誤：{e}", ["關閉"])
                fl.wait(600)
            finally:
                fl.done = True
                threading.Thread(target=self.refresh, daemon=True).start()
        threading.Thread(target=run, daemon=True).start()
        return fl.to_dict()

    def start_flow(self, kind):
        flows = {"dongle": ("準備藍芽", self._flow_dongle), "restore": ("還原藍芽", self._flow_restore),
                 "vmt": ("安裝虛擬追蹤器", self._flow_vmt), "pair": ("配對 AirPods", self._flow_pair),
                 "wizard": ("校正精靈", self._flow_wizard), "model": ("姿勢模型", self._flow_model),
                 "slimevr": ("SlimeVR Server", self._flow_slimevr)}
        if kind not in flows:
            return None
        title, fn = flows[kind]
        return self._run_flow(kind, title, fn)

    def _flow_dongle(self, fl):
        self.monitor.stop()
        self.stop()
        if not self._dongle_free(fl):
            return
        state, dev = self.dongle
        target = dev if state == "needs_driver" else None
        if target is None:
            fl.show("要用哪一個藍牙給 AirPods？\n\n"
                    "• USB藍芽（建議）：另外插一支 USB 藍芽給 AirPods 專用，電腦原本的藍芽不受影響。\n"
                    "• 內建藍芽：用筆電／主機板內建的藍芽。使用期間 Windows 的藍芽會停用，"
                    "其他藍芽耳機、手把無法連線；之後可在「進階」一鍵還原。",
                    ["USB藍芽", "內建藍芽", "取消"], "1 / 3")
            way = fl.wait()
            if way == "USB藍芽":
                fl.show("請先把要給 AirPods 用的 USB藍芽【拔掉】，然後按「下一步」。",
                        ["下一步", "取消"], "1 / 3")
                if fl.wait() != "下一步":
                    return
                fl.show("正在記錄目前的裝置…", progress=True, step="1 / 3")
                before = winsetup.list_bt_usb()
                fl.show("現在【插上】那支 USB藍芽。\n\n正在等待…（60 秒內）", ["取消"], "2 / 3", progress=True)
                target = winsetup.wait_for_new_dongle(before, 60, stop=lambda: fl.cancelled)
                if target is None:
                    if not fl.cancelled:
                        fl.show("沒有偵測到新的 USB藍芽。請確認插的是藍芽（不是 Wi-Fi 網卡），"
                                "或換一個 USB 孔再試。", ["關閉"])
                        fl.wait()
                    return
            elif way == "內建藍芽":
                fl.show("正在列出內建藍芽…", progress=True, step="2 / 3")
                devs = winsetup.list_bt_usb()
                if not devs:
                    fl.show("找不到內建藍芽（或它不是 USB 介面，無法使用）。", ["關閉"])
                    fl.wait()
                    return
                labels = [f"{d.name or '藍芽裝置'}  ({d.vid:04X}:{d.pid:04X}) — {winsetup.chip_info(d)[1]}"
                          + ("  [已可使用]" if d.is_winusb else "") for d in devs]
                pick = fl.choose("選擇要給 AirPods 用的藍芽（選一個）", labels)
                if not pick:
                    return
                target = devs[pick[0]]
            else:
                return
        support, chip = winsetup.chip_info(target)
        if support == "unsupported":
            fl.show(f"「{target.name}」\n{chip}。\n\n這種晶片需要的驅動程式 FBT Pods 還沒有，請改用 Realtek 晶片的"
                    "USB藍芽 5.x（大部分平價的都是）。", ["關閉"])
            fl.wait()
            return
        if not target.is_winusb:
            fl.show(f"找到：{target.name}\n{chip}\n\n接下來 Windows 會詢問系統管理員權限，請按「是」。\n"
                    "安裝約需 30 秒～2 分鐘。", ["安裝", "取消"], "3 / 3")
            if fl.wait() != "安裝":
                return
            fl.show("正在安裝驅動…請不要拔掉藍芽。", progress=True, step="3 / 3")
            winsetup.install_winusb(target)
        else:
            C.save_config({"backend": "bumble", "bumble_transport": target.bumble_spec,
                           "dongle": {"instance_id": target.instance_id, "name": target.name}})
        # try it: opens the adapter and loads its firmware, exactly like tracking will
        fl.show("正在測試藍芽（需要時會自動載入韌體）…", progress=True, step="3 / 3")
        for attempt in range(3):              # Windows re-enumerates the device after a driver change
            time.sleep(2.0 + 2.0 * attempt)
            ok, info = winsetup.probe_dongle(C.load_config())
            if ok:
                break
        self.refresh()
        if ok:
            fl.show(f"完成！藍芽測試成功（{info}）。\n下一步：配對 AirPods。", ["完成"], "3 / 3")
            fl.wait()
            return
        extra = ("\n\n這個晶片是實驗性支援，若一直失敗，請把「進階 → 紀錄」的內容回報，幫助改善。"
                 if support == "experimental" else "")
        fl.show(f"驅動已安裝，但藍芽測試失敗：\n{info}{extra}\n\n可以拔插後到「進階」再試一次；"
                "如果一直失敗，按「還原」把它還給 Windows。", ["還原", "關閉"], "3 / 3")
        if fl.wait() == "還原":
            st, d = winsetup.dongle_state(C.load_config())
            if d is not None and d.is_winusb:
                self._restore(fl, d)
            else:
                fl.show("沒有需要還原的藍芽。", ["關閉"])
                fl.wait()

    def _restore(self, fl, dev):
        """Give the adapter back to Windows and say what really happened (the uninstall runs in an
        elevated window we cannot read, so the device is looked at again afterwards)."""
        fl.show("正在還原…（需要系統管理員權限）", progress=True)
        winsetup.restore_driver(dev)
        ok = winsetup.wait_restored(dev)
        if ok is False:
            fl.show(f"「{dev.name}」還在使用 AirPods 專用驅動，還原沒有成功。\n\n"
                    "可以再試一次；或手動還原：開啟「裝置管理員」→ 通用序列匯流排裝置 → 在 "
                    "FBT Pods Bluetooth (WinUSB) 上按右鍵 →「解除安裝裝置」並勾選「刪除此裝置的驅動程式」，"
                    "然後拔掉再插上（內建藍芽則重新開機）。", ["關閉"])
            fl.wait()
            return False
        C.save_config({"dongle": {}})
        self.refresh()
        fl.show("已還原。USB藍芽拔掉再插上（內建藍芽則重新開機），即可給 Windows 使用。", ["完成"])
        fl.wait()
        return True

    def _flow_model(self, fl):
        """Download the HMD-Poser weights (MIT, trained on AMASS: non-commercial use)."""
        import urllib.request
        from .hmdposer import MODEL_FILE, MODEL_URL, load_checkpoint
        # an existing copy wherever it is (data, app folder); new downloads go to the user data
        # folder: the app folder is read-only when installed and is replaced by updates
        dest = C.abspath(os.path.join("models", MODEL_FILE))
        have = os.path.exists(dest)
        if not have:
            dest = os.path.join(C.DATA, "models", MODEL_FILE)
        fl.show("姿勢模型（HMD-Poser，CVPR 2024）用頭顯、手把和小腿的資料推算整個身體，膝蓋與腰的位置會更自然。\n\n"
                "• 程式碼與權重：MIT 授權（Pico AI Research）\n"
                "• 權重以 AMASS 動作資料訓練：只限非商業用途\n"
                "• 下載約 103 MB，存在 models 資料夾，不會上傳任何資料\n\n"
                + ("已經下載過了。" if have else "要下載嗎？"),
                (["刪除模型", "關閉"] if have else ["下載", "取消"]))
        a = fl.wait()
        if a == "刪除模型":
            os.remove(dest)
            if self.app is not None:
                self.app.core.close_model()
            fl.show("已刪除，改用幾何推算。", ["關閉"])
            fl.wait()
            return
        if a != "下載":
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        try:
            with urllib.request.urlopen(urllib.request.Request(MODEL_URL, headers={"User-Agent": "FBTPods"}),
                                        timeout=60) as r, open(tmp, "wb") as f:
                declared = int(r.headers.get("Content-Length") or 0)
                total = declared or 103_423_171
                got = 0
                # one showing for the whole download, only its text changes: every show() counts as
                # a new step, and answer_flow drops an answer for an earlier one, so a Cancel clicked
                # between two 1 MB chunks was ignored and the download ran to the end
                fl.show(f"下載中… {got / 1e6:.0f} / {total / 1e6:.0f} MB", ["取消"], progress=True)
                while True:
                    if fl.cancelled:
                        raise RuntimeError("已取消")
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    fl.text = f"下載中… {got / 1e6:.0f} / {total / 1e6:.0f} MB"
            if declared and got != declared:
                raise RuntimeError(f"下載不完整（{got / 1e6:.0f} / {declared / 1e6:.0f} MB），請再試一次")
            sd = load_checkpoint(tmp)                     # also checks it is the expected network
            if "pose_est.2.weight" not in sd or len(sd) < 200:
                raise RuntimeError("下載的檔案不是 HMD-Poser 模型")
            os.replace(tmp, dest)
        except Exception as e:   # noqa: BLE001
            try:
                os.remove(tmp)
            except OSError:
                pass
            fl.show(f"下載失敗：{e}", ["關閉"])
            fl.wait()
            return
        use = bool(C.load_config().get("use_pose_model", True))
        # into the running session only, and only when the setting wants it: a session that has
        # ended (SteamVR closed) already closed its model, and loading one there starts a worker
        # with ~100 MB of weights that nobody closes. The next session loads it when it starts.
        with self._life():
            app = self.app
            if use and self._alive(app):
                app.core.load_model(app.cfg)
        if use:
            fl.show("完成！姿勢模型已啟用（設定裡的「使用姿勢模型」可以關閉）。", ["完成"])
        else:
            fl.show("完成！姿勢模型已下載。設定裡的「使用姿勢模型」目前是關閉的，打開後才會使用。", ["完成"])
        fl.wait()

    def _flow_restore(self, fl):
        self.monitor.stop()
        fl.show("正在檢查藍芽…", progress=True)
        try:                               # looked up now: the cached state can be a minute old
            state, dev = winsetup.dongle_state(C.load_config())
        except Exception as e:   # noqa: BLE001
            log.warning("dongle check: %s", e)
            state, dev = self.dongle
        if dev is None or not dev.is_winusb:
            fl.show("沒有需要還原的藍芽。\n（如果藍芽沒有插著，請先插上再按一次。）", ["關閉"])
            fl.wait()
            return
        fl.show(f"把「{dev.name}」還給 Windows 一般藍牙使用？\n之後 FBT Pods 將無法使用它。", ["還原", "取消"])
        if fl.wait() != "還原":
            return
        self.stop()
        if not self._dongle_free(fl):
            return
        self._restore(fl, dev)

    SLIMEVR_SETUP_URL = "https://docs.slimevr.dev/quick-setup"

    SLIMEVR_HELP = (
        "把身體的計算交給 SlimeVR Server，FBT Pods 只負責連上 AirPods 並送出每顆耳機的方向。\n"
        "⚠ SlimeVR 的骨架需要腰或胸的追蹤器：請準備 5 顆以上（雙小腿＋雙大腿＋腰或胸）。"
        "只有 4 顆以下時腰和上半身會很不自然，請改用「FBT Pods 自己算」。\n\n"
        "【第一次設定】\n"
        "① 安裝並開啟 SlimeVR Server（含它的 SteamVR 驅動；FBT Pods 不附帶）。\n"
        "② SlimeVR 的設定精靈問追蹤器類型時，第三方追蹤器選「Wi-Fi」（不是接收器／加密狗）。\n"
        "③「透過 Wi-Fi 連接」那一頁：按「跳過 Wi-Fi 設定」，或名稱、密碼隨便填再按「送出！」。"
        "耳機不經過 Wi-Fi，電腦沒有 Wi-Fi 也沒關係。\n"
        "④「連接追蹤器」那一頁要你用 USB 接追蹤器：不用理會，把「找不到追蹤器」的視窗關掉，"
        "按「所有的追蹤器都連接好了」，一路到 SlimeVR 的首頁。\n"
        "⑤ 回到 FBT Pods 按「開始追蹤」（不用先開 SteamVR）。\n"
        "⑥ SlimeVR 會跳出「找到了新的追蹤器！」：按「好喔！」，每顆耳機一次。"
        "不要按「別管它」（按了那顆會被忽略）。停在「連接追蹤器」那一頁時不會跳出來，要在首頁等。\n"
        "⑦ SlimeVR 左邊「追蹤器分配」：替每顆指定部位。追蹤器名稱結尾＝耳機藍牙位址末五碼；"
        "動一下那顆耳機，看哪個在動也可以。小腿在 SlimeVR 裡叫「左腳踝／右腳踝」。\n"
        "⑧ SlimeVR「軀幹比例」和「配戴校正」各做一次，最後站直按「完整重置」。\n\n"
        "【之後每次】開啟 SlimeVR Server → FBT Pods 按「開始追蹤」→ 站直做完整重置。\n"
        "完整重置、快速重置（＝SlimeVR 的「左右偏擺重置」）、暫停：FBT Pods 的按鈕、手把按鍵和耳機柄"
        "都會轉交給 SlimeVR（手把按鍵要 SteamVR 開著）。\n"
        "右邊清單的「變更網路設定檔」可以按「忽略」，和耳機無關。\n\n"
        "FBT Pods 與 SlimeVR B.V. 無關，也未經其認可。")

    def _flow_slimevr(self, fl):
        fl.show(self.SLIMEVR_HELP, ["開啟 SlimeVR 安裝說明", "關閉"])
        if fl.wait(600) == "開啟 SlimeVR 安裝說明":
            import webbrowser
            webbrowser.open(self.SLIMEVR_SETUP_URL)

    def _flow_vmt(self, fl):
        fl.show("正在下載虛擬追蹤器驅動（VMT）…\n\n如果跳出 VMT 安裝程式，照預設一路按「下一步」完成即可。",
                progress=True)
        def confirm(name, sha):
            fl.show(f"無法自動驗證這個檔案的完整性：\n\n{name}\nSHA-256：{sha}\n\n"
                    "可以到 VMT 的 GitHub 下載頁核對這串數值（Assets 裡檔名旁邊的 sha256）。\n"
                    "一致才按「已核對，安裝」。",
                    ["已核對，安裝", "取消"])
            ok = fl.wait() == "已核對，安裝"
            fl.show("安裝中…", progress=True)
            return ok

        while True:
            try:
                winsetup.register_vmt(confirm)
                break
            except winsetup.VmtDownloadError as e:
                log.warning("VMT: %s", e)
                fl.show(f"自動安裝 VMT 失敗：\n{e}\n\n改用手動：\n\n"
                        "1. 按「開啟下載頁」，下載\n"
                        f"    {winsetup.VMT_FILE}（不是 Source code）\n"
                        "2. 存到「下載」資料夾，不用自己執行\n"
                        "3. 回來按「已下載，繼續」\n\n"
                        "程式會找到它並開啟安裝程式，照預設一路按「下一步」完成安裝即可；"
                        "裝好後程式會自動把它加進 SteamVR，不用再開 VMT Manager 按 Install。",
                        ["開啟下載頁", "已下載，繼續", "取消"])
                a = fl.wait()
                if a == "開啟下載頁":
                    import webbrowser
                    webbrowser.open(f"{winsetup.VMT_PAGE}/tag/{winsetup.VMT_TAG}")
                    fl.show("下載好之後按「已下載，繼續」。", ["已下載，繼續", "取消"])
                    a = fl.wait()
                if a != "已下載，繼續":
                    return
                fl.show("安裝中…", progress=True)
        fl.show("完成！如果 SteamVR 正在執行，請重新啟動它一次。", ["完成"])
        fl.wait()

    def _flow_pair(self, fl):
        from .pairing import LegIdentifier, discover_airpods, save_roles
        from .tracker import layout_for
        from .transport import open_backend
        self.stop()
        self.monitor.stop()          # the sensor test page holds the dongle otherwise
        if not self._dongle_free(fl):
            return
        cfg = C.load_config()
        start_now = False
        backend = open_backend(cfg)
        try:
            addrs = cfg.get("airpods") or []
            again = True
            known = []
            if addrs:
                fl.show(f"已經配對過 {len(addrs)} 副 AirPods（" + "、".join("…" + x[-5:] for x in addrs) + "）。\n\n"
                        "• 加入耳機：再配對一副（或把其中一副重新配對），原本的都保留\n"
                        "• 只分左右：不配對，只重新指定每一顆綁在哪裡\n"
                        "• 全部重新配對：清掉原本的，只留下這次勾選的",
                        ["加入耳機", "只分左右", "全部重新配對", "取消"])
                a = fl.wait()
                if a == "取消":
                    return
                again = a in ("全部重新配對", "加入耳機")
                known = list(addrs) if a == "加入耳機" else []
            if again:
                fl.show("讓 AirPods 進入配對模式：\n\n1. 耳機放在盒內、打開盒蓋\n2. 按住盒子背面按鈕，直到燈號閃白光\n\n然後按「搜尋」。",
                        ["搜尋", "取消"], "1 / 4")
                if fl.wait() != "搜尋":
                    return
                fl.show("搜尋中…（約 12 秒）", progress=True, step="1 / 4")
                found = discover_airpods(backend, 12)
                if not found:
                    fl.show("找不到 AirPods。確認燈號在閃白光後再試一次。", ["關閉"])
                    fl.wait()
                    return
                # an address already paired is marked: a bud that shows up under a new address is a
                # separate device (and can be used next to the pair it came from)
                choice = fl.choose("選擇你的 AirPods（兩副時兩個都選）",
                                   [f"{n or '(無名稱)'}  {a}" + ("  ← 已配對過的位址" if a in (cfg.get("airpods") or []) else "")
                                    for a, n in found])
                if not choice:
                    return
                addrs = [found[i][0] for i in choice]
                found_addrs = list(addrs)                # paired afresh below
                log.info("pairing: found %s, chosen %s, kept %s", [a for a, _ in found], addrs, known)
                # pair right away, while the AirPods are still in pairing mode (once they leave the
                # case they only accept hosts they already know)
                for n, addr in enumerate(addrs):
                    fl.show(f"配對中…（{n + 1} / {len(addrs)}）\n耳機請留在打開的盒子裡。", progress=True, step="1 / 4")
                    try:
                        pair = getattr(backend, "pair", None)
                        if pair is not None:     # bond only: the motion channel comes later
                            pair(addr)
                        else:
                            backend.open(addr).close()
                    except Exception as e:   # noqa: BLE001
                        log.warning("pairing %s: %s", addr, e)
                        fl.show(f"和 {addr} 配對失敗：{errtext(e)}\n\n"
                                "把耳機放回盒子、重新按住背面按鈕到閃白光，再試一次。\n"
                                "仍然失敗時，請到「進階 → 匯出診斷包」把檔案傳給開發者。",
                                ["關閉"])
                        fl.wait()
                        return
                addrs = known + [a for a in addrs if a not in known]
                lost = [a for a in (cfg.get("lost_pairing") or []) if a in addrs and a not in found_addrs]
                C.save_config({"airpods": addrs, "lost_pairing": lost})
            # Everything on the table first: the screen shows which bud of each pair is the live one
            # (it turns when the bud is turned), the user says where each goes, and only then are
            # they strapped on. (It used to ask "shake the left one" with four buds lying there.)
            fl.show("把要用的每一副 AirPods 都打開盒蓋，兩顆都拿出來放在桌上。先不要綁到腳上。\n\n"
                    "請先關掉 iPhone／iPad／Mac 的藍牙：AirPods 同時連著它們時不會傳送動作資料。\n\n"
                    "按「下一步」開始連線。",
                    ["下一步", "取消"], "2 / 4")
            if fl.wait() != "下一步":
                return
            expected = len(addrs)
            fl.show(f"連線中…已配對 {expected} 副，等它們都開始傳資料（最多約 1 分鐘）。\n"
                    "耳機是一副接一副連上的；沒連上的那幾副之後還會繼續嘗試。",
                    ["取消"], "2 / 4", progress=True)
            ident = LegIdentifier(backend, addrs, cfg)
            start_now = False
            try:
                # (the second-bud probe, when it is switched on, needs about half a minute more)
                # Wait for every paired pair, not for the first two: the pairs are paged one after the
                # other and one that does not answer at once takes ten seconds more, so the screen
                # came up with two, three or four buds from one try to the next.
                srcs = ident.wait_sources(max(2, expected), 75 if getattr(ident, "probe", False) else
                                          60 if expected > 1 else 30, stop=lambda: fl.cancelled,
                                          most=max(2, expected), grace=1.0)
                if fl.cancelled:
                    return
                lost = [l.addr for l in getattr(ident, "links", []) if getattr(l, "forgot", False)]
                if lost:
                    fl.show("這些 AirPods 已經不認得這台電腦（被重設過，或在盒子上重新按過配對）：\n"
                            + "\n".join("• " + a for a in lost) + "\n\n"
                            "請重新執行「配對」，選「加入耳機」：把那一副放回盒內、打開盒蓋，按住背面按鈕到白燈閃爍，"
                            "再搜尋並勾選它。其他已配對的耳機會保留。", ["關閉"])
                    fl.wait()
                    return
                if not srcs:
                    fl.show("沒有收到任何耳機的動作資料。\n\n"
                            "• 耳機要拿出盒子（在盒內不會傳資料）\n"
                            "• 關掉 iPhone／iPad／Mac 的藍牙，AirPods 連著它們時不會傳資料\n"
                            "• 仍然不行：重新執行「配對」", ["關閉"])
                    fl.wait()
                    return
                if len(srcs) < 2 and self.output_mode() != 1:
                    text = ("只收到一顆耳機的資料。\n\n一副 AirPods 只會送出其中一顆（主耳）的動作資料，"
                            "所以一副只能當一個追蹤點：兩個追蹤點需要兩副，每副各拿一顆。\n"
                            "每副沒用到的那一顆請留在盒子裡並蓋上盒蓋，否則主耳可能換到它身上，追蹤會停住。")
                    if getattr(ident, "probe", False):       # "bud_probe" in config.json (for research)
                        fl.show(text + "\n\n要做第二顆探測嗎（約 2 分鐘）？兩顆都拿在手上，照畫面提示搖。",
                                ["開始探測", "關閉"])
                        if fl.wait() == "開始探測":
                            self._relay_probe(fl, ident, backend)
                    else:
                        fl.show(text, ["關閉"])
                        fl.wait()
                    return
                def live_sources():
                    """The buds streaming now (a pair may still join, or drop out)."""
                    try:
                        live = [b["src"] for b in ident.snapshot()["buds"] if b.get("here", b.get("streaming", True))]
                    except Exception:   # noqa: BLE001 - no live picture (tests, or closing)
                        live = []
                    return live or list(srcs)
                text = ("每一副只有一顆會送出動作。拿起桌上的耳機轉一轉：畫面上跟著轉的就是那一顆。\n"
                        "1. 每一副【不會動】的那顆，收回盒內並蓋上盒蓋\n"
                        "2. 替每一顆會動的耳機，點選它要綁的位置\n"
                        "還沒連上的耳機會繼續嘗試，連上後卡片會自己出現。")
                self.flow_ident, self.flow_expected = ident, expected
                if self.output_mode() == 1:
                    # a SlimeVR Server does the body: which bud goes where is said there. Here the
                    # screen only shows which bud of each pair is the live one.
                    try:
                        fl.show("每一副只有一顆會送出動作。拿起桌上的耳機轉一轉：畫面上跟著轉的就是那一顆。\n"
                                "每一副【不會動】的那顆，收回盒內並蓋上盒蓋；會動的那顆等一下綁到身上。\n"
                                "哪一顆綁哪裡不用在這裡選：之後在 SlimeVR 的「追蹤器分配」指定。\n"
                                "還沒連上的耳機會繼續嘗試；不用等全部連上也可以按「下一步」。",
                                ["下一步", "取消"], "3 / 4", panel={"type": "buds", "expected": expected, "roles": False})
                        if fl.wait() != "下一步" or fl.cancelled:
                            return
                    finally:
                        self.flow_ident = None
                    fl.show("現在把每一副會動的那顆綁上，柄朝下、貼緊不要晃：\n\n"
                            "• 小腿：外側、腳踝上方\n• 大腿：外側、膝蓋上方\n• 腰：皮帶正後方或側邊\n\n"
                            "沒用到的耳機留在盒內、蓋著盒蓋。\n"
                            "綁好後先開啟 SlimeVR Server，再按「開始追蹤」。",
                            ["開始追蹤", "稍後再說"], "4 / 4")
                    start_now = fl.wait() == "開始追蹤"
                    return
                try:
                    note = ""
                    while True:
                        fl.show(text + note, ["下一步", "取消"], "3 / 4", panel={"type": "buds", "expected": expected})
                        if fl.wait() != "下一步" or fl.cancelled:
                            return
                        srcs = live_sources()
                        layout = layout_for(len(srcs))
                        pick = fl.choice if isinstance(fl.choice, dict) else {}
                        roles = {role: src for src, role in pick.items() if src in srcs and role in layout}
                        if len(srcs) >= 2 and len(roles) == len(layout) and len(set(roles.values())) == len(layout):
                            break
                        note = ("\n\n⚠ 至少要有 2 顆耳機在傳資料。" if len(srcs) < 2 else
                                "\n\n⚠ 每一顆耳機都要選一個位置，而且不能重複。")
                    snap = {b["src"]: b for b in ident.snapshot()["buds"]} if hasattr(ident, "snapshot") else {}
                finally:
                    self.flow_ident = None
                evidence = ident.evidence() if hasattr(ident, "evidence") else None
                save_roles(roles, evidence, forget=again)
                where = {"left_shin": "左小腿外側、腳踝上方", "right_shin": "右小腿外側、腳踝上方",
                         "left_thigh": "左大腿外側、膝蓋上方", "right_thigh": "右大腿外側、膝蓋上方",
                         "pelvis": "腰部，皮帶正後方或側邊"}
                lines = []
                for role in layout:
                    src = roles[role]
                    bud = {"left": "左耳", "right": "右耳"}.get((snap.get(src) or {}).get("primary"), "會動的那顆")
                    lines.append(f"• {ROLE_ZH[role]} ← …{src.split('#')[0][-5:]} 的{bud}：{where[role]}")
                fl.show("位置都記好了。現在把耳機綁上，柄朝下、貼緊不要晃：\n\n" + "\n".join(lines) + "\n\n"
                        "沒用到的耳機留在盒內、蓋著盒蓋。\n"
                        "綁好後按「開始追蹤並校正」：程式會開始追蹤，接著進入校正精靈。",
                        ["開始追蹤並校正", "稍後再說"], "4 / 4")
                start_now = fl.wait() == "開始追蹤並校正"
            finally:
                ident.close()
        finally:
            backend.close()
            if start_now:                    # (after the dongle is free again)
                threading.Thread(target=self._track_then_calibrate, args=(fl,), daemon=True).start()

    def _direct_probe(self, fl, ident, backend, links):
        """The buds mention a second address of their own: try to connect to it as if it were a
        device (first as it is, then with the pair's link key), next to the connection we have."""
        from .link import AirPodsLink
        found = []
        for l in links:
            for other in sorted(getattr(l, "other_addrs", ())):
                if fl.cancelled:
                    return found
                copied = False
                for attempt in ("plain", "with the pair's key"):
                    if attempt != "plain":
                        try:
                            copied = bool(backend.copy_key(l.addr, other)) if hasattr(backend, "copy_key") else False
                        except Exception as e:   # noqa: BLE001
                            log.info("direct probe: key copy failed: %s", e)
                        if not copied:
                            break
                    fl.show(f"嘗試直接連線第二顆（{other}）…\n兩顆都保持在盒外。約 30 秒。", ["取消"], progress=True)
                    log.info("direct probe: connecting to %s (%s), next to %s", other, attempt, l.addr)
                    got = []
                    rec = getattr(getattr(l, "record", None), "name", None)
                    extra = AirPodsLink(backend, other, lambda src, s: got.append(src), probe=True, sides=(0,),
                                        record_path=rec.replace(".jsonl", "-direct.jsonl") if rec else None)
                    extra.start()
                    t0 = time.time()
                    while time.time() - t0 < 28 and not got and not fl.cancelled:
                        time.sleep(0.25)
                    log.info("direct probe: %s (%s): status %s, %d samples", other, attempt, extra.status, len(got))
                    extra.stop()
                    extra.join(4)
                    if got:
                        found.append(other)
                        break
                if copied and other not in found and hasattr(backend, "forget_key"):
                    backend.forget_key(other)            # leave no key behind for an address that is no use
        return found

    def _relay_probe(self, fl, ident, backend=None):
        """Record what every sensor service sends while the user moves one bud, then the other."""
        links = [l for l in ident.links if getattr(l, "status", "") == "streaming"] or list(ident.links)
        for l in links:
            l.start_relay_probe()
        zh = {"left": "左耳", "right": "右耳"}
        prim = getattr(getattr(links[0], "buds", None), "primary", None) if links else None
        first = zh.get(prim, "其中一顆")
        second = zh.get({"left": "right", "right": "left"}.get(prim), "另一顆")
        phases = [("still", "兩顆都不要動", 8),
                  ("primary", f"只搖{first}，{second}不要動", 8),
                  ("still", "兩顆都不要動", 5),
                  ("secondary", f"只搖{second}，{first}不要動", 8),
                  ("still", "兩顆都不要動", 5),
                  ("both", "兩顆一起搖", 6)]
        for i, (key, text, seconds) in enumerate(phases):
            if fl.cancelled:
                break
            fl.show(f"探測 {i + 1} / {len(phases)}\n\n{text}（{seconds} 秒）", ["取消"], progress=True)
            time.sleep(1.5)                     # time to read it, before the phase counts
            for l in links:
                l.mark(f"{key}: {text}")
            time.sleep(seconds)
            beep(1320)
        for l in links:
            l.mark("end")
        found = []
        if backend is not None and not fl.cancelled:
            try:
                found = self._direct_probe(fl, ident, backend, links)
            except Exception as e:   # noqa: BLE001
                log.info("direct probe: %s", e)
        extra = f"\n\n直接連線第二顆成功：{'、'.join(found)}" if found else ""
        fl.show("探測完成。" + extra + "\n請到「進階 → 匯出診斷包」，把 zip 交給開發者分析。", ["關閉"])
        fl.wait()

    WIZARD = [
        ("站直、雙腳平行、看正前方。", "mount_begin", None, None, 3),
        ("膝蓋往前彎成滑雪姿勢：腳掌貼地、背打直、看前方。", "mount_finish", None, None, 3),
        ("再站直、看正前方。", "full_reset", None, None, 3),
    ]
    # the leg held out straight ahead: takes out the sideways lean a leg can show from the start
    AIM = {
        "left": ("左腳伸直，往【正前方】伸出去：腳跟點地或離地都可以，膝蓋打直，身體和頭保持朝正前方。"
                 "（站不穩可以扶著東西；不需要可按「略過」）"),
        "right": ("右腳伸直，往【正前方】伸出去：腳跟點地或離地都可以，膝蓋打直，身體和頭保持朝正前方。"
                  "（站不穩可以扶著東西；不需要可按「略過」）"),
    }
    # "The back of the controller": the face opposite the buttons. What counts is that the reset
    # later is done with the controller held the same way as here.
    _TOUCH = ("{h}手把貼在{l}正面：有搖桿和 {btn} 鍵的那一面朝外（你看得到它），"
              "另一面、也就是食指扳機那一側貼著{bone}，手把前端朝{to}。保持不動。\n"
              "之後做「貼合重置」時，要用同樣的方式貼。")
    TOUCH = {
        "left_shin": (_TOUCH.format(h="左", l="左小腿", bone="脛骨", to="腳掌", btn="X／Y"), "left_hand"),
        "right_shin": (_TOUCH.format(h="右", l="右小腿", bone="脛骨", to="腳掌", btn="A／B"), "right_hand"),
        "left_thigh": (_TOUCH.format(h="左", l="左大腿", bone="大腿", to="膝蓋", btn="X／Y"), "left_hand"),
        "right_thigh": (_TOUCH.format(h="右", l="右大腿", bone="大腿", to="膝蓋", btn="A／B"), "right_hand"),
    }
    RELAXED = [
        ("用你平常最自然的方式站著（不用刻意站直），看前方。", "relaxed_stand", None, None, 3),
        ("坐下，用平常的坐姿，看前方。（不會坐著玩可以按「略過」）",
         "relaxed_sit", None, None, 3),
    ]
    SKIPPABLE = ("relaxed_sit", "aim")

    @staticmethod
    def step_picture(action, hand, role):
        """Which drawing goes with a wizard step (web/poses.js)."""
        if action == "aim":
            return "aim_" + (role or "left").split("_")[0]
        if action == "register_touch":
            return ("touch_thigh_" if (role or "").endswith("thigh") else "touch_") + ("left" if hand == "left_hand" else "right")
        return {"mount_begin": "stand", "mount_finish": "ski", "full_reset": "stand",
                "relaxed_stand": "relaxed_stand", "relaxed_sit": "relaxed_sit"}.get(action)

    def wizard_steps(self, roles):
        """Steps for the trackers in use: mounting, each leg held out ahead, a touch registration per
        leg sensor, relaxed poses. (The touch registration comes after the aim step: it records the
        controller against the leg as it is then.)"""
        steps = list(self.WIZARD)
        for leg in ("left", "right"):
            if any(r.startswith(leg + "_") for r in roles):
                steps.append((self.AIM[leg], "aim", None, leg + "_shin", 3))
        for role in ("left_shin", "right_shin", "left_thigh", "right_thigh"):
            if role in roles:
                text, hand = self.TOUCH[role]
                steps.append((text, "register_touch", hand, role, 2))
        return steps + self.RELAXED

    def _flow_wizard(self, fl):
        app = self.app
        if app is None:
            fl.show("請先按「開始追蹤」（或試用模式），再執行校正精靈。", ["關閉"])
            fl.wait()
            return
        if getattr(app, "slime", None) is not None and not app.use_vmt:
            fl.show("目前身體交給 SlimeVR Server 計算：安裝角度校正和重置請在 SlimeVR 視窗裡做，"
                    "這裡的校正精靈只對 FBT Pods 自己的計算（VMT 輸出）有作用。\n\n"
                    "要用這個精靈：設定 →「身體演算與輸出」改成 FBT Pods 自己算，再重新開始追蹤。", ["關閉"])
            fl.wait()
            return
        def hook(hand):
            # the controller's long press is the screen's first button: 下一步 on a step, 重試 when a
            # step did not work (a fixed 下一步 there ended the wizard half done). Nothing while
            # the countdown runs: there is no button to press then.
            buttons = list(fl.buttons)
            if buttons and not fl.progress:
                fl.reply(buttons[0])
        app.wizard_hook = hook
        steps = self.wizard_steps(app.core.trackers)
        try:
            i = 0
            while i < len(steps):
                text, action, hand, role, count = steps[i]
                step = f"{i + 1} / {len(steps)}"
                skippable = action in self.SKIPPABLE
                pic = self.step_picture(action, hand, role)
                fl.show(text + "\n\n準備好後按「下一步」或【長按左手 Y／B】。嗶聲倒數結束前保持不動。",
                        ["下一步", "略過", "取消"] if skippable else ["下一步", "取消"], step, picture=pic)
                a = fl.wait()
                if a == "取消" or fl.cancelled:
                    return
                if a == "略過":
                    i += 1
                    continue
                fl.show("保持不動…", step=step, progress=True, picture=pic)
                for _ in range(count):
                    beep(660)
                    time.sleep(0.9)
                beep(1320, 200)
                app.last_message, app.last_ok = "", None
                app.request(action, hand or "left_hand", role)
                t0 = time.time()
                while app.last_ok is None and time.time() - t0 < 3:
                    time.sleep(0.05)
                if app.last_ok:
                    i += 1
                    continue
                # the step did not work (headset or controller not seen, knees not bent enough,
                # tracking stopped): say so and do it again, never carry on to "校正完成"
                if not self.running or self.app is not app:
                    fl.show("追蹤已停止，校正沒有完成。請重新「開始追蹤」後再執行校正精靈。", ["關閉"], step)
                    fl.wait()
                    return
                why = app.last_message or "沒有收到結果（耳機是否都在傳送資料？）"
                if action == "mount_finish":
                    i = 0                      # needs the standing pose again
                    why += "\n\n從「站直」重新開始。"
                fl.show(f"這一步沒有成功：\n{why}", ["重試", "略過", "取消"] if skippable else ["重試", "取消"],
                        step, picture=pic)
                a = fl.wait()
                if fl.cancelled:
                    return
                if a == "略過":
                    i += 1
                elif a not in ("重試", "下一步"):    # (a 下一步 meant for the step before: try again)
                    return
            # from here the long press is a full reset again, as the screen says: it used to only
            # close this screen
            app.wizard_hook = None
            fl.show("校正完成！\n\n之後每次開始追蹤，只要站直按「完整重置」（左手 Y 長按）。", ["完成"])
            fl.wait()
        finally:
            app.wizard_hook = None

    # ---------------------------------------------------------------- misc
    def open_data_folder(self):
        if hasattr(os, "startfile"):
            os.startfile(C.DATA)

    # files that never leave the data folder: the window's access token and the Bluetooth link keys
    # (with those, another computer could connect to the AirPods as this one)
    DIAG_NEVER = ("ui_token", "bumble_keys.json")

    def export_diagnostics(self):
        """One zip with what a bug report needs: logs, settings, calibration, versions and the
        current state. Returns its path (and shows it in Explorer)."""
        import json
        import platform
        import re
        import zipfile
        from importlib import metadata
        from .version import build_info
        out_dir = C.data_path("diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "FBTPods-diag-" + time.strftime("%Y%m%d-%H%M%S") + ".zip")
        pkgs = {}
        for name in ("numpy", "vqf", "bumble", "openvr", "python-osc", "onnxruntime", "pywebview", "pystray"):
            try:
                pkgs[name] = metadata.version(name)
            except Exception:   # noqa: BLE001
                pkgs[name] = None
        dstate, dev = self.dongle
        try:
            state = self.state()
            for k in ("logs", "pose", "flow"):
                state.pop(k, None)
        except Exception as e:   # noqa: BLE001
            state = {"error": str(e)}
        info = {
            "build": build_info(), "platform": platform.platform(), "python": sys.version,
            "frozen": C.FROZEN, "packages": pkgs,
            "dongle": {"state": dstate, "name": getattr(dev, "name", None),
                       "id": getattr(dev, "instance_id", None), "service": getattr(dev, "service", None),
                       "inf": getattr(dev, "inf", None)},
            "vmt_registered": self.vmt_ok, "vmt_pinned": winsetup.VMT_TAG, "steamvr_installed": self.steamvr_installed,
            "state": state, "settings": {s["key"]: s["value"] for s in self.settings()},
            # what each pair says about left and right (for working out the side labels: sides.py)
            "links": (self.app.status().get("links") if self.app is not None else None),
            "buds": C.load_config().get("buds"),
        }
        token = re.compile(r"([?&]t=)[\w-]+")           # older logs contain the window address

        def text_of(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                return token.sub(r"\1<removed>", f.read())

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("info.json", json.dumps(info, ensure_ascii=False, indent=2, default=str))
            z.writestr("log_recent.txt", "\n".join(f"{t} {lvl} {msg}" for _, t, lvl, msg in self.logs.since(0)))
            names = ["config.json", "calibration.json", "airpodsvr.log", "airpodsvr.log.1"]
            try:
                reports = sorted(f for f in os.listdir(C.DATA) if f.startswith("p0_report_") and f.endswith(".json"))
            except OSError:
                reports = []
            try:                                     # packets recorded while identifying buds
                rec = sorted(f for f in os.listdir(C.data_path("recordings")) if f.startswith("probe-"))
            except OSError:
                rec = []
            for name in names + reports[-3:] + [os.path.join("recordings", f) for f in rec[-6:]]:
                p = os.path.join(C.DATA, name)
                if name in self.DIAG_NEVER or not os.path.isfile(p):
                    continue
                try:
                    z.writestr(name.replace(os.sep, "/"), text_of(p))
                except OSError as e:
                    z.writestr(name + ".error.txt", str(e))
        log.info("diagnostics written: %s", path)
        if sys.platform == "win32":
            try:
                subprocess.Popen(["explorer", "/select,", path])
            except OSError:
                pass
        return path

    def run_diag(self):
        self.monitor.stop()
        self.stop()
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        exe = os.path.join(C.APP_DIR, "FBTPods.exe")
        bat = os.path.join(C.APP_DIR, "FBTPods.bat")
        console = os.path.join(C.APP_DIR, "FBTPods-console.exe")
        if C.FROZEN and os.path.exists(console):       # installed build: FBTPods.exe has no console
            subprocess.Popen([console, "diag"], cwd=C.APP_DIR, creationflags=flags)
        elif os.path.exists(exe):
            subprocess.Popen([exe, "diag"], cwd=C.APP_DIR)
        elif os.path.exists(bat):
            subprocess.Popen(["cmd", "/c", bat, "diag"], cwd=C.APP_DIR, creationflags=flags)
        else:
            subprocess.Popen([sys.executable, "-m", "airpodsvr", "diag"], cwd=C.ROOT, creationflags=flags)

    def notice(self):
        for p in (os.path.join(C.ROOT, "NOTICE"), os.path.join(C.ROOT, "licenses", "NOTICE")):
            try:
                with open(p, encoding="utf-8") as f:
                    return f.read()
            except OSError:
                pass
        return ""

