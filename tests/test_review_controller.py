"""Controller and tray fixes from the review: one session at a time, a stop that waits for the
dongle, the sensor page following a new session, the wizard driven by the controller, empty
choices, and the pose model only going into a running session."""
import os
import threading
import time
import types

import pytest

from airpodsvr import config as C


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A private user-data folder, so tests never touch (or depend on) the real one."""
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    monkeypatch.setattr(C, "CALIB", str(tmp_path / "calibration.json"))
    return tmp_path


def wait(cond, seconds=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def answer_by_text(fl, rules):
    """Answers each waiting screen by what it says: rules are (words on the screen, button) or
    (words, button, choice). By text rather than in order: an answer is never used up by a screen
    that was only passing by. (Event.wait, not time.sleep: some tests make time.sleep return at once.)"""
    shown = []

    def auto():
        unknown = (None, 0.0)
        while not fl.done:
            if fl.buttons and not fl.progress and not fl.ev.is_set():
                text = fl.text
                rule = next((r for r in rules if r[0] in text), None)
                if rule is not None:
                    shown.append(text)
                    fl.reply(rule[1], rule[2] if len(rule) > 2 else None)
                elif unknown[0] != text:
                    unknown = (text, time.monotonic())
                elif time.monotonic() - unknown[1] > 0.3:
                    # a screen no rule expects, and it stays: leave it, so the test fails rather
                    # than waits for ever (a screen merely passing by is never answered this way)
                    shown.append(text)
                    fl.reply("取消" if "取消" in fl.buttons else fl.buttons[-1])
            threading.Event().wait(0.005)
    threading.Thread(target=auto, daemon=True).start()
    return shown


def fake_app_class(build_s=0.0, shutdown_s=0.0, join_s=5.0):
    """Stands in for App: one dongle that only one session can hold, a build that takes a while
    (calibration, the pose model) and a shutdown that takes a while (links saying goodbye) and may
    outlast App.stop's own join."""
    dongle = {"holder": None, "log": []}

    class FakeApp:
        made = []

        def __init__(self, demo=False):
            time.sleep(build_s)
            self.demo = demo
            self.core = types.SimpleNamespace(tap=None)
            self.links = []
            self.stop_flag = threading.Event()
            self.release = threading.Event()
            self.release.set()
            self.thread = None
            self.error = None
            FakeApp.made.append(self)

        def start(self):
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

        def run(self):
            if dongle["holder"] is not None:
                dongle["log"].append("open REFUSED")
                self.error = "無法開啟藍芽"
                return
            dongle["holder"] = self
            dongle["log"].append("open")
            self.stop_flag.wait()
            time.sleep(shutdown_s)
            self.release.wait()
            dongle["holder"] = None
            dongle["log"].append("close")

        def stop(self):
            self.stop_flag.set()
            if self.thread is not None:
                self.thread.join(join_s)
    FakeApp.dongle = dongle
    return FakeApp


def controller(monkeypatch, app_cls=None):
    from airpodsvr import app as appmod, controller as ctl
    from airpodsvr.monitor import SensorMonitor
    if app_cls is not None:
        monkeypatch.setattr(appmod, "App", app_cls)
    c = ctl.Controller.__new__(ctl.Controller)
    c.app, c.flow = None, None
    c.monitor = SensorMonitor(c)
    return c


# ------------------------------------------------------------------ G6-5: one session at a time
def test_two_start_requests_at_once_make_one_session(data, monkeypatch):
    App = fake_app_class(build_s=0.3)
    c = controller(monkeypatch, App)
    ts = [threading.Thread(target=c.start) for _ in range(2)]       # a double click, tray + window
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert len(App.made) == 1 and c.app is App.made[0] and c.running
    assert App.dongle["log"] == ["open"]
    c.stop()
    assert not c.running and App.dongle["holder"] is None            # nothing left holding the dongle


# ------------------------------------------------------------------ G6-4: stop waits for the dongle
def test_stop_reports_running_until_the_session_let_go_of_the_dongle(data, monkeypatch):
    # the shutdown (links saying goodbye) outlasts App.stop's own join
    App = fake_app_class(shutdown_s=0.5, join_s=0.05)
    c = controller(monkeypatch, App)
    c.start()
    assert wait(lambda: App.dongle["holder"] is not None, 2)
    stopper = threading.Thread(target=c.stop)
    stopper.start()
    time.sleep(0.15)
    assert c.running and c.busy() == "tracking"     # the button must not say 開始追蹤 yet
    starter = threading.Thread(target=c.start)      # ...and a start meanwhile waits for the stop
    starter.start()
    stopper.join(5)
    starter.join(5)
    assert App.dongle["log"] == ["open", "close", "open"]             # never two at once
    assert len(App.made) == 2 and c.app is App.made[1] and App.made[1].error is None
    c.stop()


def test_a_session_that_hangs_in_its_shutdown_still_counts_as_running(data, monkeypatch):
    from airpodsvr import controller as ctl, transport
    App = fake_app_class(join_s=0.05)
    c = controller(monkeypatch, App)
    c.STOP_WAIT = 0.3
    c.start()
    assert wait(lambda: App.dongle["holder"] is not None, 2)
    hung = App.made[0]
    hung.release.clear()                             # it cannot let go of the adapter
    t0 = time.monotonic()
    c.stop()
    assert time.monotonic() - t0 < 3                 # bounded
    assert c.app is None and c.running               # but still holding the dongle: not "stopped"
    c.start()
    assert len(App.made) == 1                        # no second session on a busy dongle
    # pairing says so instead of failing to open the adapter (or, for 準備藍芽, blaming the driver)
    opened = []
    monkeypatch.setattr(transport, "open_backend", lambda cfg: opened.append(cfg))
    fl = ctl.Flow("pair", "t")
    shown = answer_by_text(fl, [("", "關閉")])
    c._flow_pair(fl)
    fl.done = True
    assert opened == [] and "還在結束中" in shown[-1]
    hung.release.set()                               # it ends after all
    assert wait(lambda: not c.running, 3)
    c.start()
    assert len(App.made) == 2 and c.running
    c.stop()
    assert not c.running and c._stopping is None


def test_a_hung_session_never_reads_as_stopped_while_stop_puts_it_aside(data, monkeypatch):
    from airpodsvr import app as appmod, controller as ctl
    from airpodsvr.monitor import SensorMonitor
    App = fake_app_class(join_s=0.05)
    monkeypatch.setattr(appmod, "App", App)
    seen = []

    class Watched(ctl.Controller):
        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "app" and value is None:
                seen.append(self.running)            # what a window poll at that moment reads
    c = Watched.__new__(Watched)
    c.app, c.flow = None, None
    c.monitor = SensorMonitor(c)
    c.STOP_WAIT = 0.3
    c.start()
    assert wait(lambda: App.dongle["holder"] is not None, 2)
    hung = App.made[0]
    hung.release.clear()
    seen.clear()
    c.stop()
    # never "not running" in between: the window, the tray or the sensor page would open the
    # adapter the session still holds
    assert seen == [True] and c.running
    hung.release.set()
    assert wait(lambda: not c.running, 3)
    c.start()
    assert len(App.made) == 2 and c._stopping is None    # the ended one is not kept any longer
    c.stop()


# ------------------------------------------------------------------ G7-15: not during a dongle step
def test_tracking_does_not_start_while_a_setup_step_has_the_dongle(data, monkeypatch):
    from airpodsvr import controller as ctl
    from airpodsvr.tray import Tray
    App = fake_app_class()
    c = controller(monkeypatch, App)
    c.ready_to_run = lambda: True
    tray = Tray(c, on_show=lambda: None, on_quit=lambda: None)
    for kind in ("pair", "dongle", "restore"):
        c.flow = ctl.Flow(kind, "t")
        assert c.setup_holds_dongle() and not tray.can_toggle()
        c.start()                                    # e.g. from the tray, or SteamVR's autostart
        assert App.made == []
        c.flow.done = True
        assert tray.can_toggle()
    c.flow = ctl.Flow("slimevr", "t")                # a help text holds nothing
    assert tray.can_toggle()
    c.start()
    assert len(App.made) == 1 and c.running
    c.flow = ctl.Flow("pair", "t")
    assert tray.can_toggle()                         # stopping is always offered
    c.flow.done = True
    c.stop()
    c.ready_to_run = lambda: False
    assert not tray.can_toggle()


def test_tray_menu_is_built_again_when_its_start_stop_item_changes(data, monkeypatch):
    # Windows shows the menu as it was built: the item's label and whether it can be pressed are not
    # looked at again when it opens, so a greyed-out item during pairing needs a rebuild
    from airpodsvr import controller as ctl
    from airpodsvr.tray import Tray
    App = fake_app_class()
    c = controller(monkeypatch, App)
    c.ready_to_run = lambda: True
    tray = Tray(c, on_show=lambda: None, on_quit=lambda: None)
    builds = []
    tray.icon = types.SimpleNamespace(title="", update_menu=lambda: builds.append((c.running, tray.can_toggle())))
    tray.update()
    tray.update()
    assert builds == [(False, True)]                 # once, not on every update
    c.flow = ctl.Flow("pair", "t")                   # pairing has the adapter: not pressable
    tray.update()
    c.flow.done = True
    tray.update()
    c.start()
    tray.update()                                    # tracking: the item says 停止追蹤 now
    c.stop()
    tray.update()
    assert builds == [(False, True), (False, False), (False, True), (True, True), (False, True)]
    tray.icon = None


# ------------------------------------------------------------------ G6-11: sensor page and a new session
def test_sensor_page_follows_a_new_session_after_one_ended_by_itself(data, monkeypatch):
    App = fake_app_class()
    c = controller(monkeypatch, App)
    c.start()
    assert c.monitor.start("live") == "tap"          # tracking runs: the page taps it
    first = App.made[0]
    first.stop_flag.set()                            # SteamVR closed: the session ends on its own
    first.thread.join(2)
    assert not c.running and c.monitor.mode == "tap"
    c.start()                                        # tracking again
    second = App.made[1]
    assert c.monitor.mode == "tap" and c.monitor.app is second
    assert second.core.tap == c.monitor.feed and first.core.tap is None    # the dead one is let go
    c.stop()
    assert c.monitor.mode is None and second.core.tap is None


def test_sensor_page_stays_on_the_session_until_it_has_ended(data, monkeypatch):
    App = fake_app_class(shutdown_s=0.4, join_s=0.05)
    c = controller(monkeypatch, App)
    c.start()
    c.monitor.start("live")
    stopper = threading.Thread(target=c.stop)
    stopper.start()
    time.sleep(0.15)
    # its start button stays hidden while the dongle is still in use (it would open it twice)
    assert c.monitor.mode == "tap"
    stopper.join(5)
    assert c.monitor.mode is None and App.dongle["holder"] is None


# ------------------------------------------------------------------ G6-9 / G7-13: empty choice
def test_choose_with_nothing_ticked_chooses_nothing():
    from airpodsvr.controller import Flow

    def ask(value, choice, items=("a", "b")):
        fl = Flow("t", "t")
        out = []
        t = threading.Thread(target=lambda: out.append(fl.choose("?", items)))
        t.start()
        assert wait(lambda: fl.items is not None, 2)
        fl.reply(value, choice)
        t.join(2)
        return out[0]
    assert ask("確定", []) is None                   # all unticked: none of these
    assert ask("確定", None) == [0]                  # no list sent at all: the first, as before
    assert ask("確定", [1]) == [1] and ask("確定", [1, 0]) == [1, 0]
    assert ask("確定", [1, 1, 7, -1, True, "0"]) == [1]       # only real items, each once
    assert ask("確定", [5]) is None
    assert ask("取消", [0]) is None


def test_pairing_with_nothing_ticked_pairs_nothing(data, monkeypatch):
    from airpodsvr import controller as ctl, pairing, transport
    C.save_config({"airpods": ["AA:AA:AA:AA:AA:AA"], "output": 0})
    paired = []
    monkeypatch.setattr(pairing, "discover_airpods", lambda backend, seconds: [
        ("11:22:33:44:55:66", "Someone else's AirPods"), ("AA:AA:AA:AA:AA:AA", "My AirPods")])
    monkeypatch.setattr(transport, "open_backend", lambda cfg: types.SimpleNamespace(
        pair=paired.append, close=lambda: None))
    c = controller(monkeypatch)
    fl = ctl.Flow("pair", "t")
    shown = answer_by_text(fl, [("已經配對過", "全部重新配對"), ("進入配對模式", "搜尋"),
                                ("選擇你的 AirPods", "確定", [])])       # everything unticked
    c._flow_pair(fl)
    fl.done = True
    assert any("選擇你的 AirPods" in t for t in shown)
    assert paired == [] and C.load_config()["airpods"] == ["AA:AA:AA:AA:AA:AA"]


# ------------------------------------------------------------------ G6-3: wizard by controller
class WizardApp:
    def __init__(self, results):
        self.core = types.SimpleNamespace(trackers={"left_shin": 1, "right_shin": 1})
        self.wizard_hook = None
        self.last_message, self.last_ok = "", None
        self.thread = types.SimpleNamespace(is_alive=lambda: True)
        self.results = iter(results)
        self.done = []

    def request(self, action, hand, role):
        self.done.append(action)
        self.last_ok = next(self.results)
        self.last_message = "" if self.last_ok else "看不到頭顯，無法執行"


def test_wizard_driven_by_the_controller_retries_a_failed_step_and_finishes(data, monkeypatch):
    from airpodsvr import controller as ctl
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    c = controller(monkeypatch)
    steps = c.wizard_steps({"left_shin": 1, "right_shin": 1})
    # the standing step fails once (head not seen), everything else works
    app = c.app = WizardApp([True, True, False] + [True] * len(steps))
    fl = ctl.Flow("wizard", "t")
    shown, hook_at_end = [], []

    def press():                                     # every screen answered with a long press of Y
        while not fl.done:
            if fl.buttons and not fl.ev.is_set():
                shown.append(fl.text)
                if "校正完成" in fl.text:
                    hook_at_end.append(app.wizard_hook)
                    fl.reply("完成")
                    return
                hook = app.wizard_hook
                if hook is not None:
                    hook("left_hand")
            threading.Event().wait(0.002)
    threading.Thread(target=press, daemon=True).start()
    c._flow_wizard(fl)
    fl.done = True
    assert any("沒有成功" in t for t in shown) and "校正完成" in shown[-1]
    assert app.done.count("full_reset") == 2                         # retried, not abandoned
    assert app.done[-1] == "relaxed_sit"
    # on the last screen the long press is a real full reset again
    assert hook_at_end == [None] and app.wizard_hook is None


def test_wizard_ignores_the_long_press_during_the_countdown(data, monkeypatch):
    from airpodsvr import controller as ctl
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    c = controller(monkeypatch)
    app = c.app = WizardApp([True] * 20)
    fl = ctl.Flow("wizard", "t")
    during = []

    def beep(*a):
        if not during and fl.progress:               # "保持不動…": nothing to answer
            app.wizard_hook("left_hand")
            during.append((fl.ev.is_set(), fl.answer))
    monkeypatch.setattr(ctl, "beep", beep)
    answer_by_text(fl, [("膝蓋往前彎", "取消"), ("站直", "下一步")])
    c._flow_wizard(fl)
    fl.done = True
    assert during == [(False, None)]


def test_wizard_takes_a_late_next_on_the_failure_screen_as_retry(data, monkeypatch):
    from airpodsvr import controller as ctl
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    c = controller(monkeypatch)
    app = c.app = WizardApp([False, True])
    fl = ctl.Flow("wizard", "t")
    # step 1 fails; its failure screen gets a 下一步 (meant for the step); step 2: cancel
    shown = answer_by_text(fl, [("沒有成功", "下一步"), ("膝蓋往前彎", "取消"), ("站直", "下一步")])
    c._flow_wizard(fl)
    fl.done = True
    assert app.done == ["mount_begin", "mount_begin"] and any("沒有成功" in t for t in shown)
    assert "膝蓋往前彎" in shown[-1]                  # it went on to step 2, not out of the wizard


# ------------------------------------------------------------------ G6-M2: pose model
class FakeResponse:
    def __init__(self, size):
        self.left = size
        self.headers = {"Content-Length": str(size)}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        n = min(n, self.left)
        self.left -= n
        return b"\0" * n


def _download(monkeypatch, c, use_model):
    import urllib.request
    from airpodsvr import controller as ctl, hmdposer
    C.save_config({"use_pose_model": use_model})
    # (only the private data folder counts: a model downloaded into this checkout would otherwise
    # be "already downloaded" and the flow would offer to delete it)
    monkeypatch.setattr(C, "APP_DIR", C.DATA)
    monkeypatch.setattr(C, "ROOT", C.DATA)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResponse(3000))
    sd = {"pose_est.2.weight": 1, **{f"w{i}": i for i in range(250)}}
    monkeypatch.setattr(hmdposer, "load_checkpoint", lambda path: sd)
    fl = ctl.Flow("model", "t")
    shown = answer_by_text(fl, [("要下載嗎", "下載"), ("完成！", "完成"), ("", "關閉")])
    c._flow_model(fl)
    fl.done = True
    os.remove(os.path.join(C.DATA, "models", hmdposer.MODEL_FILE))     # so the next one downloads again
    return shown


def test_downloaded_model_goes_into_a_running_session_only(data, monkeypatch):
    c = controller(monkeypatch)
    loaded = []
    alive = {"v": False}
    c.app = types.SimpleNamespace(cfg={}, core=types.SimpleNamespace(load_model=loaded.append),
                                  thread=types.SimpleNamespace(is_alive=lambda: alive["v"]))
    shown = _download(monkeypatch, c, True)           # the session ended by itself (SteamVR closed)
    assert loaded == [] and "已啟用" in shown[-1]
    alive["v"] = True
    shown = _download(monkeypatch, c, False)          # running, but the setting is off
    assert loaded == [] and "目前是關閉的" in shown[-1]
    _download(monkeypatch, c, True)
    assert len(loaded) == 1


def test_settings_are_not_applied_to_a_session_that_has_ended(data, monkeypatch):
    c = controller(monkeypatch)
    applied = []
    alive = {"v": False}
    c.app = types.SimpleNamespace(apply_settings=applied.append,
                                  thread=types.SimpleNamespace(is_alive=lambda: alive["v"]))
    c.update_settings({"use_pose_model": True})       # would load a model nobody closes
    assert applied == []
    alive["v"] = True
    c.update_settings({"auto_yaw": False})
    assert len(applied) == 1


def test_cancel_during_the_model_download_is_honoured(data, monkeypatch):
    """Each 1 MB chunk used to show the dialog again: a Cancel drawn from an earlier poll named an
    earlier showing, answer_flow dropped it, and the 100 MB download ran to the end."""
    import urllib.request
    from airpodsvr import controller as ctl, hmdposer
    c = controller(monkeypatch)
    monkeypatch.setattr(C, "APP_DIR", C.DATA)
    monkeypatch.setattr(C, "ROOT", C.DATA)
    fl = c.flow = ctl.Flow("model", "t")
    seen = {"reads": 0}

    class Slow(FakeResponse):
        def read(self, n):
            seen["reads"] += 1
            if seen["reads"] == 2:
                seen["snap"] = fl.to_dict()["shown"]       # what the window drew at its last poll
            if seen["reads"] == 4:
                c.answer_flow(fl.id, "取消", None, shown=seen["snap"])
            return super().read(n)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Slow(10 << 20))
    monkeypatch.setattr(hmdposer, "load_checkpoint", lambda path: pytest.fail("the download was not cancelled"))
    answer_by_text(fl, [("要下載嗎", "下載"), ("", "關閉")])
    c._flow_model(fl)
    fl.done = True
    assert fl.cancelled and seen["reads"] < 10
    assert not os.path.exists(os.path.join(C.DATA, "models", hmdposer.MODEL_FILE))
