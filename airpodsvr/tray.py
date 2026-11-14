# SPDX-License-Identifier: Apache-2.0
"""Notification-area (tray) icon: shows that FBT Pods is running even when its window is closed,
and offers open / start-stop tracking / quit."""
from __future__ import annotations

import logging
import os
import threading

from .i18n import tr

log = logging.getLogger(__name__)

ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "icon.ico")


class Tray:
    def __init__(self, ctl, on_show, on_quit):
        self.ctl, self.on_show, self.on_quit = ctl, on_show, on_quit
        self.icon = None
        self.told = False

    def start(self):
        try:
            import pystray
            from PIL import Image
        except Exception as e:   # noqa: BLE001
            log.info("tray icon unavailable: %s", e)
            return False
        ctl = self.ctl

        def toggle(icon, item):
            threading.Thread(target=(ctl.stop if ctl.running else ctl.start), daemon=True).start()

        menu = pystray.Menu(
            pystray.MenuItem(lambda it: tr("開啟 FBT Pods"), lambda i, it: self.on_show(), default=True),
            pystray.MenuItem(lambda it: tr("停止追蹤" if ctl.running else "開始追蹤"), toggle,
                             enabled=lambda it: ctl.running or ctl.ready_to_run()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(lambda it: tr("結束 FBT Pods"), lambda i, it: self.on_quit()))
        try:
            self.icon = pystray.Icon("FBTPods", Image.open(ICON), "FBT Pods", menu)
            self.icon.run_detached()
        except Exception as e:   # noqa: BLE001
            log.warning("tray icon: %s", e)
            self.icon = None
            return False
        return True

    @property
    def active(self):
        return self.icon is not None

    def update(self):
        if self.icon is not None:
            title = tr("FBT Pods — 追蹤中") if self.ctl.running else "FBT Pods"
            if self.icon.title != title:
                self.icon.title = title

    def notify_hidden(self, why="tracking"):
        """Once per run: tell where the program went when its window was closed."""
        if self.icon is None or self.told:
            return
        self.told = True
        what = "仍在背景追蹤" if why == "tracking" else "的設定步驟還在進行"
        try:
            self.icon.notify(tr(f"FBT Pods {what}。點右下角的 FBT Pods 圖示可以叫回視窗或結束。"), "FBT Pods")
        except Exception as e:   # noqa: BLE001
            log.debug("notify: %s", e)

    def stop(self):
        if self.icon is not None:
            try:
                self.icon.stop()
            except Exception:   # noqa: BLE001
                pass
            self.icon = None
