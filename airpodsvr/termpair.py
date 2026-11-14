# SPDX-License-Identifier: Apache-2.0
"""Pairing from the terminal: for `pair`, and for `diag` (the hardware check) when nothing is paired
yet. Everything else of the first-time setup is done in the window."""
from __future__ import annotations

from . import config as C


def step_pair(cfg, backend):
    print("\n=== 配對 AirPods ===")
    if cfg.get("airpods"):
        print("已設定：", ", ".join(cfg["airpods"]))
        if input("重新配對？(y/N) ").strip().lower() != "y":
            return cfg["airpods"]
    if hasattr(backend, "discover"):
        print("打開 AirPods 盒蓋，按住背面按鈕直到燈號閃白光，然後按 Enter 開始搜尋。")
        input()
        found = backend.discover(12)
        found = [f for f in found if "airpods" in f[1].lower()] or found
        for i, (addr, name) in enumerate(found):
            print(f"  [{i}] {name or '(無名稱)'}  {addr}")
        if not found:
            raise SystemExit("找不到裝置。確認耳機在配對模式、USB藍芽已在 FBT Pods 視窗按「準備藍芽」設定好。")
        pick = input("輸入編號（兩副用逗號分隔，例如 0,1）：").strip() or "0"
        addrs = [found[int(x)][0] for x in pick.split(",")]
        for addr in addrs:          # pair now, while still in pairing mode
            print(f"  配對 {addr}…（耳機留在打開的盒子裡）")
            (getattr(backend, "pair", None) or (lambda a: backend.open(a).close()))(addr)
    else:
        paired = backend.list_paired()
        for i, (addr, name) in enumerate(paired):
            print(f"  [{i}] {name}  {addr}")
        if not paired:
            raise SystemExit("沒有已配對的 AirPods，請先用 bluetoothctl 配對。")
        pick = input("輸入編號（兩副用逗號分隔）：").strip() or "0"
        addrs = [paired[int(x)][0] for x in pick.split(",")]
    C.save_config({"airpods": addrs})
    return addrs
