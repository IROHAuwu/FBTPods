# SPDX-License-Identifier: Apache-2.0
"""Export a trained checkpoint to a single-step ONNX model (x, h, c) -> (y, h, c) for the runtime."""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import PoseNet  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from airpodsvr.features import IN_DIM  # noqa: E402


def export_kwargs(export):
    """dynamo=False keeps the classic exporter, which newer torch no longer picks by default. Older
    torch (2.2-2.4, still allowed by requirements.txt) has only that exporter and no such keyword:
    passing it there stops the export with a TypeError."""
    import inspect
    try:
        params = inspect.signature(export).parameters
    except (TypeError, ValueError):
        return {}
    return {"dynamo": False} if "dynamo" in params else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/pose.pt")
    ap.add_argument("--out", default="models/pose.onnx")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    net = PoseNet(ck["hidden"], ck["layers"])
    net.load_state_dict(ck["state"])
    net.eval()
    x = torch.zeros(1, 1, IN_DIM)
    h = torch.zeros(ck["layers"], 1, ck["hidden"])
    c = torch.zeros(ck["layers"], 1, ck["hidden"])
    torch.onnx.export(net, (x, h, c), a.out, input_names=["x", "h", "c"], output_names=["y", "h_out", "c_out"],
                      opset_version=17, **export_kwargs(torch.onnx.export))
    print("exported", a.out)


if __name__ == "__main__":
    main()
