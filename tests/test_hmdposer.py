# SPDX-License-Identifier: Apache-2.0
"""HMD-Poser port: frames, forward kinematics, and (when PyTorch + the checkpoint are available)
equality with the reference network."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from airpodsvr import hmdposer as H          # noqa: E402
from airpodsvr.math3d import axis_angle      # noqa: E402

CKPT = os.environ.get("HMDPOSER_CKPT", "")


def test_frames_roundtrip_and_upright():
    R = axis_angle([0.3, 1, 0.2], 1.1)
    assert np.allclose(H.from_smpl(H.to_smpl(R)), R)
    up_smpl = H.to_smpl(np.eye(3)) @ np.array([0, 1, 0])     # SMPL Y (up) of an upright VR frame
    assert np.allclose(up_smpl, [0, 0, 1])                    # = AMASS +Z
    fwd = H.to_smpl(np.eye(3)) @ np.array([0, 0, 1])          # SMPL Z (facing) -> AMASS
    assert np.allclose(H.M.T @ fwd, [0, 0, -1])               # = VR forward (-Z)


def test_standing_skeleton_from_identity_pose():
    """All local rotations identity, root upright: legs straight below the hip, head on top."""
    Rl = np.repeat(np.eye(3)[None], 22, axis=0)
    Rl[0] = H.to_smpl(np.eye(3))
    Rg, pos = H.fk(Rl, H.REST)
    p = pos @ H.M
    assert p[H.HEAD][1] - p[H.PELVIS][1] > 0.5
    assert p[H.LANKLE][1] < p[H.LKNEE][1] < p[H.PELVIS][1]
    assert p[H.LKNEE][0] < 0 < p[H.RKNEE][0]                  # left knee on the left (VR -X)
    assert np.allclose(H.from_smpl(Rg[H.LKNEE]), np.eye(3))


def test_sixd_roundtrip():
    R = axis_angle([1, 2, 3], 0.7)
    assert np.allclose(H.sixd_to_mat(np.array(H.sixd(R))), R)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="set HMDPOSER_CKPT to the downloaded checkpoint")
def test_checkpoint_matches_torch_and_stands():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, os.path.dirname(os.path.dirname(CKPT)))
    from model.network import HMD_imu_HME_Universe
    net = HMD_imu_HME_Universe(135, 3, 256, 0.05, 8, 2)
    net.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"])
    net.eval()
    x = torch.randn(1, 10, 135) * 0.5
    with torch.no_grad():
        ref = H.sixd_to_mat(net(x)[0][0].numpy().reshape(10, 22, 6))
    mine = H.HmdPoserNet(H.load_checkpoint(CKPT))
    assert max(np.abs(mine.step(x[0, t].numpy()) - ref[t]).max() for t in range(10)) < 1e-3

    from airpodsvr.body import Body
    m = H.HmdPoser(CKPT, threaded=False)
    m.set_body(Body(1.70))
    hands = {"left_hand": (np.eye(3), np.array([-0.25, 0.95, -0.15])),
             "right_hand": (np.eye(3), np.array([0.25, 0.95, -0.15]))}
    for _ in range(90):
        pred = m.step(1 / 60, np.eye(3), np.array([0, 1.60, 0]), hands, [np.eye(3), np.eye(3)],
                      acc={"left": np.zeros(3), "right": np.zeros(3)})
    assert 0.85 < pred["pelvis_p"][1] < 1.1
    assert all(-0.05 < a[1] < 0.15 for a in pred["ankle_p"])
    lifted = axis_angle([1, 0, 0], math.radians(-60))           # left shin raised forward
    for _ in range(90):
        pred = m.step(1 / 60, np.eye(3), np.array([0, 1.60, 0]), hands, [lifted, np.eye(3)],
                      acc={"left": np.zeros(3), "right": np.zeros(3)})
    assert pred["ankle_p"][0][1] > pred["ankle_p"][1][1] + 0.15
