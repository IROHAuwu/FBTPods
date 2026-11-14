# SPDX-License-Identifier: Apache-2.0
"""Small rotation helpers (numpy). Quaternions are (w, x, y, z).

World frame everywhere = SteamVR / OpenVR: right-handed, +Y up, -Z forward, metres.
Segment frames (shin, foot, hip) use the same convention: +X right, +Y up (along the bone,
towards the proximal joint), -Z forward.
"""
from __future__ import annotations

import math

import numpy as np

UP = np.array([0.0, 1.0, 0.0])

# VQF earth frame (ENU-like, +Z up) -> SteamVR world (+Y up, -Z forward).
# x_w = x_e, y_w = z_e, z_w = -y_e
E2W = np.array([[1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0]])


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(m) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def rotvec(m) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle)."""
    m = np.asarray(m, dtype=float)
    c = max(-1.0, min(1.0, (np.trace(m) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    s = np.linalg.norm(v)
    if s < 1e-9:                       # ~180 degrees
        w, vec = np.linalg.eigh((m + np.eye(3)) / 2)
        return vec[:, np.argmax(w)] * ang
    return v / s * ang


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(rotvec(a.T @ b)))


def twist_y(m) -> float:
    """Signed rotation angle of m around world +Y (swing-twist decomposition)."""
    q = mat_to_quat(m)
    return 2.0 * math.atan2(q[2], q[0])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def heading_of_vec(v) -> float:
    """Heading of a horizontal direction; 0 = -Z (forward), positive = turning left (CCW from above)."""
    return math.atan2(-v[0], -v[2])


def heading_of(m) -> float:
    """Heading of a frame: the yaw that best aligns rot_y(heading) with m (twist about Y)."""
    return wrap(twist_y(m))


def tilt_angle(m) -> float:
    """Angle between the frame's +Y axis and world up."""
    y = np.asarray(m)[:, 1]
    return math.acos(max(-1.0, min(1.0, float(y[1]))))


def orthonormal(m) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=float))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def frame_from_up_right(up, right) -> np.ndarray:
    """Columns X(right), Y(up), Z(back) from a primary up axis and an approximate right axis."""
    y = np.asarray(up, dtype=float)
    y = y / np.linalg.norm(y)
    x = np.asarray(right, dtype=float) - y * float(np.dot(right, y))
    n = np.linalg.norm(x)
    if n < 1e-6:
        x = np.cross(y, [0.0, 0.0, 1.0])
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(y, [1.0, 0.0, 0.0])
        n = np.linalg.norm(x)
    x = x / n
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def rot6d(m) -> np.ndarray:
    """First two columns, flattened (6D rotation representation)."""
    m = np.asarray(m)
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1)


def from_rot6d(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    a, b = v[..., :3], v[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - x * np.sum(x * b, axis=-1, keepdims=True)
    y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def openvr_matrix_to_np(m34) -> tuple[np.ndarray, np.ndarray]:
    """HmdMatrix34_t -> (R 3x3, t 3)."""
    a = np.array([[m34[i][j] for j in range(4)] for i in range(3)], dtype=float)
    return orthonormal(a[:, :3]), a[:, 3].copy()
