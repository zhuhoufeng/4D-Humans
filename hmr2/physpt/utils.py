# hmr2/physpt/utils.py    （地面估计、窗口缓存、打包/解包工具）
# -*- coding: utf-8 -*-
from collections import deque
from typing import Optional, Tuple

import numpy as np


def pack_q(transl: np.ndarray, global_orient_axisang: np.ndarray, body_pose_axisang: np.ndarray) -> np.ndarray:
    """
    把 HMR2.0/PHALP 的一帧 SMPL 参数打包为 75 维向量:
      [T(3), R(3), theta(23*3)]
    """
    t = np.asarray(transl, dtype=np.float32).reshape(-1)
    r = np.asarray(global_orient_axisang, dtype=np.float32).reshape(-1)
    th = np.asarray(body_pose_axisang, dtype=np.float32).reshape(-1)
    if t.size != 3 or r.size != 3 or th.size != 69:
        raise ValueError(f"pack_q 输入尺寸错误: T={t.size}, R={r.size}, theta={th.size} (期望 3/3/69)")
    return np.concatenate([t, r, th], axis=0).astype(np.float32)


class GroundPlaneEstimator:
    """
    简化的地面估计器：用“最近若干帧脚底最低点高度”的中位数，近似水平地面 z = 常数。
    如需倾斜地面/多平面，可自行扩展为 RANSAC 平面拟合。
    返回格式: [nx, ny, nz, d] with n=[0,0,1], d=-z_med  =>  n^T x + d = 0
    """
    def __init__(self, max_hist: int = 120):
        self.z_hist = deque(maxlen=max_hist)

    def update(self, feet_points_world: Optional[np.ndarray]):
        if feet_points_world is None or feet_points_world.size == 0:
            return
        pts = np.asarray(feet_points_world, dtype=np.float32).reshape(-1, 3)
        z_val = np.percentile(pts[:, 2], 5)  # 取较低的 5% 以对抗噪声/穿插
        self.z_hist.append(float(z_val))

    def get_plane(self) -> Optional[np.ndarray]:
        if not self.z_hist:
            return None
        z_med = float(np.median(self.z_hist))
        return np.array([0.0, 0.0, 1.0, -z_med], dtype=np.float32)


class WindowBuffer:
    """
    维护每条 track 的滑动窗口:
      - q_window: (T,75)
      - beta: (10,)
      - ground plane: 累积估计
    """
    def __init__(self, T: int = 16):
        self.T = T
        self._q = deque(maxlen=T)
        self._beta = None
        self._gp = GroundPlaneEstimator()

    def push(self, q_t: np.ndarray, beta: np.ndarray, feet_points_world: Optional[np.ndarray] = None):
        self._q.append(np.asarray(q_t, dtype=np.float32))
        if self._beta is None:
            self._beta = np.asarray(beta, dtype=np.float32).reshape(-1)
        self._gp.update(feet_points_world)

    def ready(self) -> bool:
        return len(self._q) == self.T and self._beta is not None

    def as_arrays(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        q = np.stack(self._q, axis=0) if self.ready() else None
        b = self._beta
        plane = self._gp.get_plane()
        return q, b, plane
