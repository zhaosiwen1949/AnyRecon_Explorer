"""相机轨迹录制与 JSON 序列化。

固定频率采样 + 相机静止去重。不依赖 viser，可独立测试；
相机状态通过注入的 get_camera_state_fn 获取。
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np

# v2: 每帧位姿以 transform_matrix（OpenCV 格式 c2w 4x4）记录，参考 NeRF transforms.json
SCHEMA_VERSION = 2


@dataclass
class Frame:
    """单帧相机状态。t 为距录制开始的相对秒数，wxyz 为 w 在前的四元数，fov 为弧度。"""

    t: float
    position: list[float]
    wxyz: list[float]
    fov: float
    aspect: float
    look_at: list[float]


# get_camera_state_fn 返回的字典需包含 position/wxyz/fov/aspect/look_at
CameraStateFn = Callable[[], dict]


class TrajectoryRecorder:
    """后台线程按固定频率采样相机状态，相机静止时跳过重复帧。"""

    def __init__(
        self,
        sample_hz: float,
        get_camera_state: CameraStateFn,
        move_eps_pos: float = 1e-4,
        move_eps_rot: float = 1e-5,
    ):
        self.sample_hz = sample_hz
        self._get_camera_state = get_camera_state
        self._move_eps_pos = move_eps_pos
        self._move_eps_rot = move_eps_rot
        self.frames: list[Frame] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self.frames = []
        self._running = True
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[Frame]:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self.frames

    def _loop(self) -> None:
        interval = 1.0 / self.sample_hz
        i = 0
        while self._running:
            now = time.monotonic()
            try:
                state = self._get_camera_state()
            except Exception:
                state = None
            if state is not None:
                frame = Frame(
                    t=round(now - self._t0, 4),
                    position=[float(v) for v in state["position"]],
                    wxyz=[float(v) for v in state["wxyz"]],
                    fov=float(state["fov"]),
                    aspect=float(state["aspect"]),
                    look_at=[float(v) for v in state["look_at"]],
                )
                if self._moved(frame):
                    self.frames.append(frame)
            # 按绝对时间表计算下一拍，避免漂移累积
            i += 1
            next_tick = self._t0 + i * interval
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _moved(self, frame: Frame) -> bool:
        """首帧必录；否则与上一已录帧比较位置和朝向。"""
        if not self.frames:
            return True
        last = self.frames[-1]
        return not (
            np.allclose(frame.position, last.position, atol=self._move_eps_pos)
            and np.allclose(frame.wxyz, last.wxyz, atol=self._move_eps_rot)
        )


# ---------------------------------------------------------- 位姿矩阵转换
#
# viser 相机内部即 OpenCV 约定：wxyz 给出 R_world_camera（相机系基向量在世界系下
# 的表示，列为 +X 右 / +Y 下 / +Z 朝前），position 为相机中心。因此 OpenCV 格式
# 的 c2w 矩阵直接为 [[R, position], [0, 0, 0, 1]]，无需任何坐标轴翻转。


def quat_wxyz_to_matrix(wxyz) -> np.ndarray:
    """w 在前的四元数 → 3x3 旋转矩阵 R_world_camera。"""
    q = np.asarray(wxyz, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(R) -> list[float]:
    """3x3 旋转矩阵 → w 在前的四元数（w >= 0）。"""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q)) or 1.0
    q = q / n
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def c2w_matrix(position, wxyz) -> list[list[float]]:
    """由 position + wxyz 构造 OpenCV 格式 c2w 4x4 矩阵（按行的嵌套列表）。"""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = quat_wxyz_to_matrix(wxyz)
    M[:3, 3] = np.asarray(position, dtype=np.float64)
    return [[float(v) for v in row] for row in M]


def _frame_to_dict(f: Frame) -> dict:
    """单帧序列化：位姿用 transform_matrix（OpenCV c2w），保留 t/fov/aspect 供回放。"""
    return {
        "t": f.t,
        "fov": f.fov,
        "aspect": f.aspect,
        "transform_matrix": c2w_matrix(f.position, f.wxyz),
    }


def _frame_from_dict(fd: dict, version: int) -> Frame:
    if version >= 2 or "transform_matrix" in fd:
        M = np.asarray(fd["transform_matrix"], dtype=np.float64)
        R = M[:3, :3]
        position = [float(v) for v in M[:3, 3]]
        wxyz = matrix_to_quat_wxyz(R)
        # OpenCV 约定：+Z 为视线方向；以单位距离取注视点（仅用于回放结束后恢复轨道中心）
        forward = R[:, 2]
        look_at = [float(position[i] + forward[i]) for i in range(3)]
        return Frame(
            t=float(fd.get("t", 0.0)),
            position=position,
            wxyz=wxyz,
            fov=float(fd.get("fov", 1.0)),
            aspect=float(fd.get("aspect", 1.0)),
            look_at=look_at,
        )
    # 旧版 v1：直接含 position/wxyz/look_at 字段
    return Frame(**fd)


def to_dict(frames: list[Frame], point_cloud: str, sample_rate_hz: float) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "camera_model": "OPENCV",
        "point_cloud": point_cloud,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_rate_hz": sample_rate_hz,
        "frame_count": len(frames),
        "frames": [_frame_to_dict(f) for f in frames],
    }


def from_dict(d: dict) -> tuple[dict, list[Frame]]:
    """解析轨迹字典，返回 (元信息, 帧列表)。兼容 v1（position/wxyz）与 v2（transform_matrix）。"""
    version = int(d.get("version", 1))
    if version > SCHEMA_VERSION:
        raise ValueError(f"不支持的轨迹版本: {version}")
    frames = [_frame_from_dict(f, version) for f in d["frames"]]
    meta = {k: v for k, v in d.items() if k != "frames"}
    return meta, frames


def save_json(d: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_trajectories(directory: str) -> list[str]:
    """返回目录下所有轨迹 JSON 的文件名（按名称排序）。"""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(".json"))


def make_trajectory_filename(ply_path: str, when: datetime | None = None) -> str:
    """由点云文件名 + 时间戳生成轨迹文件名（只取 basename，去除路径分隔符）。"""
    base = os.path.splitext(os.path.basename(ply_path))[0]
    base = base.replace(os.sep, "_").replace("/", "_") or "trajectory"
    when = when or datetime.now()
    return f"{base}_{when.strftime('%Y%m%d_%H%M%S')}.json"
