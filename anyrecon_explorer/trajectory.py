"""相机轨迹录制与 JSON 序列化。

固定频率采样 + 相机静止去重。不依赖 viser，可独立测试；
相机状态通过注入的 get_camera_state_fn 获取。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable

import numpy as np

SCHEMA_VERSION = 1


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


def to_dict(frames: list[Frame], point_cloud: str, sample_rate_hz: float) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "point_cloud": point_cloud,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_rate_hz": sample_rate_hz,
        "frame_count": len(frames),
        "frames": [asdict(f) for f in frames],
    }


def from_dict(d: dict) -> tuple[dict, list[Frame]]:
    """解析轨迹字典，返回 (元信息, 帧列表)。version 升级时在此做迁移。"""
    version = d.get("version", 1)
    if version != SCHEMA_VERSION:
        raise ValueError(f"不支持的轨迹版本: {version}")
    frames = [Frame(**f) for f in d["frames"]]
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
