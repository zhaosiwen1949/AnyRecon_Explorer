"""PLY 点云读取与降采样。

只读取 XYZ + RGB；含 3DGS 属性（f_dc_*、opacity 等）的文件一律忽略多余属性，
按普通点云解读。不依赖 viser，可独立测试。
"""

from __future__ import annotations

import numpy as np
from plyfile import PlyData

# 无颜色时使用的统一灰色
DEFAULT_GRAY = (160, 160, 160)

# 颜色属性名的候选别名（按优先级）
_COLOR_ALIASES = [("red", "green", "blue"), ("r", "g", "b")]


def load_ply(path: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """读取 PLY 文件。

    返回 (points float32 (N,3), colors uint8 (N,3), has_color)。
    支持 ASCII / 二进制小端 / 二进制大端（plyfile 自动识别）。
    无颜色属性时 colors 为统一灰色，has_color=False。
    """
    plydata = PlyData.read(path)

    # 取 vertex 元素，缺失时回退到第一个元素
    element = None
    for el in plydata.elements:
        if el.name == "vertex":
            element = el
            break
    if element is None:
        if not plydata.elements:
            raise ValueError(f"PLY 文件没有任何元素: {path}")
        element = plydata.elements[0]

    names = {p.name for p in element.properties}
    if not {"x", "y", "z"} <= names:
        raise ValueError(f"PLY 缺少 x/y/z 属性: {path}")

    n = element.count
    points = np.column_stack(
        [element["x"], element["y"], element["z"]]
    ).astype(np.float32)
    if n == 0:
        return points.reshape(0, 3), np.zeros((0, 3), dtype=np.uint8), False

    # 颜色：优先 red/green/blue，兼容 r/g/b 别名；浮点 0–1 色缩放到 0–255
    for keys in _COLOR_ALIASES:
        if set(keys) <= names:
            raw = np.column_stack([element[k] for k in keys])
            if np.issubdtype(raw.dtype, np.floating):
                raw = np.clip(raw, 0.0, 1.0) * 255.0
            colors = np.clip(raw, 0, 255).astype(np.uint8)
            return points, colors, True

    colors = np.full((n, 3), DEFAULT_GRAY, dtype=np.uint8)
    return points, colors, False


def downsample(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """点数超过 max_points 时无放回随机降采样，否则原样返回。"""
    n = len(points)
    if max_points <= 0 or n <= max_points:
        return points, colors
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.choice(n, max_points, replace=False)
    return points[idx], colors[idx]
