"""PLY 读取与降采样。

根据 PLY 的属性自动判别两类内容并分别解析：

- **普通点云**：只取 XYZ + RGB，返回 ``PointCloud``。
- **3DGS（高斯泼溅）**：含 ``f_dc_*`` / ``opacity`` / ``scale_*`` / ``rot_*`` 属性时，
  解析出每个高斯的中心、颜色、不透明度与 3x3 协方差矩阵，返回 ``GaussianCloud``，
  供 viser 的 ``add_gaussian_splats`` 渲染。

不依赖 viser，可独立测试。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from plyfile import PlyData

# 无颜色时使用的统一灰色
DEFAULT_GRAY = (160, 160, 160)

# 颜色属性名的候选别名（按优先级）
_COLOR_ALIASES = [("red", "green", "blue"), ("r", "g", "b")]

# 判定为 3DGS 所需的最小属性集合（INRIA / 通用 3DGS 导出约定）
_GS_REQUIRED = ("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "rot_0")

# 0 阶球谐系数（DC 项 → 基础颜色）
_SH_C0 = 0.28209479177387814

# 各向异性下限：每个高斯最小标准差不低于其最大标准差的该比例。
# viser 把协方差打包成 float16，过薄（高各向异性）的高斯在 f16 下会退化为奇异/负定矩阵，
# 导致前端 conic 求逆得到 NaN → THREE 报 “Computed radius is NaN” 而无法显示。
# 0.1 表示方差各向异性最多 100:1，足以保留薄面片观感且保证 f16 下协方差正定。
_MIN_ANISO_RATIO = 0.1


@dataclass
class PointCloud:
    """普通点云：points float32 (N,3)，colors uint8 (N,3) 0–255。"""

    points: np.ndarray
    colors: np.ndarray
    has_color: bool

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def positions(self) -> np.ndarray:
        """用于计算包围盒 / 框选相机的坐标。"""
        return self.points

    def subsample(self, idx: np.ndarray) -> "PointCloud":
        return PointCloud(self.points[idx], self.colors[idx], self.has_color)


@dataclass
class GaussianCloud:
    """3DGS 高斯集合，字段与 viser ``add_gaussian_splats`` 对齐。

    - centers   float32 (N,3)
    - colors    float32 (N,3)，0–1
    - opacities float32 (N,1)，0–1
    - covariances float32 (N,3,3)
    """

    centers: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    covariances: np.ndarray

    @property
    def count(self) -> int:
        return len(self.centers)

    @property
    def positions(self) -> np.ndarray:
        return self.centers

    def subsample(self, idx: np.ndarray) -> "GaussianCloud":
        return GaussianCloud(
            self.centers[idx],
            self.colors[idx],
            self.opacities[idx],
            self.covariances[idx],
        )


def _vertex_element(plydata: PlyData, path: str):
    for el in plydata.elements:
        if el.name == "vertex":
            return el
    if not plydata.elements:
        raise ValueError(f"PLY 文件没有任何元素: {path}")
    return plydata.elements[0]


def is_gaussian_ply(names) -> bool:
    """根据属性名集合判断是否为 3DGS 文件。"""
    names = set(names)
    return all(n in names for n in _GS_REQUIRED)


def load_ply(path: str):
    """读取 PLY，自动判别返回 ``PointCloud`` 或 ``GaussianCloud``。

    支持 ASCII / 二进制小端 / 二进制大端（plyfile 自动识别）。
    """
    plydata = PlyData.read(path)
    element = _vertex_element(plydata, path)
    names = {p.name for p in element.properties}
    if not {"x", "y", "z"} <= names:
        raise ValueError(f"PLY 缺少 x/y/z 属性: {path}")

    if is_gaussian_ply(names):
        return _load_gaussian(element)
    return _load_pointcloud(element, names)


def _load_pointcloud(element, names: set) -> PointCloud:
    n = element.count
    points = np.column_stack(
        [element["x"], element["y"], element["z"]]
    ).astype(np.float32)
    if n == 0:
        return PointCloud(points.reshape(0, 3), np.zeros((0, 3), np.uint8), False)

    # 颜色：优先 red/green/blue，兼容 r/g/b 别名；浮点 0–1 色缩放到 0–255
    for keys in _COLOR_ALIASES:
        if set(keys) <= names:
            raw = np.column_stack([element[k] for k in keys])
            if np.issubdtype(raw.dtype, np.floating):
                raw = np.clip(raw, 0.0, 1.0) * 255.0
            colors = np.clip(raw, 0, 255).astype(np.uint8)
            return PointCloud(points, colors, True)

    colors = np.full((n, 3), DEFAULT_GRAY, dtype=np.uint8)
    return PointCloud(points, colors, False)


def _quats_to_matrices(q: np.ndarray) -> np.ndarray:
    """批量 wxyz 四元数 → (N,3,3) 旋转矩阵（输入需已归一化）。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    R = np.empty((n, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _load_gaussian(element) -> GaussianCloud:
    centers = np.column_stack(
        [element["x"], element["y"], element["z"]]
    ).astype(np.float32)
    n = element.count
    if n == 0:
        return GaussianCloud(
            centers.reshape(0, 3),
            np.zeros((0, 3), np.float32),
            np.zeros((0, 1), np.float32),
            np.zeros((0, 3, 3), np.float32),
        )

    # 颜色：DC 球谐 → RGB，sigmoid 之外直接线性映射并裁剪到 [0,1]
    f_dc = np.column_stack(
        [element["f_dc_0"], element["f_dc_1"], element["f_dc_2"]]
    ).astype(np.float32)
    colors = np.clip(0.5 + _SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)

    # 不透明度：logit → sigmoid
    opacity = element["opacity"].astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-opacity))).reshape(-1, 1).astype(np.float32)

    # 协方差：Σ = R diag(s²) Rᵀ，其中 s = exp(scale)，R 由四元数 rot 得到
    scales = np.exp(
        np.column_stack(
            [element["scale_0"], element["scale_1"], element["scale_2"]]
        ).astype(np.float32)
    )
    # 限制各向异性，避免 float16 协方差退化为奇异矩阵（见 _MIN_ANISO_RATIO 注释）
    scales = np.maximum(scales, scales.max(axis=1, keepdims=True) * _MIN_ANISO_RATIO)
    quats = np.column_stack(
        [element["rot_0"], element["rot_1"], element["rot_2"], element["rot_3"]]
    ).astype(np.float32)
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    quats = quats / norm
    R = _quats_to_matrices(quats)
    # M = R · diag(s)：把 R 的第 j 列乘以 s_j
    M = R * scales[:, None, :]
    covariances = (M @ np.transpose(M, (0, 2, 1))).astype(np.float32)

    cloud = GaussianCloud(centers, colors, opacities, covariances)
    return _drop_nonfinite_gaussians(cloud)


def _drop_nonfinite_gaussians(g: GaussianCloud) -> GaussianCloud:
    """防御性丢弃任何中心/颜色/不透明度/协方差含 NaN/Inf 的高斯，避免污染渲染。"""
    finite = (
        np.isfinite(g.centers).all(axis=1)
        & np.isfinite(g.colors).all(axis=1)
        & np.isfinite(g.opacities).all(axis=1)
        & np.isfinite(g.covariances).all(axis=(1, 2))
    )
    if finite.all():
        return g
    return g.subsample(np.nonzero(finite)[0])


def choose_indices(
    n: int, max_points: int, rng: np.random.Generator | None = None
) -> np.ndarray | None:
    """点数超过 max_points 时返回无放回随机下标，否则返回 None（表示全量）。"""
    if max_points <= 0 or n <= max_points:
        return None
    if rng is None:
        rng = np.random.default_rng(0)
    return rng.choice(n, max_points, replace=False)


def downsample(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """（兼容旧接口）点数超过 max_points 时无放回随机降采样，否则原样返回。"""
    idx = choose_indices(len(points), max_points, rng)
    if idx is None:
        return points, colors
    return points[idx], colors[idx]
