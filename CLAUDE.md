# CLAUDE.md

## 项目目的

浏览器端点云 / 3DGS 浏览 + 相机轨迹录制工具。用户在浏览器中选择目录下的 PLY 浏览，可一键录制相机轨迹为 JSON，并支持轨迹可视化与回放。纯 Python 技术栈。

## 技术约束（重要）

- **仅使用 viser** 提供 Web 服务与 3D 渲染，**不引入** FastAPI/Flask 等 Web 框架，**不写**自定义 JS/HTML
- PLY 解析用 **plyfile**（轻量），**不用 open3d**（重依赖，本项目不需要）
- **按 PLY 属性自动选择渲染方式**：含完整 3DGS 属性（`f_dc_*` + `opacity` + `scale_*` + `rot_*`，判定见 `ply_loader._GS_REQUIRED`）时按 3DGS 用 `scene.add_gaussian_splats` 渲染；否则按普通点云（XYZ + RGB）用 `scene.add_point_cloud` 渲染。两条路径由 `load_ply` 返回 `PointCloud` 或 `GaussianCloud` 区分
- 依赖只有：viser、numpy、plyfile（见 environment.yml，conda 环境名 `anyrecon_explorer`）

## 模块职责

```
anyrecon_explorer/
├── ply_loader.py   # 纯 IO：load_ply()（自动判别 PointCloud/GaussianCloud）/ choose_indices()，不依赖 viser，可单测
├── trajectory.py   # TrajectoryRecorder + JSON schema 序列化，不依赖 viser，可单测
└── app.py          # 所有 viser/GUI/线程逻辑 + main()，唯一依赖 viser 的模块
```

保持 `ply_loader.py` 和 `trajectory.py` 不导入 viser，便于独立测试。`load_ply` 返回两类 dataclass：`PointCloud`(points/colors/has_color) 或 `GaussianCloud`(centers/colors/opacities/covariances)，二者都提供 `count`/`positions`/`subsample(idx)`，App 据类型分派渲染。

## 数据约定

- 点坐标：`np.float32`，形状 `(N, 3)`
- 点云颜色：`np.uint8`，形状 `(N, 3)`，0–255；无颜色时统一灰色 `(160, 160, 160)`
- 3DGS：颜色由 DC 球谐还原 `rgb = clip(0.5 + 0.2820948·f_dc, 0, 1)`（float32 0–1）；不透明度 `sigmoid(opacity)`，形状 `(N,1)`；协方差 `Σ = R·diag(exp(scale)²)·Rᵀ`（R 由 `rot` 四元数得到），形状 `(N,3,3)`，传给 `add_gaussian_splats`。`f_rest_*`（高阶球谐）忽略，只用 DC 项做视角无关着色
- 3DGS 两个**必须保留**的鲁棒性处理（去掉会导致浏览器报 `THREE … Computed radius is NaN` 而黑屏）：① 各向异性下限 `_MIN_ANISO_RATIO`——viser 把协方差打包成 **float16**，过薄高斯在 f16 下退化为奇异/负定矩阵，前端 conic 求逆得 NaN，故每个高斯最小标准差不低于最大的 10%；② `_drop_nonfinite_gaussians` 丢弃含 NaN/Inf 的高斯
- 相机框选/默认尺度用 **1–99 百分位包围盒**（`app._load_and_show`），而非 min/max——3DGS 导出常有极端离群 floater（本样本 full-bbox 1.1e6 vs 1–99 百分位仅 885），用 min/max 会把相机推到远处导致看不到主体
- 四元数：**wxyz 顺序**（w 在前，viser 约定；3DGS 的 `rot_0..3` 亦按 wxyz 解读）
- FOV：垂直视场角，**弧度**
- 轨迹时间 `t`：距录制开始的相对秒数
- 轨迹位姿：JSON 中以 `transform_matrix`（**OpenCV 格式 c2w 4x4**，按行的嵌套数组）记录，参考 NeRF `transforms.json`。viser 相机内部即 OpenCV 约定（+X 右 / +Y 下 / +Z 朝前），故 c2w 直接为 `[[R, position], [0,0,0,1]]`，**不做坐标轴翻转**。`Frame` dataclass 在内存中仍保留 `position`/`wxyz`/`look_at`，仅序列化层（`trajectory.py` 的 `_frame_to_dict`/`_frame_from_dict`）做矩阵互转
- 轨迹 JSON 有 `version` 字段（当前为 2）；如做破坏性修改需升版本并在 `from_dict` 中做迁移。`from_dict` 当前兼容 v1（`position`/`wxyz`/`look_at`）与 v2（`transform_matrix`）

## 线程规则

- 录制线程（`TrajectoryRecorder._loop`）**只读**相机状态，不写场景
- 回放线程是**唯一**写 `client.camera` 的线程；多字段写入用 `with client.atomic():`
- `is_recording` / `is_playing` 标志 + 按钮禁用防止录制与回放并发
- 录制中禁止切换点云/目录

## 多客户端模型

viser 场景与 GUI 在所有浏览器标签页间共享，但相机是每客户端独立的。录制/回放只针对"活跃客户端"（最近连接者），操作前重新解析活跃客户端，无客户端连接时报错中止。

## 运行与验证

```bash
conda activate anyrecon_explorer
python -m anyrecon_explorer --dir data   # 打开 http://localhost:8080
```

`--dir` 接受**目录**（相对路径按当前 cwd 解析），递归扫描其下 `.ply`。

端到端验证：用 `data/export_last.ply`（3DGS，100 万高斯，含 `f_dc_*`/`opacity`/`scale_*`/`rot_*`）——选择加载应识别为「3DGS 高斯」并用 splatting 渲染（Point size 控件自动禁用）；换一个普通点云 PLY 应识别为「点云」并可调 Max points 降采样、调 Point size。再录制 5 秒轨迹 → 确认 `./trajectories/` 生成 JSON（每帧 `transform_matrix` 为 OpenCV c2w）、t 单调、静止帧被去重 → 加载轨迹绘制并回放。

修改 viser 场景/相机相关调用前，先核对 https://viser.studio 的当前 API 签名。
