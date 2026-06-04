# CLAUDE.md

## 项目目的

浏览器端点云浏览 + 相机轨迹录制工具。用户在浏览器中选择目录下的 PLY 点云浏览，可一键录制相机轨迹为 JSON，并支持轨迹可视化与回放。纯 Python 技术栈。

## 技术约束（重要）

- **仅使用 viser** 提供 Web 服务与 3D 渲染，**不引入** FastAPI/Flask 等 Web 框架，**不写**自定义 JS/HTML
- PLY 解析用 **plyfile**（轻量），**不用 open3d**（重依赖，本项目不需要）
- PLY 文件**始终按普通点云**解读（只取 XYZ + RGB），即使含 3DGS 属性（`f_dc_*`、`opacity`、`scale_*`、`rot_*`）也一律忽略，绝不做高斯渲染
- 依赖只有：viser、numpy、plyfile（见 environment.yml，conda 环境名 `anyrecon_explorer`）

## 模块职责

```
anyrecon_explorer/
├── ply_loader.py   # 纯 IO：load_ply() / downsample()，不依赖 viser，可单测
├── trajectory.py   # TrajectoryRecorder + JSON schema 序列化，不依赖 viser，可单测
└── app.py          # 所有 viser/GUI/线程逻辑 + main()，唯一依赖 viser 的模块
```

保持 `ply_loader.py` 和 `trajectory.py` 不导入 viser，便于独立测试。

## 数据约定

- 点坐标：`np.float32`，形状 `(N, 3)`
- 颜色：`np.uint8`，形状 `(N, 3)`，0–255；无颜色时统一灰色 `(160, 160, 160)`
- 四元数：**wxyz 顺序**（w 在前，viser 约定）
- FOV：垂直视场角，**弧度**
- 轨迹时间 `t`：距录制开始的相对秒数
- 轨迹 JSON 有 `version` 字段（当前为 1）；如做破坏性修改需升版本并在 `from_dict` 中做迁移

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
python -m anyrecon_explorer --dir .   # 打开 http://localhost:8080
```

端到端验证：用仓库根目录的 `piont_cloud.ply`（ASCII，849,346 点，含 RGB）——选择加载 → 调整 Max points 观察降采样 → 录制 5 秒轨迹 → 确认 `./trajectories/` 生成 JSON 且 t 单调、静止帧被去重 → 加载轨迹绘制并回放。

修改 viser 场景/相机相关调用前，先核对 https://viser.studio 的当前 API 签名。
