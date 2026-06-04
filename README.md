# AnyRecon Explorer

基于 [viser](https://viser.studio) 的浏览器端点云浏览与相机轨迹录制工具，纯 Python 实现。

在浏览器中选择并浏览目录下的 PLY 点云文件（始终按普通点云渲染，**不**解读为 3D Gaussian Splatting），并可一键录制浏览过程中的相机轨迹，保存为 JSON 供后续使用。

## 功能特性

- **点云浏览**：支持 ASCII / 二进制（小端、大端）PLY 文件，读取 XYZ + RGB 渲染；无颜色时显示统一灰色；含 3DGS 属性（`f_dc_*`、`opacity` 等）的文件忽略多余属性，按普通点云渲染
- **目录选择**：启动时通过 `--dir` 指定根目录，页面上也可随时输入新目录切换；下拉框递归列出目录下所有 `.ply` 文件
- **降采样**：页面可设置"最大点数"（默认 200 万），超出时随机降采样，保证大点云流畅渲染
- **点大小调节**：页面可实时调整渲染点大小，加载新点云时按场景包围盒自动给出合适默认值
- **轨迹录制**：点击按钮开始/停止录制；后台以固定频率（默认 30Hz，可调）采样相机状态，相机静止时自动去重
- **轨迹保存**：录制结束后自动保存 JSON 到服务端 `./trajectories/` 目录，同时触发浏览器下载
- **轨迹可视化**：录制完成或加载已存轨迹后，在场景中绘制轨迹线与关键帧相机视锥
- **轨迹回放**：让浏览器相机按录制时间轴沿轨迹自动飞行重放

## 环境要求

- Python 3.10+
- conda（用于创建虚拟环境）

## 安装

```bash
conda env create -f environment.yml
conda activate anyrecon_explorer
```

## 使用

```bash
python -m anyrecon_explorer --dir /path/to/pointclouds
```

然后在浏览器中打开终端打印的地址（默认 `http://localhost:8080`）。

### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dir` | 当前目录 | 点云根目录（递归扫描 `.ply`） |
| `--host` | `127.0.0.1` | 服务监听地址 |
| `--port` | `8080` | 服务端口 |
| `--max-points` | `2000000` | 默认最大渲染点数 |
| `--sample-hz` | `30` | 默认轨迹采样频率 |

### 页面操作

1. **Data 面板**：在下拉框中选择点云文件即自动加载；可修改目录路径后点 "Apply directory" 切换目录；"Max points" 控制降采样阈值；"Point size" 调整渲染点大小（加载新点云时按场景尺度自动重置默认值）
2. **Recording 面板**：点 "● Start recording" 开始录制，自由浏览点云，点 "■ Stop recording" 结束 —— 轨迹自动保存到 `./trajectories/` 并触发浏览器下载，场景中同时画出轨迹线
3. **Trajectory 面板**：从下拉框选择已保存的轨迹，点 "Load & draw" 绘制；点 "▶ Play" 让相机沿轨迹回放，"⏹ Stop playback" 中断

## 轨迹 JSON 格式

```json
{
  "version": 1,
  "point_cloud": "piont_cloud.ply",
  "created_at": "2026-06-04T16:30:00",
  "sample_rate_hz": 30,
  "frame_count": 412,
  "frames": [
    {
      "t": 0.0,
      "position": [x, y, z],
      "wxyz": [w, x, y, z],
      "fov": 1.0472,
      "aspect": 1.7778,
      "look_at": [x, y, z]
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `t` | 距录制开始的相对秒数 |
| `position` | 相机在世界坐标系下的位置 |
| `wxyz` | 相机朝向四元数，**w 在前**（viser 约定） |
| `fov` | 垂直视场角，**弧度** |
| `aspect` | 视口宽高比 |
| `look_at` | 相机注视点（轨道控制中心） |

## 限制说明

- 录制与回放针对"活跃客户端"（最近连接的浏览器标签页）；多标签页共享同一场景，但只有活跃标签页的相机被录制/驱动
- 含 3DGS 属性的 PLY 文件按普通点云渲染（仅取 XYZ + RGB，无颜色时灰色），不做高斯渲染
- 录制过程中禁止切换点云或目录

## 许可证

见 [LICENSE](LICENSE)。
