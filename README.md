# AnyRecon Explorer

基于 [viser](https://viser.studio) 的浏览器端点云浏览与相机轨迹录制工具，纯 Python 实现。

在浏览器中选择并浏览目录下的 PLY 文件，**根据文件属性自动判别**按普通点云还是 3D Gaussian Splatting（3DGS）渲染，并可一键录制浏览过程中的相机轨迹，保存为 JSON 供后续使用。

## 功能特性

- **自动判别渲染方式**：含完整 3DGS 属性（`f_dc_*` + `opacity` + `scale_*` + `rot_*`）的 PLY 按 **3DGS 高斯泼溅**渲染（DC 球谐还原颜色、`exp(scale)` + `rot` 四元数还原协方差）；否则按**普通点云**（XYZ + RGB）渲染
- **点云浏览**：支持 ASCII / 二进制（小端、大端）PLY，读取 XYZ + RGB；无颜色时显示统一灰色
- **目录选择**：启动时通过 `--dir` 指定根目录，页面上也可随时输入新目录切换；下拉框递归列出目录下所有 `.ply` 文件
- **降采样**：页面可设置"最大点数"（默认 200 万），超出时随机降采样（点云与 3DGS 通用），保证大文件流畅渲染
- **点大小调节**：普通点云可实时调整渲染点大小，加载时按场景包围盒自动给出默认值；3DGS 大小由协方差决定，该控件自动禁用
- **轨迹录制**：点击按钮开始/停止录制；后台以固定频率（默认 30Hz，可调）采样相机状态，相机静止时自动去重
- **轨迹保存**：录制结束后自动保存 JSON 到服务端 `./trajectories/` 目录（每帧位姿为 OpenCV 格式 c2w 矩阵，参考 NeRF `transforms.json`），同时触发浏览器下载
- **轨迹可视化**：录制完成或加载已存轨迹后，在场景中绘制轨迹线与关键帧相机视锥
- **轨迹回放**：让浏览器相机按录制时间轴沿轨迹自动飞行重放

## 环境要求

- Python 3.10+
- conda（用于创建虚拟环境）

## 安装

```bash
conda env create -f environment.yml   # 会以可编辑模式安装本项目（pip install -e .）
conda activate anyrecon_explorer
```

> 若环境已存在、或想单独安装，在仓库根目录执行 `pip install -e .` 即可。安装后 `python -m anyrecon_explorer` 可在任意目录下运行（无需 `cd` 到仓库根目录）。

## 使用

`--dir` 接受的是**目录**（会递归扫描其下所有 `.ply`），不是单个文件。例如扫描仓库内的 `data/` 目录：

```bash
python -m anyrecon_explorer --dir data
```

也可以用安装时注册的命令行入口：

```bash
anyrecon-explorer --dir data
```

然后在浏览器中打开终端打印的地址（默认 `http://localhost:8080`），在页面 "Data" 面板的下拉框中选择 `export_last.ply` 等点云文件加载。

### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dir` | 当前目录 | 点云根目录（递归扫描 `.ply`） |
| `--host` | `127.0.0.1` | 服务监听地址 |
| `--port` | `8080` | 服务端口 |
| `--max-points` | `2000000` | 默认最大渲染点数 |
| `--sample-hz` | `30` | 默认轨迹采样频率 |

### 页面操作

1. **Data 面板**：在下拉框中选择文件即自动加载（按属性自动判别点云 / 3DGS，状态栏会标明）；可修改目录路径后点 "Apply directory" 切换目录；"Max points" 控制降采样阈值；"Point size" 调整渲染点大小（仅普通点云可用，加载时按场景尺度自动重置默认值，3DGS 下禁用）
2. **Recording 面板**：点 "● Start recording" 开始录制，自由浏览点云，点 "■ Stop recording" 结束 —— 轨迹自动保存到 `./trajectories/` 并触发浏览器下载，场景中同时画出轨迹线
3. **Trajectory 面板**：从下拉框选择已保存的轨迹，点 "Load & draw" 绘制；点 "▶ Play" 让相机沿轨迹回放，"⏹ Stop playback" 中断

## 轨迹 JSON 格式

参考 NeRF `transforms.json` 的形式：每帧位姿用 **OpenCV 格式的 c2w（camera-to-world）4x4 矩阵** `transform_matrix` 记录，矩阵按行排列为嵌套数组。viser 相机内部即 OpenCV 约定（相机系 +X 右 / +Y 下 / +Z 朝前），故 c2w 直接为 `[[R, t], [0, 0, 0, 1]]`，无坐标轴翻转。

```json
{
  "version": 2,
  "camera_model": "OPENCV",
  "point_cloud": "piont_cloud.ply",
  "created_at": "2026-06-04T16:30:00",
  "sample_rate_hz": 30,
  "frame_count": 412,
  "frames": [
    {
      "t": 0.0,
      "fov": 1.0472,
      "aspect": 1.7778,
      "transform_matrix": [
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `camera_model` | 相机模型，固定 `OPENCV` |
| `t` | 距录制开始的相对秒数 |
| `fov` | 垂直视场角，**弧度**（回放用） |
| `aspect` | 视口宽高比（回放用） |
| `transform_matrix` | OpenCV 格式 c2w 4x4 矩阵，`R` 为相机朝向、`t`（最后一列前三行）为相机世界坐标位置 |

> 兼容性：加载轨迹时同时支持旧版 v1（含 `position` / `wxyz` / `look_at` 字段）。

## 限制说明

- 录制与回放针对"活跃客户端"（最近连接的浏览器标签页）；多标签页共享同一场景，但只有活跃标签页的相机被录制/驱动
- 3DGS 渲染只用 DC 球谐项（`f_dc_*`）做视角无关着色，忽略高阶球谐 `f_rest_*`；判别依据是是否同时含 `f_dc_*` / `opacity` / `scale_*` / `rot_*` 属性
- 录制过程中禁止切换点云或目录

## 许可证

见 [LICENSE](LICENSE)。
