"""viser 应用：点云浏览 GUI + 相机轨迹录制/回放。

本模块是唯一依赖 viser 的模块；线程规则：
录制线程只读相机，回放线程是唯一写 client.camera 的线程。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time

import numpy as np
import viser

from anyrecon_explorer.ply_loader import (
    GaussianCloud,
    PointCloud,
    choose_indices,
    load_ply,
)
from anyrecon_explorer.trajectory import (
    Frame,
    TrajectoryRecorder,
    from_dict,
    list_trajectories,
    load_json,
    make_trajectory_filename,
    save_json,
    to_dict,
)

NO_FILES = "(未找到 .ply 文件)"
NO_TRAJ = "(无已保存轨迹)"
TRAJ_DIR = "trajectories"


class App:
    def __init__(
        self,
        server: viser.ViserServer,
        root_dir: str,
        max_points: int,
        sample_hz: float,
    ):
        self.server = server
        self.root_dir = os.path.realpath(root_dir)
        self.traj_dir = os.path.abspath(TRAJ_DIR)
        os.makedirs(self.traj_dir, exist_ok=True)

        # 点云 / 3DGS 状态
        self.current_ply_rel: str | None = None
        self.full_cloud: PointCloud | GaussianCloud | None = None
        self.scene_handle = None
        self.shown_count = 0
        self.scene_center = np.zeros(3)
        self.scene_diag = 1.0
        self.rng = np.random.default_rng(0)

        # 录制/回放状态
        self.recorder: TrajectoryRecorder | None = None
        self.is_recording = False
        self.is_playing = False
        self.playback_stop = threading.Event()
        self.loaded_frames: list[Frame] = []
        self.traj_handles: list = []
        self.active_client: viser.ClientHandle | None = None
        self._suppress_file_cb = False
        self._suppress_point_size_cb = False

        self._build_gui(max_points, sample_hz)
        server.on_client_connect(self._on_client_connect)
        server.on_client_disconnect(self._on_client_disconnect)

        # 启动时加载第一个点云
        files = self._scan_ply_files()
        if files and files[0] != NO_FILES:
            self._load_and_show(files[0])

    # ------------------------------------------------------------------ GUI

    def _build_gui(self, max_points: int, sample_hz: float) -> None:
        gui = self.server.gui

        with gui.add_folder("Data"):
            self.dir_text = gui.add_text("Directory", initial_value=self.root_dir)
            self.dir_apply = gui.add_button("Apply directory")
            self.file_dd = gui.add_dropdown(
                "Point cloud (.ply)", options=(NO_FILES,)
            )
            self.max_points_num = gui.add_number(
                "Max points", initial_value=max_points, min=1000, step=1000
            )
            self.point_size_num = gui.add_number(
                "Point size", initial_value=0.01, min=0.0, step=0.001
            )
            self.status_text = gui.add_text(
                "Status", initial_value="就绪", disabled=True
            )

        with gui.add_folder("Recording"):
            self.rate_num = gui.add_number(
                "Sample rate (Hz)", initial_value=sample_hz, min=1, max=120, step=1
            )
            self.rec_start = gui.add_button("● Start recording")
            self.rec_stop = gui.add_button("■ Stop recording", disabled=True)

        with gui.add_folder("Trajectory"):
            self.traj_dd = gui.add_dropdown(
                "Saved trajectory", options=self._traj_options()
            )
            self.traj_refresh = gui.add_button("Refresh list")
            self.traj_load = gui.add_button("Load & draw")
            self.show_frustums = gui.add_checkbox(
                "Show keyframe frustums", initial_value=True
            )
            self.frustum_stride = gui.add_number(
                "Frustum every N frames", initial_value=20, min=1, step=1
            )
            self.play_btn = gui.add_button("▶ Play")
            self.play_stop_btn = gui.add_button("⏹ Stop playback", disabled=True)

        self.dir_apply.on_click(lambda _: self._apply_directory())
        self.file_dd.on_update(lambda _: self._on_file_selected())
        self.max_points_num.on_update(lambda _: self._on_max_points_changed())
        self.point_size_num.on_update(lambda _: self._on_point_size_changed())
        self.rec_start.on_click(lambda _: self._start_recording())
        self.rec_stop.on_click(lambda _: self._stop_recording())
        self.traj_refresh.on_click(lambda _: self._refresh_traj_options())
        self.traj_load.on_click(lambda _: self._load_trajectory())
        self.play_btn.on_click(lambda _: self._start_playback())
        self.play_stop_btn.on_click(lambda _: self.playback_stop.set())

    def _set_status(self, msg: str) -> None:
        self.status_text.value = msg
        print(f"[status] {msg}")

    # ----------------------------------------------------------- 客户端管理

    def _on_client_connect(self, client: viser.ClientHandle) -> None:
        # 最近连接的客户端成为活跃客户端
        self.active_client = client
        self._frame_camera(client)

    def _on_client_disconnect(self, client: viser.ClientHandle) -> None:
        if self.active_client is client:
            self.active_client = None

    def _resolve_client(self) -> viser.ClientHandle | None:
        """取活跃客户端；已断开则回退到最近连接的客户端。"""
        clients = self.server.get_clients()
        if self.active_client is not None and self.active_client.client_id in clients:
            return self.active_client
        if clients:
            self.active_client = clients[max(clients.keys())]
            return self.active_client
        return None

    def _frame_camera(self, client: viser.ClientHandle) -> None:
        """让客户端相机框住当前点云。"""
        if self.full_cloud is None or self.full_cloud.count == 0:
            return
        offset = np.array([0.0, 0.0, self.scene_diag * 0.8 + 1e-3])
        with client.atomic():
            client.camera.position = self.scene_center + offset
            client.camera.look_at = self.scene_center

    # ----------------------------------------------------------- 文件与加载

    def _scan_ply_files(self) -> list[str]:
        """递归扫描根目录下的 .ply 文件并刷新下拉框，返回相对路径列表。"""
        paths = sorted(
            os.path.relpath(p, self.root_dir)
            for p in glob.glob(
                os.path.join(glob.escape(self.root_dir), "**", "*.ply"),
                recursive=True,
            )
        )
        options = paths if paths else [NO_FILES]
        self._suppress_file_cb = True
        try:
            self.file_dd.options = tuple(options)
            self.file_dd.value = options[0]
        finally:
            self._suppress_file_cb = False
        return options

    def _apply_directory(self) -> None:
        if self.is_recording:
            self._set_status("录制中，请先停止录制再切换目录")
            return
        path = os.path.realpath(os.path.expanduser(self.dir_text.value.strip()))
        if not os.path.isdir(path):
            self._set_status(f"目录不存在: {path}")
            return
        self.root_dir = path
        self.dir_text.value = path
        files = self._scan_ply_files()
        if files[0] == NO_FILES:
            self._set_status(f"目录中没有 .ply 文件: {path}")
        else:
            self._set_status(f"找到 {len(files)} 个 .ply 文件")
            self._load_and_show(files[0])

    def _resolve_ply_path(self, rel: str) -> str | None:
        """将下拉框相对路径还原为绝对路径，并确认仍在根目录内。"""
        full = os.path.realpath(os.path.join(self.root_dir, rel))
        if full != self.root_dir and not full.startswith(self.root_dir + os.sep):
            return None
        return full

    def _on_file_selected(self) -> None:
        if self._suppress_file_cb:
            return
        rel = self.file_dd.value
        if rel == NO_FILES:
            return
        if self.is_recording:
            self._set_status("录制中，请先停止录制再切换点云")
            return
        self._load_and_show(rel)

    def _load_and_show(self, rel: str) -> None:
        full = self._resolve_ply_path(rel)
        if full is None or not os.path.isfile(full):
            self._set_status(f"无效的文件路径: {rel}")
            return
        self._set_status(f"加载中: {rel} ...")
        try:
            cloud = load_ply(full)
        except Exception as e:
            self._set_status(f"加载失败: {e}")
            return
        if cloud.count == 0:
            self._set_status(f"文件没有点: {rel}")
            return

        self.current_ply_rel = rel
        self.full_cloud = cloud
        # 用 1–99 百分位包围盒做相机框选 / 默认尺度，避免极端离群点（floater）把场景撑大
        lo = np.percentile(cloud.positions, 1.0, axis=0)
        hi = np.percentile(cloud.positions, 99.0, axis=0)
        self.scene_center = (lo + hi) / 2.0
        self.scene_diag = float(np.linalg.norm(hi - lo)) or 1.0

        is_gaussian = isinstance(cloud, GaussianCloud)
        # 点大小仅对普通点云有意义；3DGS 由协方差决定大小，禁用该控件
        self.point_size_num.disabled = is_gaussian
        if not is_gaussian:
            self._suppress_point_size_cb = True
            try:
                self.point_size_num.value = round(
                    max(self.scene_diag / 1000.0, 1e-4), 4
                )
            finally:
                self._suppress_point_size_cb = False

        self._render_cloud()
        kind = "3DGS 高斯" if is_gaussian else "点云"
        if is_gaussian:
            note = ""
        else:
            note = "" if cloud.has_color else "（无颜色，使用灰色）"
        self._set_status(
            f"已加载 {rel}（{kind}）: {cloud.count:,} 个，显示 {self.shown_count:,} 个{note}"
        )

        client = self._resolve_client()
        if client is not None:
            self._frame_camera(client)

    def _render_cloud(self) -> None:
        """从缓存的全量数据按当前 Max points 降采样并渲染（点云或 3DGS）。"""
        cloud = self.full_cloud
        if cloud is None:
            return
        idx = choose_indices(cloud.count, int(self.max_points_num.value), self.rng)
        shown = cloud if idx is None else cloud.subsample(idx)
        self.shown_count = shown.count
        if self.scene_handle is not None:
            self.scene_handle.remove()
            self.scene_handle = None
        if isinstance(shown, GaussianCloud):
            self.scene_handle = self.server.scene.add_gaussian_splats(
                "/cloud",
                centers=shown.centers,
                covariances=shown.covariances,
                rgbs=shown.colors,
                opacities=shown.opacities,
            )
        else:
            self.scene_handle = self.server.scene.add_point_cloud(
                "/cloud",
                points=shown.points,
                colors=shown.colors,
                point_size=max(float(self.point_size_num.value), 1e-5),
            )

    def _on_point_size_changed(self) -> None:
        """热更新点大小，无需重建点云（仅普通点云）。"""
        if self._suppress_point_size_cb or self.scene_handle is None:
            return
        if not isinstance(self.full_cloud, PointCloud):
            return
        self.scene_handle.point_size = max(float(self.point_size_num.value), 1e-5)

    def _on_max_points_changed(self) -> None:
        if self.full_cloud is None:
            return
        self._render_cloud()
        self._set_status(
            f"显示 {self.shown_count:,} / {self.full_cloud.count:,} 个"
        )

    # --------------------------------------------------------------- 录制

    def _start_recording(self) -> None:
        if self.is_recording:
            return
        if self.is_playing:
            self._set_status("回放中，请先停止回放再录制")
            return
        client = self._resolve_client()
        if client is None:
            self._set_status("没有已连接的浏览器客户端，无法录制")
            return

        def get_camera_state() -> dict:
            cam = client.camera
            return {
                "position": cam.position,
                "wxyz": cam.wxyz,
                "fov": cam.fov,
                "aspect": cam.aspect,
                "look_at": cam.look_at,
            }

        self.recorder = TrajectoryRecorder(
            sample_hz=float(self.rate_num.value),
            get_camera_state=get_camera_state,
        )
        self.recorder.start()
        self.is_recording = True
        self.rec_start.disabled = True
        self.rec_stop.disabled = False
        self.play_btn.disabled = True
        self._set_status(f"录制中（{self.rate_num.value} Hz）... 自由浏览点云")

    def _stop_recording(self) -> None:
        if not self.is_recording or self.recorder is None:
            return
        frames = self.recorder.stop()
        self.is_recording = False
        self.rec_start.disabled = False
        self.rec_stop.disabled = True
        self.play_btn.disabled = False

        if not frames:
            self._set_status("未录到任何帧")
            return

        d = to_dict(
            frames,
            point_cloud=os.path.basename(self.current_ply_rel or ""),
            sample_rate_hz=float(self.rate_num.value),
        )
        fname = make_trajectory_filename(self.current_ply_rel or "trajectory")
        path = os.path.join(self.traj_dir, fname)
        save_json(d, path)

        # 浏览器下载
        client = self._resolve_client()
        if client is not None:
            content = json.dumps(d, indent=2).encode("utf-8")
            client.send_file_download(fname, content)

        self.loaded_frames = frames
        self._draw_trajectory(frames)
        self._refresh_traj_options()
        self.traj_dd.value = fname
        self._set_status(f"已保存 {len(frames)} 帧到 trajectories/{fname}")

    # ----------------------------------------------------------- 轨迹绘制

    def _traj_options(self) -> tuple[str, ...]:
        files = list_trajectories(self.traj_dir)
        return tuple(files) if files else (NO_TRAJ,)

    def _refresh_traj_options(self) -> None:
        self.traj_dd.options = self._traj_options()

    def _clear_trajectory_drawing(self) -> None:
        for h in self.traj_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.traj_handles = []

    def _draw_trajectory(self, frames: list[Frame]) -> None:
        self._clear_trajectory_drawing()
        if not frames:
            return
        positions = np.array([f.position for f in frames], dtype=np.float32)
        if len(frames) >= 2:
            self.traj_handles.append(
                self.server.scene.add_spline_catmull_rom(
                    "/traj/line",
                    positions,
                    line_width=3.0,
                    color=(255, 140, 0),
                )
            )
        if self.show_frustums.value:
            stride = max(1, int(self.frustum_stride.value))
            scale = max(self.scene_diag * 0.02, 1e-3)
            for i in range(0, len(frames), stride):
                f = frames[i]
                self.traj_handles.append(
                    self.server.scene.add_camera_frustum(
                        f"/traj/frustum_{i}",
                        fov=f.fov,
                        aspect=f.aspect,
                        scale=scale,
                        color=(80, 160, 255),
                        wxyz=np.array(f.wxyz),
                        position=np.array(f.position),
                    )
                )

    def _load_trajectory(self) -> None:
        fname = self.traj_dd.value
        if fname == NO_TRAJ:
            self._set_status("没有可加载的轨迹")
            return
        path = os.path.join(self.traj_dir, os.path.basename(fname))
        try:
            meta, frames = from_dict(load_json(path))
        except Exception as e:
            self._set_status(f"轨迹加载失败: {e}")
            return
        self.loaded_frames = frames
        self._draw_trajectory(frames)
        msg = f"已加载轨迹 {fname}（{len(frames)} 帧）"
        current = os.path.basename(self.current_ply_rel or "")
        if meta.get("point_cloud") and meta["point_cloud"] != current:
            msg += f"，注意：录制时的点云为 {meta['point_cloud']}，与当前不同"
        self._set_status(msg)

    # --------------------------------------------------------------- 回放

    def _start_playback(self) -> None:
        if self.is_playing:
            return
        if self.is_recording:
            self._set_status("录制中，请先停止录制再回放")
            return
        if not self.loaded_frames:
            self._set_status("没有可回放的轨迹，请先录制或加载")
            return
        client = self._resolve_client()
        if client is None:
            self._set_status("没有已连接的浏览器客户端，无法回放")
            return
        self.is_playing = True
        self.playback_stop.clear()
        self.play_btn.disabled = True
        self.play_stop_btn.disabled = False
        self.rec_start.disabled = True
        threading.Thread(
            target=self._playback_loop,
            args=(list(self.loaded_frames), client),
            daemon=True,
        ).start()

    def _playback_loop(self, frames: list[Frame], client: viser.ClientHandle) -> None:
        self._set_status(f"回放中（{len(frames)} 帧）...")
        t_start = time.monotonic()
        try:
            for f in frames:
                if self.playback_stop.is_set():
                    self._set_status("回放已停止")
                    break
                wait = f.t - (time.monotonic() - t_start)
                if wait > 0:
                    time.sleep(wait)
                with client.atomic():
                    client.camera.position = np.array(f.position)
                    client.camera.wxyz = np.array(f.wxyz)
                    client.camera.fov = f.fov
            else:
                # 回放完整结束：恢复 look_at，方便继续轨道操作
                last = frames[-1]
                client.camera.look_at = np.array(last.look_at)
                self._set_status("回放完成")
        except Exception as e:
            self._set_status(f"回放出错: {e}")
        finally:
            self.is_playing = False
            self.play_btn.disabled = False
            self.play_stop_btn.disabled = True
            self.rec_start.disabled = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="浏览器端点云浏览 + 相机轨迹录制（viser）"
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="点云根目录（递归扫描 .ply）；相对路径按当前执行命令的目录解析",
    )
    parser.add_argument("--host", default="127.0.0.1", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8080, help="服务端口")
    parser.add_argument("--max-points", type=int, default=2_000_000, help="默认最大渲染点数")
    parser.add_argument("--sample-hz", type=float, default=30.0, help="默认轨迹采样频率")
    args = parser.parse_args()

    # 相对路径以“当前执行命令的目录”（cwd）为根解析；同时展开 ~
    resolved_dir = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(resolved_dir):
        parser.error(
            f"目录不存在: {resolved_dir}\n"
            f"（输入参数 --dir={args.dir!r}，当前工作目录 {os.getcwd()!r}）"
        )
    args.dir = resolved_dir

    server = viser.ViserServer(host=args.host, port=args.port)
    App(server, root_dir=args.dir, max_points=args.max_points, sample_hz=args.sample_hz)
    server.sleep_forever()


if __name__ == "__main__":
    main()
