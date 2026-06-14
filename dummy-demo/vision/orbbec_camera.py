"""
Orbbec Gemini 335 相机封装 (pyorbbecsdk v2 / OrbbecSDK 2.x)
==========================================================

后台线程持续取流, 缓存最新一帧 color (BGR) + depth (uint16 mm)。
对外暴露 cv2.VideoCapture 兼容接口 (isOpened / read), 方便直接替换
web/app.py 里原来的 cv2.VideoCapture(0)。

额外提供:
  - read_depth_colormap(): 深度伪彩 BGR (JET, 近=蓝 远=红), 带中心点 mm 读数
  - latest_depth(): 原始 uint16 深度 (mm), 供算法/抓取用

线程安全: 后台线程写, 任意线程读最新缓存帧 (加锁拷贝)。
"""

import time
import threading
import logging
from typing import Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class OrbbecCamera:
    """Gemini 335 后台取流 + cv2 兼容接口。"""

    def __init__(self, color_size: Tuple[int, int] = None, fps: int = 30):
        """
        Args:
            color_size: (w, h) 期望彩色分辨率; None=用默认 profile
            fps: 期望帧率
        """
        self.color_size = color_size
        self.fps = fps

        self._pipe = None
        self._sdk = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        self._color_bgr: Optional[np.ndarray] = None   # 最新彩色 (BGR)
        self._depth_mm: Optional[np.ndarray] = None     # 最新深度 (uint16, mm)
        self._depth_scale: float = 1.0                  # 深度值 -> mm 比例
        self._opened = False
        self._depth_intr = None                          # 深度相机内参 (fx,fy,cx,cy)
        self._color_intr = None                          # RGB 相机内参

    # ─── 启动 / 停止 ────────────────────────────────────────

    def start(self) -> bool:
        try:
            import pyorbbecsdk as ob
        except ImportError:
            logger.warning("pyorbbecsdk 未安装")
            return False

        try:
            self._sdk = ob
            ctx = ob.Context()
            devs = ctx.query_devices()
            if devs.get_count() == 0:
                logger.warning("未发现 Orbbec 设备")
                return False

            dev = devs.get_device_by_index(0)
            info = dev.get_device_info()
            logger.info(f"Orbbec: {info.get_name()} fw={info.get_firmware_version()} "
                        f"sn={info.get_serial_number()}")

            pipe = ob.Pipeline(dev)
            cfg = ob.Config()

            # Color profile
            clist = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            if self.color_size is not None:
                try:
                    cprof = clist.get_video_stream_profile(
                        self.color_size[0], self.color_size[1],
                        ob.OBFormat.MJPG, self.fps)
                except Exception:
                    cprof = clist.get_default_video_stream_profile()
            else:
                cprof = clist.get_default_video_stream_profile()
            cfg.enable_stream(cprof)

            # Depth profile
            dlist = pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            dprof = dlist.get_default_video_stream_profile()
            cfg.enable_stream(dprof)

            pipe.start(cfg)
            self._pipe = pipe

            # 读出出厂内参 (用于 3D 反投影 / 手眼标定)
            try:
                cam_param = pipe.get_camera_param()
                di = cam_param.depth_intrinsic
                self._depth_intr = {"fx": di.fx, "fy": di.fy, "cx": di.cx,
                                    "cy": di.cy, "width": di.width, "height": di.height}
                ci = cam_param.rgb_intrinsic
                self._color_intr = {"fx": ci.fx, "fy": ci.fy, "cx": ci.cx,
                                    "cy": ci.cy, "width": ci.width, "height": ci.height}
                logger.info(f"depth intrinsic: {self._depth_intr}")
            except Exception as e:
                logger.warning(f"读取内参失败: {e}")

            logger.info(f"Gemini335 started: color {cprof.get_width()}x{cprof.get_height()} "
                        f"{cprof.get_format()}, depth {dprof.get_width()}x{dprof.get_height()} "
                        f"{dprof.get_format()}")

            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

            # 等首帧 (最多 ~3s)
            for _ in range(60):
                with self._lock:
                    if self._color_bgr is not None:
                        break
                time.sleep(0.05)

            self._opened = self._color_bgr is not None
            return self._opened
        except Exception as e:
            logger.error(f"Orbbec 启动失败: {e}")
            return False

    def _loop(self):
        ob = self._sdk
        while self._running:
            try:
                fs = self._pipe.wait_for_frames(200)
            except Exception:
                continue
            if fs is None:
                continue

            cf = fs.get_color_frame()
            if cf is not None:
                buf = np.frombuffer(cf.get_data(), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # MJPG -> BGR
                if img is None:
                    try:
                        img = buf.reshape((cf.get_height(), cf.get_width(), 3))
                    except Exception:
                        img = None
                if img is not None:
                    with self._lock:
                        self._color_bgr = img

            df = fs.get_depth_frame()
            if df is not None:
                w, h = df.get_width(), df.get_height()
                try:
                    scale = df.get_depth_scale()
                except Exception:
                    scale = 1.0
                d = np.frombuffer(df.get_data(), dtype=np.uint16).reshape((h, w))
                with self._lock:
                    self._depth_mm = d
                    self._depth_scale = scale or 1.0

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None
        self._opened = False
        logger.info("Gemini335 stopped")

    # ─── cv2.VideoCapture 兼容接口 ──────────────────────────

    def isOpened(self) -> bool:
        return self._opened and self._running

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """返回 (ret, BGR frame) — 与 cv2.VideoCapture.read() 一致。"""
        with self._lock:
            if self._color_bgr is None:
                return False, None
            return True, self._color_bgr.copy()

    def read_color(self) -> Optional[np.ndarray]:
        """只读最新彩色帧 (BGR)。与旧 vision.camera.Camera 接口兼容。"""
        with self._lock:
            return None if self._color_bgr is None else self._color_bgr.copy()

    def release(self):
        self.stop()

    def set(self, *args, **kwargs):
        # 兼容 cv2 调用, profile 在 start() 时已定
        return True

    # ─── 深度相关 ───────────────────────────────────────────

    def latest_depth(self) -> Optional[np.ndarray]:
        """最新原始深度 (uint16, 单位 mm * depth_scale)。"""
        with self._lock:
            if self._depth_mm is None:
                return None
            return self._depth_mm.copy()

    def depth_at(self, x: int, y: int) -> Optional[float]:
        """指定像素的深度 (mm)。坐标按深度图分辨率。"""
        with self._lock:
            if self._depth_mm is None:
                return None
            h, w = self._depth_mm.shape
            if not (0 <= x < w and 0 <= y < h):
                return None
            return float(self._depth_mm[y, x]) * self._depth_scale

    def get_depth_intrinsics(self) -> Optional[dict]:
        """深度相机内参 {fx,fy,cx,cy,width,height}。"""
        return dict(self._depth_intr) if self._depth_intr else None

    def get_color_intrinsics(self) -> Optional[dict]:
        """RGB 相机内参。"""
        return dict(self._color_intr) if self._color_intr else None

    def pixel_depth_to_camera_3d(self, u: float, v: float,
                                 depth_mm: float = None) -> Optional[tuple]:
        """
        深度图像素 (u,v) + 深度 -> 相机坐标系 3D 点 (Xc,Yc,Zc), 单位 mm。
        使用深度相机内参反投影: Xc=(u-cx)*Z/fx, Yc=(v-cy)*Z/fy, Zc=Z。

        Args:
            u, v: 深度图像素坐标
            depth_mm: 该点深度(mm); None=自动查 depth_at
        Returns:
            (Xc, Yc, Zc) mm, 或 None (无内参/无深度)
        """
        if self._depth_intr is None:
            return None
        if depth_mm is None:
            depth_mm = self.depth_at(int(round(u)), int(round(v)))
        if depth_mm is None or depth_mm <= 0:
            return None
        fx = self._depth_intr["fx"]
        fy = self._depth_intr["fy"]
        cx = self._depth_intr["cx"]
        cy = self._depth_intr["cy"]
        Xc = (u - cx) * depth_mm / fx
        Yc = (v - cy) * depth_mm / fy
        Zc = float(depth_mm)
        return (float(Xc), float(Yc), Zc)

    def read_depth_colormap(self) -> Tuple[bool, Optional[np.ndarray]]:
        """深度伪彩 (BGR, JET)。无效区 (0) 染黑, 中心点标 mm 读数。"""
        d = self.latest_depth()
        if d is None:
            return False, None
        scale = self._depth_scale
        valid = d > 0
        if valid.sum() == 0:
            return True, np.zeros((*d.shape, 3), dtype=np.uint8)
        # 归一化到有效深度范围
        dmin = d[valid].min()
        dmax = d[valid].max()
        norm = np.zeros(d.shape, dtype=np.uint8)
        if dmax > dmin:
            norm[valid] = ((d[valid].astype(np.float32) - dmin) /
                           (dmax - dmin) * 255).astype(np.uint8)
        vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        vis[~valid] = (0, 0, 0)  # 无深度区域黑
        # 中心点 mm
        h, w = d.shape
        cx, cy = w // 2, h // 2
        cz = float(d[cy, cx]) * scale
        cv2.circle(vis, (cx, cy), 5, (255, 255, 255), 1)
        txt = f"center {cz/1000:.2f}m" if cz > 0 else "center: --"
        cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        rng = f"range {dmin*scale/1000:.2f}-{dmax*scale/1000:.2f}m"
        cv2.putText(vis, rng, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 255, 200), 1)
        return True, vis


# ─── 统一相机初始化 ─────────────────────────────────────────

def open_camera():
    """
    统一的相机获取入口: 优先 Gemini 335 (RGB + 深度, pyorbbecsdk v2),
    取不到流则回退到 USB cv2.VideoCapture(0) (640x480 RGB)。

    所有需要相机的入口 (web/app.py, strands_agent.run_demo, main.py)
    都应通过此函数获取, 保证视觉链路统一到新相机方案。

    Returns:
        (camera, label) — camera 可能是 OrbbecCamera / cv2.VideoCapture / None
    """
    try:
        cam = OrbbecCamera()
        if cam.start():
            return cam, "Gemini 335 (RGB + 深度, pyorbbecsdk v2)"
        logger.warning("Gemini 335 未取到流, 回退 USB")
    except Exception as e:
        logger.warning(f"Orbbec 初始化失败 ({e}), 回退 USB")

    usb = cv2.VideoCapture(0)
    if usb.isOpened():
        usb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        usb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return usb, "USB 相机 (640x480 RGB)"
    return None, "无相机"
