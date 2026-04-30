"""
相机采集模块 (实测可用版)
========================

Orbbec 深度摄像头:
  - RGB: 通过 OpenCV AVFoundation 读取 (camera index 1)
  - 深度: 通过 pyorbbecsdk v1 读取 (编译安装)

设备信息:
  - 型号: SV1301S_U3 (Orbbec)
  - RGB PID: 0x0511 (Sonix)
  - 深度 PID: 0x0614 (Orbbec)
  - 深度流: 640x400 @ 30fps
  - RGB 流: 1920x1080 (AVFoundation)

注意: RGB 和深度来自不同 pipeline，分辨率不同，
      使用时需要对齐或分别处理。
"""

import numpy as np
import cv2
import logging
import time
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """相机内参"""
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 640
    height: int = 480
    dist_coeffs: np.ndarray = None

    def __post_init__(self):
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros(5)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


class Camera:
    """
    Orbbec RGBD 相机

    RGB 通过 OpenCV AVFoundation, 深度通过 pyorbbecsdk。

    Args:
        rgb_device: OpenCV camera index for RGB (default 1 = Orbbec USB Camera)
        rgb_width: RGB 图像宽度
        rgb_height: RGB 图像高度
        enable_depth: 是否启用深度
    """

    def __init__(
        self,
        rgb_device: int = 1,
        rgb_width: int = 640,
        rgb_height: int = 480,
        enable_depth: bool = True,
    ):
        self.rgb_device = rgb_device
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.enable_depth = enable_depth

        self._cap: Optional[cv2.VideoCapture] = None
        self._ob_pipeline = None
        self._ob_config = None
        self._depth_intrinsics: Optional[CameraIntrinsics] = None
        self._running = False

    def start(self):
        """启动相机"""
        # RGB via OpenCV
        logger.info(f"启动 RGB (AVFoundation device {self.rgb_device})")
        self._cap = cv2.VideoCapture(self.rgb_device)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开 RGB 相机 (device {self.rgb_device})")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.rgb_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.rgb_height)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"RGB 分辨率: {actual_w}x{actual_h}")

        # 预热 — 前几帧可能黑屏
        for _ in range(5):
            self._cap.read()
            time.sleep(0.1)

        # 深度 via pyorbbecsdk
        if self.enable_depth:
            self._start_depth()

        self._running = True
        logger.info("相机启动完成")

    def _start_depth(self):
        """启动 Orbbec 深度流"""
        try:
            from pyorbbecsdk import Context, Pipeline, Config, OBSensorType
        except ImportError:
            logger.warning("pyorbbecsdk 未安装，深度不可用")
            self.enable_depth = False
            return

        try:
            ctx = Context()
            device_list = ctx.query_devices()
            if device_list.get_count() == 0:
                logger.warning("未找到 Orbbec 深度设备")
                self.enable_depth = False
                return

            device = device_list.get_device_by_index(0)
            info = device.get_device_info()
            logger.info(f"Orbbec 设备: {info.get_name()}, SN: {info.get_serial_number()}")

            self._ob_pipeline = Pipeline(device)
            config = Config()

            profile_list = self._ob_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(profile)

            depth_w = profile.get_width()
            depth_h = profile.get_height()
            depth_fps = profile.get_fps()
            logger.info(f"深度流: {depth_w}x{depth_h} @ {depth_fps}fps")

            self._ob_pipeline.start(config)

            # 估算深度内参
            self._depth_intrinsics = CameraIntrinsics(
                fx=depth_w * 0.85,
                fy=depth_w * 0.85,
                cx=depth_w / 2,
                cy=depth_h / 2,
                width=depth_w,
                height=depth_h,
            )

        except Exception as e:
            logger.error(f"深度启动失败: {e}")
            self.enable_depth = False
            self._ob_pipeline = None

    def stop(self):
        """停止相机"""
        if self._cap:
            self._cap.release()
            self._cap = None

        if self._ob_pipeline:
            try:
                self._ob_pipeline.stop()
            except:
                pass
            self._ob_pipeline = None

        self._running = False
        logger.info("相机已停止")

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        读取一帧

        Returns:
            (color_image, depth_image)
            - color_image: BGR uint8, shape (H, W, 3)
            - depth_image: uint16 毫米, shape (H, W) 或 None
        """
        if not self._running:
            raise RuntimeError("相机未启动")

        color = self._read_rgb()
        depth = self._read_depth() if self.enable_depth else None

        return color, depth

    def read_rgb(self) -> Optional[np.ndarray]:
        """只读取 RGB"""
        return self._read_rgb()

    def read_depth(self) -> Optional[np.ndarray]:
        """只读取深度"""
        return self._read_depth()

    def _read_rgb(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        # 如果分辨率不是目标大小，resize
        h, w = frame.shape[:2]
        if w != self.rgb_width or h != self.rgb_height:
            frame = cv2.resize(frame, (self.rgb_width, self.rgb_height))
        return frame

    def _read_depth(self) -> Optional[np.ndarray]:
        if self._ob_pipeline is None:
            return None
        try:
            frames = self._ob_pipeline.wait_for_frames(500)
            if frames is None:
                return None
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                return None
            w = depth_frame.get_width()
            h = depth_frame.get_height()
            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
            return data
        except Exception as e:
            logger.debug(f"深度读取失败: {e}")
            return None

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """深度相机内参"""
        if self._depth_intrinsics:
            return self._depth_intrinsics
        return CameraIntrinsics(
            fx=self.rgb_width * 0.8,
            fy=self.rgb_width * 0.8,
            cx=self.rgb_width / 2,
            cy=self.rgb_height / 2,
            width=self.rgb_width,
            height=self.rgb_height,
        )

    def save_snapshot(self, path: str = "snapshot") -> Tuple[str, Optional[str]]:
        """保存 RGB + 深度快照"""
        color, depth = self.read()
        rgb_path = f"{path}_rgb.jpg"
        depth_path = None

        if color is not None:
            cv2.imwrite(rgb_path, color)

        if depth is not None:
            depth_path = f"{path}_depth.png"
            cv2.imwrite(depth_path, depth)
            # 也保存伪彩色版
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(f"{path}_depth_vis.jpg", depth_color)

        return rgb_path, depth_path

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ─── 工具函数 ────────────────────────────────────────────────

def pixel_to_3d(
    u: int, v: int,
    depth_mm: float,
    intrinsics: CameraIntrinsics,
) -> Tuple[float, float, float]:
    """像素坐标 + 深度 → 相机坐标系 3D 点 (mm)"""
    z = depth_mm
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return (x, y, z)


def get_depth_at(
    depth_image: np.ndarray,
    u: int, v: int,
    kernel_size: int = 5,
) -> float:
    """获取像素点深度值 (邻域中值)"""
    h, w = depth_image.shape
    half = kernel_size // 2
    y1, y2 = max(0, v - half), min(h, v + half + 1)
    x1, x2 = max(0, u - half), min(w, u + half + 1)

    patch = depth_image[y1:y2, x1:x2]
    valid = patch[patch > 0]

    return float(np.median(valid)) if len(valid) > 0 else 0.0
