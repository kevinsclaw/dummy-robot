"""
相机采集模块

支持:
  - 普通 USB 摄像头 (OpenCV VideoCapture)
  - 奥比中光 Gemini 2 深度相机 (pyorbbecsdk)

使用:
  # 普通 USB 摄像头
  cam = Camera(device=0)

  # Gemini 2 深度相机
  cam = Camera(device="gemini2")

  cam.start()
  color, depth = cam.read()   # depth 为 None (USB) 或 np.ndarray (Gemini2)
  cam.stop()
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
    dist_coeffs: np.ndarray = None  # 畸变系数 [k1, k2, p1, p2, k3]

    def __post_init__(self):
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros(5)

    @property
    def matrix(self) -> np.ndarray:
        """3×3 内参矩阵"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


class Camera:
    """
    统一相机接口

    Args:
        device: 设备标识
            - int: OpenCV 设备号 (0, 1, ...)
            - "gemini2": 奥比中光 Gemini 2
        width: 图像宽度
        height: 图像高度
        fps: 帧率
    """

    def __init__(
        self,
        device=0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps

        self._cap = None
        self._ob_pipeline = None
        self._intrinsics = None
        self._running = False

        self._is_gemini = isinstance(device, str) and "gemini" in device.lower()

    def start(self):
        """启动相机"""
        if self._is_gemini:
            self._start_gemini()
        else:
            self._start_opencv()
        self._running = True
        logger.info(f"Camera started: {self.device} ({self.width}x{self.height}@{self.fps})")

    def stop(self):
        """停止相机"""
        if self._is_gemini:
            self._stop_gemini()
        else:
            if self._cap:
                self._cap.release()
                self._cap = None
        self._running = False
        logger.info("Camera stopped")

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        读取一帧

        Returns:
            (color_image, depth_image)
            - color_image: BGR uint8, shape (H, W, 3)
            - depth_image: uint16 毫米, shape (H, W) 或 None
        """
        if not self._running:
            raise RuntimeError("Camera not started")

        if self._is_gemini:
            return self._read_gemini()
        else:
            return self._read_opencv()

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """相机内参 (标定后可用)"""
        if self._intrinsics is None:
            # 返回默认估计值
            self._intrinsics = CameraIntrinsics(
                fx=self.width * 0.8,
                fy=self.width * 0.8,
                cx=self.width / 2,
                cy=self.height / 2,
                width=self.width,
                height=self.height,
            )
        return self._intrinsics

    def set_intrinsics(self, intrinsics: CameraIntrinsics):
        """设置标定后的内参"""
        self._intrinsics = intrinsics

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """去畸变"""
        intr = self.intrinsics
        return cv2.undistort(image, intr.matrix, intr.dist_coeffs)

    # ─── OpenCV 后端 ────────────────────────────────────────

    def _start_opencv(self):
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.device}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 读取实际分辨率
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _read_opencv(self) -> Tuple[Optional[np.ndarray], None]:
        ret, frame = self._cap.read()
        if not ret:
            return None, None
        return frame, None

    # ─── 奥比中光 Gemini 2 后端 ──────────────────────────────

    def _start_gemini(self):
        """启动 Gemini 2 (需要 pyorbbecsdk)"""
        try:
            from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
        except ImportError:
            raise ImportError(
                "pyorbbecsdk not installed. Install with:\n"
                "  pip install pyorbbecsdk\n"
                "Or download from: https://github.com/orbbec/pyorbbecsdk"
            )

        self._ob_pipeline = Pipeline()
        config = Config()

        # 配置彩色流
        color_profiles = self._ob_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(
            self.width, self.height, OBFormat.RGB888, self.fps
        )
        if color_profile is None:
            color_profile = color_profiles.get_default_video_stream_profile()
            logger.warning(f"Requested color profile not available, using default")
        config.enable_stream(color_profile)

        # 配置深度流
        depth_profiles = self._ob_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        # 启用对齐 (深度对齐到彩色)
        config.set_align_mode(True)

        self._ob_pipeline.start(config)

        # 获取内参
        try:
            cam_params = self._ob_pipeline.get_camera_param()
            intr = cam_params.rgb_intrinsic
            self._intrinsics = CameraIntrinsics(
                fx=intr.fx, fy=intr.fy,
                cx=intr.cx, cy=intr.cy,
                width=int(intr.width), height=int(intr.height),
                dist_coeffs=np.array([
                    intr.k1, intr.k2, intr.p1, intr.p2, intr.k3
                ]),
            )
            logger.info(f"Gemini 2 intrinsics: fx={intr.fx:.1f}, fy={intr.fy:.1f}")
        except Exception as e:
            logger.warning(f"Could not get intrinsics: {e}")

    def _stop_gemini(self):
        if self._ob_pipeline:
            self._ob_pipeline.stop()
            self._ob_pipeline = None

    def _read_gemini(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        frameset = self._ob_pipeline.wait_for_frameset(1000)
        if frameset is None:
            return None, None

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        color_image = None
        depth_image = None

        if color_frame:
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_image = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        if depth_frame:
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

        return color_image, depth_image

    # ─── 上下文管理器 ────────────────────────────────────────

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
    """
    像素坐标 + 深度 → 相机坐标系 3D 点 (mm)

    Args:
        u, v: 像素坐标
        depth_mm: 深度值 (毫米)
        intrinsics: 相机内参

    Returns:
        (x, y, z) 相机坐标系下的 3D 坐标 (mm)
    """
    z = depth_mm
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return (x, y, z)


def get_depth_at(
    depth_image: np.ndarray,
    u: int, v: int,
    kernel_size: int = 5,
) -> float:
    """
    获取像素点的深度值 (取邻域中值，更鲁棒)

    Args:
        depth_image: 深度图 (uint16, mm)
        u, v: 像素坐标
        kernel_size: 取值邻域大小

    Returns:
        深度值 (mm), 无效返回 0
    """
    h, w = depth_image.shape
    half = kernel_size // 2
    y1 = max(0, v - half)
    y2 = min(h, v + half + 1)
    x1 = max(0, u - half)
    x2 = min(w, u + half + 1)

    patch = depth_image[y1:y2, x1:x2]
    valid = patch[patch > 0]

    if len(valid) == 0:
        return 0.0

    return float(np.median(valid))
