"""
相机采集模块
============

支持:
  1. USB 摄像头 (通过 OpenCV)
  2. Orbbec 深度相机 (通过 pyorbbecsdk v1) — RGB + Depth

Pi5 上运行时自动选择可用的相机源。
"""

import time
import logging
import numpy as np
import cv2
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class Camera:
    """
    统一相机接口
    
    Args:
        rgb_device: OpenCV 设备索引或路径 (如 0, "/dev/video0")
        enable_depth: 是否启用 Orbbec 深度相机
        width: 期望宽度
        height: 期望高度
        fps: 期望帧率
    """
    
    def __init__(
        self,
        rgb_device=0,
        enable_depth: bool = False,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.rgb_device = rgb_device
        self.enable_depth = enable_depth
        self.width = width
        self.height = height
        self.fps = fps
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._orbbec_pipe = None
        self._started = False
    
    def start(self) -> bool:
        """启动相机"""
        success = False
        
        if self.enable_depth:
            success = self._start_orbbec()
        
        if not success:
            # Fallback 到 OpenCV USB 摄像头
            success = self._start_opencv()
        
        self._started = success
        return success
    
    def stop(self):
        """停止相机"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        if self._orbbec_pipe is not None:
            self._orbbec_pipe.stop()
            self._orbbec_pipe = None
        
        self._started = False
        logger.info("Camera stopped")
    
    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        读取一帧。
        
        Returns:
            (color_bgr, depth_uint16_mm) — depth 可能为 None
        """
        if not self._started:
            return None, None
        
        if self._orbbec_pipe:
            return self._read_orbbec()
        else:
            return self._read_opencv()
    
    def read_color(self) -> Optional[np.ndarray]:
        """只读取 RGB 帧"""
        color, _ = self.read()
        return color
    
    def save_snapshot(self, prefix: str = "snapshot") -> Tuple[Optional[str], Optional[str]]:
        """保存快照到文件"""
        color, depth = self.read()
        
        rgb_path = None
        depth_path = None
        
        if color is not None:
            rgb_path = f"/tmp/{prefix}_color.jpg"
            cv2.imwrite(rgb_path, color)
            logger.info(f"Saved: {rgb_path}")
        
        if depth is not None:
            depth_path = f"/tmp/{prefix}_depth.npy"
            np.save(depth_path, depth)
            logger.info(f"Saved: {depth_path}")
        
        return rgb_path, depth_path
    
    # ─── OpenCV USB 摄像头 ───────────────────────────────────
    
    def _start_opencv(self) -> bool:
        """启动 OpenCV 摄像头"""
        try:
            if isinstance(self.rgb_device, str):
                self._cap = cv2.VideoCapture(self.rgb_device)
            else:
                self._cap = cv2.VideoCapture(self.rgb_device)
            
            if not self._cap.isOpened():
                logger.error(f"Failed to open camera {self.rgb_device}")
                return False
            
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 读一帧确认
            ret, frame = self._cap.read()
            if ret:
                h, w = frame.shape[:2]
                logger.info(f"OpenCV camera started: {w}x{h} @ device={self.rgb_device}")
                return True
            else:
                logger.error("Camera opened but can't read frames")
                return False
                
        except Exception as e:
            logger.error(f"OpenCV camera error: {e}")
            return False
    
    def _read_opencv(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从 OpenCV 摄像头读取"""
        if self._cap is None:
            return None, None
        ret, frame = self._cap.read()
        if ret:
            return frame, None
        return None, None
    
    # ─── Orbbec 深度相机 (pyorbbecsdk v1) ───────────────────
    
    def _start_orbbec(self) -> bool:
        """启动 Orbbec 深度相机"""
        try:
            import pyorbbecsdk as sdk
            
            ctx = sdk.Context()
            dl = ctx.query_devices()
            
            if dl.get_count() == 0:
                logger.warning("No Orbbec device found")
                return False
            
            dev = dl.get_device_by_index(0)
            info = dev.get_device_info()
            logger.info(f"Orbbec device: {info.get_name()} (SN: {info.get_serial_number()})")
            
            pipe = sdk.Pipeline(dev)
            cfg = sdk.Config()
            
            # 启用 depth
            try:
                dp = pipe.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR)
                cfg.enable_stream(dp.get_default_video_stream_profile())
                logger.info("Depth stream enabled")
            except Exception as e:
                logger.warning(f"Depth not available: {e}")
            
            # 启用 color
            try:
                cp = pipe.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
                cfg.enable_stream(cp.get_default_video_stream_profile())
                logger.info("Color stream enabled")
            except Exception as e:
                logger.warning(f"Color not available: {e}")
            
            pipe.start(cfg)
            self._orbbec_pipe = pipe
            self._orbbec_sdk = sdk
            
            # 预热 (丢弃前几帧)
            for _ in range(10):
                pipe.wait_for_frames(500)
            
            logger.info("Orbbec camera started ✅")
            return True
            
        except ImportError:
            logger.warning("pyorbbecsdk not installed, falling back to OpenCV")
            return False
        except Exception as e:
            logger.error(f"Orbbec init error: {e}")
            return False
    
    def _read_orbbec(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从 Orbbec 读取 color + depth"""
        try:
            frameset = self._orbbec_pipe.wait_for_frames(1000)
            if frameset is None:
                return None, None
            
            color = None
            depth = None
            
            # Color frame (MJPG → BGR)
            cf = frameset.get_color_frame()
            if cf:
                data = bytes(cf.get_data())
                color = cv2.imdecode(
                    np.frombuffer(data, dtype=np.uint8), 
                    cv2.IMREAD_COLOR
                )
            
            # Depth frame (uint16 mm)
            df = frameset.get_depth_frame()
            if df:
                w, h = df.get_width(), df.get_height()
                depth = np.frombuffer(
                    df.get_data(), dtype=np.uint16
                ).reshape(h, w)
            
            return color, depth
            
        except Exception as e:
            logger.error(f"Orbbec read error: {e}")
            return None, None
    
    # ─── 上下文管理器 ────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
