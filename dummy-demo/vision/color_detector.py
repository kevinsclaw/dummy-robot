"""
HSV 颜色方块检测器 (RGB-only, 不需要深度相机)
================================================

纯 OpenCV 实现，通过 HSV 颜色分割检测彩色方块。
配合固定相机高度 + 已知物体大小，可以估算世界坐标。

适用场景:
  - Pi5 + USB 摄像头 (固定俯视)
  - 不需要 Hailo (纯 CPU 够用, 640x480 @ 30fps)
  - 后续可选 Hailo 加速更复杂的模型

使用:
    detector = ColorBlockDetector(camera_height_mm=300)
    frame = cv2.imread("test.jpg")
    blocks = detector.detect(frame)
    # → [Block(color="red", cx=320, cy=240, world_x=150, world_y=80, ...)]
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """检测到的方块"""
    color: str          # 颜色名 ("red", "blue", ...)
    cx: int             # 中心 x (像素)
    cy: int             # 中心 y (像素)
    width_px: int       # 宽 (像素)
    height_px: int      # 高 (像素)
    area_px: int        # 面积 (像素)
    angle: float        # 旋转角度 (度)
    contour: np.ndarray # 轮廓点
    # 世界坐标 (如果标定过)
    world_x: float = 0.0   # mm
    world_y: float = 0.0   # mm
    world_z: float = 0.0   # mm (桌面高度)


# ─── HSV 颜色范围 ────────────────────────────────────────────
# 🔧 需要根据实际光照条件调整！
# 建议用 color_tuner() 在现场标定

DEFAULT_COLOR_RANGES = {
    # red 禁用: 机械臂本身是红色, 会持续误检
    # "red": [
    #     ((0, 120, 80), (8, 255, 255)),
    #     ((172, 120, 80), (180, 255, 255)),
    # ],
    "blue": [
        ((100, 120, 60), (130, 255, 255)),
    ],
    "green": [
        ((40, 80, 60), (80, 255, 255)),
    ],
    "yellow": [
        ((22, 30, 100), (35, 255, 255)),
    ],
    "orange": [
        ((8, 150, 100), (22, 255, 255)),
    ],
    "purple": [
        ((130, 80, 60), (160, 255, 255)),
    ],
}


class ColorBlockDetector:
    """
    HSV 颜色方块检测器
    
    Args:
        color_ranges: HSV 颜色范围字典
        min_area: 最小面积 (像素²)，过滤噪声
        max_area: 最大面积 (像素²)，过滤误检
        camera_height_mm: 相机到桌面高度 (mm)，用于坐标转换
        fov_h_deg: 水平视场角 (度)
        image_width: 图像宽度 (像素)
        image_height: 图像高度 (像素)
    """
    
    def __init__(
        self,
        color_ranges: dict = None,
        min_area: int = 500,
        max_area: int = 50000,
        camera_height_mm: float = 300.0,
        fov_h_deg: float = 60.0,
        image_width: int = 640,
        image_height: int = 480,
    ):
        self.color_ranges = color_ranges or DEFAULT_COLOR_RANGES
        self.min_area = min_area
        self.max_area = max_area
        self.camera_height_mm = camera_height_mm
        self.fov_h_deg = fov_h_deg
        self.image_width = image_width
        self.image_height = image_height
        
        # 像素→mm 转换系数 (简单针孔模型)
        # 实际使用时应该通过标定获得
        fov_h_rad = np.radians(fov_h_deg)
        self.mm_per_pixel = (2 * camera_height_mm * np.tan(fov_h_rad / 2)) / image_width
        
        # 手眼标定矩阵 (像素→机器人坐标)
        # 🔧 需要现场标定！默认用简单仿射变换
        self._calibration_matrix = None
    
    def detect(self, frame: np.ndarray) -> List[Block]:
        """
        检测图像中所有彩色方块。
        
        Args:
            frame: BGR 图像
        
        Returns:
            检测到的方块列表
        """
        if frame is None or frame.size == 0:
            return []
        
        # 高斯模糊减噪
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        blocks = []
        
        for color_name, ranges in self.color_ranges.items():
            # 合并多个 HSV 范围的 mask
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                m = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = cv2.bitwise_or(mask, m)
            
            # 形态学操作 (去噪 + 填充)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area or area > self.max_area:
                    continue
                
                # 最小外接矩形
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                
                # 方块的宽高比检查 (应该接近 1:1)
                aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                if aspect < 0.5:  # 太细长的不是方块
                    continue
                
                # 计算世界坐标
                world_x, world_y = self._pixel_to_world(int(cx), int(cy))
                
                blocks.append(Block(
                    color=color_name,
                    cx=int(cx),
                    cy=int(cy),
                    width_px=int(max(w, h)),
                    height_px=int(min(w, h)),
                    area_px=int(area),
                    angle=angle,
                    contour=cnt,
                    world_x=world_x,
                    world_y=world_y,
                    world_z=0.0,  # 桌面
                ))
        
        # 按面积降序排列 (最大的最可能是真实目标)
        blocks.sort(key=lambda b: b.area_px, reverse=True)
        
        logger.info(f"Detected {len(blocks)} blocks: {[f'{b.color}@({b.cx},{b.cy})' for b in blocks]}")
        return blocks
    
    def detect_specific(self, frame: np.ndarray, color: str) -> List[Block]:
        """只检测特定颜色的方块"""
        all_blocks = self.detect(frame)
        return [b for b in all_blocks if b.color == color]
    
    # ─── 坐标转换 ────────────────────────────────────────────
    
    def _pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """
        像素坐标 → 机器人世界坐标 (mm)
        
        如果有标定矩阵用标定矩阵，否则用简单的针孔模型估算。
        """
        if self._calibration_matrix is not None:
            # 仿射变换
            pt = np.array([px, py, 1.0])
            world = self._calibration_matrix @ pt
            return float(world[0]), float(world[1])
        
        # 简单估算: 图像中心 = 机器人正前方
        # 🔧 这只是粗略估算，实际需要标定
        cx = self.image_width / 2
        cy = self.image_height / 2
        
        # 像素偏移 → mm 偏移
        dx_mm = (px - cx) * self.mm_per_pixel
        dy_mm = (cy - py) * self.mm_per_pixel  # y 轴翻转
        
        # 假设相机在机器人正上方，光轴指向桌面
        # world_x = 机器人前方 (对应图像上方)
        # world_y = 机器人左侧 (对应图像左侧)
        # 🔧 实际偏移量需要现场测量
        robot_x_offset = 200.0  # 相机到机器人基座的前方偏移 (mm)
        robot_y_offset = 0.0    # 相机到机器人基座的侧向偏移 (mm)
        
        world_x = robot_x_offset + dy_mm
        world_y = robot_y_offset - dx_mm
        
        return world_x, world_y
    
    def set_calibration(self, src_points: List[Tuple[int, int]], 
                        dst_points: List[Tuple[float, float]]):
        """
        设置手眼标定 (4 点仿射变换)
        
        Args:
            src_points: 4 个像素坐标 [(px1,py1), ...]
            dst_points: 对应的机器人世界坐标 [(wx1,wy1), ...]
        """
        src = np.float32(src_points)
        dst = np.float32(dst_points)
        
        # 计算仿射变换矩阵 (需要至少 3 点)
        if len(src_points) >= 3:
            self._calibration_matrix = cv2.getAffineTransform(
                src[:3], dst[:3]
            )
            # 如果有 4 点，用透视变换更精确
            if len(src_points) >= 4:
                M = cv2.getPerspectiveTransform(
                    np.float32([[p[0], p[1]] for p in src_points[:4]]),
                    np.float32([[p[0], p[1]] for p in dst_points[:4]])
                )
                # 转为 3x3 仿射近似 (忽略透视分量)
                self._calibration_matrix = M[:2, :]
            
            logger.info("Calibration set with {} points".format(len(src_points)))
    
    # ─── 可视化 ──────────────────────────────────────────────
    
    def draw_detections(self, frame: np.ndarray, blocks: List[Block]) -> np.ndarray:
        """在图像上画出检测结果"""
        vis = frame.copy()
        
        color_bgr = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
            "purple": (255, 0, 255),
        }
        
        for block in blocks:
            bgr = color_bgr.get(block.color, (255, 255, 255))
            
            # 画旋转矩形
            rect = ((block.cx, block.cy), 
                    (block.width_px, block.height_px), 
                    block.angle)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(vis, [box], 0, bgr, 2)
            
            # 画中心点
            cv2.circle(vis, (block.cx, block.cy), 5, bgr, -1)
            
            # 标签
            label = f"{block.color} ({block.world_x:.0f},{block.world_y:.0f})"
            cv2.putText(vis, label, (block.cx - 40, block.cy - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
        
        return vis
    
    # ─── 颜色调参工具 ────────────────────────────────────────
    
    @staticmethod
    def color_tuner(frame: np.ndarray):
        """
        交互式 HSV 颜色调参器。
        用 trackbar 调整 HSV 范围，实时预览 mask。
        
        Args:
            frame: 参考图像 (BGR)
        """
        cv2.namedWindow("Color Tuner", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        
        cv2.createTrackbar("H_lo", "Color Tuner", 0, 180, lambda x: None)
        cv2.createTrackbar("S_lo", "Color Tuner", 100, 255, lambda x: None)
        cv2.createTrackbar("V_lo", "Color Tuner", 80, 255, lambda x: None)
        cv2.createTrackbar("H_hi", "Color Tuner", 180, 180, lambda x: None)
        cv2.createTrackbar("S_hi", "Color Tuner", 255, 255, lambda x: None)
        cv2.createTrackbar("V_hi", "Color Tuner", 255, 255, lambda x: None)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        while True:
            h_lo = cv2.getTrackbarPos("H_lo", "Color Tuner")
            s_lo = cv2.getTrackbarPos("S_lo", "Color Tuner")
            v_lo = cv2.getTrackbarPos("V_lo", "Color Tuner")
            h_hi = cv2.getTrackbarPos("H_hi", "Color Tuner")
            s_hi = cv2.getTrackbarPos("S_hi", "Color Tuner")
            v_hi = cv2.getTrackbarPos("V_hi", "Color Tuner")
            
            mask = cv2.inRange(hsv, 
                              np.array([h_lo, s_lo, v_lo]),
                              np.array([h_hi, s_hi, v_hi]))
            
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            cv2.imshow("Color Tuner", result)
            cv2.imshow("Mask", mask)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key == ord('q'):
                break
        
        print(f"Selected range: (({h_lo}, {s_lo}, {v_lo}), ({h_hi}, {s_hi}, {v_hi}))")
        cv2.destroyAllWindows()
