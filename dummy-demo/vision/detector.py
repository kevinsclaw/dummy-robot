"""
物体检测模块 — 基于颜色分割

检测工作区内的彩色积木，返回位置和颜色信息。

使用:
    detector = ColorDetector()
    detector.add_color("red", (0, 120, 70), (10, 255, 255))    # HSV 范围
    detector.add_color("blue", (100, 120, 70), (130, 255, 255))
    
    objects = detector.detect(color_image)
    for obj in objects:
        print(f"{obj.color} block at pixel ({obj.cx}, {obj.cy}), area={obj.area}")
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """检测到的物体"""
    color: str            # 颜色名称
    cx: int               # 质心 x (像素)
    cy: int               # 质心 y (像素)
    area: float           # 面积 (像素²)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    angle: float = 0.0    # 旋转角度 (度)
    contour: np.ndarray = None  # 轮廓点
    depth_mm: float = 0.0      # 深度 (mm，仅深度相机可用)

    # 世界坐标 (标定后由 calibration 模块填入)
    world_x: float = 0.0
    world_y: float = 0.0
    world_z: float = 0.0

    def __str__(self):
        return (
            f"{self.color} @ pixel({self.cx},{self.cy}) "
            f"area={self.area:.0f} angle={self.angle:.1f}°"
        )


@dataclass
class ColorRange:
    """HSV 颜色范围"""
    name: str
    lower: np.ndarray  # HSV 下限
    upper: np.ndarray  # HSV 上限
    # 红色需要两段范围 (H 跨 0°)
    lower2: Optional[np.ndarray] = None
    upper2: Optional[np.ndarray] = None


# ─── 预定义颜色 (针对彩色积木在白色桌面的场景) ─────────────────

PRESET_COLORS: Dict[str, dict] = {
    "red": {
        "lower": (0, 100, 80),
        "upper": (10, 255, 255),
        "lower2": (170, 100, 80),  # 红色跨 0°
        "upper2": (180, 255, 255),
    },
    "green": {
        "lower": (35, 80, 60),
        "upper": (85, 255, 255),
    },
    "blue": {
        "lower": (100, 100, 60),
        "upper": (130, 255, 255),
    },
    "yellow": {
        "lower": (20, 80, 80),
        "upper": (35, 255, 255),
    },
    "orange": {
        "lower": (10, 100, 80),
        "upper": (20, 255, 255),
    },
    "purple": {
        "lower": (130, 60, 60),
        "upper": (160, 255, 255),
    },
}


class ColorDetector:
    """
    颜色分割检测器

    Args:
        min_area: 最小面积阈值 (像素²，过滤噪点)
        max_area: 最大面积阈值 (像素²，过滤背景)
        blur_kernel: 高斯模糊核大小
        morph_kernel: 形态学核大小
        use_presets: 是否加载预定义颜色
    """

    def __init__(
        self,
        min_area: float = 500,
        max_area: float = 50000,
        blur_kernel: int = 5,
        morph_kernel: int = 5,
        use_presets: bool = True,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel

        self._colors: List[ColorRange] = []

        if use_presets:
            for name, params in PRESET_COLORS.items():
                self.add_color(name, **params)

    def add_color(
        self,
        name: str,
        lower: Tuple[int, int, int],
        upper: Tuple[int, int, int],
        lower2: Optional[Tuple[int, int, int]] = None,
        upper2: Optional[Tuple[int, int, int]] = None,
    ):
        """
        添加颜色范围

        Args:
            name: 颜色名称
            lower: HSV 下限 (H: 0-180, S: 0-255, V: 0-255)
            upper: HSV 上限
            lower2, upper2: 第二段范围 (红色等跨 0° 的颜色)
        """
        color = ColorRange(
            name=name,
            lower=np.array(lower, dtype=np.uint8),
            upper=np.array(upper, dtype=np.uint8),
            lower2=np.array(lower2, dtype=np.uint8) if lower2 else None,
            upper2=np.array(upper2, dtype=np.uint8) if upper2 else None,
        )
        # 替换已有同名颜色
        self._colors = [c for c in self._colors if c.name != name]
        self._colors.append(color)
        logger.debug(f"Color added: {name} {lower}-{upper}")

    def remove_color(self, name: str):
        """移除颜色"""
        self._colors = [c for c in self._colors if c.name != name]

    def detect(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[DetectedObject]:
        """
        检测图像中的彩色物体

        Args:
            color_image: BGR 图像
            depth_image: 深度图 (uint16 mm)，可选
            roi: 感兴趣区域 (x, y, w, h)，仅检测该区域

        Returns:
            检测到的物体列表，按面积从大到小排序
        """
        if color_image is None:
            return []

        # ROI 裁剪
        offset_x, offset_y = 0, 0
        if roi:
            rx, ry, rw, rh = roi
            color_image = color_image[ry:ry+rh, rx:rx+rw]
            if depth_image is not None:
                depth_image = depth_image[ry:ry+rh, rx:rx+rw]
            offset_x, offset_y = rx, ry

        # 预处理
        blurred = cv2.GaussianBlur(color_image, (self.blur_kernel, self.blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        all_objects: List[DetectedObject] = []

        for color in self._colors:
            objects = self._detect_color(
                hsv, color, depth_image, offset_x, offset_y
            )
            all_objects.extend(objects)

        # 按面积排序
        all_objects.sort(key=lambda o: o.area, reverse=True)
        return all_objects

    def detect_specific(
        self,
        color_image: np.ndarray,
        color_name: str,
        depth_image: Optional[np.ndarray] = None,
    ) -> List[DetectedObject]:
        """只检测特定颜色"""
        matching = [c for c in self._colors if c.name == color_name]
        if not matching:
            logger.warning(f"Color '{color_name}' not registered")
            return []

        blurred = cv2.GaussianBlur(color_image, (self.blur_kernel, self.blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        return self._detect_color(hsv, matching[0], depth_image, 0, 0)

    def get_mask(
        self, color_image: np.ndarray, color_name: str
    ) -> Optional[np.ndarray]:
        """获取特定颜色的掩码 (调试用)"""
        matching = [c for c in self._colors if c.name == color_name]
        if not matching:
            return None

        blurred = cv2.GaussianBlur(color_image, (self.blur_kernel, self.blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return self._create_mask(hsv, matching[0])

    def draw_detections(
        self,
        image: np.ndarray,
        objects: List[DetectedObject],
        show_label: bool = True,
        show_bbox: bool = True,
        show_center: bool = True,
    ) -> np.ndarray:
        """
        在图像上绘制检测结果

        Returns:
            标注后的图像 (不修改原图)
        """
        vis = image.copy()

        # 颜色映射
        color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
            "purple": (255, 0, 255),
        }

        for obj in objects:
            draw_color = color_map.get(obj.color, (255, 255, 255))

            if show_bbox:
                x, y, w, h = obj.bbox
                cv2.rectangle(vis, (x, y), (x+w, y+h), draw_color, 2)

            if show_center:
                cv2.circle(vis, (obj.cx, obj.cy), 5, draw_color, -1)
                cv2.drawMarker(vis, (obj.cx, obj.cy), (255, 255, 255),
                               cv2.MARKER_CROSS, 15, 1)

            if show_label:
                label = f"{obj.color}"
                if obj.depth_mm > 0:
                    label += f" {obj.depth_mm:.0f}mm"
                cv2.putText(vis, label, (obj.cx + 10, obj.cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

        return vis

    # ─── 私有方法 ────────────────────────────────────────────

    def _create_mask(self, hsv: np.ndarray, color: ColorRange) -> np.ndarray:
        """创建颜色掩码"""
        mask = cv2.inRange(hsv, color.lower, color.upper)

        if color.lower2 is not None and color.upper2 is not None:
            mask2 = cv2.inRange(hsv, color.lower2, color.upper2)
            mask = cv2.bitwise_or(mask, mask2)

        # 形态学处理: 开运算去噪 + 闭运算填充
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel, self.morph_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _detect_color(
        self,
        hsv: np.ndarray,
        color: ColorRange,
        depth_image: Optional[np.ndarray],
        offset_x: int,
        offset_y: int,
    ) -> List[DetectedObject]:
        """检测单一颜色"""
        mask = self._create_mask(hsv, color)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            # 质心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"]) + offset_x
            cy = int(M["m01"] / M["m00"]) + offset_y

            # 边界框
            bx, by, bw, bh = cv2.boundingRect(contour)
            bx += offset_x
            by += offset_y

            # 最小旋转矩形 → 角度
            rect = cv2.minAreaRect(contour)
            angle = rect[2]

            # 深度
            depth_val = 0.0
            if depth_image is not None:
                local_cx = cx - offset_x
                local_cy = cy - offset_y
                if 0 <= local_cy < depth_image.shape[0] and 0 <= local_cx < depth_image.shape[1]:
                    from .camera import get_depth_at
                    depth_val = get_depth_at(depth_image, local_cx, local_cy)

            obj = DetectedObject(
                color=color.name,
                cx=cx,
                cy=cy,
                area=area,
                bbox=(bx, by, bw, bh),
                angle=angle,
                contour=contour,
                depth_mm=depth_val,
            )
            objects.append(obj)

        return objects


# ─── 调参工具 ────────────────────────────────────────────────

class ColorTuner:
    """
    HSV 颜色调参工具 (带滑动条的 GUI)

    使用:
        tuner = ColorTuner()
        tuner.run(camera)  # 传入 Camera 实例
    """

    def __init__(self):
        self.lower_h = 0
        self.lower_s = 100
        self.lower_v = 80
        self.upper_h = 180
        self.upper_s = 255
        self.upper_v = 255

    def run(self, camera=None, image: np.ndarray = None):
        """
        启动调参 GUI

        Args:
            camera: Camera 实例 (实时调参)
            image: 静态图片 (无相机时用)
        """
        window = "HSV Color Tuner"
        cv2.namedWindow(window)
        cv2.createTrackbar("H_low", window, 0, 180, lambda x: None)
        cv2.createTrackbar("S_low", window, 100, 255, lambda x: None)
        cv2.createTrackbar("V_low", window, 80, 255, lambda x: None)
        cv2.createTrackbar("H_high", window, 180, 180, lambda x: None)
        cv2.createTrackbar("S_high", window, 255, 255, lambda x: None)
        cv2.createTrackbar("V_high", window, 255, 255, lambda x: None)

        print("HSV Color Tuner - 按 'q' 退出, 按 's' 打印当前值")

        while True:
            if camera:
                frame, _ = camera.read()
                if frame is None:
                    continue
            elif image is not None:
                frame = image.copy()
            else:
                break

            h_lo = cv2.getTrackbarPos("H_low", window)
            s_lo = cv2.getTrackbarPos("S_low", window)
            v_lo = cv2.getTrackbarPos("V_low", window)
            h_hi = cv2.getTrackbarPos("H_high", window)
            s_hi = cv2.getTrackbarPos("S_high", window)
            v_hi = cv2.getTrackbarPos("V_high", window)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]),
                               np.array([h_hi, s_hi, v_hi]))

            result = cv2.bitwise_and(frame, frame, mask=mask)

            # 拼接显示
            display = np.hstack([frame, result])
            cv2.imshow(window, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f'lower=({h_lo}, {s_lo}, {v_lo}), upper=({h_hi}, {s_hi}, {v_hi})')

        cv2.destroyAllWindows()
        return (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)
