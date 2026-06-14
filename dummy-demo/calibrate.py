"""
手眼标定工具 (灵活版)
========================

Eye-to-Hand (固定俯拍) 标定:
  像素坐标 (u, v) → 机械臂笛卡尔坐标 (x, y)

方法: N 点仿射变换 (最小二乘), 网格点自动生成
标记: 夹爪夹着一只小黄鱼(黄色), 自动识别

用法:
  # 全自动标定 (推荐, Pi5 headless): 小黄鱼(黄色)自动识别
  python calibrate.py --auto

  # 换环境重新标定: 只需改工作区参数
  python calibrate.py --auto --center 230,0 --size 400,400 --calib-z 150 --grid 4

  # 验证已有标定 (点击画面, 机械臂移到对应位置)
  python calibrate.py --verify

  # 交互标定 (需显示器, 手动点击夹爪尖端)
  python calibrate.py

参数说明:
  --center x,y    工作区中心 (机械臂 XY 坐标 mm), 默认 230,0
  --size 宽,深    工作区尺寸 mm, 默认 400,400
  --calib-z z     标定高度 mm (小黄鱼标记高度), 默认 150
  --grid n        网格每边点数, 3=9点 4=16点, 默认 3
  --camera n      RGB 相机编号, 默认 0 (Pi5 /dev/video0)

依赖:
  pip install numpy opencv-python pyserial
"""

import numpy as np
import cv2
import json
import os
import sys
import logging
import time
from typing import List, Tuple, Optional
from pathlib import Path

# 添加 parent 到路径
sys.path.insert(0, str(Path(__file__).parent))
from driver.dummy_serial import DummySerial
from vision.orbbec_camera import open_camera
try:
    from vision.color_detector import ColorBlockDetector
except Exception:
    ColorBlockDetector = None


def read_rgb_frame(cam):
    """统一读帧 helper: 兼容 OrbbecCamera / cv2.VideoCapture / 旧 Camera。
    返回 BGR 帧或 None。"""
    if cam is None:
        return None
    if hasattr(cam, 'read_color'):
        return cam.read_color()
    if hasattr(cam, 'read_rgb'):
        return cam.read_rgb()
    if hasattr(cam, 'read'):
        ret = cam.read()
        if isinstance(ret, tuple):
            ok, f = ret
            return f if ok else None
        return ret
    return None

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ─── 配置 ──────────────────────────────────────────────────────

# 标定文件路径
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"

# ─── 默认工作区配置 (可被命令行覆盖) ─────────────────────────
# 工作平面中心 (机械臂 XY 坐标, mm)
DEFAULT_CENTER = (230.0, 0.0)
# 工作平面尺寸 (宽 x 深, mm) — 杨光当前约 400x400
DEFAULT_SIZE = (400.0, 400.0)
# 标定时机械臂尖端高度 (mm) — 小黄鱼夹在尖端, 当前约 150mm (15cm)
DEFAULT_CALIB_Z = 150.0
# 抓取高度 (积木顶面高度)
DEFAULT_GRAB_Z = 150.0
# 网格密度 (每边点数, 3 => 9 点, 4 => 16 点)
DEFAULT_GRID = 3
# 姿态 (末端朝下)
DEFAULT_POSE = (0.0, 90.0, 0.0)
# 相机设备号 (Pi5 上 /dev/video0 => 0)
RGB_DEVICE = 0
# 标记颜色 (夹爪夹着的小黄鱼标定物)
MARKER_COLOR = "yellow"


def generate_grid_points(center, size, z, grid=3, pose=DEFAULT_POSE):
    """
    在工作平面上自动生成网格标定点。

    Args:
        center: (cx, cy) 工作区中心 mm
        size: (w, d) 工作区宽深 mm
        z: 标定高度 mm
        grid: 每边点数 (总点数 = grid*grid)
        pose: (a, b, c) 末端姿态
    Returns:
        List[(x, y, z, a, b, c)]
    """
    cx, cy = center
    w, d = size
    a, b, c = pose
    points = []
    # 留 10% 边距, 避免到达工作区边缘极限
    half_w = w / 2 * 0.9
    half_d = d / 2 * 0.9
    for i in range(grid):
        for j in range(grid):
            # 网格均匀分布
            fx = -1 + 2 * i / (grid - 1) if grid > 1 else 0
            fy = -1 + 2 * j / (grid - 1) if grid > 1 else 0
            x = cx + fx * half_d   # X 是机械臂前后 (深)
            y = cy + fy * half_w   # Y 是机械臂左右 (宽)
            points.append((x, y, z, a, b, c))
    # 按距中心远近排序: 从中心点开始, 逐渐向外, 移动更平滑安全
    points.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    return points


def detect_red_marker(frame, detector=None):
    """
    自动检测画面中的标记 (夹爪上的标定物)。
    返回最大标记区域的中心像素 (u, v), 找不到返回 None。
    根据 MARKER_COLOR 选择 HSV 范围。
    """
    import cv2 as _cv2
    import numpy as _np
    # ⚠️ 不用 ColorBlockDetector: 它 min_area=500 会滤掉小黄夹子(≎270px²)。
    #    直接用调好的 HSV 阈值 (H下限=23, 滤掉噪点)。
    hsv = _cv2.cvtColor(_cv2.GaussianBlur(frame, (5, 5), 0), _cv2.COLOR_BGR2HSV)
    if MARKER_COLOR == "yellow":
        mask = _cv2.inRange(hsv, _np.array([23, 80, 80]), _np.array([38, 255, 255]))
    else:  # red
        mask = _cv2.inRange(hsv, _np.array([0, 100, 80]), _np.array([10, 255, 255]))
        mask |= _cv2.inRange(hsv, _np.array([170, 100, 80]), _np.array([180, 255, 255]))
    mask = _cv2.morphologyEx(mask, _cv2.MORPH_OPEN, _np.ones((3, 3), _np.uint8))
    contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=_cv2.contourArea)
    if _cv2.contourArea(biggest) < 150:  # 夹子≎270px², 滤掉 <150 的噪点
        return None
    M = _cv2.moments(biggest)
    if M['m00'] == 0:
        return None
    return (M['m10'] / M['m00'], M['m01'] / M['m00'])


# ─── 标定数据结构 ─────────────────────────────────────────────

class CalibrationData:
    """标定数据。支持两种模式:
      - "2d": 像素(u,v) -> 机械臂(x,y), 2x3 仿射 (Z 写死 grab_z)
      - "3d": 相机系 3D(Xc,Yc,Zc) -> 机械臂基座 3D(x,y,z), 4x4 刚体变换
             (需深度相机, 能抳不同高度/斜放物体)
    """

    def __init__(self, mode: str = "2d"):
        self.mode = mode
        self.pixel_points: List[Tuple[float, float]] = []   # (u, v) — 2d
        self.robot_points: List[Tuple[float, float]] = []   # (x, y) — 2d
        self.cam_points_3d: List[Tuple[float, float, float]] = []    # (Xc,Yc,Zc) — 3d
        self.robot_points_3d: List[Tuple[float, float, float]] = []  # (x,y,z) — 3d
        self.transform_matrix: Optional[np.ndarray] = None  # 2x3 (2d) 或 3x4 (3d)
        self.grab_z: float = DEFAULT_GRAB_Z
        self.error_mm: float = 0.0  # 标定误差

    def add_point(self, pixel: Tuple[float, float], robot: Tuple[float, float]):
        self.pixel_points.append(pixel)
        self.robot_points.append(robot)

    def add_point_3d(self, cam_xyz: Tuple[float, float, float],
                     robot_xyz: Tuple[float, float, float]):
        self.cam_points_3d.append(cam_xyz)
        self.robot_points_3d.append(robot_xyz)

    def compute(self) -> bool:
        """计算变换 (按 mode 分支)"""
        if self.mode == "3d":
            return self._compute_3d()
        return self._compute_2d()

    def _compute_3d(self) -> bool:
        """3D 刚体变换: 相机系 3D -> 机械臂 3D, 用 estimateAffine3D"""
        if len(self.cam_points_3d) < 4:
            logger.error("3D 标定至少需要 4 个点")
            return False
        src = np.array(self.cam_points_3d, dtype=np.float32)
        dst = np.array(self.robot_points_3d, dtype=np.float32)
        # estimateAffine3D 返回 3x4 仿射 (含轻微缩放/剪切, 对深度噪声更鲁棒)
        retval, M, inliers = cv2.estimateAffine3D(src, dst)
        if not retval or M is None:
            logger.error("estimateAffine3D 失败")
            return False
        self.transform_matrix = M  # 3x4
        self._compute_error_3d()
        return True

    def _compute_2d(self) -> bool:
        """计算 2D 仿射变换矩阵"""
        if len(self.pixel_points) < 3:
            logger.error("至少需要 3 个标定点")
            return False

        src = np.array(self.pixel_points, dtype=np.float32)
        dst = np.array(self.robot_points, dtype=np.float32)

        if len(self.pixel_points) == 3:
            self.transform_matrix = cv2.getAffineTransform(src, dst)
        else:
            # 4+ 点用最小二乘
            # 构建方程: [u, v, 1] * M^T = [x, y]
            n = len(self.pixel_points)
            A = np.zeros((n * 2, 6))
            b = np.zeros(n * 2)

            for i in range(n):
                u, v = self.pixel_points[i]
                x, y = self.robot_points[i]
                A[2*i] = [u, v, 1, 0, 0, 0]
                A[2*i+1] = [0, 0, 0, u, v, 1]
                b[2*i] = x
                b[2*i+1] = y

            # 最小二乘解
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            self.transform_matrix = result.reshape(2, 3)

        # 计算误差
        self._compute_error()
        return True

    def _compute_error_3d(self):
        """3D 重投影误差 (欧氏距离 mm)"""
        if self.transform_matrix is None:
            return
        errors = []
        for cam, robot in zip(self.cam_points_3d, self.robot_points_3d):
            pred = self.camera_to_robot(*cam)
            err = np.sqrt(sum((pred[k] - robot[k])**2 for k in range(3)))
            errors.append(err)
        self.error_mm = float(np.mean(errors))

    def camera_to_robot(self, Xc: float, Yc: float, Zc: float) -> Tuple[float, float, float]:
        """相机系 3D 点 -> 机械臂基座 3D 点 (仅 3d 模式)"""
        if self.transform_matrix is None or self.mode != "3d":
            raise RuntimeError("未进行 3D 标定")
        pt = np.array([Xc, Yc, Zc, 1.0])
        r = self.transform_matrix @ pt  # 3x4 @ 4 -> 3
        return (float(r[0]), float(r[1]), float(r[2]))

    def _compute_error(self):
        """计算标定重投影误差"""
        if self.transform_matrix is None:
            return
        errors = []
        for pixel, robot in zip(self.pixel_points, self.robot_points):
            pred = self.pixel_to_robot(pixel[0], pixel[1])
            err = np.sqrt((pred[0] - robot[0])**2 + (pred[1] - robot[1])**2)
            errors.append(err)
        self.error_mm = float(np.mean(errors))

    def pixel_to_robot(self, u: float, v: float) -> Tuple[float, float]:
        """像素坐标 → 机械臂 XY 坐标"""
        if self.transform_matrix is None:
            raise RuntimeError("未标定")
        pt = np.array([u, v, 1.0])
        result = self.transform_matrix @ pt
        return (float(result[0]), float(result[1]))

    def save(self, path: Optional[str] = None):
        """保存标定结果"""
        path = path or str(CALIBRATION_FILE)
        data = {
            'mode': self.mode,
            'pixel_points': self.pixel_points,
            'robot_points': self.robot_points,
            'cam_points_3d': self.cam_points_3d,
            'robot_points_3d': self.robot_points_3d,
            'transform_matrix': self.transform_matrix.tolist(),
            'grab_z': self.grab_z,
            'error_mm': self.error_mm,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"标定数据已保存: {path} (mode={self.mode})")
        logger.info(f"  误差: {self.error_mm:.2f} mm")

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'CalibrationData':
        """加载标定结果"""
        path = path or str(CALIBRATION_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"标定文件不存在: {path}")

        with open(path) as f:
            data = json.load(f)

        cal = cls(mode=data.get('mode', '2d'))
        cal.pixel_points = [tuple(p) for p in data.get('pixel_points', [])]
        cal.robot_points = [tuple(p) for p in data.get('robot_points', [])]
        cal.cam_points_3d = [tuple(p) for p in data.get('cam_points_3d', [])]
        cal.robot_points_3d = [tuple(p) for p in data.get('robot_points_3d', [])]
        cal.transform_matrix = np.array(data['transform_matrix'])
        cal.grab_z = data.get('grab_z', DEFAULT_GRAB_Z)
        cal.error_mm = data.get('error_mm', 0.0)
        return cal


# ─── 交互标定 ─────────────────────────────────────────────────

class InteractiveCalibrator:
    """交互式标定工具"""

    def __init__(self, robot: DummySerial, camera,
                 points: List[Tuple] = None):
        self.robot = robot
        self.camera = camera
        self.points = points or DEFAULT_CALIBRATION_POINTS
        self.calibration = CalibrationData()
        self._click_point: Optional[Tuple[int, int]] = None
        self._window_name = "Calibration - Click gripper tip"

    def run(self) -> CalibrationData:
        """执行标定流程"""
        print("\n" + "=" * 50)
        print("  Dummy V2 手眼标定")
        print("=" * 50)
        print(f"\n将依次移动到 {len(self.points)} 个标定点")
        print("在每个位置，请在画面中 点击夹爪尖端")
        print("按 ESC 取消, 按 R 重新点击当前点\n")

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

        try:
            for i, point in enumerate(self.points):
                x, y, z, a, b, c = point
                print(f"\n--- 标定点 {i+1}/{len(self.points)} ---")
                print(f"  目标: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")

                # 移动机械臂
                print("  移动机械臂...")
                self.robot.move_cartesian(x, y, z, a, b, c)
                time.sleep(1.0)  # 等到位

                # 等待用户点击
                pixel = self._wait_for_click(i + 1, len(self.points))
                if pixel is None:
                    print("\n⚠️ 标定取消")
                    return self.calibration

                self.calibration.add_point(pixel, (x, y))
                print(f"  ✓ 像素 ({pixel[0]}, {pixel[1]}) → 坐标 ({x:.1f}, {y:.1f})")

        finally:
            cv2.destroyAllWindows()

        # 计算变换
        print("\n计算变换矩阵...")
        if self.calibration.compute():
            print(f"✓ 标定完成！误差: {self.calibration.error_mm:.2f} mm")
            self.calibration.save()
        else:
            print("✗ 标定失败")

        return self.calibration

    def _wait_for_click(self, current: int, total: int) -> Optional[Tuple[float, float]]:
        """等待用户在画面中点击"""
        self._click_point = None

        while True:
            frame = read_rgb_frame(self.camera)
            if frame is None:
                time.sleep(0.1)
                continue

            # 绘制信息
            display = frame.copy()
            cv2.putText(display, f"Point {current}/{total} - Click gripper tip",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "R=redo, ESC=cancel",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 绘制已有标定点
            for j, pp in enumerate(self.calibration.pixel_points):
                cv2.circle(display, (int(pp[0]), int(pp[1])), 8, (0, 0, 255), -1)
                cv2.putText(display, str(j+1), (int(pp[0])+10, int(pp[1])-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 绘制当前点击
            if self._click_point:
                cv2.circle(display, self._click_point, 10, (0, 255, 0), 2)
                cv2.drawMarker(display, self._click_point, (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)

            cv2.imshow(self._window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                return None
            elif key == ord('r'):  # 重新点击
                self._click_point = None
            elif key == 13 or key == 32:  # Enter or Space — 确认
                if self._click_point:
                    return (float(self._click_point[0]), float(self._click_point[1]))

            # 点击后自动确认 (1 秒内没按 R 就算确认)
            if self._click_point:
                return (float(self._click_point[0]), float(self._click_point[1]))

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_point = (x, y)
            logger.info(f"点击: ({x}, {y})")


# ─── 自动标定 (小黄鱼/黄色自动识别) ────────────────────────────────

class AutoCalibrator:
    """
    全自动标定: 机械臂移动到每个网格点, 自动检测小黄鱼(黄色)像素位置。
    无需人工点击, 适合无显示器的 Pi5 (headless) 环境。
    """

    def __init__(self, robot, camera, points, detector=None,
                 grab_z=DEFAULT_GRAB_Z, settle=1.5, samples=5,
                 safe_z=280.0, confirm=False, direct_move=False, mode="2d"):
        self.robot = robot
        self.camera = camera
        self.points = points
        self.detector = detector
        self.mode = mode
        self.calibration = CalibrationData(mode=mode)
        self.calibration.grab_z = grab_z
        self.settle = settle      # 移动后稳定等待 (秒)
        self.samples = samples    # 每点采样帧数 (取中值降噪)
        self.safe_z = safe_z      # 安全过渡高度 (点间先抬到这个高度)
        self.confirm = confirm    # 逐点确认模式
        self.direct_move = direct_move  # 直接移动 (点间不抬过渡高度)

    def _pixel_to_cam3d(self, u, v):
        """彩色图像素 (u,v) -> 相机系 3D 点 (mm)。
        像素在彩色图分辨率, 需按比例映射到深度图再查深度+反投影。"""
        cam = self.camera
        if not hasattr(cam, "pixel_depth_to_camera_3d"):
            return None
        depth = cam.latest_depth() if hasattr(cam, "latest_depth") else None
        if depth is None:
            return None
        # 彩色帧分辨率
        frame = self._read_frame()
        if frame is None:
            return None
        ch, cw = frame.shape[:2]
        dh, dw = depth.shape[:2]
        du = u * dw / cw
        dv = v * dh / ch
        return cam.pixel_depth_to_camera_3d(du, dv)


    def _read_frame(self):
        cam = self.camera
        if hasattr(cam, 'read_color'):
            return cam.read_color()
        if hasattr(cam, 'read_rgb'):
            return cam.read_rgb()
        if hasattr(cam, 'read'):
            ret = cam.read()
            if isinstance(ret, tuple):
                ok, f = ret
                return f if ok else None
            return ret
        return None

    def _detect_at_point(self):
        """在当前位置多帧采样, 返回中值像素坐标"""
        us, vs = [], []
        for _ in range(self.samples):
            frame = self._read_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            pt = detect_red_marker(frame, self.detector)
            if pt is not None:
                us.append(pt[0])
                vs.append(pt[1])
            time.sleep(0.05)
        if not us:
            return None
        return (float(np.median(us)), float(np.median(vs)))

    def run(self):
        print("\n" + "=" * 50)
        print("  Dummy V2 全自动手眼标定 (逐点安全模式)")
        print("=" * 50)
        print(f"\n标定点: {len(self.points)} 个网格点 (从中心向外)")
        print(f"标记颜色: {MARKER_COLOR} (夹爪上的标定物)")
        print(f"安全过渡高度: Z={self.safe_z:.0f}mm")
        print(f"逐点确认: {'是' if self.confirm else '否'}")
        print(f"每点采样: {self.samples} 帧\n")
        sys.stdout.flush()

        failed = []
        for i, point in enumerate(self.points):
            x, y, z, a, b, c = point
            print(f"--- 点 {i+1}/{len(self.points)}: X={x:.0f} Y={y:.0f} Z={z:.0f} ---")
            sys.stdout.flush()

            # 逐点确认 (在移动前, 更安全)
            if self.confirm:
                print(f"  [确认] 即将移动到 X={x:.0f} Y={y:.0f} Z={z:.0f}")
                print(f"  输入 y/回车=继续, s=跳过, q=退出: ", end="", flush=True)
                try:
                    ans = input().strip().lower()
                except EOFError:
                    # 非交互环境 (如 SSH 后台): 不能逐点停, 直接中止
                    print("\n  ⚠️ 检测到非交互终端, --confirm 需要在 Pi5 本地运行! 中止。")
                    break
                if ans == "q":
                    print("  用户退出标定")
                    break
                if ans == "s":
                    print("  跳过此点")
                    failed.append(i + 1)
                    continue

            # 移动到标定点
            if getattr(self, 'direct_move', False):
                # 直接移动: 点间直接到目标 (x,y,z), 不抬过渡高度
                self.robot.move_cartesian(x, y, z, a, b, c)
                time.sleep(self.settle)
            else:
                # 先抬到安全高度 (同 X,Y 但高 Z), 避免贴桌面横扫
                self.robot.move_cartesian(x, y, self.safe_z, a, b, c)
                time.sleep(self.settle)
                # 再下降到标定高度
                self.robot.move_cartesian(x, y, z, a, b, c)
                time.sleep(self.settle)

            pixel = self._detect_at_point()
            if pixel is None:
                print(f"  ⚠️ 未检测到 {MARKER_COLOR} 标记, 跳过")
                sys.stdout.flush()
                failed.append(i + 1)
                continue

            if self.mode == "3d":
                cam3d = self._pixel_to_cam3d(pixel[0], pixel[1])
                if cam3d is None:
                    print(f"  ⚠️ 该点无深度/无内参, 跳过 (像素 {pixel[0]:.0f},{pixel[1]:.0f})")
                    sys.stdout.flush()
                    failed.append(i + 1)
                    continue
                # 机械臂端用真实 3D (x,y,z)
                self.calibration.add_point_3d(cam3d, (x, y, z))
                # 同时保留 2D 点备查
                self.calibration.add_point(pixel, (x, y))
                print(f"  ✓ 像素({pixel[0]:.0f},{pixel[1]:.0f}) 深度系({cam3d[0]:.0f},{cam3d[1]:.0f},{cam3d[2]:.0f}) → 机械臂({x:.0f},{y:.0f},{z:.0f})")
            else:
                self.calibration.add_point(pixel, (x, y))
                print(f"  ✓ 像素 ({pixel[0]:.0f}, {pixel[1]:.0f}) → 坐标 ({x:.0f}, {y:.0f})")
            sys.stdout.flush()

        # 标定结束, 抬回安全高度 (直接移动模式下不抬, 保持在标定平面)
        try:
            last = self.points[0]
            if not getattr(self, 'direct_move', False):
                self.robot.move_cartesian(last[0], last[1], self.safe_z, last[3], last[4], last[5])
        except Exception:
            pass

        if len(self.calibration.pixel_points) < 3:
            print(f"\n✗ 有效点不足 ({len(self.calibration.pixel_points)}), 标定失败")
            print("  检查: 标记是否在相机视野内? 光照是否充足?")
            return self.calibration

        print("\n计算变换矩阵...")
        if self.calibration.compute():
            print(f"✓ 标定完成! 有效点 {len(self.calibration.pixel_points)}, "
                  f"误差 {self.calibration.error_mm:.2f} mm")
            if failed:
                print(f"  跳过的点: {failed}")
            self.calibration.save()
        else:
            print("✗ 标定失败")
        return self.calibration


# ─── 验证工具 ─────────────────────────────────────────────────

def verify_calibration(robot: DummySerial, camera):
    """验证标定精度"""
    cal = CalibrationData.load()
    print(f"\n已加载标定 (误差: {cal.error_mm:.2f} mm)")
    print("在画面中点击任意点，机械臂将移到对应位置")
    print("按 ESC 退出\n")

    window = "Verify Calibration - Click to move"
    click_point = [None]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point[0] = (x, y)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    try:
        while True:
            frame = read_rgb_frame(camera)
            if frame is None:
                time.sleep(0.1)
                continue

            display = frame.copy()
            cv2.putText(display, "Click to move arm | ESC to exit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if click_point[0]:
                u, v = click_point[0]
                if cal.mode == "3d" and hasattr(camera, "pixel_depth_to_camera_3d"):
                    xyz = pixel_depth_to_robot(u, v, camera, cal)
                    if xyz is None:
                        print(f"  像素 ({u}, {v}) 无深度, 跳过")
                        click_point[0] = None
                    else:
                        x, y, z = xyz
                        cv2.circle(display, (u, v), 10, (0, 255, 0), 2)
                        cv2.putText(display, f"({x:.0f},{y:.0f},{z:.0f})",
                                    (u + 15, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        print(f"  像素 ({u}, {v}) → 3D 坐标 ({x:.1f}, {y:.1f}, {z:.1f})")
                        robot.move_cartesian(x, y, z, 0, 90, 0)
                        click_point[0] = None
                else:
                    x, y = cal.pixel_to_robot(u, v)
                    cv2.circle(display, (u, v), 10, (0, 255, 0), 2)
                    cv2.putText(display, f"({x:.1f}, {y:.1f})",
                                (u + 15, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    print(f"  像素 ({u}, {v}) → 坐标 ({x:.1f}, {y:.1f}, {cal.grab_z:.1f})")
                    robot.move_cartesian(x, y, cal.grab_z, 0, 90, 0)
                    click_point[0] = None

            cv2.imshow(window, display)
            if cv2.waitKey(30) & 0xFF == 27:
                break
    finally:
        cv2.destroyAllWindows()


# ─── 辅助函数 (给 demo 调用) ──────────────────────────────────

def load_calibration(path: Optional[str] = None) -> CalibrationData:
    """加载标定数据 (供外部模块调用)"""
    return CalibrationData.load(path)


def pixel_to_robot_xy(u: float, v: float, cal: Optional[CalibrationData] = None) -> Tuple[float, float]:
    """像素→机器人XY (便捷函数, 2D 模式)"""
    if cal is None:
        cal = CalibrationData.load()
    return cal.pixel_to_robot(u, v)


def pixel_depth_to_robot(u: float, v: float, camera,
                         cal: Optional[CalibrationData] = None):
    """
    3D 模式便捷函数: 彩色图像素 (u,v) + 深度相机 -> 机械臂基座 3D (x,y,z)。
    供 demo/agent 调用: 抳取前把检测到的像素转成真实三维抓取点。

    Returns:
        (x, y, z) mm 或 None (无深度/无内参/未 3D 标定)
    """
    if cal is None:
        cal = CalibrationData.load()
    if cal.mode != "3d" or not hasattr(camera, "pixel_depth_to_camera_3d"):
        return None
    depth = camera.latest_depth() if hasattr(camera, "latest_depth") else None
    if depth is None:
        return None
    # 彩色像素 -> 深度像素 (按分辨率比例)
    frame = camera.read_color() if hasattr(camera, "read_color") else None
    if frame is not None:
        ch, cw = frame.shape[:2]
        dh, dw = depth.shape[:2]
        du = u * dw / cw
        dv = v * dh / ch
    else:
        du, dv = u, v
    cam3d = camera.pixel_depth_to_camera_3d(du, dv)
    if cam3d is None:
        return None
    return cal.camera_to_robot(*cam3d)


# ─── 主入口 ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dummy V2 手眼标定")
    parser.add_argument('--verify', action='store_true', help='验证已有标定')
    parser.add_argument('--auto', action='store_true',
                        help='全自动标定 (小黄鱼/黄色识别, 无需点击, 适合 Pi5)')
    parser.add_argument('--points', type=str, help='自定义标定点 JSON 文件')
    parser.add_argument('--camera', type=int, default=RGB_DEVICE, help='RGB 相机编号')
    parser.add_argument('--grab-z', type=float, default=DEFAULT_GRAB_Z, help='抓取 Z 高度 (mm)')
    parser.add_argument('--center', type=str, default=None,
                        help='工作区中心 "x,y" mm, 默认 230,0')
    parser.add_argument('--size', type=str, default=None,
                        help='工作区尺寸 "宽,深" mm, 默认 400,400')
    parser.add_argument('--calib-z', type=float, default=DEFAULT_CALIB_Z,
                        help='标定高度 mm (小黄鱼标记高度), 默认 150')
    parser.add_argument('--grid', type=int, default=DEFAULT_GRID,
                        help='网格每边点数, 3=9点 4=16点, 默认 3')
    parser.add_argument('--samples', type=int, default=5,
                        help='自动模式每点采样帧数, 默认 5')
    parser.add_argument('--confirm', action='store_true',
                        help='逐点确认模式 (每点暂停等输入 y/s/q)')
    parser.add_argument('--safe-z', type=float, default=280.0,
                        help='安全过渡高度 mm, 默认 280')
    parser.add_argument('--direct', action='store_true',
                        help='直接移动: 点间直接到目标不抬过渡高度 (忽略 --safe-z)')
    parser.add_argument('--3d', dest='use_3d', action='store_true',
                        help='3D 手眼标定 (需深度相机): 像素+深度反投影成相机系 3D, 解 4x4 刚体变换, 能抳不同高度物体')
    args = parser.parse_args()

    # 解析工作区参数
    center = DEFAULT_CENTER
    if args.center:
        _cx, _cy = args.center.split(',')
        center = (float(_cx), float(_cy))
    size = DEFAULT_SIZE
    if args.size:
        _sw, _sd = args.size.split(',')
        size = (float(_sw), float(_sd))

    # 初始化硬件
    print("连接机械臂...")
    robot = DummySerial()
    if not robot.connect():
        print("错误: 无法连接机械臂")
        sys.exit(1)

    print("启动相机...")
    camera, cam_label = open_camera()
    if camera is None:
        print("错误: 未检测到相机")
        robot.disconnect()
        sys.exit(1)
    print(f"📷 {cam_label}")

    try:
        # 机械臂回零
        print("机械臂回零...")
        robot.home()
        robot.enable()

        if args.verify:
            verify_calibration(robot, camera)
        elif args.auto:
            # 全自动标定: 生成网格点 + 小黄鱼(黄色)识别
            if args.points:
                with open(args.points) as f:
                    points = [tuple(p) for p in json.load(f)]
            else:
                points = generate_grid_points(center, size, args.calib_z, args.grid)
            print(f"工作区: 中心={center} 尺寸={size} 高度={args.calib_z}mm 网格={args.grid}x{args.grid}")
            detector = ColorBlockDetector() if ColorBlockDetector else None
            cal_mode = "3d" if args.use_3d else "2d"
            if cal_mode == "3d" and not hasattr(camera, "pixel_depth_to_camera_3d"):
                print("⚠️  --3d 需要深度相机 (Gemini 335), 当前相机不支持, 回退 2D 模式")
                cal_mode = "2d"
            print(f"标定模式: {cal_mode.upper()}")
            cal = AutoCalibrator(robot, camera, points, detector,
                                 grab_z=args.grab_z, samples=args.samples,
                                 safe_z=args.safe_z, confirm=args.confirm,
                                 direct_move=args.direct, mode=cal_mode)
            cal.run()
        else:
            # 交互标定 (需要显示器 + 鼠标点击)
            if args.points:
                with open(args.points) as f:
                    points = [tuple(p) for p in json.load(f)]
            else:
                # 默认也用网格生成, 而非写死的 4 点
                points = generate_grid_points(center, size, args.calib_z, args.grid)

            # 设置 grab_z
            calibrator = InteractiveCalibrator(robot, camera, points)
            calibrator.calibration.grab_z = args.grab_z
            calibrator.run()

    finally:
        if hasattr(camera, 'stop'):
            camera.stop()
        elif hasattr(camera, 'release'):
            camera.release()
        robot.disconnect()


if __name__ == '__main__':
    main()
