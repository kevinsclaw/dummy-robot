"""
手眼标定工具
============

Eye-to-Hand (固定俯拍) 标定:
  像素坐标 (u, v) → 机械臂笛卡尔坐标 (x, y)

方法: 4 点仿射变换
  1. 机械臂移动到 4 个预设标定点
  2. 用户在相机画面中点击夹爪尖端
  3. 计算仿射变换矩阵
  4. 保存到 calibration.json

使用:
  python calibrate.py              # 交互标定
  python calibrate.py --verify     # 验证已有标定
  python calibrate.py --auto       # 自动标定 (需标定板)

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
from vision.camera import Camera

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ─── 配置 ──────────────────────────────────────────────────────

# 标定文件路径
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"

# 标定用的 4 个机械臂位置 (笛卡尔坐标)
# 在工作平面上均匀分布的 4 个点
# 这些值需要根据实际工作空间调整
DEFAULT_CALIBRATION_POINTS = [
    # (x, y, z, a, b, c) — 笛卡尔坐标
    # 以 HOME 位为参考: X=227.5, Y=0, Z=324.5
    # 降低 Z 到积木平面 (大约 150mm 高)
    (180.0, -80.0, 150.0, 0.0, 90.0, 0.0),   # 左前
    (180.0,  80.0, 150.0, 0.0, 90.0, 0.0),   # 右前
    (280.0,  80.0, 150.0, 0.0, 90.0, 0.0),   # 右后
    (280.0, -80.0, 150.0, 0.0, 90.0, 0.0),   # 左后
]

# 抓取高度 (积木顶面高度)
DEFAULT_GRAB_Z = 150.0

# 相机设备号
RGB_DEVICE = 1


# ─── 标定数据结构 ─────────────────────────────────────────────

class CalibrationData:
    """标定数据"""

    def __init__(self):
        self.pixel_points: List[Tuple[float, float]] = []   # (u, v)
        self.robot_points: List[Tuple[float, float]] = []   # (x, y)
        self.transform_matrix: Optional[np.ndarray] = None  # 2x3 仿射矩阵
        self.grab_z: float = DEFAULT_GRAB_Z
        self.error_mm: float = 0.0  # 标定误差

    def add_point(self, pixel: Tuple[float, float], robot: Tuple[float, float]):
        self.pixel_points.append(pixel)
        self.robot_points.append(robot)

    def compute(self) -> bool:
        """计算仿射变换矩阵"""
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
            'pixel_points': self.pixel_points,
            'robot_points': self.robot_points,
            'transform_matrix': self.transform_matrix.tolist(),
            'grab_z': self.grab_z,
            'error_mm': self.error_mm,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"标定数据已保存: {path}")
        logger.info(f"  误差: {self.error_mm:.2f} mm")

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'CalibrationData':
        """加载标定结果"""
        path = path or str(CALIBRATION_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"标定文件不存在: {path}")

        with open(path) as f:
            data = json.load(f)

        cal = cls()
        cal.pixel_points = [tuple(p) for p in data['pixel_points']]
        cal.robot_points = [tuple(p) for p in data['robot_points']]
        cal.transform_matrix = np.array(data['transform_matrix'])
        cal.grab_z = data.get('grab_z', DEFAULT_GRAB_Z)
        cal.error_mm = data.get('error_mm', 0.0)
        return cal


# ─── 交互标定 ─────────────────────────────────────────────────

class InteractiveCalibrator:
    """交互式标定工具"""

    def __init__(self, robot: DummySerial, camera: Camera,
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
            frame = self.camera.read_rgb()
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


# ─── 验证工具 ─────────────────────────────────────────────────

def verify_calibration(robot: DummySerial, camera: Camera):
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
            frame = camera.read_rgb()
            if frame is None:
                time.sleep(0.1)
                continue

            display = frame.copy()
            cv2.putText(display, "Click to move arm | ESC to exit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if click_point[0]:
                u, v = click_point[0]
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
    """像素→机器人XY (便捷函数)"""
    if cal is None:
        cal = CalibrationData.load()
    return cal.pixel_to_robot(u, v)


# ─── 主入口 ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dummy V2 手眼标定")
    parser.add_argument('--verify', action='store_true', help='验证已有标定')
    parser.add_argument('--points', type=str, help='自定义标定点 JSON 文件')
    parser.add_argument('--camera', type=int, default=RGB_DEVICE, help='RGB 相机编号')
    parser.add_argument('--grab-z', type=float, default=DEFAULT_GRAB_Z, help='抓取 Z 高度 (mm)')
    args = parser.parse_args()

    # 初始化硬件
    print("连接机械臂...")
    robot = DummySerial()
    if not robot.connect():
        print("错误: 无法连接机械臂")
        sys.exit(1)

    print("启动相机...")
    camera = Camera(rgb_device=args.camera, enable_depth=False)
    camera.start()

    try:
        # 机械臂回零
        print("机械臂回零...")
        robot.home()
        robot.enable()

        if args.verify:
            verify_calibration(robot, camera)
        else:
            # 加载自定义标定点
            points = None
            if args.points:
                with open(args.points) as f:
                    points = [tuple(p) for p in json.load(f)]

            # 设置 grab_z
            calibrator = InteractiveCalibrator(robot, camera, points)
            calibrator.calibration.grab_z = args.grab_z
            calibrator.run()

    finally:
        camera.stop()
        robot.disconnect()


if __name__ == '__main__':
    main()
