"""
Dummy V2 任务执行器 — 将 RobotAction 转换为机械臂运动序列

连接 brain (LLM) → vision (检测) → driver (执行) 的核心模块。
适配 Dummy V2 的工作空间和手爪控制方式。

使用:
    executor = TaskExecutor(dummy, camera, detector, calibration)
    executor.execute(RobotAction(type="pick_and_place", object="red_block", target="plate"))
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """任务执行参数 — 🔧 这些都需要现场调"""
    # Dummy V2 工作空间较小 (~400mm臂展)，参数相应缩小
    safe_z: float = 200.0        # 安全高度 (mm)
    pick_z: float = 20.0         # 抓取高度 (mm) 🔧
    place_z: float = 20.0        # 放置高度 (mm) 🔧
    approach_z: float = 60.0     # 接近高度 (mm)
    stack_z_step: float = 20.0   # 堆叠每层高度 (mm) 🔧

    move_speed: float = 30.0     # 运动速度 (%)
    approach_speed: float = 15.0 # 接近速度 (%)
    fast_speed: float = 50.0     # 快速移动速度 (%)

    # Dummy 的手爪通过 J6 电机控制 (不是舵机)
    gripper_open_pos: float = 0.0    # 手爪张开 (0~1)
    gripper_close_pos: float = 0.8   # 手爪闭合 (0~1) 🔧

    gripper_delay: float = 0.5   # 手爪动作延时 (秒)
    settle_delay: float = 0.3    # 运动到位后稳定延时

    # 预设位置 🔧 需要现场标定
    home_joints: tuple = (0, 0, 0, 90, 0, 0)
    wave_joints: list = None
    nod_joints: list = None

    # 分拣目标区域 (世界坐标 mm) 🔧
    sort_zones: dict = None

    def __post_init__(self):
        if self.wave_joints is None:
            self.wave_joints = [
                (0, -15, 20, 90, -30, 0),
                (0, -15, 20, 90, -30, 40),
                (0, -15, 20, 90, -30, -40),
                (0, -15, 20, 90, -30, 40),
                (0, -15, 20, 90, -30, -40),
                (0, 0, 0, 90, 0, 0),
            ]
        if self.nod_joints is None:
            self.nod_joints = [
                (0, 10, -10, 90, 0, 0),
                (0, -10, 10, 90, 0, 0),
                (0, 10, -10, 90, 0, 0),
                (0, 0, 0, 90, 0, 0),
            ]
        if self.sort_zones is None:
            # 默认分拣区域 🔧 需要现场测量 (Dummy 工作半径 ~300mm)
            self.sort_zones = {
                "red": (150, -100, 20),
                "blue": (150, -60, 20),
                "green": (150, -20, 20),
                "yellow": (150, 20, 20),
                "plate": (180, 0, 20),
                "left_zone": (100, -100, 20),
                "right_zone": (100, 100, 20),
                "center_zone": (120, 0, 20),
            }


class TaskExecutor:
    """
    Dummy V2 任务执行器

    Args:
        robot: DummyCAN 实例
        camera: Camera 实例 (可选)
        detector: ColorDetector 实例 (可选)
        calibration: HandEyeCalibration 实例 (可选)
        config: TaskConfig
        speech: SpeechEngine 实例 (可选, 语音反馈)
    """

    def __init__(
        self,
        robot,
        camera=None,
        detector=None,
        calibration=None,
        config: TaskConfig = None,
        speech=None,
    ):
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.calibration = calibration
        self.config = config or TaskConfig()
        self.speech = speech

        self._holding = None
        self._stack_count = {}

        # 导入运动学 (用于笛卡尔运动)
        try:
            import sys, os
            sim_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dummy-sim')
            sys.path.insert(0, os.path.abspath(sim_path))
            from kinematics import fk, ik
            self._fk = fk
            self._ik = ik
            logger.info("运动学库已加载")
        except ImportError:
            logger.warning("运动学库未找到，笛卡尔运动不可用")
            self._fk = None
            self._ik = None

    def execute(self, action) -> bool:
        """执行动作"""
        logger.info(f"执行: {action}")

        handlers = {
            "pick_and_place": self._do_pick_and_place,
            "pick_up": self._do_pick_up,
            "put_down": self._do_put_down,
            "point_at": self._do_point_at,
            "push": self._do_push,
            "sort_by_color": self._do_sort_by_color,
            "stack": self._do_stack,
            "count": self._do_count,
            "wave": self._do_wave,
            "nod": self._do_nod,
            "go_home": self._do_go_home,
            "stop": self._do_stop,
            "describe_scene": self._do_describe_scene,
        }

        handler = handlers.get(action.type)
        if handler is None:
            self._say(f"不支持的动作: {action.type}")
            return False

        try:
            return handler(action)
        except Exception as e:
            logger.error(f"任务失败: {e}")
            self._say(f"执行失败: {e}")
            return False

    # ─── 笛卡尔运动 ─────────────────────────────────────────

    def _move_cartesian(self, x, y, z, a=0, b=0, c=0):
        """笛卡尔空间运动 (通过 IK 转换)"""
        if self._ik is None:
            logger.error("IK 不可用，无法执行笛卡尔运动")
            return False

        joints = self._ik([x, y, z, a, b, c])
        if joints is None:
            logger.error(f"IK 无解: ({x:.1f}, {y:.1f}, {z:.1f})")
            return False

        return self.robot.set_joint_positions(joints[:6])

    def _move_joints(self, *joints):
        """关节空间运动"""
        return self.robot.set_joint_positions(list(joints))

    # ─── 动作实现 ────────────────────────────────────────────

    def _do_pick_and_place(self, action) -> bool:
        """抓取放置"""
        obj_pos = self._find_object(action.object)
        if obj_pos is None:
            self._say(f"找不到 {action.object}")
            return False

        target_pos = self._resolve_target(action.target)
        if target_pos is None:
            self._say(f"找不到目标 {action.target}")
            return False

        self._say(f"抓取 {action.object}")
        self._pick_at(*obj_pos)
        self._place_at(*target_pos)
        self._say("放好了！")
        return True

    def _do_pick_up(self, action) -> bool:
        obj_pos = self._find_object(action.object)
        if obj_pos is None:
            self._say(f"找不到 {action.object}")
            return False

        self._say(f"抓起 {action.object}")
        self._pick_at(*obj_pos)
        self._holding = action.object
        return True

    def _do_put_down(self, action) -> bool:
        target_pos = self._resolve_target(action.target)
        if target_pos is None:
            # 在当前位置下方放置
            current = self.robot.get_joint_positions()
            if self._fk:
                pose = self._fk(current[:6])
                target_pos = (pose[0], pose[1], self.config.place_z)
            else:
                target_pos = (100, 0, self.config.place_z)

        self._place_at(*target_pos)
        self._holding = None
        self._say("放下了")
        return True

    def _do_point_at(self, action) -> bool:
        obj_pos = self._find_object(action.object)
        if obj_pos is None:
            self._say(f"找不到 {action.object}")
            return False

        x, y, z = obj_pos
        self._move_cartesian(x, y, self.config.approach_z)
        self._say(f"{action.object} 在这里")
        time.sleep(1)
        return True

    def _do_push(self, action) -> bool:
        obj_pos = self._find_object(action.object)
        if obj_pos is None:
            self._say(f"找不到 {action.object}")
            return False

        target_pos = self._resolve_target(action.target)
        if target_pos is None:
            target_pos = (obj_pos[0] + 30, obj_pos[1], obj_pos[2])

        x, y, z = obj_pos
        tx, ty, tz = target_pos
        push_z = self.config.pick_z + 3

        dx, dy = tx - x, ty - y
        dist = max(1, np.sqrt(dx**2 + dy**2))
        approach_x = x - dx / dist * 20
        approach_y = y - dy / dist * 20

        self._move_cartesian(approach_x, approach_y, self.config.safe_z)
        self._move_cartesian(approach_x, approach_y, push_z)
        self._move_cartesian(tx, ty, push_z)
        self._move_cartesian(tx, ty, self.config.safe_z)
        self._say("推好了")
        return True

    def _do_sort_by_color(self, action) -> bool:
        if not self.detector or not self.camera or not self.calibration:
            self._say("需要相机和标定才能分拣")
            return False

        color, _ = self.camera.read()
        if color is None:
            return False

        objects = self.detector.detect(color)
        if not objects:
            self._say("没有检测到物体")
            return False

        self._say(f"检测到 {len(objects)} 个物体，开始分拣")
        for obj in objects:
            if obj.color in self.config.sort_zones:
                world_pos = self.calibration.pixel_to_world(obj.cx, obj.cy)
                target_pos = self.config.sort_zones[obj.color]
                self._pick_at(*world_pos)
                self._place_at(*target_pos)

        self._say("分拣完成！")
        return True

    def _do_stack(self, action) -> bool:
        if not self.detector or not self.camera or not self.calibration:
            self._say("需要相机和标定才能堆叠")
            return False

        color, _ = self.camera.read()
        objects = self.detector.detect(color)
        if len(objects) < 2:
            self._say("至少需要两个物体才能堆叠")
            return False

        base = objects[0]
        base_world = self.calibration.pixel_to_world(base.cx, base.cy)
        self._say(f"开始堆叠，底座: {base.color}")

        for i, obj in enumerate(objects[1:], 1):
            obj_world = self.calibration.pixel_to_world(obj.cx, obj.cy)
            stack_z = self.config.place_z + i * self.config.stack_z_step
            self._pick_at(*obj_world)
            self._place_at(base_world[0], base_world[1], stack_z)

        self._say(f"堆了 {len(objects)} 层！")
        return True

    def _do_count(self, action) -> bool:
        if not self.detector or not self.camera:
            self._say("需要相机才能数")
            return False

        color, _ = self.camera.read()
        objects = self.detector.detect(color)
        color_count = {}
        for obj in objects:
            color_count[obj.color] = color_count.get(obj.color, 0) + 1

        msg = f"一共 {len(objects)} 个物体"
        for c, n in color_count.items():
            msg += f"，{c} {n} 个"
        self._say(msg)
        return True

    def _do_wave(self, action) -> bool:
        self._say("你好！")
        for joints in self.config.wave_joints:
            self._move_joints(*joints)
            time.sleep(0.3)
        return True

    def _do_nod(self, action) -> bool:
        for joints in self.config.nod_joints:
            self._move_joints(*joints)
            time.sleep(0.3)
        return True

    def _do_go_home(self, action) -> bool:
        self._say("回家")
        self.robot.open_gripper()
        self._move_joints(*self.config.home_joints)
        self._holding = None
        return True

    def _do_stop(self, action) -> bool:
        self._say("紧急停止！")
        self.robot.estop()
        return True

    def _do_describe_scene(self, action) -> bool:
        if not self.detector or not self.camera:
            self._say("需要相机才能看")
            return False

        color, _ = self.camera.read()
        objects = self.detector.detect(color)
        if not objects:
            self._say("桌上什么都没有")
        else:
            descriptions = [f"{obj.color}色物体在({obj.cx}, {obj.cy})" for obj in objects]
            self._say(f"看到 {len(objects)} 个物体: " + "，".join(descriptions))
        return True

    # ─── 基础运动原语 ────────────────────────────────────────

    def _pick_at(self, x, y, z=None):
        """在指定位置抓取"""
        if z is None:
            z = self.config.pick_z
        cfg = self.config

        self.robot.open_gripper()
        time.sleep(cfg.gripper_delay)

        self._move_cartesian(x, y, cfg.safe_z)
        time.sleep(cfg.settle_delay)

        self._move_cartesian(x, y, cfg.approach_z)
        self._move_cartesian(x, y, z)
        time.sleep(cfg.settle_delay)

        self.robot.close_gripper()
        time.sleep(cfg.gripper_delay)

        self._move_cartesian(x, y, cfg.safe_z)

    def _place_at(self, x, y, z=None):
        """在指定位置放置"""
        if z is None:
            z = self.config.place_z
        cfg = self.config

        self._move_cartesian(x, y, cfg.safe_z)
        time.sleep(cfg.settle_delay)

        self._move_cartesian(x, y, cfg.approach_z)
        self._move_cartesian(x, y, z)
        time.sleep(cfg.settle_delay)

        self.robot.open_gripper()
        time.sleep(cfg.gripper_delay)

        self._move_cartesian(x, y, cfg.safe_z)
        self._holding = None

    # ─── 辅助方法 ────────────────────────────────────────────

    def _find_object(self, object_name) -> Optional[Tuple[float, float, float]]:
        """通过视觉找物体世界坐标"""
        if not self.detector or not self.camera or not self.calibration:
            logger.warning("视觉系统不可用")
            return None

        color = object_name.split("_")[0] if "_" in object_name else object_name

        frame, depth = self.camera.read()
        if frame is None:
            return None

        objects = self.detector.detect_specific(frame, color, depth)
        if not objects:
            return None

        obj = objects[0]
        wx, wy, wz = self.calibration.pixel_to_world(obj.cx, obj.cy)
        if obj.depth_mm > 0:
            wz = self.config.pick_z

        logger.info(f"找到 {object_name}: pixel({obj.cx},{obj.cy}) → world({wx:.1f},{wy:.1f},{wz:.1f})")
        return (wx, wy, wz)

    def _resolve_target(self, target_name) -> Optional[Tuple[float, float, float]]:
        """解析目标位置"""
        if target_name is None:
            return None

        if target_name in self.config.sort_zones:
            return self.config.sort_zones[target_name]

        if target_name.startswith("near_"):
            ref_obj = target_name.replace("near_", "")
            ref_pos = self._find_object(ref_obj)
            if ref_pos:
                return (ref_pos[0] + 30, ref_pos[1], ref_pos[2])

        return self._find_object(target_name)

    def _say(self, text):
        """语音/文字反馈"""
        logger.info(f"[SAY] {text}")
        if self.speech:
            self.speech.speak_async(text)
        else:
            print(f"🤖 {text}")
