"""
任务执行器 — 将 RobotAction 转换为机械臂运动序列

连接 brain (LLM) → vision (检测) → driver (执行) 的核心模块。

使用:
    executor = TaskExecutor(picker, camera, detector, calibration)
    executor.execute(RobotAction(type="pick_and_place", object="red_block", target="plate"))
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """任务执行参数 — 🔧 这些都需要现场调"""
    safe_z: float = 300.0        # 安全高度 (mm)
    pick_z: float = 30.0         # 抓取高度 (mm) 🔧
    place_z: float = 30.0        # 放置高度 (mm) 🔧
    approach_z: float = 80.0     # 接近高度 (mm)
    stack_z_step: float = 25.0   # 堆叠每层高度 (mm) 🔧

    move_speed: float = 25.0     # 运动速度 (%)
    approach_speed: float = 15.0 # 接近速度 (%)
    fast_speed: float = 40.0     # 快速移动速度 (%)

    gripper_open_pos: int = 90   # 夹爪张开角度 🔧
    gripper_close_pos: int = 30  # 夹爪闭合角度 🔧
    gripper_servo: int = 0       # 夹爪舵机编号

    gripper_delay: float = 0.5   # 夹爪动作延时 (秒)
    settle_delay: float = 0.3    # 运动到位后稳定延时

    # 预设位置 🔧 需要现场标定
    home_joints: tuple = (0, 0, 0, 0, 0, 0)
    wave_joints: list = None     # 挥手动作关节序列
    nod_joints: list = None      # 点头动作关节序列

    # 分拣目标区域 (世界坐标 mm) 🔧
    sort_zones: dict = None

    def __post_init__(self):
        if self.wave_joints is None:
            self.wave_joints = [
                (0, -20, 30, 0, -30, 0),
                (0, -20, 30, 0, -30, 40),
                (0, -20, 30, 0, -30, -40),
                (0, -20, 30, 0, -30, 40),
                (0, -20, 30, 0, -30, -40),
                (0, 0, 0, 0, 0, 0),
            ]
        if self.nod_joints is None:
            self.nod_joints = [
                (0, 10, -10, 0, 0, 0),
                (0, -10, 10, 0, 0, 0),
                (0, 10, -10, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
            ]
        if self.sort_zones is None:
            # 默认分拣区域 🔧 需要现场测量
            self.sort_zones = {
                "red": (250, -150, 30),
                "blue": (250, -100, 30),
                "green": (250, -50, 30),
                "yellow": (250, 0, 30),
                "plate": (300, 0, 30),
                "left_zone": (150, -150, 30),
                "right_zone": (150, 150, 30),
                "center_zone": (200, 0, 30),
            }


class TaskExecutor:
    """
    任务执行器

    Args:
        picker: PickerDriver 实例
        camera: Camera 实例 (可选)
        detector: ColorDetector 实例 (可选)
        calibration: HandEyeCalibration 实例 (可选)
        config: TaskConfig
        speech: SpeechEngine 实例 (可选, 语音反馈)
    """

    def __init__(
        self,
        picker,
        camera=None,
        detector=None,
        calibration=None,
        config: TaskConfig = None,
        speech=None,
    ):
        self.picker = picker
        self.camera = camera
        self.detector = detector
        self.calibration = calibration
        self.config = config or TaskConfig()
        self.speech = speech

        self._holding = None  # 当前持有的物体
        self._stack_count = {}  # 堆叠计数

    def execute(self, action) -> bool:
        """
        执行动作

        Args:
            action: RobotAction

        Returns:
            是否成功
        """
        logger.info(f"Executing: {action}")

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
            self._say(f"我不知道怎么做 {action.type}")
            return False

        try:
            return handler(action)
        except Exception as e:
            logger.error(f"Task failed: {e}")
            self._say(f"执行失败: {e}")
            return False

    # ─── 动作实现 ────────────────────────────────────────────

    def _do_pick_and_place(self, action) -> bool:
        """抓取放置"""
        # 找物体
        obj_pos = self._find_object(action.object)
        if obj_pos is None:
            self._say(f"找不到 {action.object}")
            return False

        # 找目标
        target_pos = self._resolve_target(action.target)
        if target_pos is None:
            self._say(f"找不到目标位置 {action.target}")
            return False

        self._say(f"好的，抓取 {action.object}")

        # 执行
        self._pick_at(*obj_pos)
        self._place_at(*target_pos)

        self._say("放好了！")
        return True

    def _do_pick_up(self, action) -> bool:
        """抓起物体"""
        obj_pos = self._find_object(action.object)
        if obj_pos is None:
            self._say(f"找不到 {action.object}")
            return False

        self._say(f"抓起 {action.object}")
        self._pick_at(*obj_pos)
        self._holding = action.object
        return True

    def _do_put_down(self, action) -> bool:
        """放下物体"""
        target_pos = self._resolve_target(action.target)
        if target_pos is None:
            # 放在当前位置下方
            pos = self.picker.request_position()
            target_pos = (pos.pose.x, pos.pose.y, self.config.place_z)

        self._place_at(*target_pos)
        self._holding = None
        self._say("放下了")
        return True

    def _do_point_at(self, action) -> bool:
        """指向物体"""
        obj_pos = self._find_object(action.object)
        if obj_pos is None:
            self._say(f"找不到 {action.object}")
            return False

        x, y, z = obj_pos
        self.picker.set_speed(self.config.move_speed)
        self.picker.move_cartesian(x, y, self.config.approach_z, 0, 0, 0)
        self._say(f"{action.object} 在这里")
        time.sleep(1)
        return True

    def _do_push(self, action) -> bool:
        """推动物体"""
        obj_pos = self._find_object(action.object)
        if obj_pos is None:
            self._say(f"找不到 {action.object}")
            return False

        target_pos = self._resolve_target(action.target)
        if target_pos is None:
            target_pos = (obj_pos[0] + 50, obj_pos[1], obj_pos[2])

        # 移到物体侧面 → 推过去
        x, y, z = obj_pos
        tx, ty, tz = target_pos
        push_z = self.config.pick_z + 5

        # 计算推动方向的反方向 (接近点)
        dx, dy = tx - x, ty - y
        dist = max(1, np.sqrt(dx**2 + dy**2))
        approach_x = x - dx / dist * 30
        approach_y = y - dy / dist * 30

        self.picker.set_speed(self.config.move_speed)
        self.picker.move_cartesian(approach_x, approach_y, self.config.safe_z, 0, 0, 0)
        self.picker.move_linear(approach_x, approach_y, push_z, 0, 0, 0)
        self.picker.set_speed(self.config.approach_speed)
        self.picker.move_linear(tx, ty, push_z, 0, 0, 0)
        self.picker.move_linear(tx, ty, self.config.safe_z, 0, 0, 0)

        self._say("推好了")
        return True

    def _do_sort_by_color(self, action) -> bool:
        """按颜色分拣"""
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
            target_key = obj.color
            if target_key in self.config.sort_zones:
                world_pos = self.calibration.pixel_to_world(obj.cx, obj.cy)
                target_pos = self.config.sort_zones[target_key]

                self._pick_at(*world_pos)
                self._place_at(*target_pos)

        self._say("分拣完成！")
        return True

    def _do_stack(self, action) -> bool:
        """堆叠物体"""
        if not self.detector or not self.camera or not self.calibration:
            self._say("需要相机和标定才能堆叠")
            return False

        color, _ = self.camera.read()
        objects = self.detector.detect(color)

        if len(objects) < 2:
            self._say("至少需要两个物体才能堆叠")
            return False

        # 第一个物体作为底座
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
        """数物体"""
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
        """挥手"""
        self._say("你好！")
        self.picker.set_speed(30)
        for joints in self.config.wave_joints:
            self.picker.move_joints(*joints)
            time.sleep(0.3)
        return True

    def _do_nod(self, action) -> bool:
        """点头"""
        self.picker.set_speed(25)
        for joints in self.config.nod_joints:
            self.picker.move_joints(*joints)
            time.sleep(0.3)
        return True

    def _do_go_home(self, action) -> bool:
        """回零"""
        self._say("回家")
        self.picker.set_speed(self.config.move_speed)
        self.picker.gripper_open(
            self.config.gripper_servo, self.config.gripper_open_pos
        )
        self.picker.move_joints(*self.config.home_joints)
        self._holding = None
        return True

    def _do_stop(self, action) -> bool:
        """紧急停止"""
        self._say("紧急停止！")
        # 停在当前位置 — 发送当前位置作为目标
        pos = self.picker.request_position()
        self.picker.move_joints(*pos.joints.as_tuple())
        return True

    def _do_describe_scene(self, action) -> bool:
        """描述场景"""
        if not self.detector or not self.camera:
            self._say("需要相机才能看")
            return False

        color, _ = self.camera.read()
        objects = self.detector.detect(color)

        if not objects:
            self._say("桌上什么都没有")
        else:
            descriptions = []
            for obj in objects:
                descriptions.append(f"{obj.color}色物体在图像位置({obj.cx}, {obj.cy})")
            msg = f"我看到 {len(objects)} 个物体: " + "，".join(descriptions)
            self._say(msg)

        return True

    # ─── 基础运动原语 ────────────────────────────────────────

    def _pick_at(self, x: float, y: float, z: float = None):
        """在指定位置抓取"""
        if z is None:
            z = self.config.pick_z
        cfg = self.config

        # 张开夹爪
        self.picker.gripper_open(cfg.gripper_servo, cfg.gripper_open_pos)
        time.sleep(cfg.gripper_delay)

        # 移到上方
        self.picker.set_speed(cfg.fast_speed)
        self.picker.move_cartesian(x, y, cfg.safe_z, 0, 0, 0)
        time.sleep(cfg.settle_delay)

        # 下降接近
        self.picker.set_speed(cfg.approach_speed)
        self.picker.move_linear(x, y, cfg.approach_z, 0, 0, 0)
        self.picker.move_linear(x, y, z, 0, 0, 0)
        time.sleep(cfg.settle_delay)

        # 夹取
        self.picker.gripper_close(cfg.gripper_servo, cfg.gripper_close_pos)
        time.sleep(cfg.gripper_delay)

        # 抬起
        self.picker.move_linear(x, y, cfg.safe_z, 0, 0, 0)

    def _place_at(self, x: float, y: float, z: float = None):
        """在指定位置放置"""
        if z is None:
            z = self.config.place_z
        cfg = self.config

        # 移到上方
        self.picker.set_speed(cfg.fast_speed)
        self.picker.move_cartesian(x, y, cfg.safe_z, 0, 0, 0)
        time.sleep(cfg.settle_delay)

        # 下降
        self.picker.set_speed(cfg.approach_speed)
        self.picker.move_linear(x, y, cfg.approach_z, 0, 0, 0)
        self.picker.move_linear(x, y, z, 0, 0, 0)
        time.sleep(cfg.settle_delay)

        # 松开
        self.picker.gripper_open(
            cfg.gripper_servo, cfg.gripper_open_pos
        )
        time.sleep(cfg.gripper_delay)

        # 抬起
        self.picker.move_linear(x, y, cfg.safe_z, 0, 0, 0)
        self._holding = None

    # ─── 辅助方法 ────────────────────────────────────────────

    def _find_object(self, object_name: str) -> Optional[Tuple[float, float, float]]:
        """
        通过视觉找到物体的世界坐标

        Args:
            object_name: 如 "red_block"

        Returns:
            (x, y, z) 世界坐标 (mm)，或 None
        """
        if not self.detector or not self.camera or not self.calibration:
            logger.warning("Vision not available, cannot find object")
            return None

        # 解析颜色
        color = object_name.split("_")[0] if "_" in object_name else object_name

        # 拍照检测
        frame, depth = self.camera.read()
        if frame is None:
            return None

        objects = self.detector.detect_specific(frame, color, depth)
        if not objects:
            return None

        # 取最大的那个
        obj = objects[0]

        # 像素 → 世界坐标
        wx, wy, wz = self.calibration.pixel_to_world(obj.cx, obj.cy)

        # 如果有深度数据，用深度更新 Z
        if obj.depth_mm > 0:
            wz = self.config.pick_z  # 简化处理

        logger.info(f"Found {object_name}: pixel({obj.cx},{obj.cy}) → world({wx:.1f},{wy:.1f},{wz:.1f})")
        return (wx, wy, wz)

    def _resolve_target(self, target_name: str) -> Optional[Tuple[float, float, float]]:
        """解析目标位置"""
        if target_name is None:
            return None

        # 先查预设区域
        if target_name in self.config.sort_zones:
            return self.config.sort_zones[target_name]

        # "near_xxx_block" → 找到 xxx 物体旁边
        if target_name.startswith("near_"):
            ref_obj = target_name.replace("near_", "")
            ref_pos = self._find_object(ref_obj)
            if ref_pos:
                # 旁边偏移 40mm
                return (ref_pos[0] + 40, ref_pos[1], ref_pos[2])

        # 通过视觉找目标
        return self._find_object(target_name)

    def _say(self, text: str):
        """语音/文字反馈"""
        logger.info(f"[SAY] {text}")
        if self.speech:
            self.speech.speak_async(text)
        else:
            print(f"🤖 {text}")
