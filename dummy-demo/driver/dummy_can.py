"""
Dummy V2 CAN/Fibre 通信驱动
============================

通过 USB 连接 REF 控制板，经 CAN 总线控制各关节电机。
基于 fibre 协议（参考 CLI-Tool）。
"""

import time
import logging

logger = logging.getLogger(__name__)


# Dummy V2 关节配置
JOINT_CONFIG = {
    # id: (name, reduction, inverse, angle_min, angle_max)
    0: ('J0_base',     1,  False, -180, 180),
    1: ('J1_shoulder', 50, False, -170, 170),
    2: ('J2_arm',      50, True,  -75,  90),
    3: ('J3_elbow',    50, True,  0,    180),
    4: ('J4_wrist1',   50, True,  -180, 180),
    5: ('J5_wrist2',   50, True,  -100, 120),
    6: ('J6_gripper',  5,  True,  -720, 720),
}


class DummyCAN:
    """Dummy V2 CAN 通信驱动"""

    def __init__(self, port=None):
        """
        初始化 Dummy CAN 驱动

        参数:
            port: USB 串口路径 (如 /dev/ttyACM0)，None 为自动搜索
        """
        self.port = port
        self.connected = False
        self.drive = None
        self.joint_positions = [0.0] * 7  # 7 个电机
        self.gripper_pos = 0.0

    def connect(self):
        """连接 Dummy"""
        try:
            import ref_tool
            if self.port:
                self.drive = ref_tool.find_any(f"serial:{self.port}")
            else:
                logger.info("搜索 Dummy 设备...")
                self.drive = ref_tool.find_any()
            self.connected = True
            logger.info(f"Dummy 已连接")
            return True
        except ImportError:
            logger.error("需要 ref_tool 库，请将 CLI-Tool 加入 PYTHONPATH")
            return False
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        self.connected = False
        self.drive = None
        logger.info("Dummy 已断开")

    def home(self):
        """回零位"""
        if not self.connected:
            logger.warning("未连接")
            return False
        logger.info("执行回零...")
        # TODO: 实际的回零指令
        self.joint_positions = [0.0] * 7
        return True

    def get_joint_positions(self):
        """
        读取当前关节角度

        返回:
            7 个关节角度 (度)
        """
        if not self.connected:
            return self.joint_positions
        # TODO: 通过 fibre 读取实际角度
        return self.joint_positions

    def set_joint_positions(self, positions, speed=50):
        """
        设置关节角度

        参数:
            positions: 6 或 7 个关节角度 (度)
            speed: 运动速度百分比 (1-100)
        """
        if not self.connected:
            logger.warning("未连接")
            return False

        # 检查限位
        for i, pos in enumerate(positions):
            config = JOINT_CONFIG.get(i)
            if config:
                _, _, _, lo, hi = config
                if pos < lo or pos > hi:
                    logger.error(f"J{i} = {pos}° 超出限位 [{lo}, {hi}]")
                    return False

        logger.info(f"移动到: {[f'{p:.1f}' for p in positions]}")
        # TODO: 发送 CAN 指令
        self.joint_positions[:len(positions)] = positions
        return True

    def move_joint(self, joint_id, angle, speed=50):
        """单轴运动"""
        positions = self.joint_positions.copy()
        positions[joint_id] = angle
        return self.set_joint_positions(positions, speed)

    def set_gripper(self, position):
        """
        控制手爪

        参数:
            position: 0.0 (全开) ~ 1.0 (全闭)
        """
        if not self.connected:
            return False
        angle = position * 720  # 映射到电机角度
        self.gripper_pos = position
        logger.info(f"手爪: {'闭合' if position > 0.5 else '张开'} ({position*100:.0f}%)")
        # TODO: 发送手爪指令
        return True

    def open_gripper(self):
        """张开手爪"""
        return self.set_gripper(0.0)

    def close_gripper(self):
        """闭合手爪"""
        return self.set_gripper(1.0)

    def estop(self):
        """急停"""
        logger.warning("⚠️ 急停！")
        # TODO: 发送急停指令
        return True

    def get_status(self):
        """获取状态"""
        return {
            'connected': self.connected,
            'joints': self.joint_positions,
            'gripper': self.gripper_pos,
        }
