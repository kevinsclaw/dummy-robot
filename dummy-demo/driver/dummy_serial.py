"""
Dummy V2 串口驱动
=================

通过 USB CDC 串口 (ASCII 命令) 控制机械臂。
实测验证的通信方式 — 2026-04-30。

硬件连接:
  Mac USB-C (翻面) → STM32 USB CDC → CAN 总线 → 电机

串口协议:
  !HOME      → 回零 (展开)
  !START     → 使能
  !STOP      → 急停
  !DISABLE   → 失能
  #GETJPOS   → 获取关节角度 (6轴)
  #GETLPOS   → 获取末端位姿 (XYZ + ABC)
  &j1,j2,j3,j4,j5,j6 → 关节运动
  @x,y,z,a,b,c        → 笛卡尔运动

夹爪: 通过 J6 机械联动控制
  J6 正值 → 合拢
  J6 负值 → 张开
"""

import serial
import serial.tools.list_ports
import time
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Dummy V2 STM32 USB CDC 标识
STM32_VID = 0x1209
STM32_PID = 0x0D32

# 零位关节角度 (HOME 后)
HOME_JOINTS = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]

# 零位笛卡尔位姿
HOME_POSE = [227.5, 0.0, 324.5, 0.0, 90.0, 0.0]

# 关节限位 (度)
JOINT_LIMITS = {
    0: (-180, 180),   # J1 底座旋转
    1: (-170, 170),   # J2 大臂
    2: (-75, 270),    # J3 小臂
    3: (-180, 180),   # J4 腕部旋转
    4: (-120, 120),   # J5 腕部俯仰
    5: (-720, 720),   # J6 末端旋转 / 夹爪
}

# 夹爪参数 (通过 J6 联动)
GRIPPER_OPEN_ANGLE = -60.0    # J6 角度: 张开
GRIPPER_CLOSE_ANGLE = 60.0    # J6 角度: 合拢
GRIPPER_MAX_ANGLE = 90.0      # J6 最大角度


class DummySerial:
    """Dummy V2 串口驱动"""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._ser: Optional[serial.Serial] = None
        self._joints = HOME_JOINTS.copy()  # 缓存的关节角度
        self._gripper_angle = 0.0           # J6 夹爪角度 (独立跟踪)
        self._enabled = False

    # ─── 连接管理 ────────────────────────────────────────────

    def connect(self) -> bool:
        """连接 Dummy"""
        port = self.port or self._find_port()
        if not port:
            logger.error("未找到 Dummy STM32 USB CDC 设备")
            return False

        try:
            self._ser = serial.Serial(port, self.baudrate, timeout=2)
            time.sleep(0.3)
            self._ser.read(self._ser.in_waiting)  # 清缓冲
            logger.info(f"已连接: {port}")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self._ser:
            try:
                if self._enabled:
                    self.disable()
            except:
                pass
            self._ser.close()
            self._ser = None
            logger.info("已断开")

    @property
    def connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _find_port(self) -> Optional[str]:
        """自动查找 STM32 USB CDC 串口"""
        for p in serial.tools.list_ports.comports():
            if p.vid == STM32_VID and p.pid == STM32_PID:
                logger.info(f"找到 Dummy: {p.device} (SN: {p.serial_number})")
                return p.device
        # 备用: 按名称匹配
        for p in serial.tools.list_ports.comports():
            if 'usbmodem' in p.device.lower():
                logger.info(f"可能的 Dummy 设备: {p.device}")
                return p.device
        return None

    # ─── 低级命令 ────────────────────────────────────────────

    def _send(self, cmd: str, timeout: float = 2.0) -> str:
        """发送命令并读取响应"""
        if not self.connected:
            raise RuntimeError("未连接")

        self._ser.read(self._ser.in_waiting)  # 清掉旧数据
        self._ser.write(f"{cmd}\n".encode())
        time.sleep(0.1)

        deadline = time.time() + timeout
        data = b''
        while time.time() < deadline:
            chunk = self._ser.read(self._ser.in_waiting or 1)
            data += chunk
            if b'\n' in data:
                break
            time.sleep(0.05)

        return data.decode('utf-8', errors='replace').strip()

    # ─── 基础控制 ────────────────────────────────────────────

    def home(self) -> bool:
        """回零位 (机械臂展开到 HOME)"""
        resp = self._send("!HOME")
        if "ok" in resp.lower() or "started" in resp.lower():
            logger.info("HOME 成功")
            self._joints = HOME_JOINTS.copy()
            self._gripper_angle = 0.0
            time.sleep(3)  # 等回零完成
            return True
        logger.error(f"HOME 失败: {resp}")
        return False

    def enable(self) -> bool:
        """使能 (允许运动)"""
        resp = self._send("!START")
        if "started" in resp.lower() or "ok" in resp.lower():
            self._enabled = True
            logger.info("使能成功")
            return True
        logger.error(f"使能失败: {resp}")
        return False

    def disable(self) -> bool:
        """失能"""
        resp = self._send("!DISABLE")
        self._enabled = False
        logger.info(f"失能: {resp}")
        return True

    def estop(self) -> bool:
        """急停"""
        resp = self._send("!STOP")
        self._enabled = False
        logger.warning(f"⚠️ 急停: {resp}")
        return True

    # ─── 状态查询 ────────────────────────────────────────────

    def get_joint_positions(self) -> List[float]:
        """读取当前 6 轴关节角度"""
        resp = self._send("#GETJPOS")
        # 响应格式: "ok 0.00 -0.00 90.00 0.00 0.00 0.00"
        try:
            parts = resp.split()
            if parts[0] == "ok" and len(parts) >= 7:
                self._joints = [float(x) for x in parts[1:7]]
                return self._joints.copy()
        except (ValueError, IndexError):
            pass
        logger.warning(f"GETJPOS 解析失败: {resp}")
        return self._joints.copy()

    def get_cartesian_pose(self) -> List[float]:
        """读取末端位姿 [X, Y, Z, A, B, C]"""
        resp = self._send("#GETLPOS")
        try:
            parts = resp.split()
            if parts[0] == "ok" and len(parts) >= 7:
                return [float(x) for x in parts[1:7]]
        except (ValueError, IndexError):
            pass
        logger.warning(f"GETLPOS 解析失败: {resp}")
        return HOME_POSE.copy()

    # ─── 关节运动 ────────────────────────────────────────────

    def move_joints(self, joints: List[float], wait: bool = True) -> bool:
        """
        关节运动 (6 轴)

        Args:
            joints: [J1, J2, J3, J4, J5, J6] 角度 (度)
            wait: 是否等待到位
        """
        if len(joints) != 6:
            logger.error(f"需要 6 个关节角度，收到 {len(joints)}")
            return False

        # 检查限位
        for i, angle in enumerate(joints):
            lo, hi = JOINT_LIMITS.get(i, (-720, 720))
            if angle < lo or angle > hi:
                logger.error(f"J{i+1} = {angle}° 超出限位 [{lo}, {hi}]")
                return False

        cmd = "&" + ",".join(f"{a:.2f}" for a in joints)
        resp = self._send(cmd)
        self._joints = joints.copy()

        if wait:
            # 简单等待 — 根据角度差估算时间
            max_delta = max(abs(a - b) for a, b in zip(joints, self._joints))
            wait_time = max(0.5, max_delta / 30.0)  # ~30°/s
            time.sleep(min(wait_time, 5.0))

        return True

    def move_joint_single(self, joint_id: int, angle: float) -> bool:
        """
        单轴运动 (保持其他轴不变)

        Args:
            joint_id: 0~5 (J1~J6)
            angle: 目标角度
        """
        current = self.get_joint_positions()
        current[joint_id] = angle
        return self.move_joints(current)

    def move_cartesian(self, x: float, y: float, z: float,
                       a: float = 0.0, b: float = 90.0, c: float = 0.0,
                       wait: bool = True) -> bool:
        """
        笛卡尔运动

        Args:
            x, y, z: 位置 (mm)
            a, b, c: 姿态 (度)
        """
        cmd = f"@{x:.2f},{y:.2f},{z:.2f},{a:.2f},{b:.2f},{c:.2f}"
        resp = self._send(cmd)

        if wait:
            time.sleep(1.5)  # 笛卡尔运动等待

        return True

    # ─── 夹爪控制 (通过 J6 机械联动) ────────────────────────

    def open_gripper(self, angle: Optional[float] = None) -> bool:
        """
        张开夹爪

        Args:
            angle: J6 负角度，默认 -60°
        """
        target = angle if angle is not None else GRIPPER_OPEN_ANGLE
        return self._set_gripper(target)

    def close_gripper(self, angle: Optional[float] = None) -> bool:
        """
        合拢夹爪

        Args:
            angle: J6 正角度，默认 60°
        """
        target = angle if angle is not None else GRIPPER_CLOSE_ANGLE
        return self._set_gripper(target)

    def set_gripper(self, position: float) -> bool:
        """
        设置夹爪位置

        Args:
            position: 0.0 (全开) ~ 1.0 (全闭)
        """
        angle = GRIPPER_OPEN_ANGLE + position * (GRIPPER_CLOSE_ANGLE - GRIPPER_OPEN_ANGLE)
        return self._set_gripper(angle)

    def _set_gripper(self, j6_angle: float) -> bool:
        """通过 J6 角度控制夹爪"""
        current = self.get_joint_positions()
        current[5] = j6_angle  # J6 是 index 5
        self._gripper_angle = j6_angle
        return self.move_joints(current)

    # ─── 状态 ────────────────────────────────────────────────

    def get_status(self) -> dict:
        """获取状态摘要"""
        joints = self.get_joint_positions()
        gripper_pct = (self._gripper_angle - GRIPPER_OPEN_ANGLE) / \
                      (GRIPPER_CLOSE_ANGLE - GRIPPER_OPEN_ANGLE)
        gripper_pct = max(0.0, min(1.0, gripper_pct))

        return {
            'connected': self.connected,
            'enabled': self._enabled,
            'joints': joints,
            'gripper_angle': self._gripper_angle,
            'gripper_pct': gripper_pct,
        }

    # ─── 上下文管理器 ────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
