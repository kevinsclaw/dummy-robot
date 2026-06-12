"""
Dummy V2 Strands Agent — ReAct 循环控制
========================================

用 AWS Strands Agents SDK 实现 ReAct 风格的机械臂控制。
LLM 作为 Agent，通过 tool calling 实时控制机械臂。

架构:
  用户语音/文字 → Strands Agent (Bedrock Claude) → Tool Calls → STM32 执行

优势:
  - 每一步都有视觉反馈 (detect → move → verify)
  - 失败自动重试 (抓取失败 → 重新检测 → 再试)
  - 支持多轮对话 ("再往左一点")
"""

import json
import time
import logging
import numpy as np
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ─── Tool Definitions ────────────────────────────────────────

SYSTEM_PROMPT = """你是 Dummy V2 机械臂的智能控制器。

## 硬件信息
- 6-DOF 桌面机械臂，工作半径 ~300mm
- 夹爪通过 J6 电机联动控制
- STM32 控制器，通过串口通信

## 工作空间 (毫米)
- X: [50, 350] (前方为正)
- Y: [-200, 200] (左正右负)
- Z: [0, 400] (上方为正)
- 安全高度: Z=200mm (移动时保持此高度避免碰撞)

## 操作规范
1. 移动前先确认目标在工作空间内
2. 接近物体: 先到上方 60mm，再垂直下降
3. 抓取后: 先垂直上升到安全高度，再水平移动
4. 放置: 先到目标上方，再垂直下降
5. 每次抓取/放置后，用 detect_objects 验证结果

## 安全规则
- 绝不在 Z<10mm 时水平移动 (会撞桌面)
- 夹爪张开/合拢前确认高度正确
- 如果检测不到目标物体，询问用户而不是盲目尝试
- 连续 3 次失败后停止并报告

## 颜色方块 Demo
工作台上有彩色方块 (约 25mm 边长)。
常见任务: 按颜色分拣、堆叠、放到指定位置。
"""

# ─── Strands Tool Functions ──────────────────────────────────

def create_agent_tools(robot, camera=None, detector=None, calibration=None):
    """
    创建 Strands Agent 的 tool 函数集合。
    
    Args:
        robot: DummySerial 实例
        camera: Camera 实例 (可选, OpenCV VideoCapture 也行)
        detector: ColorBlockDetector 或 HailoDetector 实例 (可选)
        calibration: 手眼标定 (可选, detector 自带简单标定)
    
    Returns:
        list of tool functions
    """
    
    from strands import tool

    @tool
    def detect_objects() -> str:
        """
        检测工作台上的所有物体。
        返回每个物体的颜色、像素坐标和世界坐标(mm)。
        如果相机/检测器不可用，返回错误信息。
        """
        if not camera or not detector:
            return json.dumps({"error": "视觉系统不可用 (相机/检测器缺失)"})
        
        # 支持两种相机接口
        if hasattr(camera, 'read_color'):
            frame = camera.read_color()
        elif hasattr(camera, 'read'):
            ret = camera.read()
            frame = ret[0] if isinstance(ret, tuple) else ret
        else:
            return json.dumps({"error": "相机接口不兼容"})
        
        if frame is None:
            return json.dumps({"error": "相机读取失败"})
        
        # 支持两种检测器 (ColorBlockDetector / HailoDetector)
        objects = detector.detect(frame)
        results = []
        for obj in objects:
            # ColorBlockDetector 返回 Block (有 world_x/y)
            # HailoDetector 返回 Detection (需要 calibration)
            if hasattr(obj, 'world_x'):
                wx, wy, wz = obj.world_x, obj.world_y, obj.world_z
            elif calibration:
                wx, wy, wz = calibration.pixel_to_world(obj.cx, obj.cy)
            else:
                wx, wy, wz = 0, 0, 0
            
            results.append({
                "name": f"{obj.color}_block",
                "color": obj.color,
                "pixel": [obj.cx, obj.cy],
                "world_mm": [round(wx, 1), round(wy, 1), round(wz, 1)],
                "size_px": getattr(obj, 'area_px', getattr(obj, 'area', 0)),
            })
        
        return json.dumps({
            "objects": results,
            "count": len(results),
            "timestamp": time.strftime("%H:%M:%S"),
        })

    @tool
    def move_to(x: float, y: float, z: float) -> str:
        """
        移动机械臂末端到指定的世界坐标位置 (笛卡尔空间)。
        
        Args:
            x: X 坐标 (mm), 前方为正, 范围 [50, 350]
            y: Y 坐标 (mm), 左为正, 范围 [-200, 200]  
            z: Z 坐标 (mm), 上方为正, 范围 [10, 400]
        
        Returns:
            执行结果
        """
        # 安全检查
        if z < 10:
            return json.dumps({"error": f"Z={z}mm 太低，最小值为 10mm"})
        if x < 50 or x > 350:
            return json.dumps({"error": f"X={x}mm 超出工作范围 [50, 350]"})
        if y < -200 or y > 200:
            return json.dumps({"error": f"Y={y}mm 超出工作范围 [-200, 200]"})
        
        success = robot.move_cartesian(x, y, z)
        if success:
            return json.dumps({"status": "ok", "position": [x, y, z]})
        else:
            return json.dumps({"error": "运动指令执行失败"})

    @tool
    def open_gripper() -> str:
        """张开夹爪，准备抓取或释放物体。"""
        robot.open_gripper()
        time.sleep(0.5)
        return json.dumps({"status": "ok", "gripper": "open"})

    @tool
    def close_gripper() -> str:
        """合拢夹爪，抓住物体。"""
        robot.close_gripper()
        time.sleep(0.5)
        return json.dumps({"status": "ok", "gripper": "closed"})

    @tool
    def home() -> str:
        """回到初始零位。机械臂展开到 HOME 姿态。"""
        robot.home()
        robot.enable()
        return json.dumps({"status": "ok", "position": "home"})

    @tool
    def get_current_position() -> str:
        """读取机械臂当前的关节角度和末端位姿。"""
        joints = robot.get_joint_positions()
        pose = robot.get_cartesian_pose()
        return json.dumps({
            "joints_deg": [round(j, 1) for j in joints],
            "cartesian_mm": [round(p, 1) for p in pose],
        })

    @tool
    def move_joints(j1: float, j2: float, j3: float, 
                    j4: float, j5: float, j6: float) -> str:
        """
        直接设置 6 个关节角度 (度)。
        一般优先使用 move_to (笛卡尔)，只在需要精确关节控制时使用此工具。
        
        Args:
            j1-j6: 各关节目标角度 (度)
        """
        success = robot.move_joints([j1, j2, j3, j4, j5, j6])
        if success:
            return json.dumps({"status": "ok", "joints": [j1, j2, j3, j4, j5, j6]})
        else:
            return json.dumps({"error": "关节运动失败 (可能超出限位)"})

    @tool  
    def set_gripper_position(position: float) -> str:
        """
        设置夹爪开合度。
        
        Args:
            position: 0.0 = 全开, 1.0 = 全闭, 0.5 = 半开
        """
        if position < 0 or position > 1:
            return json.dumps({"error": "position 必须在 0.0~1.0 之间"})
        robot.set_gripper(position)
        time.sleep(0.5)
        return json.dumps({"status": "ok", "gripper_position": position})

    return [detect_objects, move_to, open_gripper, close_gripper, 
            home, get_current_position, move_joints, set_gripper_position]


# ─── Agent Setup ─────────────────────────────────────────────

def create_dummy_agent(robot, camera=None, detector=None, calibration=None,
                       model_id: str = "qwen.qwen3-vl-235b-a22b",
                       provider: str = "bedrock",
                       region: str = "us-east-1"):
    """
    创建 Dummy V2 的 Strands Agent。
    
    Args:
        robot: DummySerial 实例 (必须已连接)
        camera: Camera 实例
        detector: 物体检测器
        calibration: 手眼标定
        model_id: 模型 ID
        provider: "bedrock" (AWS) 或 "dashscope" (阿里云)
        region: AWS region (仅 bedrock)
    
    Returns:
        Strands Agent 实例
    """
    from strands import Agent
    
    if provider == "dashscope":
        import os
        from strands.models.openai import OpenAIModel
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        model = OpenAIModel(
            client_args={
                "api_key": api_key,
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            },
            model_id=model_id,
            params={
                "max_tokens": 2048,
                "temperature": 0.7,
            }
        )
    else:
        from strands.models.bedrock import BedrockModel
        model = BedrockModel(
            model_id=model_id,
            region_name=region,
        )
    
    tools = create_agent_tools(robot, camera, detector, calibration)
    
    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    
    return agent


# ─── Demo Runner ─────────────────────────────────────────────

def run_demo(port: Optional[str] = None, mock: bool = False,
             model_id: str = "qwen-vl-max", provider: str = "dashscope"):
    """
    运行交互式 Demo。
    
    Args:
        port: STM32 串口路径 (None=自动搜索)
        mock: 使用 mock 驱动 (不需要真实硬件)
        model_id: 模型 ID
        provider: 模型提供商
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from driver.dummy_serial import DummySerial
    
    if mock:
        robot = MockRobot()
        print("🤖 Mock 模式 (无真实硬件)")
    else:
        robot = DummySerial(port=port)
        if not robot.connect():
            print("❌ 连接失败！")
            return
        robot.home()
        robot.enable()
        print("✅ 机械臂已连接并使能")
    
    # TODO: 相机和检测器在有硬件时初始化
    camera = None
    detector = None
    calibration = None
    
    agent = create_dummy_agent(robot, camera, detector, calibration,
                               model_id=model_id, provider=provider)
    
    print("\n🤖 Dummy V2 智能抓取 Demo")
    print("=" * 50)
    print("说出你的指令（输入 quit 退出）")
    print("示例:")
    print("  - 把红色方块放到蓝色旁边")
    print("  - 看看桌上有什么")
    print("  - 挥个手")
    print("  - 回零位")
    print()
    
    while True:
        try:
            user_input = input("你: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        
        # 调用 Agent
        try:
            response = agent(user_input)
            print(f"\n🤖: {response}\n")
        except Exception as e:
            print(f"\n❌ Agent 错误: {e}\n")
    
    if not mock:
        robot.disable()
        robot.disconnect()
    print("Demo 结束")


# ─── Mock Robot (用于测试) ───────────────────────────────────

class MockRobot:
    """Mock 机器人驱动，用于不接硬件时测试 Agent 逻辑"""
    
    def __init__(self):
        self._joints = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]
        self._pose = [227.5, 0.0, 324.5, 0.0, 90.0, 0.0]
        self._gripper = 0.0
        self._enabled = True
    
    def connect(self): return True
    def disconnect(self): pass
    def home(self): 
        self._joints = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]
        self._pose = [227.5, 0.0, 324.5, 0.0, 90.0, 0.0]
        print("  [MOCK] HOME")
        return True
    def enable(self): self._enabled = True; return True
    def disable(self): self._enabled = False; return True
    def estop(self): print("  [MOCK] ESTOP!"); return True
    
    def get_joint_positions(self): return self._joints.copy()
    def get_cartesian_pose(self): return self._pose.copy()
    
    def move_joints(self, joints):
        self._joints = joints.copy()
        print(f"  [MOCK] move_joints → {[f'{j:.1f}' for j in joints]}")
        time.sleep(0.3)
        return True
    
    def move_cartesian(self, x, y, z, a=0, b=90, c=0):
        self._pose = [x, y, z, a, b, c]
        print(f"  [MOCK] move_to → ({x:.1f}, {y:.1f}, {z:.1f})")
        time.sleep(0.3)
        return True
    
    def open_gripper(self):
        self._gripper = 0.0
        print("  [MOCK] gripper OPEN")
        return True
    
    def close_gripper(self):
        self._gripper = 1.0
        print("  [MOCK] gripper CLOSE")
        return True
    
    def set_gripper(self, pos):
        self._gripper = pos
        print(f"  [MOCK] gripper → {pos:.1f}")
        return True


# ─── Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dummy V2 Strands Agent Demo")
    parser.add_argument("--mock", action="store_true", help="Mock 模式 (无硬件)")
    parser.add_argument("--port", type=str, default=None, help="串口路径")
    parser.add_argument("--model", type=str, 
                        default="qwen.qwen3-vl-235b-a22b",
                        help="模型 ID")
    parser.add_argument("--provider", type=str, default="bedrock",
                        choices=["dashscope", "bedrock"],
                        help="模型提供商")
    args = parser.parse_args()
    
    run_demo(port=args.port, mock=args.mock, model_id=args.model, provider=args.provider)
