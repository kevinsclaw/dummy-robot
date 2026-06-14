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
import cv2
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

## 深度感知 (Gemini 335)
- 相机提供 RGB + 深度。detect_objects 返回的每个物体可能带 depth_mm (该像素到相机的距离)。
- 需要某点准确距离时用 measure_depth(pixel_x, pixel_y)。
- 抓取前建议先 measure_depth 确认物体真实距离, 再决定下降深度, 别只靠猜。

## 颜色方块 Demo
工作台上有彩色方块 (约 25mm 边长)。
常见任务: 按颜色分拣、堆叠、放到指定位置。
"""

# ─── Strands Tool Functions ──────────────────────────────────

def create_agent_tools(robot, camera=None, detector=None, calibration=None, hailo=None):
    """
    创建 Strands Agent 的 tool 函数集合。
    
    Args:
        robot: DummySerial 实例
        camera: Camera 实例 (可选, OpenCV VideoCapture 也行)
        detector: ColorBlockDetector 实例 (可选, HSV 颜色检测)
        calibration: 手眼标定 (可选)
        hailo: HailoDetector 实例 (可选, YOLO 物体检测)
    
    Returns:
        list of tool functions
    """
    
    from strands import tool

    @tool
    def detect_objects() -> str:
        """
        检测工作台上的所有物体。
        使用两套检测系统:
        1. Hailo-8 YOLOv6n: COCO 80类通用物体检测 (人、杯子、瓶子等)
        2. HSV 颜色分割: 精确检测彩色方块
        返回每个物体的类别、颜色、像素坐标和世界坐标(mm)。
        """
        if not camera:
            return json.dumps({"error": "视觉系统不可用 (相机缺失)"})
        
        # 拍照
        if hasattr(camera, 'read_color'):
            frame = camera.read_color()
        elif hasattr(camera, 'read'):
            ret = camera.read()
            if isinstance(ret, tuple):
                success, frame = ret
                if not success:
                    frame = None
            else:
                frame = ret
        else:
            return json.dumps({"error": "相机接口不兼容"})
        
        if frame is None:
            return json.dumps({"error": "相机读取失败"})
        
        results = []

        # 深度查询辅助: 仅当相机是 Gemini335 (有 depth_at) 时可用
        def _depth_mm_at(px, py):
            if not hasattr(camera, "depth_at"):
                return None
            try:
                # 彩色与深度分辨率可能不同, 按比例把彩色像素映射到深度图
                draw = camera.latest_depth() if hasattr(camera, "latest_depth") else None
                if draw is None:
                    return None
                ch, cw = frame.shape[:2]
                dh, dw = draw.shape[:2]
                dx = int(px * dw / cw)
                dy = int(py * dh / ch)
                z = camera.depth_at(dx, dy)
                if z is None or z <= 0:
                    return None
                return round(float(z), 1)
            except Exception:
                return None
        
        # 1. Hailo YOLO 检测 (通用物体)
        if hailo and hailo._started:
            try:
                yolo_objects = hailo.detect(frame)
                for obj in yolo_objects:
                    if calibration:
                        wx, wy, wz = calibration.pixel_to_world(obj.cx, obj.cy)
                    else:
                        wx, wy, wz = 0, 0, 0
                    results.append({
                        "name": obj.label,
                        "color": obj.color,
                        "source": "yolo",
                        "confidence": round(obj.confidence, 2),
                        "pixel": [obj.cx, obj.cy],
                        "world_mm": [round(wx, 1), round(wy, 1), round(wz, 1)],
                        "depth_mm": _depth_mm_at(obj.cx, obj.cy),
                        "bbox": obj.bbox,
                    })
            except Exception as e:
                pass  # YOLO 失败不影响 HSV
        
        # 2. HSV 颜色方块检测
        if detector:
            try:
                color_objects = detector.detect(frame)
                for obj in color_objects:
                    if hasattr(obj, 'world_x'):
                        wx, wy, wz = obj.world_x, obj.world_y, obj.world_z
                    elif calibration:
                        wx, wy, wz = calibration.pixel_to_world(obj.cx, obj.cy)
                    else:
                        wx, wy, wz = 0, 0, 0
                    results.append({
                        "name": f"{obj.color}_block",
                        "color": obj.color,
                        "source": "hsv",
                        "confidence": 1.0,
                        "pixel": [obj.cx, obj.cy],
                        "world_mm": [round(wx, 1), round(wy, 1), round(wz, 1)],
                        "depth_mm": _depth_mm_at(obj.cx, obj.cy),
                        "size_px": getattr(obj, 'area_px', getattr(obj, 'area', 0)),
                    })
            except Exception as e:
                pass  # HSV 失败不影响 YOLO
        
        # 3. Qwen3 VL 场景理解 (语义级检测)
        vlm_description = ""
        try:
            import base64
            import boto3
            _, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(jpeg_buf.tobytes()).decode('utf-8')
            
            bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
            vlm_response = bedrock.converse(
                modelId='qwen.qwen3-vl-235b-a22b',
                messages=[{
                    'role': 'user',
                    'content': [
                        {'image': {'format': 'jpeg', 'source': {'bytes': jpeg_buf.tobytes()}}},
                        {'text': '请描述图片中桌面上的所有物体，包括它们的位置关系、颜色、大小。用中文回答，简洁明了。'}
                    ]
                }],
                inferenceConfig={'maxTokens': 300}
            )
            vlm_description = vlm_response['output']['message']['content'][0]['text']
        except Exception as e:
            vlm_description = f"场景分析不可用: {str(e)[:50]}"
        
        return json.dumps({
            "objects": results,
            "count": len(results),
            "yolo_count": sum(1 for r in results if r.get("source") == "yolo"),
            "hsv_count": sum(1 for r in results if r.get("source") == "hsv"),
            "vlm_scene_description": vlm_description,
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

    @tool
    def measure_depth(pixel_x: int = None, pixel_y: int = None) -> str:
        """
        用 Gemini 335 深度相机测量距离。
        用于抓取前确认物体的真实距离 (Z 深度), 避免只靠 2D 猜高度。

        Args:
            pixel_x: 查询点的像素 X (按彩色图分辨率)。不传=画面中心。
            pixel_y: 查询点的像素 Y。不传=画面中心。

        Returns:
            该点的深度 (mm) + 整幅深度统计 (最近/中位/最远)
        """
        if not camera or not hasattr(camera, "latest_depth"):
            return json.dumps({"error": "深度不可用 (需 Gemini 335 相机)"})
        depth = camera.latest_depth()
        if depth is None:
            return json.dumps({"error": "未取到深度帧"})
        h, w = depth.shape
        # 默认画面中心
        px = w // 2 if pixel_x is None else int(pixel_x)
        py = h // 2 if pixel_y is None else int(pixel_y)
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        z = camera.depth_at(px, py)
        valid = depth[depth > 0]
        stats = {}
        if valid.size > 0:
            scale = getattr(camera, "_depth_scale", 1.0) or 1.0
            stats = {
                "nearest_mm": round(float(valid.min()) * scale, 1),
                "median_mm": round(float(np.median(valid)) * scale, 1),
                "farthest_mm": round(float(valid.max()) * scale, 1),
                "valid_ratio": round(float(valid.size) / depth.size, 2),
            }
        return json.dumps({
            "status": "ok",
            "query_pixel": [px, py],
            "depth_mm": round(float(z), 1) if (z and z > 0) else None,
            "depth_resolution": [w, h],
            "scene": stats,
        })

    return [detect_objects, move_to, open_gripper, close_gripper, 
            home, get_current_position, move_joints, set_gripper_position,
            measure_depth]


# ─── Agent Setup ─────────────────────────────────────────────

def create_dummy_agent(robot, camera=None, detector=None, calibration=None,
                       hailo=None,
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
    
    tools = create_agent_tools(robot, camera, detector, calibration, hailo)
    
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
    
    # 初始化相机 (RGB, /dev/video0 on Pi5)
    import cv2
    camera = cv2.VideoCapture(0)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("📷 相机已连接 (640x480 RGB)")
    else:
        print("⚠️  相机未检测到，视觉功能不可用")
        camera = None

    # 颜色方块检测器
    detector = None
    if camera:
        try:
            from vision.color_detector import ColorBlockDetector
            detector = ColorBlockDetector()
            print("🎨 颜色检测器已加载")
        except ImportError:
            print("⚠️  颜色检测器不可用")

    # 手眼标定 (TODO: 标定完成后加载参数)
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
