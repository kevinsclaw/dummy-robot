"""
Dummy V2 Web Frontend
=====================

简单的 Web UI，通过浏览器输入指令控制机械臂。
FastAPI + WebSocket 实时显示 Agent 思考过程和 tool calls。

启动:
    cd /home/pi/dummy-demo
    AWS_DEFAULT_REGION=us-east-1 python3 web/app.py --port /dev/dummy_arm

浏览器打开: http://<pi5-ip>:8080
"""

import sys
import os
import json
import asyncio
import argparse
import time
import threading
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from driver.dummy_serial import DummySerial
from brain.strands_agent import create_agent_tools, MockRobot, SYSTEM_PROMPT

app = FastAPI(title="Dummy V2 Controller")

# Global state
robot = None
agent_model = None
agent_tools = None
camera_instance = None  # 全局相机引用，用于视频流
hailo_detector = None   # 全局 Hailo 检测器，用于实时叠加


def get_index_html():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/")
async def index():
    return HTMLResponse(get_index_html())


from fastapi.responses import StreamingResponse
import cv2 as _cv2

def generate_mjpeg():
    """Generator 产生 MJPEG 帧，叠加 Hailo 检测结果"""
    import threading
    lock = threading.Lock()
    while True:
        if camera_instance is None or not camera_instance.isOpened():
            time.sleep(0.1)
            continue
        with lock:
            ret, frame = camera_instance.read()
        if not ret:
            time.sleep(0.03)
            continue

        # 叠加 Hailo 检测结果
        if hailo_detector and hailo_detector._started:
            try:
                detections = hailo_detector.detect(frame)
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    # 画框
                    color_map = {
                        "red": (0, 0, 255), "blue": (255, 0, 0),
                        "green": (0, 255, 0), "yellow": (0, 255, 255),
                        "orange": (0, 165, 255), "purple": (255, 0, 255),
                    }
                    box_color = color_map.get(det.color, (255, 255, 255))
                    _cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    # 标签
                    label = f"{det.label} {det.color} {det.confidence:.0%}"
                    _cv2.putText(frame, label, (x1, y1 - 8),
                                _cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                # 调试: 在左上角显示检测数量
                _cv2.putText(frame, f"YOLO: {len(detections)} obj", (10, 25),
                            _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                _cv2.putText(frame, f"YOLO ERR: {str(e)[:40]}", (10, 25),
                            _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        _, jpeg = _cv2.imencode('.jpg', frame, [_cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30fps


@app.get("/video_feed")
async def video_feed():
    """实时 MJPEG 视频流"""
    if camera_instance is None:
        return HTMLResponse("<h3>相机未连接</h3>", status_code=503)
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
    )


@app.get("/status")
async def status():
    if robot is None:
        return {"connected": False}
    try:
        pos = robot.get_cartesian_pose()
        joints = robot.get_joint_positions()
        return {
            "connected": True,
            "position": {"x": pos[0], "y": pos[1], "z": pos[2],
                         "a": pos[3], "b": pos[4], "c": pos[5]},
            "joints": joints,
        }
    except Exception as e:
        return {"connected": True, "error": str(e)}


def run_agent_with_streaming(command: str, event_queue: Queue):
    """
    Run agent and push events (thinking, tool_use, response) to queue.
    """
    from strands import Agent
    
    # Capture tool calls by wrapping
    tool_call_log = []
    
    # Create a fresh agent per request with callback
    agent = Agent(
        model=agent_model,
        tools=agent_tools,
        system_prompt=SYSTEM_PROMPT,
        callback_handler=lambda **kwargs: handle_agent_event(kwargs, event_queue),
    )
    
    try:
        result = agent(command)
        event_queue.put({"type": "response", "text": str(result)})
    except Exception as e:
        event_queue.put({"type": "error", "text": f"❌ Agent 错误: {str(e)}"})
    finally:
        event_queue.put({"type": "done"})


def handle_agent_event(event: dict, queue: Queue):
    """Process streaming events from Strands Agent"""
    # Strands callback_handler receives: data, complete, current_tool_use, etc.
    if "current_tool_use" in event and event.get("current_tool_use"):
        tool = event["current_tool_use"]
        if tool.get("name") and not tool.get("_reported"):
            queue.put({
                "type": "tool_call",
                "text": f"🔧 调用: {tool['name']}({json.dumps(tool.get('input', {}), ensure_ascii=False)[:200]})"
            })
            tool["_reported"] = True
    
    if "data" in event and event["data"]:
        # Streaming text chunk
        queue.put({"type": "thinking_chunk", "text": event["data"]})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            
            if msg.get("type") == "command":
                command = msg["text"]
                await ws.send_json({"type": "thinking", "text": f"🧠 理解指令: {command}"})
                
                # Run agent in thread, stream events back via queue
                event_queue = Queue()
                thread = threading.Thread(
                    target=run_agent_with_streaming,
                    args=(command, event_queue),
                    daemon=True
                )
                thread.start()
                
                # Stream events to WebSocket
                while True:
                    # Poll queue with small timeout
                    try:
                        event = await asyncio.to_thread(event_queue.get, timeout=0.1)
                    except:
                        if not thread.is_alive():
                            break
                        continue
                    
                    if event["type"] == "done":
                        break
                    await ws.send_json(event)
                
                # Send updated position
                try:
                    pos = robot.get_cartesian_pose()
                    joints = robot.get_joint_positions()
                    await ws.send_json({
                        "type": "status",
                        "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                        "joints": joints,
                    })
                except:
                    pass
                    
            elif msg.get("type") == "home":
                await ws.send_json({"type": "thinking", "text": "🏠 开机流程: HOME → ENABLE"})
                await asyncio.to_thread(robot.home)
                await asyncio.to_thread(time.sleep, 3)
                await asyncio.to_thread(robot.enable)
                await ws.send_json({"type": "response", "text": "✅ 开机完成！机械臂已展开并使能。"})

            elif msg.get("type") == "rehome":
                await ws.send_json({"type": "thinking", "text": "🔄 复位: HOME"})
                await asyncio.to_thread(robot.set_speed, 15)
                await asyncio.to_thread(robot.home)
                await asyncio.to_thread(time.sleep, 4)
                await asyncio.to_thread(robot.set_speed, 20)
                await ws.send_json({"type": "response", "text": "✅ 已复位到 HOME"})
                
                
            elif msg.get("type") == "reset":
                await ws.send_json({"type": "thinking", "text": "📦 关机流程: RESET (折叠) → DISABLE (失能)"})
                # RESET = 折叠收纳 (J1=0, J2=-72, J3=180, J4=0, J5=0, J6=0)
                await asyncio.to_thread(robot.move_joints, [0, -72, 180, 0, 0, 0])
                await asyncio.to_thread(time.sleep, 5)  # 等到位
                await asyncio.to_thread(robot.disable)
                await ws.send_json({"type": "response", "text": "✅ 关机完成！机械臂已折叠收纳并失能。\n现在可以安全断电。"})
                
                
            elif msg.get("type") == "estop":
                await asyncio.to_thread(robot.estop)
                await ws.send_json({"type": "response", "text": "⚠️ 急停！"})
                
    except WebSocketDisconnect:
        pass


def main():
    global robot, agent_model, agent_tools
    
    parser = argparse.ArgumentParser(description="Dummy V2 Web Controller")
    # 使用 udev symlink 固定设备名，避免 ttyACM 编号随枚举顺序变化
    # udev 规则: /etc/udev/rules.d/99-dummy-arm.rules
    # SUBSYSTEM=="tty", ATTRS{idVendor}=="1209", ATTRS{idProduct}=="0d32", SYMLINK+="dummy_arm"
    parser.add_argument("--port", type=str, default="/dev/dummy_arm", help="串口 (默认使用 udev symlink)")
    parser.add_argument("--mock", action="store_true", help="Mock 模式")
    parser.add_argument("--model", type=str, default="qwen.qwen3-vl-235b-a22b")
    parser.add_argument("--provider", type=str, default="bedrock")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8080)
    args = parser.parse_args()
    
    # Initialize robot
    if args.mock:
        robot = MockRobot()
        print("🤖 Mock 模式")
    else:
        robot = DummySerial(port=args.port)
        if not robot.connect():
            print("❌ 串口连接失败！")
            sys.exit(1)
        robot.home()
        time.sleep(3)
        robot.enable()
        print(f"✅ 机械臂已连接: {args.port}")
    
    # Initialize model
    if args.provider == "bedrock":
        from strands.models.bedrock import BedrockModel
        agent_model = BedrockModel(
            model_id=args.model,
            region_name="us-east-1",
        )
    else:
        from strands.models.openai import OpenAIModel
        agent_model = OpenAIModel(
            client_args={
                "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            },
            model_id=args.model,
        )
    
    # Initialize camera (RGB, /dev/video0 on Pi5)
    global camera_instance
    import cv2
    camera = cv2.VideoCapture(0)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera_instance = camera
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
        except ImportError as e:
            print(f"⚠️  颜色检测器不可用: {e}")

    # Hailo-8 YOLO 检测器 (实时叠加到视频流)
    global hailo_detector
    try:
        from vision.hailo_detector import HailoDetector
        hailo_detector = HailoDetector(
            model_path="/usr/share/hailo-models/yolov6n_h8.hef"
        )
        if hailo_detector.start():
            print("🧠 Hailo-8 检测器已启动 (YOLO 实时叠加)")
        else:
            print("⚠️  Hailo-8 启动失败，视频流不叠加检测")
            hailo_detector = None
    except ImportError:
        print("⚠️  Hailo SDK 未安装，视频流不叠加检测")
        hailo_detector = None

    # 手眼标定 (TODO: 标定完成后加载参数)
    calibration = None

    # Initialize tools
    agent_tools = create_agent_tools(robot, camera, detector, calibration)
    
    print(f"🧠 Model ready: {args.model}")
    print(f"🌐 Web UI: http://0.0.0.0:{args.web_port}")
    
    uvicorn.run(app, host=args.host, port=args.web_port)


if __name__ == "__main__":
    main()
