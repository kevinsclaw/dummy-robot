"""
Dummy V2 Web Frontend
=====================

简单的 Web UI，通过浏览器输入指令控制机械臂。
FastAPI + WebSocket 实时显示 Agent 思考过程和 tool calls。

启动:
    cd /home/pi/dummy-demo
    AWS_DEFAULT_REGION=us-east-1 python3 web/app.py --port /dev/ttyACM0

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


def get_index_html():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/")
async def index():
    return HTMLResponse(get_index_html())


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
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="串口")
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
    
    # Initialize tools
    agent_tools = create_agent_tools(robot)
    
    print(f"🧠 Model ready: {args.model}")
    print(f"🌐 Web UI: http://0.0.0.0:{args.web_port}")
    
    uvicorn.run(app, host=args.host, port=args.web_port)


if __name__ == "__main__":
    main()
