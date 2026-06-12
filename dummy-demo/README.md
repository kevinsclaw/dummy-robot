# Dummy Robot V2 Demo

基于稚晖君 Dummy V2 机械臂的 **LLM Agent 驱动智能抓取**演示。

## 🎬 Quick Start

```bash
# Pi5 上启动 Web 控制器
cd /home/pi/dummy-demo
AWS_DEFAULT_REGION=us-east-1 python3 web/app.py --port /dev/ttyACM0

# 浏览器打开
http://<pi5-ip>:8080
```

Mock 模式（不接硬件）：
```bash
python3 web/app.py --mock
```

## 系统架构

```
┌─── 浏览器 (Web UI) ───────────────────────────┐
│  指令输入 → WebSocket → 实时显示思考过程       │
└──────────────┬────────────────────────────────┘
               │ WebSocket
┌──────────────▼──── Pi 5 ──────────────────────┐
│                                               │
│  🧠 Strands Agent (Qwen3 VL via Bedrock)      │
│  - ReAct 循环: 思考 → 调用 Tool → 观察结果    │
│  - 8 个 Tools: move_to, gripper, detect...    │
│                                               │
│  👁 视觉 (Hailo-8 NPU / OpenCV HSV)          │
│  - YOLOv8 物体检测 266 FPS                    │
│  - HSV 颜色方块检测 (RGB-only fallback)        │
│                                               │
│  🦾 执行 (STM32 USB CDC 串口)                 │
│  - 笛卡尔/关节运动                            │
│  - 夹爪控制 (J6 联动)                         │
└───────────────────────────────────────────────┘
```

## 代码结构

```
dummy-demo/
├── web/
│   ├── app.py               # FastAPI + WebSocket 服务
│   └── index.html           # 前端 UI (实时思考过程)
├── brain/
│   └── strands_agent.py     # Strands ReAct Agent (8 tools)
├── driver/
│   └── dummy_serial.py      # STM32 USB CDC 串口驱动
├── vision/
│   ├── color_detector.py    # HSV 颜色方块检测 (RGB-only)
│   ├── hailo_detector.py    # Hailo-8 YOLO 检测器
│   └── camera.py            # 相机抽象层
├── sync_to_pi.sh            # rsync 同步到 Pi5
└── README.md
```

## Agent Tools

| Tool | 功能 | 状态 |
|------|------|------|
| `move_to` | 笛卡尔运动 (x,y,z mm) | ✅ |
| `move_joints` | 关节运动 (6轴角度) | ✅ |
| `open_gripper` | 张开夹爪 | ✅ |
| `close_gripper` | 合拢夹爪 | ✅ |
| `set_gripper_position` | 夹爪精确控制 (0-1) | ✅ |
| `home` | 回零位 | ✅ |
| `get_current_position` | 读取当前位姿 | ✅ |
| `detect_objects` | 检测工作台物体 | ✅ (RGB-only) |

## 硬件

| 设备 | 接口 | 说明 |
|------|------|------|
| Dummy V2 机械臂 | USB CDC `/dev/ttyACM0` | 6-DOF, STM32F405 + CAN |
| Hailo-8 NPU | PCIe M.2 | 8 TOPS, YOLOv8 @ 266 FPS |
| USB 摄像头 | USB 2.0 | RGB 方块检测 |
| Pi 5 | - | 主控, 运行 Agent + 视觉 + 执行 |

### USB Type-C 注意

```
USB-C 有正反面:
  一面 → STM32 CDC (需要这面)
  翻面 → ESP32 CP2102 (不要这面)
```

## 模型配置

默认使用 **Qwen3 VL 235B** (通过 AWS Bedrock)：
- 从中国 IP 直连无地理限制 ✅
- 支持视觉输入 (VL)
- Tool calling 能力强

```bash
# 使用 Bedrock (默认)
python3 web/app.py --model qwen.qwen3-vl-235b-a22b --provider bedrock

# 使用阿里云 DashScope
export DASHSCOPE_API_KEY="sk-xxx"
python3 web/app.py --model qwen-vl-max --provider dashscope
```

## 开发部署

```bash
# Mac 上开发
cd /Users/gyang/.openclaw/workspace/dummy-demo

# 同步到 Pi5
./sync_to_pi.sh
# 或手动: scp -r brain/ vision/ driver/ web/ pi@172.20.10.4:/home/pi/dummy-demo/

# Pi5 依赖
pip3 install --break-system-packages boto3 strands-agents strands-agents-bedrock fastapi uvicorn websockets pyserial opencv-python numpy
```

## 安全操作

### 开机
1. 接通电源
2. 插 USB-C（正确面 → STM32 CDC）
3. 启动 Web 服务
4. 自动 HOME + ENABLE

### 关机 ⚠️
```
1. !RESET     → 折叠收纳（等 3-5 秒）
2. !DISABLE   → 失能电机
3. 断电
```
**不要在展开状态直接断电** — 机械臂会塌下来。

## 关键位姿

| 状态 | 笛卡尔 (mm) | 关节 (°) |
|------|-------------|---------|
| HOME (展开) | 227.5, 0, 324.5 | 0, 0, 90, 0, 0, 0 |
| RESET (收纳) | - | 0, -72, 180, 0, 0, 0 |

## TODO

- [x] Strands Agent + Tool calling
- [x] Web UI (实时思考过程)
- [x] Qwen3 VL via Bedrock (无地理限制)
- [x] 真实硬件控制验证
- [x] Hailo-8 恢复 (内核 6.12.87, 266 FPS)
- [ ] 摄像头接入 + detect_objects 实测
- [ ] 手眼标定 (4 点仿射)
- [ ] 彩色方块抓取 demo
- [ ] Hailo YOLOv8 替换 HSV 检测
- [ ] 语音输入
