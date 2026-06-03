# Dummy Robot V2 Demo

基于稚晖君 Dummy V2 机械臂的 LLM 驱动智能抓取演示。

## 系统架构

### 分层部署

```
┌─── 云端 (GPU) ───────────────────────────────┐
│                                               │
│  VLA 模型 / LLM (高级规划)                     │
│  - VLA: RT-2 / Octo / π0 / OpenVLA           │
│    输入: 图像 + 语言指令 → 输出: 动作序列      │
│  - LLM: Claude (Bedrock) / GPT-4o            │
│    输入: 场景描述 + 指令 → 输出: 结构化动作     │
│                                               │
│  部署选项: Bedrock / SageMaker / 自建GPU       │
└──────────────┬────────────────────────────────┘
               │ API (~100-300ms)
┌──────────────▼──── Pi 5 (独立运行) ───────────┐
│                                               │
│  感知层 (Hailo-8 NPU, 本地实时)                │
│  - YOLOv8 物体检测/分割 30+ FPS               │
│  - 视觉伺服 / 安全检查                         │
│                                               │
│  执行层 (Pi CPU, 本地实时)                     │
│  - FK/IK 解算 + 轨迹插值                      │
│  - 碰撞检测                                    │
│  - 串口通信 → STM32 → 电机                    │
└───────────────────────────────────────────────┘
```

**分工原则：**
- 云端：思考 — "要做什么/怎么做"（延迟不敏感，100-300ms OK）
- Pi Hailo：感知 — "看到什么"（必须实时，30FPS）
- Pi CPU：执行 — "怎么动"（必须实时，运动控制不能有网络延迟）

**单次决策延迟：**
```
拍照 → Hailo 检测 (30ms) → 上传图像到云端 (50-100ms)
→ VLA/LLM 推理 (100-200ms) → 返回动作 (50ms) → IK+执行 (10ms)
总计: ~300-400ms/决策 (桌面抓取完全够用)
```

### 硬件连接

| 设备 | Pi 5 接口 | 说明 |
|------|-----------|------|
| Orbbec 深度相机 | USB 3.0 | RGB+深度，高带宽 |
| Dummy 机械臂 | USB 2.0 | STM32 CDC 串口 (`/dev/ttyACM0`) |
| Hailo-8 NPU | PCIe (M.2 HAT) | CLB AI Developer Kit |
| NVMe 256GB | PCIe (同 HAT) | YOOMAGO YM680, 挂载 `/data/` |
| 网络 | WiFi / 网线 | Bedrock API + SSH |

### 代码结构

```
dummy-demo/
├── main.py                  # 主入口 (交互/测试)
├── driver/
│   ├── dummy_serial.py      # ✅ 串口驱动 (USB CDC ASCII)
│   └── dummy_can.py         # 旧版 fibre/CAN 驱动 (未完成)
├── brain/
│   ├── llm_planner.py       # LLM 任务规划 (Bedrock Claude)
│   └── task_executor.py     # 抓取任务执行器
├── vision/
│   ├── camera.py            # ✅ Orbbec RGBD 相机
│   ├── detector.py          # 颜色检测 (旧方案)
│   └── hailo_detector.py   # TODO: Hailo NPU 检测 (新方案)
└── SETUP_LOG.md             # 调试记录
```

## 硬件

- **机械臂**: Dummy V2 (6-DOF), STM32F405 + CAN 总线 + 步进电机
- **夹爪**: 通过 J6 机械联动 (J6+ 合拢, J6- 张开, ±90° 范围)
- **相机**: Orbbec 深度摄像头 (RGB 1920x1080 + 深度 640x400)
- **控制板**: REF 主控 (STM32F405)

## 连接方式

```
Mac USB-C (翻面) → STM32 USB CDC → CAN 总线 → 电机
                   /dev/cu.usbmodemXXX @ 115200
```

**注意**: USB Type-C 有正反面，翻面才是 STM32 CDC。

## 快速开始

```bash
# 安装依赖
pip install pyserial numpy opencv-python

# 硬件测试
python main.py --test

# 相机测试 (需要 pyorbbecsdk)
python main.py --camera

# 交互控制
python main.py
```

## 交互命令

| 命令 | 说明 |
|------|------|
| `home` | 回零位并使能 |
| `status` | 查看状态 |
| `grip open/close` | 夹爪开合 |
| `j 1 45` | J1 转到 45° |
| `goto 0,0,90,0,0,0` | 6 轴关节运动 |
| `cart 227.5,0,324.5,0,90,0` | 笛卡尔运动 |
| `raw !HOME` | 发送原始命令 |

## 开机 / 关机

### 开机顺序

1. 接通电源
2. 插入 USB-C（**翻面**连 STM32）
3. `!HOME` — 回零展开
4. `!START` — 使能电机
5. 开始操作

### 关机顺序 ⚠️

```
1. !RESET     → 折叠收纳（等 3-5 秒到位）
2. !DISABLE   → 失能电机
3. 拔电源
4. 拔 USB-C
```

**⚠️ 注意：**
- **先 `!RESET` 再 `!DISABLE`**，顺序反了电机没力，臂会塌
- **不要在 `!HOME`（展开）状态直接断电** → 臂会砸下来
- 折叠到位后靠机械结构平衡，断电安全

## Hailo-8 AI 加速模块

通过 CLB AI Developer Kit 连接 Pi 5，搭载 **Hailo-8 NPU (8 TOPS)**，用于视觉推理加速。

### 对 dummy-demo 的作用

| 能力 | 原方案 (HSV 颜色过滤) | Hailo-8 加速 |
|------|----------------------|-------------|
| 物体检测 | 只能识别纯色物体 | YOLOv8 任意物体检测 |
| 分割 | 无 | 实例分割 → 精确抓取点 |
| 姿态估计 | 无 | 物体朝向 → 最优抓取角度 |
| 帧率 | 快但简陋 | 30+ FPS 实时检测 |
| 鲁棒性 | 光照敏感 | 光照/遮挡/角度鲁棒 |

### 硬件配置

```
Pi 5 + CLB AI Developer Kit (M.2 HAT) + Hailo-8 + NVMe 256GB (YOOMAGO YM680)
```

### 资料位置

- 客户资料: `/Users/gyang/Projects/树莓派5 hailo 8 AI 模块客户资料 2024-11-9/`
- 预装镜像: `1.开发环境/搭建好环境的镜像/rpi_ai_hailo-2024-11-11.zip`
- 示例代码: `2.程序文档/hailo-rpi5-examples/` (YOLOv5/v8 检测、分割、姿态)
- 系统账号: `pi` / `123456` (首次登录后改密码)

### 集成方式

```python
# vision/detector.py 替换:
# 原: HSV 颜色过滤
objects = detector.detect(frame)  # 只能按颜色

# 新: Hailo NPU 推理
objects = hailo_detector.detect(frame)  # YOLOv8 on Hailo-8, 30+ FPS
# 返回格式兼容: [{name, color, x, y, confidence}]
```

### TODO

- [ ] 烧 SD 卡 (用预装镜像)
- [ ] 挂载 NVMe 为数据盘 `/data/`
- [ ] 验证 Hailo-8 检测 (`hailortcli fw-control identify`)
- [ ] 用 hailo-rpi5-examples 跑 YOLOv8 demo
- [ ] 写 `vision/hailo_detector.py` 替换颜色检测
- [ ] 训练自定义模型 (积木、杯子等桌面物品)

## 关键位姿

| 命令 | 关节角度 | 状态 |
|------|---------|------|
| `!HOME` (零位) | `0, 0, 90, 0, 0, 0` | 展开工作位 |
| `!RESET` (收纳) | `0, -72, 180, 0, 0, 0` | 折叠，可断电 |

- 笛卡尔零位: `227.50, 0, 324.50, 0, 90, 0`
