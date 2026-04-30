# Dummy Robot V2 Demo

基于稚晖君 Dummy V2 机械臂的 LLM 驱动智能抓取演示。

## 架构

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
│   └── detector.py          # 颜色检测
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

## 关键位姿

| 命令 | 关节角度 | 状态 |
|------|---------|------|
| `!HOME` (零位) | `0, 0, 90, 0, 0, 0` | 展开工作位 |
| `!RESET` (收纳) | `0, -72, 180, 0, 0, 0` | 折叠，可断电 |

- 笛卡尔零位: `227.50, 0, 324.50, 0, 90, 0`
