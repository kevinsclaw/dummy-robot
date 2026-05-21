# Dummy Robot V2

开源 6-DOF 桌面机械臂 — 基于[稚晖君 Dummy](https://github.com/peng-zhihui/Dummy-Robot) V2 版本的 LLM 智能控制方案。

## 项目结构

```
dummy-robot/
├── dummy-sim/       # 仿真环境 (运动学 + 3D 可视化)
├── dummy-demo/      # 实机控制 (串口驱动 + LLM 规划 + RGBD 视觉)
└── README.md
```

## 硬件

| 组件 | 型号 |
|------|------|
| 机械臂 | Dummy V2 (6-DOF + 夹爪) |
| 主控 | STM32F405 (REF 控制板) |
| 深度相机 | Orbbec (RGB 1080p + 深度 640x400) |
| 通信 | USB CDC 串口 (ASCII 协议) |

## 快速开始

### 仿真

```bash
cd dummy-sim
pip install numpy matplotlib
python visualize.py
```

### 实机控制

```bash
cd dummy-demo
pip install pyserial numpy opencv-python

# 硬件连通性测试
python main.py --test

# 交互控制
python main.py
```

> ⚠️ USB Type-C 有正反面 — **翻面**才连接 STM32 CDC。

## 功能

- **运动学引擎** — 标准 DH 正/逆运动学，经过固件源码验证
- **串口控制** — 关节空间 & 笛卡尔空间运动，夹爪开合
- **RGBD 视觉** — Orbbec 深度相机，颜色检测 + 定位
- **LLM 规划** — AWS Bedrock Claude 自然语言→抓取任务 (WIP)

## 安全注意事项

- 开机顺序: 上电 → `!HOME` → `!START`
- **关机顺序: `!RESET` (折叠) → `!DISABLE` → 断电**
- 不要在展开状态断电，臂会自由坠落

## 致谢

- [稚晖君](https://github.com/peng-zhihui) — 原始设计
- 木子晓汶 — PCB 二次开发
- 任同学 — V2 版本整理

## License

MIT
