# Dummy Robot V2 Demo

基于稚晖君 Dummy V2 机械臂的 LLM 驱动智能抓取演示。

## 架构

```
dummy-demo/
├── main.py                 # 主入口
├── driver/
│   └── dummy_can.py        # CAN/fibre 通信驱动
├── brain/
│   ├── llm_planner.py      # LLM 任务规划 (Bedrock Claude)
│   └── task_executor.py    # 抓取任务执行器
└── vision/
    ├── camera.py            # 深度相机接口
    └── detector.py          # 目标检测
```

## 硬件

- Dummy V2 机械臂 (6-DOF + 手爪)
- REF 控制板 (STM32F405 + CAN 总线)
- 42/35 步进电机 + 谐波减速器
- 深度相机 (待定)

## 通信

Dummy 使用 **CAN 总线** 通过 **fibre 协议** 通信，不同于 Picker 的串口方式。
连接方式：USB → REF 控制板 → CAN 总线 → 各电机驱动板

## 运行

```bash
# 安装依赖
pip install pyserial numpy

# 连接 Dummy 后运行
python main.py
```
