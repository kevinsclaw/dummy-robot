# Dummy Robot V2 仿真环境

基于稚晖君 Dummy 机械臂 V2 版本（6-DOF + 手爪）的仿真。

## 文件结构

```
dummy-sim/
├── kinematics.py          # Python 正/逆运动学库
├── test_kinematics.py     # 单元测试
├── visualize.py           # matplotlib 3D 交互可视化
├── urdf/
│   └── dummy_v2.urdf      # 机器人 URDF 模型
└── meshes/                # STL 模型文件 (base, j1~j6)
```

## 快速开始

```bash
cd dummy-sim
source ../picker-sim/.venv/bin/activate  # 共用 venv
python visualize.py
```

## 运动学参数

| 参数 | 值 (mm) | 说明 |
|------|---------|------|
| L_BASE | 109.0 | 底座高度 |
| D_BASE | 35.0 | 底座偏移 |
| L_ARM | 146.0 | 大臂长度 |
| L_FOREARM | 115.0 | 前臂长度 |
| D_ELBOW | 52.0 | 肘部偏移 |
| L_WRIST | 72.0 | 腕部长度 |

> 连杆参数来自固件 `dummy_robot.cpp`: `DOF6Kinematic(0.109, 0.035, 0.146, 0.115, 0.052, 0.072)`（单位米）。
> **FK 用固件几何向量法**（非纯 DH 矩阵累乘），否则 D_ELBOW 会被误甩到 Y 轴产生假偏移。

### DH 参数（标准 DH 约定，仅旋转部分与固件一致）

| Joint | θ offset (rad) | d (mm) | a (mm) | α (rad) |
|-------|----------------|--------|--------|---------|
| 1 | 0 | 109.0 | 35.0 | -π/2 |
| 2 | -π/2 | 0 | 146.0 | 0 |
| 3 | +π/2 | 52.0 | 0 | +π/2 |
| 4 | 0 | 115.0 | 0 | -π/2 |
| 5 | 0 | 0 | 0 | +π/2 |
| 6 | 0 | 72.0 | 0 | 0 |

### 关节限位

| Joint | Min (°) | Max (°) | 减速比 |
|-------|---------|---------|--------|
| J1 | -180 | 180 | 1:1 |
| J2 | -170 | 170 | 50:1 |
| J3 | -75 | 90 | 50:1 |
| J4 | 0 | 180 | 50:1 |
| J5 | -180 | 180 | 50:1 |
| J6 | -100 | 120 | 50:1 |

## 运动学库

```python
from kinematics import fk, ik

# 正运动学
pose = fk([0, 0, 90, 0, 0, 0])   # 真机 HOME = J3=90
# → [222.0, 0.0, 307.0, 0, 90, 0] (mm, deg)

# 逆运动学
joints = ik([100, 50, 300, 0, -45, 0])
```

## 致谢

- 稚晖君 (Zhihui Jun) — 原始设计
- 木子晓汶 — PCB 二次开发
- 任同学 — V2 版本整理
