"""
Dummy Robot V2 运动学库
=======================

基于固件 DOF6Kinematic 的 DH 参数实现正运动学（FK）和逆运动学（IK）。
使用**标准 DH** (Standard DH) 约定，注意与 Picker 的 Modified DH 不同！

标准 DH 变换矩阵:
    T_i = Rz(θ) · Tz(d) · Tx(a) · Rx(α)

欧拉角约定: ZYX (A=Rz, B=Ry, C=Rx)，与固件一致。

单位：毫米 (mm)，角度 (degrees)
"""

import numpy as np
from numpy import sin, cos, arctan2, sqrt, pi

# ============================================================
# DH 参数定义（Standard DH Convention）
# 固件格式: {theta_offset(rad), d, a, alpha}
# 这里转换为: [theta_offset(deg), d(mm), a(mm), alpha(deg)]
# ============================================================

DH_PARAMS = np.array([
    [0,    126.5,  35.0,  -90],   # Link 1: 底座旋转
    [-90,  0,      146.0,  0],    # Link 2: 大臂
    [90,   52.0,   0,      90],   # Link 3: 肘部
    [0,    117.0,  0,     -90],   # Link 4: 前臂
    [0,    0,      0,      90],   # Link 5: 腕部偏转
    [0,    75.5,   0,      0],    # Link 6: 腕部旋转
])

# 关节限位（度）— 来自固件 DummyRobot 构造函数
JOINT_LIMITS = {
    1: (-180, 180),   # J0, 减速比 1
    2: (-170, 170),   # J1, 减速比 50
    3: (-75, 90),     # J2, 减速比 50, inverse
    4: (0, 180),      # J3, 减速比 50, inverse
    5: (-180, 180),   # J4, 减速比 50, inverse
    6: (-100, 120),   # J5, 减速比 50, inverse
}

# 连杆参数（毫米）— 来自固件 DOF6Kinematic 构造函数
L_BASE = 126.5    # 底座高度
D_BASE = 35.0     # 底座偏移
L_ARM = 146.0     # 大臂长度
L_FOREARM = 117.0 # 前臂长度
D_ELBOW = 52.0    # 肘部偏移
L_WRIST = 75.5    # 腕部到末端


def deg2rad(deg):
    """度转弧度"""
    return deg * pi / 180.0


def rad2deg(rad):
    """弧度转度"""
    return rad * 180.0 / pi


def sdh_transform(theta_deg, d, a, alpha_deg):
    """
    标准 DH 变换矩阵 (Standard DH Convention)
    T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)

    与 Picker 的 Modified DH 不同！标准 DH 的矩阵形式:
    [cos(θ)  -sin(θ)cos(α)   sin(θ)sin(α)   a·cos(θ)]
    [sin(θ)   cos(θ)cos(α)  -cos(θ)sin(α)   a·sin(θ)]
    [0        sin(α)          cos(α)          d       ]
    [0        0               0               1       ]
    """
    theta = deg2rad(theta_deg)
    alpha = deg2rad(alpha_deg)

    ct, st = cos(theta), sin(theta)
    ca, sa = cos(alpha), sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1]
    ])


def forward_kinematics(joint_angles_deg):
    """
    正运动学（FK）：关节角度 → 末端位姿 4x4 矩阵

    参数:
        joint_angles_deg: 6个关节角度 (度)

    返回:
        4x4 齐次变换矩阵
    """
    q = np.array(joint_angles_deg, dtype=float)
    T = np.eye(4)

    for i in range(6):
        theta_offset = DH_PARAMS[i, 0]
        d_i = DH_PARAMS[i, 1]
        a_i = DH_PARAMS[i, 2]
        alpha_i = DH_PARAMS[i, 3]
        theta_i = theta_offset + q[i]
        Hi = sdh_transform(theta_i, d_i, a_i, alpha_i)
        T = T @ Hi

    return T


def fk_to_xyzrpy(joint_angles_deg):
    """正运动学：关节角度 → [x, y, z, rx, ry, rz] (ZYX 欧拉角，度)"""
    T = forward_kinematics(joint_angles_deg)
    return pose_to_xyzrpy(T)


def pose_to_xyzrpy(T):
    """
    从 4x4 变换矩阵提取 [x, y, z, rx, ry, rz]

    ZYX 欧拉角分解: R = Rz(rz) · Ry(ry) · Rx(rx)
    与固件 RotMatToEulerAngle 一致:
      B = atan2(-R[2,0], sqrt(R[0,0]² + R[1,0]²))
      A = atan2(R[1,0]/cos(B), R[0,0]/cos(B))
      C = atan2(R[2,1]/cos(B), R[2,2]/cos(B))
    输出: rx=C(roll), ry=B(pitch), rz=A(yaw)
    """
    x, y, z = T[0, 3], T[1, 3], T[2, 3]

    # R[2,0] = -sin(B)
    sb = np.clip(-T[2, 0], -1.0, 1.0)
    if abs(sb) < 0.99999:
        ry = np.arcsin(sb)
        cb = cos(ry)
        rz = arctan2(T[1, 0] / cb, T[0, 0] / cb)
        rx = arctan2(T[2, 1] / cb, T[2, 2] / cb)
    else:
        # 万向节死锁
        ry = pi / 2 * np.sign(sb)
        rz = 0.0
        rx = arctan2(T[0, 1], T[1, 1])

    return np.array([x, y, z, rad2deg(rx), rad2deg(ry), rad2deg(rz)])


def xyzrpy_to_pose(xyzrpy):
    """
    [x, y, z, rx, ry, rz] → 4x4 变换矩阵

    ZYX 欧拉角: R = Rz(rz) · Ry(ry) · Rx(rx)
    与固件 EulerAngleToRotMat 一致
    """
    x, y, z = xyzrpy[0], xyzrpy[1], xyzrpy[2]
    rx = deg2rad(xyzrpy[3])  # C (roll)
    ry = deg2rad(xyzrpy[4])  # B (pitch)
    rz = deg2rad(xyzrpy[5])  # A (yaw)

    cc, sc = cos(rx), sin(rx)
    cb, sb = cos(ry), sin(ry)
    ca, sa = cos(rz), sin(rz)

    # R = Rz(A) · Ry(B) · Rx(C)
    T = np.array([
        [ca * cb,  ca * sb * sc - sa * cc,  ca * sb * cc + sa * sc, x],
        [sa * cb,  sa * sb * sc + ca * cc,  sa * sb * cc - ca * sc, y],
        [-sb,      cb * sc,                 cb * cc,                z],
        [0,        0,                       0,                      1]
    ])
    return T


def check_joint_limits(joint_angles_deg):
    """检查关节限位，返回 (valid, violations)"""
    violations = []
    for i in range(6):
        lo, hi = JOINT_LIMITS[i + 1]
        if joint_angles_deg[i] < lo - 0.01 or joint_angles_deg[i] > hi + 0.01:
            violations.append((i + 1, joint_angles_deg[i], lo, hi))
    return len(violations) == 0, violations


# ============================================================
# 数值逆运动学（基于 Jacobian + Damped Least Squares）
# ============================================================

def _jacobian(joint_angles_deg, delta=0.01):
    """数值雅可比矩阵 (6x6)"""
    J = np.zeros((6, 6))
    base = fk_to_xyzrpy(joint_angles_deg)

    for i in range(6):
        q_plus = np.array(joint_angles_deg, dtype=float)
        q_plus[i] += delta
        fk_plus = fk_to_xyzrpy(q_plus)
        diff = fk_plus - base

        # 角度差归一化到 [-180, 180]
        for j in range(3, 6):
            while diff[j] > 180:
                diff[j] -= 360
            while diff[j] < -180:
                diff[j] += 360

        J[:, i] = diff / delta

    return J


def inverse_kinematics(target_xyzrpy, joint_estimate=None, max_iter=200,
                       tol_pos=0.05, tol_ang=0.1, damping=0.5):
    """
    数值逆运动学（Damped Least Squares / Levenberg-Marquardt）

    参数:
        target_xyzrpy: [x, y, z, rx, ry, rz] (mm, 度)
        joint_estimate: 初始关节估计值 (度)，默认 [0]*6
        max_iter: 最大迭代次数
        tol_pos: 位置收敛容差 (mm)
        tol_ang: 姿态收敛容差 (度)
        damping: 阻尼系数

    返回:
        关节角度 (度) 的 numpy 数组，或 None（未收敛）
    """
    if joint_estimate is None:
        joint_estimate = [0.0, 0.0, 0.0, 90.0, 0.0, 0.0]  # Dummy 的 J3 范围是 0~180

    q = np.array(joint_estimate, dtype=float)
    target = np.array(target_xyzrpy, dtype=float)

    for iteration in range(max_iter):
        current = fk_to_xyzrpy(q)
        error = target - current

        # 角度差归一化
        for j in range(3, 6):
            while error[j] > 180:
                error[j] -= 360
            while error[j] < -180:
                error[j] += 360

        pos_err = np.linalg.norm(error[:3])
        ang_err = np.linalg.norm(error[3:6])

        if pos_err < tol_pos and ang_err < tol_ang:
            # 夹紧到限位
            for i in range(6):
                lo, hi = JOINT_LIMITS[i + 1]
                q[i] = np.clip(q[i], lo, hi)
            valid, _ = check_joint_limits(q)
            if valid:
                return q
            else:
                return q  # 接近但超限，也返回

        J = _jacobian(q)
        # Damped Least Squares: dq = J^T (J J^T + λ²I)^-1 e
        JJT = J @ J.T
        dq = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(6), error)

        # 限制步长
        max_step = 5.0  # 度
        step_norm = np.max(np.abs(dq))
        if step_norm > max_step:
            dq *= max_step / step_norm

        q += dq

        # 软限位约束
        for i in range(6):
            lo, hi = JOINT_LIMITS[i + 1]
            q[i] = np.clip(q[i], lo - 5, hi + 5)

    # 未收敛，检查是否接近
    current = fk_to_xyzrpy(q)
    error = target - current
    for j in range(3, 6):
        while error[j] > 180:
            error[j] -= 360
        while error[j] < -180:
            error[j] += 360

    pos_err = np.linalg.norm(error[:3])
    if pos_err < 1.0:  # 放宽到 1mm
        for i in range(6):
            lo, hi = JOINT_LIMITS[i + 1]
            q[i] = np.clip(q[i], lo, hi)
        return q

    return None


def inverse_kinematics_best(target_xyzrpy, joint_estimate=None):
    """
    尝试多组初始值，选择最优 IK 解

    返回:
        最优关节角度 (度) 或 None
    """
    if joint_estimate is None:
        joint_estimate = [0.0, 0.0, 0.0, 90.0, 0.0, 0.0]

    solutions = []

    # 从给定估计开始
    sol = inverse_kinematics(target_xyzrpy, joint_estimate)
    if sol is not None:
        solutions.append(sol)

    # 尝试不同初始值（注意 Dummy 的关节限位和 Picker 不同）
    seeds = [
        [0, 0, 0, 90, 0, 0],
        [0, -30, 45, 90, -45, 0],
        [0, 30, -30, 90, 45, 0],
        [90, 0, 0, 90, 0, 0],
        [-90, 0, 0, 90, 0, 0],
        [0, 0, 45, 120, 0, 0],
        [0, -60, 60, 60, 0, 0],
    ]
    for seed in seeds:
        sol = inverse_kinematics(target_xyzrpy, seed)
        if sol is not None:
            is_dup = False
            for prev in solutions:
                if np.allclose(sol, prev, atol=1.0):
                    is_dup = True
                    break
            if not is_dup:
                solutions.append(sol)

    if not solutions:
        return None

    best = min(solutions,
               key=lambda s: np.sum((np.array(s) - np.array(joint_estimate))**2))
    return best


# ============================================================
# 便捷函数
# ============================================================

def fk(joint_angles_deg):
    """FK 快捷方式：返回 [x, y, z, rx, ry, rz]"""
    return fk_to_xyzrpy(joint_angles_deg)


def ik(target_xyzrpy, joint_estimate=None):
    """IK 快捷方式：返回最优关节角度"""
    return inverse_kinematics_best(target_xyzrpy, joint_estimate)


if __name__ == "__main__":
    print("=== Dummy V2 运动学测试 ===\n")

    # 零位 FK
    zero_pos = [0, 0, 0, 0, 0, 0]
    result = fk(zero_pos)
    print(f"零位 FK: x={result[0]:.2f}, y={result[1]:.2f}, z={result[2]:.2f} mm")
    print(f"         rx={result[3]:.2f}, ry={result[4]:.2f}, rz={result[5]:.2f} deg")

    # 关节中位 FK（J3 范围 0~180，中位约 90）
    mid_pos = [0, 0, 0, 90, 0, 0]
    result_mid = fk(mid_pos)
    print(f"\nJ3=90° FK: x={result_mid[0]:.2f}, y={result_mid[1]:.2f}, z={result_mid[2]:.2f} mm")

    # FK → IK → FK 闭环测试
    test_configs = [
        [0, 0, 0, 90, 0, 0],
        [30, -20, 30, 90, -30, 0],
        [-45, 15, -15, 120, 20, -30],
    ]

    print("\n--- FK → IK → FK 闭环测试 ---")
    for q_orig in test_configs:
        target = fk(q_orig)
        q_ik = ik(target, q_orig)
        if q_ik is not None:
            fk_check = fk(q_ik)
            pos_err = np.linalg.norm(target[:3] - fk_check[:3])
            status = "✓" if pos_err < 0.1 else "✗"
            print(f"q={q_orig} → FK={np.round(target[:3],1)} → IK → FK误差={pos_err:.4f} mm {status}")
        else:
            print(f"q={q_orig} → IK 无解！ ✗")
