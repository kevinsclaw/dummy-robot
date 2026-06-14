#!/usr/bin/env python3
"""
Dummy Robot V2 Demo
===================

LLM 驱动的智能抓取演示。
硬件: Dummy V2 机械臂 + Orbbec 深度相机。

使用:
    python main.py              # 交互模式
    python main.py --test       # 硬件测试
    python main.py --camera     # 相机测试
"""

import logging
import sys
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s'
)
logger = logging.getLogger('dummy-demo')


def test_hardware():
    """硬件连接测试"""
    from driver.dummy_serial import DummySerial

    print("=" * 50)
    print("  Dummy V2 硬件测试")
    print("=" * 50)

    robot = DummySerial()

    # 连接
    print("\n[1] 连接...")
    if not robot.connect():
        print("❌ 连接失败！确认 USB-C 线已翻面插入")
        return

    print("✅ 连接成功")

    # 读取位置
    print("\n[2] 读取关节...")
    joints = robot.get_joint_positions()
    print(f"  关节: {joints}")

    pose = robot.get_cartesian_pose()
    print(f"  位姿: {pose}")

    # 使能 + 小运动
    print("\n[3] 使能...")
    robot.enable()

    print("\n[4] J1 转 10°...")
    current = robot.get_joint_positions()
    current[0] = 10.0
    robot.move_joints(current)
    print(f"  位置: {robot.get_joint_positions()}")

    print("\n[5] J1 回 0°...")
    current[0] = 0.0
    robot.move_joints(current)

    # 夹爪
    print("\n[6] 夹爪张开...")
    robot.open_gripper()

    print("\n[7] 夹爪合拢...")
    robot.close_gripper()

    print("\n[8] 夹爪回零...")
    robot.set_gripper(0.5)

    # 清理
    robot.disable()
    robot.disconnect()
    print("\n✅ 硬件测试完成！")


def test_camera():
    """相机测试 (Gemini 335 / pyorbbecsdk v2, 回退 USB)"""
    from vision.orbbec_camera import open_camera

    print("=" * 50)
    print("  相机测试")
    print("=" * 50)

    cam, label = open_camera()
    if cam is None:
        print("❌ 未检测到相机")
        return
    print(f"📷 {label}")

    color = cam.read_color() if hasattr(cam, "read_color") else (cam.read()[1] if cam.read()[0] else None)
    if color is not None:
        print(f"✅ RGB: {color.shape}")
    else:
        print("❌ RGB 读取失败")

    depth = cam.latest_depth() if hasattr(cam, "latest_depth") else None
    if depth is not None and (depth > 0).any():
        print(f"✅ 深度: {depth.shape}, 范围 {depth[depth>0].min()}-{depth.max()} mm")
    else:
        print("⚠️ 深度不可用 (非 Gemini 335 或无有效深度)")

    if hasattr(cam, "stop"):
        cam.stop()
    elif hasattr(cam, "release"):
        cam.release()
    print("\n✅ 相机测试完成！")


def interactive():
    """交互控制模式"""
    from driver.dummy_serial import DummySerial

    print("🤖 Dummy V2 Demo")
    print("=" * 50)

    robot = DummySerial()

    print("连接中...")
    if not robot.connect():
        print("❌ 连接失败！请确认:")
        print("  1. USB-C 线已翻面插入 MacBook")
        print("  2. 机械臂已上电")
        return

    print("✅ 已连接")
    joints = robot.get_joint_positions()
    print(f"当前关节: {[f'{j:.1f}' for j in joints]}")

    print("\n输入指令 (help 查看帮助, quit 退出):")
    while True:
        try:
            cmd = input("dummy> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not cmd:
            continue

        cmd_lower = cmd.lower()

        if cmd_lower in ('quit', 'exit', 'q'):
            break
        elif cmd_lower == 'help':
            print("""
可用指令:
  home            回零位 (!HOME + !START)
  start           使能
  stop            急停
  disable         失能
  status          查看状态
  grip open       张开夹爪
  grip close      合拢夹爪
  j <id> <deg>    单轴运动 (如: j 1 45)
  goto <j1>,<j2>,<j3>,<j4>,<j5>,<j6>
                  关节运动 (如: goto 0,0,90,0,0,0)
  cart <x>,<y>,<z>,<a>,<b>,<c>
                  笛卡尔运动 (如: cart 227.5,0,324.5,0,90,0)
  raw <cmd>       发送原始命令
  quit            退出
""")
        elif cmd_lower == 'home':
            robot.home()
            robot.enable()
            print("已回零并使能")
        elif cmd_lower == 'start':
            robot.enable()
        elif cmd_lower == 'stop':
            robot.estop()
        elif cmd_lower == 'disable':
            robot.disable()
        elif cmd_lower == 'status':
            s = robot.get_status()
            print(f"  连接: {'✅' if s['connected'] else '❌'}")
            print(f"  使能: {'✅' if s['enabled'] else '❌'}")
            print(f"  关节: {[f'{j:.1f}' for j in s['joints']]}")
            print(f"  夹爪: {s['gripper_angle']:.1f}° ({s['gripper_pct']*100:.0f}%)")
            pose = robot.get_cartesian_pose()
            print(f"  位姿: X={pose[0]:.1f} Y={pose[1]:.1f} Z={pose[2]:.1f}")
        elif cmd_lower == 'grip open':
            robot.open_gripper()
            print("夹爪已张开")
        elif cmd_lower == 'grip close':
            robot.close_gripper()
            print("夹爪已合拢")
        elif cmd_lower.startswith('j '):
            parts = cmd.split()
            if len(parts) == 3:
                try:
                    jid, deg = int(parts[1]) - 1, float(parts[2])  # J1=index 0
                    robot.move_joint_single(jid, deg)
                    print(f"J{jid+1} → {deg}°")
                except ValueError:
                    print("格式: j <关节号1-6> <角度>")
            else:
                print("格式: j <关节号1-6> <角度>")
        elif cmd_lower.startswith('goto '):
            try:
                values = [float(x) for x in cmd[5:].split(',')]
                if len(values) == 6:
                    robot.move_joints(values)
                    print(f"移动到: {values}")
                else:
                    print("需要 6 个值: j1,j2,j3,j4,j5,j6")
            except ValueError:
                print("格式: goto j1,j2,j3,j4,j5,j6")
        elif cmd_lower.startswith('cart '):
            try:
                values = [float(x) for x in cmd[5:].split(',')]
                if len(values) == 6:
                    robot.move_cartesian(*values)
                    print(f"笛卡尔移动: {values}")
                else:
                    print("需要 6 个值: x,y,z,a,b,c")
            except ValueError:
                print("格式: cart x,y,z,a,b,c")
        elif cmd_lower.startswith('raw '):
            raw_cmd = cmd[4:]
            resp = robot._send(raw_cmd)
            print(f"  → {resp}")
        else:
            print(f"未知指令: {cmd}，输入 help 查看帮助")

    robot.disconnect()
    print("Dummy V2 Demo 已退出")


def main():
    parser = argparse.ArgumentParser(description='Dummy V2 Demo')
    parser.add_argument('--test', action='store_true', help='硬件测试')
    parser.add_argument('--camera', action='store_true', help='相机测试')
    args = parser.parse_args()

    if args.test:
        test_hardware()
    elif args.camera:
        test_camera()
    else:
        interactive()


if __name__ == '__main__':
    main()
