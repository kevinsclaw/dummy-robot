#!/usr/bin/env python3
"""
Dummy Robot V2 Demo 主入口
==========================

LLM 驱动的智能抓取演示。
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger('dummy-demo')


def main():
    logger.info("🤖 Dummy V2 Demo 启动")
    logger.info("=" * 50)

    # 初始化驱动
    from driver.dummy_can import DummyCAN
    robot = DummyCAN()

    logger.info("尝试连接 Dummy...")
    if not robot.connect():
        logger.warning("未检测到 Dummy 硬件，进入模拟模式")

    # 回零
    robot.home()
    logger.info(f"当前关节: {robot.get_joint_positions()}")

    # 交互循环
    logger.info("\n输入指令 (help 查看帮助, quit 退出):")
    while True:
        try:
            cmd = input("dummy> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if cmd in ('quit', 'exit', 'q'):
            break
        elif cmd == 'help':
            print("""
可用指令:
  home          回零位
  status        查看状态
  grip open     张开手爪
  grip close    闭合手爪
  j <id> <deg>  单轴运动 (如: j 1 45)
  estop         急停
  quit          退出
""")
        elif cmd == 'home':
            robot.home()
        elif cmd == 'status':
            s = robot.get_status()
            print(f"  连接: {'✅' if s['connected'] else '❌'}")
            print(f"  关节: {[f'{j:.1f}' for j in s['joints']]}")
            print(f"  手爪: {s['gripper']*100:.0f}%")
        elif cmd == 'grip open':
            robot.open_gripper()
        elif cmd == 'grip close':
            robot.close_gripper()
        elif cmd.startswith('j '):
            parts = cmd.split()
            if len(parts) == 3:
                try:
                    jid, deg = int(parts[1]), float(parts[2])
                    robot.move_joint(jid, deg)
                except ValueError:
                    print("格式: j <关节号> <角度>")
        elif cmd == 'estop':
            robot.estop()
        elif cmd:
            print(f"未知指令: {cmd}，输入 help 查看帮助")

    robot.disconnect()
    logger.info("Dummy V2 Demo 已退出")


if __name__ == '__main__':
    main()
