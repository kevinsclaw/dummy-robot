#!/usr/bin/env python3
"""紧急停止 — 立即发 STOP + DISABLE 并松夹爪"""
import sys
sys.path.insert(0, "/home/pi/dummy-demo")
from driver.dummy_serial import DummySerial

robot = DummySerial()
if robot.connect():
    print("急停中...")
    robot._send("!STOP")
    robot._send("!DISABLE")
    print("已急停 + 失能")
    robot.disconnect()
else:
    print("连接失败 — 直接拔 USB 或断电!")
