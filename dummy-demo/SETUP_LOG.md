# Dummy V2 机械臂配置日志

**日期**: 2026-04-30
**设备**: Dummy V2 机械臂（稚晖君设计）
**环境**: MacBook Air (Apple Silicon, M1, 8GB), macOS Sequoia

---

## 1. USB 连接排查

**问题**: 机械臂通过 USB 连接 Mac 后无法识别

**排查过程**:
- 通过扩展坞连接 → ❌ 不识别
- 直接插 MacBook Type-C 口 → ✅ 检测到 CP2102 芯片
- 串口设备: `/dev/cu.usbserial-0001`
- USB 信息: Product ID `0xea60`, Vendor ID `0x10c4` (Silicon Labs CP2102)

**结论**: 扩展坞不兼容 CP2102 串口芯片，必须直插 MacBook

---

## 2. ESP32 固件烧录

**问题**: ESP32 启动后报 `flash read err, 1000`，持续重启循环

**诊断**:
- 芯片: ESP32-PICO-D4 (revision v1.1)
- MAC: `f8:b3:b7:10:6b:90`
- 固件未正确烧录

**烧录工具安装**:
```bash
pip3 install esptool  # v5.2.0
```

**固件来源**:
- 路径: `1.Dummy V2任同学整理/2.Firmware/esp32-iot/firmware/examples/esp32/uart-bridge/dummy-iot.hex`
- ⚠️ 顶层 `hex/dummy-iot.hex` 是空文件(0 bytes)，不要用！

**从 Intel HEX 提取 bin 并烧录**:
```bash
# 用 Python intelhex 库提取三段
pip3 install intelhex

# 提取脚本生成三个 bin 文件:
# bootloader.bin   @ 0x1000   (26.2 KB)
# partitions.bin   @ 0x8000   (3.0 KB)
# app.bin          @ 0x100000 (883.7 KB)

# 烧录
esptool --port /dev/cu.usbserial-0001 --baud 460800 --chip esp32 \
  write_flash --flash_mode dio --flash_freq 40m --flash_size 4MB \
  0x1000 bootloader.bin \
  0x8000 partitions.bin \
  0x100000 app.bin
```

**结果**: ✅ 烧录成功，ESP32 正常启动，进入 CLI 模式

**分区表**:
| 分区 | 类型 | 偏移 | 大小 |
|------|------|------|------|
| nvs | data | 0x009000 | 24KB |
| phy_init | data | 0x00F000 | 4KB |
| storage (SPIFFS) | data | 0x010000 | 64KB |
| factory (app) | app | 0x100000 | 1024KB |

---

## 3. ESP32 WiFi 配置

**架构理解**:
```
USB → CP2102 → ESP32 UART0 (debug console / CLI)
                ESP32 UART1 (GPIO26 TX / GPIO27 RX) → STM32 UART4
WiFi → HTTP :80 / TCP :4001 / WS :4002 → UART bridge → STM32
```

ESP32 UART0（USB 串口）只是 debug console，**不透传**到 STM32。
要和 STM32 通信必须通过 WiFi 网络的 TCP:4001 或 WS:4002。

**CLI 命令** (WiFi 未配置时可用):
- `wifi SSID PASSWORD` — 配置 WiFi
- `ls` — 列出 SPIFFS 文件
- `cat <file>` — 读文件
- `rm <file>` — 删文件
- `reboot` — 重启

### 第一次尝试: Guest WiFi
```
wifi Guest <password>
```
- ✅ 连接成功，IP: 172.24.50.184
- ❌ Mac (172.24.50.83) ping 不通 ESP32 — **Guest WiFi 开了客户端隔离**
- ARP 显示 `(incomplete)`，确认是 AP 层面隔离

### 排除 VPN 干扰
- 检查路由表: 172.24.50.184 走 en0，路由正常
- 有多个 utun 隧道但只影响 IPv6 default route
- 确认是 WiFi AP 隔离问题，不是 VPN

### 第二次尝试: 手机热点
1. 擦除 SPIFFS 清除旧 WiFi 配置:
   ```bash
   esptool --port /dev/cu.usbserial-0001 --baud 460800 --chip esp32 \
     erase_region 0x010000 0x010000
   ```
2. 中文 SSID "一加13" → ❌ WiFi auth 失败（ESP32 中文编码问题）
3. 再次擦除 SPIFFS
4. 英文 SSID "oneplus13" → ✅ 连接成功
   ```
   wifi oneplus13 fbvf8755
   ```
5. `reboot` 重启后服务全部就绪

**最终网络状态**:
- Mac IP: 10.185.6.108
- ESP32 IP: 10.185.6.109
- Ping: ✅ 通了
- HTTP :80: ✅ (`/api/hi` 返回 "hi")
- TCP :4001: ✅ 可连接
- WS :4002: ✅ 可连接

### VPN 代理坑
- Mac 上 VPN 设置了 `ALL_PROXY=socks5://127.0.0.1:0`
- curl/Python 默认走代理导致连接失败
- 解决: `curl --noproxy '*'` 或 Python 中清除 `ALL_PROXY` 环境变量

---

## 4. STM32 主控板通信测试

**测试方式**: 通过 TCP:4001 → ESP32 UART bridge → STM32 UART4

**发送的命令**:
- 文本: `h`, `help`, `v`, `info`, `p` + 换行
- fibre 二进制: `0xAA 0x03 0x00`
- nc 直连

**结果**: ❌ **全部无响应**

**结论**: STM32F405 大概率没有烧录固件

---

## 5. macOS STM32 编译环境搭建

**安装工具链**:
```bash
# ARM 交叉编译器 (手动解压，绕过 brew cask 的 sudo 问题)
curl -L "https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/binrel/arm-gnu-toolchain-15.2.rel1-darwin-arm64-arm-none-eabi.tar.xz" -o /tmp/arm-toolchain.tar.xz
mkdir -p /opt/homebrew/opt/arm-toolchain
tar xf /tmp/arm-toolchain.tar.xz -C /opt/homebrew/opt/arm-toolchain --strip-components=1

# ST-Link 烧录工具
brew install stlink  # v1.8.0

# DFU 烧录工具
brew install dfu-util  # v0.11
```

**编译 STM32 主控固件**:
```bash
cd "1.Dummy V2任同学整理/2.Firmware/dummy-ref-core-fw"
export PATH="/opt/homebrew/opt/arm-toolchain/bin:$PATH"
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

**编译结果**: ✅ 成功
- 输出: `Core-STM32F4-fw.elf` / `.hex` / `.bin`
- Flash: 357KB / 1MB (34%)
- RAM: 46KB / 128KB (35%)
- CCMRAM: 64KB / 64KB (100%)

---

## 6. 虚拟机包分析

**文件**: `dummy-demo/vm.zip` (33GB)

**内容**: VirtualBox x86_64 虚拟机
- OS: Ubuntu 22.04 (kernel 6.8.0-94-generic)
- 配置: 8 核, 22GB RAM
- 硬盘: Clion1.vdi (82GB)

**结论**: ❌ 无法在 Apple Silicon Mac 上运行
- VirtualBox 不支持 ARM Mac 运行 x86 guest
- 内存需求 22GB > Mac 的 8GB
- 不需要了，本地编译环境已搭好
- **建议删除 vm.zip 省 33GB 空间**

---

## 7. 当前状态 & 下一步

### 已完成 ✅
- [x] ESP32 固件烧录
- [x] ESP32 WiFi 配置（手机热点）
- [x] ESP32 UART Bridge 服务运行
- [x] Mac ↔ ESP32 网络连通
- [x] macOS ARM 交叉编译环境
- [x] STM32 主控固件编译

### 待完成 ❌
- [ ] **烧录 STM32F405 主控固件** — 需要 ST-Link 调试器
- [ ] 确认 6pin 排针是否为 SWD 调试口（看丝印标注）
- [ ] 烧录后验证 STM32 ↔ ESP32 UART 通信
- [ ] 烧录 STM32F103 电机驱动板固件（35/42 步进电机板）
- [ ] fibre 协议 CLI 工具测试
- [ ] 运行 dummy-demo 代码控制机械臂

---

## 8. 成功控制机械臂！🎉

### 关键发现：USB Type-C 有正反面！

使用指南（Page 10）原文：
> USB（typeC）口有正反，上电后电脑识别不到别慌，拔出来，翻一面

**同一个 Type-C 口**：
- **一面** → ESP32 CP2102 串口 (`/dev/cu.usbserial-0001`) — 用于 ESP32 固件烧录和 WiFi 配置
- **翻面** → STM32 USB CDC (`/dev/cu.usbmodem2067307F534B1`) — 用于直接控制机械臂

之前一直连的是 ESP32 那面，翻转后立刻识别到 STM32：
```
REF 1.0 CDC Interface
Product ID: 0x0d32
Vendor ID: 0x1209
Serial Number: 2067307F534B
Manufacturer: Robot Embedded Framework
```

### STM32 串口通信确认

```python
import serial
ser = serial.Serial('/dev/cu.usbmodem2067307F534B1', 115200, timeout=2)
```

### 控制流程

```
1. !HOME      → 校准回零（关节归零位）
2. !START     → 使能电机
3. &J1,J2,J3,J4,J5,J6   → 关节角度控制
   @X,Y,Z,A,B,C         → 末端位姿控制（笛卡尔）
4. !DISABLE   → 失能电机
5. !STOP      → 紧急停止
```

### 命令参考

| 命令 | 说明 | 示例 |
|------|------|------|
| `!HOME` | 校准回零位 | `!HOME` |
| `!START` | 使能电机 | `!START` → `Started ok` |
| `!STOP` | 紧急停止 | `!STOP` → `Stopped ok` |
| `!DISABLE` | 失能电机 | `!DISABLE` → `Disabled ok` |
| `!CALIBRATION` | 校准零位偏移 | `!CALIBRATION` → `calibration ok` |
| `!RESET` | 复位 | `!RESET` |
| `#GETJPOS` | 读关节角度 | → `ok 0.00 0.00 90.00 0.00 0.00 0.00` |
| `#GETLPOS` | 读笛卡尔位置 | → `ok 227.50 0.00 324.50 0.00 90.00 0.00` |
| `#CMDMODE N` | 设置命令模式 | `#CMDMODE 1` |
| `&J1,J2,J3,J4,J5,J6` | 发送关节角度 | `&10,0,90,0,0,0` |
| `@X,Y,Z,A,B,C` | 发送末端位姿 | `@227,0,340,0,90,0` |
| `>...` | 顺序轨迹点 | |

### 零位参数
- 关节零位: `0, 0, 90, 0, 0, 0`
- 笛卡尔零位: `X=227.50, Y=0.00, Z=324.50, A=0.00, B=90.00, C=0.00`

### 运动测试结果 ✅

```
!HOME       → 回零成功
!START      → Started ok
&10,0,90,0,0,0  → J1 转 10°，到位 ✓
&0,0,90,0,0,0   → J1 回零，到位 ✓
@227,0,340,0,90,0  → 末端前移，到位 ✓
@227.5,0,324.5,0,90,0 → 末端回零，到位 ✓
```

---

## 总结

### 关键路径（两条独立通道）
```
控制通道: Mac → USB-C(翻面) → STM32 USB CDC → 电机
配置通道: Mac → USB-C(正面) → CP2102 → ESP32 → WiFi/UART bridge
```

### 设备标识
| 设备 | 串口 | 用途 |
|------|------|------|
| ESP32 CP2102 | `/dev/cu.usbserial-0001` | ESP32 固件烧录/WiFi 配置 |
| STM32 USB CDC | `/dev/cu.usbmodem2067307F534B1` | **机械臂控制** |

### 踩坑总结
1. **扩展坞不识别 CP2102** — 直插 MacBook
2. **ESP32 固件为空** — 用 esptool 从 HEX 提取 bin 烧录
3. **中文 WiFi SSID 连接失败** — 改英文名
4. **Guest WiFi 客户端隔离** — 换手机热点
5. **VPN 代理劫持本地流量** — `--noproxy '*'`
6. **ESP32 UART bridge 发命令 STM32 无响应** — 因为一直连的是 ESP32 那面，**翻转 USB-C 后直连 STM32 才是正确方式**
7. **USB Type-C 有正反面** — 这是最关键的发现！
