#!/bin/bash
# Dummy Demo — 同步到 Pi5
# 用法: ./sync_to_pi.sh [pi_host]

PI_HOST="${1:-pi@172.20.10.4}"
PI_PATH="/home/pi/dummy-demo"

echo "Syncing dummy-demo → ${PI_HOST}:${PI_PATH}"

rsync -avz --progress \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='vm.zip' \
    --exclude='*.egg-info/' \
    "$(dirname "$0")/" \
    "${PI_HOST}:${PI_PATH}/"

echo ""
echo "✅ Done! Run on Pi5:"
echo "   ssh ${PI_HOST}"
echo "   cd ${PI_PATH}"
echo "   python3 -m brain.strands_agent --mock  # 测试"
echo "   python3 -m brain.strands_agent          # 真实硬件"
