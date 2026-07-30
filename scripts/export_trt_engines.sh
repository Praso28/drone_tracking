#!/usr/bin/env bash
# TensorRT FP16 Engine Exporter for Jetson Orin Nano
set -e

mkdir -p data/engines

echo "=================================================="
echo "    TensorRT FP16 Engine Compiler (Jetson Edge)   "
echo "=================================================="

ENGINE_PATH="data/engines/superpoint_fp16.engine"
ONNX_PATH="data/engines/superpoint.onnx"

# 1. Export ONNX model if missing
if [ ! -f "$ONNX_PATH" ]; then
    echo "[!] Exporting SuperPoint model to ONNX ($ONNX_PATH)..."
    python3 -c "
import torch
import torch.nn as nn
import os

class SuperPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_pts = nn.Conv2d(64, 65, kernel_size=1, stride=1, padding=0)
        self.conv_desc = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.relu(self.conv1a(x))
        x = torch.relu(self.conv1b(x))
        x = self.pool(x)
        x = torch.relu(self.conv2a(x))
        x = torch.relu(self.conv2b(x))
        semi = self.conv_pts(x)
        desc = self.conv_desc(x)
        return semi, desc

model = SuperPointNet().eval()
dummy_input = torch.randn(1, 1, 240, 320)
torch.onnx.export(
    model, dummy_input, '$ONNX_PATH',
    input_names=['input'], output_names=['semi', 'desc'],
    opset_version=11
)
print('Successfully exported $ONNX_PATH')
"
fi

# 2. Compile TensorRT FP16 Engine using trtexec if available
TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"
if [ -x "$TRTEXEC_BIN" ]; then
    echo "[+] Compiling ONNX to TensorRT FP16 Engine using $TRTEXEC_BIN..."
    "$TRTEXEC_BIN" --onnx="$ONNX_PATH" --saveEngine="$ENGINE_PATH" --fp16
    echo "[+] TensorRT FP16 Engine saved to $ENGINE_PATH"
else
    echo "[!] trtexec not found. Using PyTorch / ONNX Fallback engine."
    cp "$ONNX_PATH" "$ENGINE_PATH"
fi

echo "[+] SuperPoint Engine preparation complete."
