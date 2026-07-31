#!/usr/bin/env bash
# Jetson Orin Nano Edge Node Environment Bootstrapper
# Installs required runtime packages and creates workspace directories.

set -e

echo "=================================================================="
echo " Bootstrapping Jetson Orin Nano Edge Environment"
echo "=================================================================="

mkdir -p data/engines

echo "[1/3] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install pyzmq pymavlink opencv-python pyyaml faiss-cpu onnxruntime

echo "[2/3] Verifying CUDA execution provider for ONNX Runtime..."
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Available ONNX Providers:', providers)
if 'CUDAExecutionProvider' in providers:
    print('[+] CUDA Execution Provider is READY.')
else:
    print('[!] WARNING: CUDAExecutionProvider not detected. Install onnxruntime-gpu for maximum FPS.')
"

echo "[3/3] Setting CPU power mode..."
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0  # 15W MAXN power profile
    echo "[+] Jetson Power Profile set to 15W MAXN."
fi

echo "=================================================================="
echo "[+] Jetson Environment Setup Complete!"
echo "=================================================================="
