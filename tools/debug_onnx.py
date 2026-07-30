"""
Diagnostic script to inspect SuperPoint ONNX output shapes on Jetson.
"""

import os
import sys
import numpy as np
import onnxruntime as ort

onnx_path = os.path.expanduser("~/Desktop/gazeebo_drone/data/engines/superpoint.onnx")
if not os.path.exists(onnx_path):
    onnx_path = "data/engines/superpoint.onnx"

print(f"Loading ONNX model from: {onnx_path}")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

print("=== ONNX Inputs ===")
for inp in session.get_inputs():
    print("Input:", inp.name, inp.shape, inp.type)

print("=== ONNX Outputs ===")
for out in session.get_outputs():
    print("Output:", out.name, out.shape, out.type)

dummy_img = np.random.randint(0, 255, (240, 320), dtype=np.uint8).astype(np.float32)[None, None, :, :] / 255.0
outs = session.run(None, {session.get_inputs()[0].name: dummy_img})

print("=== Run Outputs ===")
for i, o in enumerate(outs):
    arr = np.array(o)
    print(f"Output [{i}] shape:", arr.shape, "dtype:", arr.dtype)
    if arr.ndim >= 2:
        print(f"  Slice [0, :5]:", arr.reshape(-1)[:5])
