"""
TensorRT / ONNX / CPU SuperPoint feature extractor engine wrapper.
Parses raw 65-channel semi-dense heatmap logits and 256-channel descriptor maps
from SuperPoint ONNX into keypoints (x, y), confidence scores, and normalized descriptors.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any

from shared.logging_cfg import setup_logger

logger = setup_logger("trt_engine")


def decode_superpoint(
    semi: np.ndarray,
    desc: np.ndarray,
    max_keypoints: int = 512,
    threshold: float = 0.005
):
    """
    Decodes raw SuperPoint ONNX output tensors:
    semi: (1, 65, H_c, W_c) -> softmax -> heatmap -> (N, 2) keypoints
    desc: (1, 256, H_c, W_c) -> bilinear sample at keypoints -> (N, 256) descriptors
    """
    if semi.ndim == 3:
        semi = semi[None, ...]
    if desc.ndim == 3:
        desc = desc[None, ...]

    # 1. Softmax over 65 channels
    dense = np.exp(semi - np.max(semi, axis=1, keepdims=True))
    dense /= np.sum(dense, axis=1, keepdims=True)
    nodust = dense[:, :-1, :, :]  # remove dustbin channel -> (1, 64, H_c, W_c)

    _, _, H_c, W_c = nodust.shape
    # Reshape (1, 64, H_c, W_c) -> full resolution heatmap (H_c*8, W_c*8)
    heatmap = nodust[0].transpose(1, 2, 0).reshape(H_c, W_c, 8, 8)
    heatmap = heatmap.transpose(0, 2, 1, 3).reshape(H_c * 8, W_c * 8)

    # 2. Extract keypoints above confidence threshold
    ys, xs = np.where(heatmap > threshold)
    if len(xs) == 0:
        ys, xs = np.where(heatmap > threshold * 0.2)

    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0, 256), dtype=np.float32)

    scores = heatmap[ys, xs]

    # Top-K sorting
    if len(xs) > max_keypoints:
        top_idx = np.argsort(scores)[::-1][:max_keypoints]
        xs = xs[top_idx]
        ys = ys[top_idx]
        scores = scores[top_idx]

    keypoints_240 = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2) -> (x, y)

    # 3. Sample 256-dim descriptors from desc map (1, 256, H_c, W_c)
    desc_map = desc[0]  # (256, H_c, W_c)
    C_dim = desc_map.shape[0]
    descriptors = np.zeros((len(keypoints_240), C_dim), dtype=np.float32)

    for i in range(len(keypoints_240)):
        kx = float(keypoints_240[i][0])
        ky = float(keypoints_240[i][1])
        gx = int(clamp(kx * (W_c / 320.0), 0, W_c - 1))
        gy = int(clamp(ky * (H_c / 240.0), 0, H_c - 1))
        descriptors[i] = desc_map[:, gy, gx]

    # L2 normalization
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    descriptors /= norms

    return keypoints_240, scores.astype(np.float32), descriptors.astype(np.float32)


class SuperPointEngine:
    """SuperPoint feature extractor with TRT CUDA engine / ONNX / OpenCV feature fallback."""

    def __init__(
        self,
        engine_path: str = "data/engines/superpoint_fp16.engine",
        onnx_path: str = "data/engines/superpoint.onnx",
        max_keypoints: int = 512,
        descriptor_dim: int = 256
    ):
        self.engine_path = engine_path
        self.onnx_path = onnx_path
        self.max_keypoints = max_keypoints
        self.descriptor_dim = descriptor_dim
        self.ort_session = None
        
        # TRT state
        self.trt_context = None
        self.trt_engine = None
        self.d_input = None
        self.d_semi = None
        self.d_desc = None
        self.h_input = None
        self.h_semi = None
        self.h_desc = None
        self.stream = None

        if os.path.exists(engine_path):
            try:
                import tensorrt as trt
                import pycuda.driver as cuda
                import pycuda.autoinit
                
                TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                    self.trt_engine = runtime.deserialize_cuda_engine(f.read())
                
                self.trt_context = self.trt_engine.create_execution_context()
                
                # Allocate buffers (assuming fixed 1x1x240x320 input, 1x65x120x160 semi, 1x256x120x160 desc)
                self.h_input = cuda.pagelocked_empty((1, 1, 240, 320), dtype=np.float32)
                self.h_semi = cuda.pagelocked_empty((1, 65, 120, 160), dtype=np.float32)
                self.h_desc = cuda.pagelocked_empty((1, 256, 120, 160), dtype=np.float32)
                
                self.d_input = cuda.mem_alloc(self.h_input.nbytes)
                self.d_semi = cuda.mem_alloc(self.h_semi.nbytes)
                self.d_desc = cuda.mem_alloc(self.h_desc.nbytes)
                
                self.stream = cuda.Stream()
                logger.info(f"Loaded SuperPoint TensorRT engine from {engine_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorRT session: {e}")
                self.trt_context = None

        if self.trt_context is None and os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                logger.info(f"Loaded SuperPoint ONNX model from {onnx_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX session: {e}")

    def extract(self, image_gray: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts real visual keypoints, confidence scores, and L2-normalized descriptors from grayscale frame.
        """
        if image_gray.ndim == 3:
            image_gray = cv2.cvtColor(image_gray, cv2.COLOR_RGB2GRAY)

        orig_H, orig_W = image_gray.shape[:2]

        img_resized = cv2.resize(image_gray, (320, 240)) if (orig_H != 240 or orig_W != 320) else image_gray
        scale_x = float(orig_W) / 320.0
        scale_y = float(orig_H) / 240.0

        # 1. Run TensorRT if session is loaded
        if self.trt_context is not None:
            try:
                import pycuda.driver as cuda
                np.copyto(self.h_input, img_resized.astype(np.float32)[None, None, :, :] / 255.0)
                cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
                
                try:
                    # TensorRT 8.5+ API (Required for TRT 10)
                    self.trt_context.set_tensor_address("input", int(self.d_input))
                    self.trt_context.set_tensor_address("semi", int(self.d_semi))
                    self.trt_context.set_tensor_address("desc", int(self.d_desc))
                    self.trt_context.execute_async_v3(stream_handle=self.stream.handle)
                except AttributeError:
                    # Fallback for older TensorRT versions
                    self.trt_context.execute_async_v2(
                        bindings=[int(self.d_input), int(self.d_semi), int(self.d_desc)],
                        stream_handle=self.stream.handle
                    )
                
                cuda.memcpy_dtoh_async(self.h_semi, self.d_semi, self.stream)
                cuda.memcpy_dtoh_async(self.h_desc, self.d_desc, self.stream)
                self.stream.synchronize()
                
                semi_tensor = self.h_semi
                desc_tensor = self.h_desc
                
                keypoints_240, scores, descriptors = decode_superpoint(
                    semi_tensor, desc_tensor, max_keypoints=self.max_keypoints, threshold=0.005
                )

                # Rescale keypoints back to original frame resolution
                keypoints = keypoints_240.copy()
                if len(keypoints) > 0:
                    keypoints[:, 0] *= scale_x
                    keypoints[:, 1] *= scale_y

                # Adjust descriptors to target dimension if needed
                if descriptors.ndim == 2 and descriptors.shape[1] < self.descriptor_dim:
                    descriptors = np.pad(descriptors, ((0, 0), (0, self.descriptor_dim - descriptors.shape[1])))
                elif descriptors.ndim == 2 and descriptors.shape[1] > self.descriptor_dim:
                    descriptors = descriptors[:, :self.descriptor_dim]

                return {"keypoints": keypoints, "scores": scores, "descriptors": descriptors}
                
            except Exception as e:
                logger.warning(f"TensorRT extraction failed: {e}. Falling back.")

        # 2. Run ONNX Runtime if session is loaded
        if self.ort_session is not None:
            try:
                inp_tensor = img_resized.astype(np.float32)[None, None, :, :] / 255.0
                inp_name = self.ort_session.get_inputs()[0].name
                outs = self.ort_session.run(None, {inp_name: inp_tensor})

                semi_tensor = outs[0]
                desc_tensor = outs[1]

                keypoints_240, scores, descriptors = decode_superpoint(
                    semi_tensor, desc_tensor, max_keypoints=self.max_keypoints, threshold=0.005
                )

                # Rescale keypoints back to original frame resolution
                keypoints = keypoints_240.copy()
                if len(keypoints) > 0:
                    keypoints[:, 0] *= scale_x
                    keypoints[:, 1] *= scale_y

                # Adjust descriptors to target dimension if needed
                if descriptors.ndim == 2 and descriptors.shape[1] < self.descriptor_dim:
                    descriptors = np.pad(descriptors, ((0, 0), (0, self.descriptor_dim - descriptors.shape[1])))
                elif descriptors.ndim == 2 and descriptors.shape[1] > self.descriptor_dim:
                    descriptors = descriptors[:, :self.descriptor_dim]

                return {"keypoints": keypoints, "scores": scores, "descriptors": descriptors}

            except Exception as e:
                logger.warning(f"ONNX extraction failed: {e}. Falling back to FAST/gradient feature extractor.")

        # 2. Deterministic Corner & Gradient Visual Feature Extractor
        fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
        kp_objs = fast.detect(image_gray, None)

        if len(kp_objs) < 16:
            corners = cv2.goodFeaturesToTrack(image_gray, maxCorners=self.max_keypoints, qualityLevel=0.01, minDistance=5)
            if corners is not None:
                keypoints = corners.reshape(-1, 2).astype(np.float32)
            else:
                keypoints = np.zeros((0, 2), dtype=np.float32)
        else:
            kp_objs = sorted(kp_objs, key=lambda x: x.response, reverse=True)[:self.max_keypoints]
            keypoints = np.array([kp.pt for kp in kp_objs], dtype=np.float32)

        num_kps = len(keypoints)
        if num_kps == 0:
            return {
                "keypoints": np.zeros((0, 2), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "descriptors": np.zeros((0, self.descriptor_dim), dtype=np.float32)
            }

        descriptors = np.zeros((num_kps, self.descriptor_dim), dtype=np.float32)
        scores = np.ones(num_kps, dtype=np.float32)

        gx = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)

        for i, (x, y) in enumerate(keypoints):
            ix, iy = int(round(x)), int(round(y))
            patch_r = 8
            x0, x1 = max(0, ix - patch_r), min(orig_W, ix + patch_r)
            y0, y1 = max(0, iy - patch_r), min(orig_H, iy + patch_r)

            patch_gx = gx[y0:y1, x0:x1].flatten()
            patch_gy = gy[y0:y1, x0:x1].flatten()
            patch_desc = np.concatenate([patch_gx, patch_gy])

            if len(patch_desc) < self.descriptor_dim:
                patch_desc = np.pad(patch_desc, (0, self.descriptor_dim - len(patch_desc)))
            else:
                patch_desc = patch_desc[:self.descriptor_dim]

            norm = np.linalg.norm(patch_desc)
            descriptors[i] = patch_desc / norm if norm > 1e-6 else patch_desc

        return {
            "keypoints": keypoints,
            "scores": scores,
            "descriptors": descriptors
        }


def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))
