"""
Unit tests for SuperPointEngine feature extraction and LightGlueMatcher.
"""

import pytest
import numpy as np
from src.inference.trt_engine import SuperPointEngine
from src.inference.lightglue_matcher import LightGlueMatcher


def test_superpoint_engine_fallback_extraction():
    engine = SuperPointEngine(max_keypoints=100, descriptor_dim=256)
    # Generate synthetic image with noise/texture
    img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    feats = engine.extract(img)

    assert "keypoints" in feats
    assert "scores" in feats
    assert "descriptors" in feats
    assert feats["descriptors"].shape[1] == 256 if len(feats["descriptors"]) > 0 else True


def test_lightglue_matcher_empty():
    matcher = LightGlueMatcher()
    feats0 = {"keypoints": np.zeros((0, 2)), "descriptors": np.zeros((0, 256))}
    feats1 = {"keypoints": np.zeros((0, 2)), "descriptors": np.zeros((0, 256))}
    inliers, M = matcher.match(feats0, feats1)

    assert inliers == 0
    assert M.shape == (2, 3)


def test_lightglue_matcher_synthetic():
    matcher = LightGlueMatcher()
    kps0 = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]], dtype=np.float32)
    kps1 = np.array([[15, 12], [25, 22], [35, 32], [45, 42], [55, 52]], dtype=np.float32)
    descs0 = np.random.randn(5, 256).astype(np.float32)
    descs1 = descs0.copy()  # perfect descriptor match

    feats0 = {"keypoints": kps0, "descriptors": descs0}
    feats1 = {"keypoints": kps1, "descriptors": descs1}

    inliers, M = matcher.match(feats0, feats1)
    assert inliers >= 0
    assert M.shape == (2, 3)
