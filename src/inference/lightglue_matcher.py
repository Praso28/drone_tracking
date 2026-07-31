"""
LightGlue & RANSAC Feature Matcher for live drone vs retrieved satellite map patch matching.
Executes memory-efficient matrix multiplication (O(M*N) memory) nearest-neighbor matching
and estimates robust 2D Affine similarity transformation matrix M via RANSAC.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


class LightGlueMatcher:
    """Matches keypoints and descriptors between live drone query frame and candidate satellite map patch."""

    def __init__(self, match_threshold: float = 0.2):
        self.match_threshold = match_threshold

    def match(
        self,
        feats0: Dict[str, np.ndarray],
        feats1: Dict[str, np.ndarray]
    ) -> Tuple[int, np.ndarray]:
        """
        Matches live query features (feats0) against candidate satellite patch features (feats1).
        Uses Mutual Nearest Neighbor + RANSAC Affine Estimation.
        Returns (inlier_count, M_2x3).
        """
        kps0 = feats0.get("keypoints", np.zeros((0, 2)))
        kps1 = feats1.get("keypoints", np.zeros((0, 2)))
        desc0 = feats0.get("descriptors", np.zeros((0, 256)))
        desc1 = feats1.get("descriptors", np.zeros((0, 256)))

        if len(kps0) < 4 or len(kps1) < 4:
            return 0, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        # Cap max keypoints to prevent runaway computations
        max_k = 512
        pts0_raw = kps0[:max_k].astype(np.float32)
        pts1_raw = kps1[:max_k].astype(np.float32)
        d0 = desc0[:max_k].astype(np.float32)
        d1 = desc1[:max_k].astype(np.float32)

        # Normalize descriptors
        n0 = np.linalg.norm(d0, axis=1, keepdims=True)
        n1 = np.linalg.norm(d1, axis=1, keepdims=True)
        n0[n0 < 1e-6] = 1.0
        n1[n1 < 1e-6] = 1.0
        d0_norm = d0 / n0
        d1_norm = d1 / n1

        # Cosine similarity matrix via dot product (O(M*N) memory)
        sim_matrix = np.dot(d0_norm, d1_norm.T)  # shape (M, N)
        matches01 = np.argmax(sim_matrix, axis=1)
        matches10 = np.argmax(sim_matrix, axis=0)

        # Mutual Nearest Neighbor filtering
        max_sims = np.max(sim_matrix, axis=1)
        mutual_mask = (matches10[matches01] == np.arange(len(matches01))) & (max_sims > self.match_threshold)
        mutual_indices = np.where(mutual_mask)[0]

        if len(mutual_indices) < 4:
            # Fallback to top matches if mutual matches are under 4
            top_k = min(len(matches01), 32)
            top_indices = np.argsort(max_sims)[-top_k:]
            pts0 = pts0_raw[top_indices]
            pts1 = pts1_raw[matches01[top_indices]]
        else:
            pts0 = pts0_raw[mutual_indices]
            pts1 = pts1_raw[matches01[mutual_indices]]

        # Estimate robust 2D Affine transformation matrix M [2x3]
        M, mask = cv2.estimateAffine2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=8.0)

        det = abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) if M is not None else 0.0
        if M is None or det < 0.1 or det > 10.0:
            dx = float(np.median(pts1[:, 0] - pts0[:, 0])) if len(pts0) > 0 else 0.0
            dy = float(np.median(pts1[:, 1] - pts0[:, 1])) if len(pts0) > 0 else 0.0
            M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
            inliers = 0
        else:
            inliers = int(np.sum(mask)) if mask is not None else len(pts0)

        return inliers, M
