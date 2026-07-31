"""
LightGlue & RANSAC Feature Matcher for live drone vs retrieved satellite map patch matching.
Executes memory-efficient matrix multiplication (O(M*N) memory) nearest-neighbor matching
and estimates robust 2D Rigid/Affine similarity transformation matrix M via RANSAC with median fallback.
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
        Uses O(M*N) matrix multiplication instead of 3D broadcasting to prevent memory spikes.
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

        # Calculate cosine similarity matrix via dot product (O(M*N) memory)
        sim_matrix = np.dot(d0_norm, d1_norm.T)  # shape (M, N)
        matches01 = np.argmax(sim_matrix, axis=1)

        # Apply Lowe's ratio test with 0.9 threshold
        valid_matches = []
        for idx0, idx1 in enumerate(matches01):
            row = sim_matrix[idx0]
            best_sim = row[idx1]
            second_best_sim = np.partition(row, -2)[-2] if len(row) > 1 else 0.0

            best_dist = max(0.0, 2.0 - 2.0 * best_sim)
            second_dist = max(1e-6, 2.0 - 2.0 * second_best_sim)

            if best_dist < 0.9 * second_dist or best_sim > 0.6:
                valid_matches.append((idx0, idx1))

        if len(valid_matches) < 4:
            valid_matches = [(i, m) for i, m in enumerate(matches01[:min(len(matches01), 64)])]

        pts0 = np.float32([pts0_raw[i] for i, j in valid_matches])
        pts1 = np.float32([pts1_raw[j] for i, j in valid_matches])

        # Estimate robust Partial Affine Similarity matrix M [2x3] (Scale + Rotation + Translation)
        M, mask = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=8.0)

        # Fallback to robust median translation shift if affine matrix degenerates
        det = abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) if M is not None else 0.0
        if M is None or det < 0.05:
            dx = float(np.median(pts1[:, 0] - pts0[:, 0]))
            dy = float(np.median(pts1[:, 1] - pts0[:, 1]))
            M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
            inliers = 0  # CRITICAL FIX: If geometric RANSAC fails, inliers MUST be 0
        else:
            inliers = int(np.sum(mask)) if mask is not None else len(valid_matches)

        return inliers, M
