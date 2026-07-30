"""
Homography-to-GeoPose calculations, 4-rotation patch augmentation,
multi-candidate RANSAC voting, and metric evaluation routines.
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Any
from .tile_math import pixel_to_latlon, haversine_distance


def augment_patch_rotations(patch_img: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Generates 4 rotated versions of a patch image: 0°, 90°, 180°, 270°."""
    rotations = [
        (0, patch_img),
        (90, np.rot90(patch_img, k=1)),
        (180, np.rot90(patch_img, k=2)),
        (270, np.rot90(patch_img, k=3)),
    ]
    return rotations


def homography_to_translation(
    M: np.ndarray,
    patch_center_px: Tuple[float, float]
) -> Tuple[float, float, float]:
    """
    Computes (dx_px, dy_px, yaw_deg) from a 2x3 Affine matrix or 3x3 Homography matrix M.
    Maps query center to reference patch coordinate system.
    """
    if M is None:
        return 0.0, 0.0, 0.0

    cx, cy = patch_center_px

    if M.shape == (2, 3):
        # 2x3 Affine Similarity Transformation Matrix [ [s*cos(theta), -s*sin(theta), tx], [s*sin(theta), s*cos(theta), ty] ]
        tx = float(M[0, 2])
        ty = float(M[1, 2])
        yaw_rad = math.atan2(M[1, 0], M[0, 0])
        yaw_deg = math.degrees(yaw_rad)

        # Center displacement offset in pixels
        dx_px = tx + M[0, 0] * cx + M[0, 1] * cy - cx
        dy_px = ty + M[1, 0] * cx + M[1, 1] * cy - cy
        return float(dx_px), float(dy_px), float(yaw_deg)

    elif M.shape == (3, 3):
        # Fallback 3x3 Homography projection
        query_center = np.array([cx, cy, 1.0], dtype=np.float64)
        ref_proj = M @ query_center
        if abs(ref_proj[2]) > 1e-8:
            ref_proj /= ref_proj[2]

        dx_px = ref_proj[0] - cx
        dy_px = ref_proj[1] - cy

        rotation_submatrix = M[:2, :2]
        u, _, vt = np.linalg.svd(rotation_submatrix)
        R = u @ vt
        yaw_rad = math.atan2(R[1, 0], R[0, 0])
        yaw_deg = math.degrees(yaw_rad)
        return float(dx_px), float(dy_px), float(yaw_deg)

    return 0.0, 0.0, 0.0


def ransac_pose_vote(
    candidates: List[Dict[str, Any]],
    inlier_threshold: int = 15
) -> Dict[str, Any]:
    """
    Selects the best visual matching candidate using RANSAC inlier count
    and rotation consensus voting.
    """
    if not candidates:
        return {"valid": False, "reason": "No candidates provided"}

    sorted_candidates = sorted(candidates, key=lambda c: c.get("inliers", 0), reverse=True)
    best = sorted_candidates[0]

    if best.get("inliers", 0) < inlier_threshold:
        return {
            "valid": False,
            "reason": f"Insufficient inliers ({best.get('inliers', 0)} < {inlier_threshold})",
            "best_candidate": best
        }

    return {
        "valid": True,
        "best_candidate": best,
        "inliers": best.get("inliers", 0),
        "patch_id": best.get("patch_id"),
        "dx_px": best.get("dx_px", 0.0),
        "dy_px": best.get("dy_px", 0.0),
        "yaw_deg": best.get("yaw_deg", 0.0),
    }


def compute_geopose(
    ref_lat: float,
    ref_lon: float,
    dx_px: float,
    dy_px: float,
    yaw_deg: float,
    gsd_m_per_px: float
) -> Dict[str, float]:
    """Calculates final latitude, longitude, heading from reference coordinate and offsets."""
    lat, lon = pixel_to_latlon(ref_lat, ref_lon, dx_px, dy_px, gsd_m_per_px)
    return {
        "latitude": lat,
        "longitude": lon,
        "heading_deg": (yaw_deg + 360.0) % 360.0,
    }
