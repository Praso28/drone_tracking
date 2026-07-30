"""
Unit tests for shared/geo/tile_math.py and pose_math.py modules.
"""

import pytest
import math
import numpy as np
from shared.geo.tile_math import (
    latlon_to_tile,
    tile_to_latlon,
    calculate_gsd,
    pixel_to_latlon,
    haversine_distance,
)
from shared.geo.pose_math import (
    augment_patch_rotations,
    homography_to_translation,
    ransac_pose_vote,
    compute_geopose,
)


def test_latlon_tile_roundtrip():
    lat, lon, zoom = 27.2000, 76.2350, 18
    tx, ty = latlon_to_tile(lat, lon, zoom)
    rec_lat, rec_lon = tile_to_latlon(tx, ty, zoom)
    assert abs(rec_lat - lat) < 0.01
    assert abs(rec_lon - lon) < 0.01


def test_calculate_gsd():
    gsd = calculate_gsd(27.2000, zoom=18, tile_size_px=256)
    assert 0.1 < gsd < 2.0


def test_haversine_distance_known():
    # Delhi to Jaipur ~ 235km
    delhi_lat, delhi_lon = 28.6139, 77.2090
    jaipur_lat, jaipur_lon = 26.9124, 75.7873
    dist_m = haversine_distance(delhi_lat, delhi_lon, jaipur_lat, jaipur_lon)
    assert 220000 < dist_m < 260000


def test_haversine_distance_zero():
    dist = haversine_distance(27.2000, 76.2350, 27.2000, 76.2350)
    assert dist == 0.0


def test_pixel_to_latlon():
    lat, lon = 27.2000, 76.2350
    new_lat, new_lon = pixel_to_latlon(lat, lon, dx_px=10.0, dy_px=0.0, gsd_m_per_px=0.5)
    assert new_lat == lat
    assert new_lon > lon


def test_augment_patch_rotations():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    rotations = augment_patch_rotations(img)
    assert len(rotations) == 4
    angles = [r[0] for r in rotations]
    assert angles == [0, 90, 180, 270]


def test_homography_to_translation():
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = 5.0  # dx
    H[1, 2] = -3.0 # dy
    dx, dy, yaw = homography_to_translation(H, (100.0, 100.0))
    assert abs(dx - 5.0) < 1e-5
    assert abs(dy - (-3.0)) < 1e-5
    assert abs(yaw) < 1e-5


def test_ransac_pose_vote_valid():
    candidates = [
        {"patch_id": 1, "inliers": 50, "dx_px": 2.0, "dy_px": 1.0, "yaw_deg": 0.0},
        {"patch_id": 2, "inliers": 10, "dx_px": 0.0, "dy_px": 0.0, "yaw_deg": 90.0},
    ]
    res = ransac_pose_vote(candidates, inlier_threshold=15)
    assert res["valid"] is True
    assert res["patch_id"] == 1


def test_ransac_pose_vote_invalid():
    candidates = [
        {"patch_id": 1, "inliers": 5, "dx_px": 2.0, "dy_px": 1.0, "yaw_deg": 0.0},
    ]
    res = ransac_pose_vote(candidates, inlier_threshold=15)
    assert res["valid"] is False


def test_compute_geopose():
    res = compute_geopose(27.2000, 76.2350, dx_px=0.0, dy_px=0.0, yaw_deg=45.0, gsd_m_per_px=0.5)
    assert abs(res["latitude"] - 27.2000) < 1e-6
    assert abs(res["longitude"] - 76.2350) < 1e-6
    assert res["heading_deg"] == 45.0
