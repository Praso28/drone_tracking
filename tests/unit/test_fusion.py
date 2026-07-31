"""
Unit tests for EKF fusion and PoseSmoother modules.
"""

import pytest
from src.fusion.ekf_node import SimpleEKFFusion, PoseSmoother


def test_pose_smoother_acceptance():
    smoother = PoseSmoother(window_size=5, max_distance_m=15.0)
    p1 = {"latitude": 27.2000, "longitude": 76.2350}
    p2 = {"latitude": 27.2001, "longitude": 76.2351}

    res1 = smoother.filter(p1)
    res2 = smoother.filter(p2)
    assert res1["accepted"] is True
    assert res2["accepted"] is True


def test_pose_smoother_outlier_rejection():
    smoother = PoseSmoother(window_size=5, max_distance_m=15.0, warmup_fixes=1)
    p1 = {"latitude": 27.2000, "longitude": 76.2350}
    p_outlier = {"latitude": 28.2000, "longitude": 77.2350}  # ~140km away

    smoother.filter(p1)
    res_outlier = smoother.filter(p_outlier)
    assert res_outlier["accepted"] is False
    assert "Outlier rejected" in res_outlier["reason"]


def test_simple_ekf_fusion_cold_start():
    ekf = SimpleEKFFusion(init_lat=27.2000, init_lon=76.2350)
    fix1 = {"latitude": 27.2010, "longitude": 76.2360, "heading_deg": 10.0}
    
    # Cold-start snap: ekf should snap instantly to fix1
    updated1 = ekf.update_visual_fix(fix1)
    assert updated1["latitude"] == 27.2010
    assert updated1["longitude"] == 76.2360
    assert updated1["heading_deg"] == 10.0

    # Warm update: blending alpha=0.9
    fix2 = {"latitude": 27.2020, "longitude": 76.2370, "heading_deg": 20.0}
    updated2 = ekf.update_visual_fix(fix2)
    # lat = 0.1 * 27.2010 + 0.9 * 27.2020 = 27.2019
    assert abs(updated2["latitude"] - 27.2019) < 1e-6
    assert abs(updated2["longitude"] - 76.2369) < 1e-6
    assert updated2["heading_deg"] == 20.0
