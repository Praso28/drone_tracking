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


def test_simple_ekf_fusion():
    ekf = SimpleEKFFusion(init_lat=27.2000, init_lon=76.2350)
    fix = {"latitude": 27.2010, "longitude": 76.2360, "heading_deg": 10.0}
    updated = ekf.update_visual_fix(fix)

    assert 27.2000 < updated["latitude"] < 27.2010
    assert 76.2350 < updated["longitude"] < 76.2360
    assert updated["heading_deg"] == 10.0
