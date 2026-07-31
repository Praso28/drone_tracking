"""
Unit tests for NavigationPhaseController state machine transitions.
"""

import pytest
from src.fusion.health_monitor import NavigationPhaseController


def test_phase_controller_unanchored_to_tracking():
    ctrl = NavigationPhaseController(anchor_inliers_thresh=50, tracking_min_inliers=15)
    assert ctrl.phase == "UNANCHORED_ACQUISITION"
    assert ctrl.anchored is False

    # Low inliers -> stays unanchored
    res1 = ctrl.evaluate_step(inliers=10, pose_accepted=True, raw_pose={"latitude": 29.76, "longitude": 115.97})
    assert res1["phase"] == "UNANCHORED_ACQUISITION"
    assert res1["use_visual_fix"] is False

    # High inliers -> anchors to tracking
    res2 = ctrl.evaluate_step(inliers=60, pose_accepted=True, raw_pose={"latitude": 29.76, "longitude": 115.97})
    assert res2["phase"] == "HIGH_CONFIDENCE_TRACKING"
    assert res2["use_visual_fix"] is True
    assert ctrl.anchored is True


def test_phase_controller_tracking_to_imu_dead_reckoning():
    ctrl = NavigationPhaseController(anchor_inliers_thresh=50, tracking_min_inliers=15, max_dead_reckoning_sec=5.0)
    # Anchor first
    ctrl.evaluate_step(inliers=60, pose_accepted=True, raw_pose={"latitude": 29.76, "longitude": 115.97})

    # Drop inliers -> transitions to IMU_DEAD_RECKONING
    res = ctrl.evaluate_step(inliers=5, pose_accepted=False, raw_pose={"latitude": 29.76, "longitude": 115.97})
    assert res["phase"] == "IMU_DEAD_RECKONING"
    assert res["use_visual_fix"] is False
