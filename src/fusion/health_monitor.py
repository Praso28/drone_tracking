"""
Sensor Health Monitor & Fail-Safe Fallback Cascade.
Evaluates per-frame visual tracking quality score and manages smooth transitions between
visual tracking (TRACKING), IMU dead-reckoning fallback (DEGRADED), and MAVLink Emergency Loiter/RTL (EMERGENCY).
"""

import time
from typing import Dict, Any
from shared.logging_cfg import setup_logger

logger = setup_logger("health_monitor")


class SensorHealthMonitor:
    """Monitors visual localization quality and controls emergency fallback cascade."""

    def __init__(
        self,
        min_inliers: int = 15,
        max_lost_duration_sec: float = 10.0
    ):
        self.min_inliers = min_inliers
        self.max_lost_duration_sec = max_lost_duration_sec
        self.state = "TRACKING"
        self.last_valid_fix_time = time.time()

    def evaluate_step(
        self,
        inliers: int,
        pose_accepted: bool
    ) -> Dict[str, Any]:
        """
        Evaluates current frame tracking metrics and updates system health state.
        Returns health evaluation status and recommended flight action.
        """
        now = time.time()
        is_valid = pose_accepted and (inliers >= self.min_inliers)

        if is_valid:
            self.last_valid_fix_time = now
            self.state = "TRACKING"
            return {
                "health_state": "TRACKING",
                "quality_score": min(1.0, inliers / 100.0),
                "action": "NORMAL_FLIGHT"
            }

        lost_duration = now - self.last_valid_fix_time

        if lost_duration <= self.max_lost_duration_sec:
            self.state = "DEGRADED"
            logger.warning(f"Visual tracking degraded ({lost_duration:.1f}s lost). Falling back to IMU dead-reckoning.")
            return {
                "health_state": "DEGRADED",
                "quality_score": 0.3,
                "action": "IMU_DEAD_RECKONING"
            }

        self.state = "EMERGENCY"
        logger.error(f"EMERGENCY: Visual tracking lost for {lost_duration:.1f}s > {self.max_lost_duration_sec}s! Triggering MAVLink RTL/Loiter.")
        return {
            "health_state": "EMERGENCY",
            "quality_score": 0.0,
            "action": "MAVLINK_LOITER_RTL"
        }
