"""
Intelligent Multi-Phase Navigation State Controller & Health Monitor.
Manages operational state transitions across:
  - Phase 1: UNANCHORED_ACQUISITION (Cold-Start Global Map Search)
  - Phase 2: HIGH_CONFIDENCE_TRACKING (Visual Primary Navigation)
  - Phase 3: IMU_DEAD_RECKONING (Sensor Propagation during Visual Dropout)
  - Phase 4: GLOBAL_REFIX (Trajectory Re-Anchoring upon High-Confidence Match)
  - Phase 5: EMERGENCY_HOLD (MAVLink Fail-Safe Loiter / Return-to-Launch)
"""

import time
from typing import Dict, Any
from shared.logging_cfg import setup_logger

logger = setup_logger("health_monitor")


class NavigationPhaseController:
    """Intelligent multi-stage state machine for GPS-Denied navigation."""

    def __init__(
        self,
        anchor_inliers_thresh: int = 150,
        tracking_min_inliers: int = 30,
        refix_inliers_thresh: int = 100,
        max_dead_reckoning_sec: float = 15.0
    ):
        self.anchor_inliers_thresh = anchor_inliers_thresh
        self.tracking_min_inliers = tracking_min_inliers
        self.refix_inliers_thresh = refix_inliers_thresh
        self.max_dead_reckoning_sec = max_dead_reckoning_sec

        self.phase = "UNANCHORED_ACQUISITION"
        self.anchored = False
        self.last_valid_fix_time = time.time()
        self.dead_reckoning_start = None

    def evaluate_step(
        self,
        inliers: int,
        pose_accepted: bool,
        raw_pose: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluates per-step visual match quality and controls multi-phase state transitions.
        """
        now = time.time()

        # PHASE 1: UNANCHORED_ACQUISITION (Cold-Start Global Search)
        if not self.anchored:
            if pose_accepted and inliers >= self.anchor_inliers_thresh:
                self.anchored = True
                self.phase = "HIGH_CONFIDENCE_TRACKING"
                self.last_valid_fix_time = now
                logger.info(f"PHASE 1 COMPLETE: Map Anchored with {inliers} inliers. Transitioning to HIGH_CONFIDENCE_TRACKING.")
                return {
                    "phase": "HIGH_CONFIDENCE_TRACKING",
                    "action": "GLOBAL_ANCHOR_SUCCESS",
                    "use_visual_fix": True,
                    "reset_imu": True,
                    "confidence": min(1.0, inliers / 300.0)
                }
            else:
                logger.info(f"PHASE 1: Searching global map... (Inliers: {inliers} < {self.anchor_inliers_thresh})")
                return {
                    "phase": "UNANCHORED_ACQUISITION",
                    "action": "SEARCHING_GLOBAL_MAP",
                    "use_visual_fix": False,
                    "reset_imu": False,
                    "confidence": 0.1
                }

        # PHASE 2 & 3: HIGH_CONFIDENCE_TRACKING or IMU_DEAD_RECKONING
        is_strong_match = pose_accepted and (inliers >= self.tracking_min_inliers)

        if is_strong_match:
            is_refix = (self.phase == "IMU_DEAD_RECKONING") and (inliers >= self.refix_inliers_thresh)
            self.phase = "HIGH_CONFIDENCE_TRACKING"
            self.last_valid_fix_time = now
            self.dead_reckoning_start = None

            if is_refix:
                logger.info(f"PHASE 4 (GLOBAL_REFIX): High-confidence visual match ({inliers} inliers) re-anchored trajectory.")
                action = "GLOBAL_REFIX_REANCHOR"
            else:
                action = "VISUAL_TRACKING"

            return {
                "phase": "HIGH_CONFIDENCE_TRACKING",
                "action": action,
                "use_visual_fix": True,
                "reset_imu": True,
                "confidence": min(1.0, inliers / 200.0)
            }

        # Visual Match Degraded -> Fallback to IMU Dead-Reckoning
        if self.dead_reckoning_start is None:
            self.dead_reckoning_start = now

        dead_rec_duration = now - self.dead_reckoning_start

        if dead_rec_duration <= self.max_dead_reckoning_sec:
            self.phase = "IMU_DEAD_RECKONING"
            logger.warning(f"PHASE 3 (IMU_DEAD_RECKONING): Low inliers ({inliers}). Relying on IMU propagation ({dead_rec_duration:.1f}s).")
            return {
                "phase": "IMU_DEAD_RECKONING",
                "action": "IMU_PROPAGATION",
                "use_visual_fix": False,
                "reset_imu": False,
                "confidence": max(0.1, 1.0 - (dead_rec_duration / self.max_dead_reckoning_sec))
            }

        # PHASE 5: EMERGENCY_HOLD (MAVLink Fail-Safe)
        self.phase = "EMERGENCY_HOLD"
        logger.error(f"PHASE 5 (EMERGENCY_HOLD): Visual tracking lost for {dead_rec_duration:.1f}s. Triggering MAVLink Loiter/RTL.")
        return {
            "phase": "EMERGENCY_HOLD",
            "action": "MAVLINK_EMERGENCY_LOITER",
            "use_visual_fix": False,
            "reset_imu": False,
            "confidence": 0.0
        }
