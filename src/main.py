"""
Main control loop for GPS-Denied Visual Navigation System (Runs on System 1 - Jetson Orin Nano).
Executes Intelligent Multi-Phase Navigation State Machine:
  - Phase 1: UNANCHORED_ACQUISITION (Cold-Start Map Anchoring from Takeoff Pose)
  - Phase 2: HIGH_CONFIDENCE_TRACKING (Anchored Candidate Retrieval & Direct Visual Geopose Override)
  - Phase 3: IMU_DEAD_RECKONING (Sensor Propagation during Visual Dropout)
  - Phase 4: GLOBAL_REFIX (Trajectory Re-Anchoring upon High-Confidence Match)
  - Phase 5: EMERGENCY_HOLD (PyMAVLink Fail-Safe Stream)
Includes candidate patch rotation un-winding into North/East geographic frame.
"""

import cv2
import time
import math
import argparse
import yaml
import numpy as np
from typing import Dict, Any

from shared.logging_cfg import setup_logger
from shared.geo.pose_math import homography_to_translation, compute_geopose, ransac_pose_vote
from src.camera.sim_camera import SimCamera
from src.camera.zmq_receiver import ZMQReceiver
from src.inference.trt_engine import SuperPointEngine
from src.inference.lightglue_matcher import LightGlueMatcher
from src.retrieval.local_faiss import LocalFaissRetriever
from src.fusion.ekf_node import SimpleEKFFusion, PoseSmoother
from src.fusion.health_monitor import NavigationPhaseController
from src.comms.mavlink_bridge import MAVLinkBridge

logger = setup_logger("main")


class GPSDeniedPipeline:
    """Intelligent Multi-Phase GPS-Denied Edge Navigation Pipeline."""

    def __init__(self, config_path: str, mode: str = "zmq", pc_host: str = "10.1.1.13"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        edge_cfg = self.cfg.get("edge", {})
        map_cfg = self.cfg.get("map", {})
        self.start_pose = edge_cfg.get("start_pose", {"latitude": 29.760960, "longitude": 115.974797})

        texture_path = map_cfg.get("satellite_texture_path", "data/satellite01.jpg")
        bbox = tuple(map_cfg.get("bbox", [29.702283, 115.970635, 29.774065, 115.996851]))

        self.mode = mode
        self.camera = ZMQReceiver(host=pc_host, port=5555) if mode == "zmq" else SimCamera(texture_path=texture_path, bbox=bbox)
        self.sat_sampler = SimCamera(texture_path=texture_path, bbox=bbox)
        self.sp_engine = SuperPointEngine(descriptor_dim=256)
        self.matcher = LightGlueMatcher()
        self.retriever = LocalFaissRetriever(
            index_path=self.cfg.get("retrieval", {}).get("index_path", "data/ajabgarh_ivfpq.index"),
            db_path=self.cfg.get("retrieval", {}).get("db_path", "data/georef.sqlite")
        )
        self.ekf = SimpleEKFFusion(self.start_pose["latitude"], self.start_pose["longitude"])
        self.smoother = PoseSmoother(max_distance_m=3000.0)
        self.phase_controller = NavigationPhaseController(anchor_inliers_thresh=100, tracking_min_inliers=30)
        self.mavlink = MAVLinkBridge(
            connection_str=self.cfg.get("comms", {}).get("mavlink_connection", "udp:127.0.0.1:14540"),
            origin_lat=self.start_pose["latitude"],
            origin_lon=self.start_pose["longitude"]
        )
        
        self.anchor_lat = self.start_pose["latitude"]
        self.anchor_lon = self.start_pose["longitude"]
        self.last_timestamp = None

    def run_step(self) -> Dict[str, Any]:
        """Executes a single step of the intelligent multi-phase edge navigation loop."""
        if self.mode == "zmq":
            success, frame, telemetry = self.camera.read()
        else:
            success, frame = self.camera.read()
            telemetry = None

        if not success or frame is None:
            return {"status": "error", "message": "Frame acquisition failed"}

        alt_m = telemetry.gt_altitude_m if telemetry else 405.0

        # Calculate exact inter-frame timestamp delta dt
        now_ts = telemetry.timestamp if telemetry else time.time()
        if self.last_timestamp is not None:
            dt = max(0.01, min(1.0, now_ts - self.last_timestamp))
        else:
            dt = 1.0 / 30.0
        self.last_timestamp = now_ts

        # 1. IMU Dead-Reckoning State Prediction (Only used during IMU propagation phase)
        if telemetry is not None and not self.phase_controller.anchored:
            self.ekf.predict_imu(
                telemetry.imu_ax,
                telemetry.imu_ay,
                telemetry.imu_az,
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=telemetry.gt_heading_deg,
                dt=dt
            )

        # 2. Extract visual features from live drone camera frame
        gray_live = np.mean(frame, axis=2).astype(np.uint8) if frame.ndim == 3 else frame
        live_feats = self.sp_engine.extract(gray_live)

        if len(live_feats["keypoints"]) == 0:
            return {"status": "warning", "message": "No keypoints detected"}

        # 3. Intelligent Map Retrieval (Spatial Prior Search from anchored takeoff origin)
        spatial_prior = {"latitude": self.anchor_lat, "longitude": self.anchor_lon}
        query_vec = np.mean(live_feats["descriptors"], axis=0)
        candidates = self.retriever.search(query_vec, spatial_prior=spatial_prior, radius_km=8.0)

        # 4. Multi-candidate RANSAC voting consensus evaluation
        evaluated_candidates = []
        for cand in candidates[:3]:
            cand_lat = cand.get("center_lat", self.anchor_lat)
            cand_lon = cand.get("center_lon", self.anchor_lon)
            gsd_m_per_px = cand.get("gsd_m_per_px", 0.2781)
            patch_rot = cand.get("rotation_deg", 0)

            sat_patch_bgr = self.sat_sampler.get_frame_at_pose(cand_lat, cand_lon, alt_m=alt_m, heading_deg=patch_rot)
            gray_sat = np.mean(sat_patch_bgr, axis=2).astype(np.uint8) if sat_patch_bgr.ndim == 3 else sat_patch_bgr
            sat_feats = self.sp_engine.extract(gray_sat)

            inliers, M = self.matcher.match(live_feats, sat_feats)
            dx_px, dy_px, yaw_deg = homography_to_translation(M, (frame.shape[1] / 2.0, frame.shape[0] / 2.0))

            evaluated_candidates.append({
                "cand": cand,
                "inliers": inliers,
                "cand_lat": cand_lat,
                "cand_lon": cand_lon,
                "gsd_m_per_px": gsd_m_per_px,
                "patch_rot": patch_rot,
                "dx_px": dx_px,
                "dy_px": dy_px,
                "yaw_deg": yaw_deg
            })

        vote_res = ransac_pose_vote(evaluated_candidates, inlier_threshold=15)
        if vote_res.get("valid"):
            best = vote_res["best_candidate"]
        else:
            best = evaluated_candidates[0] if evaluated_candidates else {
                "inliers": 0, "cand_lat": self.anchor_lat, "cand_lon": self.anchor_lon,
                "gsd_m_per_px": 0.2781, "patch_rot": 0, "dx_px": 0.0, "dy_px": 0.0, "yaw_deg": 0.0
            }

        inliers = best["inliers"]
        cand_lat = best["cand_lat"]
        cand_lon = best["cand_lon"]
        gsd_m_per_px = best["gsd_m_per_px"]
        patch_rot = best["patch_rot"]
        dx_px = best["dx_px"]
        dy_px = best["dy_px"]
        yaw_deg = best["yaw_deg"]
        fov_base_px = 512.0
        crop_size = int(fov_base_px * (alt_m / 100.0))
        scale_factor = crop_size / 640.0

        dx_map = dx_px * scale_factor
        dy_map = dy_px * scale_factor

        # Un-rotate pixel displacement vector into North/East geographic frame
        rot_rad = math.radians(-patch_rot)
        dx_rot = dx_map * math.cos(rot_rad) - dy_map * math.sin(rot_rad)
        dy_rot = dx_map * math.sin(rot_rad) + dy_map * math.cos(rot_rad)

        raw_pose = compute_geopose(
            ref_lat=cand_lat,
            ref_lon=cand_lon,
            dx_px=dx_rot,
            dy_px=dy_rot,
            yaw_deg=yaw_deg + patch_rot,
            gsd_m_per_px=gsd_m_per_px
        )

        # 7. Evaluate Multi-Phase State Transitions & Apply Absolute Visual Fix
        smooth_res = self.smoother.filter(raw_pose)
        phase_res = self.phase_controller.evaluate_step(inliers, smooth_res["accepted"], raw_pose)

        if phase_res["use_visual_fix"]:
            # Direct 100% visual override during High-Confidence Visual Tracking
            final_pose = self.ekf.update_visual_fix({"latitude": raw_pose["latitude"], "longitude": raw_pose["longitude"], "heading_deg": yaw_deg + patch_rot})
            self.anchor_lat = final_pose["latitude"]
            self.anchor_lon = final_pose["longitude"]
            self.mavlink.send_vision_position_estimate(final_pose)
        else:
            final_pose = {"latitude": self.ekf.lat, "longitude": self.ekf.lon}

        gt_lat = telemetry.gt_latitude if telemetry else None
        gt_lon = telemetry.gt_longitude if telemetry else None

        return {
            "status": "success",
            "state": phase_res["phase"],
            "phase": phase_res["phase"],
            "action": phase_res["action"],
            "inliers": inliers,
            "pose": final_pose,
            "gt_pose": {"latitude": gt_lat, "longitude": gt_lon} if gt_lat else None
        }

    def close(self):
        """Releases pipeline resources cleanly."""
        if hasattr(self.camera, "release"):
            try:
                self.camera.release()
            except Exception:
                pass
        if hasattr(self.sat_sampler, "release"):
            try:
                self.sat_sampler.release()
            except Exception:
                pass
        if hasattr(self.retriever, "close"):
            try:
                self.retriever.close()
            except Exception:
                pass
        if hasattr(self.mavlink, "close"):
            try:
                self.mavlink.close()
            except Exception:
                pass
        logger.info("Pipeline resources released cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Run Intelligent GPS-Denied navigation control loop.")
    parser.add_argument("--config", type=str, default="config/jetson.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="sim", choices=["sim", "zmq"], help="Frame acquisition mode")
    parser.add_argument("--pc-host", type=str, default="10.1.1.13", help="PC host IP for ZMQ stream")
    parser.add_argument("--steps", type=int, default=10, help="Number of control loop steps to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose diagnostic logging")
    args = parser.parse_args()

    pipeline = GPSDeniedPipeline(args.config, mode=args.mode, pc_host=args.pc_host)
    logger.info(f"Initialized Intelligent GPS-Denied Navigation State Machine [{args.mode} mode]")

    try:
        for step in range(args.steps):
            res = pipeline.run_step()
            if res.get("status") == "success":
                gt_str = f" | GT: ({res['gt_pose']['latitude']:.6f}, {res['gt_pose']['longitude']:.6f})" if res.get("gt_pose") else ""
                logger.info(f"Step {step+1:02d}/{args.steps:02d} [{res['phase']} | {res['action']}] Pred: ({res['pose']['latitude']:.6f}, {res['pose']['longitude']:.6f}) (Inliers: {res['inliers']}){gt_str}")
            else:
                logger.warning(f"Step {step+1:02d}/{args.steps:02d} failed: {res.get('message')}")
    except KeyboardInterrupt:
        logger.info("Pipeline loop interrupted by user.")
    finally:
        pipeline.close()

    logger.info("Pipeline test loop complete.")


if __name__ == "__main__":
    main()
