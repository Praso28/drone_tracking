"""
Main control loop for GPS-Denied Visual Navigation System (Runs on System 1 - Jetson Orin Nano).
Executes real-time visual localization, spatial-prior FAISS map search, LightGlue 2D Affine matching,
IMU EKF fusion, Sensor Health Monitoring, and PyMAVLink output stream.
Includes wide-radius 8.0km spatial search window for robust takeoff position acquisition.
"""

import cv2
import time
import argparse
import yaml
import numpy as np
from typing import Dict, Any

from shared.logging_cfg import setup_logger
from shared.geo.pose_math import homography_to_translation, compute_geopose
from src.camera.sim_camera import SimCamera
from src.camera.zmq_receiver import ZMQReceiver
from src.inference.trt_engine import SuperPointEngine
from src.inference.lightglue_matcher import LightGlueMatcher
from src.retrieval.local_faiss import LocalFaissRetriever
from src.fusion.ekf_node import SimpleEKFFusion, PoseSmoother
from src.fusion.health_monitor import SensorHealthMonitor
from src.comms.mavlink_bridge import MAVLinkBridge

logger = setup_logger("main")


class GPSDeniedPipeline:
    """State machine pipeline executing edge navigation loop on Jetson."""

    def __init__(self, config_path: str, mode: str = "zmq", pc_host: str = "10.1.1.13"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        edge_cfg = self.cfg.get("edge", {})
        map_cfg = self.cfg.get("map", {})
        start_pose = edge_cfg.get("start_pose", {"latitude": 29.739041, "longitude": 115.985113})

        texture_path = map_cfg.get("satellite_texture_path", "data/satellite01.jpg")
        bbox = tuple(map_cfg.get("bbox", [29.702283, 115.970635, 29.774065, 115.996851]))

        self.mode = mode
        self.state = "INIT"

        if mode == "zmq":
            self.camera = ZMQReceiver(host=pc_host, port=5555)
        else:
            self.camera = SimCamera(texture_path=texture_path, bbox=bbox)

        self.sat_sampler = SimCamera(texture_path=texture_path, bbox=bbox)
        self.sp_engine = SuperPointEngine(descriptor_dim=256)
        self.matcher = LightGlueMatcher()
        self.retriever = LocalFaissRetriever(
            index_path=self.cfg.get("retrieval", {}).get("index_path", "data/ajabgarh_ivfpq.index"),
            db_path=self.cfg.get("retrieval", {}).get("db_path", "data/georef.sqlite")
        )
        self.ekf = SimpleEKFFusion(start_pose["latitude"], start_pose["longitude"])
        self.smoother = PoseSmoother(max_distance_m=5000.0)
        self.health_monitor = SensorHealthMonitor()
        self.mavlink = MAVLinkBridge(connection_str=self.cfg.get("comms", {}).get("mavlink_connection", "udp:127.0.0.1:14540"))

    def run_step(self) -> Dict[str, Any]:
        """Executes a single step of the edge navigation state machine."""
        if self.mode == "zmq":
            success, frame, telemetry = self.camera.read()
        else:
            success, frame = self.camera.read()
            telemetry = None

        if not success or frame is None:
            return {"status": "error", "message": "Frame acquisition failed"}

        alt_m = telemetry.gt_altitude_m if telemetry else 405.0

        # 1. IMU state prediction update with gravity compensation
        if telemetry is not None:
            self.ekf.predict_imu(
                telemetry.imu_ax,
                telemetry.imu_ay,
                telemetry.imu_az,
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=telemetry.gt_heading_deg
            )

        # 2. Extract visual features from live drone camera frame (256-dim)
        gray_live = np.mean(frame, axis=2).astype(np.uint8) if frame.ndim == 3 else frame
        live_feats = self.sp_engine.extract(gray_live)

        if len(live_feats["keypoints"]) == 0:
            return {"status": "warning", "message": "No keypoints detected"}

        # 3. Retrieve candidate satellite map patch using wide 8.0km spatial search radius
        spatial_prior = {"latitude": self.ekf.lat, "longitude": self.ekf.lon}
        query_vector = np.mean(live_feats["descriptors"], axis=0)
        candidates = self.retriever.search(query_vector, spatial_prior=spatial_prior, radius_km=8.0)
        top_cand = candidates[0] if candidates else {}

        cand_lat = top_cand.get("center_lat", self.ekf.lat)
        cand_lon = top_cand.get("center_lon", self.ekf.lon)
        gsd_m_per_px = top_cand.get("gsd_m_per_px", 0.2781)

        # 4. Lazy-crop candidate satellite map patch from disk at retrieved candidate coordinate
        sat_patch_bgr = self.sat_sampler.get_frame_at_pose(cand_lat, cand_lon, alt_m=alt_m, heading_deg=top_cand.get("rotation_deg", 0))
        gray_sat = np.mean(sat_patch_bgr, axis=2).astype(np.uint8) if sat_patch_bgr.ndim == 3 else sat_patch_bgr
        sat_feats = self.sp_engine.extract(gray_sat)

        # 5. Perform 2D Partial Affine cross-matching between live drone frame and retrieved satellite map patch
        inliers, M = self.matcher.match(live_feats, sat_feats)

        # 6. Convert Affine transformation to translation & WGS84 geopose with altitude scale factor
        dx_px, dy_px, yaw_deg = homography_to_translation(M, (frame.shape[1] / 2.0, frame.shape[0] / 2.0))

        fov_base_px = 512.0
        crop_size = int(fov_base_px * (alt_m / 100.0))
        scale_factor = crop_size / 640.0

        dx_map_px = dx_px * scale_factor
        dy_map_px = dy_px * scale_factor

        raw_pose = compute_geopose(
            ref_lat=cand_lat,
            ref_lon=cand_lon,
            dx_px=dx_map_px,
            dy_px=dy_map_px,
            yaw_deg=yaw_deg,
            gsd_m_per_px=gsd_m_per_px
        )

        # 7. Outlier filtering, EKF state update, and Sensor Health Evaluation
        smooth_res = self.smoother.filter(raw_pose)
        health_res = self.health_monitor.evaluate_step(inliers, smooth_res["accepted"])

        if smooth_res["accepted"]:
            final_pose = self.ekf.update_visual_fix(smooth_res["pose"])
            self.mavlink.send_vision_position_estimate(final_pose)
            self.state = health_res["health_state"]
        else:
            self.state = health_res["health_state"]
            final_pose = {"latitude": self.ekf.lat, "longitude": self.ekf.lon}

        gt_lat = telemetry.gt_latitude if telemetry else None
        gt_lon = telemetry.gt_longitude if telemetry else None

        return {
            "status": "success",
            "state": self.state,
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
    parser = argparse.ArgumentParser(description="Run GPS-Denied navigation control loop.")
    parser.add_argument("--config", type=str, default="config/jetson.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="sim", choices=["sim", "zmq"], help="Frame acquisition mode")
    parser.add_argument("--pc-host", type=str, default="10.1.1.13", help="PC host IP for ZMQ stream")
    parser.add_argument("--steps", type=int, default=10, help="Number of control loop steps to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose diagnostic logging")
    args = parser.parse_args()

    pipeline = GPSDeniedPipeline(args.config, mode=args.mode, pc_host=args.pc_host)
    logger.info(f"Initialized GPS-Denied Pipeline state machine [{args.mode} mode]: {pipeline.state}")

    try:
        for step in range(args.steps):
            res = pipeline.run_step()
            if res.get("status") == "success":
                gt_str = f" | GT: ({res['gt_pose']['latitude']:.6f}, {res['gt_pose']['longitude']:.6f})" if res.get("gt_pose") else ""
                logger.info(f"Step {step+1}/{args.steps} [{res['state']}] Pred Pose: Lat {res['pose']['latitude']:.6f}, Lon {res['pose']['longitude']:.6f} (Inliers: {res['inliers']}){gt_str}")
            else:
                logger.warning(f"Step {step+1}/{args.steps} failed: {res.get('message')}")
    except KeyboardInterrupt:
        logger.info("Pipeline loop interrupted by user.")
    finally:
        pipeline.close()

    logger.info("Pipeline test loop complete.")


if __name__ == "__main__":
    main()
