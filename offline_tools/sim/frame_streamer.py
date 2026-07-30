"""
ZMQ Frame & Telemetry Streamer Server (Runs on System 2 - PC Ground Station).
Simulates realistic down-looking camera video and IMU/baro telemetry along a waypointed flight path
and streams packets over ZMQ to the Jetson Orin Nano edge node.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import cv2
import zmq
import yaml
import argparse
import numpy as np

from shared.logging_cfg import setup_logger
from shared.protocol.telemetry_frame import TelemetryFrame
from src.camera.sim_camera import SimCamera

logger = setup_logger("frame_streamer")


def run_frame_streamer(config_path: str = "config/sim.yaml", port: int = 5555):
    """Publishes continuous video frames and flight telemetry over ZMQ PUB socket."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    map_cfg = cfg.get("map", {})
    bbox = map_cfg.get("bbox", [27.1800, 76.2100, 27.2200, 76.2600])

    camera = SimCamera(
        texture_path=map_cfg.get("satellite_texture_path", "data/ajabgarh_sat.png"),
        bbox=tuple(bbox),
        width=640,
        height=480,
        fps=30
    )

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://0.0.0.0:{port}")
    logger.info(f"Started ZMQ Frame & Telemetry Streamer on tcp://0.0.0.0:{port}")

    waypoints = [
        (27.2000, 76.2350),
        (27.2050, 76.2350),
        (27.2050, 76.2400),
        (27.2000, 76.2400)
    ]

    frame_id = 0
    start_time = time.time()
    num_wp = len(waypoints)

    try:
        while True:
            frame_id += 1
            now = time.time()
            elapsed = now - start_time

            # Interpolate position along lawnmower waypoints
            total_duration = 30.0  # seconds per loop
            progress = (elapsed % total_duration) / total_duration
            segment = progress * num_wp
            idx0 = int(segment) % num_wp
            idx1 = (idx0 + 1) % num_wp
            alpha = segment - int(segment)

            wp0 = waypoints[idx0]
            wp1 = waypoints[idx1]

            cur_lat = (1 - alpha) * wp0[0] + alpha * wp1[0]
            cur_lon = (1 - alpha) * wp0[1] + alpha * wp1[1]
            alt_m = 100.0 + 2.0 * np.sin(elapsed * 0.5)
            heading_deg = math.degrees(math.atan2(wp1[1] - wp0[1], wp1[0] - wp0[0])) % 360.0

            # Render realistic camera crop from satellite map
            frame_bgr = camera.get_frame_at_pose(cur_lat, cur_lon, alt_m, heading_deg)
            success, frame_jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            if not success:
                continue

            # Generate synthetic IMU readings (gravity + small vibration noise)
            imu_ax = float(np.random.normal(0.0, 0.05))
            imu_ay = float(np.random.normal(0.0, 0.05))
            imu_az = float(9.81 + np.random.normal(0.0, 0.05))
            imu_gx = float(np.random.normal(0.0, 0.01))
            imu_gy = float(np.random.normal(0.0, 0.01))
            imu_gz = float(np.random.normal(0.0, 0.01))

            telemetry = TelemetryFrame(
                frame_id=frame_id,
                timestamp=now,
                gt_latitude=cur_lat,
                gt_longitude=cur_lon,
                gt_altitude_m=alt_m,
                gt_heading_deg=heading_deg,
                gt_speed_mps=5.0,
                imu_ax=imu_ax,
                imu_ay=imu_ay,
                imu_az=imu_az,
                imu_gx=imu_gx,
                imu_gy=imu_gy,
                imu_gz=imu_gz
            )

            packet = telemetry.pack_payload(frame_jpg.tobytes())
            socket.send(packet)

            if frame_id % 30 == 0:
                logger.info(f"Streamed Frame #{frame_id} -> GT Pose: ({cur_lat:.6f}, {cur_lon:.6f}, Alt: {alt_m:.1f}m)")

            time.sleep(1.0 / 30.0)

    except KeyboardInterrupt:
        logger.info("Stopping frame streamer...")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    import math
    parser = argparse.ArgumentParser(description="Stream video frames and telemetry over ZMQ.")
    parser.add_argument("--config", type=str, default="config/sim.yaml", help="Path to config file")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port to publish on")
    args = parser.parse_args()

    run_frame_streamer(args.config, args.port)
