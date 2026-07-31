"""
UAV-VisLoc Professional Telemetry & Real Video Streamer.
Reads real high-resolution drone photographs and actual flight telemetry (center Lat/Lon,
flying height, roll/pitch/yaw angles) from the UAV-VisLoc dataset and streams packets
over ZMQ PUB socket to the Jetson Orin Nano edge node at 30 Hz.
Includes startup delay parameter to allow seamless synchronization with Jetson control loop launch.
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

logger = setup_logger("uav_visloc_streamer")


def run_uav_visloc_streamer(
    sequence_id: str = "01",
    dataset_root: str = os.environ.get("UAV_VISLOC_ROOT", "data/uav_visloc"),
    port: int = 5555,
    fps: float = 30.0,
    startup_delay: float = 3.0,
    loop: bool = True
):
    """Publishes real drone video frames and authentic flight telemetry over ZMQ PUB socket."""
    csv_path = os.path.join(dataset_root, f"{sequence_id}/{sequence_id}.csv")
    drone_dir = os.path.join(dataset_root, f"{sequence_id}/drone")

    if not os.path.exists(csv_path):
        logger.error(f"Metadata CSV not found: {csv_path}")
        return

    # Read flight telemetry CSV
    telemetry_records = []
    with open(csv_path, "r") as f:
        header = f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 9:
                telemetry_records.append({
                    "num": int(parts[0]),
                    "filename": parts[1],
                    "date": parts[2],
                    "lat": float(parts[3]),
                    "lon": float(parts[4]),
                    "height": float(parts[5]),
                    "omega": float(parts[6]),  # pitch
                    "kappa": float(parts[7]),  # roll
                    "phi1": float(parts[8])    # yaw / heading
                })

    logger.info(f"Loaded {len(telemetry_records)} real flight telemetry records from {csv_path}")

    # Bind ZMQ Socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://0.0.0.0:{port}")
    logger.info(f"Started Professional UAV-VisLoc Telemetry Streamer on tcp://0.0.0.0:{port}")

    if startup_delay > 0:
        logger.info(f"Pausing {startup_delay}s for Jetson Edge Node connection launch...")
        time.sleep(startup_delay)

    frame_id = 0
    record_idx = 0

    try:
        while True:
            rec = telemetry_records[record_idx]
            img_path = os.path.join(drone_dir, rec["filename"])

            if not os.path.exists(img_path):
                logger.warning(f"Drone image missing: {img_path}")
                record_idx = (record_idx + 1) % len(telemetry_records)
                continue

            frame_id += 1
            now = time.time()

            # Load real drone photograph and resize to 640x480 for edge pipeline input
            frame_bgr = cv2.imread(img_path)
            if frame_bgr is None:
                record_idx = (record_idx + 1) % len(telemetry_records)
                continue

            frame_resized = cv2.resize(frame_bgr, (640, 480))
            success, frame_jpg = cv2.imencode(".jpg", frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            if not success:
                record_idx = (record_idx + 1) % len(telemetry_records)
                continue

            # Real telemetry attributes
            cur_lat = rec["lat"]
            cur_lon = rec["lon"]
            alt_m = rec["height"]
            heading_deg = rec["phi1"]

            # Convert roll/pitch angles to synthetic IMU accelerations
            pitch_rad = np.radians(rec["omega"])
            roll_rad = np.radians(rec["kappa"])
            imu_ax = float(9.81 * np.sin(pitch_rad) + np.random.normal(0, 0.02))
            imu_ay = float(-9.81 * np.sin(roll_rad) + np.random.normal(0, 0.02))
            imu_az = float(9.81 * np.cos(pitch_rad) * np.cos(roll_rad) + np.random.normal(0, 0.02))

            telemetry_packet = TelemetryFrame(
                frame_id=frame_id,
                timestamp=now,
                gt_latitude=cur_lat,
                gt_longitude=cur_lon,
                gt_altitude_m=alt_m,
                gt_heading_deg=heading_deg,
                gt_speed_mps=15.0,  # real drone flight speed ~15 m/s
                imu_ax=imu_ax,
                imu_ay=imu_ay,
                imu_az=imu_az,
                imu_gx=0.0,
                imu_gy=0.0,
                imu_gz=0.0
            )

            packet = telemetry_packet.pack_payload(frame_jpg.tobytes())
            socket.send(packet)

            if frame_id % 10 == 0 or frame_id == 1:
                logger.info(f"Streamed Frame #{frame_id} ({rec['filename']}) -> GT Pose: Lat {cur_lat:.6f}, Lon {cur_lon:.6f}, Alt: {alt_m:.1f}m, Heading: {heading_deg:.1f}°")

            record_idx += 1
            if record_idx >= len(telemetry_records):
                if loop:
                    logger.info("Reached end of UAV-VisLoc sequence. Looping flight trajectory...")
                    record_idx = 0
                else:
                    break

            time.sleep(1.0 / fps)

    except KeyboardInterrupt:
        logger.info("Stopping UAV-VisLoc streamer...")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream real UAV-VisLoc drone video and authentic telemetry over ZMQ.")
    parser.add_argument("--sequence", type=str, default="01", help="Dataset sequence ID (01 - 11)")
    parser.add_argument("--dataset-root", type=str, default=os.environ.get("UAV_VISLOC_ROOT", "data/uav_visloc"), help="Path to UAV-VisLoc dataset directory")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ publishing port")
    parser.add_argument("--fps", type=float, default=30.0, help="Stream FPS rate")
    parser.add_argument("--startup-delay", type=float, default=3.0, help="Startup delay seconds before publishing")
    args = parser.parse_args()

    run_uav_visloc_streamer(sequence_id=args.sequence, dataset_root=args.dataset_root, port=args.port, fps=args.fps, startup_delay=args.startup_delay)
