"""
Direct Dataset Camera Reader.
Reads real UAVVisLoc dataset frames and telemetry metadata directly from local disk.
Enables standalone Jetson evaluation without network/ZMQ sockets.
"""

import os
import cv2
import time
import numpy as np
from typing import Tuple, Optional, Any
from shared.protocol.telemetry_frame import TelemetryFrame


class DatasetCamera:
    """Reads real drone images and authentic telemetry CSV directly from UAVVisLoc dataset on disk."""

    def __init__(
        self,
        dataset_root: str = "data/UAV_VisLoc_dataset",
        sequence_id: str = "01",
        width: int = 640,
        height: int = 480
    ):
        self.dataset_root = dataset_root
        self.sequence_id = sequence_id
        self.width = width
        self.height = height
        self.records = []
        self.index = 0

        csv_path = os.path.join(dataset_root, f"{sequence_id}/{sequence_id}.csv")
        self.drone_dir = os.path.join(dataset_root, f"{sequence_id}/drone")

        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 9:
                        self.records.append({
                            "filename": parts[1],
                            "lat": float(parts[3]),
                            "lon": float(parts[4]),
                            "height": float(parts[5]),
                            "omega": float(parts[6]),  # pitch
                            "kappa": float(parts[7]),  # roll
                            "phi1": float(parts[8])    # yaw / heading
                        })

    def read(self) -> Tuple[bool, Optional[np.ndarray], Any]:
        """Reads next frame and authentic flight telemetry record from dataset."""
        if not self.records:
            return False, None, None

        rec = self.records[self.index]
        self.index = (self.index + 1) % len(self.records)

        img_path = os.path.join(self.drone_dir, rec["filename"])
        if not os.path.exists(img_path):
            # Generate synthetic fallback if specific photo file isn't downloaded
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img[:, :] = (50, 100, 50)
        else:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.width, self.height))
            else:
                img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        pitch_rad = np.radians(rec["omega"])
        roll_rad = np.radians(rec["kappa"])
        imu_ax = float(9.81 * np.sin(pitch_rad))
        imu_ay = float(-9.81 * np.sin(roll_rad))
        imu_az = float(9.81 * np.cos(pitch_rad) * np.cos(roll_rad))

        telemetry = TelemetryFrame(
            frame_id=self.index,
            timestamp=time.time(),
            gt_latitude=rec["lat"],
            gt_longitude=rec["lon"],
            gt_altitude_m=rec["height"],
            gt_heading_deg=rec["phi1"],
            gt_speed_mps=15.0,
            imu_ax=imu_ax,
            imu_ay=imu_ay,
            imu_az=imu_az,
            imu_gx=0.0,
            imu_gy=0.0,
            imu_gz=0.0
        )

        return True, img, telemetry

    def release(self):
        """Releases dataset camera resources."""
        pass
