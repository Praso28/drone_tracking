"""
Protocol buffer data structure for PC Ground Station to Jetson Edge Node telemetry streaming.
Transmits compressed camera frames alongside ground-truth flight state, IMU, and sensor data.
"""

import json
import struct
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any


@dataclass
class TelemetryFrame:
    frame_id: int
    timestamp: float
    gt_latitude: float
    gt_longitude: float
    gt_altitude_m: float
    gt_heading_deg: float
    gt_speed_mps: float
    imu_ax: float
    imu_ay: float
    imu_az: float
    imu_gx: float
    imu_gy: float
    imu_gz: float

    def to_json(self) -> str:
        """Serializes telemetry metadata to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "TelemetryFrame":
        """Deserializes telemetry metadata from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def pack_payload(self, frame_jpg_bytes: bytes) -> bytes:
        """Packs metadata JSON length, metadata payload, and JPEG frame bytes into a binary packet."""
        meta_bytes = self.to_json().encode("utf-8")
        header = struct.pack(">II", len(meta_bytes), len(frame_jpg_bytes))
        return header + meta_bytes + frame_jpg_bytes

    @classmethod
    def unpack_payload(cls, raw_bytes: bytes) -> Tuple["TelemetryFrame", bytes]:
        """Unpacks binary packet into (TelemetryFrame, frame_jpg_bytes)."""
        header_size = struct.calcsize(">II")
        meta_len, frame_len = struct.unpack(">II", raw_bytes[:header_size])
        meta_bytes = raw_bytes[header_size : header_size + meta_len]
        frame_bytes = raw_bytes[header_size + meta_len : header_size + meta_len + frame_len]
        frame = cls.from_json(meta_bytes.decode("utf-8"))
        return frame, frame_bytes
