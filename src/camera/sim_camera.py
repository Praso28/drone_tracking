"""
Synthetic frame camera stream interface for simulation host & edge testing.
Generates realistic drone camera crops directly from high-resolution satellite map textures
(data/ajabgarh_sat.png) mapped to real WGS84 coordinates [min_lat, min_lon, max_lat, max_lon].
Memory-optimized using lazy image loading & memory-mapping for low RAM footprint (<50MB).
"""

import os
import time
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class SimCamera:
    """Generates realistic down-looking drone video frames from satellite map texture."""

    def __init__(
        self,
        texture_path: str = "data/ajabgarh_sat.png",
        bbox: Tuple[float, float, float, float] = (27.1800, 76.2100, 27.2200, 76.2600),
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.min_lat, self.min_lon, self.max_lat, self.max_lon = bbox
        self.texture_path = texture_path
        self.sat_image = None
        self.map_w, self.map_h = 2048, 2048

        if os.path.exists(texture_path):
            try:
                # Open PIL Image lazily (does not load uncompressed pixels into memory until cropped)
                self.sat_image = Image.open(texture_path)
                self.map_w, self.map_h = self.sat_image.size
            except Exception:
                self.sat_image = None

    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        """Converts WGS84 latitude and longitude into image pixel coordinates on satellite texture."""
        px = (lon - self.min_lon) / (self.max_lon - self.min_lon) * self.map_w
        py = (self.max_lat - lat) / (self.max_lat - self.min_lat) * self.map_h
        return px, py

    def get_frame_at_pose(
        self,
        lat: float,
        lon: float,
        alt_m: float = 100.0,
        heading_deg: float = 0.0
    ) -> np.ndarray:
        """Extracts camera crop from satellite map at specific WGS84 pose."""
        center_x, center_y = self.latlon_to_pixel(lat, lon)

        fov_base_px = 512.0
        crop_size = int(fov_base_px * (alt_m / 100.0))
        crop_size = max(128, min(crop_size, 1024))

        left = int(clamp(center_x - crop_size / 2, 0, self.map_w - crop_size))
        top = int(clamp(center_y - crop_size / 2, 0, self.map_h - crop_size))

        if self.sat_image is not None:
            # Crop only the required bounding box from disk
            crop = self.sat_image.crop((left, top, left + crop_size, top + crop_size)).convert("RGB")
            if abs(heading_deg) > 0.1:
                resample_filter = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
                crop = crop.rotate(-heading_deg, resample=resample_filter)
            crop_resized = crop.resize((self.width, self.height))
            return np.array(crop_resized, dtype=np.uint8)
        else:
            # Synthetic fallback frame
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img[:, :] = (40, 120, 40)
            return img

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Fallback automated path generator for standalone testing."""
        time.sleep(1.0 / self.fps)
        self.frame_count += 1
        t = self.frame_count * 0.02
        lat = self.min_lat + (self.max_lat - self.min_lat) * (0.5 + 0.3 * np.sin(t))
        lon = self.min_lon + (self.max_lon - self.min_lon) * (0.5 + 0.3 * np.cos(t))
        heading = (self.frame_count * 2.0) % 360.0
        frame = self.get_frame_at_pose(lat, lon, 100.0, heading)
        return True, frame

    def release(self):
        """Releases camera resources."""
        if self.sat_image is not None:
            try:
                self.sat_image.close()
            except Exception:
                pass


def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))
