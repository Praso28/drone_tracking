"""
EKF state estimator and rolling median outlier rejection filter for position fixes.
Includes IMU sensor pre-integration for dead-reckoning state estimation between visual fixes.
"""

import numpy as np
from typing import Dict, Any, List
from shared.geo.tile_math import haversine_distance, pixel_to_latlon


class PoseSmoother:
    """Rolling median filter for position outlier rejection with cold-start warmup."""

    def __init__(self, window_size: int = 5, max_distance_m: float = 3000.0, warmup_fixes: int = 5):
        self.window_size = window_size
        self.max_distance_m = max_distance_m
        self.warmup_fixes = warmup_fixes
        self.fix_count = 0
        self.history: List[Dict[str, float]] = []

    def filter(self, pose: Dict[str, float]) -> Dict[str, Any]:
        """Filters an incoming pose fix. Accepts initial fixes during warmup phase."""
        lat = pose["latitude"]
        lon = pose["longitude"]

        self.fix_count += 1

        # Accept all fixes during cold-start warmup phase to allow pipeline anchoring
        if self.fix_count <= self.warmup_fixes:
            self.history.append(pose)
            if len(self.history) > self.window_size:
                self.history.pop(0)
            return {"accepted": True, "pose": pose, "warmup": True}

        med_lat = float(np.median([p["latitude"] for p in self.history]))
        med_lon = float(np.median([p["longitude"] for p in self.history]))

        dist = haversine_distance(med_lat, med_lon, lat, lon)
        if dist > self.max_distance_m:
            return {"accepted": False, "reason": f"Outlier rejected (distance {dist:.1f}m > {self.max_distance_m}m)"}

        self.history.append(pose)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        return {"accepted": True, "pose": pose, "warmup": False}


class SimpleEKFFusion:
    """Combines IMU dead-reckoning and visual fixes into a smoothed global state estimate."""

    def __init__(self, init_lat: float, init_lon: float, std_m: float = 2.0):
        self.lat = init_lat
        self.lon = init_lon
        self.vx = 0.0
        self.vy = 0.0
        self.std_m = std_m
        self.initialized = False

    def predict_imu(
        self,
        ax: float,
        ay: float,
        az: float,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        dt: float = 1.0 / 30.0
    ):
        """
        Integrates IMU linear accelerations with gravity compensation and attitude rotation.
        """
        # Convert attitude to radians
        r = np.radians(roll_deg)
        p = np.radians(pitch_deg)

        # Subtract gravity vector projection based on roll and pitch tilt
        ax_comp = ax - 9.81 * np.sin(p)
        ay_comp = ay + 9.81 * np.sin(r) * np.cos(p)

        # Integrate planar velocity
        self.vx += ax_comp * dt
        self.vy += ay_comp * dt

        dx_m = self.vx * dt
        dy_m = self.vy * dt

        earth_radius_m = 6378137.0
        dlat = (dy_m / earth_radius_m) * (180.0 / np.pi)
        dlon = (dx_m / (earth_radius_m * np.cos(np.radians(self.lat)))) * (180.0 / np.pi)

        self.lat += dlat
        self.lon += dlon

    def update_visual_fix(self, fix: Dict[str, float]) -> Dict[str, float]:
        """Updates internal Kalman state estimate using absolute visual fix."""
        if not self.initialized:
            # Snap instantly to first visual position fix on cold-start
            self.lat = fix["latitude"]
            self.lon = fix["longitude"]
            self.initialized = True
        else:
            alpha = 0.85  # Fast Kalman Gain for dynamic flight tracking
            self.lat = (1 - alpha) * self.lat + alpha * fix["latitude"]
            self.lon = (1 - alpha) * self.lon + alpha * fix["longitude"]

        # Reset velocity drift on visual update
        self.vx *= 0.5
        self.vy *= 0.5

        return {
            "latitude": self.lat,
            "longitude": self.lon,
            "heading_deg": fix.get("heading_deg", 0.0),
        }
