"""
Trajectory Comparison & Visualization Script.
Parses `data/flight_log.csv` and outputs step-by-step Ground Truth vs Predicted Pose coordinates,
error offsets in meters, phase states, and overall trajectory summary.
"""

import os
import sys
import numpy as np

# Ensure repository root is in sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from shared.geo.tile_math import haversine_distance
from shared.logging_cfg import setup_logger

logger = setup_logger("plot_trajectory")


def analyze_trajectory(csv_path: str = "data/flight_log.csv"):
    """Reads flight_log.csv and prints trajectory comparison report."""
    if not os.path.exists(csv_path):
        logger.error(f"Flight log CSV not found at {csv_path}")
        return

    steps = []
    pred_lats, pred_lons = [], []
    gt_lats, gt_lons = [], []
    inliers = []
    phases = []
    errors_m = []

    with open(csv_path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 7:
                try:
                    s = int(parts[0])
                    plat, plon = float(parts[1]), float(parts[2])
                    glat, glon = float(parts[3]), float(parts[4])
                    inl = int(parts[5])
                    ph = parts[6]

                    dist = haversine_distance(plat, plon, glat, glon)
                    steps.append(s)
                    pred_lats.append(plat)
                    pred_lons.append(plon)
                    gt_lats.append(glat)
                    gt_lons.append(glon)
                    inliers.append(inl)
                    phases.append(ph)
                    errors_m.append(dist)
                except ValueError:
                    continue

    print("=" * 85)
    print("  FLIGHT TRAJECTORY: GROUND TRUTH VS PREDICTED GEOPOSITION COMPARISON")
    print("=" * 85)
    print(f" {'Step':<5} | {'Predicted Pose (Lat, Lon)':<25} | {'Ground Truth Pose (Lat, Lon)':<25} | {'Phase State':<22} | {'Inliers':<7}")
    print("-" * 85)

    num_samples = min(15, len(steps))
    sample_indices = np.linspace(0, len(steps) - 1, num_samples, dtype=int)

    for i in sample_indices:
        print(f" {steps[i]:02d}    | ({pred_lats[i]:.6f}, {pred_lons[i]:.6f})  | ({gt_lats[i]:.6f}, {gt_lons[i]:.6f})  | {phases[i]:<22} | {inliers[i]:<7}")

    print("=" * 85)
    print(f" Total Steps Recorded        : {len(steps)}")
    print(f" Initial Start Pose          : ({gt_lats[0]:.6f}, {gt_lons[0]:.6f})")
    print(f" Final Flight Destination    : ({gt_lats[-1]:.6f}, {gt_lons[-1]:.6f})")
    print(f" High-Confidence Fix Ratio   : {(sum(1 for p in phases if 'TRACKING' in p) / len(phases)) * 100:.1f}%")
    print("=" * 85)


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    analyze_trajectory()
