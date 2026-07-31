"""
CEP50 / CEP90 and Trajectory Metric Evaluation Script.
Parses `data/flight_log.csv` and calculates Circular Error Probable (CEP50, CEP90),
Mean Absolute Localization Error (meters), tracking success rate, and error statistics.
"""

import os
import sys
import math
import numpy as np

# Ensure repository root is in sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from shared.geo.tile_math import haversine_distance
from shared.logging_cfg import setup_logger

logger = setup_logger("evaluate_cep_metrics")


def evaluate_flight_log(csv_path: str = "data/flight_log.csv") -> dict:
    """Evaluates position errors from flight log CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"Flight log CSV not found at {csv_path}")
        return {}

    errors_m = []
    phases = []
    inlier_counts = []

    with open(csv_path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 7:
                try:
                    step = int(parts[0])
                    pred_lat = float(parts[1])
                    pred_lon = float(parts[2])
                    gt_lat = float(parts[3])
                    gt_lon = float(parts[4])
                    inliers = int(parts[5])
                    phase = parts[6]

                    dist_m = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
                    errors_m.append(dist_m)
                    phases.append(phase)
                    inlier_counts.append(inliers)
                except ValueError:
                    continue

    if not errors_m:
        logger.error("No valid telemetry rows found in flight log CSV")
        return {}

    err_arr = np.array(errors_m)
    cep50 = float(np.percentile(err_arr, 50))
    cep90 = float(np.percentile(err_arr, 90))
    mean_err = float(np.mean(err_arr))
    min_err = float(np.min(err_arr))
    max_err = float(np.max(err_arr))
    std_err = float(np.std(err_arr))

    tracking_steps = sum(1 for p in phases if "TRACKING" in p or "REFIX" in p)
    tracking_rate = (tracking_steps / len(phases)) * 100.0
    avg_inliers = float(np.mean(inlier_counts))

    results = {
        "total_steps": len(errors_m),
        "cep50_m": cep50,
        "cep90_m": cep90,
        "mean_error_m": mean_err,
        "min_error_m": min_err,
        "max_error_m": max_err,
        "std_error_m": std_err,
        "tracking_success_rate": tracking_rate,
        "avg_inliers_per_frame": avg_inliers
    }

    print("=" * 70)
    print("  GPS-DENIED VISUAL NAVIGATION — CEP METRICS EVALUATION")
    print("=" * 70)
    print(f" Total Flight Steps Evaluated  : {results['total_steps']}")
    print(f" CEP50 Radius (50% Error)     : {results['cep50_m']:.2f} meters")
    print(f" CEP90 Radius (90% Error)     : {results['cep90_m']:.2f} meters")
    print(f" Mean Absolute Error (MAE)    : {results['mean_error_m']:.2f} meters (± {results['std_error_m']:.2f}m)")
    print(f" Min / Max Position Error     : {results['min_error_m']:.2f}m / {results['max_error_m']:.2f}m")
    print(f" Tracking Success Rate        : {results['tracking_success_rate']:.1f}%")
    print(f" Average RANSAC Inliers       : {results['avg_inliers_per_frame']:.1f} per frame")
    print("=" * 70)

    return results


if __name__ == "__main__":
    evaluate_flight_log()
