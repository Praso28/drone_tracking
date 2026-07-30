"""
Accuracy evaluation tool for trajectory benchmarking.
Calculates Median Error, Mean Error, RMSE, Sub-2m %, and Sub-5m % accuracy metrics.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
from typing import List, Tuple, Dict
from shared.geo.tile_math import haversine_distance
from shared.logging_cfg import setup_logger

logger = setup_logger("eval_accuracy")


def evaluate_trajectory_errors(
    ground_truth: List[Tuple[float, float]],
    estimated: List[Tuple[float, float]]
) -> Dict[str, float]:
    """Computes positioning error metrics across ground truth and estimated trajectories."""
    if len(ground_truth) != len(estimated) or len(ground_truth) == 0:
        logger.error("Trajectory length mismatch or empty trajectory.")
        return {}

    errors = []
    for (gt_lat, gt_lon), (est_lat, est_lon) in zip(ground_truth, estimated):
        err_m = haversine_distance(gt_lat, gt_lon, est_lat, est_lon)
        errors.append(err_m)

    errors_arr = np.array(errors)
    median_err = float(np.median(errors_arr))
    mean_err = float(np.mean(errors_arr))
    rmse_err = float(np.sqrt(np.mean(errors_arr ** 2)))
    sub_2m_pct = float(np.mean(errors_arr <= 2.0) * 100.0)
    sub_5m_pct = float(np.mean(errors_arr <= 5.0) * 100.0)

    metrics = {
        "total_frames": len(errors),
        "median_error_m": median_err,
        "mean_error_m": mean_err,
        "rmse_m": rmse_err,
        "sub_2m_pct": sub_2m_pct,
        "sub_5m_pct": sub_5m_pct,
    }

    logger.info("=== Trajectory Evaluation Metrics ===")
    logger.info(f"Total Frames: {metrics['total_frames']}")
    logger.info(f"Median Error: {metrics['median_error_m']:.2f} m")
    logger.info(f"Mean Error:   {metrics['mean_error_m']:.2f} m")
    logger.info(f"RMSE:         {metrics['rmse_m']:.2f} m")
    logger.info(f"Sub-2m Acc:   {metrics['sub_2m_pct']:.1f}%")
    logger.info(f"Sub-5m Acc:   {metrics['sub_5m_pct']:.1f}%")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectory accuracy metrics.")
    args = parser.parse_args()

    # Synthetic demo trajectory evaluation
    gt = [(27.2000 + i*0.0001, 76.2350 + i*0.0001) for i in range(50)]
    est = [(lat + np.random.normal(0, 0.00001), lon + np.random.normal(0, 0.00001)) for lat, lon in gt]
    evaluate_trajectory_errors(gt, est)
