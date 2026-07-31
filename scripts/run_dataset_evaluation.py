"""
Single-Step Dataset Evaluation Script.
Executes UAVVisLoc flight tracking on Jetson, outputs trajectory comparison, and prints CEP accuracy metrics.
"""

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.main import GPSDeniedPipeline
from scripts.evaluate_cep_metrics import evaluate_flight_log
from scripts.plot_flight_trajectory import analyze_trajectory
from shared.logging_cfg import setup_logger

logger = setup_logger("run_dataset_evaluation")


def run_full_dataset_eval(steps: int = 20):
    print("=" * 80)
    print("  EXECUTING GPS-DENIED EDGE NAVIGATION ON REAL UAV-VISLOC DATASET")
    print("=" * 80)

    pipeline = GPSDeniedPipeline("config/jetson.yaml", mode="dataset")
    try:
        for s in range(steps):
            res = pipeline.run_step()
            if res.get("status") == "success":
                gt_str = f" | GT: ({res['gt_pose']['latitude']:.6f}, {res['gt_pose']['longitude']:.6f})" if res.get("gt_pose") else ""
                print(f" Step {s+1:02d}/{steps:02d} [{res['phase']}] Pred: ({res['pose']['latitude']:.6f}, {res['pose']['longitude']:.6f}) Inliers: {res['inliers']}{gt_str}")
    finally:
        pipeline.close()

    print("\n")
    analyze_trajectory("data/flight_log.csv")
    print("\n")
    evaluate_flight_log("data/flight_log.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run complete dataset evaluation in one command.")
    parser.add_argument("--steps", type=int, default=20, help="Number of flight steps to evaluate")
    args = parser.parse_args()

    run_full_dataset_eval(steps=args.steps)
