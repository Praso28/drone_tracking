"""
CEP Position Accuracy Evaluator for GPS-Denied Navigation Pipeline.
Calculates Circular Error Probable (CEP-50, CEP-95), Mean Error (meters), and RMSE
comparing predicted WGS84 coordinates against authentic ground-truth drone telemetry.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from typing import List, Dict, Any
from shared.geo.tile_math import haversine_distance
from shared.logging_cfg import setup_logger

logger = setup_logger("cep_evaluator")


class CEPEvaluator:
    """Tracks and calculates quantitative position accuracy metrics."""

    def __init__(self):
        self.errors_m: List[float] = []

    def add_sample(self, pred_lat: float, pred_lon: float, gt_lat: float, gt_lon: float) -> float:
        """Adds a predicted vs ground-truth pose sample and returns position error in meters."""
        err_m = float(haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon))
        self.errors_m.append(err_m)
        return err_m

    def compute_metrics(self) -> Dict[str, float]:
        """Calculates CEP-50, CEP-95, Mean Error, Max Error, and RMSE in meters."""
        if not self.errors_m:
            return {"cep_50_m": 0.0, "cep_95_m": 0.0, "mean_m": 0.0, "max_m": 0.0, "rmse_m": 0.0, "samples": 0}

        errs = np.array(self.errors_m)
        cep_50 = float(np.percentile(errs, 50))
        cep_95 = float(np.percentile(errs, 95))
        mean_err = float(np.mean(errs))
        max_err = float(np.max(errs))
        rmse_err = float(np.sqrt(np.mean(errs**2)))

        return {
            "cep_50_m": cep_50,
            "cep_95_m": cep_95,
            "mean_m": mean_err,
            "max_m": max_err,
            "rmse_m": rmse_err,
            "samples": len(errs)
        }

    def print_summary(self):
        """Prints a clean, professional CEP accuracy summary table."""
        m = self.compute_metrics()
        print("\n" + "=" * 65)
        print("GPS-DENIED VISUAL LOCALIZATION POSITION ACCURACY REPORT")
        print("=" * 65)
        print(f" Total Flight Samples Evaluated:  {m['samples']}")
        print(f" CEP-50 (50% Horizontal Error):   {m['cep_50_m']:.2f} meters")
        print(f" CEP-95 (95% Horizontal Error):   {m['cep_95_m']:.2f} meters")
        print(f" Mean Horizontal Error:           {m['mean_m']:.2f} meters")
        print(f" Maximum Position Deviation:      {m['max_m']:.2f} meters")
        print(f" Root Mean Square Error (RMSE):   {m['rmse_m']:.2f} meters")
        print("=" * 65 + "\n")
