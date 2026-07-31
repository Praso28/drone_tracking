"""
Visual Trajectory Comparator for GPS-Denied Navigation Pipeline.
Reads `data/flight_log.csv` and renders a 2D plot comparing the Authentic Ground Truth path
against the Jetson predicted path.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

# Ensure matplotlib runs in headless mode if no display
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.evaluate_cep_metrics import CEPEvaluator
from shared.logging_cfg import setup_logger

logger = setup_logger("plot_trajectory")

def plot_flight_trajectory(csv_path: str = "data/flight_log.csv", output_img: str = "data/trajectory_comparison.png"):
    if not os.path.exists(csv_path):
        logger.error(f"Flight log CSV not found at {csv_path}. Did you run the main engine first?")
        return

    logger.info(f"Loading flight log from {csv_path}...")
    df = pd.read_csv(csv_path)

    if df.empty:
        logger.error("Flight log is empty.")
        return

    # Extract coordinates
    pred_lats = df['pred_lat'].values
    pred_lons = df['pred_lon'].values
    gt_lats = df['gt_lat'].values
    gt_lons = df['gt_lon'].values
    phases = df['phase'].values

    # Evaluate CEP metrics
    evaluator = CEPEvaluator()
    for p_lat, p_lon, g_lat, g_lon in zip(pred_lats, pred_lons, gt_lats, gt_lons):
        evaluator.add_sample(p_lat, p_lon, g_lat, g_lon)
    
    metrics = evaluator.compute_metrics()
    evaluator.print_summary()

    # Create the plot
    plt.figure(figsize=(12, 10), facecolor='#1E1E1E')
    ax = plt.gca()
    ax.set_facecolor('#1E1E1E')
    ax.grid(color='#333333', linestyle='--', linewidth=0.5)
    
    # Plot Ground Truth
    plt.plot(gt_lons, gt_lats, color='#00FF00', linewidth=3, alpha=0.7, label='Authentic Ground Truth (UAV-VisLoc)')
    plt.scatter(gt_lons, gt_lats, color='#00FF00', s=20, alpha=0.5)

    # Plot Prediction
    plt.plot(pred_lons, pred_lats, color='#FF3366', linewidth=2, linestyle='--', label='Jetson Edge Prediction')
    plt.scatter(pred_lons, pred_lats, color='#FF3366', s=20, marker='x')

    # Mark Start/End
    plt.scatter([gt_lons[0]], [gt_lats[0]], color='white', s=150, marker='*', label='Start Point', zorder=5)
    plt.scatter([gt_lons[-1]], [gt_lats[-1]], color='cyan', s=150, marker='X', label='End Point', zorder=5)

    # Style
    plt.title("UAV-VisLoc: Jetson GPS-Denied Edge Navigation vs Ground Truth", color='white', pad=20, fontsize=16)
    plt.xlabel("Longitude", color='white', fontsize=12)
    plt.ylabel("Latitude", color='white', fontsize=12)
    plt.tick_params(colors='white')
    
    # Add metrics text box
    metrics_text = (
        f"Flight Steps: {metrics['samples']}\n"
        f"CEP-50 Error: {metrics['cep_50_m']:.2f} m\n"
        f"CEP-95 Error: {metrics['cep_95_m']:.2f} m\n"
        f"RMSE: {metrics['rmse_m']:.2f} m"
    )
    plt.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8, edgecolor='none'), color='white')

    plt.legend(facecolor='#333333', edgecolor='none', labelcolor='white', loc='lower right')
    plt.tight_layout()

    # Save output
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    logger.info(f"Saved trajectory comparison visualization to {output_img}")

if __name__ == "__main__":
    plot_flight_trajectory()
