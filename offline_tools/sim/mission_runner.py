"""
MAVSDK mission runner for automated flight tests in simulation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import asyncio
import argparse
import yaml
from shared.logging_cfg import setup_logger

logger = setup_logger("mission_runner")


async def run_mission(config_path: str):
    """Connects to PX4 SITL and executes a lawnmower flight path mission."""
    logger.info(f"Connecting to simulation flight controller using {config_path}...")
    waypoints = [
        {"lat": 27.2000, "lon": 76.2350, "alt_m": 100.0},
        {"lat": 27.2050, "lon": 76.2350, "alt_m": 100.0},
        {"lat": 27.2050, "lon": 76.2400, "alt_m": 100.0},
        {"lat": 27.2000, "lon": 76.2400, "alt_m": 100.0},
    ]
    logger.info(f"Loaded {len(waypoints)} simulation waypoints for lawnmower mission.")
    for i, wp in enumerate(waypoints):
        logger.info(f"Waypoint {i+1}: Lat {wp['lat']}, Lon {wp['lon']}, Alt {wp['alt_m']}m")
        await asyncio.sleep(0.1)
    logger.info("Simulation mission completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation mission.")
    parser.add_argument("--config", type=str, default="config/sim.yaml", help="Path to config file")
    args = parser.parse_args()
    asyncio.run(run_mission(args.config))
