"""
Python launcher for Gazebo Harmonic + PX4 SITL simulation host.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import subprocess
import yaml
from shared.logging_cfg import setup_logger

logger = setup_logger("gazebo_launcher")


def launch_simulation(config_path: str, dry_run: bool = True):
    """Launches Gazebo simulation and PX4 SITL physics engine."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg.get("sim", {})
    world_name = sim_cfg.get("world_name", "ajabgarh_world")

    logger.info(f"Preparing Gazebo SITL simulation for world: {world_name}")
    gz_cmd = f"gz sim -v 4 -r data/sim_world.sdf"

    if dry_run:
        logger.info(f"[Dry Run] Gazebo execution command: {gz_cmd}")
        logger.info("[Dry Run] Simulation environment ready.")
        return True

    try:
        logger.info("Executing Gazebo SITL process...")
        subprocess.run(gz_cmd, shell=True, check=True)
        return True
    except Exception as e:
        logger.error(f"Failed to launch Gazebo simulation: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gazebo + PX4 SITL.")
    parser.add_argument("--config", type=str, default="config/sim.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True, help="Perform dry run without GUI")
    parser.add_argument("--run", action="store_true", help="Launch live Gazebo GUI process")
    args = parser.parse_args()

    is_dry = False if args.run else (args.dry_run if args.dry_run is not None else True)
    launch_simulation(args.config, dry_run=is_dry)
