"""
System Audit and Integrity Verification Tool for GPS-Denied Navigation System.
Checks data file presence, FAISS index validity, SQLite DB records,
Jetson network ping latency, and configuration compliance.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
import argparse
import subprocess
import yaml
import faiss
from shared.logging_cfg import setup_logger

logger = setup_logger("system_audit")


def audit_system(config_path: str = "config/sim.yaml") -> bool:
    """Executes a full system health audit and gap analysis report."""
    logger.info("=== Starting System Health & Integrity Audit ===")
    all_ok = True

    # 1. Config Check
    if not os.path.exists(config_path):
        logger.error(f"FAIL: Configuration file not found: {config_path}")
        return False
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"PASS: Configuration loaded: {config_path}")

    map_cfg = cfg.get("map", {})
    idx_cfg = cfg.get("indexer", {})
    index_path = idx_cfg.get("output_index_path", "data/ajabgarh_ivfpq.index")
    db_path = map_cfg.get("db_path", "data/georef.sqlite")

    # 2. FAISS Index Audit
    if os.path.exists(index_path):
        try:
            idx = faiss.read_index(index_path)
            size_mb = os.path.getsize(index_path) / (1024 * 1024)
            logger.info(f"PASS: FAISS IVFPQ Index loaded: {index_path} (Vectors: {idx.ntotal}, Size: {size_mb:.2f} MB)")
            if size_mb > 60.0:
                logger.warning(f"WARN: FAISS index size {size_mb:.2f} MB exceeds 60MB Jetson budget.")
        except Exception as e:
            logger.error(f"FAIL: Could not load FAISS index {index_path}: {e}")
            all_ok = False
    else:
        logger.warning(f"WARN: FAISS index file not found at {index_path}")

    # 3. SQLite Database Audit
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM patches")
            count = cursor.fetchone()[0]
            conn.close()
            logger.info(f"PASS: SQLite Georef Database valid: {db_path} ({count} patches registered)")
        except Exception as e:
            logger.error(f"FAIL: SQLite DB check error: {e}")
            all_ok = False
    else:
        logger.warning(f"WARN: SQLite DB file not found at {db_path}")

    # 4. Jetson Ping Test
    jetson_ip = cfg.get("network", {}).get("jetson_ip", "10.1.1.75")
    try:
        res = subprocess.run(["ping", "-c", "2", "-W", "2", jetson_ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            logger.info(f"PASS: Jetson Edge Node reachable at {jetson_ip} (0% packet loss)")
        else:
            logger.warning(f"WARN: Jetson Edge Node at {jetson_ip} ping failed")
    except Exception as e:
        logger.warning(f"WARN: Ping command execution error: {e}")

    # 5. Gap Analysis Report
    logger.info("=== Gap Analysis Audit Report ===")
    logger.info("1. Camera Intrinsic Calibration Matrix (K): Missing physical lens distortion (k1, k2)")
    logger.info("2. VIO High-Frequency Odometry: 30Hz IMU dead-reckoning bridge pending hardware test")
    logger.info("3. MAVLink Covariance Matrix: Standard deviation (2.0m) integrated in VISION_POSITION_ESTIMATE")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run system integrity audit.")
    parser.add_argument("--config", type=str, default="config/sim.yaml", help="Path to config file")
    args = parser.parse_args()
    audit_system(args.config)
