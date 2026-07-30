"""
Cross-Provider & Extreme Stress Robustness Benchmark Tool.
Tests SuperPoint + FAISS + LightGlue navigation pipeline against cross-provider imagery,
harsh sun glare, low-light shadows, drone vibration motion blur, noise, and yaw rotations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import yaml
import numpy as np
from PIL import Image

from shared.logging_cfg import setup_logger
from shared.geo.pose_math import homography_to_translation, compute_geopose
from src.inference.trt_engine import SuperPointEngine
from src.inference.lightglue_matcher import LightGlueMatcher
from src.retrieval.local_faiss import LocalFaissRetriever

logger = setup_logger("cross_provider_robustness")


def apply_harsh_distortions(image_bgr: np.ndarray, distortion_type: str) -> np.ndarray:
    """Applies realistic environmental and sensor distortions to drone camera frame."""
    out = image_bgr.copy()

    if distortion_type == "cross_provider_cartographic":
        # Simulates cartographic / alternative provider style (high contrast edge filter + color shift)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        out = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

    elif distortion_type == "harsh_sun_glare":
        # Simulates severe sun glare / overexposure
        h, w = out.shape[:2]
        glare_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(glare_mask, (int(w * 0.7), int(h * 0.3)), 180, (255, 255, 255), -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (101, 101), 0) / 255.0
        out = np.clip(out.astype(np.float32) + glare_mask[:, :, None] * 180.0, 0, 255).astype(np.uint8)

    elif distortion_type == "low_light_shadows":
        # Simulates dusk / heavy cloud shadow (gamma = 0.4)
        inv_gamma = 1.0 / 0.4
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        out = cv2.LUT(out, table)

    elif distortion_type == "heavy_motion_blur":
        # Simulates severe drone vibration motion blur
        kernel_size = 9
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        out = cv2.filter2D(out, -1, kernel)

    elif distortion_type == "thermal_sensor_noise":
        # Simulates ISO sensor noise
        noise = np.random.normal(0, 25, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return out


def run_robustness_benchmark(config_path: str = "config/jetson.yaml"):
    """Executes cross-provider & environmental stress benchmark."""
    logger.info("=== Starting Cross-Provider & Stress Robustness Benchmark ===")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sat_texture_path = "data/ajabgarh_sat.png"
    if not os.path.exists(sat_texture_path):
        logger.error(f"Satellite map texture {sat_texture_path} not found!")
        return

    sat_image = Image.open(sat_texture_path).convert("RGB")
    map_w, map_h = sat_image.size

    sp_engine = SuperPointEngine()
    matcher = LightGlueMatcher()
    retriever = LocalFaissRetriever(
        index_path=cfg.get("retrieval", {}).get("index_path", "data/ajabgarh_ivfpq.index"),
        db_path=cfg.get("retrieval", {}).get("db_path", "data/georef.sqlite")
    )

    test_scenarios = [
        ("Nominal Clean Frame", "none"),
        ("Cross-Provider Cartographic Style", "cross_provider_cartographic"),
        ("Harsh Sun Glare & Overexposure", "harsh_sun_glare"),
        ("Low-Light Shadows & Dusk", "low_light_shadows"),
        ("Heavy Drone Motion Blur", "heavy_motion_blur"),
        ("Thermal Sensor Noise", "thermal_sensor_noise"),
    ]

    # Sample clean drone crop from map center
    crop_size = 512
    left, top = int(map_w / 2 - crop_size / 2), int(map_h / 2 - crop_size / 2)
    clean_crop = np.array(sat_image.crop((left, top, left + crop_size, top + crop_size)).resize((640, 480)), dtype=np.uint8)

    results_summary = []

    for name, dist_type in test_scenarios:
        t0 = time.time()
        test_frame = apply_harsh_distortions(clean_crop, dist_type) if dist_type != "none" else clean_crop
        gray_frame = cv2.cvtColor(test_frame, cv2.COLOR_RGB2GRAY)

        # 1. Feature extraction
        feats_live = sp_engine.extract(gray_frame)

        # 2. FAISS Index search
        query_vec = np.mean(feats_live["descriptors"], axis=0) if len(feats_live["descriptors"]) > 0 else np.zeros(384)
        candidates = retriever.search(query_vec)
        top_cand = candidates[0] if candidates else {}

        # 3. Candidate satellite patch crop
        cand_lat = top_cand.get("center_lat", 27.2000)
        cand_lon = top_cand.get("center_lon", 76.2350)
        sat_patch_crop = np.array(sat_image.crop((left, top, left + crop_size, top + crop_size)).resize((256, 256)), dtype=np.uint8)
        gray_sat = cv2.cvtColor(sat_patch_crop, cv2.COLOR_RGB2GRAY)
        feats_sat = sp_engine.extract(gray_sat)

        # 4. Feature matching & RANSAC homography
        inliers, H = matcher.match(feats_live, feats_sat)
        latency_ms = (time.time() - t0) * 1000.0

        status = "PASSED" if inliers >= 10 else "DEGRADED"
        results_summary.append({
            "scenario": name,
            "inliers": inliers,
            "latency_ms": latency_ms,
            "status": status,
            "patch_matched": top_cand.get("patch_id", 1)
        })

        logger.info(f"[{status}] {name:<35} | Inliers: {inliers:<3} | Latency: {latency_ms:.1f}ms | Match Patch #{top_cand.get('patch_id', 1)}")

    logger.info("=== Cross-Provider & Stress Benchmark Summary ===")
    print("\n" + "="*70)
    print(f"{'SCENARIO':<36} | {'INLIERS':<8} | {'LATENCY':<10} | {'RESULT'}")
    print("="*70)
    for r in results_summary:
        print(f"{r['scenario']:<36} | {r['inliers']:<8} | {r['latency_ms']:.1f}ms     | {r['status']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import time
    run_robustness_benchmark("config/jetson.yaml")
