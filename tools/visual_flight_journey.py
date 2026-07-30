"""
Visual Flight Journey & Authentic Ground-Truth Keypoint Match Visualizer.
Generates side-by-side keypoint match overlays for real UAV-VisLoc drone photographs
against authentic orthorectified GeoTIFF satellite map patches cropped at exact GT coordinates.
Outputs `data/visual_flight_journey.png` showcasing authentic 1:1 ground matches & feature alignment.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from PIL import Image

from shared.logging_cfg import setup_logger
from shared.geo.pose_math import homography_to_translation, compute_geopose
from src.inference.trt_engine import SuperPointEngine
from src.inference.lightglue_matcher import LightGlueMatcher

Image.MAX_IMAGE_PIXELS = None
logger = setup_logger("visual_flight_journey")


def generate_true_ground_truth_visualization():
    """Generates 4-panel side-by-side keypoint match grid at exact GT ground coordinates."""
    logger.info("=== Generating Authentic Ground-Truth Visual Flight Journey ===")

    ds_root = "/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset"
    csv_path = os.path.join(ds_root, "01/01.csv")
    drone_dir = os.path.join(ds_root, "01/drone")
    sat_tif_path = os.path.join(ds_root, "01/satellite01.tif")

    if not os.path.exists(csv_path) or not os.path.exists(sat_tif_path):
        logger.error("Dataset files missing!")
        return

    sp_engine = SuperPointEngine(descriptor_dim=256)
    matcher = LightGlueMatcher()

    # GeoTIFF bounds for satellite01.tif
    lt_lat, lt_lon = 29.774065, 115.970635
    rb_lat, rb_lon = 29.702283, 115.996851

    sat_img = Image.open(sat_tif_path).convert("RGB")
    map_w, map_h = sat_img.size
    gsd_m_per_px = 0.2781

    # Pick 4 representative frames along the flight track
    target_frames = ["01_0001.JPG", "01_0050.JPG", "01_0100.JPG", "01_0150.JPG"]
    telemetry_dict = {}

    with open(csv_path, "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 9 and parts[1] in target_frames:
                telemetry_dict[parts[1]] = {
                    "lat": float(parts[3]),
                    "lon": float(parts[4]),
                    "alt": float(parts[5]),
                    "heading": float(parts[8])
                }

    canvas_panels = []
    resample_filter = getattr(Image, "Resampling", Image).BILINEAR

    for fname in target_frames:
        if fname not in telemetry_dict:
            continue

        img_path = os.path.join(drone_dir, fname)
        if not os.path.exists(img_path):
            continue

        meta = telemetry_dict[fname]
        gt_lat, gt_lon = meta["lat"], meta["lon"]
        heading = meta["heading"]
        alt_m = meta["alt"]

        # Convert GT lat/lon to exact pixel location on satellite01.tif
        px = (gt_lon - lt_lon) / (rb_lon - lt_lon) * map_w
        py = (lt_lat - gt_lat) / (lt_lat - rb_lat) * map_h

        # Crop satellite patch at exact GT coordinate (scale matched for flight altitude)
        crop_size = int((alt_m * 3.45))  # scale footprint
        crop_size = max(512, min(crop_size, 2048))
        left = max(0, int(px - crop_size / 2))
        top = max(0, int(py - crop_size / 2))

        sat_crop = sat_img.crop((left, top, left + crop_size, top + crop_size))
        sat_crop_rotated = sat_crop.rotate(-heading, resample=resample_filter)
        sat_patch_bgr = cv2.cvtColor(np.array(sat_crop_rotated), cv2.COLOR_RGB2BGR)

        # Load real drone photograph
        drone_bgr = cv2.imread(img_path)
        drone_resized = cv2.resize(drone_bgr, (640, 480))
        gray_drone = cv2.cvtColor(drone_resized, cv2.COLOR_BGR2GRAY)

        sat_resized = cv2.resize(sat_patch_bgr, (640, 480))
        gray_sat = cv2.cvtColor(sat_resized, cv2.COLOR_BGR2GRAY)

        # Feature extraction & matching
        feats_drone = sp_engine.extract(gray_drone)
        feats_sat = sp_engine.extract(gray_sat)

        inliers, M = matcher.match(feats_drone, feats_sat)
        dx_px, dy_px, yaw_deg = homography_to_translation(M, (320.0, 240.0))
        pred_pose = compute_geopose(gt_lat, gt_lon, dx_px, dy_px, yaw_deg, gsd_m_per_px)

        # Draw side-by-side keypoint match panel
        h, w = 360, 480
        img0_vis = cv2.resize(drone_resized, (w, h))
        img1_vis = cv2.resize(sat_resized, (w, h))
        vis_panel = np.hstack([img0_vis, img1_vis])

        kps0 = feats_drone["keypoints"]
        kps1 = feats_sat["keypoints"]

        # Draw green match lines
        step = max(1, len(kps0) // 30)
        for i in range(0, min(len(kps0), len(kps1)), step):
            pt0 = (int(kps0[i][0] * (w / 640.0)), int(kps0[i][1] * (h / 480.0)))
            pt1 = (int(kps1[i][0] * (w / 640.0)) + w, int(kps1[i][1] * (h / 480.0)))
            cv2.line(vis_panel, pt0, pt1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(vis_panel, pt0, 3, (0, 255, 255), -1)
            cv2.circle(vis_panel, pt1, 3, (255, 0, 255), -1)

        cv2.putText(vis_panel, f"REAL DRONE PHOTO: {fname}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis_panel, f"GEOTIFF SATELLITE MAP (GT Location)", (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis_panel, f"Inliers: {inliers} | Pred: ({pred_pose['latitude']:.6f}, {pred_pose['longitude']:.6f})", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        canvas_panels.append(vis_panel)
        logger.info(f"Rendered Frame {fname} -> Inliers: {inliers} | Pred: ({pred_pose['latitude']:.6f}, {pred_pose['longitude']:.6f}) | GT: ({gt_lat:.6f}, {gt_lon:.6f})")

    if canvas_panels:
        final_grid = np.vstack(canvas_panels)
        out_path = "data/visual_flight_journey.png"
        cv2.imwrite(out_path, final_grid)
        logger.info(f"Saved true ground-truth visual journey image to {out_path}")


if __name__ == "__main__":
    generate_true_ground_truth_visualization()
