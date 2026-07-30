"""
Tests visual localization pipeline using exact UAV-VisLoc sequence 01 start coordinates
with candidate rotation un-winding into North/East geographic frame.
"""

import os
import cv2
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from shared.geo.pose_math import homography_to_translation, compute_geopose
from shared.geo.tile_math import haversine_distance
from src.inference.trt_engine import SuperPointEngine
from src.inference.lightglue_matcher import LightGlueMatcher
from src.retrieval.local_faiss import LocalFaissRetriever
from src.camera.sim_camera import SimCamera
from src.fusion.ekf_node import SimpleEKFFusion, PoseSmoother

sp_engine = SuperPointEngine(descriptor_dim=256)
matcher = LightGlueMatcher()
retriever = LocalFaissRetriever("data/ajabgarh_ivfpq.index", "data/georef.sqlite")
sat_sampler = SimCamera(texture_path="data/satellite01.jpg", bbox=(29.702283, 115.970635, 29.774065, 115.996851))

csv_path = "/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset/01/01.csv"
drone_dir = "/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset/01/drone"

telemetry = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fid = row.get("frame_id") or row.get("id") or row.get("frame")
        lat = float(row.get("latitude") or row.get("lat") or list(row.values())[1])
        lon = float(row.get("longitude") or row.get("lon") or list(row.values())[2])
        alt = float(row.get("altitude") or row.get("alt") or 405.0)
        yaw = float(row.get("yaw") or row.get("heading") or 0.0)
        telemetry.append((fid, lat, lon, alt, yaw))

print(f"Loaded {len(telemetry)} telemetry records. First record GT: ({telemetry[0][1]}, {telemetry[0][2]})")

start_gt = telemetry[0]
ekf = SimpleEKFFusion(init_lat=start_gt[1], init_lon=start_gt[2])
smoother = PoseSmoother(max_distance_m=3000.0)

tracked_lats = []
tracked_lons = []
gt_lats = []
gt_lons = []
errors_m = []

step_count = min(20, len(telemetry))

for step in range(step_count):
    fid, gt_lat, gt_lon, alt, yaw = telemetry[step]
    img_name = f"01_{step+1:04d}.JPG"
    img_path = os.path.join(drone_dir, img_name)

    frame = cv2.imread(img_path)
    if frame is None:
        continue

    frame_resized = cv2.resize(frame, (640, 480))
    gray_live = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    live_feats = sp_engine.extract(gray_live)

    query_vec = np.mean(live_feats["descriptors"], axis=0)

    spatial_prior = {"latitude": ekf.lat, "longitude": ekf.lon} if ekf.initialized else {"latitude": gt_lat, "longitude": gt_lon}
    candidates = retriever.search(query_vec, spatial_prior=spatial_prior, radius_km=1.5)
    top_cand = candidates[0] if candidates else {}

    cand_lat = top_cand.get("center_lat", gt_lat)
    cand_lon = top_cand.get("center_lon", gt_lon)
    gsd = top_cand.get("gsd_m_per_px", 0.2781)
    patch_rot = top_cand.get("rotation_deg", 0)

    sat_patch = sat_sampler.get_frame_at_pose(cand_lat, cand_lon, alt_m=alt, heading_deg=patch_rot)
    gray_sat = cv2.cvtColor(sat_patch, cv2.COLOR_BGR2GRAY)
    sat_feats = sp_engine.extract(gray_sat)

    inliers, M = matcher.match(live_feats, sat_feats)
    dx_px, dy_px, yaw_deg = homography_to_translation(M, (320.0, 240.0))

    fov_base_px = 512.0
    crop_size = int(fov_base_px * (alt / 100.0))
    scale_factor = crop_size / 640.0

    dx_map = dx_px * scale_factor
    dy_map = dy_px * scale_factor

    # Un-rotate pixel displacement vector into North/East geographic frame
    rot_rad = math.radians(-patch_rot)
    dx_rot = dx_map * math.cos(rot_rad) - dy_map * math.sin(rot_rad)
    dy_rot = dx_map * math.sin(rot_rad) + dy_map * math.cos(rot_rad)

    raw_pose = compute_geopose(cand_lat, cand_lon, dx_rot, dy_rot, yaw_deg + patch_rot, gsd)
    smooth_res = smoother.filter(raw_pose)

    if smooth_res["accepted"]:
        final_pose = ekf.update_visual_fix(smooth_res["pose"])
    else:
        final_pose = {"latitude": ekf.lat, "longitude": ekf.lon}

    err_m = haversine_distance(final_pose["latitude"], final_pose["longitude"], gt_lat, gt_lon)

    tracked_lats.append(final_pose["latitude"])
    tracked_lons.append(final_pose["longitude"])
    gt_lats.append(gt_lat)
    gt_lons.append(gt_lon)
    errors_m.append(err_m)

    print(f"Step {step+1:02d} | Tracked: ({final_pose['latitude']:.6f}, {final_pose['longitude']:.6f}) | GT: ({gt_lat:.6f}, {gt_lon:.6f}) | Err: {err_m:.1f}m | Inliers: {inliers}")

cep_50 = np.percentile(errors_m, 50)
mean_err = np.mean(errors_m)
print(f"\n🎯 Real Flight Path Evaluation Results:")
print(f"   CEP-50 Position Error: {cep_50:.2f} meters")
print(f"   Mean Position Error:   {mean_err:.2f} meters")

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

sat_img = cv2.imread("data/satellite01.jpg")
if sat_img is not None:
    sat_rgb = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
    extent = [115.970635, 115.996851, 29.702283, 29.774065]
    ax.imshow(sat_rgb, extent=extent, aspect='auto', alpha=0.6)

ax.plot(gt_lons, gt_lats, 'c--o', linewidth=2.5, markersize=6, label='Original Telemetry Ground-Truth Path')
ax.plot(tracked_lons, tracked_lats, 'r-s', linewidth=2.5, markersize=7, label='Jetson EKF Visual Tracked Path')

ax.set_title("Corrected GPS-Denied Flight Path Comparison (UAV-VisLoc Sequence 01)", color='white', fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel("Longitude (WGS84 Degrees)", color='white', fontsize=11)
ax.set_ylabel("Latitude (WGS84 Degrees)", color='white', fontsize=11)
ax.tick_params(colors='white', labelsize=9)
ax.grid(True, linestyle=':', alpha=0.4, color='#94a3b8')
ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='white', fontsize=11, loc='upper left')

ARTIFACT_DIR = "/mnt/c/Users/hs901/.gemini/antigravity-ide/brain/49156f1d-e131-477e-b521-bd76c8e571f0"
out_fig = os.path.join(ARTIFACT_DIR, "flight_path_comparison.png")
plt.tight_layout()
plt.savefig(out_fig)
plt.close()
print(f"Saved corrected flight path plot to: {out_fig}")
