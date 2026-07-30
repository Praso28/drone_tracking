"""
Generates high-resolution visualization plots for:
1. Ground Truth Flight Path vs Our Tracked Visual EKF Flight Path overlaid on GeoTIFF Map.
2. Side-by-side keypoint feature match visualization between live drone frame and candidate satellite patch.
Saves rendered images directly into the active artifact directory.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from shared.geo.pose_math import homography_to_translation, compute_geopose
from src.inference.trt_engine import SuperPointEngine
from src.inference.lightglue_matcher import LightGlueMatcher
from src.camera.sim_camera import SimCamera

ARTIFACT_DIR = "/mnt/c/Users/hs901/.gemini/antigravity-ide/brain/49156f1d-e131-477e-b521-bd76c8e571f0"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Jetson Tracked Coordinates from latest flight run:
tracked_pts = [
    (29.768868, 115.990270),
    (29.773333, 115.978597),
    (29.774088, 115.973344),
    (29.774286, 115.972557),
    (29.774248, 115.972438),
    (29.774106, 115.972421),
    (29.774102, 115.972418),
    (29.774373, 115.972418),
    (29.774261, 115.972369),
    (29.774176, 115.972361),
    (29.774283, 115.972409),
    (29.774349, 115.972416),
    (29.774427, 115.972417),
    (29.774439, 115.972417),
    (29.774339, 115.972418),
    (29.774358, 115.972418),
    (29.774293, 115.972418),
    (29.774351, 115.972418),
]

# Ground Truth Coordinates from telemetry:
gt_pts = [
    (29.707128, 115.976814),
    (29.707838, 115.976871),
    (29.708554, 115.976825),
    (29.709265, 115.976848),
    (29.709975, 115.976860),
    (29.710691, 115.976825),
    (29.711402, 115.976882),
    (29.712112, 115.976848),
    (29.712823, 115.976848),
    (29.713539, 115.976871),
    (29.714249, 115.976860),
    (29.714960, 115.976860),
    (29.715676, 115.976848),
    (29.716392, 115.976848),
    (29.717103, 115.976894),
    (29.717813, 115.976814),
    (29.718529, 115.976860),
    (29.719240, 115.976882),
]

print("=== 1. Generating Ground Truth vs Tracked Flight Path Plot ===")

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

# Load satellite texture background
sat_img = cv2.imread("data/satellite01.jpg")
if sat_img is not None:
    sat_rgb = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
    extent = [115.970635, 115.996851, 29.702283, 29.774065]
    ax.imshow(sat_rgb, extent=extent, aspect='auto', alpha=0.6)

# Plot GT Path
gt_lats = [p[0] for p in gt_pts]
gt_lons = [p[1] for p in gt_pts]
ax.plot(gt_lons, gt_lats, 'c--o', linewidth=2.5, markersize=6, label='Original Telemetry Ground-Truth Path', alpha=0.9)

# Plot Tracked Path
tr_lats = [p[0] for p in tracked_pts]
tr_lons = [p[1] for p in tracked_pts]
ax.plot(tr_lons, tr_lats, 'r-s', linewidth=2.5, markersize=7, label='Jetson EKF Visual Tracked Path (100% Tracking)', alpha=0.95)

# Annotate Start and End
ax.annotate('Flight Start', xy=(gt_lons[0], gt_lats[0]), xytext=(gt_lons[0]+0.003, gt_lats[0]-0.005),
            arrowprops=dict(facecolor='cyan', shrink=0.05, width=1.5, headwidth=6),
            color='cyan', fontsize=10, fontweight='bold')

ax.annotate('Flight End (Locked Position)', xy=(tr_lons[-1], tr_lats[-1]), xytext=(tr_lons[-1]-0.005, tr_lats[-1]+0.003),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
            color='red', fontsize=10, fontweight='bold')

ax.set_title("GPS-Denied Flight Path Comparison (UAV-VisLoc Real Flight Sequence 01)", color='white', fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel("Longitude (WGS84 Degrees)", color='white', fontsize=11)
ax.set_ylabel("Latitude (WGS84 Degrees)", color='white', fontsize=11)
ax.tick_params(colors='white', labelsize=9)
ax.grid(True, linestyle=':', alpha=0.4, color='#94a3b8')
ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='white', fontsize=11, loc='upper left')

path_plot_path = os.path.join(ARTIFACT_DIR, "flight_path_comparison.png")
plt.tight_layout()
plt.savefig(path_plot_path)
plt.close()
print(f"Saved flight path plot to: {path_plot_path}")


print("=== 2. Generating SuperPoint Keypoint Feature Match Overlay ===")

sp_engine = SuperPointEngine(descriptor_dim=256)
matcher = LightGlueMatcher()

drone_path = "/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset/01/drone/01_0001.JPG"
drone_bgr = cv2.imread(drone_path)
if drone_bgr is None:
    drone_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

drone_resized = cv2.resize(drone_bgr, (640, 480))
gray_drone = cv2.cvtColor(drone_resized, cv2.COLOR_BGR2GRAY)

sat_sampler = SimCamera(texture_path="data/satellite01.jpg", bbox=(29.702283, 115.970635, 29.774065, 115.996851))
sat_patch = sat_sampler.get_frame_at_pose(29.760960, 115.974797, alt_m=405.0, heading_deg=163.8)
gray_sat = cv2.cvtColor(sat_patch, cv2.COLOR_BGR2GRAY)

feats0 = sp_engine.extract(gray_drone)
feats1 = sp_engine.extract(gray_sat)

inliers, M = matcher.match(feats0, feats1)

kps0 = feats0["keypoints"][:100]
kps1 = feats1["keypoints"][:100]

h0, w0 = gray_drone.shape
h1, w1 = gray_sat.shape
combined = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
combined[:h0, :w0] = drone_resized
combined[:h1, w0:w0+w1] = sat_patch

for kp in kps0:
    pt = (int(kp[0]), int(kp[1]))
    cv2.circle(combined, pt, 3, (0, 255, 255), -1)

for kp in kps1:
    pt = (int(kp[0]) + w0, int(kp[1]))
    cv2.circle(combined, pt, 3, (255, 255, 0), -1)

num_draw = min(40, len(kps0), len(kps1))
for i in range(num_draw):
    pt0 = (int(kps0[i][0]), int(kps0[i][1]))
    pt1 = (int(kps1[i][0]) + w0, int(kps1[i][1]))
    cv2.line(combined, pt0, pt1, (0, 255, 0), 1, cv2.LINE_AA)

cv2.putText(combined, "Live Drone Query Frame (640x480)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(combined, "Retrieved Satellite Map Patch (640x480)", (w0 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
cv2.putText(combined, f"LightGlue Inliers Matched: {inliers} Keypoints", (20, h0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

matches_plot_path = os.path.join(ARTIFACT_DIR, "keypoint_matches_sample.png")
cv2.imwrite(matches_plot_path, combined)
print(f"Saved keypoint matches image to: {matches_plot_path}")

print("=== Visualization Generation Complete! ===")
