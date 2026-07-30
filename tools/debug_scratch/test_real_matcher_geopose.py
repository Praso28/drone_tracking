import os
import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from shared.geo.pose_math import homography_to_translation, compute_geopose
from src.inference.trt_engine import SuperPointEngine
from src.inference.lightglue_matcher import LightGlueMatcher
from src.camera.sim_camera import SimCamera

sp_engine = SuperPointEngine(descriptor_dim=256)
matcher = LightGlueMatcher()

drone_path = "/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset/01/drone/01_0001.JPG"
drone_bgr = cv2.imread(drone_path)
drone_resized = cv2.resize(drone_bgr, (640, 480))
gray_drone = cv2.cvtColor(drone_resized, cv2.COLOR_BGR2GRAY)

cand_lat = 29.760960
cand_lon = 115.974797
heading = 163.87
alt_m = 405.76
gsd = 0.2781

sat_sampler = SimCamera(texture_path="data/satellite01.jpg", bbox=(29.702283, 115.970635, 29.774065, 115.996851))
sat_patch = sat_sampler.get_frame_at_pose(cand_lat, cand_lon, alt_m=alt_m, heading_deg=heading)
gray_sat = cv2.cvtColor(sat_patch, cv2.COLOR_BGR2GRAY)

feats0 = sp_engine.extract(gray_drone)
feats1 = sp_engine.extract(gray_sat)

inliers, M = matcher.match(feats0, feats1)
print(f"Inliers: {inliers}")
print(f"M Matrix:\n{M}")

dx_px, dy_px, yaw_deg = homography_to_translation(M, (320.0, 240.0))
print(f"dx_px: {dx_px:.2f}, dy_px: {dy_px:.2f}, yaw_deg: {yaw_deg:.2f}")

crop_size = int(512.0 * (alt_m / 100.0))
scale_factor = crop_size / 640.0
dx_map_px = dx_px * scale_factor
dy_map_px = dy_px * scale_factor

pose = compute_geopose(cand_lat, cand_lon, dx_map_px, dy_map_px, yaw_deg, gsd)
print("Corrected Calculated Geopose:")
print(pose)
print(f"Ground Truth Geopose: ({cand_lat}, {cand_lon})")
