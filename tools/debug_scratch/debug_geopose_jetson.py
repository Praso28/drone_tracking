import cv2
import numpy as np
from src.inference.trt_engine import SuperPointEngine
from src.inference.lightglue_matcher import LightGlueMatcher
from src.retrieval.local_faiss import LocalFaissRetriever
from shared.geo.pose_math import homography_to_translation, compute_geopose
from src.camera.sim_camera import SimCamera

sp_engine = SuperPointEngine(descriptor_dim=256)
matcher = LightGlueMatcher()
retriever = LocalFaissRetriever("data/ajabgarh_ivfpq.index", "data/georef.sqlite")
sat_sampler = SimCamera(texture_path="data/satellite01.jpg", bbox=(29.702283, 115.970635, 29.774065, 115.996851))

# Load a test drone frame
img_path = "/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset/01/drone/01_0001.JPG"
frame = cv2.imread(img_path)
if frame is None:
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

frame_resized = cv2.resize(frame, (640, 480))
gray_live = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

live_feats = sp_engine.extract(gray_live)
query_vec = np.mean(live_feats["descriptors"], axis=0)

candidates = retriever.search(query_vec)
top_cand = candidates[0] if candidates else {}

print("Top Candidate Raw Item from FAISS / SQLite:")
print(top_cand)

cand_lat = top_cand.get("center_lat", 29.760960)
cand_lon = top_cand.get("center_lon", 115.974797)
gsd_m_per_px = top_cand.get("gsd_m_per_px", 0.2781)

print(f"cand_lat: {cand_lat}, cand_lon: {cand_lon}, gsd: {gsd_m_per_px}")

sat_patch_bgr = sat_sampler.get_frame_at_pose(cand_lat, cand_lon, alt_m=405.0, heading_deg=top_cand.get("rotation_deg", 0))
gray_sat = cv2.cvtColor(sat_patch_bgr, cv2.COLOR_BGR2GRAY)
sat_feats = sp_engine.extract(gray_sat)

inliers, M = matcher.match(live_feats, sat_feats)
print(f"Inliers: {inliers}, M matrix:\n{M}")

dx_px, dy_px, yaw_deg = homography_to_translation(M, (320.0, 240.0))
print(f"dx_px: {dx_px}, dy_px: {dy_px}, yaw_deg: {yaw_deg}")

raw_pose = compute_geopose(cand_lat, cand_lon, dx_px, dy_px, yaw_deg, gsd_m_per_px)
print("Resulting Raw Pose:")
print(raw_pose)
