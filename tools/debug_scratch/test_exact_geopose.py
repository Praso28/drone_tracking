import cv2
import numpy as np
import math
from shared.geo.tile_math import pixel_to_latlon
from shared.geo.pose_math import homography_to_translation, compute_geopose

# Test with real candidate values
cand_lat = 29.740294
cand_lon = 115.974797
gsd_m_per_px = 0.2781

# M matrix from OpenCV estimateAffinePartial2D
# Suppose drone photo center is matched near satellite patch center
M = np.array([
    [0.96, -0.27, 25.0],
    [0.27,  0.96, -15.0]
], dtype=np.float32)

cx, cy = 320.0, 240.0
proj_x = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
proj_y = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]

dx_px = proj_x - cx
dy_px = proj_y - cy

print(f"proj_x: {proj_x:.2f}, proj_y: {proj_y:.2f}")
print(f"dx_px: {dx_px:.2f}, dy_px: {dy_px:.2f}")

res = pixel_to_latlon(cand_lat, cand_lon, dx_px, dy_px, gsd_m_per_px)
print(f"Input Ref: ({cand_lat}, {cand_lon}) -> Output Geopose: {res}")
