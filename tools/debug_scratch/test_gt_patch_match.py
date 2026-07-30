import os
import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ds_root = "/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset"
csv_path = os.path.join(ds_root, "01/01.csv")
drone_img_path = os.path.join(ds_root, "01/drone/01_0001.JPG")
sat_tif_path = os.path.join(ds_root, "01/satellite01.tif")

# Read GT lat/lon for 01_0001.JPG
gt_lat, gt_lon, alt_m, heading = None, None, None, None
with open(csv_path, "r") as f:
    f.readline()
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 9 and parts[1] == "01_0001.JPG":
            gt_lat = float(parts[3])
            gt_lon = float(parts[4])
            alt_m = float(parts[5])
            heading = float(parts[8])
            break

print(f"01_0001.JPG GT Telemetry -> Lat: {gt_lat}, Lon: {gt_lon}, Alt: {alt_m}m, Heading: {heading}°")

# GeoTIFF bounds for satellite01.tif
lt_lat, lt_lon = 29.774065, 115.970635
rb_lat, rb_lon = 29.702283, 115.996851

sat_img = Image.open(sat_tif_path).convert("RGB")
map_w, map_h = sat_img.size

# Convert GT lat/lon to exact pixel location on satellite01.tif
px = (gt_lon - lt_lon) / (rb_lon - lt_lon) * map_w
py = (lt_lat - gt_lat) / (lt_lat - rb_lat) * map_h

print(f"Satellite GeoTIFF Center Pixel for GT: ({px:.1f}, {py:.1f}) on {map_w}x{map_h} map")

# Crop satellite patch at exact GT coordinate (scale matched for 405m altitude)
crop_size = 1400  # approx 400m footprint on 0.28m/px map
left = max(0, int(px - crop_size / 2))
top = max(0, int(py - crop_size / 2))

sat_crop = sat_img.crop((left, top, left + crop_size, top + crop_size))
resample_filter = getattr(Image, "Resampling", Image).BILINEAR
sat_crop_rotated = sat_crop.rotate(-heading, resample=resample_filter)

# Load drone image
drone_img = Image.open(drone_img_path).convert("RGB")

# Save side-by-side comparison image
d_vis = np.array(drone_img.resize((640, 480)))
s_vis = np.array(sat_crop_rotated.resize((640, 480)))
side_by_side = np.hstack([d_vis, s_vis])

out_path = "data/gt_ground_truth_comparison.png"
cv2.imwrite(out_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
print(f"Saved true ground truth comparison image to {out_path}")
