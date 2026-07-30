import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ds_path = "/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset/01/satellite01.tif"
out_path = "data/satellite01.jpg"

if os.path.exists(ds_path):
    print(f"Converting {ds_path} to {out_path}...")
    img = Image.open(ds_path).convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, quality=85)
    print(f"Saved {out_path} ({img.size})")
else:
    print("GeoTIFF file not found:", ds_path)
