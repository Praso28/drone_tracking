import os
import sqlite3
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ds_path = "/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset"
img_path = os.path.join(ds_path, "01/drone/01_0001.JPG")
sat_path = os.path.join(ds_path, "01/satellite01.tif")

if os.path.exists(img_path):
    img = Image.open(img_path)
    print("Drone Image Size:", img.size)
else:
    print("Drone image not found:", img_path)

if os.path.exists(sat_path):
    sat = Image.open(sat_path)
    print("Satellite GeoTIFF Size:", sat.size)
else:
    print("Satellite TIF not found:", sat_path)
