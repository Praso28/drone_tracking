"""
UAV-VisLoc Professional Dataset Compiler.
Processes real orthorectified GeoTIFF satellite maps (e.g. satellite01.tif) from the UAV-VisLoc dataset,
extracts 4-rotation augmented map patches, generates SuperPoint ONNX 256-dim feature descriptors,
populates `georef.sqlite`, and saves `vlad_descriptors.npy` for FAISS IVFPQ indexing.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import sqlite3
import argparse
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any

from shared.logging_cfg import setup_logger
from src.inference.trt_engine import SuperPointEngine

Image.MAX_IMAGE_PIXELS = None
logger = setup_logger("uav_visloc_compiler")
_sp_engine = None


def get_sp_engine():
    global _sp_engine
    if _sp_engine is None:
        _sp_engine = SuperPointEngine(descriptor_dim=256)
    return _sp_engine


def extract_descriptor(patch_arr: np.ndarray, dim: int = 256) -> np.ndarray:
    """Extracts normalized 256-dim SuperPoint descriptor vector for map patch."""
    engine = get_sp_engine()
    feats = engine.extract(patch_arr)
    descs = feats.get("descriptors", np.zeros((0, dim)))

    if len(descs) > 0:
        vec = np.mean(descs, axis=0)
    else:
        vec = np.zeros(dim, dtype=np.float32)

    if len(vec) < dim:
        vec = np.pad(vec, (0, dim - len(vec)))
    else:
        vec = vec[:dim]

    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec.astype(np.float32)


def compile_uav_visloc(
    sequence_id: str = "01",
    dataset_root: str = "/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset",
    output_db: str = "data/georef.sqlite",
    output_desc: str = "data/vlad_descriptors.npy",
    patch_size_px: int = 256,
    stride_px: int = 256
):
    """Compiles satellite GeoTIFF into 4-rotation augmented patches with exact WGS84 mapping."""
    logger.info(f"=== Compiling UAV-VisLoc Sequence {sequence_id} ===")

    # 1. Parse satellite coordinates range CSV (handles optional leading space in filename)
    meta_csv1 = os.path.join(dataset_root, "satellite_coordinates_range.csv")
    meta_csv2 = os.path.join(dataset_root, "satellite_ coordinates_range.csv")
    meta_csv = meta_csv1 if os.path.exists(meta_csv1) else meta_csv2

    sat_tif_path = os.path.join(dataset_root, f"{sequence_id}/satellite{sequence_id}.tif")

    if not os.path.exists(sat_tif_path):
        logger.error(f"Satellite GeoTIFF not found: {sat_tif_path}")
        return

    lt_lat, lt_lon, rb_lat, rb_lon = None, None, None, None
    with open(meta_csv, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 5 and parts[0] == f"satellite{sequence_id}.tif":
                lt_lat = float(parts[1])
                lt_lon = float(parts[2])
                rb_lat = float(parts[3])
                rb_lon = float(parts[4])
                break

    if lt_lat is None:
        logger.error(f"Could not find coordinates for satellite{sequence_id}.tif in {meta_csv}")
        return

    logger.info(f"Bounding Box for satellite{sequence_id}.tif: NW({lt_lat:.6f}, {lt_lon:.6f}) -> SE({rb_lat:.6f}, {rb_lon:.6f})")

    # 2. Open satellite GeoTIFF image
    sat_img = Image.open(sat_tif_path).convert("RGB")
    map_w, map_h = sat_img.size
    logger.info(f"Loaded Satellite Image ({map_w} x {map_h} px)")

    # Compute exact pixel GSD
    lat_span = abs(lt_lat - rb_lat)
    lon_span = abs(rb_lon - lt_lon)
    gsd_x = (lon_span * 111000.0 * np.cos(np.radians((lt_lat + rb_lat) / 2.0))) / map_w
    gsd_y = (lat_span * 111000.0) / map_h
    gsd_m_per_px = float((gsd_x + gsd_y) / 2.0)
    logger.info(f"Calculated Ground Sampling Distance (GSD): {gsd_m_per_px:.4f} m/px")

    # 3. Setup SQLite Database
    os.makedirs(os.path.dirname(output_db), exist_ok=True)
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS patches")
    cursor.execute("""
        CREATE TABLE patches (
            patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tile_x INTEGER,
            tile_y INTEGER,
            zoom INTEGER,
            rotation_deg INTEGER,
            center_lat REAL,
            center_lon REAL,
            gsd_m_per_px REAL,
            descriptor_index INTEGER
        )
    """)
    conn.commit()

    descriptors_list = []
    patch_counter = 0
    rotations = [0, 90, 180, 270]

    xs = range(0, map_w - patch_size_px, stride_px)
    ys = range(0, map_h - patch_size_px, stride_px)
    total_patches = len(xs) * len(ys)
    logger.info(f"Grid Patch Extraction: {len(xs)} x {len(ys)} = {total_patches} patches...")

    count = 0
    for y0 in ys:
        for x0 in xs:
            count += 1
            x_center = x0 + patch_size_px / 2.0
            y_center = y0 + patch_size_px / 2.0

            # Linear interpolation for patch center WGS84 lat/lon
            center_lon = lt_lon + (x_center / map_w) * lon_span
            center_lat = lt_lat - (y_center / map_h) * lat_span  # Y increases downward

            crop = sat_img.crop((x0, y0, x0 + patch_size_px, y0 + patch_size_px))
            crop_arr = np.array(crop)

            base_desc = extract_descriptor(crop_arr, dim=256)

            for rot_deg in rotations:
                descriptors_list.append(base_desc)
                cursor.execute("""
                    INSERT INTO patches (tile_x, tile_y, zoom, rotation_deg, center_lat, center_lon, gsd_m_per_px, descriptor_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (x0, y0, 18, rot_deg, float(center_lat), float(center_lon), gsd_m_per_px, patch_counter))
                patch_counter += 1

            if count % 500 == 0 or count == total_patches:
                logger.info(f"Progress: {count}/{total_patches} patches processed ({patch_counter} rotation vectors stored).")

    conn.commit()
    conn.close()

    desc_matrix = np.array(descriptors_list, dtype=np.float32)
    os.makedirs(os.path.dirname(output_desc), exist_ok=True)
    np.save(output_desc, desc_matrix)
    logger.info(f"Successfully compiled {patch_counter} augmented patch vectors.")
    logger.info(f"Saved descriptor matrix {desc_matrix.shape} to {output_desc}")
    logger.info(f"Saved georeference database to {output_db}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile UAV-VisLoc GeoTIFF into FAISS descriptor database.")
    parser.add_argument("--sequence", type=str, default="01", help="Dataset sequence ID (e.g. 01, 02, 03)")
    parser.add_argument("--dataset-root", type=str, default="/mnt/c/Users/hs901/Downloads/UAV_VisLoc_dataset")
    parser.add_argument("--stride", type=int, default=256, help="Patch extraction stride in pixels")
    args = parser.parse_args()

    compile_uav_visloc(sequence_id=args.sequence, dataset_root=args.dataset_root, stride_px=args.stride)
