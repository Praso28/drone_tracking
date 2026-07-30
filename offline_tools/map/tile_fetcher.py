"""
Multithreaded satellite tile fetcher for PC Ground Station.
Downloads map tiles based on bounding box coordinates (min_lat, min_lon, max_lat, max_lon)
and zoom level.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import argparse
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from shared.geo.tile_math import latlon_to_tile
from shared.logging_cfg import setup_logger

logger = setup_logger("tile_fetcher")


def fetch_tile(
    x: int, y: int, zoom: int, out_dir: str, tile_url_template: str
) -> Tuple[bool, str]:
    """Downloads a single satellite tile."""
    out_path = os.path.join(out_dir, f"{zoom}_{x}_{y}.png")
    if os.path.exists(out_path):
        return True, out_path

    url = tile_url_template.format(x=x, y=y, z=zoom)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GPSDenied/1.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(response.content)
            return True, out_path
        else:
            logger.warning(f"Failed to fetch tile {x},{y} at z={zoom}, HTTP {response.status_code}")
            return False, ""
    except Exception as e:
        logger.error(f"Error fetching tile {x},{y}: {e}")
        return False, ""


def download_bbox_tiles(
    bbox: List[float],
    zoom: int,
    out_dir: str,
    tile_url_template: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    max_workers: int = 8
) -> List[str]:
    """Downloads all tiles covering a given bounding box [min_lat, min_lon, max_lat, max_lon]."""
    min_lat, min_lon, max_lat, max_lon = bbox

    x_min, y_max = latlon_to_tile(min_lat, min_lon, zoom)
    x_max, y_min = latlon_to_tile(max_lat, max_lon, zoom)

    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    logger.info(f"Downloading tiles for BBox {bbox} at zoom {zoom} (X: {x_min}..{x_max}, Y: {y_min}..{y_max})")
    tiles_to_download = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles_to_download.append((x, y))

    downloaded_paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fetch_tile, x, y, zoom, out_dir, tile_url_template)
            for x, y in tiles_to_download
        ]
        for future in futures:
            success, path = future.result()
            if success and path:
                downloaded_paths.append(path)

    logger.info(f"Successfully downloaded/verified {len(downloaded_paths)}/{len(tiles_to_download)} tiles.")
    return downloaded_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download satellite tiles for flight map.")
    parser.add_argument("--config", type=str, default="config/sim.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    map_cfg = cfg["map"]
    url_template = map_cfg.get(
        "tile_url_template",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    download_bbox_tiles(
        bbox=map_cfg["bbox"],
        zoom=map_cfg["zoom_level"],
        out_dir=map_cfg["tile_dir"],
        tile_url_template=url_template
    )
