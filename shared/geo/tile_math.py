"""
Pure geospatial math utilities for GPS-Denied visual navigation.
Provides conversion functions between lat/lon, tile coordinates (XYZ),
pixel offsets, Ground Sampling Distance (GSD), and Haversine distance calculations.
Includes correct sign inversion for image-to-camera coordinate transformation.
"""

import math
from typing import Tuple


def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Converts latitude and longitude to tile (X, Y) coordinates at a given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_latlon(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Converts tile (X, Y) coordinates to northwest corner latitude and longitude."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def calculate_gsd(lat: float, zoom: int, tile_size_px: int = 256) -> float:
    """Calculates Ground Sampling Distance (GSD) in meters per pixel."""
    earth_radius_m = 6378137.0
    lat_rad = math.radians(lat)
    initial_resolution = 2.0 * math.pi * earth_radius_m / tile_size_px
    resolution = initial_resolution * math.cos(lat_rad) / (2.0 ** zoom)
    return resolution


def pixel_to_latlon(
    anchor_lat: float,
    anchor_lon: float,
    dx_px: float,
    dy_px: float,
    gsd_m_per_px: float
) -> Tuple[float, float]:
    """
    Converts pixel offsets (dx, dy) relative to an anchor coordinate into new lat/lon.
    Note: Live frame feature shift +X (right) relative to patch means camera is situated to West (-X).
    Live frame feature shift +Y (down) relative to patch means camera is situated to North (-Y).
    """
    earth_radius_m = 6378137.0
    dx_m = -dx_px * gsd_m_per_px
    dy_m = -dy_px * gsd_m_per_px

    dlat = (dy_m / earth_radius_m) * (180.0 / math.pi)
    dlon = (dx_m / (earth_radius_m * math.cos(math.radians(anchor_lat)))) * (180.0 / math.pi)

    return anchor_lat + dlat, anchor_lon + dlon


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great-circle distance between two points in meters using Haversine formula."""
    earth_radius_m = 6378137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_m * c
