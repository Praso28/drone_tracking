"""
Unit tests for coordinate frame transformations:
WGS84 Geodetic <-> Local ENU (East-North-Up) <-> Pixel offsets.
"""

import pytest
import math
import numpy as np
from shared.geo.tile_math import (
    calculate_gsd,
    pixel_to_latlon,
    haversine_distance,
    latlon_to_tile,
    tile_to_latlon,
)
from shared.geo.pose_math import compute_geopose


def test_wgs84_to_local_enu_conversion():
    lat0, lon0 = 27.2000, 76.2350
    # Move ~100m East, ~50m North
    dx_east_m = 100.0
    dy_north_m = 50.0

    # Convert ENU offset to Lat/Lon
    earth_radius_m = 6378137.0
    dlat = (dy_north_m / earth_radius_m) * (180.0 / math.pi)
    dlon = (dx_east_m / (earth_radius_m * math.cos(math.radians(lat0)))) * (180.0 / math.pi)

    lat_target = lat0 + dlat
    lon_target = lon0 + dlon

    # Compute Haversine distance
    dist_m = haversine_distance(lat0, lon0, lat_target, lon_target)
    expected_dist = math.sqrt(dx_east_m**2 + dy_north_m**2)

    assert abs(dist_m - expected_dist) < 0.1  # Sub-10cm precision


def test_gsd_scaling_with_altitude():
    lat = 27.2000
    zoom = 18
    gsd_base = calculate_gsd(lat, zoom, tile_size_px=256)

    # At higher zoom level (zoom 19), GSD should halve (higher resolution)
    gsd_high = calculate_gsd(lat, zoom + 1, tile_size_px=256)
    assert abs(gsd_high - (gsd_base / 2.0)) < 1e-4


def test_pixel_displacement_to_geopose():
    ref_lat, ref_lon = 27.2000, 76.2350
    gsd = 0.5  # 0.5 m/px

    # Image shift of 20px Right (East), 10px Up (North)
    pose = compute_geopose(
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        dx_px=20.0,
        dy_px=-10.0,  # Image dy is inverted (-dy is North)
        yaw_deg=15.0,
        gsd_m_per_px=gsd
    )

    dist = haversine_distance(ref_lat, ref_lon, pose["latitude"], pose["longitude"])
    expected_dist = math.sqrt((20.0 * 0.5)**2 + (10.0 * 0.5)**2)  # ~11.18m

    assert abs(dist - expected_dist) < 0.05
    assert pose["heading_deg"] == 15.0
    # dy_px=-10 corresponds to North movement -> latitude increases
    assert pose["latitude"] > ref_lat
    # dx_px=20 corresponds to West camera displacement -> longitude decreases
    assert pose["longitude"] < ref_lon
