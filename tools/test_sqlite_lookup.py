import sqlite3
from shared.geo.tile_math import haversine_distance

conn = sqlite3.connect("data/georef.sqlite")
c = conn.cursor()

prior_lat = 29.760960
prior_lon = 115.974797

min_lat = prior_lat - 0.05
max_lat = prior_lat + 0.05
min_lon = prior_lon - 0.05
max_lon = prior_lon + 0.05

c.execute("""
    SELECT patch_id, center_lat, center_lon, rotation_deg, gsd_m_per_px, descriptor_index
    FROM patches
    WHERE center_lat BETWEEN ? AND ? AND center_lon BETWEEN ? AND ?
""", (min_lat, max_lat, min_lon, max_lon))

rows = c.fetchall()
print(f"Found {len(rows)} matching patches in bounding box.")

rows_sorted = sorted(rows, key=lambda r: haversine_distance(prior_lat, prior_lon, r[1], r[2]))

print("\nTop 5 Spatially Closest Patches:")
for r in rows_sorted[:5]:
    dist = haversine_distance(prior_lat, prior_lon, r[1], r[2])
    print(f"Patch ID: {r[0]}, center_lat: {r[1]:.6f}, center_lon: {r[2]:.6f}, dist: {dist:.1f}m, rot: {r[3]}")
