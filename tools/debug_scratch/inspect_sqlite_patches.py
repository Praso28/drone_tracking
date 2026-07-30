import sqlite3

conn = sqlite3.connect("data/georef.sqlite")
c = conn.cursor()
c.execute("SELECT patch_id, center_lat, center_lon, gsd_m_per_px FROM patches LIMIT 10")
rows = c.fetchall()
print("Sample SQLite Patches:")
for r in rows:
    print(r)
conn.close()
