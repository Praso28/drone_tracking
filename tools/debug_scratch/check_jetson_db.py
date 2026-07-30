import sqlite3
import numpy as np
from src.retrieval.local_faiss import LocalFaissRetriever

print("=== Checking Jetson Local FAISS & SQLite Database ===")

conn = sqlite3.connect("data/georef.sqlite")
c = conn.cursor()
c.execute("SELECT patch_id, center_lat, center_lon, gsd_m_per_px FROM patches LIMIT 5")
rows = c.fetchall()
print("SQLite Database First 5 Patches:")
for r in rows:
    print(r)
conn.close()

retriever = LocalFaissRetriever("data/ajabgarh_ivfpq.index", "data/georef.sqlite")
dummy_query = np.zeros((1, 256), dtype=np.float32)
results = retriever.search(dummy_query)
print("FAISS Search Results for Dummy Query:")
for item in results:
    print(item)
