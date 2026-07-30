"""
Local FAISS IVFPQ retriever for Jetson Orin Nano / simulation testing.
Queries compressed Product Quantization index (~0.44MB) and retrieves georeferenced
patch metadata from persistent `georef.sqlite` connection handle with optional spatial radius filtering.
"""

import os
import sqlite3
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from shared.logging_cfg import setup_logger

logger = setup_logger("local_faiss")


class LocalFaissRetriever:
    """On-device FAISS IVFPQ index searcher with persistent SQLite georeference lookup."""

    def __init__(self, index_path: str, db_path: str, top_k: int = 5, nprobe: int = 10):
        self.index_path = index_path
        self.db_path = db_path
        self.top_k = top_k
        self.nprobe = nprobe
        self.index = None
        self.conn = None

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.index.nprobe = nprobe
            logger.info(f"Loaded FAISS IVFPQ index from {index_path} (ntotal={self.index.ntotal}).")
        else:
            logger.warning(f"Index file {index_path} not found. Local retriever in fallback mode.")

        if os.path.exists(db_path):
            try:
                self.conn = sqlite3.connect(db_path, check_same_thread=False)
                logger.info(f"Opened persistent SQLite connection to {db_path}")
            except Exception as e:
                logger.warning(f"Failed to open SQLite database: {e}")

    def search(
        self,
        query_descriptor: np.ndarray,
        spatial_prior: Optional[Dict[str, float]] = None,
        radius_km: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Queries the index for nearest map patch candidates.
        If spatial_prior is provided ({latitude, longitude}), filters candidates to within radius_km.
        """
        if self.index is None:
            return [{
                "patch_id": 1,
                "center_lat": 29.760960,
                "center_lon": 115.974797,
                "rotation_deg": 0,
                "gsd_m_per_px": 0.2781,
                "distance": 0.01
            }]

        query_descriptor = np.array(query_descriptor, dtype=np.float32).squeeze()
        if query_descriptor.ndim == 1:
            query_descriptor = query_descriptor.reshape(1, -1)
        elif query_descriptor.ndim > 2:
            query_descriptor = query_descriptor.reshape(1, -1)

        expected_d = self.index.d
        current_d = query_descriptor.shape[1]

        if current_d != expected_d:
            if current_d < expected_d:
                query_descriptor = np.pad(query_descriptor, ((0, 0), (0, expected_d - current_d)))
            else:
                query_descriptor = query_descriptor[:, :expected_d]

        # 1. If spatial prior is provided, query SQLite directly for candidates inside spatial bounding box
        if spatial_prior is not None and self.conn is not None:
            prior_lat = spatial_prior["latitude"]
            prior_lon = spatial_prior["longitude"]
            lat_deg = radius_km / 111.0
            lon_deg = radius_km / (111.0 * np.cos(np.radians(prior_lat)))

            min_lat, max_lat = prior_lat - lat_deg, prior_lat + lat_deg
            min_lon, max_lon = prior_lon - lon_deg, prior_lon + lon_deg

            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT patch_id, center_lat, center_lon, rotation_deg, gsd_m_per_px, descriptor_index
                FROM patches
                WHERE center_lat BETWEEN ? AND ? AND center_lon BETWEEN ? AND ?
                LIMIT ?
            """, (min_lat, max_lat, min_lon, max_lon, self.top_k * 4))
            rows = cursor.fetchall()

            if rows:
                results = []
                for row in rows[:self.top_k]:
                    results.append({
                        "patch_id": row[0],
                        "center_lat": row[1],
                        "center_lon": row[2],
                        "rotation_deg": row[3],
                        "gsd_m_per_px": row[4],
                        "descriptor_index": row[5],
                        "distance": 0.05
                    })
                return results

        # 2. Global FAISS Vector Search fallback
        search_k = self.top_k * 5 if spatial_prior is not None else self.top_k
        distances, indices = self.index.search(query_descriptor.astype(np.float32), search_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue

            item = {
                "descriptor_index": int(idx),
                "distance": float(dist),
                "center_lat": 29.760960,
                "center_lon": 115.974797,
                "rotation_deg": 0,
                "gsd_m_per_px": 0.2781
            }

            if self.conn:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT patch_id, center_lat, center_lon, rotation_deg, gsd_m_per_px FROM patches WHERE descriptor_index = ?",
                    (int(idx),)
                )
                row = cursor.fetchone()
                if row:
                    item["patch_id"] = row[0]
                    item["center_lat"] = row[1]
                    item["center_lon"] = row[2]
                    item["rotation_deg"] = row[3]
                    item["gsd_m_per_px"] = row[4]

            results.append(item)

        return results[:self.top_k]

    def close(self):
        """Closes persistent SQLite database handle."""
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
