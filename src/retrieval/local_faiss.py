"""
SQLite + FAISS Vector Database Retriever for satellite map patches.
Supports spatial radius filtering, SQL bounding box queries,
and fast persistent handle searching.
"""

import os
import sqlite3
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from shared.geo.tile_math import haversine_distance
from shared.logging_cfg import setup_logger

logger = setup_logger("local_faiss")


class LocalFaissRetriever:
    """Manages FAISS vector index and SQLite metadata for candidate map retrieval."""

    def __init__(
        self,
        index_path: str = "data/ajabgarh_ivfpq.index",
        db_path: str = "data/georef.sqlite",
        top_k: int = 5
    ):
        self.index_path = index_path
        self.db_path = db_path
        self.top_k = top_k
        self.index = None
        self.conn = None

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS IVFPQ index from {index_path} (ntotal={self.index.ntotal}).")
        else:
            logger.warning(f"FAISS index file not found at {index_path}.")

        if os.path.exists(db_path):
            self.conn = sqlite3.connect(db_path)
            logger.info(f"Opened persistent SQLite connection to {db_path}")

    def search(
        self,
        query_vector: np.ndarray,
        spatial_prior: Optional[Dict[str, float]] = None,
        radius_km: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Searches index for top_k candidate map patches.
        When spatial_prior is provided, filters and ranks candidates by geographic distance to EKF position.
        """
        query_descriptor = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query_descriptor.ndim == 1:
            query_descriptor = np.expand_dims(query_descriptor, axis=0)

        # 1. Spatial Prior Radius Search via SQLite Bounding Box Query
        if spatial_prior is not None and self.conn is not None:
            prior_lat = spatial_prior["latitude"]
            prior_lon = spatial_prior["longitude"]
            lat_deg = radius_km / 111.0
            lon_deg = radius_km / (111.0 * max(0.1, np.cos(np.radians(prior_lat))))

            min_lat, max_lat = prior_lat - lat_deg, prior_lat + lat_deg
            min_lon, max_lon = prior_lon - lon_deg, prior_lon + lon_deg

            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT patch_id, center_lat, center_lon, rotation_deg, gsd_m_per_px, descriptor_index
                FROM patches
                WHERE center_lat BETWEEN ? AND ? AND center_lon BETWEEN ? AND ?
                LIMIT ?
            """, (min_lat, max_lat, min_lon, max_lon, self.top_k * 10))

            rows = cursor.fetchall()
            if rows:
                # Rank candidates by spatial Haversine distance to EKF position estimate
                rows_sorted = sorted(
                    rows,
                    key=lambda r: haversine_distance(prior_lat, prior_lon, r[1], r[2])
                )
                results = []
                for r in rows_sorted[:self.top_k]:
                    results.append({
                        "patch_id": r[0],
                        "center_lat": r[1],
                        "center_lon": r[2],
                        "rotation_deg": r[3],
                        "gsd_m_per_px": r[4],
                        "descriptor_index": r[5],
                        "distance": haversine_distance(prior_lat, prior_lon, r[1], r[2])
                    })
                return results

        # 2. Global FAISS Vector Search fallback
        if self.index is None:
            return []

        search_k = self.top_k * 5
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
        """Closes active SQLite connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
