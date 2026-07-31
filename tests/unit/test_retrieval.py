"""
Unit tests for LocalFaissRetriever module.
"""

import os
import sqlite3
import pytest
import numpy as np
from src.retrieval.local_faiss import LocalFaissRetriever


def test_retriever_nonexistent_files():
    retriever = LocalFaissRetriever(index_path="data/nonexistent.faiss", db_path="data/nonexistent.sqlite")
    query_vec = np.random.randn(256).astype(np.float32)
    results = retriever.search(query_vec)
    assert isinstance(results, list)
    assert len(results) == 0
    retriever.close()


def test_retriever_sqlite_bounding_box(tmp_path):
    db_path = str(tmp_path / "test_map.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE patches (
            patch_id INTEGER PRIMARY KEY,
            tile_x INTEGER, tile_y INTEGER, zoom INTEGER,
            rotation_deg INTEGER, center_lat REAL, center_lon REAL,
            gsd_m_per_px REAL, descriptor_index INTEGER
        )
    """)
    cursor.execute("INSERT INTO patches VALUES (1, 10, 20, 18, 0, 29.7609, 115.9747, 0.2781, 0)")
    cursor.execute("INSERT INTO patches VALUES (2, 10, 21, 18, 0, 29.7700, 115.9800, 0.2781, 1)")
    conn.commit()
    conn.close()

    retriever = LocalFaissRetriever(index_path="data/nonexistent.faiss", db_path=db_path, top_k=5)
    query_vec = np.random.randn(256).astype(np.float32)
    spatial_prior = {"latitude": 29.7610, "longitude": 115.9750}

    results = retriever.search(query_vec, spatial_prior=spatial_prior, radius_km=5.0)
    assert len(results) >= 1
    assert results[0]["patch_id"] == 1
    retriever.close()
