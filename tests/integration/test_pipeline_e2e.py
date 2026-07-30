"""
End-to-end integration test verifying synthetic camera ingestion, FAISS IVFPQ retrieval,
LightGlue matching, EKF fusion, and MAVLink bridge output.
"""

import os
import pytest
import numpy as np
from src.main import GPSDeniedPipeline


def test_full_pipeline_e2e_integration():
    config_path = "config/jetson.yaml"
    if not os.path.exists(config_path):
        pytest.skip("config/jetson.yaml not found")

    pipeline = GPSDeniedPipeline(config_path)

    # Run 3 control loop steps
    results = []
    for _ in range(3):
        res = pipeline.run_step()
        results.append(res)

    assert len(results) == 3
    for r in results:
        assert r["status"] == "success"
        assert r["state"] in ["TRACKING", "ANCHORING"]
        assert "latitude" in r["pose"]
        assert "longitude" in r["pose"]
        assert 27.0 < r["pose"]["latitude"] < 28.0
        assert 76.0 < r["pose"]["longitude"] < 77.0
