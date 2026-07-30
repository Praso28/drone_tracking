"""
End-to-end integration tests for full GPS-Denied Visual Navigation Pipeline.
"""

import os
import pytest
from src.main import GPSDeniedPipeline


def test_full_pipeline_e2e_integration():
    config_path = "config/jetson.yaml"
    if not os.path.exists(config_path):
        pytest.skip("config/jetson.yaml not found")

    # Initialize pipeline in standalone simulation mode for unit testing
    pipeline = GPSDeniedPipeline(config_path, mode="sim")

    # Run 3 control loop steps
    results = []
    try:
        for _ in range(3):
            res = pipeline.run_step()
            results.append(res)
    finally:
        pipeline.close()

    assert len(results) == 3
    for r in results:
        assert r["status"] == "success"
        assert r["state"] in ["HIGH_CONFIDENCE_TRACKING", "UNANCHORED_ACQUISITION", "TRACKING"]
        assert "latitude" in r["pose"]
        assert "longitude" in r["pose"]
