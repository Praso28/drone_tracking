"""
Unit tests for camera stream interfaces.
"""

import pytest
import numpy as np
from src.camera.sim_camera import SimCamera


def test_sim_camera_read():
    cam = SimCamera(width=320, height=240, fps=100)
    success, frame = cam.read()
    assert success is True
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    cam.release()
