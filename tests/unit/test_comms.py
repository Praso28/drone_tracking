"""
Unit tests for MAVLinkBridge module.
"""

import pytest
from src.comms.mavlink_bridge import MAVLinkBridge


def test_mavlink_bridge_init_and_send():
    bridge = MAVLinkBridge(connection_str="udp:127.0.0.1:14555", origin_lat=29.7609, origin_lon=115.9747)
    pose = {"latitude": 29.7610, "longitude": 115.9748, "altitude_m": 100.0, "heading_deg": 90.0}

    success = bridge.send_vision_position_estimate(pose)
    assert success is True
    bridge.close()
