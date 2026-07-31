"""
MAVLink bridge for sending visual position fixes to PX4 / ArduPilot flight controllers.
Constructs and transmits VISION_POSITION_ESTIMATE (MAVLink #102) packets over UDP.
"""

import time
import socket
import numpy as np
from typing import Dict, Any
from shared.logging_cfg import setup_logger

logger = setup_logger("mavlink_bridge")


class MAVLinkBridge:
    """Interface to flight controller via PyMAVLink / UDP socket."""

    def __init__(
        self,
        connection_str: str = "udp:127.0.0.1:14540",
        origin_lat: float = 27.2000,
        origin_lon: float = 76.2350
    ):
        self.connection_str = connection_str
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.mav = None
        self.udp_sock = None
        self.target_ip = "127.0.0.1"
        self.target_port = 14540

        if connection_str.startswith("udp:"):
            parts = connection_str.split(":")
            if len(parts) >= 3:
                self.target_ip = parts[1]
                self.target_port = int(parts[2])

        try:
            from pymavlink import mavutil
            self.mav = mavutil.mavlink_connection(connection_str, source_system=1, source_component=191)
            logger.info(f"Initialized PyMAVLink connection endpoint: {connection_str} (Origin: {self.origin_lat}, {self.origin_lon})")
        except Exception as e:
            logger.warning(f"PyMAVLink init failed ({e}). Initializing raw UDP socket fallback to {self.target_ip}:{self.target_port}.")
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_vision_position_estimate(
        self,
        pose: Dict[str, float],
        covariance_std_m: float = 2.0
    ) -> bool:
        """
        Sends VISION_POSITION_ESTIMATE MAVLink message to flight controller.
        """
        usec = int(time.time() * 1e6)
        lat = pose["latitude"]
        lon = pose["longitude"]
        alt_m = pose.get("altitude_m", 100.0)
        roll = 0.0
        pitch = 0.0
        yaw_rad = float(np.radians(pose.get("heading_deg", 0.0))) if "heading_deg" in pose else 0.0

        if self.mav is not None:
            try:
                # Convert lat/lon to local ENU approximation relative to origin
                x_m = float((lat - self.origin_lat) * 111000.0)
                y_m = float((lon - self.origin_lon) * 111000.0 * np.cos(np.radians(self.origin_lat)))
                z_m = float(-alt_m)

                cov = [covariance_std_m**2] + [0.0] * 20
                self.mav.mav.vision_position_estimate_send(
                    usec, x_m, y_m, z_m, roll, pitch, yaw_rad, cov
                )
                logger.debug(f"PyMAVLink VISION_POSITION_ESTIMATE sent: ({x_m:.2f}m, {y_m:.2f}m, {z_m:.2f}m)")
                return True
            except Exception as e:
                logger.error(f"Failed to send MAVLink message: {e}")
                return False

        if self.udp_sock is not None:
            try:
                msg = f"VISION_POSE:{usec}:{lat:.6f}:{lon:.6f}:{alt_m:.1f}\n".encode("utf-8")
                self.udp_sock.sendto(msg, (self.target_ip, self.target_port))
                logger.debug(f"UDP fallback pose sent -> {msg.decode().strip()}")
                return True
            except Exception as e:
                logger.error(f"UDP fallback error: {e}")
                return False

        return False

    def close(self):
        """Closes MAVLink / UDP socket resources."""
        if self.udp_sock is not None:
            try:
                self.udp_sock.close()
            except Exception:
                pass
            self.udp_sock = None
