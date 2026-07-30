"""
ZMQ Frame & Telemetry Receiver for Jetson Orin Nano Edge Node.
Subscribes to PC Ground Station ZMQ streamer (tcp://10.1.1.13:5555) and decodes
compressed camera frames alongside real flight telemetry and IMU data.
"""

import cv2
import zmq
import numpy as np
from typing import Tuple, Optional
from shared.logging_cfg import setup_logger
from shared.protocol.telemetry_frame import TelemetryFrame

logger = setup_logger("zmq_receiver")


class ZMQReceiver:
    """ZMQ subscriber client receiving camera frames and sensor telemetry from PC host."""
    def __init__(self, host: str = "10.1.1.13", port: int = 5555, timeout_ms: int = 1000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.endpoint = f"tcp://{host}:{port}"
        self.socket.connect(self.endpoint)
        logger.info(f"Subscribed to ZMQ Frame Streamer at {self.endpoint}")

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[TelemetryFrame]]:
        """
        Receives next packet over ZMQ socket.
        Returns (success, frame_rgb, telemetry_frame).
        """
        try:
            raw_bytes = self.socket.recv()
            telemetry, frame_jpg_bytes = TelemetryFrame.unpack_payload(raw_bytes)

            np_arr = np.frombuffer(frame_jpg_bytes, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                return False, None, telemetry

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return True, frame_rgb, telemetry

        except zmq.Again:
            logger.warning(f"ZMQ recv timeout ({self.timeout_ms}ms) from {self.endpoint}")
            return False, None, None
        except Exception as e:
            logger.error(f"ZMQ frame unpack error: {e}")
            return False, None, None

    def release(self):
        """Closes ZMQ socket connection."""
        self.socket.close()
        self.context.term()
