"""
Physical CSI/USB camera frame capture interface using V4L2 / OpenCV.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class V4L2Camera:
    """Reads frames from physical CSI or USB camera hardware."""
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(device_id)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads frame from camera."""
        if not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        """Releases OpenCV video capture device."""
        if self.cap.isOpened():
            self.cap.release()
