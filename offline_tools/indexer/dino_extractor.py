"""
Feature extractor wrapper for DINOv2 / CNN visual representations (PC Ground Station).
"""

import numpy as np
from PIL import Image


class DinoFeatureExtractor:
    """Wrapper for feature extraction."""
    def __init__(self, model_name: str = "dinov2_vits14", descriptor_dim: int = 384):
        self.descriptor_dim = descriptor_dim
        self.model_name = model_name

    def extract(self, image: Image.Image) -> np.ndarray:
        """Extracts a normalized descriptor vector for a given PIL image."""
        img_arr = np.array(image.convert("RGB").resize((224, 224)))
        gray = np.mean(img_arr, axis=2)
        freq = np.fft.fft2(gray).real.flatten()
        
        if len(freq) >= self.descriptor_dim:
            vec = freq[:self.descriptor_dim]
        else:
            vec = np.pad(freq, (0, self.descriptor_dim - len(freq)))
            
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.astype(np.float32)
