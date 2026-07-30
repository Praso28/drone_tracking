"""
NetVLAD aggregator with intra-normalization.
"""

import numpy as np


class VladEncoder:
    """Encodes local features into a single global NetVLAD descriptor vector."""
    def __init__(self, num_clusters: int = 16, feature_dim: int = 128):
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.centers = np.random.randn(num_clusters, feature_dim).astype(np.float32)

    def encode(self, local_features: np.ndarray) -> np.ndarray:
        """
        Aggregates local descriptors into a NetVLAD vector.
        Applies L2 intra-normalization and global L2 normalization.
        """
        if local_features is None or len(local_features) == 0:
            return np.zeros(self.num_clusters * self.feature_dim, dtype=np.float32)

        N, D = local_features.shape
        dists = np.linalg.norm(local_features[:, None, :] - self.centers[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)

        vlad = np.zeros((self.num_clusters, D), dtype=np.float32)
        for i in range(N):
            c_idx = assignments[i]
            vlad[c_idx] += local_features[i] - self.centers[c_idx]

        # Intra-normalization
        vlad_norms = np.linalg.norm(vlad, axis=1, keepdims=True)
        vlad_norms[vlad_norms < 1e-6] = 1.0
        vlad = vlad / vlad_norms

        # Global normalization
        vlad_vec = vlad.flatten()
        g_norm = np.linalg.norm(vlad_vec)
        if g_norm > 1e-6:
            vlad_vec /= g_norm

        return vlad_vec.astype(np.float32)
