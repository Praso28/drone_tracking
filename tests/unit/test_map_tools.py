"""
Unit tests for offline map compiler and IVFPQ index trainer.
"""

import os
import pytest
import numpy as np
from offline_tools.map.map_compiler import extract_synthetic_descriptor
from offline_tools.indexer.vlad_encoder import VladEncoder
from offline_tools.indexer.dino_extractor import DinoFeatureExtractor
from offline_tools.indexer.train_ivfpq import train_and_save_ivfpq_index


def test_extract_synthetic_descriptor():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    desc = extract_synthetic_descriptor(img, dim=384)
    assert desc.shape == (384,)
    assert abs(np.linalg.norm(desc) - 1.0) < 1e-4


def test_vlad_encoder():
    encoder = VladEncoder(num_clusters=8, feature_dim=128)
    local_feats = np.random.randn(50, 128).astype(np.float32)
    vlad_vec = encoder.encode(local_feats)
    assert vlad_vec.shape == (8 * 128,)
    assert abs(np.linalg.norm(vlad_vec) - 1.0) < 1e-4


def test_train_ivfpq_index(tmp_path):
    desc_path = str(tmp_path / "test_desc.npy")
    index_path = str(tmp_path / "test_ivfpq.index")

    # Generate 300 synthetic descriptors
    desc_matrix = np.random.randn(300, 384).astype(np.float32)
    norms = np.linalg.norm(desc_matrix, axis=1, keepdims=True)
    desc_matrix /= norms
    np.save(desc_path, desc_matrix)

    success = train_and_save_ivfpq_index(
        descriptors_path=desc_path,
        output_index_path=index_path,
        nlist=10,
        m=16,
        nbits=8,
        verify=True
    )
    assert success is True
    assert os.path.exists(index_path)
