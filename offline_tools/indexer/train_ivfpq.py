"""
Offline FAISS IVFPQ Trainer (PC Ground Station).
Compresses heavy FlatL2 descriptor matrices (2GB+) into lightweight Product Quantized
IVFPQ index files (~60MB target) for Jetson edge deployment.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import numpy as np
import faiss
from shared.logging_cfg import setup_logger

logger = setup_logger("train_ivfpq")


def train_and_save_ivfpq_index(
    descriptors_path: str,
    output_index_path: str,
    nlist: int = 100,
    m: int = 16,
    nbits: int = 8,
    verify: bool = True
) -> bool:
    """Trains an IndexIVFPQ index from a matrix of descriptors and saves it to disk."""
    if not os.path.exists(descriptors_path):
        logger.error(f"Descriptors file not found: {descriptors_path}")
        return False

    descriptors = np.load(descriptors_path).astype(np.float32)
    N, D = descriptors.shape
    logger.info(f"Loaded descriptor matrix: {N} vectors of dimension {D}.")

    # Ensure D is divisible by m
    if D % m != 0:
        logger.warning(f"Dimension {D} is not divisible by m={m}. Padding dimension.")
        padded_D = int(np.ceil(D / m) * m)
        padding = np.zeros((N, padded_D - D), dtype=np.float32)
        descriptors = np.hstack([descriptors, padding])
        D = padded_D

    # Ensure nbits is compatible with number of training points N
    max_bits = max(1, int(np.floor(np.log2(max(1, N)))))
    nbits_actual = min(nbits, max_bits)

    nlist_actual = min(nlist, max(1, N // 4))
    quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFPQ(quantizer, D, nlist_actual, m, nbits_actual)

    logger.info(f"Training IndexIVFPQ (D={D}, nlist={nlist_actual}, m={m}, nbits={nbits_actual})...")
    index.train(descriptors)
    index.add(descriptors)
    logger.info(f"Successfully added {index.ntotal} vectors to index.")

    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    faiss.write_index(index, output_index_path)
    file_size_mb = os.path.getsize(output_index_path) / (1024 * 1024)
    logger.info(f"Saved compressed index to {output_index_path} ({file_size_mb:.2f} MB).")

    if verify:
        logger.info("Verifying index load and query...")
        loaded_index = faiss.read_index(output_index_path)
        loaded_index.nprobe = min(10, nlist_actual)
        query_vec = descriptors[:1]
        distances, indices = loaded_index.search(query_vec, k=5)
        logger.info(f"Query verification passed. Top-1 index: {indices[0][0]}, distance: {distances[0][0]:.4f}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train compressed FAISS IVFPQ index.")
    parser.add_argument("--descriptors", type=str, default="data/vlad_descriptors.npy", help="Input descriptors .npy")
    parser.add_argument("--output", type=str, default="data/map_index.faiss", help="Output .faiss index file")
    parser.add_argument("--nlist", type=int, default=100, help="Number of Voronoi cells")
    parser.add_argument("--m", type=int, default=16, help="Number of sub-quantizers")
    parser.add_argument("--nbits", type=int, default=8, help="Bits per sub-quantizer")
    parser.add_argument("--verify", action="store_true", help="Verify index search after saving")
    args = parser.parse_args()

    train_and_save_ivfpq_index(
        descriptors_path=args.descriptors,
        output_index_path=args.output,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        verify=args.verify
    )
