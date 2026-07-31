#!/usr/bin/env bash
# Production Satellite Map Database Compiler & FAISS Indexer
# Step 1: Extracts satellite map patches & georef database from UAV-VisLoc dataset.
# Step 2: Trains compressed FAISS IVFPQ index file.

set -e

SEQUENCE_ID="${1:-01}"
DATASET_ROOT="${2:-${UAV_VISLOC_ROOT:-data/uav_visloc}}"
OUTPUT_DB="data/map_db.sqlite"
OUTPUT_INDEX="data/map_index.faiss"
DESCRIPTOR_NPY="data/vlad_descriptors.npy"

echo "=================================================================="
echo " Building Satellite Map Database for Sequence ${SEQUENCE_ID}"
echo " Dataset Root: ${DATASET_ROOT}"
echo "=================================================================="

# Step 1: Compile satellite patches & SQLite database
echo "[Step 1/2] Compiling satellite map patches..."
python3 offline_tools/dataset/uav_visloc_compiler.py \
    --sequence "${SEQUENCE_ID}" \
    --dataset-root "${DATASET_ROOT}"

# Step 2: Train FAISS index
echo "[Step 2/2] Training FAISS IVFPQ index..."
python3 offline_tools/indexer/train_ivfpq.py \
    --descriptors "${DESCRIPTOR_NPY}" \
    --output "${OUTPUT_INDEX}" \
    --verify

echo "=================================================================="
echo "[+] Satellite Map Build Complete!"
echo "    - SQLite Database: ${OUTPUT_DB}"
echo "    - FAISS Index:     ${OUTPUT_INDEX}"
echo "=================================================================="
