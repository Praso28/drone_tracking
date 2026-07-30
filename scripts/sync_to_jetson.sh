#!/usr/bin/env bash
# Synchronizes clean codebase and dataset index to Jetson Orin Nano edge node
JETSON_IP="10.1.1.75"
JETSON_USER="jetson"
TARGET_DIR="~/Desktop/gazeebo_drone"

echo "=== Syncing gazeebo_drone codebase to Jetson (${JETSON_USER}@${JETSON_IP}) ==="

rsync -avz --exclude='.git' --exclude='data/tiles' --exclude='data/*.png' --exclude='*.JPG' \
    src/ config/ shared/ offline_tools/ tools/ tests/ scripts/ README.md \
    ${JETSON_USER}@${JETSON_IP}:${TARGET_DIR}/

echo "=== Syncing FAISS index and georef database ==="
scp data/ajabgarh_ivfpq.index data/georef.sqlite ${JETSON_USER}@${JETSON_IP}:${TARGET_DIR}/data/

echo "=== Sync Complete! ==="
