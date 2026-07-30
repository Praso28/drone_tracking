#!/usr/bin/env bash
# Synchronizes Jetson Edge Node runtime codebase, vector index, and satellite texture map.
# Clean separation: PC Ground Station (offline_tools, tests, dataset) stays on PC.
JETSON_IP="10.1.1.75"
JETSON_USER="jetson"
TARGET_DIR="~/Desktop/gazeebo_drone"

echo "=== Syncing Edge Runtime (src, shared, config, scripts) to Jetson (${JETSON_USER}@${JETSON_IP}) ==="

rsync -avz --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/tiles' \
    --exclude='data/*.png' \
    --exclude='*.JPG' \
    src config shared scripts README.md \
    ${JETSON_USER}@${JETSON_IP}:${TARGET_DIR}/

echo "=== Syncing FAISS index, SQLite georef DB, and satellite map texture ==="
scp data/ajabgarh_ivfpq.index data/georef.sqlite data/satellite01.jpg ${JETSON_USER}@${JETSON_IP}:${TARGET_DIR}/data/

echo "=== Edge Node Sync Complete! ==="
