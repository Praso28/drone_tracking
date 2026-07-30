#!/usr/bin/env bash
# Synchronizes Jetson Edge Node runtime codebase directly into ~/Desktop/gazeebo_drone
# Clean separation: PC Ground Station (offline_tools, tests, dataset) stays on PC.

JETSON_IP="10.1.1.75"
JETSON_USER="jetson"

echo "=== Syncing Edge Runtime (src, shared, config, scripts) to Jetson (${JETSON_USER}@${JETSON_IP}:Desktop/gazeebo_drone/) ==="

rsync -avz --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/tiles' \
    --exclude='data/*.png' \
    --exclude='*.JPG' \
    src config shared scripts README.md \
    ${JETSON_USER}@${JETSON_IP}:Desktop/gazeebo_drone/

echo "=== Syncing FAISS index, SQLite georef DB, and satellite map texture ==="
scp data/ajabgarh_ivfpq.index data/georef.sqlite data/satellite01.jpg ${JETSON_USER}@${JETSON_IP}:Desktop/gazeebo_drone/data/

echo "=== Edge Node Sync Complete! ==="
