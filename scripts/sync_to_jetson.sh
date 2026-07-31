#!/usr/bin/env bash
# Production Edge Code & Map Data Synchronization Script
# Transfers runtime source code, configurations, FAISS index, and SQLite georeference DB to Jetson Edge Node.

set -e

JETSON_IP="${JETSON_IP:-10.1.1.75}"
JETSON_USER="${JETSON_USER:-jetson}"
TARGET_DIR="${TARGET_DIR:-drone_tracking}"

echo "=================================================================="
echo " Syncing Edge Runtime to Jetson (${JETSON_USER}@${JETSON_IP}:~/${TARGET_DIR}/)"
echo "=================================================================="

# 1. Sync production codebase & configurations
rsync -avz --delete --no-perms \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='data/tiles' \
    --exclude='data/*.png' \
    src config shared scripts README.md LICENSE \
    "${JETSON_USER}@${JETSON_IP}:~/${TARGET_DIR}/"

# 2. Transfer compiled satellite map index & SQLite georef database if present locally
echo "--- Transferring Map Index & SQLite Georef Database ---"
if [ -f "data/map_index.faiss" ] && [ -f "data/map_db.sqlite" ]; then
    scp data/map_index.faiss data/map_db.sqlite "${JETSON_USER}@${JETSON_IP}:~/${TARGET_DIR}/data/"
    echo "[+] Successfully transferred map_index.faiss and map_db.sqlite."
else
    echo "[!] Warning: data/map_index.faiss or data/map_db.sqlite missing locally."
    echo "[!] Run 'bash scripts/build_map.sh' to compile map database before flying."
fi

echo "=================================================================="
echo "[+] Jetson Edge Node Synchronization Complete!"
echo "=================================================================="
