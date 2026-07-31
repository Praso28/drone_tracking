# Operational Runbook: GPS-Denied Visual Navigation System

This document outlines the complete operational procedures for map database compilation, PC-to-Jetson data synchronization, telemetry streaming, and edge flight execution.

---

## 1. System Topology

```
+-------------------------------------------------------+
|                 PC Ground Station                     |
|  • UAV-VisLoc Dataset / Raw Satellite GeoTIFF          |
|  • Map Compiler (build_map.sh)                        |
|  • Telemetry Streamer (uav_visloc_streamer.py)       |
+---------------------------+---------------------------+
                            |
           ZMQ Stream (tcp://10.1.1.13:5555)
           [JPEG Frame + Telemetry Packet]
                            |
                            v
+-------------------------------------------------------+
|             Jetson Orin Nano (Edge Node)              |
|  • Repository Location: ~/drone_tracking/              |
|  • Edge Engine: src/main.py                           |
|  • Local Map Data: data/map_index.faiss, map_db.sqlite|
+---------------------------+---------------------------+
                            |
                 MAVLink (udp:127.0.0.1:14540)
                            v
+-------------------------------------------------------+
|            PX4 / ArduPilot Flight Controller           |
+-------------------------------------------------------+
```

---

## 2. Satellite Map Compilation Workflow (PC Ground Station)

Before running visual navigation on a new flight area or dataset sequence, compile the satellite map database on the PC Ground Station:

### Command:
```bash
# Compile map for Sequence 01
bash scripts/build_map.sh 01 /path/to/UAV_VisLoc_dataset
```

### Artifacts Generated in `data/`:
1. `data/map_db.sqlite`: SQLite database containing patch georeference coordinates, scale, and rotations.
2. `data/vlad_descriptors.npy`: Feature descriptor matrix.
3. `data/map_index.faiss`: Product Quantized FAISS vector index file.

---

## 3. Data Transfer & Jetson Synchronization

To transfer compiled map database files and updated codebase to the Jetson Orin Nano:

### Automated Sync:
```bash
# Set target Jetson IP and sync
JETSON_IP="10.1.1.75" bash scripts/sync_to_jetson.sh
```

### Manual SCP (Map Database Only):
```bash
scp data/map_index.faiss data/map_db.sqlite jetson@<JETSON_IP>:~/drone_tracking/data/
```

---

## 4. Execution Sequence

### Phase 1: Start Telemetry Streamer on PC Ground Station
```bash
python3 offline_tools/dataset/uav_visloc_streamer.py \
    --sequence 01 \
    --dataset-root /path/to/UAV_VisLoc_dataset \
    --fps 30
```

### Phase 2: Start Edge Engine on Jetson Orin Nano
Log into the Jetson Orin Nano terminal and execute:
```bash
cd ~/drone_tracking
python3 -m src.main --config config/jetson.yaml --mode zmq --pc-host 10.1.1.13 --steps 50
```

---

## 5. Automated Testing & Accuracy Benchmark

### Run Unit & Integration Tests:
```bash
python3 -m pytest tests/ -v
```

### Compute Trajectory Accuracy Metrics (CEP-50, CEP-95, RMSE):
```bash
python3 tools/evaluate_cep_metrics.py
```
