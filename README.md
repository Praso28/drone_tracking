# GPS-Denied Visual Navigation & Local Geopose Engine

High-precision visual localization and state estimation engine for UAV flight in GPS-denied environments. Matches live downward camera frames against pre-compiled georeferenced satellite map databases using SuperPoint ONNX, FAISS Product Quantization (IVFPQ), LightGlue RANSAC Homography, and Extended Kalman Filter (EKF) state estimation.

---

## Architecture & Directory Structure

```
drone_tracking/
├── config/                   # System configuration specs
│   ├── jetson.yaml           # Edge node runtime configuration (Jetson Orin Nano)
│   ├── sim.yaml              # Simulation configuration
│   └── uav_visloc.yaml       # UAV-VisLoc dataset spec
│
├── docs/                     # Operations & Hardware Documentation
│   ├── OPERATIONS.md         # Operational runbook & data sync procedures
│   └── HARDWARE.md           # Jetson hardware specs & MAVLink setup
│
├── src/                      # Production Edge Navigation Engine
│   ├── camera/               # Video frame ingestion (ZMQ Receiver, Sim Camera)
│   ├── inference/            # Feature extraction (SuperPoint ONNX, TensorRT)
│   ├── retrieval/            # Vector map retrieval (FAISS IVFPQ, SQLite Georef)
│   ├── fusion/               # EKF state estimation & health monitoring
│   ├── comms/                # MAVLink telemetry bridge (VISION_POSITION_ESTIMATE)
│   └── main.py               # Edge state machine control loop
│
├── shared/                   # Shared Protocol & Geospatial Utilities
│   ├── geo/                  # WGS84 tile math, GSD scaling, pose homography
│   ├── protocol/             # Binary ZMQ telemetry packet definitions
│   └── logging_cfg.py        # Structured logging configuration
│
├── offline_tools/            # PC Ground Station Pipeline Tools
│   ├── dataset/              # UAV-VisLoc compiler & telemetry streamer
│   ├── indexer/              # FAISS IVFPQ vector index trainer
│   └── map/                  # Satellite patch compiler
│
├── scripts/                  # Operational Utility Scripts
│   ├── build_map.sh          # Two-step satellite map compiler script
│   ├── sync_to_jetson.sh     # Codebase & map database synchronization script
│   ├── setup_jetson.sh       # Edge node environment bootstrapper
│   └── export_trt_engines.sh # TensorRT FP16 model exporter
│
├── tests/                    # Automated Test Suite (Pytest)
│   ├── unit/                 # Unit tests for geometry, fusion, camera, map tools
│   └── integration/          # End-to-end multi-phase state machine test
└── README.md
```

---

## Operational Workflow

### 1. Build Satellite Map Database (PC Ground Station)

Compile the satellite map database and train the compressed FAISS index for a target flight sequence:

```bash
bash scripts/build_map.sh 01 /path/to/UAV_VisLoc_dataset
```

This generates `data/map_db.sqlite` and `data/map_index.faiss`.

### 2. Synchronize Map Data to Jetson Orin Nano

Transfer the codebase and compiled map database to the Jetson edge node:

```bash
JETSON_IP="10.1.1.75" bash scripts/sync_to_jetson.sh
```

### 3. Start Telemetry Streamer (PC Ground Station)

Stream real drone video frames and authentic flight telemetry over ZMQ:

```bash
python3 offline_tools/dataset/uav_visloc_streamer.py --sequence 01 --fps 30
```

### 4. Execute Edge Engine (Jetson Orin Nano)

Run real-time edge navigation on the Jetson Orin Nano:

```bash
python3 -m src.main --config config/jetson.yaml --mode zmq --pc-host 10.1.1.13 --steps 50
```

---

## Verification & Automated Test Suite

Run the automated test suite:

```bash
python3 -m pytest tests/ -v
```

Evaluate horizontal positioning error metrics (CEP-50, CEP-95, RMSE):

```bash
python3 tools/evaluate_cep_metrics.py
```

---

## Documentation Links

* [Operational Runbook](docs/OPERATIONS.md)
* [Hardware & MAVLink Specifications](docs/HARDWARE.md)
* [Contributor Guidelines](CONTRIBUTING.md)
* [Software License](LICENSE)
