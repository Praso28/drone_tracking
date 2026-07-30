# GPS-Denied Visual Navigation & Local Geopose Engine

> High-precision visual localization and state estimation engine for UAV flights in GPS-denied environments.
> Matches live downward drone video against georeferenced satellite map databases using SuperPoint ONNX, FAISS Product Quantization (IVFPQ), LightGlue RANSAC Homography, and EKF Sensor Fusion.

---

## 🏗️ Architecture & Component Separation

```
gazeebo_drone/
├── config/                   # Configuration specs (YAML)
│   ├── jetson.yaml           # Edge node configuration (Jetson Orin Nano)
│   ├── sim.yaml              # Simulation configuration
│   └── uav_visloc.yaml       # Authentic UAV-VisLoc dataset spec
│
├── src/                      # Production Engine Source Code
│   ├── camera/               # Video ingestion (ZMQ Receiver, Sim Camera)
│   ├── inference/            # Neural inference (SuperPoint ONNX, LightGlue Matcher)
│   ├── retrieval/            # Map database search (FAISS IVFPQ, SQLite Georef)
│   ├── fusion/               # State estimation (EKF IMU Fusion, PoseSmoother)
│   ├── comms/                # Hardware telemetry output (PyMAVLink Bridge)
│   └── main.py               # Edge state machine control loop
│
├── shared/                   # Shared Protocol & Geospatial Utilities
│   ├── geo/                  # WGS84 tile math, GSD scaling, pose homography
│   ├── protocol/             # Binary ZMQ telemetry packer/unpacker
│   └── logging_cfg.py        # Structured logging
│
├── offline_tools/            # PC Ground Station Tools
│   ├── dataset/              # UAV-VisLoc compiler & telemetry streamer
│   ├── map/                  # Satellite patch compiler
│   ├── indexer/              # FAISS IVFPQ index trainer
│   └── sim/                  # Simulation streamer
│
├── tools/                    # Benchmark & Diagnostic Tools
│   ├── evaluate_cep_metrics.py
│   ├── test_cross_provider_robustness.py
│   └── test_drone_vs_satellite.py
│
├── tests/                    # Automated Test Suite (Pytest)
│   ├── unit/
│   └── integration/
└── README.md
```

---

## ⚡ Quick Start

### 1. Ground PC (Streamer / Compiler)

To stream authentic UAV-VisLoc real drone video and flight telemetry:

```bash
python3 offline_tools/dataset/uav_visloc_streamer.py --sequence 01 --fps 10
```

### 2. Edge Node (Jetson Orin Nano)

To execute edge navigation against PC stream:

```bash
python3 -m src.main --config config/jetson.yaml --mode zmq --pc-host 10.1.1.13 --steps 50
```

### 3. Automated Test Suite

To run unit & integration tests:

```bash
python3 -m pytest tests/ -v
```

---

## 📊 Verification & Accuracy Benchmarks

- **FAISS Search Latency**: `< 1.0 ms`
- **Feature Consensus**: `150 – 295 Inliers / Frame`
- **Memory Footprint**: `< 40 MB Active RAM`
- **Test Suite Status**: `21/21 Passed`
