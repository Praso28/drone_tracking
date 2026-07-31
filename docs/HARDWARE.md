# Hardware Architecture & Deployment Specification

This document details hardware requirements, connection interfaces, power management, and autopilot integration for the **NVIDIA Jetson Orin Nano Edge Node**.

---

## 1. Hardware Specifications

| Component | Specification | Operational Note |
|---|---|---|
| **Edge Compute** | NVIDIA Jetson Orin Nano (8GB / 4GB) | Unified LPDDR5 RAM |
| **GPU Compute** | 1024 CUDA Cores, 32 Tensor Cores | FP16 Inference Mode |
| **Flight Controller** | PX4 Autopilot / ArduPilot | Pixhawk 6C / Cube Orange |
| **Optical Sensor** | Downward Nadir Camera (640x480 @ 30 FPS) | 75° FOV |
| **Communication** | Ethernet / Serial UART / ZMQ TCP | MAVLink Protocol |

---

## 2. Power Management (Jetson Orin Nano)

For flight operations, set the Jetson power mode to **15W MAXN** to ensure maximum compute throughput for TensorRT and OpenCV:

```bash
# Enable 15W MAXN Power Profile
sudo nvpmodel -m 0

# Verify active profile
sudo nvpmodel -q
```

---

## 3. Flight Controller MAVLink Interface

The edge navigation engine transmits visual position fixes to PX4/ArduPilot using the `VISION_POSITION_ESTIMATE` (MAVLink message #102) packet.

### Port Connection Options:
1. **UDP Socket (Software-in-the-Loop / Ethernet Bridge):** `udp:127.0.0.1:14540`
2. **Serial UART (Direct Pixhawk TELEM2 Connection):** `/dev/ttyTHS1` at 921600 baud.

### MAVLink Configuration in `config/jetson.yaml`:
```yaml
comms:
  mavlink_connection: "udp:127.0.0.1:14540"
  system_id: 1
  component_id: 191  # MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY
```
