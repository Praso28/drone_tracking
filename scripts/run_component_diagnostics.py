"""
Comprehensive Per-Component Diagnostic & Health Audit Script.
Evaluates every core subsystem independently and outputs a machine-readable health report.
"""

import os
import sys
import json
import time
import numpy as np

# Ensure root directory is in sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from shared.logging_cfg import setup_logger

logger = setup_logger("diagnostics")


def audit_camera_subsystem() -> dict:
    """Audit Camera and ZMQ receiver modules."""
    try:
        from src.camera.sim_camera import SimCamera
        cam = SimCamera(width=320, height=240, fps=30)
        success, frame, telemetry = cam.read()
        cam.release()

        if success and frame is not None and frame.shape == (240, 320, 3):
            return {"status": "PASS", "details": f"SimCamera frame shape: {frame.shape}, telemetry present: {telemetry is not None}"}
        return {"status": "FAIL", "details": "SimCamera returned invalid frame or failure code"}
    except Exception as e:
        return {"status": "FAIL", "details": str(e)}


def audit_inference_subsystem() -> dict:
    """Audit SuperPointEngine and LightGlueMatcher modules."""
    try:
        from src.inference.trt_engine import SuperPointEngine
        from src.inference.lightglue_matcher import LightGlueMatcher

        sp = SuperPointEngine(descriptor_dim=256)
        matcher = LightGlueMatcher()

        img0 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        feats0 = sp.extract(img0)
        feats1 = sp.extract(img1)

        inliers, M = matcher.match(feats0, feats1)
        if getattr(sp, "trt_context", None) is not None:
            backend = "TensorRT FP16"
        elif sp.ort_session is not None:
            backend = "ONNX"
        else:
            backend = "FAST/Gradient CPU Fallback"

        return {
            "status": "PASS",
            "details": f"Backend: [{backend}], Extracted kps: ({len(feats0['keypoints'])}, {len(feats1['keypoints'])}), Match inliers: {inliers}"
        }
    except Exception as e:
        return {"status": "FAIL", "details": str(e)}


def audit_retrieval_subsystem() -> dict:
    """Audit LocalFaissRetriever module."""
    try:
        from src.retrieval.local_faiss import LocalFaissRetriever

        retriever = LocalFaissRetriever(index_path="data/map_index.faiss", db_path="data/map_db.sqlite")
        query_vec = np.random.randn(256).astype(np.float32)
        spatial_prior = {"latitude": 29.760960, "longitude": 115.974797}
        results = retriever.search(query_vec, spatial_prior=spatial_prior, radius_km=5.0)
        retriever.close()

        index_status = "Loaded" if os.path.exists("data/map_index.faiss") else "Not found (Fallback active)"
        db_status = "Loaded" if os.path.exists("data/map_db.sqlite") else "Not found"

        return {
            "status": "PASS",
            "details": f"FAISS Index: [{index_status}], SQLite DB: [{db_status}], Search returned {len(results)} candidates"
        }
    except Exception as e:
        return {"status": "FAIL", "details": str(e)}


def audit_fusion_subsystem() -> dict:
    """Audit EKF, PoseSmoother, and NavigationPhaseController modules."""
    try:
        from src.fusion.ekf_node import SimpleEKFFusion, PoseSmoother
        from src.fusion.health_monitor import NavigationPhaseController

        ekf = SimpleEKFFusion(29.760960, 115.974797)
        ekf.predict_imu(0.1, 0.0, 9.81, dt=0.033)
        updated = ekf.update_visual_fix({"latitude": 29.7610, "longitude": 115.9748, "heading_deg": 15.0})

        smoother = PoseSmoother(window_size=5, max_distance_m=100.0)
        smooth_res = smoother.filter({"latitude": 29.7610, "longitude": 115.9748})

        ctrl = NavigationPhaseController(anchor_inliers_thresh=15, tracking_min_inliers=10)
        phase_res = ctrl.evaluate_step(inliers=20, pose_accepted=True, raw_pose={"latitude": 29.7610, "longitude": 115.9748})

        return {
            "status": "PASS",
            "details": f"EKF Pose: ({updated['latitude']:.6f}, {updated['longitude']:.6f}), Smoother Accepted: {smooth_res['accepted']}, Phase: {phase_res['phase']}"
        }
    except Exception as e:
        return {"status": "FAIL", "details": str(e)}


def audit_comms_subsystem() -> dict:
    """Audit MAVLink bridge and TelemetryFrame protocol modules."""
    try:
        from src.comms.mavlink_bridge import MAVLinkBridge
        from shared.protocol.telemetry_frame import TelemetryFrame

        bridge = MAVLinkBridge(connection_str="udp:127.0.0.1:14540")
        sent = bridge.send_vision_position_estimate({"latitude": 29.760960, "longitude": 115.974797, "heading_deg": 0.0})
        bridge.close()

        frame_obj = TelemetryFrame(1, time.time(), 29.76, 115.97, 100.0, 0.0, 5.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
        packed = frame_obj.pack_payload(b"fake_jpeg")
        unpacked_obj, unpacked_bytes = TelemetryFrame.unpack_payload(packed)

        if sent and unpacked_obj.frame_id == 1 and unpacked_bytes == b"fake_jpeg":
            return {"status": "PASS", "details": "MAVLink UDP broadcast OK, TelemetryFrame pack/unpack payload OK"}
        return {"status": "WARN", "details": "TelemetryFrame pack/unpack OK, MAVLink packet sending unconfirmed"}
    except Exception as e:
        return {"status": "FAIL", "details": str(e)}


def audit_pipeline_e2e() -> dict:
    """Audit full end-to-end main control pipeline loop."""
    try:
        from src.main import GPSDeniedPipeline
        pipeline = GPSDeniedPipeline("config/jetson.yaml", mode="sim")
        res = pipeline.run_step()
        pipeline.close()

        if res.get("status") == "success":
            return {"status": "PASS", "details": f"Pipeline step executed successfully: Phase [{res['phase']}], Inliers: {res['inliers']}"}
        return {"status": "FAIL", "details": f"Pipeline step failed: {res.get('message')}"}
    except Exception as e:
        return {"status": "FAIL", "details": str(e)}


def run_all_diagnostics():
    print("=" * 70)
    print("  GPS-DENIED VISUAL NAVIGATION SYSTEM — COMPONENT HEALTH AUDIT")
    print("=" * 70)

    subsystems = {
        "Camera & Streamer": audit_camera_subsystem,
        "Inference & Feature Matching": audit_inference_subsystem,
        "Map Database & FAISS Retrieval": audit_retrieval_subsystem,
        "EKF Fusion & Phase Controller": audit_fusion_subsystem,
        "Telemetry & MAVLink Comms": audit_comms_subsystem,
        "End-to-End Pipeline Loop": audit_pipeline_e2e,
    }

    report = {}
    all_pass = True

    for name, audit_fn in subsystems.items():
        res = audit_fn()
        report[name] = res
        status_color = "\033[92m[PASS]\033[0m" if res["status"] == "PASS" else ("\033[93m[WARN]\033[0m" if res["status"] == "WARN" else "\033[91m[FAIL]\033[0m")
        print(f" {status_color} {name:<32} | {res['details']}")
        if res["status"] == "FAIL":
            all_pass = False

    os.makedirs("data", exist_ok=True)
    with open("data/component_health.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print(f" Diagnostic JSON written to data/component_health.json")
    print(f" OVERALL SYSTEM AUDIT STATUS: {'SUCCESS - READY FOR TESTING' if all_pass else 'FAILURES DETECTED'}")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_all_diagnostics())
