"""
Report Visualization Script.
Generates static visualization images for the GPS-Denied Navigation Technical Report.
Each image captures a key stage of the data flow pipeline.
"""

import os
import sys
import numpy as np
import cv2
import math

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

OUT = "data/report_visuals"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FIGURE 1 — PIPELINE ARCHITECTURE DATA FLOW
# ─────────────────────────────────────────────────────────────
def fig_pipeline_overview():
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 16)
    ax.set_ylim(-1.5, 2.5)
    ax.axis("off")
    ax.set_title("GPS-Denied Visual Navigation — Data Flow Architecture", color="white", fontsize=14, fontweight="bold", pad=12)

    boxes = [
        (0.5,  0.5, "#1f4e79", "DATASET\nCAMERA\n(DatasetCamera)",          "UAVVisLoc\n01.csv + JPG"),
        (3.5,  0.5, "#174117", "SUPERPOINT\nTRT FP16\n(GPU)",               "Keypoints +\nDescriptors (512×256)"),
        (6.5,  0.5, "#4a1f1f", "MAP RETRIEVAL\n(FAISS IVFPQ\n+ SQLite)",    "Top-3 Candidate\nSat Patches"),
        (9.5,  0.5, "#3d2f0a", "FEATURE\nMATCHING\n(MNN + RANSAC)",         "Affine M [2×3]\n+ Inlier Count"),
        (12.5, 0.5, "#1a1a4e", "EKF FUSION\n+ Phase\nController",           "Smoothed\nWGS84 Pose"),
    ]

    for (x, y, color, label, sublabel) in boxes:
        rect = mpatches.FancyBboxPatch((x, y), 2.5, 1.5, boxstyle="round,pad=0.08",
                                       linewidth=1.5, edgecolor="#58a6ff", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + 1.25, y + 1.05, label, ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold", multialignment="center")
        ax.text(x + 1.25, y + 0.35, sublabel, ha="center", va="center", color="#8b949e",
                fontsize=7.5, multialignment="center")

    # Arrows
    for x in [3.0, 6.0, 9.0, 12.0]:
        ax.annotate("", xy=(x + 0.5, 1.25), xytext=(x, 1.25),
                    arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=1.8))

    # Labels above arrows
    arrow_labels = ["Frame\n+ Telemetry", "Visual\nFeatures", "FAISS\nQuery", "Best\nMatch"]
    arrow_x      = [3.05, 6.05, 9.05, 12.05]
    for lbl, ax_x in zip(arrow_labels, arrow_x):
        ax.text(ax_x + 0.25, 1.7, lbl, ha="center", va="bottom", color="#8b949e", fontsize=7)

    # MAVLink output box
    rect = mpatches.FancyBboxPatch((13.0, -0.8), 2.5, 0.9, boxstyle="round,pad=0.08",
                                   linewidth=1.2, edgecolor="#3fb950", facecolor="#0e2a0e")
    ax.add_patch(rect)
    ax.text(14.25, -0.35, "MAVLink UDP\nudp:127.0.0.1:14540", ha="center", va="center",
            color="#3fb950", fontsize=7.5, multialignment="center")
    ax.annotate("", xy=(14.25, -0.0), xytext=(14.25, 0.5),
                arrowprops=dict(arrowstyle="->", color="#3fb950", lw=1.5))

    plt.tight_layout()
    path = f"{OUT}/1_pipeline_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — DATASET CAMERA: REAL DRONE FRAME SAMPLE
# ─────────────────────────────────────────────────────────────
def fig_dataset_camera():
    import pandas as pd

    csv_path = "data/UAV_VisLoc_dataset/01/01.csv"
    img_dir  = "data/UAV_VisLoc_dataset/01/drone"
    df = pd.read_csv(csv_path, header=0,
                     names=["num","filename","date","latitude","longitude","altitude","omega","kappa","phi1","phi2"])
    row = df.iloc[0]

    img_path = os.path.join(img_dir, str(row["filename"]))
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Cannot load image: {img_path}, generating synthetic.")
        img = np.zeros((480, 640, 3), dtype=np.uint8) + 80

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Frame: {row['filename']}", color="white", fontsize=11, pad=8)
    axes[0].axis("off")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#58a6ff")

    meta_labels = [
        ("Latitude",  f"{float(row['latitude']):.6f}°"),
        ("Longitude", f"{float(row['longitude']):.6f}°"),
        ("Altitude",  f"{float(row['altitude']):.1f} m"),
        ("Kappa (Heading)", f"{float(row['kappa']):.2f}°"),
        ("Omega (Roll)",    f"{float(row['omega']):.2f}°"),
        ("Date",      str(row['date'])[:10]),
    ]
    axes[1].set_facecolor("#0d1117")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1); axes[1].axis("off")
    axes[1].set_title("CSV Telemetry Fields (01.csv)", color="white", fontsize=11, pad=8)
    for i, (k, v) in enumerate(meta_labels):
        y = 0.85 - i * 0.13
        rect = mpatches.FancyBboxPatch((0.05, y - 0.05), 0.9, 0.1, boxstyle="round,pad=0.02",
                                       facecolor="#161b22", edgecolor="#21262d")
        axes[1].add_patch(rect)
        axes[1].text(0.12, y, k, color="#8b949e", fontsize=10, va="center")
        axes[1].text(0.88, y, v, color="#e6edf3", fontsize=10, va="center", ha="right", fontweight="bold")

    plt.suptitle("Stage 1: Dataset Camera — Raw Drone Frame + CSV Telemetry Ingestion",
                 color="#58a6ff", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{OUT}/2_dataset_camera.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# FIGURE 3 — SUPERPOINT KEYPOINT EXTRACTION
# ─────────────────────────────────────────────────────────────
def fig_superpoint_keypoints():
    from src.inference.trt_engine import SuperPointEngine

    img_dir = "data/UAV_VisLoc_dataset/01/drone"
    img_files = sorted(os.listdir(img_dir))[:4]
    imgs_rgb, imgs_gray, feats_list = [], [], []

    engine = SuperPointEngine()

    for fname in img_files:
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = engine.extract(gray)
        imgs_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imgs_gray.append(gray)
        feats_list.append(feats)

    n = len(imgs_rgb)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.patch.set_facecolor("#0d1117")
    if n == 1:
        axes = [axes]

    for i, (img, feats) in enumerate(zip(imgs_rgb, feats_list)):
        kps = feats["keypoints"]
        scores = feats.get("scores", np.ones(len(kps)))
        axes[i].imshow(img)
        if len(kps) > 0:
            sc = np.clip(scores / scores.max(), 0, 1) if scores.max() > 0 else scores
            axes[i].scatter(kps[:, 0], kps[:, 1], s=6, c=sc, cmap="plasma", alpha=0.7)
        axes[i].set_title(f"{img_files[i]}\n{len(kps)} keypoints", color="white", fontsize=9, pad=6)
        axes[i].axis("off")

    plt.suptitle("Stage 2: SuperPoint TensorRT FP16 — Keypoint Extraction (color = confidence score)",
                 color="#58a6ff", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{OUT}/3_superpoint_keypoints.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# FIGURE 4 — FAISS CANDIDATE RETRIEVAL
# ─────────────────────────────────────────────────────────────
def fig_faiss_retrieval():
    import sqlite3

    conn = sqlite3.connect("data/map_db.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT center_lat, center_lon, patch_id, rotation_deg FROM patches LIMIT 100")
    rows = cursor.fetchall()
    conn.close()

    lats = [r[0] for r in rows]
    lons = [r[1] for r in rows]
    rots = [r[3] for r in rows]

    # Simulate a query point from CSV
    query_lat, query_lon = 29.760960, 115.974797

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    sc = ax.scatter(lons, lats, c=rots, cmap="coolwarm", s=80, alpha=0.7,
                    edgecolors="#8b949e", linewidths=0.4, label="Satellite Map Patches")
    ax.scatter([query_lon], [query_lat], c="#f0f000", s=200, marker="*",
               zorder=10, label="Query Position (EKF Prior)")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Rotation Augmentation (°)", color="#8b949e")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.get_yticklabels(), color="#8b949e")

    # Draw search radius circle
    import matplotlib.patches as mpatches
    radius_deg = 0.072  # ~8km in degrees lat
    circle = plt.Circle((query_lon, query_lat), radius_deg, color="#58a6ff",
                         fill=False, linewidth=1.5, linestyle="--", label="Search Radius (8 km)")
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude", color="#8b949e")
    ax.set_ylabel("Latitude", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    leg = ax.legend(facecolor="#161b22", edgecolor="#21262d", labelcolor="#e6edf3", fontsize=9)
    ax.set_title("Stage 3: FAISS IVFPQ Retrieval — Map Patch Search Space\n(UAVVisLoc Sequence 01 Coverage Area)",
                 color="#58a6ff", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{OUT}/4_faiss_retrieval.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# FIGURE 5 — FEATURE MATCHING (live drone vs sat patch)
# ─────────────────────────────────────────────────────────────
def fig_feature_matching():
    from src.inference.trt_engine import SuperPointEngine
    from src.inference.lightglue_matcher import LightGlueMatcher
    from src.camera.sim_camera import SimCamera

    engine = SuperPointEngine()
    matcher = LightGlueMatcher()

    img_dir = "data/UAV_VisLoc_dataset/01/drone"
    img_file = sorted(os.listdir(img_dir))[0]
    live_bgr = cv2.imread(os.path.join(img_dir, img_file))
    live_gray = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2GRAY)
    live_feats = engine.extract(live_gray)

    # Get sat patch at same area
    bbox = (29.702283, 115.970635, 29.774065, 115.996851)
    sat_cam = SimCamera(texture_path="data/satellite01.jpg", bbox=bbox)
    sat_bgr = sat_cam.get_frame_at_pose(29.760960, 115.974797, alt_m=405.0, heading_deg=0.0)
    sat_gray = cv2.cvtColor(sat_bgr, cv2.COLOR_BGR2GRAY)
    sat_feats = engine.extract(sat_gray)

    # Match
    inliers, M = matcher.match(live_feats, sat_feats)

    live_rgb = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2RGB)
    sat_rgb  = cv2.cvtColor(sat_bgr,  cv2.COLOR_BGR2RGB)

    kps0 = live_feats["keypoints"]
    kps1 = sat_feats["keypoints"]
    desc0 = live_feats["descriptors"]
    desc1 = sat_feats["descriptors"]

    if len(kps0) > 4 and len(kps1) > 4:
        n0 = np.linalg.norm(desc0, axis=1, keepdims=True); n0[n0 < 1e-6] = 1.0
        n1 = np.linalg.norm(desc1, axis=1, keepdims=True); n1[n1 < 1e-6] = 1.0
        sim = np.dot(desc0 / n0, (desc1 / n1).T)
        m01 = np.argmax(sim, axis=1)
        m10 = np.argmax(sim, axis=0)
        mnn = (m10[m01] == np.arange(len(m01))) & (np.max(sim, axis=1) > 0.2)
        mnn_idx = np.where(mnn)[0]
    else:
        mnn_idx = np.array([], dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")

    axes[0].imshow(live_rgb)
    if len(mnn_idx) > 0:
        axes[0].scatter(kps0[mnn_idx, 0], kps0[mnn_idx, 1], s=12, c="#f78166", alpha=0.8, label=f"{len(mnn_idx)} matched kps")
    axes[0].scatter(kps0[:, 0], kps0[:, 1], s=3, c="#8b949e", alpha=0.3, label=f"{len(kps0)} total kps")
    axes[0].set_title(f"Live Drone Frame\n{img_file}", color="white", fontsize=10, pad=6)
    axes[0].axis("off")
    axes[0].legend(facecolor="#161b22", edgecolor="#21262d", labelcolor="#e6edf3", fontsize=8, loc="lower right")

    axes[1].imshow(sat_rgb)
    if len(mnn_idx) > 0:
        matched1 = m01[mnn_idx]
        axes[1].scatter(kps1[matched1, 0], kps1[matched1, 1], s=12, c="#3fb950", alpha=0.8, label=f"{len(mnn_idx)} matched kps")
    axes[1].scatter(kps1[:, 0], kps1[:, 1], s=3, c="#8b949e", alpha=0.3, label=f"{len(kps1)} total kps")
    axes[1].set_title(f"Retrieved Satellite Map Patch\n(Center: 29.760960°N, 115.974797°E)", color="white", fontsize=10, pad=6)
    axes[1].axis("off")
    axes[1].legend(facecolor="#161b22", edgecolor="#21262d", labelcolor="#e6edf3", fontsize=8, loc="lower right")

    fig.suptitle(f"Stage 4: Feature Matching — MNN + RANSAC Affine Estimation\nRANSAC Inliers: {inliers} | Affine M[:, 2] = [{M[0,2]:.1f}, {M[1,2]:.1f}] px",
                 color="#58a6ff", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{OUT}/5_feature_matching.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# FIGURE 6 — TRAJECTORY: GROUND TRUTH VS PREDICTED
# ─────────────────────────────────────────────────────────────
def fig_trajectory():
    csv_path = "data/flight_log.csv"
    if not os.path.exists(csv_path):
        print("[WARN] flight_log.csv not found, run the evaluation first.")
        return

    steps, pred_lats, pred_lons, gt_lats, gt_lons, inliers, phases = [], [], [], [], [], [], []
    with open(csv_path, "r") as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 7:
                try:
                    steps.append(int(p[0]))
                    pred_lats.append(float(p[1])); pred_lons.append(float(p[2]))
                    gt_lats.append(float(p[3]));   gt_lons.append(float(p[4]))
                    inliers.append(int(p[5]))
                    phases.append(p[6])
                except ValueError:
                    continue

    if not steps:
        print("[WARN] No data rows in flight_log.csv")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")

    # Trajectory plot
    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.plot(gt_lons, gt_lats,   "o-", color="#58a6ff", linewidth=2.0, markersize=5,  label="Ground Truth Path")
    ax.plot(pred_lons, pred_lats, "s--", color="#f78166", linewidth=1.5, markersize=5, label="Predicted Path")
    ax.plot(gt_lons[0], gt_lats[0],   "^", color="#3fb950", markersize=12, zorder=10, label="Start")
    ax.plot(gt_lons[-1], gt_lats[-1], "X", color="#ff7b72", markersize=12, zorder=10, label="End")
    for i, (glat, glon, plat, plon) in enumerate(zip(gt_lats, gt_lons, pred_lats, pred_lons)):
        ax.plot([glon, plon], [glat, plat], "-", color="#8b949e", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Longitude", color="#8b949e"); ax.set_ylabel("Latitude", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    leg = ax.legend(facecolor="#161b22", edgecolor="#21262d", labelcolor="#e6edf3", fontsize=9)
    ax.set_title("Trajectory Comparison\n(Ground Truth vs Predicted)", color="white", fontsize=11, pad=8)

    # Inlier & Phase timeline
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    colors_bar = ["#3fb950" if "HIGH" in ph else "#d29922" if "IMU" in ph else "#f78166" for ph in phases]
    ax2.bar(steps, inliers, color=colors_bar, edgecolor="#0d1117", linewidth=0.5)
    ax2.set_xlabel("Flight Step", color="#8b949e"); ax2.set_ylabel("RANSAC Inliers", color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#21262d")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3fb950", label="HIGH_CONFIDENCE_TRACKING"),
        Patch(facecolor="#d29922", label="IMU_DEAD_RECKONING"),
        Patch(facecolor="#f78166", label="UNANCHORED_ACQUISITION"),
    ]
    ax2.legend(handles=legend_elements, facecolor="#161b22", edgecolor="#21262d",
               labelcolor="#e6edf3", fontsize=8)
    ax2.set_title("RANSAC Inlier Count per Step\n(Color = Navigation Phase)", color="white", fontsize=11, pad=8)

    plt.suptitle("Stage 6: EKF Fusion — Trajectory Accuracy & Phase State Timeline",
                 color="#58a6ff", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{OUT}/6_trajectory.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# FIGURE 7 — CEP ACCURACY RING CHART
# ─────────────────────────────────────────────────────────────
def fig_cep_rings():
    csv_path = "data/flight_log.csv"
    if not os.path.exists(csv_path):
        return

    from shared.geo.tile_math import haversine_distance

    pred_lats, pred_lons, gt_lats, gt_lons = [], [], [], []
    with open(csv_path, "r") as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 5:
                try:
                    pred_lats.append(float(p[1])); pred_lons.append(float(p[2]))
                    gt_lats.append(float(p[3]));   gt_lons.append(float(p[4]))
                except ValueError:
                    continue

    errors = [haversine_distance(plat, plon, glat, glon)
              for plat, plon, glat, glon in zip(pred_lats, pred_lons, gt_lats, gt_lons)]
    errors_sorted = sorted(errors)
    cep50 = np.percentile(errors, 50)
    cep90 = np.percentile(errors, 90)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#0d1117")

    # CEP rings on scatter
    ax1.set_facecolor("#161b22")
    dx = [haversine_distance(plat, plon, glat, plon) * (1 if plon > glon else -1)
          for plat, plon, glat, glon in zip(pred_lats, pred_lons, gt_lats, gt_lons)]
    dy = [haversine_distance(plat, plon, plat, glon) * (1 if plat > glat else -1)
          for plat, plon, glat, glon in zip(pred_lats, pred_lons, gt_lats, gt_lons)]
    ax1.scatter(dx, dy, c="#f78166", s=60, zorder=5, label="Position Errors", edgecolors="#0d1117", linewidths=0.5)
    for r, color, lbl in [(cep50, "#3fb950", f"CEP50 = {cep50:.0f}m"), (cep90, "#d29922", f"CEP90 = {cep90:.0f}m")]:
        circle = plt.Circle((0, 0), r, color=color, fill=False, linewidth=2.0, linestyle="--", label=lbl)
        ax1.add_patch(circle)
    ax1.axhline(0, color="#8b949e", linewidth=0.5); ax1.axvline(0, color="#8b949e", linewidth=0.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("East Error (m)", color="#8b949e"); ax1.set_ylabel("North Error (m)", color="#8b949e")
    ax1.tick_params(colors="#8b949e")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#21262d")
    ax1.legend(facecolor="#161b22", edgecolor="#21262d", labelcolor="#e6edf3", fontsize=9)
    ax1.set_title("CEP50 / CEP90 Error Rings\n(Position Scatter from Ground Truth)", color="white", fontsize=11, pad=8)

    # CDF
    ax2.set_facecolor("#161b22")
    cdf = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
    ax2.plot(errors_sorted, cdf, color="#58a6ff", linewidth=2.5, label="CDF")
    ax2.axvline(cep50, color="#3fb950", linestyle="--", linewidth=1.8, label=f"CEP50 = {cep50:.0f}m")
    ax2.axvline(cep90, color="#d29922", linestyle="--", linewidth=1.8, label=f"CEP90 = {cep90:.0f}m")
    ax2.axhline(0.5, color="#3fb950", linestyle=":", linewidth=1.0, alpha=0.5)
    ax2.axhline(0.9, color="#d29922", linestyle=":", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("Position Error (m)", color="#8b949e"); ax2.set_ylabel("Cumulative Probability", color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#21262d")
    ax2.legend(facecolor="#161b22", edgecolor="#21262d", labelcolor="#e6edf3", fontsize=9)
    ax2.set_title("Empirical CDF of Position Error\n(GPS-Denied Visual Navigation)", color="white", fontsize=11, pad=8)

    plt.suptitle("Stage 7: Accuracy Evaluation — CEP Metrics & Error Distribution",
                 color="#58a6ff", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{OUT}/7_cep_accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating report visualizations...")
    fig_pipeline_overview()
    fig_dataset_camera()
    fig_superpoint_keypoints()
    fig_faiss_retrieval()
    fig_feature_matching()

    # Run a mini flight so flight_log.csv exists
    from src.main import GPSDeniedPipeline
    pipeline = GPSDeniedPipeline("config/jetson.yaml", mode="dataset")
    for _ in range(10):
        pipeline.run_step()
    pipeline.close()

    fig_trajectory()
    fig_cep_rings()
    print(f"\nAll visuals saved to: {OUT}/")
