"""
Compute average optical flow magnitude per video as a temporal dependency proxy.
Uses Farneback dense optical flow on uniformly sampled frame pairs.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import cv2
import numpy as np
from tqdm import tqdm

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
VIDEO_DIR = VDG_DATA_ROOT + "/videomme_full/videos"
OUTPUT_PATH = VDG_RESULTS_ROOT + "/full_study/optical_flow.json"
NUM_PAIRS = 20  # number of frame pairs to sample per video


def compute_avg_flow(video_path, num_pairs=NUM_PAIRS):
    """Compute average optical flow magnitude across sampled frame pairs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return 0.0

    # Sample frame indices uniformly
    indices = np.linspace(0, total_frames - 2, min(num_pairs + 1, total_frames), dtype=int)

    magnitudes = []
    for i in range(len(indices) - 1):
        idx1, idx2 = int(indices[i]), int(indices[i + 1])
        if idx1 == idx2:
            idx2 = min(idx1 + 1, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx1)
        ret1, frame1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx2)
        ret2, frame2 = cap.read()

        if not ret1 or not ret2:
            continue

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Resize for speed
        h, w = gray1.shape
        scale = min(1.0, 320.0 / max(h, w))
        if scale < 1.0:
            gray1 = cv2.resize(gray1, None, fx=scale, fy=scale)
            gray2 = cv2.resize(gray2, None, fx=scale, fy=scale)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))

    cap.release()

    if not magnitudes:
        return 0.0
    return float(np.mean(magnitudes))


def main():
    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    # Get unique videos with their task types
    video_tasks = {}
    for s in samples:
        vid = s["videoID"]
        if vid not in video_tasks:
            video_tasks[vid] = set()
        video_tasks[vid].add(s["task_type"])

    unique_vids = sorted(video_tasks.keys())
    print(f"Computing optical flow for {len(unique_vids)} videos...")

    results = {}
    for vid in tqdm(unique_vids, desc="Optical flow"):
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"  Missing: {vid}")
            continue
        avg_flow = compute_avg_flow(video_path)
        results[vid] = {
            "avg_flow": avg_flow,
            "task_types": sorted(video_tasks[vid]),
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary per task type
    print(f"\n=== Average Optical Flow by Task Type ===")
    task_flows = {}
    for vid, info in results.items():
        for tt in info["task_types"]:
            if tt not in task_flows:
                task_flows[tt] = []
            if info["avg_flow"] is not None:
                task_flows[tt].append(info["avg_flow"])

    for tt in sorted(task_flows.keys()):
        vals = task_flows[tt]
        print(f"  {tt:<25} mean={np.mean(vals):.3f}  median={np.median(vals):.3f}  n={len(vals)}")

    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
