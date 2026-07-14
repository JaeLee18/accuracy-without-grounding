"""
Find semantic frame pairs in Video-MME videos for LGIP-video extension.
Identifies frames where something semantically changes (object/color/count)
and tests whether H.264 compression degrades VLM ability to spot the difference.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import sys
import subprocess
import numpy as np
import cv2
import torch
import clip
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from imageio_ffmpeg import get_ffmpeg_exe

# ── Paths ──────────────────────────────────────────────────────────────────────
SAMPLE_PATH   = VDG_DATA_ROOT + "/videomme/pilot_sample.json"
VIDEO_DIR     = VDG_DATA_ROOT + "/videomme/videos"
OUTPUT_DIR    = VDG_RESULTS_ROOT + "/semantic_pairs"
REPORT_PATH   = VDG_RESULTS_ROOT + "/semantic_pairs/report.json"

# ── Thresholds ─────────────────────────────────────────────────────────────────
CLIP_DIST_MIN  = 0.08   # min cosine distance (too low = near-identical)
CLIP_DIST_MAX  = 0.35   # max cosine distance (too high = scene cut)
TIME_GAP_MIN   = 1.0    # seconds
TIME_GAP_MAX   = 30.0   # seconds
TOP_PAIRS_PER_VIDEO = 5
SAVE_PAIRS_PER_VIDEO = 3
FPS_EXTRACT    = 1.0    # frames per second to extract

FFMPEG = os.environ.get("FFMPEG_BIN", get_ffmpeg_exe())


# ── Step 1: Load CLIP ──────────────────────────────────────────────────────────
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ViT-B/32 on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print("CLIP loaded.")
    return model, preprocess, device


# ── Step 2: Extract frames ─────────────────────────────────────────────────────
def extract_frames(video_path, fps=FPS_EXTRACT):
    """Extract frames at given fps using OpenCV. Returns list of (frame_idx, timestamp, np_bgr)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 25.0
    frame_interval = max(1, round(native_fps / fps))
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            ts = frame_idx / native_fps
            frames.append((frame_idx, ts, frame.copy()))
        frame_idx += 1
    cap.release()
    return frames


# ── Step 3: Embed frames with CLIP ─────────────────────────────────────────────
def embed_frames(frames, model, preprocess, device, batch_size=32):
    """Returns (N, 512) numpy array of CLIP embeddings."""
    embeddings = []
    imgs = [preprocess(Image.fromarray(cv2.cvtColor(f[2], cv2.COLOR_BGR2RGB))) for f in frames]
    for i in range(0, len(imgs), batch_size):
        batch = torch.stack(imgs[i:i+batch_size]).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 512))


# ── Step 4: Find candidate pairs ───────────────────────────────────────────────
def find_candidate_pairs(frames, embeddings, top_n=TOP_PAIRS_PER_VIDEO):
    candidates = []
    n = len(frames)
    for i in range(n):
        for j in range(i + 1, n):
            ts_a, ts_b = frames[i][1], frames[j][1]
            time_gap = ts_b - ts_a
            if time_gap < TIME_GAP_MIN or time_gap > TIME_GAP_MAX:
                continue
            clip_dist = float(cosine(embeddings[i], embeddings[j]))
            if clip_dist < CLIP_DIST_MIN or clip_dist > CLIP_DIST_MAX:
                continue
            score = clip_dist / (1.0 + 0.1 * time_gap)
            candidates.append({
                "frame_idx_a": frames[i][0],
                "frame_idx_b": frames[j][0],
                "timestamp_a": round(ts_a, 2),
                "timestamp_b": round(ts_b, 2),
                "time_gap": round(time_gap, 2),
                "clip_distance": round(clip_dist, 4),
                "score": round(score, 4),
                "_i": i,
                "_j": j,
            })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


# ── Step 5: Classify change type ───────────────────────────────────────────────
def classify_change(frame_a_bgr, frame_b_bgr):
    h, w = frame_a_bgr.shape[:2]
    a_resized = cv2.resize(frame_a_bgr, (224, 224))
    b_resized = cv2.resize(frame_b_bgr, (224, 224))

    # Absolute pixel diff
    diff = cv2.absdiff(a_resized, b_resized).astype(np.float32)
    diff_gray = diff.mean(axis=2)

    # HSV diff for color analysis
    a_hsv = cv2.cvtColor(a_resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    b_hsv = cv2.cvtColor(b_resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_diff = np.abs(a_hsv[:, :, 0] - b_hsv[:, :, 0])
    sat_diff = np.abs(a_hsv[:, :, 1] - b_hsv[:, :, 1])
    val_diff = np.abs(a_hsv[:, :, 2] - b_hsv[:, :, 2])

    # Spatial concentration: divide into 4x4 grid, check if diff is localized
    grid = 4
    cell_h, cell_w = 224 // grid, 224 // grid
    cell_diffs = []
    for gi in range(grid):
        for gj in range(grid):
            cell = diff_gray[gi*cell_h:(gi+1)*cell_h, gj*cell_w:(gj+1)*cell_w]
            cell_diffs.append(cell.mean())
    cell_diffs = np.array(cell_diffs)
    total_mean = cell_diffs.mean()
    max_cell = cell_diffs.max()
    concentration = max_cell / (total_mean + 1e-6)

    color_signal = (hue_diff.mean() + sat_diff.mean()) / (val_diff.mean() + 1e-6)

    if color_signal > 1.5 and hue_diff.mean() > 10:
        change_type = "color"
    elif concentration > 3.0:
        change_type = "object"
    else:
        change_type = "unknown"

    return change_type, diff_gray


# ── Step 6: Save pair outputs ───────────────────────────────────────────────────
def save_pair(video_id, pair_n, frame_a_bgr, frame_b_bgr, meta, diff_gray):
    pair_dir = f"{OUTPUT_DIR}/{video_id}/pair_{pair_n}"
    os.makedirs(pair_dir, exist_ok=True)

    # Save individual frames
    frame_a_path = f"{pair_dir}/frame_a.jpg"
    frame_b_path = f"{pair_dir}/frame_b.jpg"
    cv2.imwrite(frame_a_path, frame_a_bgr)
    cv2.imwrite(frame_b_path, frame_b_bgr)

    # Side-by-side + diff heatmap
    a_rgb = cv2.cvtColor(frame_a_bgr, cv2.COLOR_BGR2RGB)
    b_rgb = cv2.cvtColor(frame_b_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(a_rgb)
    axes[0].set_title(f"Frame A  t={meta['timestamp_a']}s", fontsize=10)
    axes[0].axis('off')
    axes[1].imshow(b_rgb)
    axes[1].set_title(f"Frame B  t={meta['timestamp_b']}s", fontsize=10)
    axes[1].axis('off')
    im = axes[2].imshow(diff_gray, cmap='hot', vmin=0)
    axes[2].set_title(
        f"Diff heatmap\nCLIP dist={meta['clip_distance']}  type={meta['change_type']}",
        fontsize=10
    )
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.suptitle(
        f"{video_id} | pair {pair_n} | gap={meta['time_gap']}s | score={meta['score']}",
        fontsize=11
    )
    plt.tight_layout()
    comparison_path = f"{pair_dir}/comparison.png"
    plt.savefig(comparison_path, dpi=100, bbox_inches='tight')
    plt.close()

    # JSON metadata
    meta_out = {k: v for k, v in meta.items() if not k.startswith("_")}
    meta_out.update({
        "video_id": video_id,
        "pair_n": pair_n,
        "frame_a_path": frame_a_path,
        "frame_b_path": frame_b_path,
        "comparison_path": comparison_path,
    })
    with open(f"{pair_dir}/meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    return comparison_path, meta_out


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load sample
    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    video_ids = sorted(set(s["videoID"] for s in samples))
    print(f"Videos to process: {len(video_ids)}")

    # Load CLIP
    model, preprocess, device = load_clip()

    all_pairs = []
    comparison_paths = []
    skipped = []
    change_type_counts = {"color": 0, "object": 0, "unknown": 0}

    for video_id in tqdm(video_ids, desc="Videos"):
        video_path = f"{VIDEO_DIR}/{video_id}.mp4"
        if not os.path.exists(video_path):
            skipped.append(video_id)
            continue

        # Extract frames
        frames = extract_frames(video_path)
        if len(frames) < 2:
            skipped.append(video_id)
            continue

        # Embed
        embeddings = embed_frames(frames, model, preprocess, device)
        if embeddings.shape[0] < 2:
            skipped.append(video_id)
            continue

        # Find pairs
        candidates = find_candidate_pairs(frames, embeddings)
        if not candidates:
            continue

        # Classify + save top pairs
        video_pairs = []
        for pair_n, cand in enumerate(candidates[:SAVE_PAIRS_PER_VIDEO]):
            fa_bgr = frames[cand["_i"]][2]
            fb_bgr = frames[cand["_j"]][2]
            change_type, diff_gray = classify_change(fa_bgr, fb_bgr)
            cand["change_type"] = change_type
            change_type_counts[change_type] += 1

            comp_path, meta_out = save_pair(video_id, pair_n, fa_bgr, fb_bgr, cand, diff_gray)
            comparison_paths.append(comp_path)
            video_pairs.append(meta_out)
            all_pairs.append(meta_out)

        # Also record remaining candidates (not saved to disk)
        for cand in candidates[SAVE_PAIRS_PER_VIDEO:]:
            fa_bgr = frames[cand["_i"]][2]
            fb_bgr = frames[cand["_j"]][2]
            change_type, _ = classify_change(fa_bgr, fb_bgr)
            cand["change_type"] = change_type
            change_type_counts[change_type] += 1
            cand_out = {k: v for k, v in cand.items() if not k.startswith("_")}
            cand_out["video_id"] = video_id
            all_pairs.append(cand_out)

    # ── Step 6: Report ─────────────────────────────────────────────────────────
    all_pairs.sort(key=lambda x: x["score"], reverse=True)
    good_pairs = [p for p in all_pairs if p["score"] > 0.1]
    verdict = (
        "SUFFICIENT PAIRS FOUND" if len(good_pairs) >= 50
        else "INSUFFICIENT — adjust thresholds"
    )

    print(f"\n{'='*60}")
    print(f"Videos processed : {len(video_ids) - len(skipped)} / {len(video_ids)}")
    print(f"Skipped          : {len(skipped)}")
    print(f"Total pairs found: {len(all_pairs)}")
    print(f"Pairs score>0.1  : {len(good_pairs)}")
    print(f"Change types     : {change_type_counts}")
    print(f"Verdict          : {verdict}")
    print(f"\nTop 10 pairs:")
    for i, p in enumerate(all_pairs[:10]):
        print(f"  {i+1:2d}. {p['video_id']}  "
              f"t={p['timestamp_a']}->{p['timestamp_b']}s  "
              f"clip={p['clip_distance']}  type={p['change_type']}  score={p['score']}")

    report = {
        "videos_processed": len(video_ids) - len(skipped),
        "videos_skipped": len(skipped),
        "total_pairs": len(all_pairs),
        "good_pairs_score_gt_0.1": len(good_pairs),
        "change_type_counts": change_type_counts,
        "verdict": verdict,
        "top_10_pairs": all_pairs[:10],
        "all_pairs": all_pairs,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {REPORT_PATH}")

    # ── Step 7: Open top 5 comparison images ───────────────────────────────────
    print("\nOpening top 5 comparison images...")
    for path in comparison_paths[:5]:
        if os.path.exists(path):
            subprocess.Popen(f'start "" "{path}"', shell=True)

    if verdict == "INSUFFICIENT — adjust thresholds":
        print("\nSuggestions:")
        if len(all_pairs) == 0:
            print("  - No pairs found at all. Try widening CLIP_DIST_MIN to 0.05 or TIME_GAP_MAX to 60s.")
        elif len(good_pairs) < 50:
            print(f"  - Only {len(good_pairs)} pairs with score>0.1.")
            print("  - Try lowering CLIP_DIST_MIN to 0.05 (capture subtler changes)")
            print("  - Or expand to full Video-MME dataset (400+ videos)")


if __name__ == "__main__":
    main()
