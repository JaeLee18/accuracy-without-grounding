"""
Phase 3 preprocessing (CPU-only):
  1. Select Exp D subset (100 VGG=+1 questions, high-VGG task types)
  2. Generate blur videos (sigma=3,6,10)
  3. Generate framedrop videos (every 2nd,3rd,4th frame)
  4. Generate shuffled_sample.json for Exp F
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import random
import subprocess
from tqdm import tqdm
from imageio_ffmpeg import get_ffmpeg_exe

QWEN_PATH   = VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json"
BLACK_PATH  = VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json"
SAMPLE_PATH = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
VIDEO_DIR   = VDG_DATA_ROOT + "/videomme_full/videos"
ABLATION_BASE = VDG_DATA_ROOT + "/videomme_full/ablation"
EXP_D_SAMPLE  = VDG_DATA_ROOT + "/videomme_full/exp_d_sample.json"
SHUFFLED_SAMPLE = VDG_DATA_ROOT + "/videomme_full/shuffled_sample.json"

FFMPEG = os.environ.get("FFMPEG_BIN", get_ffmpeg_exe())

# High-VGG task types (skip TR and Action Reasoning — long videos, low VGG)
HIGH_VGG_TYPES = {"Attribute Perception", "Object Recognition", "Action Recognition", "OCR Problems"}
CONDITIONS = ["original", "crf18", "crf23", "crf28", "crf33", "crf38"]

# Task-type VGG order for sorting (descending)
VGG_ORDER = {
    "Attribute Perception": 0.40,
    "Object Recognition":   0.25,
    "Action Recognition":   0.21,
    "OCR Problems":         0.20,
}


def run_ffmpeg(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")


def verify(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


# ── Step 1: Select Exp D subset ───────────────────────────────────────────────
print("=== Step 1: Select Exp D subset ===")

with open(QWEN_PATH) as f:
    qwen = json.load(f)
with open(BLACK_PATH) as f:
    black = json.load(f)
with open(SAMPLE_PATH) as f:
    samples = json.load(f)

qwen_idx  = {(r["question_id"], r["condition"]): r.get("correct", False) for r in qwen}
black_idx = {r["question_id"]: r.get("correct", False) for r in black}
qid_to_sample = {s["question_id"]: s for s in samples}

# VGG=+1: orig=correct, black=wrong, all 6 conditions present, high-VGG task type
exp_d_candidates = []
for qid, sample in qid_to_sample.items():
    tt = sample["task_type"]
    if tt not in HIGH_VGG_TYPES:
        continue
    if not all((qid, c) in qwen_idx for c in CONDITIONS):
        continue
    orig_correct  = qwen_idx.get((qid, "original"), False)
    black_correct = black_idx.get(qid, False)
    if orig_correct and not black_correct:
        exp_d_candidates.append((VGG_ORDER.get(tt, 0), qid, sample))

# Sort by task-type VGG descending, take 100
exp_d_candidates.sort(key=lambda x: -x[0])
exp_d_samples = [s for _, _, s in exp_d_candidates[:100]]
print(f"Exp D subset: {len(exp_d_samples)} questions")

from collections import Counter
print("Task type breakdown:", Counter(s["task_type"] for s in exp_d_samples))

with open(EXP_D_SAMPLE, "w") as f:
    json.dump(exp_d_samples, f, indent=2)
print(f"Saved: {EXP_D_SAMPLE}")


# ── Step 2: Blur videos ───────────────────────────────────────────────────────
print("\n=== Step 2: Generate blur videos ===")

video_ids = sorted(set(s["videoID"] for s in exp_d_samples))
print(f"Unique videos: {len(video_ids)}")

blur_levels = [("blur_s3", "gblur=sigma=3"), ("blur_s6", "gblur=sigma=6"), ("blur_s10", "gblur=sigma=10")]

for dir_name, vf_filter in blur_levels:
    out_dir = f"{ABLATION_BASE}/{dir_name}"
    os.makedirs(out_dir, exist_ok=True)
    done = skipped = failed = 0
    for vid in tqdm(video_ids, desc=dir_name):
        src = f"{VIDEO_DIR}/{vid}.mp4"
        dst = f"{out_dir}/{vid}.mp4"
        if verify(dst):
            skipped += 1
            continue
        try:
            run_ffmpeg([FFMPEG, "-y", "-i", src, "-vf", vf_filter,
                        "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast",
                        "-c:a", "copy", "-loglevel", "error", dst])
            if verify(dst):
                done += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ERROR {vid}: {e}")
    print(f"  {dir_name}: {done} new, {skipped} skipped, {failed} failed")


# ── Step 3: Framedrop videos ──────────────────────────────────────────────────
print("\n=== Step 3: Generate framedrop videos ===")

drop_levels = [
    ("framedrop2", "select='not(mod(n,2))',setpts=N/FRAME_RATE/TB"),
    ("framedrop3", "select='not(mod(n,3))',setpts=N/FRAME_RATE/TB"),
    ("framedrop4", "select='not(mod(n,4))',setpts=N/FRAME_RATE/TB"),
]

for dir_name, vf_filter in drop_levels:
    out_dir = f"{ABLATION_BASE}/{dir_name}"
    os.makedirs(out_dir, exist_ok=True)
    done = skipped = failed = 0
    for vid in tqdm(video_ids, desc=dir_name):
        src = f"{VIDEO_DIR}/{vid}.mp4"
        dst = f"{out_dir}/{vid}.mp4"
        if verify(dst):
            skipped += 1
            continue
        try:
            run_ffmpeg([FFMPEG, "-y", "-i", src, "-vf", vf_filter,
                        "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast",
                        "-an", "-loglevel", "error", dst])
            if verify(dst):
                done += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ERROR {vid}: {e}")
    print(f"  {dir_name}: {done} new, {skipped} skipped, {failed} failed")


# ── Step 4: Shuffled sample for Exp F ─────────────────────────────────────────
print("\n=== Step 4: Generate shuffled_sample.json ===")

letters = ["A", "B", "C", "D"]
shuffled_samples = []

for i, s in enumerate(samples):
    opts = s["options"][:]          # ["A. ...", "B. ...", "C. ...", "D. ..."]
    rng = random.Random(42 + i)
    new_order = [0, 1, 2, 3]
    rng.shuffle(new_order)

    shuffled_opts = [opts[j] for j in new_order]
    # Remap correct answer: find which new position holds the original correct
    orig_letter = s["answer"]       # "A", "B", "C", or "D"
    orig_idx = letters.index(orig_letter)
    new_pos = new_order.index(orig_idx)
    shuffled_answer = letters[new_pos]

    shuffle_map = {letters[new_i]: letters[orig_i] for new_i, orig_i in enumerate(new_order)}

    entry = dict(s)
    entry["shuffled_options"] = shuffled_opts
    entry["shuffled_answer"]  = shuffled_answer
    entry["shuffle_map"]      = shuffle_map
    shuffled_samples.append(entry)

with open(SHUFFLED_SAMPLE, "w") as f:
    json.dump(shuffled_samples, f, indent=2)
print(f"Saved: {SHUFFLED_SAMPLE}  ({len(shuffled_samples)} entries)")

# Sanity check
n_changed = sum(1 for s in shuffled_samples if s["shuffled_answer"] != s["answer"])
print(f"Questions where correct letter changed: {n_changed}/{len(shuffled_samples)}")

print("\n=== Preprocessing complete ===")
