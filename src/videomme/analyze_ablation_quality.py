"""
Pre-inference analysis of ablation videos:
  1. SSIM/PSNR between original and each degradation condition
  2. CLIP embedding distance per condition
  3. Temporal coherence (frame-to-frame SSIM)
  4. Shuffled sample validation
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import numpy as np
import cv2
import torch
import clip
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
VIDEO_DIR     = VDG_DATA_ROOT + "/videomme_full/videos"
ABLATION_BASE = VDG_DATA_ROOT + "/videomme_full/ablation"
CRF_BASE      = VDG_DATA_ROOT + "/videomme_full"
EXP_D_SAMPLE  = VDG_DATA_ROOT + "/videomme_full/exp_d_sample.json"
SHUFFLED_PATH  = VDG_DATA_ROOT + "/videomme_full/shuffled_sample.json"
PLOTS_DIR      = VDG_RESULTS_ROOT + "/full_study/plots"
REPORT_PATH    = VDG_RESULTS_ROOT + "/full_study/ablation_quality_report.md"

os.makedirs(PLOTS_DIR, exist_ok=True)

CONDITIONS = {
    "crf18":      f"{CRF_BASE}/crf18",
    "crf38":      f"{CRF_BASE}/crf38",
    "blur_s3":    f"{ABLATION_BASE}/blur_s3",
    "blur_s6":    f"{ABLATION_BASE}/blur_s6",
    "blur_s10":   f"{ABLATION_BASE}/blur_s10",
    "framedrop2": f"{ABLATION_BASE}/framedrop2",
    "framedrop3": f"{ABLATION_BASE}/framedrop3",
    "framedrop4": f"{ABLATION_BASE}/framedrop4",
}

with open(EXP_D_SAMPLE) as f:
    exp_d = json.load(f)
video_ids = sorted(set(s["videoID"] for s in exp_d))

N_SAMPLE_FRAMES = 5   # frames per video to sample
N_VIDEOS_LIMIT  = 30  # limit for speed (subset of 92)

sample_vids = video_ids[:N_VIDEOS_LIMIT]
print(f"Analyzing {len(sample_vids)}/{len(video_ids)} videos, {N_SAMPLE_FRAMES} frames each")


# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_frames(video_path, n=N_SAMPLE_FRAMES):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, n, dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def compute_ssim(img1, img2):
    """Simple SSIM on grayscale."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2
    s1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    s2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    s12   = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu12
    num   = (2 * mu12 + c1) * (2 * s12 + c2)
    den   = (mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2)
    ssim_map = num / den
    return float(ssim_map.mean())


def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


# ── 1. SSIM / PSNR ────────────────────────────────────────────────────────────
print("\n=== 1. SSIM / PSNR per condition ===")

metrics = defaultdict(lambda: {"ssim": [], "psnr": []})

for vid in tqdm(sample_vids, desc="SSIM/PSNR"):
    orig_path = f"{VIDEO_DIR}/{vid}.mp4"
    orig_frames = extract_frames(orig_path)
    if not orig_frames:
        continue

    for cond, cond_dir in CONDITIONS.items():
        cond_path = f"{cond_dir}/{vid}.mp4"
        if not os.path.exists(cond_path):
            continue
        cond_frames = extract_frames(cond_path)
        n = min(len(orig_frames), len(cond_frames))
        for i in range(n):
            # Resize to match if needed
            h, w = orig_frames[i].shape[:2]
            cf = cv2.resize(cond_frames[i], (w, h))
            metrics[cond]["ssim"].append(compute_ssim(orig_frames[i], cf))
            metrics[cond]["psnr"].append(compute_psnr(orig_frames[i], cf))

print(f"\n{'Condition':<15} {'SSIM':>8} {'PSNR (dB)':>10} {'n frames':>10}")
print("-" * 45)
quality_summary = {}
for cond in ["crf18", "crf38", "blur_s3", "blur_s6", "blur_s10", "framedrop2", "framedrop3", "framedrop4"]:
    m = metrics[cond]
    if m["ssim"]:
        ssim_mean = np.mean(m["ssim"])
        psnr_mean = np.mean(m["psnr"])
        print(f"{cond:<15} {ssim_mean:>8.4f} {psnr_mean:>10.2f} {len(m['ssim']):>10}")
        quality_summary[cond] = {"ssim": round(ssim_mean, 4), "psnr": round(psnr_mean, 2)}


# ── 2. CLIP embedding distance ────────────────────────────────────────────────
print("\n=== 2. CLIP embedding distance ===")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP ViT-B/32 on {device}...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

clip_dists = defaultdict(list)

for vid in tqdm(sample_vids[:15], desc="CLIP dist"):  # fewer videos for speed
    orig_frames = extract_frames(f"{VIDEO_DIR}/{vid}.mp4", n=3)
    if not orig_frames:
        continue

    # Embed original frames
    orig_tensors = [preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in orig_frames]
    orig_batch = torch.stack(orig_tensors).to(device)
    with torch.no_grad():
        orig_embs = clip_model.encode_image(orig_batch).float()
        orig_embs = orig_embs / orig_embs.norm(dim=-1, keepdim=True)

    for cond, cond_dir in CONDITIONS.items():
        cond_path = f"{cond_dir}/{vid}.mp4"
        if not os.path.exists(cond_path):
            continue
        cond_frames = extract_frames(cond_path, n=3)
        n = min(len(orig_frames), len(cond_frames))
        cond_tensors = [preprocess(Image.fromarray(cv2.cvtColor(cond_frames[i], cv2.COLOR_BGR2RGB)))
                        for i in range(n)]
        cond_batch = torch.stack(cond_tensors).to(device)
        with torch.no_grad():
            cond_embs = clip_model.encode_image(cond_batch).float()
            cond_embs = cond_embs / cond_embs.norm(dim=-1, keepdim=True)
        for i in range(n):
            dist = 1.0 - float(torch.dot(orig_embs[i], cond_embs[i]))
            clip_dists[cond].append(dist)

print(f"\n{'Condition':<15} {'CLIP dist':>10} {'n':>5}")
print("-" * 35)
clip_summary = {}
for cond in ["crf18", "crf38", "blur_s3", "blur_s6", "blur_s10", "framedrop2", "framedrop3", "framedrop4"]:
    d = clip_dists[cond]
    if d:
        mean_d = np.mean(d)
        print(f"{cond:<15} {mean_d:>10.4f} {len(d):>5}")
        clip_summary[cond] = round(mean_d, 4)


# ── 3. Temporal coherence (framedrop only) ─────────────────────────────────────
print("\n=== 3. Temporal coherence (frame-to-frame SSIM) ===")

temp_coherence = {}
for cond in ["original", "framedrop2", "framedrop3", "framedrop4"]:
    if cond == "original":
        cond_dir = VIDEO_DIR
    else:
        cond_dir = f"{ABLATION_BASE}/{cond}"

    f2f_ssims = []
    for vid in sample_vids[:20]:
        path = f"{cond_dir}/{vid}.mp4"
        if not os.path.exists(path):
            continue
        frames = extract_frames(path, n=10)
        for i in range(len(frames) - 1):
            f2f_ssims.append(compute_ssim(frames[i], frames[i+1]))

    if f2f_ssims:
        mean_tc = np.mean(f2f_ssims)
        temp_coherence[cond] = round(mean_tc, 4)
        print(f"  {cond:<15} frame-to-frame SSIM: {mean_tc:.4f}")


# ── 4. Shuffled sample validation ──────────────────────────────────────────────
print("\n=== 4. Shuffled sample validation ===")

with open(SHUFFLED_PATH) as f:
    shuffled = json.load(f)

orig_letter_dist = Counter(s["answer"] for s in shuffled)
shuf_letter_dist = Counter(s["shuffled_answer"] for s in shuffled)
n_changed = sum(1 for s in shuffled if s["shuffled_answer"] != s["answer"])

print(f"  Total questions: {len(shuffled)}")
print(f"  Correct letter changed: {n_changed}/{len(shuffled)} ({100*n_changed/len(shuffled):.1f}%)")
print(f"  Original answer distribution: {dict(orig_letter_dist)}")
print(f"  Shuffled answer distribution: {dict(shuf_letter_dist)}")


# ── Plot: degradation ladder ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

conds_ordered = ["crf18", "crf38", "blur_s3", "blur_s6", "blur_s10", "framedrop2", "framedrop3", "framedrop4"]
labels = ["CRF18", "CRF38", "Blur s3", "Blur s6", "Blur s10", "Drop 1/2", "Drop 1/3", "Drop 1/4"]
colors_map = {
    "crf18": "#4CAF50", "crf38": "#2E7D32",
    "blur_s3": "#2196F3", "blur_s6": "#1565C0", "blur_s10": "#0D47A1",
    "framedrop2": "#FF9800", "framedrop3": "#EF6C00", "framedrop4": "#E65100",
}

# SSIM bars
ssim_vals = [quality_summary.get(c, {}).get("ssim", 0) for c in conds_ordered]
bars1 = ax1.bar(labels, ssim_vals, color=[colors_map[c] for c in conds_ordered], edgecolor='black', linewidth=0.5)
ax1.set_ylabel("SSIM (vs original)")
ax1.set_title("Objective Quality: SSIM")
ax1.set_ylim(0.4, 1.05)
ax1.axhline(1.0, color='gray', linestyle=':', linewidth=1)
for bar, val in zip(bars1, ssim_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
             ha='center', va='bottom', fontsize=8, rotation=45)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# CLIP distance bars
clip_vals = [clip_summary.get(c, 0) for c in conds_ordered]
bars2 = ax2.bar(labels, clip_vals, color=[colors_map[c] for c in conds_ordered], edgecolor='black', linewidth=0.5)
ax2.set_ylabel("CLIP cosine distance (vs original)")
ax2.set_title("Perceptual Distance: CLIP ViT-B/32")
ax2.set_ylim(0, max(clip_vals) * 1.3 if clip_vals else 0.1)
for bar, val in zip(bars2, clip_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.4f}",
             ha='center', va='bottom', fontsize=8, rotation=45)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle("Degradation Ladder: Objective Quality vs Perceptual Distance", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ablation_degradation_ladder.png", dpi=150)
plt.close()
print(f"\nFig saved: {PLOTS_DIR}/ablation_degradation_ladder.png")


# ── Write report ──────────────────────────────────────────────────────────────
lines = []
lines.append("# Ablation Video Quality Analysis")
lines.append(f"**Videos sampled:** {len(sample_vids)}/{len(video_ids)}")
lines.append("")
lines.append("## Objective Quality (SSIM / PSNR)")
lines.append(f"| Condition | SSIM | PSNR (dB) |")
lines.append("|-----------|------|-----------|")
for c in conds_ordered:
    q = quality_summary.get(c, {})
    lines.append(f"| {c} | {q.get('ssim', '-')} | {q.get('psnr', '-')} |")
lines.append("")
lines.append("## CLIP Embedding Distance")
lines.append(f"| Condition | CLIP dist |")
lines.append("|-----------|-----------|")
for c in conds_ordered:
    lines.append(f"| {c} | {clip_summary.get(c, '-')} |")
lines.append("")
lines.append("## Temporal Coherence (frame-to-frame SSIM)")
for c, v in temp_coherence.items():
    lines.append(f"- {c}: {v}")
lines.append("")
lines.append("## Shuffled Sample")
lines.append(f"- Changed answer letter: {n_changed}/{len(shuffled)}")
lines.append(f"- Original distribution: {dict(orig_letter_dist)}")
lines.append(f"- Shuffled distribution: {dict(shuf_letter_dist)}")
lines.append("")
lines.append("## Pre-registered Predictions")
lines.append("Based on the quality measurements above:")
lines.append("1. If CLIP distance for blur_s10 >> CRF38, but inference shows both flat -> model ignores even severe spatial degradation")
lines.append("2. If CLIP distance for framedrop ≈ 0 -> model only sees per-frame content, temporal destruction invisible")
lines.append("3. If blur_s10 hurts accuracy but CRF38 doesn't -> model uses mid-frequency features that H.264 preserves but blur destroys")

report = "\n".join(lines)
with open(REPORT_PATH, "w") as f:
    f.write(report)
print(f"Report saved: {REPORT_PATH}")
print("\n=== DONE ===")
