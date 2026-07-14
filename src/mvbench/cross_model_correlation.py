"""
Cross-model VGG correlation + black screen floor analysis.
Exclusion rule: any task type with fewer than 30 questions in the sample is excluded.
Final set: 9 task types (all n>=30, 7 with n=50, object_existence n=46, episodic_reasoning n=31/50).
Reports aggregate VGG and per-task table for paper.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import numpy as np
from collections import defaultdict
from scipy import stats

QWEN_PATH  = VDG_RESULTS_ROOT + "/mvbench/qwen2vl_mvbench_results.json"
LLAVA_PATH = VDG_RESULTS_ROOT + "/mvbench/llava_mvbench_results.json"
IV2_PATH   = VDG_RESULTS_ROOT + "/mvbench/internvl2_mvbench_results.json"
N_THRESHOLD = 30

TASK_TYPES_FULL = {
    "action_antonym", "action_prediction", "counterfactual_inference",
    "egocentric_navigation", "episodic_reasoning", "object_existence",
    "scene_transition", "state_change", "unexpected_action",
}

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def per_task_stats(results):
    by_tt = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("error"):
            continue
        by_tt[r["task_type"]][r["condition"]].append(int(r.get("correct", False)))
    out = {}
    for tt, conds in by_tt.items():
        n_orig = len(conds.get("original", []))
        if n_orig < N_THRESHOLD:
            continue
        out[tt] = {c: (sum(v)/len(v), len(v)) for c, v in conds.items()}
    return out

qwen  = load(QWEN_PATH)
llava = load(LLAVA_PATH)
iv2   = load(IV2_PATH)

q_stats  = per_task_stats(qwen)
l_stats  = per_task_stats(llava)
iv_stats = per_task_stats(iv2)

all_tasks = sorted(set(q_stats) | set(l_stats) | set(iv_stats))

print(f"Exclusion rule: n_original < {N_THRESHOLD} per task type")
print(f"Tasks remaining after exclusion: {len(all_tasks)}")
print(f"Full-sample tasks (n>=30): {len([t for t in all_tasks if t in TASK_TYPES_FULL])}\n")

hdr = f"  {'Task Type':<31} {'Q_orig':>7} {'Q_blk':>7} {'Q_VGG':>7} | {'L_orig':>7} {'L_blk':>7} {'L_VGG':>7} | {'IV_orig':>7} {'IV_blk':>7} {'IV_VGG':>7}"
print(hdr)
print("-" * len(hdr))

q_vgg_full, l_vgg_full, iv_vgg_full, tasks_full = [], [], [], []

for tt in all_tasks:
    q  = q_stats.get(tt, {})
    l  = l_stats.get(tt, {})
    iv = iv_stats.get(tt, {})

    def get_vgg(s):
        o = s.get("original", (None, 0))
        b = s.get("black",    (None, 0))
        vgg = (o[0] - b[0]) if (o[0] is not None and b[0] is not None) else None
        return o[0], b[0], vgg, o[1]

    q_o, q_b, q_vgg, n_q   = get_vgg(q)
    l_o, l_b, l_vgg, n_l   = get_vgg(l)
    iv_o, iv_b, iv_vgg, n_iv = get_vgg(iv)

    def fs(v): return f"{v:.3f}" if v is not None else "  N/A"

    print(f"  {tt:<31} {fs(q_o):>7} {fs(q_b):>7} {fs(q_vgg):>7} | "
          f"{fs(l_o):>7} {fs(l_b):>7} {fs(l_vgg):>7} | "
          f"{fs(iv_o):>7} {fs(iv_b):>7} {fs(iv_vgg):>7}")

    if tt in TASK_TYPES_FULL and q_vgg is not None and l_vgg is not None and iv_vgg is not None:
        q_vgg_full.append(q_vgg)
        l_vgg_full.append(l_vgg)
        iv_vgg_full.append(iv_vgg)
        tasks_full.append(tt)

# Pairwise and 3-way correlations
print(f"\n{'='*60}")
print("CROSS-MODEL VGG CORRELATION (3 models, full tasks)")
print(f"{'='*60}")

pairs = [
    ("Qwen vs LLaVA",     q_vgg_full,  l_vgg_full),
    ("Qwen vs InternVL2", q_vgg_full,  iv_vgg_full),
    ("LLaVA vs InternVL2",l_vgg_full,  iv_vgg_full),
]
for label, a, b in pairs:
    r_p, p_p = stats.pearsonr(np.array(a), np.array(b))
    r_s, p_s = stats.spearmanr(np.array(a), np.array(b))
    print(f"\n  {label} (n={len(tasks_full)})")
    print(f"    Pearson  r = {r_p:.3f}  (p = {p_p:.4f})")
    print(f"    Spearman r = {r_s:.3f}  (p = {p_s:.4f})")

# Black screen floor
print(f"\n{'='*60}")
print("BLACK SCREEN ACCURACY FLOOR")
print(f"{'='*60}")

for label, results in [("Qwen2-VL-7B", qwen), ("LLaVA-Video-7B", llava), ("InternVL2-8B", iv2)]:
    valid = [r for r in results if not r.get("error")]
    orig  = [r["correct"] for r in valid if r["condition"] == "original"]
    blk   = [r["correct"] for r in valid if r["condition"] == "black"]
    crf38 = [r["correct"] for r in valid if r["condition"] == "crf38"]
    print(f"\n  {label}:")
    print(f"    original = {np.mean(orig):.4f}  (n={len(orig)})")
    print(f"    black    = {np.mean(blk):.4f}  (n={len(blk)})")
    print(f"    crf38    = {np.mean(crf38):.4f}  (n={len(crf38)})")
    print(f"    VGG      = {np.mean(orig)-np.mean(blk):.4f}")

# Case study contrast
print(f"\n{'='*60}")
print("CASE STUDY: object_existence vs unexpected_action")
print(f"{'='*60}")

for tt in ["object_existence", "unexpected_action"]:
    print(f"\n  {tt}:")
    for label, results in [("Qwen2-VL", qwen), ("LLaVA-Video", llava), ("InternVL2", iv2)]:
        rows = [r for r in results if r["task_type"] == tt and not r.get("error")]
        for cond in ["original", "crf38", "black"]:
            c = [r["correct"] for r in rows if r["condition"] == cond]
            if c:
                print(f"    {label} {cond:<12}: {sum(c)}/{len(c)} = {sum(c)/len(c):.3f}")
