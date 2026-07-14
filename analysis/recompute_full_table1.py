"""Recompute all Table 1 numbers: per-task VGG, orig acc, black acc, CIs for all 3 models."""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, numpy as np
from collections import defaultdict

def load(path):
    return json.load(open(path))

q  = load(VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json")
l  = load(VDG_RESULTS_ROOT + "/full_study/llava_results.json")
iv = load(VDG_RESULTS_ROOT + "/full_study/internvl2_results.json")
qb  = load(VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json")
lb  = load(VDG_RESULTS_ROOT + "/full_study/llava_black_results.json")
ivb = load(VDG_RESULTS_ROOT + "/full_study/internvl2_black_results.json")

rng = np.random.default_rng(42)

def make_dicts(orig_res, blk_res):
    orig = {r["question_id"]: r for r in orig_res
            if r["condition"]=="original" and not r.get("error")}
    blk  = {r["question_id"]: r for r in blk_res if not r.get("error")}
    return orig, blk

def task_stats(orig_d, blk_d):
    tasks = defaultdict(list)
    common = set(orig_d) & set(blk_d)
    for qid in common:
        tt = orig_d[qid].get("task_type","unknown")
        tasks[tt].append((orig_d[qid]["correct"], blk_d[qid]["correct"]))
    return tasks

def ci(arr, n_boot=2000):
    a = np.array(arr)
    n = len(a)
    boots = [np.mean(a[rng.integers(0,n,n)]) for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])

models = [("Qwen2-VL", q, qb), ("LLaVA-Video", l, lb), ("InternVL2", iv, ivb)]

# Overall VGG with correct CIs
print("=== OVERALL VGG ===")
for name, orig_r, blk_r in models:
    od, bd = make_dicts(orig_r, blk_r)
    common = sorted(set(od)&set(bd))
    vgg = [od[q]["correct"]-bd[q]["correct"] for q in common]
    lo, hi = ci(vgg)
    print(f"{name}: n={len(common)}, VGG={np.mean(vgg):.4f}, 95%CI=[{lo:.4f},{hi:.4f}]")

# Per-task-type VGG
print("\n=== PER-TASK VGG (Video-MME) ===")
# Collect all task types
all_tasks = set()
for name, orig_r, blk_r in models:
    od, bd = make_dicts(orig_r, blk_r)
    ts = task_stats(od, bd)
    all_tasks.update(ts.keys())

task_order = sorted(all_tasks)

for tt in task_order:
    row = [tt]
    for name, orig_r, blk_r in models:
        od, bd = make_dicts(orig_r, blk_r)
        ts = task_stats(od, bd)
        if tt not in ts or len(ts[tt]) < 5:
            row.append("N/A")
            continue
        pairs = ts[tt]
        orig_acc = np.mean([p[0] for p in pairs])
        blk_acc  = np.mean([p[1] for p in pairs])
        vgg_val  = orig_acc - blk_acc
        vgg_arr  = [p[0]-p[1] for p in pairs]
        lo, hi   = ci(vgg_arr)
        row.append(f"VGG={vgg_val:.3f}[{lo:.3f},{hi:.3f}] orig={orig_acc:.3f} blk={blk_acc:.3f} n={len(pairs)}")
    print(f"\n{tt}:")
    for name, val in zip([m[0] for m in models], row[1:]):
        print(f"  {name}: {val}")
