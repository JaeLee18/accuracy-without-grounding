"""Recompute VGG and bootstrap 95% CIs for all 3 models on Video-MME."""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, numpy as np

def load(path):
    return json.load(open(path))

q  = load(VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json")
l  = load(VDG_RESULTS_ROOT + "/full_study/llava_results.json")
iv = load(VDG_RESULTS_ROOT + "/full_study/internvl2_results.json")
qb  = load(VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json")
lb  = load(VDG_RESULTS_ROOT + "/full_study/llava_black_results.json")
ivb = load(VDG_RESULTS_ROOT + "/full_study/internvl2_black_results.json")

def vgg_overall(orig_results, black_results):
    orig = {r["question_id"]: r for r in orig_results
            if r["condition"]=="original" and not r.get("error")}
    blk  = {r["question_id"]: r for r in black_results if not r.get("error")}
    common = sorted(set(orig) & set(blk))
    vgg_vals = [orig[q]["correct"] - blk[q]["correct"] for q in common]
    return np.array(vgg_vals), common

rng = np.random.default_rng(42)

for name, orig_res, blk_res in [("Qwen2-VL", q, qb), ("LLaVA-Video", l, lb), ("InternVL2", iv, ivb)]:
    vgg, common = vgg_overall(orig_res, blk_res)
    n = len(vgg)
    point = np.mean(vgg)
    boots = [np.mean(vgg[rng.integers(0, n, n)]) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{name}: n={n}, VGG={point:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")
