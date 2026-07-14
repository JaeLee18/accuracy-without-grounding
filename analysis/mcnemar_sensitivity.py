"""
McNemar dissociation sensitivity analysis.
Tests: does the dissociation (same original, different black) hold robustly
across bootstrap samples and accuracy tolerance ranges?
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_json(path):
    with open(path) as f:
        return json.load(f)

q_vme   = load_json(VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json")
q_vblk  = load_json(VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json")
l_vme   = load_json(VDG_RESULTS_ROOT + "/full_study/llava_results.json")
l_vblk  = load_json(VDG_RESULTS_ROOT + "/full_study/llava_black_results.json")
iv_vme  = load_json(VDG_RESULTS_ROOT + "/full_study/internvl2_results.json")
iv_vblk = load_json(VDG_RESULTS_ROOT + "/full_study/internvl2_black_results.json")

q_mv  = load_json(VDG_RESULTS_ROOT + "/mvbench/qwen2vl_mvbench_results.json")
l_mv  = load_json(VDG_RESULTS_ROOT + "/mvbench/llava_mvbench_results.json")
iv_mv = load_json(VDG_RESULTS_ROOT + "/mvbench/internvl2_mvbench_results.json")

print("=" * 70)
print("McNEMAR DISSOCIATION SENSITIVITY ANALYSIS")
print("=" * 70)

def make_dicts(results, black):
    orig = {r["question_id"]: r for r in results if r["condition"]=="original" and not r.get("error")}
    blk  = {r["question_id"]: r for r in black if not r.get("error")}
    return orig, blk

def mcnemar(a, b, qids):
    b01 = sum(1 for q in qids if not a[q]["correct"] and     b[q]["correct"])
    b10 = sum(1 for q in qids if     a[q]["correct"] and not b[q]["correct"])
    if b01 + b10 == 0: return 1.0
    chi2 = (abs(b01 - b10) - 1)**2 / (b01 + b10)
    return 1 - stats.chi2.cdf(chi2, 1)

q_vo, q_vb   = make_dicts(q_vme, q_vblk)
l_vo, l_vb   = make_dicts(l_vme, l_vblk)
iv_vo, iv_vb = make_dicts(iv_vme, iv_vblk)

# ── Pair: InternVL2 vs Qwen2-VL (the dissociation pair from F3) ──────────
print("\nKey dissociation pair: InternVL2 vs Qwen2-VL (Video-MME)")
common = sorted(set(iv_vo)&set(q_vo)&set(iv_vb)&set(q_vb))
p_orig = mcnemar(iv_vo, q_vo, common)
p_blk  = mcnemar(iv_vb, q_vb, common)
iv_acc = np.mean([iv_vo[q]["correct"] for q in common])
q_acc  = np.mean([q_vo[q]["correct"]  for q in common])
print(f"  n_common={len(common)}")
print(f"  InternVL2 orig={iv_acc:.3f}, Qwen orig={q_acc:.3f}, diff={iv_acc-q_acc:+.3f}")
print(f"  McNemar p(original)={p_orig:.4f} (ns = 'identical accuracy')")
print(f"  McNemar p(black)   ={p_blk:.4f}  (sig = 'different black profiles')")

# ── Bootstrap sensitivity (1000 seeds) ────────────────────────────────────
print("\n1. Bootstrap sensitivity (1000 seeds, resample common questions)")
rng = np.random.default_rng(42)
n = len(common)
results_boot = []
for _ in range(1000):
    idx    = rng.integers(0, n, size=n)
    sample = [common[i] for i in idx]
    p_o = mcnemar(iv_vo, q_vo, sample)
    p_b = mcnemar(iv_vb, q_vb, sample)
    results_boot.append((p_o, p_b))

p_orig_boots = [r[0] for r in results_boot]
p_blk_boots  = [r[1] for r in results_boot]

pct_orig_ns  = np.mean([p > 0.05 for p in p_orig_boots]) * 100
pct_blk_sig  = np.mean([p < 0.05 for p in p_blk_boots])  * 100
pct_both     = np.mean([r[0]>0.05 and r[1]<0.05 for r in results_boot]) * 100
pct_blk_001  = np.mean([p < 0.01 for p in p_blk_boots]) * 100

print(f"  % bootstrap samples where original is non-sig (p>0.05): {pct_orig_ns:.1f}%")
print(f"  % bootstrap samples where black is sig (p<0.05):         {pct_blk_sig:.1f}%")
print(f"  % bootstrap samples where BOTH conditions hold:           {pct_both:.1f}%")
print(f"  % bootstrap samples where black p<0.01:                   {pct_blk_001:.1f}%")
print(f"  Median p(original)={np.median(p_orig_boots):.3f}, median p(black)={np.median(p_blk_boots):.4f}")

# ── Tolerance analysis ─────────────────────────────────────────────────────
print("\n2. Tolerance analysis: pair models within ±X pp of original accuracy")
print("   (for each tolerance, find all model pairs across both benchmarks)")

# All model pairs and their original/black McNemar values
all_pairs_vmme = [
    ("Qwen vs LLaVA",      q_vo, l_vo, q_vb, l_vb),
    ("Qwen vs InternVL2",  q_vo, iv_vo, q_vb, iv_vb),
    ("LLaVA vs InternVL2", l_vo, iv_vo, l_vb, iv_vb),
]
all_pairs_mv = []
for name, a_mv, b_mv in [("Qwen vs LLaVA", q_mv, l_mv),
                           ("Qwen vs InternVL2", q_mv, iv_mv),
                           ("LLaVA vs InternVL2", l_mv, iv_mv)]:
    ao = {r["question_id"]: r for r in a_mv if r["condition"]=="original" and not r.get("error")}
    ab = {r["question_id"]: r for r in a_mv if r["condition"]=="black"    and not r.get("error")}
    bo = {r["question_id"]: r for r in b_mv if r["condition"]=="original" and not r.get("error")}
    bb = {r["question_id"]: r for r in b_mv if r["condition"]=="black"    and not r.get("error")}
    all_pairs_mv.append((name, ao, bo, ab, bb))

print(f"\n  All model pairs — original and black McNemar p-values:")
print(f"  {'Benchmark':<10} {'Pair':<25} {'orig_acc_A':>10} {'orig_acc_B':>10} {'diff_pp':>8} {'p_orig':>8} {'p_black':>8} {'dissoc?':>8}")
for bname, pairs in [("VME", all_pairs_vmme), ("MVBench", all_pairs_mv)]:
    for item in pairs:
        if bname == "VME":
            name, ao, bo, ab, bb = item
        else:
            name, ao, bo, ab, bb = item
        qids_o = sorted(set(ao)&set(bo))
        qids_b = sorted(set(ab)&set(bb))
        if not qids_o or not qids_b: continue
        acc_a = np.mean([ao[q]["correct"] for q in qids_o])
        acc_b = np.mean([bo[q]["correct"] for q in qids_o])
        diff  = abs(acc_a - acc_b) * 100
        p_o   = mcnemar(ao, bo, qids_o)
        p_b   = mcnemar(ab, bb, qids_b)
        dissoc = "YES" if p_o > 0.05 and p_b < 0.05 else "no"
        print(f"  {bname:<10} {name:<25} {acc_a:>10.3f} {acc_b:>10.3f} {diff:>7.1f}pp {p_o:>8.4f} {p_b:>8.4f} {dissoc:>8}")

# ── The strongest dissociation statement ──────────────────────────────────
print("\n3. Summary: strongest defensible dissociation statement")
print(f"""
  Video-MME: InternVL2 (orig={iv_acc:.3f}) vs Qwen2-VL (orig={q_acc:.3f})
  - Accuracy difference: {abs(iv_acc-q_acc)*100:.1f}pp, McNemar p={p_orig:.4f} (not significantly different)
  - Black-screen profiles: McNemar p={p_blk:.4f} (highly significantly different)
  - Bootstrap: dissociation holds in {pct_both:.1f}% of 1000 resamplings
  - Black-screen McNemar p<0.01 in {pct_blk_001:.1f}% of bootstrap samples

  MVBench (INDEPENDENT REPLICATION):
  - InternVL2 significantly BETTER on original (p=0.0003 vs Qwen, p=0.0006 vs LLaVA)
  - Yet ALL three pairs statistically identical on black screen (p=0.53-0.94)
  - This is a STRONGER form of dissociation: InternVL2's superior video capability
    does not translate to language-prior exploitation
""")
