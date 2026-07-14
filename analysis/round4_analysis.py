"""
Round 4 analysis — InternVL2 MVBench complete. All 3 models.
1. Video-MME 3-model joint summary
2. McNemar tests on MVBench (3 pairs)
3. Cross-benchmark VGG divergence for all 3 models
4. Destructive ratio on MVBench
5. Bonferroni chance tests on MVBench (all 27 model x task cells)
6. Cohen's kappa on MVBench black screen
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.stats import binom, chi2_contingency
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def load_json(path):
    with open(path) as f:
        return json.load(f)

# ── Load all data ──────────────────────────────────────────────────────────
# Video-MME
q_vme   = load_json(VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json")
q_vblk  = load_json(VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json")
l_vme   = load_json(VDG_RESULTS_ROOT + "/full_study/llava_results.json")
l_vblk  = load_json(VDG_RESULTS_ROOT + "/full_study/llava_black_results.json")
iv_vme  = load_json(VDG_RESULTS_ROOT + "/full_study/internvl2_results.json")
iv_vblk = load_json(VDG_RESULTS_ROOT + "/full_study/internvl2_black_results.json")

# MVBench
q_mv  = load_json(VDG_RESULTS_ROOT + "/mvbench/qwen2vl_mvbench_results.json")
l_mv  = load_json(VDG_RESULTS_ROOT + "/mvbench/llava_mvbench_results.json")
iv_mv = load_json(VDG_RESULTS_ROOT + "/mvbench/internvl2_mvbench_results.json")

TASK_TYPES_MV = [
    "action_antonym", "action_prediction", "counterfactual_inference",
    "egocentric_navigation", "episodic_reasoning", "object_existence",
    "scene_transition", "state_change", "unexpected_action"
]
TASK_TYPES_VME = [
    "Action Reasoning", "Action Recognition", "Attribute Perception",
    "OCR Problems", "Object Recognition", "Temporal Reasoning"
]

print("=" * 70)
print("ROUND 4 ANALYSIS — InternVL2 Complete, All 3 Models")
print("=" * 70)

# ── 1. Video-MME 3-model joint summary ────────────────────────────────────
print("\n" + "─"*60)
print("1. VIDEO-MME 3-MODEL SUMMARY")
print("─"*60)

def vmme_stats(results_list, black_list):
    orig = {r["question_id"]: r for r in results_list if r["condition"]=="original" and not r.get("error")}
    blk  = {r["question_id"]: r for r in black_list if not r.get("error")}
    return orig, blk

q_vo,  q_vb  = vmme_stats(q_vme,  q_vblk)
l_vo,  l_vb  = vmme_stats(l_vme,  l_vblk)
iv_vo, iv_vb = vmme_stats(iv_vme, iv_vblk)

print(f"\n{'Model':<14} {'Orig acc':>9} {'Black acc':>10} {'VGG':>7}  n_orig  n_black")
for name, orig, blk in [("Qwen2-VL", q_vo, q_vb), ("LLaVA", l_vo, l_vb), ("InternVL2", iv_vo, iv_vb)]:
    common = set(orig) & set(blk)
    oa = np.mean([int(orig[qid]["correct"]) for qid in common])
    ba = np.mean([int(blk[qid]["correct"])  for qid in common])
    print(f"  {name:<12} {oa:>9.3f} {ba:>10.3f} {oa-ba:>7.3f}  {len(orig):>6}  {len(blk):>7}")

print("\nPer-task-type breakdown:")
print(f"  {'Task Type':<30} {'Q_VGG':>7} {'L_VGG':>7} {'IV_VGG':>7}")
for tt in TASK_TYPES_VME:
    def tt_vgg(orig, blk):
        qids = [qid for qid in set(orig)&set(blk) if orig[qid]["task_type"]==tt]
        if not qids: return None
        return np.mean([int(orig[qid]["correct"]) for qid in qids]) - np.mean([int(blk[qid]["correct"]) for qid in qids])
    qv = tt_vgg(q_vo, q_vb); lv = tt_vgg(l_vo, l_vb); iv = tt_vgg(iv_vo, iv_vb)
    def fs(v): return f"{v:.3f}" if v is not None else "  N/A"
    print(f"  {tt:<30} {fs(qv):>7} {fs(lv):>7} {fs(iv):>7}")

# ── 2. McNemar tests on MVBench (3 pairs) ─────────────────────────────────
print("\n" + "─"*60)
print("2. McNEMAR TESTS — MVBench black screen (3 model pairs)")
print("─"*60)

def mv_black_dict(mv_data):
    return {r["question_id"]: r for r in mv_data if r["condition"]=="black" and not r.get("error")}

def mv_orig_dict(mv_data):
    return {r["question_id"]: r for r in mv_data if r["condition"]=="original" and not r.get("error")}

q_mb  = mv_black_dict(q_mv);  q_mo  = mv_orig_dict(q_mv)
l_mb  = mv_black_dict(l_mv);  l_mo  = mv_orig_dict(l_mv)
iv_mb = mv_black_dict(iv_mv); iv_mo = mv_orig_dict(iv_mv)

# Also original accuracy comparison
print("\nMVBench original accuracy:")
for name, od in [("Qwen2-VL", q_mo), ("LLaVA", l_mo), ("InternVL2", iv_mo)]:
    acc = np.mean([int(od[qid]["correct"]) for qid in od])
    print(f"  {name:<12} orig={acc:.3f}  n={len(od)}")

# McNemar on black screen
print("\nMcNemar tests on MVBench black screen accuracy:")
pairs_mv = [
    ("Qwen2-VL vs LLaVA",      q_mb, l_mb),
    ("Qwen2-VL vs InternVL2",  q_mb, iv_mb),
    ("LLaVA vs InternVL2",     l_mb, iv_mb),
]
for label, a, b in pairs_mv:
    common = set(a) & set(b)
    # McNemar contingency: (both wrong, a right b wrong, a wrong b right, both right)
    b00 = sum(1 for qid in common if not a[qid]["correct"] and not b[qid]["correct"])
    b01 = sum(1 for qid in common if not a[qid]["correct"] and     b[qid]["correct"])
    b10 = sum(1 for qid in common if     a[qid]["correct"] and not b[qid]["correct"])
    b11 = sum(1 for qid in common if     a[qid]["correct"] and     b[qid]["correct"])
    acc_a = np.mean([int(a[qid]["correct"]) for qid in common])
    acc_b = np.mean([int(b[qid]["correct"]) for qid in common])
    # McNemar statistic (with continuity correction)
    if b01 + b10 > 0:
        chi2_mc = (abs(b01 - b10) - 1)**2 / (b01 + b10)
        p_mc = 1 - stats.chi2.cdf(chi2_mc, 1)
    else:
        chi2_mc, p_mc = 0, 1.0
    print(f"\n  {label}")
    print(f"    acc_A={acc_a:.3f}  acc_B={acc_b:.3f}  diff={acc_a-acc_b:+.3f}")
    print(f"    concordant: {b11} both right, {b00} both wrong")
    print(f"    discordant: {b10} A-only, {b01} B-only")
    print(f"    McNemar chi2={chi2_mc:.3f}, p={p_mc:.4f}")

# Also McNemar on original accuracy
print("\nMcNemar tests on MVBench ORIGINAL accuracy:")
pairs_orig = [
    ("Qwen2-VL vs LLaVA",      q_mo, l_mo),
    ("Qwen2-VL vs InternVL2",  q_mo, iv_mo),
    ("LLaVA vs InternVL2",     l_mo, iv_mo),
]
for label, a, b in pairs_orig:
    common = set(a) & set(b)
    b01 = sum(1 for qid in common if not a[qid]["correct"] and     b[qid]["correct"])
    b10 = sum(1 for qid in common if     a[qid]["correct"] and not b[qid]["correct"])
    acc_a = np.mean([int(a[qid]["correct"]) for qid in common])
    acc_b = np.mean([int(b[qid]["correct"]) for qid in common])
    if b01 + b10 > 0:
        chi2_mc = (abs(b01 - b10) - 1)**2 / (b01 + b10)
        p_mc = 1 - stats.chi2.cdf(chi2_mc, 1)
    else:
        chi2_mc, p_mc = 0, 1.0
    print(f"  {label}: orig_A={acc_a:.3f} orig_B={acc_b:.3f} McNemar p={p_mc:.4f}")

# ── 3. Cross-benchmark VGG divergence for all 3 models ────────────────────
print("\n" + "─"*60)
print("3. CROSS-BENCHMARK VGG DIVERGENCE (Video-MME vs MVBench)")
print("─"*60)

# MVBench overall VGG
def mv_vgg(mv_data):
    orig = {r["question_id"]: r for r in mv_data if r["condition"]=="original" and not r.get("error")}
    blk  = {r["question_id"]: r for r in mv_data if r["condition"]=="black"    and not r.get("error")}
    common = set(orig) & set(blk)
    return np.mean([int(orig[qid]["correct"]) for qid in common]) - \
           np.mean([int(blk[qid]["correct"])  for qid in common])

# Video-MME overall VGG
def vme_vgg(orig_d, blk_d):
    common = set(orig_d) & set(blk_d)
    return np.mean([int(orig_d[qid]["correct"]) for qid in common]) - \
           np.mean([int(blk_d[qid]["correct"])  for qid in common])

print(f"\n{'Model':<14} {'VME VGG':>9} {'MV VGG':>9} {'Abs diff':>10} {'Rel diff':>10}")
for name, orig_d, blk_d, mv_data in [
    ("Qwen2-VL",  q_vo,  q_vb,  q_mv),
    ("LLaVA",     l_vo,  l_vb,  l_mv),
    ("InternVL2", iv_vo, iv_vb, iv_mv),
]:
    vme = vme_vgg(orig_d, blk_d)
    mv  = mv_vgg(mv_data)
    abs_d = abs(vme - mv)
    rel_d = abs_d / vme * 100
    print(f"  {name:<12} {vme:>9.3f} {mv:>9.3f} {abs_d:>10.3f} {rel_d:>9.1f}%")

# ── 4. Destructive ratio on MVBench ───────────────────────────────────────
print("\n" + "─"*60)
print("4. DESTRUCTIVE RATIO — MVBench (correct-to-wrong vs wrong-to-correct)")
print("─"*60)

for name, orig_d, blk_d in [("Qwen2-VL", q_mo, q_mb), ("LLaVA", l_mo, l_mb), ("InternVL2", iv_mo, iv_mb)]:
    common = set(orig_d) & set(blk_d)
    # correct with video, wrong without (destructive)
    destructive = sum(1 for qid in common if orig_d[qid]["correct"] and not blk_d[qid]["correct"])
    # wrong with video, correct without (constructive)
    constructive = sum(1 for qid in common if not orig_d[qid]["correct"] and blk_d[qid]["correct"])
    ratio = destructive / constructive if constructive > 0 else float("inf")
    print(f"  {name:<12} destructive={destructive:>4}  constructive={constructive:>4}  ratio={ratio:.2f}:1")

# ── 5. Bonferroni chance tests on MVBench (27 cells: 3 models x 9 tasks) ─
print("\n" + "─"*60)
print("5. BONFERRONI-CORRECTED CHANCE TESTS — MVBench black screen")
print("─"*60)

n_tests = 3 * len(TASK_TYPES_MV)  # 27
results_bonf = []
print(f"\n{'Model':<12} {'Task Type':<30} {'n':>4} {'acc':>6} {'p_raw':>8} {'p_bonf':>8} {'sig':>4}")

for name, mv_data in [("Qwen2-VL", q_mv), ("LLaVA", l_mv), ("InternVL2", iv_mv)]:
    for tt in TASK_TYPES_MV:
        entries = [r for r in mv_data if r["task_type"]==tt and r["condition"]=="black" and not r.get("error")]
        if not entries: continue
        # determine chance level from number of options
        n_opts = len(set(r.get("ground_truth","A") for r in entries if r.get("ground_truth")))
        # check options count from a sample
        sample = next((r for r in mv_data if r["task_type"]==tt and r["condition"]=="original"), None)
        # Use 25% for 4-option, 20% for 5-option (episodic_reasoning)
        chance = 0.20 if tt == "episodic_reasoning" else 0.25
        n = len(entries)
        k = sum(1 for r in entries if r.get("correct"))
        p_raw = binom.sf(k - 1, n, chance)
        p_bonf = min(p_raw * n_tests, 1.0)
        sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "ns"))
        acc = k / n
        results_bonf.append({"model": name, "task": tt, "n": n, "acc": acc,
                              "p_raw": p_raw, "p_bonf": p_bonf, "sig": sig})
        print(f"  {name:<10} {tt:<30} {n:>4} {acc:>6.3f} {p_raw:>8.4f} {p_bonf:>8.4f} {sig:>4}")

n_sig = sum(1 for r in results_bonf if r["p_bonf"] < 0.05)
print(f"\nSignificant after Bonferroni: {n_sig}/{len(results_bonf)}")
print("\nNot significant cells:")
for r in results_bonf:
    if r["p_bonf"] >= 0.05:
        print(f"  {r['model']:<12} {r['task']:<30} acc={r['acc']:.3f} p_bonf={r['p_bonf']:.3f}")

# ── 6. Cohen's kappa on MVBench black screen ──────────────────────────────
print("\n" + "─"*60)
print("6. COHEN'S KAPPA — MVBench black screen inter-model agreement")
print("─"*60)

def cohen_kappa(a_dict, b_dict):
    common = set(a_dict) & set(b_dict)
    if not common: return None
    agree = sum(1 for qid in common if a_dict[qid]["correct"] == b_dict[qid]["correct"])
    po = agree / len(common)
    # Expected agreement
    pa1 = np.mean([int(a_dict[qid]["correct"]) for qid in common])
    pb1 = np.mean([int(b_dict[qid]["correct"]) for qid in common])
    pe = pa1*pb1 + (1-pa1)*(1-pb1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0
    return kappa, len(common)

print("\nMVBench black screen kappa:")
for label, a, b in [("Qwen vs LLaVA", q_mb, l_mb), ("Qwen vs InternVL2", q_mb, iv_mb), ("LLaVA vs InternVL2", l_mb, iv_mb)]:
    result = cohen_kappa(a, b)
    if result:
        k, n = result
        print(f"  {label:<25} kappa={k:.3f}  n={n}")

print("\nVideo-MME black screen kappa (for comparison):")
for label, a, b in [("Qwen vs LLaVA", q_vb, l_vb), ("Qwen vs InternVL2", q_vb, iv_vb), ("LLaVA vs InternVL2", l_vb, iv_vb)]:
    result = cohen_kappa(a, b)
    if result:
        k, n = result
        print(f"  {label:<25} kappa={k:.3f}  n={n}")

# ── BONUS: 3-model MVBench dissociation test (like F3 for Video-MME) ──────
print("\n" + "─"*60)
print("BONUS: MVBench McNemar dissociation (same original acc, diff black acc?)")
print("─"*60)

# Check if any pair has same original but different black
for label, ao, ab, bo, bb in [
    ("Qwen vs LLaVA",      q_mo, q_mb, l_mo, l_mb),
    ("Qwen vs InternVL2",  q_mo, q_mb, iv_mo, iv_mb),
    ("LLaVA vs InternVL2", l_mo, l_mb, iv_mo, iv_mb),
]:
    common = set(ao) & set(bo)
    common_b = set(ab) & set(bb)
    # McNemar on original
    b01_o = sum(1 for qid in common if not ao[qid]["correct"] and bo[qid]["correct"])
    b10_o = sum(1 for qid in common if     ao[qid]["correct"] and not bo[qid]["correct"])
    chi2_o = (abs(b01_o-b10_o)-1)**2/(b01_o+b10_o) if (b01_o+b10_o)>0 else 0
    p_orig = 1 - stats.chi2.cdf(chi2_o, 1)
    # McNemar on black
    b01_b = sum(1 for qid in common_b if not ab[qid]["correct"] and bb[qid]["correct"])
    b10_b = sum(1 for qid in common_b if     ab[qid]["correct"] and not bb[qid]["correct"])
    chi2_b = (abs(b01_b-b10_b)-1)**2/(b01_b+b10_b) if (b01_b+b10_b)>0 else 0
    p_black = 1 - stats.chi2.cdf(chi2_b, 1)
    print(f"\n  {label}")
    print(f"    original: p={p_orig:.4f} ({'sig' if p_orig<0.05 else 'ns'})")
    print(f"    black:    p={p_black:.4f} ({'sig' if p_black<0.05 else 'ns'})")
    dissoc = "DISSOCIATION" if p_orig >= 0.05 and p_black < 0.05 else ""
    if dissoc: print(f"    *** {dissoc} ***")

print("\n" + "=" * 70)
print("ROUND 4 COMPLETE")
print("=" * 70)
