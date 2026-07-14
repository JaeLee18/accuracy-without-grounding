"""
Round 3 analysis targeting reviewer's remaining 4 gaps:
Gap A: Error-rate audit across all model x task x CRF cells
Gap B: GT letter D vs A HC-rate with sample sizes and chi-square
Gap C: Predict mixed-sign vs HC vs non-positive category (logistic regression)
Gap D: Verify GT label distribution for scene_transition and unexpected_action
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Video-MME results
q_results  = load_json(VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json")
q_black    = load_json(VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json")
l_results  = load_json(VDG_RESULTS_ROOT + "/full_study/llava_results.json")
l_black    = load_json(VDG_RESULTS_ROOT + "/full_study/llava_black_results.json")
iv_results = load_json(VDG_RESULTS_ROOT + "/full_study/internvl2_results.json")
iv_black   = load_json(VDG_RESULTS_ROOT + "/full_study/internvl2_black_results.json")

# MVBench results
q_mv  = load_json(VDG_RESULTS_ROOT + "/mvbench/qwen2vl_mvbench_results.json")
l_mv  = load_json(VDG_RESULTS_ROOT + "/mvbench/llava_mvbench_results.json")
iv_mv = load_json(VDG_RESULTS_ROOT + "/mvbench/internvl2_mvbench_results.json")

print("=" * 70)
print("ROUND 3 ANALYSIS — 4 REMAINING GAPS")
print("=" * 70)

# ── GAP A: Error-rate audit across all model x task x CRF cells ────────────
print("\n" + "─"*60)
print("GAP A: Error-rate audit (all model x task x CRF for Video-MME)")
print("─"*60)

task_types = sorted(set(r["task_type"] for r in q_results if not r.get("error")))
conditions_vmme = ["original", "crf18", "crf23", "crf28", "crf33", "crf38"]

models_vmme = {
    "Qwen2-VL":  q_results,
    "LLaVA":     l_results,
    "InternVL2": iv_results,
}

# Build error rate table
print(f"\n{'Model':<12} {'Task Type':<30} {'cond':>8} {'n_total':>8} {'n_error':>8} {'err%':>6}")
max_err_cells = []
for model_name, results in models_vmme.items():
    for tt in task_types:
        for cond in conditions_vmme:
            entries = [r for r in results if r["task_type"]==tt and r["condition"]==cond]
            n_total = len(entries)
            n_error = sum(1 for r in entries if r.get("error"))
            err_pct = n_error/n_total*100 if n_total else 0
            if err_pct > 5:  # only print high-error cells
                print(f"  {model_name:<10} {tt:<30} {cond:>8} {n_total:>8} {n_error:>8} {err_pct:>5.1f}%")
                max_err_cells.append((model_name, tt, cond, err_pct))

if not max_err_cells:
    print("  No cells with error rate >5% in Video-MME. CRF analysis is clean.")
else:
    print(f"\nCells with >5% error rate: {len(max_err_cells)}")

# Check MVBench error rates
print("\nMVBench error-rate audit (cells >5%):")
models_mv = {
    "Qwen2-VL":  q_mv,
    "LLaVA":     l_mv,
    "InternVL2": iv_mv,
}
TASK_TYPES_FULL = [
    "action_antonym", "action_prediction", "counterfactual_inference",
    "egocentric_navigation", "episodic_reasoning", "object_existence",
    "scene_transition", "state_change", "unexpected_action"
]
mv_conditions = ["original", "crf38", "black"]
any_high_mv = False
for model_name, mv_data in models_mv.items():
    for tt in TASK_TYPES_FULL:
        for cond in mv_conditions:
            entries = [r for r in mv_data if r["task_type"]==tt and r["condition"]==cond]
            n_total = len(entries)
            n_error = sum(1 for r in entries if r.get("error"))
            err_pct = n_error/n_total*100 if n_total else 0
            if err_pct > 5:
                any_high_mv = True
                print(f"  {model_name:<10} {tt:<30} {cond:>8} n={n_total:>3} err={n_error:>3} ({err_pct:.1f}%)")

if not any_high_mv:
    print("  No cells with error rate >5% in MVBench main tasks.")

# Also check ALL MVBench task types for action_antonym specifically
print("\naction_antonym CRF38 detailed audit:")
for model_name, mv_data in models_mv.items():
    aa_crf = [r for r in mv_data if r["task_type"]=="action_antonym" and r["condition"]=="crf38"]
    aa_err = [r for r in aa_crf if r.get("error")]
    aa_ok  = [r for r in aa_crf if not r.get("error")]
    print(f"  {model_name:<10}: total={len(aa_crf)}, errors={len(aa_err)}, valid={len(aa_ok)}")
    if aa_err:
        # Sample error messages
        errs = set(r["error"][:80] for r in aa_err[:3])
        for e in errs:
            print(f"    Error example: {e}")

# ── GAP B: GT letter D vs A HC-rate with statistics ────────────────────────
print("\n" + "─"*60)
print("GAP B: GT letter D vs A hard-core rate with statistics")
print("─"*60)

def build_vmme_dicts(results_list, black_list):
    orig = {r["question_id"]: r for r in results_list if r["condition"]=="original" and not r.get("error")}
    black = {r["question_id"]: r for r in black_list if not r.get("error")}
    return orig, black

q_orig, q_blk   = build_vmme_dicts(q_results, q_black)
l_orig, l_blk   = build_vmme_dicts(l_results, l_black)
iv_orig, iv_blk = build_vmme_dicts(iv_results, iv_black)

common_qids = set(q_orig) & set(l_orig) & set(iv_orig) & set(q_blk) & set(l_blk) & set(iv_blk)

def vgg_per_q(orig_d, blk_d, qids):
    return {qid: int(orig_d[qid]["correct"]) - int(blk_d[qid]["correct"]) for qid in qids}

q_vgg_q  = vgg_per_q(q_orig, q_blk, common_qids)
l_vgg_q  = vgg_per_q(l_orig, l_blk, common_qids)
iv_vgg_q = vgg_per_q(iv_orig, iv_blk, common_qids)

hard_core = {qid for qid in common_qids
             if q_vgg_q[qid]>0 and l_vgg_q[qid]>0 and iv_vgg_q[qid]>0}

# GT letter analysis with full contingency table
gt_letters = "ABCD"
letter_counts = {let: [] for let in gt_letters}  # HC=1, non-HC=0

for qid in common_qids:
    gt = q_orig[qid]["ground_truth"]
    if gt in gt_letters:
        letter_counts[gt].append(1 if qid in hard_core else 0)

print(f"\n{'GT Letter':<12} {'HC':>5} {'non-HC':>7} {'Total':>6} {'HC%':>7}")
for let in gt_letters:
    data = letter_counts[let]
    n_hc   = sum(data)
    n_nonhc = len(data) - n_hc
    n_total = len(data)
    pct = n_hc/n_total*100 if n_total else 0
    print(f"  {let:<10} {n_hc:>5} {n_nonhc:>7} {n_total:>6} {pct:>6.1f}%")

# 2x4 contingency table: HC vs non-HC by letter
ct = np.array([[sum(letter_counts[let]) for let in gt_letters],
               [len(letter_counts[let]) - sum(letter_counts[let]) for let in gt_letters]])
print(f"\nContingency table (HC/non-HC by GT letter):\n  HC:     {ct[0]}\n  non-HC: {ct[1]}")
chi2, p, dof, expected = chi2_contingency(ct)
print(f"Chi-square: chi2={chi2:.3f}, p={p:.4f}, dof={dof}")

# Direct 2x2 test: D vs A
a_hc = sum(letter_counts["A"]); a_n = len(letter_counts["A"])
d_hc = sum(letter_counts["D"]); d_n = len(letter_counts["D"])
ct_ad = np.array([[a_hc, a_n-a_hc], [d_hc, d_n-d_hc]])
chi2_ad, p_ad, _, _ = chi2_contingency(ct_ad)
print(f"\nA vs D 2x2 chi-square: chi2={chi2_ad:.3f}, p={p_ad:.4f}")
print(f"  A: {a_hc}/{a_n} = {a_hc/a_n*100:.1f}%  D: {d_hc}/{d_n} = {d_hc/d_n*100:.1f}%")

# ── GAP C: Predict mixed-sign vs HC vs non-positive ───────────────────────
print("\n" + "─"*60)
print("GAP C: Predicting VGG sign category from question properties")
print("─"*60)

# Categories: 0=all non-positive, 1=mixed, 2=HC (all positive)
cat_map = {}
for qid in common_qids:
    q_pos  = q_vgg_q[qid] > 0
    l_pos  = l_vgg_q[qid] > 0
    iv_pos = iv_vgg_q[qid] > 0
    n_pos  = sum([q_pos, l_pos, iv_pos])
    if n_pos == 3:
        cat_map[qid] = 2  # HC
    elif n_pos == 0:
        cat_map[qid] = 0  # all non-positive
    else:
        cat_map[qid] = 1  # mixed

print(f"Category distribution: HC={sum(1 for v in cat_map.values() if v==2)}, "
      f"mixed={sum(1 for v in cat_map.values() if v==1)}, "
      f"non-positive={sum(1 for v in cat_map.values() if v==0)}")

# Feature 1: Task type (one-hot) -- correlation with HC membership
print("\nTask type vs HC rate (chi2):")
tt_cat = defaultdict(lambda: [0, 0, 0])  # [non-pos, mixed, HC]
for qid in common_qids:
    tt = q_orig[qid]["task_type"]
    tt_cat[tt][cat_map[qid]] += 1

for tt in sorted(task_types):
    counts = tt_cat[tt]
    n = sum(counts)
    print(f"  {tt:<30} non-pos={counts[0]:>3} ({counts[0]/n*100:.0f}%) mixed={counts[1]:>3} ({counts[1]/n*100:.0f}%) HC={counts[2]:>3} ({counts[2]/n*100:.0f}%)")

# Feature 2: Number of models correct on original (consensus correct)
print("\nConsensus correctness vs category:")
consensus_map = {}
for qid in common_qids:
    n_correct = sum([int(q_orig[qid]["correct"]), int(l_orig[qid]["correct"]), int(iv_orig[qid]["correct"])])
    consensus_map[qid] = n_correct  # 0, 1, 2, or 3

for n_correct in range(4):
    qids_this = [qid for qid in common_qids if consensus_map[qid] == n_correct]
    if not qids_this: continue
    cats = [cat_map[qid] for qid in qids_this]
    hc_rate = sum(1 for c in cats if c==2) / len(cats)
    mix_rate = sum(1 for c in cats if c==1) / len(cats)
    print(f"  n_correct_orig={n_correct}: n={len(qids_this)}, HC%={hc_rate*100:.1f}%, mixed%={mix_rate*100:.1f}%")

# Logistic regression: HC vs rest, predictors = task_type (one-hot) + n_correct_orig
# Using scipy and numpy since sklearn may not be available
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    # Features
    tt_encoder = LabelEncoder()
    tt_encoded = tt_encoder.fit_transform([q_orig[qid]["task_type"] for qid in common_qids])

    # One-hot encode
    n_tt = len(tt_encoder.classes_)
    X = np.zeros((len(common_qids), n_tt + 1))
    for i, qid in enumerate(common_qids):
        X[i, tt_encoded[i]] = 1
        X[i, n_tt] = consensus_map[qid]

    y_hc   = np.array([1 if cat_map[qid]==2 else 0 for qid in common_qids])
    y_mixed = np.array([1 if cat_map[qid]==1 else 0 for qid in common_qids])

    lr_hc = LogisticRegression(max_iter=1000, random_state=42)
    lr_hc.fit(X, y_hc)
    from sklearn.metrics import roc_auc_score
    auc_hc = roc_auc_score(y_hc, lr_hc.predict_proba(X)[:, 1])
    print(f"\nLogistic regression (HC vs rest): AUC={auc_hc:.3f}")
    print(f"  Features: task_type (one-hot, {n_tt} levels) + n_models_correct_original")
    print(f"  n_correct_orig coefficient: {lr_hc.coef_[0][-1]:.3f} (positive = more correct -> more likely HC)")

except ImportError:
    print("\nsklearn not available, skipping logistic regression")
    # Use point-biserial correlation instead
    hc_binary = np.array([1 if cat_map[qid]==2 else 0 for qid in common_qids])
    consensus_arr = np.array([consensus_map[qid] for qid in common_qids])
    r, p = stats.pointbiserialr(hc_binary, consensus_arr)
    print(f"\nPoint-biserial correlation HC ~ n_models_correct_original: r={r:.3f}, p={p:.6f}")

# ── GAP D: GT label distribution for scene_transition and unexpected_action ─
print("\n" + "─"*60)
print("GAP D: GT label distribution for scene_transition and unexpected_action")
print("─"*60)

for tt in ["scene_transition", "unexpected_action", "episodic_reasoning",
           "action_prediction", "state_change"]:
    entries = [r for r in q_mv if r["task_type"]==tt and r["condition"]=="original" and not r.get("error")]
    if not entries:
        print(f"  {tt}: no data in Qwen MVBench")
        continue
    from collections import Counter
    gt_dist = Counter(r["ground_truth"] for r in entries)
    n = len(entries)
    dom_let = max(gt_dist, key=lambda k: gt_dist[k])
    dom_pct = gt_dist[dom_let] / n * 100
    blk_entries = [r for r in l_mv if r["task_type"]==tt and r["condition"]=="black" and not r.get("error")]
    blk_acc = np.mean([r["correct"] for r in blk_entries]) if blk_entries else None
    blk_str = f"{blk_acc:.3f}" if blk_acc is not None else "N/A"
    dist_str = " ".join(f"{k}:{v/n*100:.0f}%" for k, v in sorted(gt_dist.items()))
    print(f"  {tt:<30} n={n:>3} dominant={dom_let}({dom_pct:.0f}%) GT: {dist_str}  LLaVA_blk={blk_str}")

# Number of options for scene_transition questions
st_opts = [r for r in q_mv if r["task_type"]=="scene_transition" and r["condition"]=="original" and not r.get("error")]
if st_opts:
    from collections import Counter
    # Try to check options structure
    print(f"\n  scene_transition example ground truths: {[r['ground_truth'] for r in st_opts[:10]]}")
    print(f"  scene_transition all ground truths: {Counter(r['ground_truth'] for r in st_opts)}")

# Also compute Qwen2-VL letter strategy on scene_transition black screen
print("\nModel letter strategies on scene_transition black screen:")
for model_name, mv_data in [("Qwen2-VL", q_mv), ("LLaVA", l_mv), ("InternVL2", iv_mv)]:
    entries = [r for r in mv_data if r["task_type"]=="scene_transition" and r["condition"]=="black" and not r.get("error")]
    if entries:
        from collections import Counter
        pred_dist = Counter(r["prediction"] for r in entries if r.get("prediction"))
        n = len(entries)
        acc = np.mean([r["correct"] for r in entries])
        dist_str = " ".join(f"{k}:{v/n*100:.0f}%" for k, v in sorted(pred_dist.items()))
        print(f"  {model_name:<12} n={n:>3} acc={acc:.3f} predictions: {dist_str}")

print("\n" + "=" * 70)
print("ROUND 3 ANALYSIS COMPLETE")
print("=" * 70)
