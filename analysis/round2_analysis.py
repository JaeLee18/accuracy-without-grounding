"""
Round 2 statistical analysis targeting reviewer's 6 directions:
1. Characterize the 10.3% hard-core questions
2. Explain LLaVA's 39% cross-benchmark VGG divergence mechanistically
3. Explain +48pp action_antonym CRF anomaly
4. Predict question-level VGG agreement from question properties
5. GT label skew as predictor of above-chance black-screen performance
6. Bonferroni-corrected chance test for all 18 model×task cells
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import binom
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

# Video-MME results
q_results = load_json(VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json")
q_black   = load_json(VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json")
l_results = load_json(VDG_RESULTS_ROOT + "/full_study/llava_results.json")
l_black   = load_json(VDG_RESULTS_ROOT + "/full_study/llava_black_results.json")
iv_results = load_json(VDG_RESULTS_ROOT + "/full_study/internvl2_results.json")
iv_black   = load_json(VDG_RESULTS_ROOT + "/full_study/internvl2_black_results.json")

# MVBench results
q_mv   = load_json(VDG_RESULTS_ROOT + "/mvbench/qwen2vl_mvbench_results.json")
l_mv   = load_json(VDG_RESULTS_ROOT + "/mvbench/llava_mvbench_results.json")
iv_mv  = load_json(VDG_RESULTS_ROOT + "/mvbench/internvl2_mvbench_results.json")

print("=" * 70)
print("ROUND 2 ANALYSIS — 6 REVIEWER DIRECTIONS")
print("=" * 70)

# ── Helper: build per-question dicts for Video-MME ─────────────────────────
def build_vmme_dicts(results_list, black_list):
    orig = {r["question_id"]: r for r in results_list if r["condition"] == "original" and not r.get("error")}
    black = {r["question_id"]: r for r in black_list if not r.get("error")}
    return orig, black

q_orig, q_blk   = build_vmme_dicts(q_results, q_black)
l_orig, l_blk   = build_vmme_dicts(l_results, l_black)
iv_orig, iv_blk = build_vmme_dicts(iv_results, iv_black)

# Common question IDs across all 3 models
common_qids = set(q_orig) & set(l_orig) & set(iv_orig) & set(q_blk) & set(l_blk) & set(iv_blk)
print(f"\nCommon Video-MME question IDs (all 3 models, orig+black): {len(common_qids)}")

# ── DIRECTION 1: Characterize the 10.3% hard-core questions ───────────────
print("\n" + "─"*60)
print("DIRECTION 1: Hard-core questions (VGG>0 for all 3 models)")
print("─"*60)

# VGG per question per model
def vgg_per_question(orig_dict, blk_dict, qids):
    return {qid: int(orig_dict[qid]["correct"]) - int(blk_dict[qid]["correct"]) for qid in qids}

q_vgg_q  = vgg_per_question(q_orig, q_blk, common_qids)
l_vgg_q  = vgg_per_question(l_orig, l_blk, common_qids)
iv_vgg_q = vgg_per_question(iv_orig, iv_blk, common_qids)

# Hard-core: VGG>0 for all 3 models simultaneously
hard_core = {qid for qid in common_qids
             if q_vgg_q[qid] > 0 and l_vgg_q[qid] > 0 and iv_vgg_q[qid] > 0}
pct_hc = len(hard_core) / len(common_qids) * 100
print(f"Hard-core questions (VGG>0 all 3): {len(hard_core)} / {len(common_qids)} = {pct_hc:.1f}%")

# Task type breakdown of hard-core vs rest
task_types = sorted(set(q_orig[qid]["task_type"] for qid in common_qids))
hc_by_task = defaultdict(int)
all_by_task = defaultdict(int)
for qid in common_qids:
    tt = q_orig[qid]["task_type"]
    all_by_task[tt] += 1
    if qid in hard_core:
        hc_by_task[tt] += 1

print("\nTask-type breakdown of hard-core questions:")
print(f"{'Task Type':<30} {'HC':>5} {'All':>5} {'HC%':>6}")
for tt in task_types:
    n_hc = hc_by_task[tt]
    n_all = all_by_task[tt]
    pct = n_hc/n_all*100 if n_all else 0
    print(f"  {tt:<28} {n_hc:>5} {n_all:>5} {pct:>5.1f}%")

# Chi-square test: is hard-core concentration non-uniform?
observed = np.array([hc_by_task[tt] for tt in task_types])
expected_uniform = np.array([all_by_task[tt] * len(hard_core) / len(common_qids) for tt in task_types])
chi2, p_chi2 = stats.chisquare(observed, expected_uniform)
print(f"\nChi-square test (HC task distribution vs uniform): chi2={chi2:.2f}, p={p_chi2:.6f}")

# Fraction of total VGG accounted for by hard-core
total_vgg = sum(q_vgg_q[qid] + l_vgg_q[qid] + iv_vgg_q[qid] for qid in common_qids)
hc_vgg    = sum(q_vgg_q[qid] + l_vgg_q[qid] + iv_vgg_q[qid] for qid in hard_core)
print(f"\nTotal VGG across all questions (sum of per-q VGG, 3 models): {total_vgg}")
print(f"Hard-core VGG contribution: {hc_vgg} / {total_vgg} = {hc_vgg/total_vgg*100:.1f}%")

# Accuracy profile of hard-core vs non-hard-core
hc_orig_acc = np.mean([q_orig[qid]["correct"] for qid in hard_core])
nonhc_orig_acc = np.mean([q_orig[qid]["correct"] for qid in common_qids - hard_core])
hc_blk_acc = np.mean([q_blk[qid]["correct"] for qid in hard_core])
nonhc_blk_acc = np.mean([q_blk[qid]["correct"] for qid in common_qids - hard_core])
print(f"\nHard-core:     orig_acc={hc_orig_acc:.3f}, black_acc={hc_blk_acc:.3f}")
print(f"Non-hard-core: orig_acc={nonhc_orig_acc:.3f}, black_acc={nonhc_blk_acc:.3f}")

# ── DIRECTION 2: LLaVA 39% cross-benchmark VGG divergence ─────────────────
print("\n" + "─"*60)
print("DIRECTION 2: LLaVA's cross-benchmark VGG divergence")
print("─"*60)

# LLaVA VGG on Video-MME
l_vmme_orig_acc = np.mean([r["correct"] for r in l_results if r["condition"]=="original" and not r.get("error")])
l_vmme_blk_acc  = np.mean([r["correct"] for r in l_black if not r.get("error")])
l_vmme_vgg = l_vmme_orig_acc - l_vmme_blk_acc
print(f"LLaVA Video-MME: orig={l_vmme_orig_acc:.3f}, black={l_vmme_blk_acc:.3f}, VGG={l_vmme_vgg:.3f}")

# LLaVA VGG on MVBench
l_mv_orig = {r["question_id"]: r for r in l_mv if r["condition"]=="original" and not r.get("error")}
l_mv_blk  = {r["question_id"]: r for r in l_mv if r["condition"]=="black" and not r.get("error")}
common_mv = set(l_mv_orig) & set(l_mv_blk)
l_mv_orig_acc = np.mean([l_mv_orig[qid]["correct"] for qid in common_mv])
l_mv_blk_acc  = np.mean([l_mv_blk[qid]["correct"] for qid in common_mv])
l_mv_vgg = l_mv_orig_acc - l_mv_blk_acc
print(f"LLaVA MVBench:   orig={l_mv_orig_acc:.3f}, black={l_mv_blk_acc:.3f}, VGG={l_mv_vgg:.3f}")
print(f"VGG divergence: {abs(l_mv_vgg - l_vmme_vgg)*100:.1f}pp")

# Qwen for comparison
q_vmme_orig_acc = np.mean([r["correct"] for r in q_results if r["condition"]=="original" and not r.get("error")])
q_vmme_blk_acc  = np.mean([r["correct"] for r in q_black if not r.get("error")])
q_vmme_vgg = q_vmme_orig_acc - q_vmme_blk_acc

q_mv_orig = {r["question_id"]: r for r in q_mv if r["condition"]=="original" and not r.get("error")}
q_mv_blk  = {r["question_id"]: r for r in q_mv if r["condition"]=="black" and not r.get("error")}
common_mv_q = set(q_mv_orig) & set(q_mv_blk)
q_mv_vgg = np.mean([q_mv_orig[qid]["correct"] for qid in common_mv_q]) - np.mean([q_mv_blk[qid]["correct"] for qid in common_mv_q])
print(f"\nQwen Video-MME VGG={q_vmme_vgg:.3f}, MVBench VGG={q_mv_vgg:.3f}, divergence={abs(q_mv_vgg-q_vmme_vgg)*100:.1f}pp")

# Which MVBench task types drive LLaVA's high black-screen accuracy?
print("\nLLaVA MVBench black accuracy by task type:")
l_mv_blk_by_tt = defaultdict(list)
for qid in common_mv:
    tt = l_mv_blk[qid]["task_type"]
    l_mv_blk_by_tt[tt].append(l_mv_blk[qid]["correct"])

for tt in sorted(l_mv_blk_by_tt, key=lambda t: -np.mean(l_mv_blk_by_tt[t])):
    n = len(l_mv_blk_by_tt[tt])
    acc = np.mean(l_mv_blk_by_tt[tt])
    print(f"  {tt:<30} n={n:>3}, black_acc={acc:.3f}")

# Letter distribution analysis — MVBench vs Video-MME
print("\nLLaVA letter distribution (black screen):")
from collections import Counter

l_vmme_letters = Counter(r["prediction"] for r in l_black if r.get("prediction") and len(r["prediction"])==1)
l_mv_letters   = Counter(l_mv_blk[qid]["prediction"] for qid in common_mv if l_mv_blk[qid].get("prediction") and len(l_mv_blk[qid]["prediction"])==1)
total_vmme = sum(l_vmme_letters.values())
total_mv   = sum(l_mv_letters.values())
print(f"Video-MME: {dict(sorted(l_vmme_letters.items()))}, total={total_vmme}")
for let, cnt in sorted(l_vmme_letters.items()):
    print(f"  {let}: {cnt/total_vmme*100:.1f}%")
print(f"MVBench:   {dict(sorted(l_mv_letters.items()))}, total={total_mv}")
for let, cnt in sorted(l_mv_letters.items()):
    print(f"  {let}: {cnt/total_mv*100:.1f}%")

# Chi-square test for letter distribution difference
vmme_dist = [l_vmme_letters.get(x, 0) for x in "ABCD"]
mv_dist   = [l_mv_letters.get(x, 0) for x in "ABCD"]
ct_table = np.array([vmme_dist, mv_dist])
chi2_let, p_let, dof_let, _ = chi2_contingency(ct_table)
print(f"\nChi-square for letter distribution difference (Video-MME vs MVBench): chi2={chi2_let:.2f}, p={p_let:.4f}")

# ── DIRECTION 3: action_antonym CRF anomaly ────────────────────────────────
print("\n" + "─"*60)
print("DIRECTION 3: action_antonym +48pp CRF anomaly (MVBench)")
print("─"*60)

# Check action_antonym in all MVBench models
for model_name, mv_data in [("Qwen2-VL", q_mv), ("LLaVA", l_mv), ("InternVL2", iv_mv)]:
    aa_orig = [r for r in mv_data if r["task_type"]=="action_antonym" and r["condition"]=="original" and not r.get("error")]
    aa_crf  = [r for r in mv_data if r["task_type"]=="action_antonym" and r["condition"]=="crf38" and not r.get("error")]
    aa_blk  = [r for r in mv_data if r["task_type"]=="action_antonym" and r["condition"]=="black" and not r.get("error")]
    aa_err  = [r for r in mv_data if r["task_type"]=="action_antonym" and r["condition"]=="crf38" and r.get("error")]
    print(f"{model_name}: n_orig={len(aa_orig)}, n_crf38_valid={len(aa_crf)}, n_crf38_error={len(aa_err)}, n_black={len(aa_blk)}")
    if aa_orig: print(f"  orig_acc={np.mean([r['correct'] for r in aa_orig]):.3f}")
    if aa_crf:  print(f"  crf38_acc={np.mean([r['correct'] for r in aa_crf]):.3f}")
    if aa_blk:  print(f"  black_acc={np.mean([r['correct'] for r in aa_blk]):.3f}")

# Check how many of the 50 action_antonym questions have CRF38 files
print("\nChecking CRF38 file existence for action_antonym:")
import os
aa_ids = list(set(r["videoID"] for r in q_mv if r["task_type"]=="action_antonym"))
print(f"Unique video IDs for action_antonym: {len(aa_ids)}")
crf38_dir = VDG_DATA_ROOT + "/mvbench/crf38"
found = sum(1 for vid in aa_ids if os.path.exists(f"{crf38_dir}/{vid}.mp4"))
print(f"CRF38 files found: {found}/{len(aa_ids)}")

# ── DIRECTION 4: Predict question-level VGG agreement ─────────────────────
print("\n" + "─"*60)
print("DIRECTION 4: Predicting question-level VGG from question properties")
print("─"*60)

# For each common question, compute: VGG agreement (do Q and L agree on sign of VGG?)
# Potential predictors: task_type, GT letter, question_length

# VGG categories: both positive, both negative/zero, mixed
q_vgg_sign  = {qid: 1 if q_vgg_q[qid] > 0 else (-1 if q_vgg_q[qid] < 0 else 0) for qid in common_qids}
l_vgg_sign  = {qid: 1 if l_vgg_q[qid] > 0 else (-1 if l_vgg_q[qid] < 0 else 0) for qid in common_qids}
iv_vgg_sign = {qid: 1 if iv_vgg_q[qid] > 0 else (-1 if iv_vgg_q[qid] < 0 else 0) for qid in common_qids}

# Agreement = all 3 models have same VGG sign (all positive or all non-positive)
agree_pos   = {qid for qid in common_qids if q_vgg_sign[qid]>0 and l_vgg_sign[qid]>0 and iv_vgg_sign[qid]>0}
agree_neg   = {qid for qid in common_qids if q_vgg_sign[qid]<=0 and l_vgg_sign[qid]<=0 and iv_vgg_sign[qid]<=0}
disagree    = common_qids - agree_pos - agree_neg

print(f"VGG sign agreement:")
print(f"  All positive (HC):   {len(agree_pos)} ({len(agree_pos)/len(common_qids)*100:.1f}%)")
print(f"  All non-positive:    {len(agree_neg)} ({len(agree_neg)/len(common_qids)*100:.1f}%)")
print(f"  Mixed (disagree):    {len(disagree)} ({len(disagree)/len(common_qids)*100:.1f}%)")

# Task type predicts agreement?
tt_agree_rate = defaultdict(lambda: [0, 0])
for qid in common_qids:
    tt = q_orig[qid]["task_type"]
    tt_agree_rate[tt][1] += 1
    if qid in agree_pos or qid in agree_neg:
        tt_agree_rate[tt][0] += 1

print("\nVGG sign agreement rate by task type:")
for tt in sorted(tt_agree_rate, key=lambda t: -tt_agree_rate[t][0]/tt_agree_rate[t][1]):
    a, n = tt_agree_rate[tt]
    print(f"  {tt:<30} {a}/{n} = {a/n*100:.1f}% agree")

# GT letter as predictor
gt_agree_rate = defaultdict(lambda: [0, 0])
for qid in common_qids:
    gt = q_orig[qid]["ground_truth"]
    gt_agree_rate[gt][1] += 1
    if qid in agree_pos:
        gt_agree_rate[gt][0] += 1

print("\nHard-core rate by GT letter:")
for let in "ABCD":
    a, n = gt_agree_rate[let]
    print(f"  {let}: {a}/{n} = {a/n*100:.1f}% hard-core")

# ── DIRECTION 5: GT label skew as predictor ────────────────────────────────
print("\n" + "─"*60)
print("DIRECTION 5: GT label skew predicts black-screen performance?")
print("─"*60)

# For each task type in Video-MME, compute GT label distribution and black-screen accuracy
tt_gt_skew = defaultdict(lambda: defaultdict(int))
tt_blk_acc = defaultdict(list)

for qid in common_qids:
    tt  = q_orig[qid]["task_type"]
    gt  = q_orig[qid]["ground_truth"]
    tt_gt_skew[tt][gt] += 1
    # Average black-screen accuracy across 3 models
    blk_acc = np.mean([int(q_blk[qid]["correct"]), int(l_blk[qid]["correct"]), int(iv_blk[qid]["correct"])])
    tt_blk_acc[tt].append(blk_acc)

print(f"{'Task Type':<30} {'GT entropy':>10} {'Blk acc':>8}")
entropy_vals = []
blk_vals = []
for tt in sorted(task_types):
    dist = tt_gt_skew[tt]
    n = sum(dist.values())
    if n == 0:
        continue
    probs = np.array([dist.get(x, 0)/n for x in "ABCD"])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    avg_blk = np.mean(tt_blk_acc[tt])
    print(f"  {tt:<28} {entropy:>10.3f} {avg_blk:>8.3f}")
    entropy_vals.append(entropy)
    blk_vals.append(avg_blk)

r_spear, p_spear = stats.spearmanr(entropy_vals, blk_vals)
r_pear, p_pear   = stats.pearsonr(entropy_vals, blk_vals)
print(f"\nCorrelation GT-entropy vs black-screen: Spearman r={r_spear:.3f} p={p_spear:.3f}, Pearson r={r_pear:.3f} p={p_pear:.3f}")

# Dominant letter analysis (letter most likely to appear in GT)
print("\nDominant GT letter by task type:")
for tt in sorted(task_types):
    dist = tt_gt_skew[tt]
    n = sum(dist.values())
    if n == 0: continue
    dom_let = max(dist, key=lambda k: dist[k])
    dom_pct = dist[dom_let] / n * 100
    blk_acc_tt = np.mean(tt_blk_acc[tt])
    print(f"  {tt:<30} dominant={dom_let} ({dom_pct:.0f}%), blk_acc={blk_acc_tt:.3f}")

# ── DIRECTION 6: Bonferroni-corrected chance tests ─────────────────────────
print("\n" + "─"*60)
print("DIRECTION 6: Bonferroni-corrected chance tests (18 model×task cells)")
print("─"*60)

# For each model × task_type, test if black-screen accuracy > chance (0.25)
MODELS = {
    "Qwen2-VL":  (q_blk, q_orig),
    "LLaVA":     (l_blk, l_orig),
    "InternVL2": (iv_blk, iv_orig),
}

results_chance = []
print(f"{'Model':<12} {'Task Type':<30} {'n':>4} {'acc':>6} {'p_raw':>8} {'p_bonf':>8} {'sig':>4}")

n_tests = len(MODELS) * len(task_types)
for model_name, (blk_dict, orig_dict) in MODELS.items():
    for tt in task_types:
        qids_tt = [qid for qid in common_qids if orig_dict[qid]["task_type"] == tt]
        n = len(qids_tt)
        if n == 0: continue
        k = sum(int(blk_dict[qid]["correct"]) for qid in qids_tt if qid in blk_dict)
        actual_n = sum(1 for qid in qids_tt if qid in blk_dict)
        if actual_n == 0: continue
        # Binomial test (one-sided: acc > 0.25)
        p_raw = binom.sf(k - 1, actual_n, 0.25)  # P(X >= k) = 1 - P(X <= k-1)
        p_bonf = min(p_raw * n_tests, 1.0)
        sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "ns"))
        acc = k / actual_n
        results_chance.append({
            "model": model_name, "task_type": tt, "n": actual_n, "k": k,
            "acc": acc, "p_raw": p_raw, "p_bonf": p_bonf, "sig": sig
        })
        print(f"  {model_name:<10} {tt:<30} {actual_n:>4} {acc:>6.3f} {p_raw:>8.4f} {p_bonf:>8.4f} {sig:>4}")

n_sig = sum(1 for r in results_chance if r["p_bonf"] < 0.05)
n_ns  = sum(1 for r in results_chance if r["p_bonf"] >= 0.05)
print(f"\nSignificant after Bonferroni: {n_sig}/{len(results_chance)}")
print(f"Not significant: {n_ns}/{len(results_chance)}")
print("\nNot significant cells (≥ chance even after Bonferroni):")
for r in results_chance:
    if r["p_bonf"] >= 0.05:
        print(f"  {r['model']:<12} {r['task_type']:<30} acc={r['acc']:.3f} p_bonf={r['p_bonf']:.3f}")

# ── NEW FINDING: McNemar follow-up — video necessity at question level ─────
print("\n" + "─"*60)
print("BONUS: Question-level necessity — what fraction REQUIRE video to be correct?")
print("─"*60)

# A question "requires video" if: orig_correct AND black_wrong for model
# At least 1 model
req_any = 0
req_all = 0
never_correct_orig = 0
for qid in common_qids:
    q_needs   = q_orig[qid]["correct"] and not q_blk[qid]["correct"]
    l_needs   = l_orig[qid]["correct"] and not l_blk[qid]["correct"]
    iv_needs  = iv_orig[qid]["correct"] and not iv_blk[qid]["correct"]
    if q_needs or l_needs or iv_needs:
        req_any += 1
    if q_needs and l_needs and iv_needs:
        req_all += 1
    if not q_orig[qid]["correct"] and not l_orig[qid]["correct"] and not iv_orig[qid]["correct"]:
        never_correct_orig += 1

print(f"Questions where ≥1 model requires video: {req_any} ({req_any/len(common_qids)*100:.1f}%)")
print(f"Questions where ALL 3 models require video: {req_all} ({req_all/len(common_qids)*100:.1f}%)")
print(f"Questions nobody gets right (with original video): {never_correct_orig} ({never_correct_orig/len(common_qids)*100:.1f}%)")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
