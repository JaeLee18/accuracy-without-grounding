"""
Exp A + B: VGG-stratified CRF analysis and CRF-sensitive enrichment test.
Reads existing Qwen results — no new inference needed.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress, fisher_exact
from collections import defaultdict

QWEN_PATH    = VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json"
BLACK_PATH   = VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json"
SAMPLE_PATH  = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
REPORT_PATH  = VDG_RESULTS_ROOT + "/full_study/exp_ab_report.txt"
PLOTS_DIR    = VDG_RESULTS_ROOT + "/full_study/plots"
CONDITIONS   = ["original", "crf18", "crf23", "crf28", "crf33", "crf38"]
CRF_VALUES   = [0, 18, 23, 28, 33, 38]
N_BOOTSTRAP  = 2000

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(QWEN_PATH) as f:
    qwen = json.load(f)
with open(BLACK_PATH) as f:
    black = json.load(f)
with open(SAMPLE_PATH) as f:
    samples = json.load(f)

# Index: (question_id, condition) -> correct bool
qwen_idx = {(r["question_id"], r["condition"]): r.get("correct", False)
            for r in qwen if not r.get("error")}
black_idx = {r["question_id"]: r.get("correct", False) for r in black}

# Questions with all 6 conditions present (handle partial TR)
all_qids = set(r["question_id"] for r in qwen)
complete_qids = {qid for qid in all_qids
                 if all((qid, c) in qwen_idx for c in CONDITIONS)}
print(f"Questions with all 6 conditions: {len(complete_qids)} / {len(all_qids)}")

# Map question_id -> task_type
qid_to_type = {s["question_id"]: s["task_type"] for s in samples}
task_types = sorted(set(qid_to_type.values()))

# ── Per-task-type accuracy ─────────────────────────────────────────────────────
tt_orig_acc  = {}
tt_black_acc = {}

for tt in task_types:
    qids = [q for q in complete_qids if qid_to_type.get(q) == tt]
    if not qids:
        continue
    orig  = [qwen_idx[(q, "original")] for q in qids]
    blk   = [black_idx[q] for q in qids if q in black_idx]
    tt_orig_acc[tt]  = np.mean(orig)
    tt_black_acc[tt] = np.mean(blk) if blk else 0.0

tt_vgg = {tt: tt_orig_acc[tt] - tt_black_acc[tt] for tt in tt_orig_acc}

print("\nPer-task-type VGG:")
for tt in sorted(tt_vgg, key=tt_vgg.get, reverse=True):
    print(f"  {tt:<30} orig={tt_orig_acc[tt]:.3f}  black={tt_black_acc[tt]:.3f}  VGG={tt_vgg[tt]:.3f}")

# ── Stratum assignment ─────────────────────────────────────────────────────────
def get_stratum(tt):
    v = tt_vgg.get(tt, 0)
    if v > 0.30:   return "High VGG (>0.30)"
    if v > 0.15:   return "Mid VGG (0.15–0.30)"
    return             "Low VGG (<0.15)"

strata = ["High VGG (>0.30)", "Mid VGG (0.15–0.30)", "Low VGG (<0.15)"]
stratum_colors = {"High VGG (>0.30)": "#2196F3", "Mid VGG (0.15–0.30)": "#FF9800", "Low VGG (<0.15)": "#F44336"}

# ── EXP A: CRF degradation by stratum ─────────────────────────────────────────
print("\n=== EXP A: CRF degradation by VGG stratum ===")

stratum_results = {}
for stratum in strata:
    qids = [q for q in complete_qids if get_stratum(qid_to_type.get(q, "")) == stratum]
    if not qids:
        continue
    accs = []
    for c, crf in zip(CONDITIONS, CRF_VALUES):
        vals = [qwen_idx[(q, c)] for q in qids]
        accs.append(np.mean(vals))

    slope, intercept, r, p, se = linregress(CRF_VALUES, accs)

    # Bootstrap CI on CRF38 - original degradation
    degs = []
    rng = np.random.default_rng(42)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(qids), size=len(qids))
        bq = [qids[i] for i in idx]
        orig_b = np.mean([qwen_idx[(q, "original")] for q in bq])
        crf38_b = np.mean([qwen_idx[(q, "crf38")] for q in bq])
        degs.append(crf38_b - orig_b)
    ci_lo, ci_hi = np.percentile(degs, [2.5, 97.5])
    deg_obs = accs[-1] - accs[0]

    stratum_results[stratum] = {
        "n": len(qids), "accs": accs, "slope": slope,
        "slope_p": p, "deg": deg_obs, "ci_lo": ci_lo, "ci_hi": ci_hi
    }
    print(f"\n  {stratum} (n={len(qids)})")
    for c, a in zip(CONDITIONS, accs):
        print(f"    {c:<10} {a:.3f}")
    print(f"    Slope: {slope:.5f}  p={p:.3f}")
    print(f"    CRF38 deg: {deg_obs:+.3f}  95%CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")

# Plot Exp A
fig, ax = plt.subplots(figsize=(7, 5))
for stratum, res in stratum_results.items():
    color = stratum_colors[stratum]
    ax.plot(CRF_VALUES, res["accs"], marker='o', label=f"{stratum} (n={res['n']})", color=color)
    deg = res["deg"]
    ci_lo, ci_hi = res["ci_lo"], res["ci_hi"]
    ax.fill_between(CRF_VALUES,
                    [a + ci_lo for a in res["accs"]],
                    [a + ci_hi for a in res["accs"]],
                    alpha=0.15, color=color)

ax.axhline(0.25, color='gray', linestyle=':', linewidth=1, label='Chance (0.25)')
ax.set_xlabel("CRF level (0 = original)")
ax.set_ylabel("Accuracy")
ax.set_title("CRF Degradation by Visual Grounding Stratum (Qwen2-VL)")
ax.legend(fontsize=9)
ax.set_ylim(0.15, 0.85)
ax.set_xticks(CRF_VALUES)
ax.set_xticklabels(["Orig", "18", "23", "28", "33", "38"])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp_a_vgg_stratified_crf.png", dpi=150)
plt.close()
print(f"\nFig saved: {PLOTS_DIR}/exp_a_vgg_stratified_crf.png")

# ── EXP B: CRF-sensitive enrichment ───────────────────────────────────────────
print("\n=== EXP B: CRF-sensitive question enrichment ===")

sensitive = []
non_sensitive = []

for qid in complete_qids:
    orig_correct  = qwen_idx.get((qid, "original"), False)
    crf38_correct = qwen_idx.get((qid, "crf38"), False)
    black_correct = black_idx.get(qid, False)

    is_sensitive = orig_correct and not crf38_correct   # compression hurt this question
    has_vgg      = orig_correct and not black_correct   # model uses vision here

    if is_sensitive:
        sensitive.append(has_vgg)
    else:
        non_sensitive.append(has_vgg)

n_sens = len(sensitive)
n_vgg_sens = sum(sensitive)
n_nonsens = len(non_sensitive)
n_vgg_nonsens = sum(non_sensitive)

rate_sens    = n_vgg_sens / n_sens if n_sens else 0
rate_nonsens = n_vgg_nonsens / n_nonsens if n_nonsens else 0

contingency = [[n_vgg_sens, n_sens - n_vgg_sens],
               [n_vgg_nonsens, n_nonsens - n_vgg_nonsens]]
odds_ratio, p_value = fisher_exact(contingency, alternative='greater')

print(f"\n  CRF-sensitive questions (orig=correct, crf38=wrong): {n_sens}")
print(f"    VGG=+1 rate: {n_vgg_sens}/{n_sens} = {rate_sens:.3f}")
print(f"  Non-sensitive questions: {n_nonsens}")
print(f"    VGG=+1 rate: {n_vgg_nonsens}/{n_nonsens} = {rate_nonsens:.3f}")
print(f"  Odds ratio: {odds_ratio:.2f}  Fisher exact p={p_value:.4f}")

# Task type breakdown of sensitive questions
print(f"\n  Task type breakdown of CRF-sensitive questions:")
sens_qids = [q for q in complete_qids
             if qwen_idx.get((q,"original")) and not qwen_idx.get((q,"crf38"))]
type_counts = defaultdict(int)
for q in sens_qids:
    type_counts[qid_to_type.get(q, "?")] += 1
for tt, n in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"    {tt:<30} {n}")

# Plot Exp B
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["CRF-sensitive\n(n={})".format(n_sens),
               "Non-sensitive\n(n={})".format(n_nonsens)],
              [rate_sens, rate_nonsens],
              color=["#E53935", "#1E88E5"], width=0.5, edgecolor='black', linewidth=0.8)
ax.axhline(rate_nonsens, color='#1E88E5', linestyle='--', linewidth=1)
ax.set_ylabel("VGG=+1 rate (model uses vision)")
ax.set_title(f"CRF-Sensitive Questions Are More Visually Grounded\n(Fisher exact p={p_value:.4f})")
ax.set_ylim(0, 0.85)
for bar, val in zip(bars, [rate_sens, rate_nonsens]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
            ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/exp_b_vgg_enrichment.png", dpi=150)
plt.close()
print(f"\nFig saved: {PLOTS_DIR}/exp_b_vgg_enrichment.png")

# ── Write report ───────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("EXP A+B REPORT")
lines.append("=" * 70)
lines.append(f"Questions analysed: {len(complete_qids)} (all with complete 6-condition data)")
lines.append("")
lines.append("--- Task-type VGG ---")
for tt in sorted(tt_vgg, key=tt_vgg.get, reverse=True):
    lines.append(f"  {tt:<30} orig={tt_orig_acc[tt]:.3f}  black={tt_black_acc[tt]:.3f}  VGG={tt_vgg[tt]:.3f}  stratum={get_stratum(tt)}")
lines.append("")
lines.append("--- EXP A: CRF degradation by stratum ---")
for stratum, res in stratum_results.items():
    lines.append(f"  {stratum} (n={res['n']})")
    lines.append(f"    CRF slope: {res['slope']:.5f}  p={res['slope_p']:.3f}")
    lines.append(f"    CRF38 deg: {res['deg']:+.3f}  95%CI [{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}]")
lines.append("")
lines.append("--- EXP B: CRF-sensitive enrichment ---")
lines.append(f"  CRF-sensitive: {n_sens}  VGG=+1 rate: {rate_sens:.3f}")
lines.append(f"  Non-sensitive: {n_nonsens}  VGG=+1 rate: {rate_nonsens:.3f}")
lines.append(f"  Odds ratio: {odds_ratio:.2f}  Fisher exact p={p_value:.4f}")
lines.append(f"  Task type breakdown: {dict(type_counts)}")

report = "\n".join(lines)
print("\n" + report)
with open(REPORT_PATH, "w") as f:
    f.write(report)
print(f"\nReport saved: {REPORT_PATH}")
