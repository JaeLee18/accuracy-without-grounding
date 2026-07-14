"""Compute per-model positional bias on black-screen condition (Video-MME)."""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, numpy as np
from collections import Counter

def load(path):
    return json.load(open(path))

qb  = load(VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json")
lb  = load(VDG_RESULTS_ROOT + "/full_study/llava_black_results.json")
ivb = load(VDG_RESULTS_ROOT + "/full_study/internvl2_black_results.json")

def analyze(blk_results, name):
    recs = [r for r in blk_results if not r.get("error") and r.get("prediction")]
    n = len(recs)
    gt_counts   = Counter(r["ground_truth"] for r in recs)
    pred_counts = Counter(r["prediction"] for r in recs)
    correct = sum(1 for r in recs if r.get("correct"))
    black_acc = correct / n

    # Above-chance accuracy (chance = 25% for 4 options)
    above_chance = black_acc - 0.25

    # Most frequent GT letter
    most_freq_gt = gt_counts.most_common(1)[0][0]
    gt_freq = gt_counts[most_freq_gt] / n

    # Model prediction rate for most frequent GT letter
    pred_freq_most_gt = pred_counts[most_freq_gt] / n

    # Alignment: model correctly guesses most frequent GT letter
    aligned = sum(1 for r in recs if r["prediction"] == most_freq_gt and r["ground_truth"] == most_freq_gt)
    aligned_rate = aligned / n

    # Fraction of above-chance accuracy explained by letter alignment
    # alignment_contribution = (pred_freq - 0.25) * gt_freq (approximate)
    # Actually: above-chance accuracy from alignment = (aligned_rate - 0.0625)
    # where 0.0625 = chance of both model and GT picking same letter = 0.25 * 0.25
    alignment_contrib = aligned_rate - 0.0625  # above chance from pure alignment
    if above_chance > 0:
        pct_explained = alignment_contrib / above_chance * 100
    else:
        pct_explained = 0

    print(f"\n{name} (n={n}):")
    print(f"  Black-screen accuracy: {black_acc:.3f} ({black_acc*100:.1f}%)")
    print(f"  Above-chance: {above_chance:.3f} ({above_chance*100:.1f}pp)")
    print(f"  Most frequent GT letter: {most_freq_gt} ({gt_freq*100:.1f}%)")
    print(f"  GT distribution: {dict(sorted(gt_counts.items()))}")
    print(f"  Pred distribution: {dict(sorted(pred_counts.items()))}")
    print(f"  Letter alignment explains: {pct_explained:.1f}% of above-chance accuracy")
    return pct_explained, black_acc

pq, bq = analyze(qb, "Qwen2-VL")
pl, bl = analyze(lb, "LLaVA-Video")
pi, bi = analyze(ivb, "InternVL2")

print(f"\nSummary for paper:")
print(f"  Range of alignment explanation: {min(pq,pl,pi):.0f}–{max(pq,pl,pi):.0f}%")

# Shared-wrong consensus: all 3 models agree on same wrong letter
# Need to find common questions
qb_d  = {r["question_id"]: r for r in qb  if not r.get("error") and r.get("prediction")}
lb_d  = {r["question_id"]: r for r in lb  if not r.get("error") and r.get("prediction")}
ivb_d = {r["question_id"]: r for r in ivb if not r.get("error") and r.get("prediction")}
common = sorted(set(qb_d)&set(lb_d)&set(ivb_d))

wrong_consensus = 0
for qid in common:
    qr = qb_d[qid]; lr = lb_d[qid]; ir = ivb_d[qid]
    # All 3 agree on same letter AND all 3 wrong
    if (qr.get("prediction") and qr["prediction"] == lr["prediction"] == ir["prediction"]
            and not qr.get("correct") and not lr.get("correct") and not ir.get("correct")):
        wrong_consensus += 1

print(f"\nShared-wrong consensus: {wrong_consensus}/{len(common)} = {wrong_consensus/len(common)*100:.1f}%")
