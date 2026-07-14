"""
Mine existing Qwen + black screen data for deeper patterns:
  1. Cross-condition agreement (same prediction across all CRF levels)
  2. Answer letter bias
  3. Black vs original prediction agreement by task type
  4. Error patterns (confusion matrix)
  5. Questions where black screen outperforms original (video hurts)
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
from collections import Counter, defaultdict

with open(VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json") as f:
    qwen = json.load(f)
with open(VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json") as f:
    black = json.load(f)

CONDITIONS = ["original", "crf18", "crf23", "crf28", "crf33", "crf38"]

by_q = defaultdict(dict)
for r in qwen:
    by_q[r["question_id"]][r["condition"]] = r
black_by_q = {r["question_id"]: r for r in black}
tt_map = {r["question_id"]: r["task_type"] for r in qwen}

complete = {q for q, d in by_q.items() if all(c in d for c in CONDITIONS)}
print(f"Complete questions: {len(complete)}")

# === 1. Cross-condition agreement ===
same_pred = 0
for q in complete:
    preds = [by_q[q][c]["prediction"] for c in CONDITIONS]
    if len(set(preds)) == 1:
        same_pred += 1
print(f"\n=== 1. Cross-condition agreement ===")
print(f"Same prediction across all 6 CRF levels: {same_pred}/{len(complete)} ({100*same_pred/len(complete):.1f}%)")

# By task type
tt_same = defaultdict(lambda: [0, 0])
for q in complete:
    tt = tt_map.get(q, "?")
    tt_same[tt][1] += 1
    preds = [by_q[q][c]["prediction"] for c in CONDITIONS]
    if len(set(preds)) == 1:
        tt_same[tt][0] += 1
print("By task type:")
for tt, (s, n) in sorted(tt_same.items(), key=lambda x: -x[1][0]/max(x[1][1], 1)):
    print(f"  {tt:<30} {s}/{n} ({100*s/n:.1f}%)")

# === 2. Answer letter bias ===
print(f"\n=== 2. Answer letter distribution ===")
for label, source in [("original", "original"), ("crf38", "crf38"), ("black", "black")]:
    if label == "black":
        preds = [r["prediction"] for r in black
                 if r.get("prediction") and r["prediction"] in "ABCD"]
    else:
        preds = [by_q[q][source]["prediction"] for q in complete
                 if by_q[q][source].get("prediction") and by_q[q][source]["prediction"] in "ABCD"]
    dist = Counter(preds)
    total = sum(dist.values())
    parts = [f"{l}:{dist[l]:>3} ({100*dist[l]/total:.0f}%)" for l in "ABCD"]
    print(f"  {label:<10} {'  '.join(parts)}")

gt_dist = Counter(by_q[q]["original"]["ground_truth"] for q in complete)
total_gt = sum(gt_dist.values())
parts = [f"{l}:{gt_dist[l]:>3} ({100*gt_dist[l]/total_gt:.0f}%)" for l in "ABCD"]
print(f"  {'GT':<10} {'  '.join(parts)}")

# === 3. Black vs original: same prediction? ===
both = [q for q in complete if q in black_by_q]
same_as_black = 0
for q in both:
    if by_q[q]["original"]["prediction"] == black_by_q[q]["prediction"]:
        same_as_black += 1
print(f"\n=== 3. Black screen agreement ===")
print(f"Same prediction (original vs black): {same_as_black}/{len(both)} ({100*same_as_black/len(both):.1f}%)")

by_tt_agree = defaultdict(lambda: [0, 0])
for q in both:
    tt = tt_map.get(q, "?")
    by_tt_agree[tt][1] += 1
    if by_q[q]["original"]["prediction"] == black_by_q[q]["prediction"]:
        by_tt_agree[tt][0] += 1
print("By task type:")
for tt, (s, n) in sorted(by_tt_agree.items(), key=lambda x: -x[1][0]/max(x[1][1], 1)):
    print(f"  {tt:<30} {s}/{n} ({100*s/n:.1f}%)")

# === 4. Confusion matrix ===
print(f"\n=== 4. Error patterns (original condition) ===")
confusion = defaultdict(lambda: defaultdict(int))
for q in complete:
    r = by_q[q]["original"]
    pred = r.get("prediction")
    if not r.get("correct") and pred and pred in "ABCD":
        confusion[r["ground_truth"]][pred] += 1
print("When wrong, GT -> predicted:")
for gt in "ABCD":
    row = [f"{p}:{confusion[gt][p]:>3}" for p in "ABCD" if p != gt]
    print(f"  GT={gt}: {'  '.join(row)}")

# === 5. Black better than original ===
black_better = 0
orig_better = 0
for q in both:
    bc = black_by_q[q].get("correct", False)
    oc = by_q[q]["original"].get("correct", False)
    if bc and not oc:
        black_better += 1
    if oc and not bc:
        orig_better += 1
print(f"\n=== 5. Black screen vs original ===")
print(f"Black correct, original wrong: {black_better}/{len(both)} ({100*black_better/len(both):.1f}%) -- video HURTS")
print(f"Original correct, black wrong: {orig_better}/{len(both)} ({100*orig_better/len(both):.1f}%) -- video HELPS")
print(f"Net visual benefit: {orig_better - black_better} questions")

by_tt_hurt = defaultdict(lambda: [0, 0, 0])  # [video_helps, video_hurts, total]
for q in both:
    tt = tt_map.get(q, "?")
    bc = black_by_q[q].get("correct", False)
    oc = by_q[q]["original"].get("correct", False)
    by_tt_hurt[tt][2] += 1
    if oc and not bc:
        by_tt_hurt[tt][0] += 1  # helps
    if bc and not oc:
        by_tt_hurt[tt][1] += 1  # hurts
print("By task type (video helps / hurts):")
for tt, (helps, hurts, n) in sorted(by_tt_hurt.items(), key=lambda x: x[1][0]-x[1][1], reverse=True):
    net = helps - hurts
    print(f"  {tt:<30} helps={helps:>3}  hurts={hurts:>3}  net={net:>+4}  ({100*net/n:>+.1f}%)")
