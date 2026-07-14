"""
Fix R3-2: Compute corrected episodic_reasoning VGG for Qwen2-VL
using matched valid-question subsets (same questions at both conditions).
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import numpy as np
from collections import defaultdict

def load_json(path):
    with open(path) as f:
        return json.load(f)

q_mv = load_json(VDG_RESULTS_ROOT + "/mvbench/qwen2vl_mvbench_results.json")

print("Episodic_reasoning corrected VGG analysis")
print("=" * 60)

# All episodic_reasoning entries
ep_orig  = {r["question_id"]: r for r in q_mv if r["task_type"]=="episodic_reasoning" and r["condition"]=="original"}
ep_black = {r["question_id"]: r for r in q_mv if r["task_type"]=="episodic_reasoning" and r["condition"]=="black"}

print(f"Original entries: {len(ep_orig)} (includes errors)")
print(f"Black entries:    {len(ep_black)} (includes errors)")

# Valid only
ep_orig_valid  = {qid: r for qid, r in ep_orig.items() if not r.get("error")}
ep_black_valid = {qid: r for qid, r in ep_black.items() if not r.get("error")}

print(f"\nOriginal valid: {len(ep_orig_valid)}")
print(f"Black valid:    {len(ep_black_valid)}")

# Unmatched VGG (current method — may be biased)
orig_acc_all_valid  = np.mean([r["correct"] for r in ep_orig_valid.values()])
black_acc_all_valid = np.mean([r["correct"] for r in ep_black_valid.values()])
vgg_unmatched = orig_acc_all_valid - black_acc_all_valid
print(f"\nUnmatched (current method):")
print(f"  orig_acc  = {orig_acc_all_valid:.3f}  (n={len(ep_orig_valid)})")
print(f"  black_acc = {black_acc_all_valid:.3f}  (n={len(ep_black_valid)})")
print(f"  VGG       = {vgg_unmatched:.3f}")

# Matched VGG (same question IDs, both valid)
matched_qids = set(ep_orig_valid) & set(ep_black_valid)
print(f"\nMatched (same valid question IDs): n={len(matched_qids)}")
orig_acc_matched  = np.mean([ep_orig_valid[qid]["correct"]  for qid in matched_qids])
black_acc_matched = np.mean([ep_black_valid[qid]["correct"] for qid in matched_qids])
vgg_matched = orig_acc_matched - black_acc_matched
print(f"  orig_acc  = {orig_acc_matched:.3f}")
print(f"  black_acc = {black_acc_matched:.3f}")
print(f"  VGG       = {vgg_matched:.3f}")

# Bootstrap CI on matched VGG
rng = np.random.default_rng(42)
matched_list = list(matched_qids)
boot_vggs = []
for _ in range(2000):
    sample = rng.choice(matched_list, len(matched_list), replace=True)
    oa = np.mean([ep_orig_valid[qid]["correct"]  for qid in sample])
    ba = np.mean([ep_black_valid[qid]["correct"] for qid in sample])
    boot_vggs.append(oa - ba)
ci_lo, ci_hi = np.percentile(boot_vggs, [2.5, 97.5])
print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

print(f"\nDifference between matched and unmatched VGG: {(vgg_matched - vgg_unmatched)*100:.1f}pp")

# Check which questions errored at original (they all succeed at black but not original)
errored_at_orig = set(ep_orig) - set(ep_orig_valid)
print(f"\nQuestion IDs that errored at original (n={len(errored_at_orig)}): {sorted(errored_at_orig)[:5]}...")

# Do those questions that errored at original do better at black?
errored_in_black = {qid: ep_black_valid[qid] for qid in errored_at_orig if qid in ep_black_valid}
if errored_in_black:
    acc = np.mean([r["correct"] for r in errored_in_black.values()])
    print(f"Among the {len(errored_in_black)} questions that errored at original but have valid black:")
    print(f"  black_acc = {acc:.3f}")
    print(f"  These questions: {acc:.3f} black acc vs {black_acc_matched:.3f} for matched questions")

# Also check other models for comparison
print("\n--- Comparison across models on episodic_reasoning ---")
for model_file, model_name in [
    (VDG_RESULTS_ROOT + "/mvbench/llava_mvbench_results.json", "LLaVA"),
    (VDG_RESULTS_ROOT + "/mvbench/internvl2_mvbench_results.json", "InternVL2"),
]:
    mv = load_json(model_file)
    orig_v  = {r["question_id"]: r for r in mv if r["task_type"]=="episodic_reasoning" and r["condition"]=="original" and not r.get("error")}
    black_v = {r["question_id"]: r for r in mv if r["task_type"]=="episodic_reasoning" and r["condition"]=="black" and not r.get("error")}
    match   = set(orig_v) & set(black_v)
    if match:
        oa = np.mean([orig_v[qid]["correct"]  for qid in match])
        ba = np.mean([black_v[qid]["correct"] for qid in match])
        print(f"  {model_name:<12} n_matched={len(match)}, orig={oa:.3f}, black={ba:.3f}, VGG={oa-ba:.3f}")

print("\nConclusion:")
print(f"Corrected Qwen2-VL episodic_reasoning VGG = {vgg_matched:.3f} (matched)")
print(f"Uncorrected VGG = {vgg_unmatched:.3f}")
if abs(vgg_matched - vgg_unmatched) < 0.05:
    print("=> Difference is negligible (<5pp). R1-F14 finding stands.")
else:
    print("=> Meaningful difference. Update F14 with corrected estimate.")
