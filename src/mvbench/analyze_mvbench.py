"""
MVBench analysis: VGG per task type for Qwen2-VL and LLaVA-Video.
Outputs:
  - per-model accuracy tables (original / crf38 / black)
  - VGG = original_acc - black_acc per task type
  - CRF38 degradation = original_acc - crf38_acc
  - Bootstrap 95% CI on VGG
  - Summary saved to results/mvbench/mvbench_paper_numbers.txt
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os
import numpy as np
from collections import defaultdict

QWEN_PATH  = VDG_RESULTS_ROOT + "/mvbench/qwen2vl_mvbench_results.json"
LLAVA_PATH = VDG_RESULTS_ROOT + "/mvbench/llava_mvbench_results.json"
IV2_PATH   = VDG_RESULTS_ROOT + "/mvbench/internvl2_mvbench_results.json"
OUT_PATH   = VDG_RESULTS_ROOT + "/mvbench/mvbench_paper_numbers.txt"

# Only include task types with >=10 questions available
TASK_TYPES_FULL = [
    "action_antonym", "action_prediction", "counterfactual_inference",
    "egocentric_navigation", "episodic_reasoning", "object_existence",
    "scene_transition", "state_change", "unexpected_action",
]
# Include all available task types in per-task tables (even partial)
TASK_TYPES = [
    "action_antonym", "action_count", "action_localization", "action_prediction",
    "action_sequence", "character_order", "counterfactual_inference",
    "egocentric_navigation", "episodic_reasoning", "fine_grained_action",
    "moving_attribute", "moving_count", "moving_direction", "object_existence",
    "object_interaction", "object_shuffle", "scene_transition", "state_change",
    "unexpected_action",
]

CONDITIONS = ["original", "crf38", "black"]
RNG = np.random.default_rng(42)
N_BOOT = 2000


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_index(results):
    """Returns {(question_id, condition): correct} excluding errors."""
    return {
        (r["question_id"], r["condition"]): r.get("correct", False)
        for r in results
        if not r.get("error")
    }


def acc_table(results):
    """Returns {task_type: {condition: accuracy}}"""
    by_tt = defaultdict(lambda: defaultdict(list))
    idx = build_index(results)
    for r in results:
        if r.get("error"):
            continue
        by_tt[r["task_type"]][r["condition"]].append(
            idx[(r["question_id"], r["condition"])]
        )
    table = {}
    for tt, conds in by_tt.items():
        table[tt] = {c: (sum(v) / len(v) if v else None) for c, v in conds.items()}
    return table


def vgg_bootstrap(results, task_type=None):
    """
    Bootstrap 95% CI on VGG = original_acc - black_acc.
    Uses only questions with BOTH original and black completed (same set).
    If task_type is None, uses all tasks.
    """
    idx = build_index(results)
    # Gather question_ids with both conditions
    orig_keys = {qid for (qid, cond) in idx if cond == "original"}
    black_keys = {qid for (qid, cond) in idx if cond == "black"}
    shared = sorted(orig_keys & black_keys)

    # Filter by task type if specified
    task_map = {r["question_id"]: r["task_type"] for r in results}
    if task_type:
        shared = [qid for qid in shared if task_map.get(qid) == task_type]

    if len(shared) < 5:
        return None, None, None

    orig_arr = np.array([idx[(qid, "original")] for qid in shared], dtype=float)
    black_arr = np.array([idx[(qid, "black")] for qid in shared], dtype=float)
    vgg_obs = orig_arr.mean() - black_arr.mean()

    boot = []
    for _ in range(N_BOOT):
        sel = RNG.integers(0, len(shared), size=len(shared))
        boot.append(orig_arr[sel].mean() - black_arr[sel].mean())
    ci_lo = np.percentile(boot, 2.5)
    ci_hi = np.percentile(boot, 97.5)
    return vgg_obs, ci_lo, ci_hi


def print_model_section(name, results, f):
    table = acc_table(results)

    f.write(f"\n{'='*60}\n")
    f.write(f"MODEL: {name}\n")
    f.write(f"{'='*60}\n\n")

    def write_task_row(tt):
        if tt not in table:
            f.write(f"  {tt:<33} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>7}\n")
            return
        row = table[tt]
        orig  = row.get("original")
        crf38 = row.get("crf38")
        black = row.get("black")
        vgg = (orig - black) if (orig is not None and black is not None) else None
        o_s = f"{orig:.3f}"  if orig  is not None else "N/A"
        c_s = f"{crf38:.3f}" if crf38 is not None else "N/A"
        b_s = f"{black:.3f}" if black is not None else "N/A"
        v_s = f"{vgg:.3f}"   if vgg   is not None else "N/A"
        f.write(f"  {tt:<33} {o_s:>6} {c_s:>6} {b_s:>6} {v_s:>7}\n")

    # Main table: full task types only (n=50 each)
    f.write(f"{'Task Type':<35} {'orig':>6} {'crf38':>6} {'black':>6} {'VGG':>7}\n")
    f.write("-" * 65 + "\n")
    for tt in TASK_TYPES_FULL:
        write_task_row(tt)

    # Partial task types (n<30, excluded from paper analysis)
    partial = [tt for tt in TASK_TYPES if tt not in TASK_TYPES_FULL]
    f.write(f"\n  [Excluded task types — n<30, not used in paper analysis]\n")
    f.write(f"  {'Task Type':<33} {'orig':>6} {'crf38':>6} {'black':>6} {'VGG':>7}  n\n")
    f.write("  " + "-" * 60 + "\n")
    task_map = {r["question_id"]: r["task_type"] for r in results}
    for tt in partial:
        n_tt = sum(1 for r in results if not r.get("error") and r["task_type"] == tt and r["condition"] == "original")
        if n_tt == 0:
            continue
        row = table.get(tt, {})
        orig  = row.get("original")
        crf38 = row.get("crf38")
        black = row.get("black")
        vgg = (orig - black) if (orig is not None and black is not None) else None
        o_s = f"{orig:.3f}"  if orig  is not None else "N/A"
        c_s = f"{crf38:.3f}" if crf38 is not None else "N/A"
        b_s = f"{black:.3f}" if black is not None else "N/A"
        v_s = f"{vgg:.3f}"   if vgg   is not None else "N/A"
        f.write(f"  {tt:<33} {o_s:>6} {c_s:>6} {b_s:>6} {v_s:>7}  {n_tt}\n")

    # Overall
    all_orig  = [r["correct"] for r in results if not r.get("error") and r["condition"] == "original"]
    all_crf38 = [r["correct"] for r in results if not r.get("error") and r["condition"] == "crf38"]
    all_black = [r["correct"] for r in results if not r.get("error") and r["condition"] == "black"]
    if all_orig and all_black:
        overall_vgg = np.mean(all_orig) - np.mean(all_black)
        crf38_s  = f"{np.mean(all_crf38):.3f}" if all_crf38 else "N/A"
        orig_s   = f"{np.mean(all_orig):.3f}"
        black_s  = f"{np.mean(all_black):.3f}"
        ovgg_s   = f"{overall_vgg:.3f}"
        f.write(f"\n  {'OVERALL':<33} {orig_s:>6} {crf38_s:>6} {black_s:>6} {ovgg_s:>7}\n")

    # Bootstrap CI on overall VGG
    vgg_obs, ci_lo, ci_hi = vgg_bootstrap(results)
    if vgg_obs is not None:
        f.write(f"\n  Overall VGG = {vgg_obs:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] 95% CI\n")

    # Per-task VGG with CI (full task types only)
    f.write(f"\n  Per-task VGG bootstrap (full task types, sorted by VGG):\n")
    task_vggs = []
    for tt in TASK_TYPES_FULL:
        v, lo, hi = vgg_bootstrap(results, task_type=tt)
        if v is not None:
            task_vggs.append((tt, v, lo, hi))
    task_vggs.sort(key=lambda x: -x[1])
    for tt, v, lo, hi in task_vggs:
        f.write(f"    {tt:<33} VGG={v:.3f} [{lo:.3f}, {hi:.3f}]\n")

    # CRF38 degradation (full task types only)
    f.write(f"\n  CRF38 degradation (original - crf38, full task types):\n")
    for tt in TASK_TYPES_FULL:
        if tt not in table:
            continue
        row = table[tt]
        orig  = row.get("original")
        crf38 = row.get("crf38")
        if orig is not None and crf38 is not None:
            deg = orig - crf38
            f.write(f"    {tt:<33} {deg:+.3f}\n")

    f.write(f"\n  Completed entries: {len([r for r in results if not r.get('error')])}\n")
    f.write(f"  Error entries: {len([r for r in results if r.get('error')])}\n")


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    qwen_results  = load(QWEN_PATH)
    llava_results = load(LLAVA_PATH)
    iv2_results   = load(IV2_PATH)

    with open(OUT_PATH, "w") as f:
        f.write("MVBench Analysis — Paper Numbers\n")
        f.write(f"Date: 2026-03-18\n")

        if qwen_results:
            print_model_section("Qwen2-VL-7B-Instruct", qwen_results, f)
        else:
            f.write("\nQwen results not yet available.\n")

        if llava_results:
            print_model_section("LLaVA-Video-7B-Qwen2", llava_results, f)
        else:
            f.write("\nLLaVA results not yet available.\n")

        if iv2_results:
            print_model_section("InternVL2-8B", iv2_results, f)
        else:
            f.write("\nInternVL2 results not yet available.\n")

        # Cross-model VGG comparison (3 models)
        if qwen_results and llava_results and iv2_results:
            f.write(f"\n{'='*60}\n")
            f.write("CROSS-MODEL VGG COMPARISON (3 models)\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"{'Task Type':<35} {'Qwen VGG':>10} {'LLaVA VGG':>10} {'IV2 VGG':>10}\n")
            f.write("-" * 68 + "\n")
            q_table  = acc_table(qwen_results)
            l_table  = acc_table(llava_results)
            iv_table = acc_table(iv2_results)
            for tt in TASK_TYPES_FULL:
                def get_vgg(t):
                    r = t.get(tt, {})
                    o, b = r.get("original"), r.get("black")
                    return (o - b) if (o is not None and b is not None) else None
                qv_s  = f"{get_vgg(q_table):.3f}"  if get_vgg(q_table)  is not None else "N/A"
                lv_s  = f"{get_vgg(l_table):.3f}"  if get_vgg(l_table)  is not None else "N/A"
                iv_s  = f"{get_vgg(iv_table):.3f}" if get_vgg(iv_table) is not None else "N/A"
                f.write(f"  {tt:<33} {qv_s:>10} {lv_s:>10} {iv_s:>10}\n")

    print(f"Saved to {OUT_PATH}")

    # Print to stdout too
    with open(OUT_PATH) as f:
        print(f.read())


if __name__ == "__main__":
    main()
