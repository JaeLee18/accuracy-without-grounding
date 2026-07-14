# -*- coding: utf-8 -*-
"""
Round 5 Experiments - Five supplementary analyses for ACM MM paper.
Analyses:
  A. Question-level language prior taxonomy (4 regimes)
  B. Answer-flip analysis across CRF levels (Video-MME)
  C. Positional bias analysis (black-screen letter preferences)
  D. VGG decomposition: destructive vs constructive rate per task
  E. Three-way consensus analysis

Data paths:
  results/full_study/  -- Video-MME (6 CRF + black)
  results/mvbench/     -- MVBench (original + crf38 + black)
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------

import json
import os
import sys
import math
from collections import defaultdict, Counter

# -- scipy import (optional) ---------------------------------------------------
try:
    from scipy.stats import chi2_contingency, pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available; chi2 and correlation tests will be skipped.")

# =============================================================================
# DATA LOADING
# =============================================================================

VMME_DIR  = VDG_RESULTS_ROOT + "/full_study"
MVB_DIR   = VDG_RESULTS_ROOT + "/mvbench"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("Loading Video-MME data ...")
qwen_vmme_all   = load_json(f"{VMME_DIR}/qwen2vl_results.json")
qwen_vmme_black = load_json(f"{VMME_DIR}/qwen2vl_black_results.json")
llava_vmme_all  = load_json(f"{VMME_DIR}/llava_results.json")
llava_vmme_black= load_json(f"{VMME_DIR}/llava_black_results.json")
ivl_vmme_all    = load_json(f"{VMME_DIR}/internvl2_results.json")
ivl_vmme_black  = load_json(f"{VMME_DIR}/internvl2_black_results.json")

print("Loading MVBench data ...")
qwen_mvb_all    = load_json(f"{MVB_DIR}/qwen2vl_mvbench_results.json")
llava_mvb_all   = load_json(f"{MVB_DIR}/llava_mvbench_results.json")
ivl_mvb_all     = load_json(f"{MVB_DIR}/internvl2_mvbench_results.json")

# -- convenience helpers -------------------------------------------------------

def index_by_condition(records):
    """Returns {condition: {question_id: record}}."""
    d = defaultdict(dict)
    for r in records:
        d[r["condition"]][r["question_id"]] = r
    return d

def index_single(records):
    """Returns {question_id: record} (single-condition list)."""
    return {r["question_id"]: r for r in records}

# Video-MME indices
qwen_vmme_idx  = index_by_condition(qwen_vmme_all)
llava_vmme_idx = index_by_condition(llava_vmme_all)
ivl_vmme_idx   = index_by_condition(ivl_vmme_all)

qwen_vmme_black_idx  = index_single(qwen_vmme_black)
llava_vmme_black_idx = index_single(llava_vmme_black)
ivl_vmme_black_idx   = index_single(ivl_vmme_black)

# MVBench indices
qwen_mvb_idx   = index_by_condition(qwen_mvb_all)
llava_mvb_idx  = index_by_condition(llava_mvb_all)
ivl_mvb_idx    = index_by_condition(ivl_mvb_all)

VMME_CONDS = ["original", "crf18", "crf23", "crf28", "crf33", "crf38"]
VMME_QIDs  = sorted(qwen_vmme_idx["original"].keys())
MVB_QIDs   = sorted(qwen_mvb_idx["original"].keys())

MODELS_VMME = {
    "Qwen2-VL-7B":  (qwen_vmme_idx,  qwen_vmme_black_idx),
    "LLaVA-Video-7B": (llava_vmme_idx, llava_vmme_black_idx),
    "InternVL2-8B": (ivl_vmme_idx,   ivl_vmme_black_idx),
}
MODELS_MVB = {
    "Qwen2-VL-7B":  qwen_mvb_idx,
    "LLaVA-Video-7B": llava_mvb_idx,
    "InternVL2-8B": ivl_mvb_idx,
}

SEP = "=" * 72

# =============================================================================
# A. QUESTION-LEVEL LANGUAGE PRIOR TAXONOMY
# =============================================================================

print(f"\n{SEP}")
print("ANALYSIS A -- Language Prior Taxonomy (4 Regimes)")
print(SEP)

REGIME_LABELS = {
    "I":   "Correct w/ video, Wrong w/o  (pure visual dependency)",
    "II":  "Correct both ways            (language prior + redundant visual)",
    "III": "Wrong w/ video, Correct w/o  (video hurts -- LP overrides)",
    "IV":  "Wrong both ways              (hard, neither helps)",
}

def classify_regime(correct_video, correct_black):
    if correct_video and not correct_black:
        return "I"
    elif correct_video and correct_black:
        return "II"
    elif not correct_video and correct_black:
        return "III"
    else:
        return "IV"

def regime_analysis(model_name, orig_idx, black_idx, qids, label=""):
    """Returns (regime_counts_overall, regime_counts_per_task)."""
    regime_counts   = Counter()
    task_regime     = defaultdict(Counter)  # task_type -> regime -> count

    for qid in qids:
        if qid not in orig_idx or qid not in black_idx:
            continue
        cv  = orig_idx[qid]["correct"]
        cb  = black_idx[qid]["correct"]
        tt  = orig_idx[qid]["task_type"]
        reg = classify_regime(cv, cb)
        regime_counts[reg] += 1
        task_regime[tt][reg] += 1

    total = sum(regime_counts.values())
    print(f"\n  [{label} -- {model_name}]")
    print(f"  {'Regime':<6} {'N':>6}  {'%':>6}   Description")
    for r in ["I", "II", "III", "IV"]:
        n = regime_counts[r]
        pct = 100 * n / total if total else 0
        print(f"  {r:<6} {n:>6}  {pct:>5.1f}%   {REGIME_LABELS[r]}")

    return regime_counts, task_regime

def chi2_regime_vs_task(task_regime_dict, model_name, bench):
    """Chi-square test: regime distribution differs across task types."""
    task_types = sorted(task_regime_dict.keys())
    regimes    = ["I", "II", "III", "IV"]
    # Build contingency table (tasks x regimes)
    table = []
    for tt in task_types:
        row = [task_regime_dict[tt][r] for r in regimes]
        table.append(row)

    if not HAS_SCIPY:
        return

    import numpy as np
    arr = np.array(table)
    # Drop rows/cols that are all zero to avoid issues
    arr = arr[:, arr.sum(axis=0) > 0]
    arr = arr[arr.sum(axis=1) > 0, :]
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        print(f"  Chi2 skipped (degenerate table) for {model_name} {bench}")
        return

    chi2, p, dof, _ = chi2_contingency(arr)
    print(f"  Chi2 regime x task ({model_name}, {bench}): chi2={chi2:.2f}, dof={dof}, p={p:.4e}")

# -- Video-MME -----------------------------------------------------------------
print("\n--- Video-MME ---")
vmme_regime_by_model = {}
for mname, (cond_idx, black_idx) in MODELS_VMME.items():
    rc, tr = regime_analysis(mname, cond_idx["original"], black_idx, VMME_QIDs, label="Video-MME")
    vmme_regime_by_model[mname] = (rc, tr)
    chi2_regime_vs_task(tr, mname, "Video-MME")

# -- MVBench -------------------------------------------------------------------
print("\n--- MVBench ---")
mvb_regime_by_model = {}
for mname, cond_idx in MODELS_MVB.items():
    orig_idx  = cond_idx["original"]
    black_idx = cond_idx["black"]
    rc, tr = regime_analysis(mname, orig_idx, black_idx, MVB_QIDs, label="MVBench")
    mvb_regime_by_model[mname] = (rc, tr)
    chi2_regime_vs_task(tr, mname, "MVBench")

# -- Pooled % across benchmarks ------------------------------------------------
print("\n  --- Pooled regime % across all models x Video-MME ---")
pool_vmme = Counter()
for rc, _ in vmme_regime_by_model.values():
    pool_vmme.update(rc)
tot = sum(pool_vmme.values())
for r in ["I", "II", "III", "IV"]:
    print(f"  {r}: {pool_vmme[r]:>5}  ({100*pool_vmme[r]/tot:.1f}%)")

print("\n  --- Pooled regime % across all models x MVBench ---")
pool_mvb = Counter()
for rc, _ in mvb_regime_by_model.values():
    pool_mvb.update(rc)
tot = sum(pool_mvb.values())
for r in ["I", "II", "III", "IV"]:
    print(f"  {r}: {pool_mvb[r]:>5}  ({100*pool_mvb[r]/tot:.1f}%)")


# =============================================================================
# B. ANSWER-FLIP ANALYSIS ACROSS CRF LEVELS (Video-MME)
# =============================================================================

print(f"\n{SEP}")
print("ANALYSIS B -- Answer-Flip Across CRF Levels (Video-MME)")
print(SEP)

def flip_analysis(model_name, cond_idx, black_idx, qids):
    """
    For each question, track prediction across 6 CRF levels.
    Returns per-question flip counts and correctness profile.
    """
    flip_counts    = []  # flip count per question
    n_stable       = 0   # 0 flips
    n_one_flip     = 0   # exactly 1 flip
    n_multi_flip   = 0   # 2+ flips
    vgg_by_q       = {}  # question_id -> VGG (1/0 diff)
    flip_by_q      = {}  # question_id -> flip count

    # VGG definition: correct at original - correct at black
    for qid in qids:
        preds = []
        for cond in VMME_CONDS:
            if qid in cond_idx.get(cond, {}):
                preds.append(cond_idx[cond][qid]["prediction"])
            else:
                preds.append(None)

        # Count flips (transitions)
        flips = 0
        for i in range(1, len(preds)):
            if preds[i] is not None and preds[i-1] is not None:
                if preds[i] != preds[i-1]:
                    flips += 1

        # VGG for this question (binary: 1 if orig correct and black wrong, etc.)
        orig_correct  = cond_idx["original"][qid]["correct"] if qid in cond_idx["original"] else None
        black_correct = black_idx[qid]["correct"] if qid in black_idx else None

        if orig_correct is None or black_correct is None:
            continue

        vgg_q = int(orig_correct) - int(black_correct)  # -1, 0, or +1

        flip_counts.append(flips)
        vgg_by_q[qid]  = vgg_q
        flip_by_q[qid] = flips

        if flips == 0:
            n_stable += 1
        elif flips == 1:
            n_one_flip += 1
        else:
            n_multi_flip += 1

    total = n_stable + n_one_flip + n_multi_flip
    print(f"\n  [{model_name}]")
    print(f"  Stable (0 flips):        {n_stable:>4}  ({100*n_stable/total:.1f}%)")
    print(f"  One flip:                {n_one_flip:>4}  ({100*n_one_flip/total:.1f}%)")
    print(f"  Multi-flip (2+):         {n_multi_flip:>4}  ({100*n_multi_flip/total:.1f}%)")

    # Correlation: flip count vs VGG
    if HAS_SCIPY and len(vgg_by_q) > 1:
        common = sorted(set(vgg_by_q.keys()) & set(flip_by_q.keys()))
        fc_arr  = [flip_by_q[q]  for q in common]
        vgg_arr = [vgg_by_q[q]   for q in common]
        r, p = spearmanr(fc_arr, vgg_arr)
        print(f"  Spearman r(flip_count, VGG): r={r:.3f}, p={p:.4e}")
    else:
        print("  Correlation skipped (scipy unavailable).")

    # Flip rate for decision-boundary questions
    # "Decision boundary" = correct at some CRF, wrong at others (mix of T/F across CRF)
    n_boundary   = 0
    n_non_bound  = 0
    flip_boundary = 0.0
    flip_non_bound = 0.0
    for qid in flip_by_q:
        corrects = []
        for cond in VMME_CONDS:
            if qid in cond_idx.get(cond, {}):
                corrects.append(cond_idx[cond][qid]["correct"])
        if len(corrects) < 2:
            continue
        is_boundary = (len(set(corrects)) > 1)  # mixed T/F
        if is_boundary:
            n_boundary += 1
            flip_boundary += flip_by_q[qid]
        else:
            n_non_bound += 1
            flip_non_bound += flip_by_q[qid]

    avg_flip_b  = flip_boundary  / n_boundary   if n_boundary   else 0
    avg_flip_nb = flip_non_bound / n_non_bound   if n_non_bound  else 0
    print(f"  Decision-boundary Qs:    {n_boundary:>4}  avg flips={avg_flip_b:.2f}")
    print(f"  Non-boundary Qs:         {n_non_bound:>4}  avg flips={avg_flip_nb:.2f}")

for mname, (cond_idx, black_idx) in MODELS_VMME.items():
    flip_analysis(mname, cond_idx, black_idx, VMME_QIDs)


# =============================================================================
# C. POSITIONAL BIAS ANALYSIS
# =============================================================================

print(f"\n{SEP}")
print("ANALYSIS C -- Positional Bias (Black-Screen Letter Preferences)")
print(SEP)

LETTERS = ["A", "B", "C", "D"]

def positional_bias(model_name, black_idx, orig_gt_idx, bench_label):
    """
    For black-screen predictions, compute letter preference distribution.
    Test if modal prediction letter matches most frequent GT letter.
    Compute fraction of above-chance black-screen accuracy explained by
    letter alignment (if model always predicted its modal letter).
    """
    pred_counts = Counter()
    gt_counts   = Counter()
    n_correct   = 0
    n_total     = 0

    for qid, rec in black_idx.items():
        raw_pred = rec.get("prediction", "") or ""
        pred = raw_pred.strip().upper()
        if len(pred) >= 1:
            pred_letter = pred[0]
        else:
            pred_letter = "?"
        pred_counts[pred_letter] += 1

        # GT from original (or the black record itself)
        raw_gt = rec.get("ground_truth", "") or ""
        gt = raw_gt.strip().upper()
        if len(gt) >= 1:
            gt_counts[gt[0]] += 1

        if rec.get("correct", False):
            n_correct += 1
        n_total += 1

    modal_pred = pred_counts.most_common(1)[0][0] if pred_counts else "?"
    modal_gt   = gt_counts.most_common(1)[0][0]   if gt_counts   else "?"

    # Chance level = 1/4 for 4-choice
    chance_acc = 0.25
    actual_acc = n_correct / n_total if n_total else 0
    above_chance_gap = actual_acc - chance_acc

    # If model always predicted modal letter, what accuracy would it get?
    n_gt_modal = gt_counts.get(modal_pred, 0)
    modal_acc  = n_gt_modal / n_total if n_total else 0
    modal_above_chance = modal_acc - chance_acc

    # Fraction of above-chance accuracy explained by letter bias
    if above_chance_gap > 0:
        frac_explained = modal_above_chance / above_chance_gap
    else:
        frac_explained = float("nan")

    print(f"\n  [{model_name} -- {bench_label}]")
    print(f"  Black-screen accuracy:   {actual_acc:.3f} (chance=0.250, gap={above_chance_gap:+.3f})")
    print(f"  Prediction distribution: " +
          ", ".join(f"{k}:{pred_counts[k]}" for k in LETTERS if k in pred_counts))
    print(f"  GT distribution:         " +
          ", ".join(f"{k}:{gt_counts[k]}"   for k in LETTERS if k in gt_counts))
    print(f"  Modal prediction letter: {modal_pred}  |  Modal GT letter: {modal_gt}")
    print(f"  Match (modal pred == modal GT): {modal_pred == modal_gt}")
    if not math.isnan(frac_explained):
        print(f"  Acc if always predict {modal_pred}: {modal_acc:.3f}  =>  {100*frac_explained:.1f}% of above-chance gap explained")
    else:
        print(f"  Above-chance gap <= 0; letter-alignment fraction N/A")

    return {
        "pred_dist": dict(pred_counts), "gt_dist": dict(gt_counts),
        "modal_pred": modal_pred, "modal_gt": modal_gt,
        "actual_acc": actual_acc, "modal_acc": modal_acc,
        "frac_explained": frac_explained,
    }

print("\n--- Video-MME ---")
for mname, (cond_idx, black_idx) in MODELS_VMME.items():
    positional_bias(mname, black_idx, cond_idx["original"], "Video-MME")

print("\n--- MVBench ---")
for mname, cond_idx in MODELS_MVB.items():
    black_idx = cond_idx["black"]
    positional_bias(mname, black_idx, cond_idx["original"], "MVBench")


# =============================================================================
# D. VGG DECOMPOSITION -- DESTRUCTIVE vs CONSTRUCTIVE RATE
# =============================================================================

print(f"\n{SEP}")
print("ANALYSIS D -- VGG Decomposition: Destructive vs Constructive Rate")
print(SEP)

# VGG = P(orig_correct, black_wrong) - P(orig_wrong, black_correct)
#      = destructive_rate - constructive_rate

def vgg_decompose(model_name, orig_idx, black_idx, qids, bench_label):
    task_stats = defaultdict(lambda: {"D": 0, "C": 0, "n": 0})  # D=destructive, C=constructive

    for qid in qids:
        if qid not in orig_idx or qid not in black_idx:
            continue
        cv  = orig_idx[qid]["correct"]
        cb  = black_idx[qid]["correct"]
        tt  = orig_idx[qid]["task_type"]
        task_stats[tt]["n"] += 1
        if cv and not cb:
            task_stats[tt]["D"] += 1
        elif not cv and cb:
            task_stats[tt]["C"] += 1

    total_n = sum(s["n"] for s in task_stats.values())
    total_D = sum(s["D"] for s in task_stats.values())
    total_C = sum(s["C"] for s in task_stats.values())

    print(f"\n  [{model_name} -- {bench_label}]")
    print(f"  {'Task Type':<35} {'n':>5}  {'D_rate':>7}  {'C_rate':>7}  {'VGG':>7}  {'type':>10}")

    for tt in sorted(task_stats.keys()):
        s  = task_stats[tt]
        n  = s["n"]
        dr = s["D"] / n if n else 0
        cr = s["C"] / n if n else 0
        vgg = dr - cr
        # characterise
        if dr > 0.05 and cr < 0.03:
            tag = "pure-dest"
        elif dr > 0.05 and cr > 0.03:
            tag = "both"
        elif dr < 0.03 and cr > 0.05:
            tag = "pure-const"
        else:
            tag = "neutral"
        print(f"  {tt:<35} {n:>5}  {dr:>7.3f}  {cr:>7.3f}  {vgg:>+7.3f}  {tag:>10}")

    print(f"  {'TOTAL':<35} {total_n:>5}  {total_D/total_n:>7.3f}  {total_C/total_n:>7.3f}  {(total_D-total_C)/total_n:>+7.3f}")

print("\n--- Video-MME ---")
for mname, (cond_idx, black_idx) in MODELS_VMME.items():
    vgg_decompose(mname, cond_idx["original"], black_idx, VMME_QIDs, "Video-MME")

print("\n--- MVBench ---")
for mname, cond_idx in MODELS_MVB.items():
    vgg_decompose(mname, cond_idx["original"], cond_idx["black"], MVB_QIDs, "MVBench")


# =============================================================================
# E. THREE-WAY CONSENSUS ANALYSIS
# =============================================================================

print(f"\n{SEP}")
print("ANALYSIS E -- Three-Way Consensus Analysis")
print(SEP)

def consensus_analysis(qids, model_orig_idxs, model_black_idxs, bench_label,
                       model_names):
    """
    For each question, count how many models answer correctly with video
    vs without video.
    """
    # consensus counts: n_correct_video -> count, n_correct_black -> count
    video_consensus  = Counter()   # 0..3
    black_consensus  = Counter()   # 0..3

    # transition matrix: video_count -> black_count -> n
    transition = defaultdict(Counter)   # transition[v][b] += 1

    # For HC: questions where all 3 fail on black
    all_fail_black   = 0
    all_correct_vid  = 0
    both_all         = 0   # 3/3 video AND 3/3 black -> trivial questions
    pure_visual_all  = 0   # 3/3 video AND 0/3 black

    for qid in qids:
        n_vid = 0
        n_blk = 0
        valid = True
        for oi, bi in zip(model_orig_idxs, model_black_idxs):
            if qid not in oi or qid not in bi:
                valid = False
                break
            if oi[qid]["correct"]:
                n_vid += 1
            if bi[qid]["correct"]:
                n_blk += 1
        if not valid:
            continue

        video_consensus[n_vid] += 1
        black_consensus[n_blk] += 1
        transition[n_vid][n_blk] += 1

        if n_blk == 0:
            all_fail_black += 1
        if n_vid == 3:
            all_correct_vid += 1
        if n_vid == 3 and n_blk == 3:
            both_all += 1
        if n_vid == 3 and n_blk == 0:
            pure_visual_all += 1

    total = sum(video_consensus.values())

    print(f"\n  [{bench_label}]")
    print(f"\n  Video-correct consensus:")
    for k in range(4):
        n = video_consensus[k]
        pct = 100 * n / total if total else 0
        # what % of these have all fail on black?
        all_blk_fail_here = transition[k][0]
        pct_hc = 100 * all_blk_fail_here / n if n else 0
        print(f"    {k}/3 models correct w/ video:  n={n:>4}  ({pct:>5.1f}%)  -- "
              f"{all_blk_fail_here} ({pct_hc:.1f}%) have 0/3 correct on black")

    print(f"\n  Black-correct consensus:")
    for k in range(4):
        n = black_consensus[k]
        pct = 100 * n / total if total else 0
        print(f"    {k}/3 models correct w/ black:  n={n:>4}  ({pct:>5.1f}%)")

    print(f"\n  Transition matrix (rows=video_count, cols=black_count):")
    header = "  " + " " * 20 + "  ".join(f"black={k}" for k in range(4))
    print(header)
    for v in range(4):
        row_total = sum(transition[v].values())
        row_str = f"  video={v}  (n={row_total:>4})  " + \
                  "    ".join(f"{transition[v][b]:>5}" for b in range(4))
        print(row_str)

    print(f"\n  Summary:")
    print(f"  3/3 correct w/ video:       {all_correct_vid:>4}  ({100*all_correct_vid/total:.1f}%)")
    print(f"    of which 3/3 also black:  {both_all:>4}  ({100*both_all/total:.1f}%) [trivial]")
    print(f"    of which 0/3 on black:    {pure_visual_all:>4}  ({100*pure_visual_all/total:.1f}%) [pure visual]")
    print(f"  0/3 correct even w/ video:  {video_consensus[0]:>4}  ({100*video_consensus[0]/total:.1f}%) [beyond all models]")
    print(f"  0/3 correct on black:       {all_fail_black:>4}  ({100*all_fail_black/total:.1f}%) [HC-eligible]")

    # Net gain from video vs black: E[n_correct_video] - E[n_correct_black]
    ev_vid = sum(k * video_consensus[k] for k in range(4)) / total
    ev_blk = sum(k * black_consensus[k] for k in range(4)) / total
    print(f"\n  E[correct_models | video] = {ev_vid:.3f}")
    print(f"  E[correct_models | black] = {ev_blk:.3f}")
    print(f"  Average video benefit:      {ev_vid - ev_blk:+.3f} extra models correct per question")

print("\n--- Video-MME ---")
consensus_analysis(
    VMME_QIDs,
    [qwen_vmme_idx["original"], llava_vmme_idx["original"], ivl_vmme_idx["original"]],
    [qwen_vmme_black_idx, llava_vmme_black_idx, ivl_vmme_black_idx],
    "Video-MME",
    ["Qwen2-VL-7B", "LLaVA-Video-7B", "InternVL2-8B"],
)

print("\n--- MVBench ---")
consensus_analysis(
    MVB_QIDs,
    [qwen_mvb_idx["original"], llava_mvb_idx["original"], ivl_mvb_idx["original"]],
    [qwen_mvb_idx["black"],    llava_mvb_idx["black"],    ivl_mvb_idx["black"]],
    "MVBench",
    ["Qwen2-VL-7B", "LLaVA-Video-7B", "InternVL2-8B"],
)

print(f"\n{SEP}")
print("ALL ANALYSES COMPLETE")
print(SEP)
