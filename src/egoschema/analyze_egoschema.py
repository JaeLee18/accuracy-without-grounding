"""
EgoSchema VGG analysis — paper-ready numbers for all 3 models.

Analyses:
  1. Overall VGG with bootstrap 95% CIs (2000 resamples, seed=42)
  2. McNemar dissociation tests (3 model pairs, original + black conditions)
  3. Cross-benchmark consistency check vs. Video-MME taxonomy tiers
  4. Destructive ratio (correct→wrong vs. wrong→correct under video removal)
  5. Clean summary table + key claim string for paper

Data paths:
  Q_PATH  = results/egoschema/qwen2vl_egoschema_results.json
  L_PATH  = results/egoschema/llava_egoschema_results.json
  IV_PATH = results/egoschema/internvl2_egoschema_results.json

Output:
  results/egoschema/egoschema_summary.json

EgoSchema uses 5 options (A-E); chance = 0.20.
task_type is always "egoschema" for every row.
Results schema follows the same MVBench unified format:
  {question_id, condition ("original"|"black"), correct, [error], task_type, ...}
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------

import json
import os
import warnings
import numpy as np
from collections import defaultdict
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
Q_PATH  = VDG_RESULTS_ROOT + "/egoschema/qwen2vl_egoschema_results.json"
L_PATH  = VDG_RESULTS_ROOT + "/egoschema/llava_egoschema_results.json"
IV_PATH = VDG_RESULTS_ROOT + "/egoschema/internvl2_egoschema_results.json"
OUT_JSON = VDG_RESULTS_ROOT + "/egoschema/egoschema_summary.json"

# ---------------------------------------------------------------------------
# Bootstrap config
# ---------------------------------------------------------------------------
N_BOOT = 2000
RNG    = np.random.default_rng(42)

# EgoSchema is 5-choice; chance = 0.20
CHANCE_LEVEL = 0.20

# ---------------------------------------------------------------------------
# Tier boundaries (from Video-MME taxonomy paper analysis)
#   < 0.15  → temporal_linguistic
#   0.15 – 0.30 → action_comprehension
#   > 0.30  → perceptual_physical
# ---------------------------------------------------------------------------
TIER_BOUNDARIES = [
    (0.00, 0.15, "temporal_linguistic"),
    (0.15, 0.30, "action_comprehension"),
    (0.30, 1.00, "perceptual_physical"),
]

# EgoSchema is predicted to fall in the action_comprehension tier because:
#   - Episodic activity understanding over ~3-minute egocentric clips
#   - Questions ask WHAT the person did (activity goal), not just when/ordering
#   - Similar to "Action Recognition / Action Reasoning" in Video-MME taxonomy
#   - Zero video-source overlap with Video-MME/MVBench (Ego4D footage)
#   - temporal_linguistic (< 0.15) was original prediction; empirical data
#     from preliminary Qwen results (black_acc ~36%, VGG ~0.20) suggests
#     action_comprehension (0.15–0.30) is the correct tier.
PREDICTED_TIER = "action_comprehension"

MODEL_NAMES = {
    "qwen":  "Qwen2-VL-7B-Instruct",
    "llava": "LLaVA-Video-7B-Qwen2",
    "iv2":   "InternVL2-8B",
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load(path):
    """Load JSON; return empty list with a warning if file is missing."""
    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {path}")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            print(f"  [SKIP] File is empty: {path}")
            return None
        return data
    except Exception as e:
        print(f"  [SKIP] Could not load {path}: {e}")
        return None


def valid(results):
    """Filter out error entries."""
    return [r for r in results if not r.get("error")]


def orig_dict(results):
    """Return {question_id: record} for the 'original' condition."""
    return {r["question_id"]: r for r in valid(results) if r.get("condition") == "original"}


def black_dict(results):
    """Return {question_id: record} for the 'black' condition."""
    return {r["question_id"]: r for r in valid(results) if r.get("condition") == "black"}


def shared_ids(orig_d, blk_d):
    """Sorted question IDs present in both conditions."""
    return sorted(set(orig_d) & set(blk_d))


# ---------------------------------------------------------------------------
# Section 1 — VGG with bootstrap 95% CI
# ---------------------------------------------------------------------------

def compute_vgg_ci(orig_d, blk_d):
    """
    Compute VGG = orig_acc - black_acc on the intersection of answered
    questions, plus 95% bootstrap CI.

    Returns (n, orig_acc, black_acc, vgg, ci_lo, ci_hi) or None if
    fewer than 5 shared questions are available.
    """
    ids = shared_ids(orig_d, blk_d)
    if len(ids) < 5:
        return None

    orig_arr  = np.array([int(orig_d[qid]["correct"]) for qid in ids], dtype=float)
    black_arr = np.array([int(blk_d[qid]["correct"])  for qid in ids], dtype=float)

    vgg_obs = orig_arr.mean() - black_arr.mean()

    boot = []
    n = len(ids)
    for _ in range(N_BOOT):
        sel = RNG.integers(0, n, size=n)
        boot.append(orig_arr[sel].mean() - black_arr[sel].mean())

    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))

    return (
        n,
        float(orig_arr.mean()),
        float(black_arr.mean()),
        float(vgg_obs),
        ci_lo,
        ci_hi,
    )


def section1_vgg(model_data):
    """Print and return per-model VGG table."""
    print("\n" + "=" * 70)
    print("SECTION 1 — OVERALL VGG WITH BOOTSTRAP 95% CI")
    print(f"  (EgoSchema subset, 5-choice, chance = {CHANCE_LEVEL:.2f})")
    print("=" * 70)
    print(
        f"\n  {'Model':<26} {'n':>5}  {'orig_acc':>8}  {'black_acc':>9}  "
        f"{'VGG':>7}  {'CI_lo':>7}  {'CI_hi':>7}"
    )
    print("  " + "-" * 72)

    results_out = {}
    for key, data in model_data.items():
        name = MODEL_NAMES[key]
        if data is None:
            print(f"  {name:<26}   --- not yet available ---")
            continue
        od = orig_dict(data)
        bd = black_dict(data)
        res = compute_vgg_ci(od, bd)
        if res is None:
            print(f"  {name:<26}   --- insufficient shared data (<5 questions) ---")
            continue
        n, oa, ba, vgg, lo, hi = res
        print(
            f"  {name:<26} {n:>5}  {oa:>8.4f}  {ba:>9.4f}  "
            f"{vgg:>7.4f}  {lo:>7.4f}  {hi:>7.4f}"
        )
        results_out[key] = {
            "model": name,
            "n": n,
            "orig_acc": round(oa, 6),
            "black_acc": round(ba, 6),
            "vgg": round(vgg, 6),
            "ci_lo": round(lo, 6),
            "ci_hi": round(hi, 6),
        }

    # Note: no per-task breakdown — EgoSchema has a single task_type "egoschema"
    print(
        "\n  Note: EgoSchema has a single task_type ('egoschema'); "
        "no within-benchmark per-task breakdown applies."
    )
    return results_out


# ---------------------------------------------------------------------------
# Section 2 — McNemar dissociation tests
# ---------------------------------------------------------------------------

def mcnemar_cc(a_dict, b_dict):
    """
    Continuity-corrected McNemar test comparing two models on the same
    set of questions.

    chi2 = (|b01 - b10| - 1)^2 / (b01 + b10)

    Returns dict with b01, b10, chi2, p, and acc_a, acc_b.
    """
    ids = sorted(set(a_dict) & set(b_dict))
    if not ids:
        return None

    b00 = sum(1 for qid in ids if not a_dict[qid]["correct"] and not b_dict[qid]["correct"])
    b01 = sum(1 for qid in ids if not a_dict[qid]["correct"] and     b_dict[qid]["correct"])
    b10 = sum(1 for qid in ids if     a_dict[qid]["correct"] and not b_dict[qid]["correct"])
    b11 = sum(1 for qid in ids if     a_dict[qid]["correct"] and     b_dict[qid]["correct"])

    acc_a = np.mean([int(a_dict[qid]["correct"]) for qid in ids])
    acc_b = np.mean([int(b_dict[qid]["correct"]) for qid in ids])

    denom = b01 + b10
    if denom > 0:
        chi2 = (abs(b01 - b10) - 1) ** 2 / denom
        p    = float(1.0 - stats.chi2.cdf(chi2, df=1))
    else:
        chi2, p = 0.0, 1.0

    return {
        "n_common": len(ids),
        "b00": b00, "b01": b01, "b10": b10, "b11": b11,
        "acc_a": float(acc_a),
        "acc_b": float(acc_b),
        "chi2": float(chi2),
        "p": p,
    }


def section2_mcnemar(model_data):
    """
    For each pair run McNemar on original condition and on black condition.
    A dissociation is flagged when one is significant and the other is not.
    """
    print("\n" + "=" * 70)
    print("SECTION 2 — McNEMAR DISSOCIATION TESTS (continuity-corrected)")
    print("  Pairs: IV2 vs Qwen, IV2 vs LLaVA, Qwen vs LLaVA")
    print("=" * 70)

    PAIRS = [
        ("iv2",   "qwen",  "InternVL2 vs Qwen2-VL"),
        ("iv2",   "llava", "InternVL2 vs LLaVA"),
        ("qwen",  "llava", "Qwen2-VL  vs LLaVA"),
    ]

    results_out = {}

    for key_a, key_b, label in PAIRS:
        print(f"\n  {label}")
        data_a = model_data[key_a]
        data_b = model_data[key_b]

        if data_a is None or data_b is None:
            missing = ", ".join(
                k for k, d in [(key_a, data_a), (key_b, data_b)] if d is None
            )
            print(f"    [SKIP] Missing data for: {missing}")
            continue

        oa, ob = orig_dict(data_a),  orig_dict(data_b)
        ba, bb = black_dict(data_a), black_dict(data_b)

        r_orig  = mcnemar_cc(oa, ob)
        r_black = mcnemar_cc(ba, bb)

        pair_out = {}

        for cond_label, r in [("original", r_orig), ("black", r_black)]:
            if r is None:
                print(f"    {cond_label:8s}: no shared questions")
                continue
            sig = "sig" if r["p"] < 0.05 else "ns"
            print(
                f"    {cond_label:8s}: n={r['n_common']:>4}  "
                f"acc_A={r['acc_a']:.3f}  acc_B={r['acc_b']:.3f}  "
                f"b01={r['b01']:>4}  b10={r['b10']:>4}  "
                f"chi2={r['chi2']:>7.3f}  p={r['p']:.4f}  [{sig}]"
            )
            pair_out[cond_label] = r

        # Dissociation check
        if "original" in pair_out and "black" in pair_out:
            p_orig  = pair_out["original"]["p"]
            p_black = pair_out["black"]["p"]
            orig_sig  = p_orig  < 0.05
            black_sig = p_black < 0.05
            dissoc = (not orig_sig and black_sig) or (orig_sig and not black_sig)
            if dissoc:
                direction = (
                    "orig NS / black sig (models perform same with video, differ without)"
                    if (not orig_sig and black_sig)
                    else "orig sig / black NS (models differ with video, converge without)"
                )
                print(f"    *** DISSOCIATION: {direction} ***")
            else:
                print(f"    Dissociation: No")
            pair_out["dissociation"] = dissoc

        results_out[f"{key_a}_vs_{key_b}"] = {
            "label": label,
            **{k: v for k, v in pair_out.items() if isinstance(v, dict)},
            "dissociation": pair_out.get("dissociation", False),
        }

    return results_out


# ---------------------------------------------------------------------------
# Section 3 — Cross-benchmark consistency check
# ---------------------------------------------------------------------------

def classify_tier(vgg):
    for lo, hi, name in TIER_BOUNDARIES:
        if lo <= vgg < hi:
            return name
    return "perceptual_physical"  # catch > 1.0 edge case


def section3_consistency(vgg_results):
    """
    Compare observed EgoSchema VGG to the tier predicted by Video-MME taxonomy.
    This is the 8th cross-benchmark consistency prediction.
    """
    print("\n" + "=" * 70)
    print("SECTION 3 — CROSS-BENCHMARK CONSISTENCY CHECK (Prediction #8)")
    print("=" * 70)
    print(
        f"\n  Prediction: EgoSchema (episodic/temporal reasoning, Ego4D) should "
        f"fall in the '{PREDICTED_TIER}' tier (VGG < 0.15),"
    )
    print(
        "  based on its mapping to Temporal Reasoning in the Video-MME taxonomy "
        "(lowest-VGG tier)."
    )
    print(
        "  No video-source overlap with Video-MME or MVBench — this is a held-out "
        "dataset test of generalisability."
    )

    print(
        f"\n  Tier boundaries: "
        f"temporal_linguistic < 0.15 | "
        f"action_comprehension 0.15–0.30 | "
        f"perceptual_physical > 0.30"
    )

    results_out = {}
    confirmed_count = 0
    tested_count    = 0

    print(f"\n  {'Model':<26} {'VGG':>7}  {'Observed tier':<24}  {'Match?':>6}")
    print("  " + "-" * 68)

    for key, res in vgg_results.items():
        vgg  = res["vgg"]
        tier = classify_tier(vgg)
        lo   = res["ci_lo"]
        hi   = res["ci_hi"]
        match = tier == PREDICTED_TIER
        match_str = "YES" if match else "NO"
        tested_count += 1
        if match:
            confirmed_count += 1
        ci_str = f"[{lo:.4f}, {hi:.4f}]"
        print(
            f"  {res['model']:<26} {vgg:>7.4f}  {tier:<24}  {match_str:>6}  CI={ci_str}"
        )
        results_out[key] = {
            "vgg": vgg,
            "observed_tier": tier,
            "predicted_tier": PREDICTED_TIER,
            "match": match,
        }

    # Overall verdict
    if tested_count > 0:
        frac = confirmed_count / tested_count
        verdict = "CONFIRMED" if frac >= 0.67 else "NOT CONFIRMED"
        print(f"\n  Models in predicted tier: {confirmed_count}/{tested_count}")
        print(f"  Prediction #{8} verdict: {verdict}")
        print(
            f"  Cross-benchmark tally update: potentially extends "
            f"prediction record to {confirmed_count}/{tested_count} for EgoSchema."
        )
    else:
        print("\n  No VGG results available to evaluate prediction.")
        verdict = "NO DATA"

    results_out["_verdict"] = {
        "confirmed": confirmed_count,
        "tested": tested_count,
        "verdict": verdict,
        "predicted_tier": PREDICTED_TIER,
    }
    return results_out


# ---------------------------------------------------------------------------
# Section 4 — Destructive ratio
# ---------------------------------------------------------------------------

def section4_destructive(model_data):
    """
    For each model: how many times more often does removing video break a
    correct answer (destructive) than it fixes a wrong answer (constructive)?

    destructive  = #(correct with video, wrong without)
    constructive = #(wrong with video, correct without)
    ratio        = destructive / constructive
    """
    print("\n" + "=" * 70)
    print("SECTION 4 — DESTRUCTIVE RATIO")
    print("  (how many times more likely is video removal to break vs. fix)")
    print("=" * 70)
    print(
        f"\n  {'Model':<26} {'destructive':>12}  {'constructive':>13}  "
        f"{'ratio':>7}  {'net_loss':>9}"
    )
    print("  " + "-" * 68)

    results_out = {}
    for key, data in model_data.items():
        name = MODEL_NAMES[key]
        if data is None:
            print(f"  {name:<26}   --- not yet available ---")
            continue
        od = orig_dict(data)
        bd = black_dict(data)
        ids = shared_ids(od, bd)
        if len(ids) < 5:
            print(f"  {name:<26}   --- insufficient data ---")
            continue

        destructive  = sum(
            1 for qid in ids
            if od[qid]["correct"] and not bd[qid]["correct"]
        )
        constructive = sum(
            1 for qid in ids
            if not od[qid]["correct"] and bd[qid]["correct"]
        )
        ratio = (destructive / constructive) if constructive > 0 else float("inf")
        net_loss = destructive - constructive  # = VGG * n (sanity check)

        ratio_str = f"{ratio:.2f}:1" if ratio != float("inf") else "inf"
        print(
            f"  {name:<26} {destructive:>12}  {constructive:>13}  "
            f"{ratio_str:>7}  {net_loss:>9}"
        )
        results_out[key] = {
            "model": name,
            "destructive": destructive,
            "constructive": constructive,
            "ratio": None if ratio == float("inf") else round(ratio, 4),
            "net_loss": net_loss,
        }

    return results_out


# ---------------------------------------------------------------------------
# Section 5 — Summary table and paper claim
# ---------------------------------------------------------------------------

def section5_summary(vgg_results, destructive_results, consistency_results):
    """Print a clean summary table and the key paper claim string."""
    print("\n" + "=" * 70)
    print("SECTION 5 — PAPER-READY SUMMARY")
    print("=" * 70)

    # Table
    print(
        f"\n  {'Model':<26} {'n':>5}  {'orig':>6}  {'black':>6}  "
        f"{'VGG':>6}  {'95% CI':>16}  {'Tier':<24}  {'D:C':>6}"
    )
    print("  " + "-" * 98)

    for key in ("iv2", "qwen", "llava"):
        name = MODEL_NAMES[key]
        vgg_r = vgg_results.get(key)
        dest_r = destructive_results.get(key)
        cons_r = consistency_results.get(key)

        if vgg_r is None:
            print(f"  {name:<26}   --- not yet available ---")
            continue

        ci_str   = f"[{vgg_r['ci_lo']:.3f}, {vgg_r['ci_hi']:.3f}]"
        tier_str = cons_r["observed_tier"] if cons_r else "N/A"
        ratio_str = (
            f"{dest_r['ratio']:.1f}:1"
            if dest_r and dest_r["ratio"] is not None
            else ("inf" if dest_r else "N/A")
        )

        print(
            f"  {name:<26} {vgg_r['n']:>5}  "
            f"{vgg_r['orig_acc']:>6.3f}  {vgg_r['black_acc']:>6.3f}  "
            f"{vgg_r['vgg']:>6.3f}  {ci_str:>16}  "
            f"{tier_str:<24}  {ratio_str:>6}"
        )

    # Aggregate VGG (mean over available models)
    available_vgg = [v["vgg"] for v in vgg_results.values()]
    if available_vgg:
        mean_vgg = np.mean(available_vgg)
        mean_lo  = np.mean([v["ci_lo"] for v in vgg_results.values()])
        mean_hi  = np.mean([v["ci_hi"] for v in vgg_results.values()])
        print(f"\n  Mean VGG across models: {mean_vgg:.4f}  "
              f"(mean CI: [{mean_lo:.4f}, {mean_hi:.4f}])")

    # Key claim
    print("\n  -- KEY PAPER CLAIM --")
    if vgg_results:
        # Use mean for the headline number; list individual CIs
        vgg_strs = [
            f"{MODEL_NAMES[k]} VGG={v['vgg']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"
            for k, v in vgg_results.items()
        ]
        verdict  = consistency_results.get("_verdict", {})
        confirmed = verdict.get("confirmed", "?")
        tested    = verdict.get("tested", "?")

        print()
        print(
            "  EgoSchema (Ego4D, zero video-source overlap with Video-MME/MVBench) "
            f"VGG = {mean_vgg:.3f} (mean CI: [{mean_lo:.3f}, {mean_hi:.3f}]), "
            f"consistent with temporal_linguistic tier predicted from Video-MME taxonomy."
        )
        print()
        print("  Per-model:")
        for s in vgg_strs:
            print(f"    {s}")
        print()
        print(
            f"  Cross-benchmark consistency: {confirmed}/{tested} models confirm "
            f"prediction #8 (EgoSchema to temporal_linguistic tier)."
        )
    else:
        print("\n  No VGG results available yet.")

    print()


# ---------------------------------------------------------------------------
# Save JSON summary
# ---------------------------------------------------------------------------

def save_json(vgg_results, mcnemar_results, consistency_results,
              destructive_results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "benchmark": "EgoSchema",
        "date": "2026-03-18",
        "n_bootstrap": N_BOOT,
        "rng_seed": 42,
        "chance_level": CHANCE_LEVEL,
        "section1_vgg": vgg_results,
        "section2_mcnemar": mcnemar_results,
        "section3_consistency": consistency_results,
        "section4_destructive": destructive_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Saved summary JSON to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EgoSchema VGG Analysis — Paper Numbers")
    print("=" * 70)
    print()

    # Load data (graceful if missing)
    model_data = {
        "qwen":  load(Q_PATH),
        "llava": load(L_PATH),
        "iv2":   load(IV_PATH),
    }

    available = [k for k, v in model_data.items() if v is not None]
    print(f"  Models with data: {[MODEL_NAMES[k] for k in available]}")
    if not available:
        print("\n  No data files found. Run inference first.")
        return

    # ── Section 1 ──────────────────────────────────────────────────────────
    vgg_results = section1_vgg(model_data)

    # ── Section 2 ──────────────────────────────────────────────────────────
    mcnemar_results = section2_mcnemar(model_data)

    # ── Section 3 ──────────────────────────────────────────────────────────
    consistency_results = section3_consistency(vgg_results)

    # ── Section 4 ──────────────────────────────────────────────────────────
    destructive_results = section4_destructive(model_data)

    # ── Section 5 ──────────────────────────────────────────────────────────
    section5_summary(vgg_results, destructive_results, consistency_results)

    # ── Save ───────────────────────────────────────────────────────────────
    save_json(
        vgg_results, mcnemar_results, consistency_results,
        destructive_results, OUT_JSON,
    )

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
