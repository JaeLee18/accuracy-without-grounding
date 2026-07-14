"""
Step 5: Analyze pilot results. Compute accuracy, degradation, statistical tests, and plots.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

RESULTS_PATH = VDG_RESULTS_ROOT + "/pilot_results.json"
PLOT_DIR = VDG_RESULTS_ROOT + "/plots"
CONDITIONS = ["original", "crf18", "crf38"]
CRF_MAP = {"original": 0, "crf18": 18, "crf38": 38}


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # Filter out errors
    errors = df[df["prediction"].isna()]
    if len(errors) > 0:
        print(f"WARNING: {len(errors)} failed inference results (excluded from analysis)")
        df = df[df["prediction"].notna()].copy()

    task_types = sorted(df["task_type"].unique())
    print(f"Task types: {task_types}")
    print(f"Total valid results: {len(df)}")

    # Sanity check: at least 20 per task type
    for tt in task_types:
        for cond in CONDITIONS:
            n = len(df[(df.task_type == tt) & (df.condition == cond)])
            if n < 20:
                print(f"ALERT: {tt}/{cond} has only {n} results (< 20)")

    # === Compute accuracy ===
    acc_table = {}
    for tt in task_types:
        acc_table[tt] = {}
        for cond in CONDITIONS:
            sub = df[(df.task_type == tt) & (df.condition == cond)]
            acc = sub["correct"].mean() if len(sub) > 0 else 0
            acc_table[tt][cond] = acc

    # === Print accuracy table ===
    print(f"\n{'='*60}")
    print("ABSOLUTE ACCURACY")
    print(f"{'='*60}")
    print(f"{'Task Type':<25} | {'original':>8} | {'crf18':>8} | {'crf38':>8}")
    print("-" * 60)
    for tt in task_types:
        vals = [f"{acc_table[tt][c]:.3f}" for c in CONDITIONS]
        print(f"{tt:<25} | {vals[0]:>8} | {vals[1]:>8} | {vals[2]:>8}")

    # === Relative degradation ===
    print(f"\n{'='*60}")
    print("RELATIVE DEGRADATION (CRF 38 vs Original)")
    print(f"{'='*60}")
    degradation = {}
    for tt in task_types:
        orig = acc_table[tt]["original"]
        crf38 = acc_table[tt]["crf38"]
        if orig > 0:
            deg = (orig - crf38) / orig * 100
        else:
            deg = 0.0
        degradation[tt] = deg
        print(f"  {tt}: {deg:+.1f}%")

    # === Statistical test ===
    # Compare degradation between OCR and Action Recognition using Fisher's exact test
    # Build 2x2 contingency table: correct/incorrect at original vs crf38 for each type
    print(f"\n{'='*60}")
    print("STATISTICAL TEST")
    print(f"{'='*60}")

    # For each task type, compute the difference in correct answers between original and crf38
    # Then test if the degradation pattern differs between task types
    stat_results = {}
    for tt in task_types:
        orig_sub = df[(df.task_type == tt) & (df.condition == "original")]
        crf38_sub = df[(df.task_type == tt) & (df.condition == "crf38")]
        stat_results[tt] = {
            "orig_correct": int(orig_sub["correct"].sum()),
            "orig_total": len(orig_sub),
            "crf38_correct": int(crf38_sub["correct"].sum()),
            "crf38_total": len(crf38_sub),
        }

    # Build a 2x2 table for interaction test:
    # Rows: task type (OCR vs Action Recognition)
    # We test: does compression affect OCR more than Action Recognition?
    # Use Breslow-Day or simpler: compare the proportion of "degraded" answers

    # Approach: For each task type, build a 2x2 of (correct, incorrect) x (original, crf38)
    # Then use Cochran-Mantel-Haenszel or just compare degradation with Fisher's

    # Simpler: test if the drop in accuracy is significantly different between task types
    # using a chi-square test on the interaction

    tt_list = task_types  # should be 2 types
    if len(tt_list) == 2:
        # Build combined 2x2x2: task_type x condition x correct
        # Flatten to test interaction
        tables = []
        for tt in tt_list:
            s = stat_results[tt]
            # [correct_orig, incorrect_orig], [correct_crf38, incorrect_crf38]
            table = [
                [s["orig_correct"], s["orig_total"] - s["orig_correct"]],
                [s["crf38_correct"], s["crf38_total"] - s["crf38_correct"]]
            ]
            tables.append(np.array(table))

        # Test interaction: does compression affect the two task types differently?
        # Breslow-Day test for homogeneity of odds ratios
        # Simple approach: compute odds ratio for each task type and compare

        or_values = []
        for i, tt in enumerate(tt_list):
            t = tables[i]
            # Odds ratio: (correct_orig * incorrect_crf38) / (incorrect_orig * correct_crf38)
            a, b = t[0]  # correct_orig, incorrect_orig
            c, d = t[1]  # correct_crf38, incorrect_crf38
            # Add 0.5 correction to avoid division by zero
            or_val = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
            or_values.append(or_val)
            print(f"  {tt}: OR = {or_val:.3f}")

        # Direct approach: test if degradation proportions differ
        # Build 2x2: (degraded, not_degraded) x (OCR, Action Recognition)
        # A question "degraded" if it was correct at original but incorrect at crf38
        degraded_counts = []
        not_degraded_counts = []
        for tt in tt_list:
            orig = df[(df.task_type == tt) & (df.condition == "original")].set_index("question_id")
            crf38 = df[(df.task_type == tt) & (df.condition == "crf38")].set_index("question_id")
            common_ids = orig.index.intersection(crf38.index)
            degraded = 0
            not_degraded = 0
            for qid in common_ids:
                was_correct = orig.loc[qid, "correct"]
                now_correct = crf38.loc[qid, "correct"]
                if was_correct and not now_correct:
                    degraded += 1
                else:
                    not_degraded += 1
            degraded_counts.append(degraded)
            not_degraded_counts.append(not_degraded)

        contingency = np.array([degraded_counts, not_degraded_counts])
        print(f"\n  Contingency table (degraded vs not, by task type):")
        print(f"  {'':>15} {tt_list[0]:<20} {tt_list[1]:<20}")
        print(f"  {'Degraded':>15} {degraded_counts[0]:<20} {degraded_counts[1]:<20}")
        print(f"  {'Not degraded':>15} {not_degraded_counts[0]:<20} {not_degraded_counts[1]:<20}")

        # Fisher's exact test (better for small samples)
        odds_ratio, p_value = sp_stats.fisher_exact(contingency.T)
        print(f"\n  Fisher's exact test:")
        print(f"    Odds ratio: {odds_ratio:.3f}")
        print(f"    p-value:    {p_value:.4f}")

        # Also run chi-square for reference
        if all(c > 5 for c in degraded_counts + not_degraded_counts):
            chi2, p_chi2, dof, expected = sp_stats.chi2_contingency(contingency.T)
            print(f"\n  Chi-square test:")
            print(f"    chi2: {chi2:.3f}, p-value: {p_chi2:.4f}")
    else:
        p_value = 1.0
        print("  Cannot run interaction test with != 2 task types")

    # === Verdict ===
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    ocr_type = [tt for tt in tt_list if "OCR" in tt][0]
    ar_type = [tt for tt in tt_list if "Action" in tt][0]
    ocr_deg = degradation[ocr_type]
    ar_deg = degradation[ar_type]

    print(f"  OCR degradation:              {ocr_deg:+.1f}%")
    print(f"  Action Recognition degradation: {ar_deg:+.1f}%")
    print(f"  p-value:                       {p_value:.4f}")

    if ocr_deg > ar_deg and p_value < 0.05:
        verdict = "HYPOTHESIS CONFIRMED"
    elif ocr_deg <= ar_deg:
        verdict = "HYPOTHESIS REJECTED"
    elif p_value >= 0.05:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  >>> {verdict} <<<")
    if verdict == "HYPOTHESIS CONFIRMED":
        print("  OCR tasks degrade significantly more than Action Recognition under H.264 compression.")
    elif verdict == "HYPOTHESIS REJECTED":
        print("  OCR tasks do NOT degrade more than Action Recognition (or degrade less).")
    else:
        print("  Trend observed but not statistically significant at p<0.05.")
        print("  Consider increasing sample size for a definitive answer.")

    # === PLOTS ===
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Prepare data for plotting
    plot_data = []
    for tt in task_types:
        for cond in CONDITIONS:
            sub = df[(df.task_type == tt) & (df.condition == cond)]
            acc = sub["correct"].mean()
            n = len(sub)
            se = np.sqrt(acc * (1 - acc) / n) if n > 0 else 0
            plot_data.append({
                "Task Type": tt,
                "Condition": cond,
                "Accuracy": acc,
                "SE": se,
                "CRF": CRF_MAP[cond],
                "n": n,
            })
    pdf = pd.DataFrame(plot_data)

    # Plot 1: Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    x = np.arange(len(CONDITIONS))
    width = 0.35
    for i, tt in enumerate(task_types):
        sub = pdf[pdf["Task Type"] == tt]
        bars = ax.bar(x + i * width, sub["Accuracy"], width,
                      yerr=sub["SE"], capsize=5, label=tt, alpha=0.85)
    ax.set_xlabel("Compression Condition", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Task Type and Compression Level", fontsize=14)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(CONDITIONS)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    bar_path = os.path.join(PLOT_DIR, "accuracy_bar_chart.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"\nSaved bar chart to {bar_path}")

    # Plot 2: Line chart (degradation curve)
    fig, ax = plt.subplots(figsize=(8, 6))
    for tt in task_types:
        sub = pdf[pdf["Task Type"] == tt].sort_values("CRF")
        ax.errorbar(sub["CRF"], sub["Accuracy"], yerr=sub["SE"],
                     marker="o", linewidth=2, capsize=5, label=tt)
    ax.set_xlabel("CRF Value", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Compression Level (CRF)", fontsize=14)
    ax.set_xticks([0, 18, 38])
    ax.set_xticklabels(["Original\n(CRF 0)", "CRF 18", "CRF 38"])
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    line_path = os.path.join(PLOT_DIR, "degradation_curve.png")
    plt.savefig(line_path, dpi=150)
    plt.close()
    print(f"Saved line chart to {line_path}")


if __name__ == "__main__":
    main()
