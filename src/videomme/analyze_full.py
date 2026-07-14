"""
Full study analysis — two-layer argument:
  Layer 1: Optical flow (motion quantity) does NOT predict compression sensitivity
  Layer 2: Temporal dependency (inter-frame reasoning need) DOES predict sensitivity

Outputs:
- Accuracy per task type per CRF
- Degradation curves (main figure — expect two-cluster separation)
- Task type ranking by compression sensitivity (slope)
- Spearman correlations: flow vs sensitivity (null), temporal dep vs sensitivity (significant)
- Key figure: flow vs degradation scatter with weak/no correlation
- One-way ANOVA across task types for degradation magnitude
- Cohen's d for high vs low sensitivity groups
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

RESULTS_DIR = VDG_RESULTS_ROOT + "/full_study"
PLOT_DIR = VDG_RESULTS_ROOT + "/full_study/plots"
FLOW_PATH = os.path.join(RESULTS_DIR, "optical_flow.json")
CONDITIONS = ["original", "crf18", "crf23", "crf28", "crf33", "crf38"]
CRF_VALUES = [0, 18, 23, 28, 33, 38]  # 0 = original

TASK_TYPES = [
    "OCR Problems",
    "Object Recognition",
    "Attribute Perception",
    "Action Recognition",
    "Action Reasoning",
    "Temporal Reasoning",
]

# "Single-frame answerability" — hand-labeled ordinal proxy for temporal dependency.
# Higher = requires more inter-frame reasoning, less answerable from a single frame.
# This is the core predictor variable for the paper's argument.
TEMPORAL_DEPENDENCY = {
    "OCR Problems": 1,           # text is per-frame, fully static
    "Object Recognition": 1,     # object identity is static
    "Attribute Perception": 2,   # often static properties, occasionally needs context
    "Action Recognition": 4,     # sometimes single-frame, often needs motion
    "Action Reasoning": 4,       # requires before/after reasoning
    "Temporal Reasoning": 5,     # explicitly requires sequence understanding
}

SINGLE_FRAME_ANSWERABLE = {
    "OCR Problems": "Yes",
    "Object Recognition": "Yes",
    "Attribute Perception": "Often",
    "Action Recognition": "Sometimes",
    "Action Reasoning": "Rarely",
    "Temporal Reasoning": "No",
}

STATIC_TYPES = {"OCR Problems", "Object Recognition", "Attribute Perception"}
TEMPORAL_TYPES = {"Action Recognition", "Action Reasoning", "Temporal Reasoning"}


def load_results(model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_results.json")
    with open(path) as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df = df[df["prediction"].notna()].copy()  # exclude errors
    return df


def compute_accuracy_table(df):
    """Returns dict[task_type][condition] = accuracy."""
    table = {}
    for tt in TASK_TYPES:
        table[tt] = {}
        for cond in CONDITIONS:
            sub = df[(df.task_type == tt) & (df.condition == cond)]
            table[tt][cond] = sub["correct"].mean() if len(sub) > 0 else 0
    return table


def compute_degradation_slope(acc_table):
    """Compute linear slope of accuracy vs CRF per task type."""
    slopes = {}
    for tt in TASK_TYPES:
        accs = [acc_table[tt][c] for c in CONDITIONS]
        slope, intercept, r, p, se = sp_stats.linregress(CRF_VALUES, accs)
        slopes[tt] = {"slope": slope, "intercept": intercept, "r": r, "p": p}
    return slopes


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_model(model_name, df):
    print(f"\n{'#'*80}")
    print(f"  MODEL: {model_name}")
    print(f"{'#'*80}")

    # Check sample sizes
    for tt in TASK_TYPES:
        for cond in CONDITIONS:
            n = len(df[(df.task_type == tt) & (df.condition == cond)])
            if n < 20:
                print(f"  ALERT: {tt}/{cond} has only {n} results (< 20)")

    acc_table = compute_accuracy_table(df)

    # === Accuracy table ===
    print(f"\n{'Task Type':<25} | " + " | ".join(f"{c:>8}" for c in CONDITIONS))
    print("-" * (26 + 11 * len(CONDITIONS)))
    for tt in TASK_TYPES:
        vals = " | ".join(f"{acc_table[tt][c]:>8.3f}" for c in CONDITIONS)
        print(f"{tt:<25} | {vals}")

    # === Degradation slopes ===
    slopes = compute_degradation_slope(acc_table)
    print(f"\n=== DEGRADATION RANKING (by slope, most negative = most vulnerable) ===")
    ranked = sorted(slopes.items(), key=lambda x: x[1]["slope"])
    for i, (tt, info) in enumerate(ranked):
        cluster = "TEMPORAL" if tt in TEMPORAL_TYPES else "STATIC"
        print(f"  {i+1}. {tt:<25} slope={info['slope']:>+.5f}  r={info['r']:>.3f}  [{cluster}]")

    # === Relative degradation at CRF 38 ===
    print(f"\n=== RELATIVE DEGRADATION (CRF 38 vs Original) ===")
    deg_values = {}
    for tt in TASK_TYPES:
        orig = acc_table[tt]["original"]
        crf38 = acc_table[tt]["crf38"]
        deg = (orig - crf38) / orig * 100 if orig > 0 else 0
        deg_values[tt] = deg
        cluster = "TEMPORAL" if tt in TEMPORAL_TYPES else "STATIC"
        print(f"  {tt:<25} {deg:>+6.1f}%  [{cluster}]")

    # === TWO-LAYER ARGUMENT ===
    slope_vals = [slopes[tt]["slope"] for tt in TASK_TYPES]

    # Layer 2: Temporal dependency (semantic) DOES predict sensitivity
    temp_dep_scores = [TEMPORAL_DEPENDENCY[tt] for tt in TASK_TYPES]
    rho_dep, p_dep = sp_stats.spearmanr(temp_dep_scores, slope_vals)
    print(f"\n=== LAYER 2: Temporal dependency vs degradation slope ===")
    print(f"  (Higher temporal dep = more inter-frame reasoning needed)")
    print(f"  Spearman rho = {rho_dep:.4f}, p = {p_dep:.4f}")
    for tt in TASK_TYPES:
        print(f"    {tt:<25} dep={TEMPORAL_DEPENDENCY[tt]}  "
              f"single-frame={SINGLE_FRAME_ANSWERABLE[tt]:<12} "
              f"slope={slopes[tt]['slope']:>+.5f}")

    # === ANOVA: degradation magnitude across task types ===
    print(f"\n=== ONE-WAY ANOVA: degradation across task types ===")
    # Per-question degradation: accuracy_drop = correct@original - correct@crf38
    groups = []
    group_labels = []
    for tt in TASK_TYPES:
        orig = df[(df.task_type == tt) & (df.condition == "original")].set_index("question_id")["correct"]
        crf38 = df[(df.task_type == tt) & (df.condition == "crf38")].set_index("question_id")["correct"]
        common = orig.index.intersection(crf38.index)
        drops = orig.loc[common].astype(int) - crf38.loc[common].astype(int)
        groups.append(drops.values)
        group_labels.append(tt)

    f_stat, p_anova = sp_stats.f_oneway(*groups)
    print(f"  F = {f_stat:.4f}, p = {p_anova:.4f}")

    # === Cohen's d: static vs temporal groups ===
    static_drops = np.concatenate([g for g, tt in zip(groups, TASK_TYPES) if tt in STATIC_TYPES])
    temporal_drops = np.concatenate([g for g, tt in zip(groups, TASK_TYPES) if tt in TEMPORAL_TYPES])
    d = cohens_d(temporal_drops, static_drops)
    print(f"\n=== COHEN'S D: temporal vs static degradation ===")
    print(f"  d = {d:.4f}")
    print(f"  temporal mean drop = {np.mean(temporal_drops):.4f}")
    print(f"  static mean drop   = {np.mean(static_drops):.4f}")

    return acc_table, slopes, deg_values


def plot_degradation_curves(acc_tables, model_names):
    """Main figure: degradation curves for all 6 task types, one plot per model."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Color scheme: warm colors for temporal, cool for static
    colors = {
        "OCR Problems": "#2196F3",
        "Object Recognition": "#4CAF50",
        "Attribute Perception": "#00BCD4",
        "Action Recognition": "#F44336",
        "Action Reasoning": "#FF9800",
        "Temporal Reasoning": "#9C27B0",
    }
    markers = {
        "OCR Problems": "s",
        "Object Recognition": "^",
        "Attribute Perception": "D",
        "Action Recognition": "o",
        "Action Reasoning": "v",
        "Temporal Reasoning": "P",
    }

    for model_name, acc_table in zip(model_names, acc_tables):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.set_style("whitegrid")

        for tt in TASK_TYPES:
            accs = [acc_table[tt][c] for c in CONDITIONS]
            cluster = "temporal" if tt in TEMPORAL_TYPES else "static"
            linestyle = "-" if cluster == "temporal" else "--"
            ax.plot(CRF_VALUES, accs, marker=markers[tt], color=colors[tt],
                    linewidth=2.5, markersize=8, linestyle=linestyle, label=tt)

        ax.set_xlabel("CRF Value", fontsize=13)
        ax.set_ylabel("Accuracy", fontsize=13)
        ax.set_title(f"Accuracy vs Compression Level — {model_name}", fontsize=14)
        ax.set_xticks(CRF_VALUES)
        ax.set_xticklabels(["Orig", "18", "23", "28", "33", "38"])
        ax.legend(loc="lower left", fontsize=10)
        ax.set_ylim(0, 1.05)

        # Add shaded regions for clusters
        ax.axhline(y=0, color='gray', linewidth=0.5)

        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"degradation_curves_{model_name}.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")

    # Combined figure (both models side by side)
    if len(acc_tables) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        for idx, (model_name, acc_table) in enumerate(zip(model_names, acc_tables)):
            ax = axes[idx]
            for tt in TASK_TYPES:
                accs = [acc_table[tt][c] for c in CONDITIONS]
                cluster = "temporal" if tt in TEMPORAL_TYPES else "static"
                linestyle = "-" if cluster == "temporal" else "--"
                ax.plot(CRF_VALUES, accs, marker=markers[tt], color=colors[tt],
                        linewidth=2.5, markersize=8, linestyle=linestyle, label=tt)
            ax.set_xlabel("CRF Value", fontsize=13)
            ax.set_title(model_name, fontsize=14)
            ax.set_xticks(CRF_VALUES)
            ax.set_xticklabels(["Orig", "18", "23", "28", "33", "38"])
            ax.set_ylim(0, 1.05)
            if idx == 0:
                ax.set_ylabel("Accuracy", fontsize=13)
            ax.legend(loc="lower left", fontsize=9)

        fig.suptitle("Differential Compression Sensitivity Across Task Types", fontsize=15, y=1.02)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, "degradation_curves_combined.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def plot_sensitivity_ranking(all_slopes, model_names):
    """Bar chart ranking task types by degradation slope."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(TASK_TYPES))
    width = 0.35

    for i, (model_name, slopes) in enumerate(zip(model_names, all_slopes)):
        ranked = sorted(TASK_TYPES, key=lambda tt: slopes[tt]["slope"])
        vals = [slopes[tt]["slope"] * 1000 for tt in ranked]  # scale for readability
        colors_bar = ["#F44336" if tt in TEMPORAL_TYPES else "#2196F3" for tt in ranked]
        offset = (i - 0.5 * (len(model_names) - 1)) * width
        bars = ax.bar(x + offset, vals, width, label=model_name, alpha=0.8,
                      color=colors_bar if len(model_names) == 1 else None)
        if i == 0:
            ax.set_xticks(x)
            ax.set_xticklabels(ranked, rotation=30, ha="right", fontsize=10)

    ax.set_ylabel("Degradation Slope (×10⁻³)", fontsize=12)
    ax.set_title("Task Type Ranking by Compression Sensitivity", fontsize=14)
    ax.axhline(y=0, color='black', linewidth=0.5)
    if len(model_names) > 1:
        ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "sensitivity_ranking.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def analyze_optical_flow(df, slopes):
    """
    Layer 1 analysis: show that optical flow (motion quantity) does NOT
    predict compression sensitivity. This is a deliberate negative result.
    """
    if not os.path.exists(FLOW_PATH):
        print("\nOptical flow data not found, skipping flow analysis.")
        return

    with open(FLOW_PATH) as f:
        flow_data = json.load(f)

    print(f"\n{'='*60}")
    print("  LAYER 1: Optical flow does NOT predict compression sensitivity")
    print(f"{'='*60}")

    # --- Per-video level: flow vs accuracy drop ---
    orig = df[df.condition == "original"].set_index(["videoID", "question_id"])["correct"]
    crf38 = df[df.condition == "crf38"].set_index(["videoID", "question_id"])["correct"]
    common = orig.index.intersection(crf38.index)

    video_drops = {}
    video_task = {}
    for (vid, qid) in common:
        drop = int(orig.loc[(vid, qid)]) - int(crf38.loc[(vid, qid)])
        if vid not in video_drops:
            video_drops[vid] = []
        video_drops[vid].append(drop)
        # Track task type for coloring
        row = df[(df.videoID == vid) & (df.question_id == qid) & (df.condition == "original")]
        if len(row) > 0:
            video_task[vid] = row.iloc[0]["task_type"]

    flows_all, drops_all, tasks_all = [], [], []
    for vid, drops in video_drops.items():
        if vid in flow_data and flow_data[vid]["avg_flow"] is not None:
            flows_all.append(flow_data[vid]["avg_flow"])
            drops_all.append(np.mean(drops))
            tasks_all.append(video_task.get(vid, "Unknown"))

    if len(flows_all) > 5:
        rho, p = sp_stats.spearmanr(flows_all, drops_all)
        r_pearson, p_pearson = sp_stats.pearsonr(flows_all, drops_all)
        print(f"\n  Per-video correlation (n={len(flows_all)}):")
        print(f"    Spearman: rho={rho:.4f}, p={p:.4f}")
        print(f"    Pearson:  r={r_pearson:.4f}, p={p_pearson:.4f}")
        if p > 0.05:
            print(f"    >> NOT SIGNIFICANT — motion quantity alone does not predict vulnerability")
        else:
            print(f"    >> Significant but likely weak")

        # Per-video scatter colored by cluster
        os.makedirs(PLOT_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        for flow_v, drop_v, task_v in zip(flows_all, drops_all, tasks_all):
            color = "#F44336" if task_v in TEMPORAL_TYPES else "#2196F3"
            marker = "o" if task_v in TEMPORAL_TYPES else "s"
            ax.scatter(flow_v, drop_v, c=color, marker=marker, alpha=0.4, s=25, edgecolors='none')
        # Legend
        ax.scatter([], [], c="#F44336", marker="o", label="Temporal tasks", s=50)
        ax.scatter([], [], c="#2196F3", marker="s", label="Static tasks", s=50)
        ax.set_xlabel("Average Optical Flow Magnitude", fontsize=12)
        ax.set_ylabel("Average Accuracy Drop (orig→CRF38)", fontsize=12)
        ax.set_title(f"Motion Quantity vs Accuracy Drop (ρ={rho:.3f}, p={p:.3f})", fontsize=13)
        z = np.polyfit(flows_all, drops_all, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(min(flows_all), max(flows_all), 100)
        ax.plot(x_line, p_line(x_line), "k--", alpha=0.4, linewidth=1)
        ax.legend(fontsize=11)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, "layer1_flow_vs_drop_pervideo.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"    Saved: {path}")

    # --- Per-task-type level: avg flow vs degradation slope ---
    # This is the key figure: one point per task type
    task_flows = {}
    for vid, info in flow_data.items():
        if info["avg_flow"] is not None:
            for tt in info["task_types"]:
                if tt in TASK_TYPES:
                    if tt not in task_flows:
                        task_flows[tt] = []
                    task_flows[tt].append(info["avg_flow"])

    if slopes and task_flows:
        flow_means = [np.mean(task_flows[tt]) for tt in TASK_TYPES if tt in task_flows]
        slope_vals = [slopes[tt]["slope"] for tt in TASK_TYPES if tt in task_flows]
        dep_scores = [TEMPORAL_DEPENDENCY[tt] for tt in TASK_TYPES if tt in task_flows]
        tt_labels = [tt for tt in TASK_TYPES if tt in task_flows]

        rho_agg, p_agg = sp_stats.spearmanr(flow_means, slope_vals)
        print(f"\n  Per-task-type correlation (n={len(tt_labels)}):")
        print(f"    Flow vs slope — Spearman rho={rho_agg:.4f}, p={p_agg:.4f}")

        rho_dep, p_dep = sp_stats.spearmanr(dep_scores, slope_vals)
        print(f"    Temporal dep vs slope — Spearman rho={rho_dep:.4f}, p={p_dep:.4f}")

        # THE KEY FIGURE: two-panel comparison
        colors_tt = {
            "OCR Problems": "#2196F3", "Object Recognition": "#4CAF50",
            "Attribute Perception": "#00BCD4", "Action Recognition": "#F44336",
            "Action Reasoning": "#FF9800", "Temporal Reasoning": "#9C27B0",
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel A: Flow vs degradation slope (expect weak/no correlation)
        for fm, sv, tt in zip(flow_means, slope_vals, tt_labels):
            ax1.scatter(fm, sv * 1000, c=colors_tt[tt], s=120, zorder=5, edgecolors='black', linewidth=0.5)
            ax1.annotate(tt.replace(" Problems", "").replace(" Recognition", " Rec.").replace(" Perception", " Per.").replace(" Reasoning", " Reas."),
                         (fm, sv * 1000), textcoords="offset points", xytext=(8, 4), fontsize=9)
        ax1.set_xlabel("Mean Optical Flow", fontsize=12)
        ax1.set_ylabel("Degradation Slope (×10⁻³)", fontsize=12)
        ax1.set_title(f"(a) Motion Quantity — ρ={rho_agg:.3f}, p={p_agg:.2f}", fontsize=12)

        # Panel B: Temporal dependency vs degradation slope (expect strong correlation)
        for ds, sv, tt in zip(dep_scores, slope_vals, tt_labels):
            ax2.scatter(ds, sv * 1000, c=colors_tt[tt], s=120, zorder=5, edgecolors='black', linewidth=0.5)
            ax2.annotate(tt.replace(" Problems", "").replace(" Recognition", " Rec.").replace(" Perception", " Per.").replace(" Reasoning", " Reas."),
                         (ds, sv * 1000), textcoords="offset points", xytext=(8, 4), fontsize=9)
        ax2.set_xlabel("Temporal Dependency Score", fontsize=12)
        ax2.set_ylabel("Degradation Slope (×10⁻³)", fontsize=12)
        ax2.set_title(f"(b) Temporal Dependency — ρ={rho_dep:.3f}, p={p_dep:.2f}", fontsize=12)
        ax2.set_xticks([1, 2, 3, 4, 5])

        fig.suptitle("What Predicts Compression Vulnerability?", fontsize=14, y=1.01)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, "two_layer_argument.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\n  >> KEY FIGURE saved: {path}")
        print(f"     This figure tells the whole story in one panel.")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    acc_tables = []
    all_slopes = []
    model_names = []

    for model_file, model_name in [
        ("qwen2vl_results.json", "Qwen2-VL-7B"),
        ("llava_results.json", "LLaVA-Video-7B"),
    ]:
        path = os.path.join(RESULTS_DIR, model_file)
        if not os.path.exists(path):
            print(f"Skipping {model_name} — results not found at {path}")
            continue

        df = load_results(model_file.replace("_results.json", ""))
        acc_table, slopes, deg_values = analyze_model(model_name, df)
        acc_tables.append(acc_table)
        all_slopes.append(slopes)
        model_names.append(model_name)

        # Optical flow analysis (once, using first model's data)
        if len(model_names) == 1:
            analyze_optical_flow(df, slopes)

    if acc_tables:
        plot_degradation_curves(acc_tables, model_names)
        plot_sensitivity_ranking(all_slopes, model_names)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
