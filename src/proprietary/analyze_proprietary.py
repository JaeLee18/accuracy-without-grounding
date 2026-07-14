"""
Analyze proprietary model results: compute VDG, compare with open-weight models.

Usage:
    python analyze_proprietary.py --model gemini-flash
    python analyze_proprietary.py --model gpt
    python analyze_proprietary.py --model all
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

RESULTS_DIR = VDG_RESULTS_ROOT + "/proprietary"
MODELS = ["gemini-flash", "gpt"]

DATASETS = {
    "videomme": {
        "qid_field": "question_id",
        "task_type_field": "task_type",
        "n_choices": 4,
    },
    "mvbench": {
        "qid_field": "question_id",
        "task_type_field": "task_type",
        "n_choices": 4,
    },
    "egoschema": {
        "qid_field": "question_id",
        "task_type_field": "task_type",
        "n_choices": 5,
    },
}

# Open-weight baselines for comparison
BASELINES = {
    "videomme": {
        "Qwen2-VL-7B":    {"orig": 0.700, "black": 0.393, "vdg": 0.240},
        "LLaVA-Video-7B":  {"orig": 0.638, "black": 0.351, "vdg": 0.287},
        "InternVL2-8B":    {"orig": 0.583, "black": 0.312, "vdg": 0.272},
    },
    "mvbench": {
        "Qwen2-VL-7B":    {"orig": 0.670, "black": 0.460, "vdg": 0.208},
        "LLaVA-Video-7B":  {"orig": 0.662, "black": 0.476, "vdg": 0.186},
        "InternVL2-8B":    {"orig": 0.742, "black": 0.476, "vdg": 0.266},
    },
    "egoschema": {
        "Qwen2-VL-7B":    {"orig": 0.568, "black": 0.304, "vdg": 0.264},
        "LLaVA-Video-7B":  {"orig": 0.514, "black": 0.206, "vdg": 0.308},
        "InternVL2-8B":    {"orig": 0.600, "black": 0.318, "vdg": 0.282},
    },
}


def load_results(model: str, dataset: str, condition: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, f"{model}_{dataset}_{condition}_results.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def bootstrap_ci(values, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi


def analyze_dataset(model: str, dataset: str):
    cfg = DATASETS[dataset]
    orig_results = load_results(model, dataset, "original")
    black_results = load_results(model, dataset, "black")

    if not orig_results or not black_results:
        print(f"  {dataset}: Missing results (orig={len(orig_results)}, black={len(black_results)})")
        return None

    # Index by question_id
    qid = cfg["qid_field"]
    orig_idx = {r[qid]: r for r in orig_results if not r.get("error")}
    black_idx = {r[qid]: r for r in black_results if not r.get("error")}
    common = sorted(set(orig_idx) & set(black_idx))

    if not common:
        print(f"  {dataset}: No common questions between conditions")
        return None

    # Compute VDG
    by_task = defaultdict(lambda: {"vdg": [], "orig": [], "black": []})
    all_vdg = []

    for q in common:
        o = orig_idx[q]
        b = black_idx[q]
        vdg = int(o.get("correct", False)) - int(b.get("correct", False))
        tt = o.get(cfg["task_type_field"], "unknown")
        by_task[tt]["vdg"].append(vdg)
        by_task[tt]["orig"].append(int(o.get("correct", False)))
        by_task[tt]["black"].append(int(b.get("correct", False)))
        all_vdg.append(vdg)

    # Overall stats
    n = len(common)
    orig_acc = np.mean([int(orig_idx[q].get("correct", False)) for q in common])
    black_acc = np.mean([int(black_idx[q].get("correct", False)) for q in common])
    vdg_mean = np.mean(all_vdg)
    ci_lo, ci_hi = bootstrap_ci(all_vdg)

    chance = 1.0 / cfg["n_choices"]

    print(f"\n  === {dataset.upper()} ({model}) ===")
    print(f"  N = {n} matched questions (chance = {chance:.0%})")
    print(f"  Original accuracy:    {orig_acc:.3f}")
    print(f"  Black-screen accuracy:{black_acc:.3f}")
    print(f"  VDG:                  {vdg_mean:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")
    print()

    # Per task type
    print(f"  {'Task Type':<30s} {'Orig':>6s} {'Black':>6s} {'VDG':>6s} {'n':>5s}")
    print(f"  {'-'*55}")
    for tt in sorted(by_task.keys()):
        d = by_task[tt]
        n_tt = len(d["vdg"])
        o = np.mean(d["orig"])
        b = np.mean(d["black"])
        v = np.mean(d["vdg"])
        print(f"  {tt:<30s} {o:6.3f} {b:6.3f} {v:+6.3f} {n_tt:5d}")

    # Compare with baselines
    if dataset in BASELINES:
        print(f"\n  --- Comparison with open-weight models ---")
        print(f"  {'Model':<25s} {'Orig':>6s} {'Black':>6s} {'VDG':>6s}")
        print(f"  {'-'*45}")
        print(f"  {model + ' (proprietary)':<25s} {orig_acc:6.3f} {black_acc:6.3f} {vdg_mean:+6.3f}")
        for bname, bvals in BASELINES[dataset].items():
            print(f"  {bname:<25s} {bvals['orig']:6.3f} {bvals['black']:6.3f} {bvals['vdg']:+6.3f}")

    # Destructive ratio
    destructive = sum(1 for v in all_vdg if v == 1)
    constructive = sum(1 for v in all_vdg if v == -1)
    ratio = destructive / constructive if constructive > 0 else float("inf")
    print(f"\n  Destructive ratio: {destructive}:{constructive} = {ratio:.1f}:1")

    return {
        "dataset": dataset,
        "model": model,
        "n": n,
        "orig_acc": orig_acc,
        "black_acc": black_acc,
        "vdg": vdg_mean,
        "ci": [ci_lo, ci_hi],
        "destructive": destructive,
        "constructive": constructive,
        "by_task": {tt: {k: float(np.mean(v)) for k, v in d.items()} for tt, d in by_task.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=MODELS + ["all"])
    args = parser.parse_args()

    models = MODELS if args.model == "all" else [args.model]
    all_results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model}")
        print(f"{'='*60}")
        for dataset in DATASETS:
            result = analyze_dataset(model, dataset)
            if result:
                all_results.append(result)

    # Save summary
    if all_results:
        summary_path = os.path.join(RESULTS_DIR, "proprietary_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
