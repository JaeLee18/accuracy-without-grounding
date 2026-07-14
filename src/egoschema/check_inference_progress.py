"""Quick progress check for all EgoSchema inference results."""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os
from collections import defaultdict

RESULTS = {
    "qwen":  VDG_RESULTS_ROOT + "/egoschema/qwen2vl_egoschema_results.json",
    "llava": VDG_RESULTS_ROOT + "/egoschema/llava_egoschema_results.json",
    "iv2":   VDG_RESULTS_ROOT + "/egoschema/internvl2_egoschema_results.json",
}

for name, path in RESULTS.items():
    if not os.path.exists(path):
        print(f"{name:<8}: not started")
        continue
    with open(path) as f:
        data = json.load(f)
    cond_stats = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
    for r in data:
        c = r.get("condition", "?")
        cond_stats[c]["total"] += 1
        if r.get("error"):
            cond_stats[c]["errors"] += 1
        elif r.get("correct"):
            cond_stats[c]["correct"] += 1

    for cond in ["original", "black"]:
        s = cond_stats[cond]
        n = s["total"]
        if n == 0:
            acc_str = "N/A"
        else:
            valid = n - s["errors"]
            acc_str = f"{s['correct']}/{valid} ({s['correct']/max(1,valid)*100:.1f}%)"
        print(f"{name:<8} {cond:<10}: {n:>4} total, {s['errors']:>3} errors, acc={acc_str}")
