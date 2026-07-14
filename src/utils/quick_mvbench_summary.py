
# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
from collections import defaultdict
r = json.load(open(VDG_RESULTS_ROOT + "/mvbench/qwen2vl_mvbench_results.json"))
errors = [x for x in r if x.get("error")]
correct = sum(1 for x in r if x.get("correct") and not x.get("error"))
valid = sum(1 for x in r if not x.get("error"))
print(f"Entries: {len(r)}, Errors: {len(errors)}, Acc: {correct}/{valid} = {correct/valid:.3f}")
by_task = defaultdict(lambda: defaultdict(list))
for x in r:
    if not x.get("error"):
        by_task[x["task_type"]][x["condition"]].append(x["correct"])
print(f"\n{'Task':<30} {'orig':>6} {'crf38':>6}")
print("-" * 45)
for tt in sorted(by_task):
    d = by_task[tt]
    orig  = f"{sum(d['original'])/len(d['original']):.3f}" if d.get("original") else "N/A"
    crf38 = f"{sum(d['crf38'])/len(d['crf38']):.3f}" if d.get("crf38") else "N/A"
    print(f"  {tt:<28} {orig:>6} {crf38:>6}")
