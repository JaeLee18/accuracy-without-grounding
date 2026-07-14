
# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os, time
path = VDG_RESULTS_ROOT + "/mvbench/qwen2vl_mvbench_results.json"
r = json.load(open(path))
mtime = os.path.getmtime(path)
conds = {}
for x in r:
    conds[x["condition"]] = conds.get(x["condition"], 0) + 1
n = len(r)
total = 1386
print(f"Qwen MVBench: {n}/{total} ({n/total*100:.1f}%)")
print(f"Conditions: {conds}")
print(f"File last modified: {time.strftime('%H:%M:%S', time.localtime(mtime))}")
print(f"Current time: {time.strftime('%H:%M:%S')}")
