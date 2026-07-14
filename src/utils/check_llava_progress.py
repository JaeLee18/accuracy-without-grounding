
# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os, time
path = VDG_RESULTS_ROOT + "/full_study/llava_results.json"
r = json.load(open(path))
mtime = os.path.getmtime(path)
n = len(r)
errors = sum(1 for x in r if x.get("error"))
conditions = {}
for x in r:
    conditions[x["condition"]] = conditions.get(x["condition"], 0) + 1
print(f"Total entries: {n}/3600")
print(f"Errors: {errors}")
print(f"Per condition: {conditions}")
print(f"File last modified: {time.strftime('%H:%M:%S', time.localtime(mtime))}")
print(f"Progress: {n/3600*100:.1f}%")
print(f"Current time: {time.strftime('%H:%M:%S')}")
