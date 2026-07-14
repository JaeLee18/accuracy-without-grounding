
# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
r = json.load(open(VDG_RESULTS_ROOT + "/mvbench/llava_mvbench_results.json"))
conds = {}
for x in r:
    conds[x["condition"]] = conds.get(x["condition"], 0) + 1
errors = sum(1 for x in r if x.get("error"))
print(f"LLaVA MVBench: {len(r)} entries, errors={errors}")
print(f"Conditions: {conds}")
