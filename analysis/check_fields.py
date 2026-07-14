
# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
r = json.load(open(VDG_RESULTS_ROOT + '/full_study/qwen2vl_black_results.json'))
print(list(r[0].keys()))
print(r[0])
