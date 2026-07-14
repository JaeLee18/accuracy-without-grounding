
# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
from collections import defaultdict

with open(VDG_RESULTS_ROOT + "/full_study/internvl2_results.json") as f:
    orig_data = json.load(f)
with open(VDG_RESULTS_ROOT + "/full_study/internvl2_black_results.json") as f:
    black_data = json.load(f)

stats = defaultdict(lambda: defaultdict(lambda: {"c": 0, "n": 0}))
for r in orig_data:
    if not r.get("error") and r.get("condition") == "original":
        stats[r["task_type"]]["orig"]["n"] += 1
        stats[r["task_type"]]["orig"]["c"] += int(r.get("correct", False))
for r in black_data:
    if not r.get("error"):
        stats[r["task_type"]]["black"]["n"] += 1
        stats[r["task_type"]]["black"]["c"] += int(r.get("correct", False))

b26 = {"Action Recognition": 0.44, "Action Reasoning": 0.35, "Temporal Reasoning": 0.30,
       "Attribute Perception": 0.29, "Object Recognition": 0.27, "OCR Problems": 0.27}

print("InternVL2-8B vs 26B Black-Screen Comparison (Video-MME):")
print(f"{'Task Type':<25} {'8B-Orig':>8} {'8B-Black':>9} {'8B-VGG':>7} {'26B-Black':>10} {'Delta':>7}")
print("-" * 70)
tc8o, tn8o, tc8b, tn8b = 0, 0, 0, 0
for tt in sorted(stats.keys()):
    o8 = stats[tt]["orig"]
    b8 = stats[tt]["black"]
    o8a = o8["c"] / o8["n"] if o8["n"] else 0
    b8a = b8["c"] / b8["n"] if b8["n"] else 0
    b26a = b26.get(tt, 0)
    delta = b26a - b8a
    print(f"{tt:<25} {o8a:>8.3f} {b8a:>9.3f} {o8a-b8a:>7.3f} {b26a:>10.3f} {delta:>+7.3f}")
    tc8o += o8["c"]; tn8o += o8["n"]
    tc8b += b8["c"]; tn8b += b8["n"]
if tn8o and tn8b:
    o8a = tc8o/tn8o; b8a = tc8b/tn8b
    print(f"{'OVERALL':<25} {o8a:>8.3f} {b8a:>9.3f} {o8a-b8a:>7.3f} {0.32:>10.3f} {0.32-b8a:>+7.3f}")
