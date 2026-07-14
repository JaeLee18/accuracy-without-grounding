"""Verify egoschema_subset.json format that inference scripts will consume."""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json
with open(VDG_DATA_ROOT + "/egoschema/egoschema_subset.json") as f:
    samples = json.load(f)
print(f"Total: {len(samples)}")
s = samples[0]
print("Keys:", list(s.keys()))
print("question_id:", s["question_id"])
print("video_uid:", s["video_uid"])
print("question:", s["question"][:80])
for opt in s["options"]:
    print(" ", opt)
print("answer:", s["answer"])
print("task_type:", s["task_type"])

# Check answer distribution
from collections import Counter
ans_dist = Counter(s["answer"] for s in samples)
print("\nAnswer distribution:", dict(sorted(ans_dist.items())))
