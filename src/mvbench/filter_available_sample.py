"""
Filter mvbench_sample.json to only include questions for successfully extracted videos.
Saves to data/mvbench/mvbench_available_sample.json
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json, os

SAMPLE_PATH = VDG_DATA_ROOT + "/mvbench/mvbench_sample.json"
OUT_PATH = VDG_DATA_ROOT + "/mvbench/mvbench_available_sample.json"
VIDEOS_DIR = VDG_DATA_ROOT + "/mvbench/videos"

with open(SAMPLE_PATH, encoding="latin-1") as f:
    samples = json.load(f)

extracted = set(os.path.splitext(f)[0] for f in os.listdir(VIDEOS_DIR))
available = [s for s in samples if s["videoID"] in extracted]

from collections import Counter
by_task = Counter(s["task_type"] for s in available)
print(f"Total available: {len(available)}/{len(samples)} questions")
print("\nPer task:")
for tt in sorted(by_task):
    print(f"  {tt}: {by_task[tt]}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(available, f, indent=2, ensure_ascii=True)
print(f"\nSaved to {OUT_PATH}")
