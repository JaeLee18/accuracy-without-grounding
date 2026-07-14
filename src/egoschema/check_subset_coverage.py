"""Check how many EgoSchema subset videos are already downloaded."""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json, os

VIDEOS_DIR   = VDG_DATA_ROOT + "/egoschema/videos"
SAMPLES_PATH = VDG_DATA_ROOT + "/egoschema/egoschema_subset.json"

with open(SAMPLES_PATH) as f:
    samples = json.load(f)

needed  = {s["video_uid"] for s in samples}
present = {fn.replace(".mp4", "") for fn in os.listdir(VIDEOS_DIR) if fn.endswith(".mp4")}
missing = needed - present
found   = needed & present

print(f"Subset needs:  {len(needed)}")
print(f"Present:       {len(found)}")
print(f"Missing:       {len(missing)}")
print(f"Coverage:      {len(found)/len(needed)*100:.1f}%")
print(f"\nTotal mp4s in videos dir: {len(present)}")
