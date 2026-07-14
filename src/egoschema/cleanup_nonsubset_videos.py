"""Delete non-subset EgoSchema videos to free disk space for chunk 5 download."""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json, os, shutil

VIDEOS_DIR = VDG_DATA_ROOT + "/egoschema/videos"
SAMPLES    = VDG_DATA_ROOT + "/egoschema/egoschema_subset.json"

with open(SAMPLES) as f:
    samples = json.load(f)

needed_uids = {s["video_uid"] for s in samples}
all_mp4s    = {fn.replace(".mp4", ""): fn for fn in os.listdir(VIDEOS_DIR) if fn.endswith(".mp4")}

non_subset  = {uid: fn for uid, fn in all_mp4s.items() if uid not in needed_uids}
subset_present = {uid for uid in needed_uids if uid in all_mp4s}

print(f"Total mp4s in directory: {len(all_mp4s)}")
print(f"Subset videos present:   {len(subset_present)}")
print(f"Non-subset videos:       {len(non_subset)}")

free_before = shutil.disk_usage(VDG_DATA_ROOT).free / 1e9
print(f"Free space before: {free_before:.1f} GB")

print(f"\nDeleting {len(non_subset)} non-subset videos ...")
deleted = 0
for uid, fn in non_subset.items():
    path = os.path.join(VIDEOS_DIR, fn)
    os.remove(path)
    deleted += 1
    if deleted % 500 == 0:
        print(f"  Deleted {deleted}/{len(non_subset)} ...")

free_after = shutil.disk_usage(VDG_DATA_ROOT).free / 1e9
print(f"\nDeleted {deleted} files.")
print(f"Free space after: {free_after:.1f} GB  (freed {free_after - free_before:.1f} GB)")
print(f"Subset videos remaining: {sum(1 for fn in os.listdir(VIDEOS_DIR) if fn.endswith('.mp4'))}")
