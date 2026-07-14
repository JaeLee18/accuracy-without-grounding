
# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json, zipfile, os

with open(VDG_DATA_ROOT + "/mvbench/mvbench_sample.json", encoding="latin-1") as f:
    samples = json.load(f)

# Check scene_qa.zip
print("=== scene_qa.zip ===")
with zipfile.ZipFile(VDG_DATA_ROOT + "/mvbench/raw/scene_qa.zip") as zf:
    members = zf.namelist()
mp4s = [m for m in members if m.lower().endswith('.mp4')]
print(f"MP4 files: {len(mp4s)}")
print("First 5:", mp4s[:5])

# Check ssv2_video.zip
print("\n=== ssv2_video.zip ===")
with zipfile.ZipFile(VDG_DATA_ROOT + "/mvbench/raw/ssv2_video.zip") as zf:
    members2 = zf.namelist()
mp4s2 = [m for m in members2 if m.lower().endswith('.mp4') or m.lower().endswith('.webm')]
print(f"Video files: {len(mp4s2)}")
print("First 5:", mp4s2[:5])

# Check which of our failing tasks might find videos in these
basenames_scene_qa = set(os.path.basename(m) for m in members if m.lower().endswith('.mp4'))
basenames_ssv2 = set(os.path.basename(m) for m in members2 if m.lower().endswith('.mp4') or m.lower().endswith('.webm'))

# Check failing tasks
failing_tasks = ["action_antonym", "action_count", "action_sequence", "character_order",
                 "fine_grained_action", "moving_attribute", "moving_count", "moving_direction",
                 "object_shuffle", "scene_transition"]
print("\n=== Coverage check for failing tasks ===")
for tt in failing_tasks:
    task_samples = [s for s in samples if s["task_type"] == tt]
    needed = set(os.path.basename(s["video_raw"]) for s in task_samples)
    in_scene_qa = needed & basenames_scene_qa
    in_ssv2 = needed & basenames_ssv2
    print(f"  {tt}: in_scene_qa={len(in_scene_qa)} in_ssv2={len(in_ssv2)}")
