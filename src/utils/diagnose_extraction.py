
# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json, zipfile, os

with open(VDG_DATA_ROOT + "/mvbench/mvbench_sample.json", encoding="latin-1") as f:
    samples = json.load(f)

videos_dir = VDG_DATA_ROOT + "/mvbench/videos"
extracted = set(os.path.splitext(f)[0] for f in os.listdir(videos_dir))

from collections import defaultdict
by_task = defaultdict(lambda: {"ok": 0, "fail": 0, "fail_ids": []})

for s in samples:
    vid = s["videoID"]
    tt = s["task_type"]
    if vid in extracted:
        by_task[tt]["ok"] += 1
    else:
        by_task[tt]["fail"] += 1
        by_task[tt]["fail_ids"].append(vid)

print(f"Extracted: {len(extracted)} unique videos\n")
print(f"{'Task':<35} {'OK':>4} {'FAIL':>5}")
print("-" * 47)
for tt in sorted(by_task):
    d = by_task[tt]
    print(f"  {tt:<33} {d['ok']:>4} {d['fail']:>5}")

# TVQA: check if needed clips/videos are in zip
print("\n=== Checking TVQA zip for needed items ===")
with zipfile.ZipFile(VDG_DATA_ROOT + "/mvbench/raw/tvqa.zip") as zf:
    members = set(zf.namelist())

# episodic_reasoning - frame dirs
ep_samples = [s for s in samples if s["task_type"] == "episodic_reasoning"]
ep_found = 0
for s in ep_samples:
    clip_name = s["video_raw"]
    full_path = f"tvqa/frames_fps3_hq/{clip_name}/"
    # Check if any member starts with this path
    found = any(m.startswith(f"tvqa/frames_fps3_hq/{clip_name}/") for m in members)
    if found:
        ep_found += 1

print(f"Episodic clips found in tvqa.zip: {ep_found}/{len(ep_samples)}")

# character_order - MP4 files
co_samples = [s for s in samples if s["task_type"] == "character_order"]
co_found = 0
for s in co_samples:
    basename = os.path.basename(s["video_raw"])
    found = any(os.path.basename(m) == basename for m in members)
    if found:
        co_found += 1

print(f"Character order videos found in tvqa.zip: {co_found}/{len(co_samples)}")

# MIT - check scene_transition
print("\n=== MIT zip ===")
mit_samples = [s for s in samples if s["task_type"] == "scene_transition"]
print(f"Scene transition needs {len(mit_samples)} videos")
print("First needed:", [s["video_raw"] for s in mit_samples[:3]])
with zipfile.ZipFile(VDG_DATA_ROOT + "/mvbench/raw/Moments_in_Time_Raw.zip") as zf:
    mit_members = set(os.path.basename(m) for m in zf.namelist())
mit_found = sum(1 for s in mit_samples if os.path.basename(s["video_raw"]) in mit_members)
print(f"Found in MIT zip: {mit_found}/{len(mit_samples)}")
