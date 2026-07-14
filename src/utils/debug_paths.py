
# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json, zipfile

# Check scene_transition raw JSON
print("=== scene_transition video names ===")
with open(VDG_DATA_ROOT + "/mvbench/raw/scene_transition.json", encoding="latin-1") as f:
    data = json.load(f)
print("First 5:", [d["video"] for d in data[:5]])

# Check episodic_reasoning raw JSON
print("\n=== episodic_reasoning video names ===")
with open(VDG_DATA_ROOT + "/mvbench/raw/episodic_reasoning.json", encoding="latin-1") as f:
    data = json.load(f)
print("First 5:", [d["video"] for d in data[:5]])

# Check character_order raw JSON
print("\n=== character_order video names ===")
with open(VDG_DATA_ROOT + "/mvbench/raw/character_order.json", encoding="latin-1") as f:
    data = json.load(f)
print("First 5:", [d["video"] for d in data[:5]])

# Check what's actually in Moments_in_Time_Raw.zip
print("\n=== MIT zip contents - first 5 mp4s ===")
with zipfile.ZipFile(VDG_DATA_ROOT + "/mvbench/raw/Moments_in_Time_Raw.zip") as zf:
    mp4s = [m for m in zf.namelist() if m.endswith('.mp4')]
print("Count:", len(mp4s), "MP4s")
print("Sample:", mp4s[:5])

# Check what's in tvqa.zip - look for frames_fps3_hq prefix
print("\n=== tvqa.zip - sample paths ===")
with zipfile.ZipFile(VDG_DATA_ROOT + "/mvbench/raw/tvqa.zip") as zf:
    members = zf.namelist()
print("Total entries:", len(members))
# Get unique frame dirs
frame_dirs = set()
for m in members:
    parts = m.split('/')
    if len(parts) >= 3 and parts[1] == 'frames_fps3_hq':
        frame_dirs.add(parts[2])
print(f"Unique frame dirs: {len(frame_dirs)}")
# Check if castle clips are there
castle = [d for d in frame_dirs if 'castle' in d.lower()]
friends = [d for d in frame_dirs if 'friends' in d.lower()]
print(f"Castle clips: {len(castle)}, first 3:", sorted(castle)[:3])
print(f"Friends clips: {len(friends)}, first 3:", sorted(friends)[:3])
print("Sample frame dirs:", sorted(list(frame_dirs))[:10])
