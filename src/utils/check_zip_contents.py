
# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import zipfile, os

# Check Moments_in_Time_Raw.zip
mit_zip = VDG_DATA_ROOT + "/mvbench/raw/Moments_in_Time_Raw.zip"
print(f"=== {mit_zip} ===")
with zipfile.ZipFile(mit_zip) as zf:
    members = zf.namelist()
print(f"Total entries: {len(members)}")
mp4s = [m for m in members if m.lower().endswith('.mp4')]
print(f"MP4 files: {len(mp4s)}")
if mp4s:
    print("First 10:", mp4s[:10])

print()

# Check tvqa.zip - what's in it
tvqa_zip = VDG_DATA_ROOT + "/mvbench/raw/tvqa.zip"
print(f"=== {tvqa_zip} ===")
with zipfile.ZipFile(tvqa_zip) as zf:
    members = zf.namelist()
print(f"Total entries: {len(members)}")
# Count unique top-level directories
top_dirs = set()
for m in members:
    parts = m.split('/')
    if len(parts) > 1:
        top_dirs.add(parts[0])
print(f"Top-level dirs: {len(top_dirs)}")
print("Sample dirs:", sorted(top_dirs)[:10])
# Check for jpg files
jpgs = [m for m in members if m.lower().endswith('.jpg')]
print(f"JPG files: {len(jpgs)}")
if jpgs:
    print("Sample jpgs:", jpgs[:5])
