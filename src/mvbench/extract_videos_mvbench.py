"""
Extract needed videos from MVBench source zips to data/mvbench/videos/.
Handles episodic_reasoning (frame directories -> mp4 via ffmpeg at 3fps).
Resume: skip existing files >10KB (>50KB for episodic assembled clips).
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json, os, zipfile, shutil, subprocess, re
from collections import defaultdict
import imageio_ffmpeg

SAMPLE_PATH = VDG_DATA_ROOT + "/mvbench/mvbench_sample.json"
RAW_DIR     = VDG_DATA_ROOT + "/mvbench/raw"
OUT_DIR     = VDG_DATA_ROOT + "/mvbench/videos"
TMP_DIR     = VDG_DATA_ROOT + "/mvbench/_tmp_episodic"
EPISODIC_FPS = 3

os.makedirs(OUT_DIR, exist_ok=True)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

with open(SAMPLE_PATH, encoding="latin-1") as f:
    samples = json.load(f)

# Deduplicate: one entry per unique (videoID, source_zip)
seen = {}
for s in samples:
    key = s["videoID"]
    if key not in seen:
        seen[key] = s

unique_samples = list(seen.values())
print(f"Unique videos to extract: {len(unique_samples)}")

by_zip = defaultdict(list)
for s in unique_samples:
    by_zip[s["source_zip"]].append(s)

def is_done(path, min_kb=10):
    return os.path.exists(path) and os.path.getsize(path) > min_kb * 1024

def assemble_episodic(frame_dir_name, zip_path, video_id, out_path):
    """Extract frame dir from zip, concatenate frames to mp4 at EPISODIC_FPS."""
    tmp = os.path.join(TMP_DIR, video_id)
    os.makedirs(tmp, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            all_names = zf.namelist()
            # tvqa.zip has paths like tvqa/frames_fps3_hq/{clip_name}/NNNNN.jpg
            tvqa_prefix = f"tvqa/frames_fps3_hq/{frame_dir_name}/"
            members = [m for m in all_names
                       if (m.startswith(tvqa_prefix) or m.startswith(frame_dir_name + "/"))
                       and m.lower().endswith(".jpg")]
            if not members:
                raise RuntimeError(f"No jpg frames found for {frame_dir_name} in {zip_path}")
            for m in members:
                zf.extract(m, tmp)

        # Find the actual extracted frame dir (may have tvqa/frames_fps3_hq/ prefix)
        frame_dir = os.path.join(tmp, frame_dir_name)
        if not os.path.isdir(frame_dir):
            frame_dir = os.path.join(tmp, "tvqa", "frames_fps3_hq", frame_dir_name)
        cmd = [ffmpeg, "-y", "-framerate", str(EPISODIC_FPS),
               "-i", os.path.join(frame_dir, "%05d.jpg"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[-200:]}")
        if not is_done(out_path, min_kb=50):
            if os.path.exists(out_path):
                os.remove(out_path)
            raise RuntimeError(f"Assembled clip too small: {out_path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

done, errors = 0, []

for zip_name, zip_samples in sorted(by_zip.items()):
    zip_path = os.path.join(RAW_DIR, zip_name)
    if not os.path.exists(zip_path):
        print(f"\nWARNING: {zip_path} not found — skipping {len(zip_samples)} videos")
        for s in zip_samples:
            errors.append((s["videoID"], f"zip not found: {zip_name}"))
        continue

    is_episodic_zip = (zip_name == "tvqa.zip")
    print(f"\n{zip_name} ({len(zip_samples)} videos) ...")

    # Build zip member index once
    with zipfile.ZipFile(zip_path) as zf:
        all_members = set(zf.namelist())

    for s in zip_samples:
        video_id  = s["videoID"]
        video_raw = s["video_raw"]
        out_path  = os.path.join(OUT_DIR, f"{video_id}.mp4")

        is_ep = (s["task_type"] == "episodic_reasoning")
        min_kb = 50 if is_ep else 10

        if is_done(out_path, min_kb=min_kb):
            done += 1
            continue

        if is_ep:
            try:
                assemble_episodic(video_raw, zip_path, video_id, out_path)
                done += 1
                print(f"  [assembled] {video_id}")
            except Exception as e:
                errors.append((video_id, str(e)))
                print(f"  [ERROR] {video_id}: {e}")
        else:
            # Find mp4 in zip (video_raw = filename, possibly with subdirectory)
            basename = os.path.basename(video_raw)
            candidates = [m for m in all_members if os.path.basename(m) == basename]
            if not candidates:
                errors.append((video_id, f"'{basename}' not found in {zip_name}"))
                continue
            member = candidates[0]
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    with zf.open(member) as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                if not is_done(out_path, min_kb=10):
                    errors.append((video_id, "extracted file too small"))
                    os.remove(out_path)
                else:
                    done += 1
            except Exception as e:
                errors.append((video_id, str(e)))

print(f"\n=== Extraction complete: {done}/{len(unique_samples)} ===")
print(f"Errors: {len(errors)}")
if errors:
    print("First 20 errors:")
    for vid, err in errors[:20]:
        print(f"  {vid}: {err}")
