"""
Compress MVBench videos to CRF38 (H.264 libx264, preset medium).
Resume: skip existing files >10KB.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import os, subprocess
import imageio_ffmpeg

VIDEOS_DIR = VDG_DATA_ROOT + "/mvbench/videos"
CRF38_DIR  = VDG_DATA_ROOT + "/mvbench/crf38"
CRF = 38

os.makedirs(CRF38_DIR, exist_ok=True)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

videos = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")]
print(f"Compressing {len(videos)} videos to CRF{CRF} ...")

done, errors = 0, []
for i, fname in enumerate(videos):
    inp = os.path.join(VIDEOS_DIR, fname)
    out = os.path.join(CRF38_DIR, fname)
    if os.path.exists(out) and os.path.getsize(out) > 10 * 1024:
        done += 1
        continue
    cmd = [ffmpeg, "-y", "-i", inp, "-c:v", "libx264",
           "-crf", str(CRF), "-preset", "medium", "-an", out]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) < 1024:
        errors.append(fname)
    else:
        done += 1
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(videos)} — done={done}, errors={len(errors)}")

print(f"\nDone: {done}/{len(videos)}, Errors: {len(errors)}")
if errors:
    print("Failed:", errors[:10])
