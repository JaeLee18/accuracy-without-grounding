"""
Step 3: Compress pilot videos at CRF 18 and CRF 38 using H.264 (libx264).
Reads pilot_sample.json for needed video IDs. Resumable (skips existing files).
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json
import os
import subprocess
from tqdm import tqdm
from imageio_ffmpeg import get_ffmpeg_exe

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme/pilot_sample.json"
VIDEO_DIR = VDG_DATA_ROOT + "/videomme/videos"
CRF_LEVELS = [18, 38]
OUTPUT_DIRS = {
    18: VDG_DATA_ROOT + "/videomme/crf18",
    38: VDG_DATA_ROOT + "/videomme/crf38",
}
FFMPEG = os.environ.get("FFMPEG_BIN", get_ffmpeg_exe())

def get_video_ids():
    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    return sorted(set(s["videoID"] for s in samples))

def compress_video(src_path, dst_path, crf):
    """Compress a video with H.264 at the given CRF level."""
    cmd = [
        FFMPEG, "-y", "-i", src_path,
        "-c:v", "libx264", "-crf", str(crf),
        "-preset", "medium",
        "-c:a", "copy",  # preserve audio
        "-loglevel", "error",
        dst_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

def verify_file(path):
    """Check file exists and is non-zero size."""
    return os.path.exists(path) and os.path.getsize(path) > 0

def main():
    video_ids = get_video_ids()
    print(f"Videos to compress: {len(video_ids)}")

    # Verify all source videos exist
    missing = []
    for vid in video_ids:
        src = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        if not os.path.exists(src):
            missing.append(vid)
    if missing:
        print(f"ERROR: {len(missing)} source videos missing:")
        for m in missing:
            print(f"  {m}")
        return

    # Create output dirs
    for crf in CRF_LEVELS:
        os.makedirs(OUTPUT_DIRS[crf], exist_ok=True)

    # Compress
    stats = {crf: {"original_mb": 0, "compressed_mb": 0, "skipped": 0, "done": 0, "failed": 0}
             for crf in CRF_LEVELS}

    for crf in CRF_LEVELS:
        print(f"\n=== Compressing at CRF {crf} ===")
        for vid in tqdm(video_ids, desc=f"CRF {crf}"):
            src = os.path.join(VIDEO_DIR, f"{vid}.mp4")
            dst = os.path.join(OUTPUT_DIRS[crf], f"{vid}.mp4")
            src_size = os.path.getsize(src) / (1024 * 1024)
            stats[crf]["original_mb"] += src_size

            # Skip if already done
            if verify_file(dst):
                stats[crf]["compressed_mb"] += os.path.getsize(dst) / (1024 * 1024)
                stats[crf]["skipped"] += 1
                continue

            try:
                compress_video(src, dst, crf)
                if verify_file(dst):
                    stats[crf]["compressed_mb"] += os.path.getsize(dst) / (1024 * 1024)
                    stats[crf]["done"] += 1
                else:
                    stats[crf]["failed"] += 1
                    print(f"\n  WARNING: {dst} is invalid after compression")
            except Exception as e:
                stats[crf]["failed"] += 1
                print(f"\n  ERROR compressing {vid} at CRF {crf}: {e}")

    # Report
    print("\n=== FILE SIZE STATISTICS ===")
    print(f"{'Condition':<12} {'Original (MB)':<16} {'Compressed (MB)':<18} {'Ratio':<10} {'Skipped':<10} {'Failed':<10}")
    print("-" * 76)
    for crf in CRF_LEVELS:
        s = stats[crf]
        ratio = s["compressed_mb"] / s["original_mb"] if s["original_mb"] > 0 else 0
        print(f"CRF {crf:<7} {s['original_mb']:<16.1f} {s['compressed_mb']:<18.1f} {ratio:<10.2f} {s['skipped']:<10} {s['failed']:<10}")

if __name__ == "__main__":
    main()
