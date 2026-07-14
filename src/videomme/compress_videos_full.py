"""
Full study: Compress videos at 5 CRF levels (18, 23, 28, 33, 38).
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

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
VIDEO_DIR = VDG_DATA_ROOT + "/videomme_full/videos"
BASE_DIR = VDG_DATA_ROOT + "/videomme_full"
CRF_LEVELS = [18, 23, 28, 33, 38]
FFMPEG = os.environ.get("FFMPEG_BIN", get_ffmpeg_exe())

def get_video_ids():
    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    return sorted(set(s["videoID"] for s in samples))

def compress_video(src_path, dst_path, crf):
    cmd = [
        FFMPEG, "-y", "-i", src_path,
        "-c:v", "libx264", "-crf", str(crf),
        "-preset", "medium",
        "-c:a", "copy",
        "-loglevel", "error",
        dst_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

def verify_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def main():
    video_ids = get_video_ids()
    print(f"Videos to compress: {len(video_ids)}")

    # Verify sources
    missing = [v for v in video_ids if not os.path.exists(os.path.join(VIDEO_DIR, f"{v}.mp4"))]
    if missing:
        print(f"ERROR: {len(missing)} source videos missing")
        for m in missing[:10]:
            print(f"  {m}")
        return

    stats = {}
    for crf in CRF_LEVELS:
        out_dir = os.path.join(BASE_DIR, f"crf{crf}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== CRF {crf} ===")
        done = skipped = failed = 0
        orig_mb = comp_mb = 0.0

        for vid in tqdm(video_ids, desc=f"CRF {crf}"):
            src = os.path.join(VIDEO_DIR, f"{vid}.mp4")
            dst = os.path.join(out_dir, f"{vid}.mp4")
            orig_mb += os.path.getsize(src) / (1024 * 1024)

            if verify_file(dst):
                comp_mb += os.path.getsize(dst) / (1024 * 1024)
                skipped += 1
                continue

            try:
                compress_video(src, dst, crf)
                if verify_file(dst):
                    comp_mb += os.path.getsize(dst) / (1024 * 1024)
                    done += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"\n  ERROR {vid}: {e}")

        ratio = comp_mb / orig_mb if orig_mb > 0 else 0
        stats[crf] = {"orig_mb": orig_mb, "comp_mb": comp_mb, "ratio": ratio, "done": done, "skipped": skipped, "failed": failed}

    print(f"\n{'='*80}")
    print(f"{'CRF':<8} {'Original (MB)':<16} {'Compressed (MB)':<18} {'Ratio':<10} {'New':<8} {'Skip':<8} {'Fail':<8}")
    print("-" * 76)
    for crf in CRF_LEVELS:
        s = stats[crf]
        print(f"{crf:<8} {s['orig_mb']:<16.1f} {s['comp_mb']:<18.1f} {s['ratio']:<10.2f} {s['done']:<8} {s['skipped']:<8} {s['failed']:<8}")

if __name__ == "__main__":
    main()
