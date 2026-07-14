"""
Sanity check: Inspect action recognition questions that degraded at CRF 38.
Find questions correct at original but wrong at CRF 38, then probe video properties.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import subprocess
from imageio_ffmpeg import get_ffmpeg_exe

RESULTS_PATH = VDG_RESULTS_ROOT + "/pilot_results.json"
VIDEO_DIR = VDG_DATA_ROOT + "/videomme/videos"
FFPROBE = os.environ.get("FFMPEG_BIN", get_ffmpeg_exe())

def get_video_info(video_path):
    """Get duration, fps, resolution, bitrate via ffmpeg."""
    cmd = [FFPROBE, "-i", video_path, "-hide_banner"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    info = r.stderr  # ffmpeg prints info to stderr
    duration = fps = resolution = bitrate = "?"
    for line in info.split("\n"):
        if "Duration:" in line:
            duration = line.split("Duration:")[1].split(",")[0].strip()
            br = line.split("bitrate:")
            if len(br) > 1:
                bitrate = br[1].strip()
        if "Video:" in line and "fps" in line:
            parts = line.split(",")
            for p in parts:
                if "fps" in p:
                    fps = p.strip()
                if "x" in p and ("yuv" not in p.lower()) and ("0x" not in p.lower()):
                    res_candidate = p.strip()
                    if any(c.isdigit() for c in res_candidate):
                        resolution = res_candidate
            # better resolution extraction
            for p in parts:
                p = p.strip()
                if "x" in p:
                    tokens = p.split()
                    for t in tokens:
                        if "x" in t and t.replace("x", "").replace(" ", "").split("x")[0].isdigit():
                            resolution = t
    return duration, fps, resolution, bitrate

def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # Find AR questions: correct at original, wrong at crf38
    ar_results = [r for r in results if r["task_type"] == "Action Recognition"]

    by_qid = {}
    for r in ar_results:
        qid = r["question_id"]
        if qid not in by_qid:
            by_qid[qid] = {}
        by_qid[qid][r["condition"]] = r

    degraded = []
    for qid, conds in by_qid.items():
        if conds.get("original", {}).get("correct") and not conds.get("crf38", {}).get("correct"):
            degraded.append(conds)

    print(f"Action Recognition questions that degraded (correct@orig -> wrong@crf38): {len(degraded)}")
    print(f"Total AR questions: {len(by_qid)}")
    print()

    for i, conds in enumerate(degraded):
        orig = conds["original"]
        crf38 = conds["crf38"]
        vid = orig["videoID"]
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")

        duration, fps, resolution, bitrate = get_video_info(video_path)

        print(f"{'='*80}")
        print(f"DEGRADED EXAMPLE {i+1}/{len(degraded)}")
        print(f"{'='*80}")
        print(f"  Question ID: {orig['question_id']}")
        print(f"  Video ID:    {vid}")
        print(f"  Duration:    {duration}")
        print(f"  FPS:         {fps}")
        print(f"  Resolution:  {resolution}")
        print(f"  Bitrate:     {bitrate}")
        print(f"  Question:    {orig['question']}")
        print(f"  Ground truth: {orig['ground_truth']}")
        print(f"  Pred@orig:   {orig['prediction']} (CORRECT)")
        print(f"  Pred@crf18:  {conds.get('crf18', {}).get('prediction', '?')}"
              f" ({'OK' if conds.get('crf18', {}).get('correct') else 'WRONG'})")
        print(f"  Pred@crf38:  {crf38['prediction']} (WRONG)")
        print()

    # Also show AR questions that stayed correct across all conditions
    stable = []
    for qid, conds in by_qid.items():
        if all(conds.get(c, {}).get("correct") for c in ["original", "crf18", "crf38"]):
            stable.append(conds)

    print(f"\nFor comparison: {len(stable)} AR questions stayed correct across ALL conditions")
    print(f"{len(degraded)} degraded at CRF 38, {len(by_qid) - len(stable) - len(degraded)} had other patterns")

if __name__ == "__main__":
    main()
