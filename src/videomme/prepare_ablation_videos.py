"""
Prepare ablation videos for spatial/temporal decomposition experiment.
Creates two new video conditions:
  1. Shuffled frames - randomly reorder frames (preserves appearance, destroys temporal order)
  2. Single frame   - middle frame repeated (preserves one frame's spatial content, removes all temporal info)

Videos are cached to avoid regeneration. Uses fixed random seed for reproducibility.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import random
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
from imageio_ffmpeg import get_ffmpeg_exe

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
VIDEO_BASE = VDG_DATA_ROOT + "/videomme_full/videos"
SHUFFLED_DIR = VDG_RESULTS_ROOT + "/full_study/shuffled_videos"
SINGLEFRAME_DIR = VDG_RESULTS_ROOT + "/full_study/singleframe_videos"
TEMP_DIR = VDG_RESULTS_ROOT + "/full_study/_tmp_frames"
SEED = 42

FFMPEG = os.environ.get("FFMPEG_BIN", get_ffmpeg_exe())


def get_video_info(video_path):
    """Get duration and fps of a video."""
    cmd = [FFMPEG, "-i", video_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    duration = 10.0
    fps = 30.0
    import re
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if m:
        h, mn, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = h * 3600 + mn * 60 + sec
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*fps", r.stderr)
    if m2:
        fps = float(m2.group(1))
    return duration, fps


def extract_frames(video_path, frame_dir):
    """Extract all frames from video to frame_dir as numbered PNGs."""
    os.makedirs(frame_dir, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-i", video_path,
        "-q:v", "2",
        "-loglevel", "error",
        os.path.join(frame_dir, "frame_%06d.png"),
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(Path(frame_dir).glob("frame_*.png"))
    return [str(f) for f in frames]


def make_shuffled_video(video_id, video_path, seed=SEED):
    """Create a video with shuffled frame order. Cached by video_id."""
    os.makedirs(SHUFFLED_DIR, exist_ok=True)
    out_path = os.path.join(SHUFFLED_DIR, f"{video_id}.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    duration, fps = get_video_info(video_path)
    frame_dir = os.path.join(TEMP_DIR, f"shuf_{video_id}")
    try:
        frames = extract_frames(video_path, frame_dir)
        if not frames:
            raise RuntimeError(f"No frames extracted from {video_path}")

        # Shuffle with deterministic seed per video
        rng = random.Random(seed)
        shuffled = list(frames)
        rng.shuffle(shuffled)

        # Create concat file for ffmpeg
        reordered_dir = os.path.join(TEMP_DIR, f"shuf_reord_{video_id}")
        os.makedirs(reordered_dir, exist_ok=True)
        for i, src in enumerate(shuffled):
            dst = os.path.join(reordered_dir, f"frame_{i:06d}.png")
            shutil.copy2(src, dst)

        # Re-encode from reordered frames
        cmd = [
            FFMPEG, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(reordered_dir, "frame_%06d.png"),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        # Cleanup temp frames
        for d in [frame_dir, os.path.join(TEMP_DIR, f"shuf_reord_{video_id}")]:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

    return out_path


def make_singleframe_video(video_id, video_path):
    """Create a video where the middle frame is repeated for the original duration."""
    os.makedirs(SINGLEFRAME_DIR, exist_ok=True)
    out_path = os.path.join(SINGLEFRAME_DIR, f"{video_id}.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    duration, fps = get_video_info(video_path)

    # Extract middle frame
    mid_time = duration / 2.0
    frame_path = os.path.join(TEMP_DIR, f"mid_{video_id}.png")
    os.makedirs(TEMP_DIR, exist_ok=True)
    cmd = [
        FFMPEG, "-y",
        "-ss", f"{mid_time:.2f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-loglevel", "error",
        frame_path,
    ]
    subprocess.run(cmd, check=True)

    if not os.path.exists(frame_path):
        # Fallback: extract first frame
        cmd2 = [
            FFMPEG, "-y", "-i", video_path,
            "-frames:v", "1", "-q:v", "2",
            "-loglevel", "error", frame_path,
        ]
        subprocess.run(cmd2, check=True)

    # Create video from single frame looped for original duration
    cmd = [
        FFMPEG, "-y",
        "-loop", "1",
        "-i", frame_path,
        "-t", f"{duration:.1f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-r", "1",  # 1 fps is enough since it's all the same frame
        "-loglevel", "error",
        out_path,
    ]
    subprocess.run(cmd, check=True)

    # Cleanup
    if os.path.exists(frame_path):
        os.remove(frame_path)

    return out_path


def main():
    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    # Get unique video IDs
    video_ids = sorted(set(s["videoID"] for s in samples))
    print(f"Total videos to process: {len(video_ids)}")

    # Prepare shuffled videos
    print("\n=== Creating shuffled-frame videos ===")
    errors_shuf = 0
    for vid in tqdm(video_ids, desc="Shuffled"):
        video_path = os.path.join(VIDEO_BASE, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"  SKIP {vid}: source not found")
            errors_shuf += 1
            continue
        try:
            make_shuffled_video(vid, video_path)
        except Exception as e:
            print(f"  ERROR {vid}: {e}")
            errors_shuf += 1

    # Prepare single-frame videos
    print("\n=== Creating single-frame videos ===")
    errors_sf = 0
    for vid in tqdm(video_ids, desc="Single frame"):
        video_path = os.path.join(VIDEO_BASE, f"{vid}.mp4")
        if not os.path.exists(video_path):
            errors_sf += 1
            continue
        try:
            make_singleframe_video(vid, video_path)
        except Exception as e:
            print(f"  ERROR {vid}: {e}")
            errors_sf += 1

    # Cleanup temp dir
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    shuf_count = len(list(Path(SHUFFLED_DIR).glob("*.mp4"))) if os.path.exists(SHUFFLED_DIR) else 0
    sf_count = len(list(Path(SINGLEFRAME_DIR).glob("*.mp4"))) if os.path.exists(SINGLEFRAME_DIR) else 0
    print(f"\nDone! Shuffled: {shuf_count} videos ({errors_shuf} errors), "
          f"Single-frame: {sf_count} videos ({errors_sf} errors)")


if __name__ == "__main__":
    main()
