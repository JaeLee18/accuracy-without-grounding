"""
Proprietary model inference via OpenRouter API.
Runs Video-MME, MVBench, and EgoSchema with original + black screen conditions.
Fully resumable. Saves results incrementally.
 pip install requests tqdm opencv-python
 # Set OPENROUTER_API_KEY in your environment before running.
Usage:
    python run_openrouter.py --model gemini-flash --dataset videomme
    python run_openrouter.py --model gpt --dataset mvbench
    python run_openrouter.py --model gemini-flash --dataset all

Models:
    gemini-flash  -> google/gemini-2.5-flash
    gpt           -> openai/gpt-4o-mini  (change MODEL_IDS below for gpt-5-mini etc.)

Requires: OPENROUTER_API_KEY environment variable or --api-key flag.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------

import argparse
import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path

import subprocess
import tempfile

from dotenv import load_dotenv
load_dotenv()

import requests
from tqdm import tqdm

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("WARNING: opencv not found, using ffmpeg for frame extraction")

# ── Model IDs on OpenRouter ──────────────────────────────────────────────────
MODEL_IDS = {
    "gemini-flash": "google/gemini-2.5-flash",
    "gpt": "openai/gpt-4o-mini",
}

# ── Dataset configs ──────────────────────────────────────────────────────────
DATASETS = {
    "videomme": {
        "sample_path": VDG_DATA_ROOT + "/videomme_full/full_sample.json",
        "video_dir": VDG_DATA_ROOT + "/videomme_full/videos",
        "black_dir": VDG_RESULTS_ROOT + "/full_study/black_videos",
        "video_ext": ".mp4",
        "video_id_field": "videoID",
        "question_id_field": "question_id",
        "task_type_field": "task_type",
        "question_field": "question",
        "options_field": "options",
        "answer_field": "answer",
        "num_choices": 4,
    },
    "mvbench": {
        "sample_path": VDG_DATA_ROOT + "/mvbench/mvbench_available_sample.json",
        "video_dir": VDG_DATA_ROOT + "/mvbench/videos",
        "black_dir": VDG_DATA_ROOT + "/mvbench/black",
        "video_ext": ".mp4",
        "video_id_field": "videoID",
        "question_id_field": "question_id",
        "task_type_field": "task_type",
        "question_field": "question",
        "options_field": "options",
        "answer_field": "answer",
        "num_choices": 4,
    },
    "egoschema": {
        "sample_path": VDG_DATA_ROOT + "/egoschema/egoschema_subset.json",
        "video_dir": VDG_DATA_ROOT + "/egoschema/videos",
        "black_dir": VDG_DATA_ROOT + "/egoschema/black",
        "video_ext": ".mp4",
        "video_id_field": "video_uid",
        "question_id_field": "question_id",
        "task_type_field": "task_type",
        "question_field": "question",
        "options_field": "options",
        "answer_field": "answer",
        "num_choices": 5,
    },
}

RESULTS_DIR = VDG_RESULTS_ROOT + "/proprietary"
FRAME_CACHE_DIR = VDG_RESULTS_ROOT + "/proprietary/frame_cache"
FPS = 0.25
MAX_FRAMES = 16  # cap frames for very long videos
RETRY_LIMIT = 5
RETRY_BACKOFF = 2  # seconds, doubles each retry

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path: str, fps: float = 0.25, max_frames: int = 16) -> list[str]:
    """Extract frames with disk cache. Same video+condition reuses cached frames."""
    # Cache key: video filename + fps + max_frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    parent_dir = os.path.basename(os.path.dirname(video_path))
    cache_key = f"{parent_dir}__{video_name}__fps{fps}__max{max_frames}"
    cache_path = os.path.join(FRAME_CACHE_DIR, f"{cache_key}.json")

    # Try cache first
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    # Extract frames
    if HAS_CV2:
        frames = _extract_frames_cv2(video_path, fps, max_frames)
    else:
        frames = _extract_frames_ffmpeg(video_path, fps, max_frames)

    # Save to cache
    os.makedirs(FRAME_CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(frames, f)

    return frames


def _extract_frames_cv2(video_path: str, fps: float, max_frames: int) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    frame_interval = max(1, int(video_fps / fps))
    frames_b64 = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            h, w = frame.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames_b64.append(base64.b64encode(buf).decode("utf-8"))
            if len(frames_b64) >= max_frames:
                break
        frame_idx += 1
    cap.release()

    if not frames_b64:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frames_b64


def _extract_frames_ffmpeg(video_path: str, fps: float, max_frames: int) -> list[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps},scale='min(512,iw)':min'(512,ih)':force_original_aspect_ratio=decrease",
            "-frames:v", str(max_frames),
            "-q:v", "5",
            os.path.join(tmpdir, "frame_%04d.jpg"),
            "-y", "-loglevel", "error",
        ]
        subprocess.run(cmd, check=True, timeout=60)

        frames_b64 = []
        for fname in sorted(os.listdir(tmpdir)):
            if fname.endswith(".jpg"):
                with open(os.path.join(tmpdir, fname), "rb") as f:
                    frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))

    if not frames_b64:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frames_b64


# ── API call ─────────────────────────────────────────────────────────────────

def call_openrouter(
    api_key: str,
    model_id: str,
    frames_b64: list[str],
    prompt_text: str,
) -> str:
    """Send frames + prompt to OpenRouter, return model response text."""

    # Build content: frames as image_url, then text
    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low",
            },
        })
    content.append({"type": "text", "text": prompt_text})

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 32,
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/acm-mm-vdg",
        "X-Title": "VDG Benchmark Study",
    }

    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)

            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"\n  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                error_text = resp.text[:200]
                if attempt < RETRY_LIMIT - 1:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    print(f"\n  HTTP {resp.status_code}: {error_text}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"HTTP {resp.status_code}: {error_text}")

            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            if attempt < RETRY_LIMIT - 1:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"\n  Timeout, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Failed after {RETRY_LIMIT} retries")


# ── Prompt construction ──────────────────────────────────────────────────────

def build_prompt(question: str, options: list[str], num_choices: int) -> str:
    """Build the MCQ prompt matching the format used for open-weight models."""
    options_text = "\n".join(options)
    letters = "A, B, C, D" if num_choices == 4 else "A, B, C, D, or E"
    return (
        f"{question}\n\n"
        f"{options_text}\n\n"
        f"Answer with only the letter ({letters}) of the correct option."
    )


def extract_prediction(response: str, num_choices: int) -> str | None:
    """Extract answer letter from model response."""
    valid = set("ABCDE"[:num_choices])
    # Try first letter that's valid
    for char in response.upper():
        if char in valid:
            return char
    return response[:10]  # fallback: raw response


# ── Results I/O ──────────────────────────────────────────────────────────────

def results_path(model_short: str, dataset: str, condition: str) -> str:
    return os.path.join(RESULTS_DIR, f"{model_short}_{dataset}_{condition}_results.json")


def load_results(path: str) -> list[dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_results(path: str, results: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Main inference loop ──────────────────────────────────────────────────────

def run_dataset(
    api_key: str,
    model_short: str,
    model_id: str,
    dataset_name: str,
    condition: str,  # "original" or "black"
):
    cfg = DATASETS[dataset_name]

    # Load samples
    with open(cfg["sample_path"]) as f:
        samples = json.load(f)
    print(f"\n{'='*60}")
    print(f"  Model: {model_id}")
    print(f"  Dataset: {dataset_name} ({len(samples)} questions)")
    print(f"  Condition: {condition}")
    print(f"{'='*60}")

    video_dir = cfg["video_dir"] if condition == "original" else cfg["black_dir"]

    # Load existing results for resume
    rpath = results_path(model_short, dataset_name, condition)
    results = load_results(rpath)
    completed = {r[cfg["question_id_field"]] for r in results}
    print(f"  Already completed: {len(completed)}/{len(samples)}")

    remaining = [s for s in samples if s[cfg["question_id_field"]] not in completed]
    if not remaining:
        print("  All done!")
        return results

    correct_count = sum(1 for r in results if r.get("correct"))
    total_done = len([r for r in results if not r.get("error")])

    pbar = tqdm(remaining, desc=f"{model_short}/{dataset_name}/{condition}")
    for sample in pbar:
        qid = sample[cfg["question_id_field"]]
        vid = str(sample[cfg["video_id_field"]])
        video_path = os.path.join(video_dir, vid + cfg["video_ext"])

        if not os.path.exists(video_path):
            # Try without extension change
            alt_path = os.path.join(video_dir, vid + ".mp4")
            if os.path.exists(alt_path):
                video_path = alt_path
            else:
                entry = {
                    cfg["question_id_field"]: qid,
                    cfg["video_id_field"]: vid,
                    "task_type": sample.get(cfg["task_type_field"], "unknown"),
                    "condition": condition,
                    "ground_truth": sample[cfg["answer_field"]],
                    "prediction": None,
                    "correct": False,
                    "error": f"Video not found: {video_path}",
                }
                results.append(entry)
                save_results(rpath, results)
                continue

        try:
            frames = extract_frames(video_path, fps=FPS, max_frames=MAX_FRAMES)
            prompt = build_prompt(
                sample[cfg["question_field"]],
                sample[cfg["options_field"]],
                cfg["num_choices"],
            )
            response = call_openrouter(api_key, model_id, frames, prompt)
            pred = extract_prediction(response, cfg["num_choices"])
            gt = sample[cfg["answer_field"]]

            entry = {
                cfg["question_id_field"]: qid,
                cfg["video_id_field"]: vid,
                "task_type": sample.get(cfg["task_type_field"], "unknown"),
                "condition": condition,
                "ground_truth": gt,
                "prediction": pred,
                "correct": pred == gt,
                "raw_response": response,
            }
            correct_count += int(pred == gt)
            total_done += 1

        except Exception as e:
            traceback.print_exc()
            entry = {
                cfg["question_id_field"]: qid,
                cfg["video_id_field"]: vid,
                "task_type": sample.get(cfg["task_type_field"], "unknown"),
                "condition": condition,
                "ground_truth": sample[cfg["answer_field"]],
                "prediction": None,
                "correct": False,
                "error": str(e),
            }

        results.append(entry)
        completed.add(qid)
        save_results(rpath, results)

        if total_done > 0:
            pbar.set_postfix_str(f"acc={correct_count}/{total_done} ({correct_count/total_done:.1%})")

        # Small delay to avoid rate limits
        time.sleep(0.3)

    print(f"\n  Done: {correct_count}/{total_done} correct ({correct_count/total_done:.1%})")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Proprietary model VDG inference via OpenRouter")
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS.keys()),
                        help="Model shorthand (gemini-flash, gpt)")
    parser.add_argument("--dataset", required=True,
                        choices=["videomme", "mvbench", "egoschema", "all"],
                        help="Dataset to run")
    parser.add_argument("--condition", default="both",
                        choices=["original", "black", "both"],
                        help="Condition to run (default: both)")
    parser.add_argument("--api-key", default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--fps", type=float, default=0.25,
                        help="Frame sampling rate (default: 0.25)")
    parser.add_argument("--max-frames", type=int, default=16,
                        help="Max frames per video (default: 16)")
    args = parser.parse_args()

    global FPS, MAX_FRAMES
    FPS = args.fps
    MAX_FRAMES = args.max_frames

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY env var or pass --api-key")
        sys.exit(1)

    model_id = MODEL_IDS[args.model]
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    conditions = ["original", "black"] if args.condition == "both" else [args.condition]

    print(f"Model: {model_id}")
    print(f"Datasets: {datasets}")
    print(f"Conditions: {conditions}")
    print(f"FPS: {FPS}, Max frames: {MAX_FRAMES}")
    print(f"Results dir: {RESULTS_DIR}")

    for ds in datasets:
        for cond in conditions:
            run_dataset(api_key, args.model, model_id, ds, cond)

    print("\n=== ALL INFERENCE COMPLETE ===")

    # Print summary
    print("\n--- Summary ---")
    for ds in datasets:
        for cond in conditions:
            rpath = results_path(args.model, ds, cond)
            if os.path.exists(rpath):
                results = load_results(rpath)
                n = len([r for r in results if not r.get("error")])
                c = len([r for r in results if r.get("correct")])
                e = len([r for r in results if r.get("error")])
                acc = c / n if n > 0 else 0
                print(f"  {ds}/{cond}: {c}/{n} correct ({acc:.1%}), {e} errors")


if __name__ == "__main__":
    main()
