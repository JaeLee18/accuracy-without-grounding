"""
Qwen2-VL-7B-Instruct inference on EgoSchema subset (500 questions).
Two conditions: original | black.

All results go into a single JSON file. Resume key: (question_id, condition).
Condition order: original -> black (original must be processed before black so
duration can be read from the original video).

Videos expected at: data/egoschema/videos/{q_uid}.mp4
Black screens go to: data/egoschema/black/{q_uid}.mp4
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------

import json
import os
import re
import subprocess
import traceback

import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import imageio_ffmpeg

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SAMPLE_PATH  = VDG_DATA_ROOT + "/egoschema/egoschema_subset.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/egoschema/qwen2vl_egoschema_results.json"
VIDEOS_DIR   = VDG_DATA_ROOT + "/egoschema/videos"
BLACK_DIR    = VDG_DATA_ROOT + "/egoschema/black"
MODEL_ID     = "Qwen/Qwen2-VL-7B-Instruct"
CONDITIONS   = ["original", "black"]
MAX_PIXELS   = 256 * 256
MIN_PIXELS   = 28 * 28
# FPS chosen so that ~3-min EgoSchema clips yield ~32 frames (≈ LLaVA-Video cap):
#   180 s × 0.178 fps ≈ 32 frames
FPS          = 0.178

VALID_LETTERS = "ABCDE"

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(BLACK_DIR, exist_ok=True)

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def get_video_duration(video_path):
    """Return video duration in seconds by parsing ffmpeg stderr."""
    result = subprocess.run(
        [ffmpeg_exe, "-i", video_path],
        capture_output=True, text=True,
    )
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    if m:
        h, mn, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mn * 60 + sec
    return 180.0  # EgoSchema default fallback (~3 min)


def make_black_video(video_id, duration):
    """Create a black screen mp4 of the given duration; return its path."""
    out = os.path.join(BLACK_DIR, f"{video_id}.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s=320x240:d={duration:.2f}",
        "-c:v", "libx264",
        "-t", str(duration),
        out,
    ]
    subprocess.run(cmd, capture_output=True)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    print(f"Loading {MODEL_ID} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    print("Model loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_single(model, processor, video_path, question, options):
    """
    Run one question through Qwen2-VL. Returns the predicted letter (A-E).
    Falls back to the raw response prefix if no valid letter is found.
    """
    options_text = "\n".join(options)           # already formatted "A. ..."
    prompt = (
        f"{question}\n\n"
        f"{options_text}\n\n"
        f"Answer with only the letter (A, B, C, D, or E) of the correct option."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": MAX_PIXELS,
                    "min_pixels": MIN_PIXELS,
                    "fps": FPS,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32)

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    # Extract first valid letter from response
    for char in response:
        if char.upper() in VALID_LETTERS:
            return char.upper()
    return response[:5]   # fallback: keep raw snippet for inspection


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert torch.cuda.is_available(), "CUDA not available — aborting."
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Dataset: {len(samples)} questions × {len(CONDITIONS)} conditions "
          f"= {len(samples) * len(CONDITIONS)} total items")

    model, processor = load_model()

    results   = load_results()
    # Exclude error entries so missing videos can be retried after download
    completed = {(r["question_id"], r["condition"]) for r in results if not r.get("error")}
    print(f"Already done (no-error): {len(completed)} | Total results: {len(results)} | Target: {len(samples) * len(CONDITIONS)}")

    # Build work list, respecting condition order:
    # black requires the original video to exist (to read duration), so
    # we only queue black if original is already completed.
    work = []
    for s in samples:
        for cond in CONDITIONS:
            if (s["question_id"], cond) in completed:
                continue
            if cond == "black" and (s["question_id"], "original") not in completed:
                # Original not yet done; will be re-queued in a future run.
                continue
            work.append((s, cond))

    print(f"Remaining: {len(work)}")
    if not work:
        print("Nothing to do. Exiting.")
        _print_summary(results)
        return

    pbar = tqdm(work, desc="Qwen-EgoSchema")
    for item, condition in pbar:
        qid      = item["question_id"]
        video_id = item["video_uid"]
        orig_path = os.path.abspath(
            os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
        ).replace("\\", "/")

        if condition == "black":
            duration   = get_video_duration(orig_path)
            video_path = make_black_video(video_id, duration)
        else:
            video_path = orig_path

        video_path = os.path.abspath(video_path).replace("\\", "/")

        try:
            pred_letter = run_single(
                model, processor, video_path,
                item["question"], item["options"],
            )
            entry = {
                "question_id": qid,
                "video_uid":   video_id,
                "task_type":   item["task_type"],
                "condition":   condition,
                "ground_truth": item["answer"],
                "prediction":  pred_letter,
                "correct":     pred_letter == item["answer"],
                "error":       None,
            }
        except Exception as e:
            traceback.print_exc()
            entry = {
                "question_id": qid,
                "video_uid":   video_id,
                "task_type":   item["task_type"],
                "condition":   condition,
                "ground_truth": item["answer"],
                "prediction":  None,
                "correct":     False,
                "error":       str(e),
            }

        results.append(entry)
        completed.add((qid, condition))
        save_results(results)

        # Live accuracy in progress bar (exclude errored entries)
        n_correct = sum(1 for r in results if r.get("correct"))
        n_valid   = sum(1 for r in results if r.get("error") is None)
        pbar.set_postfix_str(f"acc={n_correct}/{n_valid}")

    print("\n=== QWEN EgoSchema COMPLETE ===")
    _print_summary(results)


def _print_summary(results):
    """Print per-condition accuracy from a completed (or partial) result list."""
    from collections import defaultdict

    cond_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r.get("error") is None:
            cond = r["condition"]
            cond_stats[cond]["total"]   += 1
            cond_stats[cond]["correct"] += int(r.get("correct", False))

    print("\nPer-condition accuracy (error-free entries):")
    for cond in CONDITIONS:
        s = cond_stats[cond]
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
            print(f"  {cond:<10} {s['correct']:>4}/{s['total']:<4}  acc={acc:.4f}")
        else:
            print(f"  {cond:<10} no results yet")

    # Overall
    total   = sum(s["total"]   for s in cond_stats.values())
    correct = sum(s["correct"] for s in cond_stats.values())
    if total:
        print(f"\n  Overall     {correct:>4}/{total:<4}  acc={correct/total:.4f}")

    n_errors = sum(1 for r in results if r.get("error") is not None)
    if n_errors:
        print(f"\n  Errors: {n_errors} (check results file for details)")


if __name__ == "__main__":
    main()
