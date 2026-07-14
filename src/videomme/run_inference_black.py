"""
Black screen baseline for Qwen2-VL-7B-Instruct.
Replaces every video with a black video of matching duration,
so the model must answer from language priors alone.

Three-way comparison: black ≈ original ≈ CRF38  →  language prior story
                      black < original ≈ CRF38  →  model uses *some* vision, but
                                                    compression doesn't touch what it cares about
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import subprocess
import sys
import tempfile
import traceback
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from imageio_ffmpeg import get_ffmpeg_exe

SAMPLE_PATH  = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json"
VIDEO_BASE   = VDG_DATA_ROOT + "/videomme_full/videos"   # used only for duration probe
MODEL_ID     = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS   = 256 * 256
MIN_PIXELS   = 28 * 28
FPS          = 0.25   # match main inference setting

FFMPEG = os.environ.get("FFMPEG_BIN", get_ffmpeg_exe())

BLACK_VIDEO_DIR = VDG_RESULTS_ROOT + "/full_study/black_videos"


def get_video_duration(video_path):
    """Return duration in seconds via ffprobe."""
    cmd = [
        FFMPEG.replace("ffmpeg-win", "ffprobe-win").replace("ffmpeg", "ffprobe"),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    # ffprobe may not exist separately — fall back to ffmpeg
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        # use ffmpeg with -i and parse stderr
        r2 = subprocess.run(
            [FFMPEG, "-i", video_path, "-f", "null", "-"],
            capture_output=True, text=True,
        )
        for line in r2.stderr.splitlines():
            if "Duration" in line:
                parts = line.strip().split(",")[0].split()[-1].split(":")
                h, m, s = parts
                return float(h) * 3600 + float(m) * 60 + float(s)
        return 10.0   # fallback
    return max(1.0, float(result.stdout.strip()))


def make_black_video(video_id, duration):
    """Create a black mp4 of given duration. Cached by video_id."""
    os.makedirs(BLACK_VIDEO_DIR, exist_ok=True)
    out_path = f"{BLACK_VIDEO_DIR}/{video_id}.mp4"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:size=224x224:rate=1:duration={duration:.1f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-loglevel", "error",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def load_model():
    print("Loading Qwen2-VL-7B-Instruct...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    print("Model loaded.")
    return model, processor


def run_single(model, processor, video_path, question, options):
    options_text = "\n".join(options)
    prompt = (
        f"{question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path,
             "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "fps": FPS},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    for char in response:
        if char.upper() in "ABCD":
            return char.upper()
    return response[:5]


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    print(f"Questions: {len(samples)}")

    model, processor = load_model()

    results = load_results()
    completed = {r["question_id"] for r in results}
    work = [s for s in samples if s["question_id"] not in completed]
    print(f"Remaining: {len(work)} / {len(samples)}")

    if not work:
        print("All done!")
    else:
        pbar = tqdm(work, desc="Black screen")
        for sample in pbar:
            # Build black video matching original duration
            orig_path = f"{VIDEO_BASE}/{sample['videoID']}.mp4"
            try:
                duration = get_video_duration(orig_path)
            except Exception:
                duration = 10.0
            try:
                black_path = make_black_video(sample["videoID"], duration)
                black_path_fwd = black_path.replace("\\", "/")
                pred = run_single(model, processor, black_path_fwd,
                                  sample["question"], sample["options"])
                entry = {
                    "question_id": sample["question_id"],
                    "videoID": sample["videoID"],
                    "task_type": sample["task_type"],
                    "condition": "black",
                    "ground_truth": sample["answer"],
                    "prediction": pred,
                    "correct": pred == sample["answer"],
                }
            except Exception as e:
                traceback.print_exc()
                entry = {
                    "question_id": sample["question_id"],
                    "videoID": sample["videoID"],
                    "task_type": sample["task_type"],
                    "condition": "black",
                    "ground_truth": sample["answer"],
                    "prediction": None,
                    "correct": False,
                    "error": str(e),
                }
            results.append(entry)
            completed.add(sample["question_id"])
            save_results(results)
            done = len(results)
            acc = sum(1 for r in results if r.get("correct")) / done
            pbar.set_postfix_str(f"acc={acc:.3f}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== BLACK SCREEN BASELINE ===")
    from collections import defaultdict
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["task_type"]][1] += 1
        if r.get("correct"):
            by_type[r["task_type"]][0] += 1

    overall_c = sum(1 for r in results if r.get("correct"))
    print(f"Overall: {overall_c}/{len(results)} = {overall_c/len(results):.3f}  (chance=0.250)")
    print()

    # Load original + CRF38 for comparison
    try:
        with open(VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json") as f:
            main_results = json.load(f)
        orig = {r["question_id"]: r for r in main_results if r["condition"] == "original"}
        crf38 = {r["question_id"]: r for r in main_results if r["condition"] == "crf38"}

        print(f"{'Task Type':<30} {'Black':>7} {'Original':>9} {'CRF38':>7} {'Gap(B-O)':>9}")
        print("-" * 65)
        for tt in sorted(by_type):
            c, n = by_type[tt]
            black_acc = c / n if n else 0
            o_vals = [v["correct"] for v in orig.values()
                      if v.get("task_type") == tt]
            c38_vals = [v["correct"] for v in crf38.values()
                        if v.get("task_type") == tt]
            orig_acc = sum(o_vals) / len(o_vals) if o_vals else 0
            crf38_acc = sum(c38_vals) / len(c38_vals) if c38_vals else 0
            gap = black_acc - orig_acc
            print(f"{tt:<30} {black_acc:>7.3f} {orig_acc:>9.3f} {crf38_acc:>7.3f} {gap:>+9.3f}")
    except Exception:
        for tt, (c, n) in sorted(by_type.items()):
            print(f"  {tt:<30} {c}/{n} = {c/n:.3f}")

    print("\nInterpretation:")
    overall_acc = overall_c / len(results) if results else 0
    if overall_acc > 0.50:
        print("  Black >> chance (0.25): STRONG language prior signal.")
    elif overall_acc > 0.35:
        print("  Black > chance (0.25): moderate language prior signal.")
    else:
        print("  Black near chance (0.25): model uses visual signal substantially.")


if __name__ == "__main__":
    main()
