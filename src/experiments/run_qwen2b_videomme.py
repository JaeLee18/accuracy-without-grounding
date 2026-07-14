"""
Scale test: Qwen2-VL-2B-Instruct on Video-MME (600 questions).
Two conditions: original + black screen. Fully resumable.
Compare VGG spectrum to the 7B model.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os, subprocess, re, traceback
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import imageio_ffmpeg

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/experiments/qwen2b_videomme_results.json"
VIDEO_DIR = VDG_DATA_ROOT + "/videomme_full/videos"
BLACK_DIR = VDG_DATA_ROOT + "/videomme_full/black_2b"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MAX_PIXELS = 256 * 256
MIN_PIXELS = 28 * 28
FPS = 0.25
CONDITIONS = ["original", "black"]

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(BLACK_DIR, exist_ok=True)
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()


def get_video_duration(video_path):
    result = subprocess.run([ffmpeg_exe, "-i", video_path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 30.0


def make_black_video(video_id, duration):
    out = os.path.join(BLACK_DIR, f"{video_id}.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    cmd = [ffmpeg_exe, "-y", "-f", "lavfi",
           "-i", f"color=c=black:s=320x240:d={duration:.2f}",
           "-c:v", "libx264", "-t", str(duration), out]
    subprocess.run(cmd, capture_output=True)
    return out


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
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    print(f"Dataset: {len(samples)} questions x {len(CONDITIONS)} conditions")

    model, processor = load_model()
    results = load_results()
    completed = {(r["question_id"], r["condition"]) for r in results if not r.get("error")}
    print(f"Already done: {len(completed)} | Target: {len(samples) * len(CONDITIONS)}")

    # Build work list — original first so we can read duration for black
    work = []
    for s in samples:
        for cond in CONDITIONS:
            if (s["question_id"], cond) in completed:
                continue
            if cond == "black" and (s["question_id"], "original") not in completed:
                continue
            work.append((s, cond))

    print(f"Remaining: {len(work)}")
    if not work:
        print("All done!")
        _print_summary(results)
        return

    pbar = tqdm(work, desc="Qwen2-VL-2B")
    for sample, condition in pbar:
        video_id = sample["videoID"]
        orig_path = os.path.abspath(os.path.join(VIDEO_DIR, f"{video_id}.mp4")).replace("\\", "/")

        if condition == "black":
            duration = get_video_duration(orig_path)
            video_path = make_black_video(video_id, duration)
        else:
            video_path = orig_path
        video_path = os.path.abspath(video_path).replace("\\", "/")

        try:
            pred = run_single(model, processor, video_path, sample["question"], sample["options"])
            entry = {
                "question_id": sample["question_id"],
                "videoID": video_id,
                "task_type": sample["task_type"],
                "condition": condition,
                "ground_truth": sample["answer"],
                "prediction": pred,
                "correct": pred == sample["answer"],
                "error": None,
            }
        except Exception as e:
            traceback.print_exc()
            entry = {
                "question_id": sample["question_id"],
                "videoID": video_id,
                "task_type": sample["task_type"],
                "condition": condition,
                "ground_truth": sample["answer"],
                "prediction": None,
                "correct": False,
                "error": str(e),
            }

        results.append(entry)
        completed.add((sample["question_id"], condition))
        save_results(results)

        n_ok = sum(1 for r in results if not r.get("error"))
        n_correct = sum(1 for r in results if r.get("correct"))
        pbar.set_postfix_str(f"acc={n_correct}/{n_ok}")

    print("\n=== QWEN2-VL-2B VIDEO-MME COMPLETE ===")
    _print_summary(results)


def _print_summary(results):
    from collections import defaultdict
    stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for r in results:
        if not r.get("error"):
            stats[r["task_type"]][r["condition"]]["total"] += 1
            stats[r["task_type"]][r["condition"]]["correct"] += int(r.get("correct", False))

    print("\nQwen2-VL-2B Video-MME Results:")
    print(f"{'Task Type':<25} {'Orig':>8} {'Black':>8} {'VGG':>8}")
    print("-" * 52)
    for tt in sorted(stats.keys()):
        orig = stats[tt].get("original", {"correct": 0, "total": 0})
        black = stats[tt].get("black", {"correct": 0, "total": 0})
        orig_acc = orig["correct"] / orig["total"] if orig["total"] else 0
        black_acc = black["correct"] / black["total"] if black["total"] else 0
        vgg = orig_acc - black_acc
        print(f"{tt:<25} {orig_acc:>8.3f} {black_acc:>8.3f} {vgg:>8.3f}")

    # Overall
    all_orig = sum(stats[tt].get("original", {"correct": 0})["correct"] for tt in stats)
    all_orig_n = sum(stats[tt].get("original", {"total": 0})["total"] for tt in stats)
    all_black = sum(stats[tt].get("black", {"correct": 0})["correct"] for tt in stats)
    all_black_n = sum(stats[tt].get("black", {"total": 0})["total"] for tt in stats)
    if all_orig_n and all_black_n:
        print(f"{'OVERALL':<25} {all_orig/all_orig_n:>8.3f} {all_black/all_black_n:>8.3f} {all_orig/all_orig_n - all_black/all_black_n:>8.3f}")


if __name__ == "__main__":
    main()
