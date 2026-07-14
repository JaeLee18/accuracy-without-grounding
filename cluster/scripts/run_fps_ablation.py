"""
FPS Ablation: Qwen2-VL-7B on Temporal Reasoning questions at 0.5, 1.0, 2.0 FPS.
Baseline is 0.25 FPS (already in full_study results).
Runs both original video and black screen at each FPS level for VGG computation.

Usage:
  export DATA_ROOT=/path/to/cluster/data   # contains full_sample.json + videos/
  export RESULTS_ROOT=/path/to/cluster/results
  python run_fps_ablation.py
"""
import json, os, subprocess, re, traceback
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import imageio_ffmpeg

DATA_ROOT = os.environ.get("DATA_ROOT", "../data")
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "../results")

SAMPLE_PATH = os.path.join(DATA_ROOT, "full_sample.json")
RESULTS_PATH = os.path.join(RESULTS_ROOT, "fps_ablation_results.json")
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
BLACK_DIR = os.path.join(DATA_ROOT, "black_fps")
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 256 * 256
MIN_PIXELS = 28 * 28
FPS_LEVELS = [0.5, 1.0, 2.0]
CONDITIONS = ["original", "black"]

os.makedirs(RESULTS_ROOT, exist_ok=True)
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


def run_single(model, processor, video_path, question, options, fps):
    options_text = "\n".join(options)
    prompt = (
        f"{question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path,
             "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "fps": fps},
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
        all_samples = json.load(f)
    samples = [s for s in all_samples if s["task_type"] == "Temporal Reasoning"]
    print(f"Temporal Reasoning questions: {len(samples)}")

    model, processor = load_model()
    results = load_results()
    completed = {(r["question_id"], r["condition"], r["fps"]) for r in results if not r.get("error")}
    print(f"Already done: {len(completed)}")

    work = []
    for s in samples:
        for fps in FPS_LEVELS:
            for cond in CONDITIONS:
                if (s["question_id"], cond, fps) not in completed:
                    work.append((s, cond, fps))

    print(f"Remaining: {len(work)}")
    if not work:
        _print_summary(results)
        return

    pbar = tqdm(work, desc="FPS-Ablation")
    for sample, condition, fps in pbar:
        video_id = sample["videoID"]
        orig_path = os.path.abspath(os.path.join(VIDEO_DIR, f"{video_id}.mp4"))

        if condition == "black":
            duration = get_video_duration(orig_path)
            video_path = make_black_video(video_id, duration)
        else:
            video_path = orig_path

        try:
            pred = run_single(model, processor, video_path, sample["question"], sample["options"], fps)
            entry = {
                "question_id": sample["question_id"],
                "videoID": video_id,
                "task_type": sample["task_type"],
                "condition": condition,
                "fps": fps,
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
                "fps": fps,
                "ground_truth": sample["answer"],
                "prediction": None,
                "correct": False,
                "error": str(e),
            }

        results.append(entry)
        completed.add((sample["question_id"], condition, fps))
        save_results(results)
        torch.cuda.empty_cache()

        n_ok = sum(1 for r in results if not r.get("error"))
        n_correct = sum(1 for r in results if r.get("correct"))
        pbar.set_postfix_str(f"acc={n_correct}/{n_ok}")

    print("\n=== FPS ABLATION COMPLETE ===")
    _print_summary(results)


def _print_summary(results):
    from collections import defaultdict
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if not r.get("error"):
            key = f"{r['condition']}@{r['fps']}fps"
            stats[key]["total"] += 1
            stats[key]["correct"] += int(r.get("correct", False))

    print("\nFPS Ablation Results (Temporal Reasoning, Qwen2-VL-7B):")
    print(f"{'Condition':<20} {'Acc':>10} {'n':>5}")
    print("-" * 38)
    for key in sorted(stats.keys()):
        s = stats[key]
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"{key:<20} {acc:>10.3f} {s['total']:>5}")

    print("\nVGG by FPS level:")
    for fps in FPS_LEVELS:
        orig_key = f"original@{fps}fps"
        black_key = f"black@{fps}fps"
        if orig_key in stats and black_key in stats:
            orig_acc = stats[orig_key]["correct"] / stats[orig_key]["total"]
            black_acc = stats[black_key]["correct"] / stats[black_key]["total"]
            print(f"  {fps} FPS: orig={orig_acc:.3f} black={black_acc:.3f} VGG={orig_acc - black_acc:.3f}")


if __name__ == "__main__":
    main()
