"""
Scale test: Qwen2-VL-2B-Instruct on Video-MME (600 questions).
Two conditions: original + black screen. Fully resumable.

Usage:
  export DATA_ROOT=/path/to/cluster/data
  export RESULTS_ROOT=/path/to/cluster/results
  python run_qwen2b_videomme.py
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
RESULTS_PATH = os.path.join(RESULTS_ROOT, "qwen2b_videomme_results.json")
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
BLACK_DIR = os.path.join(DATA_ROOT, "black_2b")
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MAX_PIXELS = 256 * 256
MIN_PIXELS = 28 * 28
FPS = 0.25
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
        _print_summary(results)
        return

    pbar = tqdm(work, desc="Qwen2-VL-2B")
    for sample, condition in pbar:
        video_id = sample["videoID"]
        orig_path = os.path.abspath(os.path.join(VIDEO_DIR, f"{video_id}.mp4"))

        if condition == "black":
            duration = get_video_duration(orig_path)
            video_path = make_black_video(video_id, duration)
        else:
            video_path = orig_path

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
        torch.cuda.empty_cache()

        n_ok = sum(1 for r in results if not r.get("error"))
        n_correct = sum(1 for r in results if r.get("correct"))
        pbar.set_postfix_str(f"acc={n_correct}/{n_ok}")

    # Need second pass for black conditions of newly completed originals
    results2 = load_results()
    completed2 = {(r["question_id"], r["condition"]) for r in results2 if not r.get("error")}
    work2 = []
    for s in samples:
        if (s["question_id"], "black") not in completed2 and (s["question_id"], "original") in completed2:
            work2.append((s, "black"))
    if work2:
        print(f"\nSecond pass: {len(work2)} black conditions remaining")
        pbar2 = tqdm(work2, desc="Qwen2-VL-2B-pass2")
        for sample, condition in pbar2:
            video_id = sample["videoID"]
            orig_path = os.path.abspath(os.path.join(VIDEO_DIR, f"{video_id}.mp4"))
            duration = get_video_duration(orig_path)
            video_path = make_black_video(video_id, duration)
            try:
                pred = run_single(model, processor, video_path, sample["question"], sample["options"])
                entry = {"question_id": sample["question_id"], "videoID": video_id,
                         "task_type": sample["task_type"], "condition": "black",
                         "ground_truth": sample["answer"], "prediction": pred,
                         "correct": pred == sample["answer"], "error": None}
            except Exception as e:
                traceback.print_exc()
                entry = {"question_id": sample["question_id"], "videoID": video_id,
                         "task_type": sample["task_type"], "condition": "black",
                         "ground_truth": sample["answer"], "prediction": None,
                         "correct": False, "error": str(e)}
            results2.append(entry)
            save_results(results2)
            torch.cuda.empty_cache()
        results = results2

    print("\n=== QWEN2-VL-2B VIDEO-MME COMPLETE ===")
    _print_summary(results)


def _print_summary(results):
    from collections import defaultdict
    stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for r in results:
        if not r.get("error"):
            stats[r["task_type"]][r["condition"]]["total"] += 1
            stats[r["task_type"]][r["condition"]]["correct"] += int(r.get("correct", False))

    print(f"\n{'Task Type':<25} {'Orig':>8} {'Black':>8} {'VGG':>8}")
    print("-" * 52)
    all_o_c, all_o_n, all_b_c, all_b_n = 0, 0, 0, 0
    for tt in sorted(stats.keys()):
        o = stats[tt].get("original", {"correct": 0, "total": 0})
        b = stats[tt].get("black", {"correct": 0, "total": 0})
        oa = o["correct"] / o["total"] if o["total"] else 0
        ba = b["correct"] / b["total"] if b["total"] else 0
        print(f"{tt:<25} {oa:>8.3f} {ba:>8.3f} {oa - ba:>8.3f}")
        all_o_c += o["correct"]; all_o_n += o["total"]
        all_b_c += b["correct"]; all_b_n += b["total"]
    if all_o_n and all_b_n:
        oa = all_o_c / all_o_n; ba = all_b_c / all_b_n
        print(f"{'OVERALL':<25} {oa:>8.3f} {ba:>8.3f} {oa - ba:>8.3f}")


if __name__ == "__main__":
    main()
