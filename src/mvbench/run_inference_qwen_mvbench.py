"""
Qwen2-VL-7B-Instruct inference on MVBench: original, CRF38, black screen.
All 3 conditions in one results file. Resume key: (question_id, condition).
Condition order: original -> crf38 -> black (original must exist before black).
Supports variable candidate counts (2-5 options: A/B/C/D/E).
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os, traceback, re, subprocess, torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import imageio_ffmpeg

SAMPLE_PATH  = VDG_DATA_ROOT + "/mvbench/mvbench_available_sample.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/mvbench/qwen2vl_mvbench_results.json"
VIDEOS_DIR   = VDG_DATA_ROOT + "/mvbench/videos"
CRF38_DIR    = VDG_DATA_ROOT + "/mvbench/crf38"
BLACK_DIR    = VDG_DATA_ROOT + "/mvbench/black"
MODEL_ID     = "Qwen/Qwen2-VL-7B-Instruct"
CONDITIONS   = ["original", "crf38", "black"]
MAX_PIXELS   = 256 * 256
MIN_PIXELS   = 28 * 28
FPS          = 1.0

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(BLACK_DIR, exist_ok=True)
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()


def get_video_duration(video_path):
    result = subprocess.run([ffmpeg_exe, "-i", video_path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    if m:
        h, mn, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mn * 60 + sec
    return 10.0


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
    print("Loading Qwen2-VL-7B-Instruct ...")
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
    n_opts = len(options)
    valid_letters = "ABCDE"[:n_opts]
    prompt = (f"{question}\n\n{options_text}\n\n"
              f"Answer with only the letter ({', '.join(valid_letters)}) of the correct option.")
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path,
         "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "fps": FPS},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    for char in response:
        if char.upper() in valid_letters:
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
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(SAMPLE_PATH, encoding="latin-1") as f:
        samples = json.load(f)

    model, processor = load_model()
    results = load_results()
    completed = {(r["question_id"], r["condition"]) for r in results}
    print(f"Existing: {len(results)} | Total target: {len(samples) * 3}")

    work = []
    for s in samples:
        for cond in CONDITIONS:
            if (s["question_id"], cond) not in completed:
                if cond == "black" and (s["question_id"], "original") not in completed:
                    continue
                work.append((s, cond))
    print(f"Remaining: {len(work)}")

    pbar = tqdm(work, desc="Qwen-MVBench")
    for sample, condition in pbar:
        video_id = sample["videoID"]
        if condition == "black":
            orig_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
            duration = get_video_duration(orig_path)
            video_path = make_black_video(video_id, duration)
        elif condition == "crf38":
            video_path = os.path.join(CRF38_DIR, f"{video_id}.mp4")
        else:
            video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
        video_path = os.path.abspath(video_path).replace("\\", "/")

        try:
            pred = run_single(model, processor, video_path,
                              sample["question"], sample["options"])
            entry = {
                "question_id": sample["question_id"],
                "videoID": video_id,
                "task_type": sample["task_type"],
                "condition": condition,
                "ground_truth": sample["answer"],
                "prediction": pred,
                "correct": pred == sample["answer"],
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
        n_correct = sum(1 for r in results if r.get("correct"))
        n_valid = sum(1 for r in results if not r.get("error"))
        pbar.set_postfix_str(f"acc={n_correct}/{n_valid}")

    print("\n=== QWEN MVBench COMPLETE ===")
    from collections import defaultdict
    tt_accs = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r.get("error"):
            tt_accs[r["task_type"]][r["condition"]].append(r["correct"])
    for tt in sorted(tt_accs):
        accs = {c: sum(v)/len(v) for c, v in tt_accs[tt].items() if v}
        print(f"  {tt:<30} orig={accs.get('original',0):.3f} "
              f"crf38={accs.get('crf38',0):.3f} black={accs.get('black',0):.3f}")


if __name__ == "__main__":
    main()
