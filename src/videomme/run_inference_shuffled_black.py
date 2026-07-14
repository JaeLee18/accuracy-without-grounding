"""
Exp F: Black screen with shuffled answer options.
Tests whether language prior comes from option content/ordering vs question text.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import subprocess
import traceback
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from imageio_ffmpeg import get_ffmpeg_exe

SHUFFLED_SAMPLE = VDG_DATA_ROOT + "/videomme_full/shuffled_sample.json"
RESULTS_PATH    = VDG_RESULTS_ROOT + "/full_study/qwen2vl_shuffled_black_results.json"
VIDEO_BASE      = VDG_DATA_ROOT + "/videomme_full/videos"
BLACK_VIDEO_DIR = VDG_RESULTS_ROOT + "/full_study/black_videos"
MODEL_ID        = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS      = 256 * 256
MIN_PIXELS      = 28 * 28
FPS             = 0.25

FFMPEG = os.environ.get("FFMPEG_BIN", get_ffmpeg_exe())


def get_video_duration(video_path):
    r = subprocess.run(
        [FFMPEG, "-i", video_path, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for line in r.stderr.splitlines():
        if "Duration" in line:
            parts = line.strip().split(",")[0].split()[-1].split(":")
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
    return 10.0


def make_black_video(video_id, duration):
    os.makedirs(BLACK_VIDEO_DIR, exist_ok=True)
    out_path = f"{BLACK_VIDEO_DIR}/{video_id}.mp4"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    cmd = [FFMPEG, "-y", "-f", "lavfi",
           "-i", f"color=c=black:size=224x224:rate=1:duration={duration:.1f}",
           "-c:v", "libx264", "-crf", "18", "-preset", "fast",
           "-loglevel", "error", out_path]
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

    with open(SHUFFLED_SAMPLE) as f:
        samples = json.load(f)

    model, processor = load_model()

    results = load_results()
    completed = {r["question_id"] for r in results}
    work = [s for s in samples if s["question_id"] not in completed]
    print(f"Remaining: {len(work)} / {len(samples)}")

    if not work:
        print("All done!")
    else:
        pbar = tqdm(work, desc="Shuffled black")
        for sample in pbar:
            orig_path = f"{VIDEO_BASE}/{sample['videoID']}.mp4"
            try:
                duration = get_video_duration(orig_path)
                black_path = make_black_video(sample["videoID"], duration).replace("\\", "/")
                pred = run_single(model, processor, black_path,
                                  sample["question"], sample["shuffled_options"])
                # Correctness: pred matches shuffled_answer (the relocated correct option)
                correct_shuffled = pred == sample["shuffled_answer"]
                # Also track: did pred happen to match the ORIGINAL letter position?
                correct_original_letter = pred == sample["answer"]
                entry = {
                    "question_id": sample["question_id"],
                    "videoID": sample["videoID"],
                    "task_type": sample["task_type"],
                    "condition": "shuffled_black",
                    "ground_truth": sample["shuffled_answer"],
                    "original_answer": sample["answer"],
                    "prediction": pred,
                    "correct": correct_shuffled,              # follows shuffled answer
                    "correct_original_letter": correct_original_letter,  # position bias
                    "shuffle_map": sample["shuffle_map"],
                }
            except Exception as e:
                traceback.print_exc()
                entry = {
                    "question_id": sample["question_id"],
                    "videoID": sample["videoID"],
                    "task_type": sample["task_type"],
                    "condition": "shuffled_black",
                    "ground_truth": sample["shuffled_answer"],
                    "original_answer": sample["answer"],
                    "prediction": None,
                    "correct": False,
                    "correct_original_letter": False,
                    "error": str(e),
                }
            results.append(entry)
            completed.add(sample["question_id"])
            save_results(results)
            acc = sum(1 for r in results if r.get("correct")) / len(results)
            pbar.set_postfix_str(f"acc={acc:.3f}")

    # Summary
    print("\n=== SHUFFLED BLACK SCREEN RESULTS ===")
    total = len(results)
    correct_shuffled  = sum(1 for r in results if r.get("correct"))
    correct_orig_lett = sum(1 for r in results if r.get("correct_original_letter"))
    print(f"Overall (shuffled answer): {correct_shuffled}/{total} = {correct_shuffled/total:.3f}")
    print(f"Original letter bias:      {correct_orig_lett}/{total} = {correct_orig_lett/total:.3f}")
    print(f"Chance baseline:           0.250")
    print()

    # Load original black for comparison
    orig_black_path = VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json"
    if os.path.exists(orig_black_path):
        with open(orig_black_path) as f:
            orig_black = json.load(f)
        orig_acc = sum(1 for r in orig_black if r.get("correct")) / len(orig_black)
        print(f"Original black screen acc: {orig_acc:.3f}")
        drop = correct_shuffled/total - orig_acc
        print(f"Drop from shuffling: {drop:+.3f}")
        print()
        if drop < -0.05:
            print("Interpretation: Significant drop -> model exploits option content/ordering")
        else:
            print("Interpretation: No significant drop -> prior comes from question text, not option content")

    from collections import defaultdict
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["task_type"]][1] += 1
        if r.get("correct"):
            by_type[r["task_type"]][0] += 1
    print("Per task type (shuffled):")
    for tt, (c, n) in sorted(by_type.items()):
        print(f"  {tt:<30} {c}/{n} = {c/n:.3f}")


if __name__ == "__main__":
    main()
