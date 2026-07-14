"""
Shuffled-frames and single-frame ablation for Qwen2-VL-7B-Instruct on Video-MME.
Instead of creating new video files, we modify frames in-memory before passing to the model.
Runs both conditions sequentially with one model load.
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
import traceback
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
VIDEO_BASE = VDG_DATA_ROOT + "/videomme_full/videos"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 256 * 256
MIN_PIXELS = 28 * 28
FPS = 0.25
SEED = 42

RESULTS = {
    "shuffled":    VDG_RESULTS_ROOT + "/full_study/qwen2vl_shuffled_results.json",
    "singleframe": VDG_RESULTS_ROOT + "/full_study/qwen2vl_singleframe_results.json",
}


def load_model():
    print(f"Loading {MODEL_ID}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return model, processor


def run_single(model, processor, video_path, question, options):
    """Standard inference with a video file path."""
    options_text = "\n".join(options)
    prompt = (f"{question}\n\n{options_text}\n\n"
              "Answer with only the letter (A, B, C, or D) of the correct option.")
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path,
         "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "fps": FPS},
        {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # Apply frame manipulation based on condition
    return _generate(model, processor, text, image_inputs, video_inputs)


def run_with_frames(model, processor, frames, question, options):
    """Inference with pre-loaded PIL frames (for shuffled/single-frame conditions)."""
    options_text = "\n".join(options)
    prompt = (f"{question}\n\n{options_text}\n\n"
              "Answer with only the letter (A, B, C, or D) of the correct option.")
    # Build message with image list instead of video path
    content = []
    for frame in frames:
        content.append({"type": "image", "image": frame,
                        "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return _generate(model, processor, text, image_inputs, video_inputs)


def _generate(model, processor, text, image_inputs, video_inputs):
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    for char in response:
        if char.upper() in "ABCD":
            return char.upper()
    return response[:5]


def extract_frames_from_video(video_path, fps=FPS):
    """Extract frames from video at given FPS using the same logic Qwen would use."""
    from decord import VideoReader, cpu
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    except Exception:
        return []
    total = len(vr)
    avg_fps = vr.get_avg_fps()
    if avg_fps <= 0:
        avg_fps = 30.0
    # Sample at FPS rate
    step = max(1, round(avg_fps / fps))
    indices = list(range(0, total, step))
    if not indices:
        indices = [0]
    frames_np = vr.get_batch(indices).asnumpy()
    pil_frames = [Image.fromarray(f) for f in frames_np]
    return pil_frames


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_condition(model, processor, samples, condition):
    results_path = RESULTS[condition]
    results = load_results(results_path)
    completed = {r["question_id"] for r in results}
    work = [s for s in samples if s["question_id"] not in completed]
    print(f"\n=== Qwen {condition.upper()} === Remaining: {len(work)} / {len(samples)}")

    if not work:
        print("All done!")
        return results

    rng = random.Random(SEED)
    pbar = tqdm(work, desc=f"Qwen {condition}")
    for sample in pbar:
        video_path = os.path.join(VIDEO_BASE, f"{sample['videoID']}.mp4")
        try:
            frames = extract_frames_from_video(video_path, FPS)
            if not frames:
                raise RuntimeError(f"No frames from {video_path}")

            if condition == "shuffled":
                # Deterministic shuffle per video
                frame_seed = hash(sample["videoID"]) + SEED
                local_rng = random.Random(frame_seed)
                local_rng.shuffle(frames)
            elif condition == "singleframe":
                # Use middle frame, repeated
                mid = len(frames) // 2
                frames = [frames[mid]] * len(frames)

            pred = run_with_frames(model, processor, frames,
                                   sample["question"], sample["options"])
            entry = {"question_id": sample["question_id"], "videoID": sample["videoID"],
                     "task_type": sample["task_type"], "condition": condition,
                     "ground_truth": sample["answer"], "prediction": pred,
                     "correct": pred == sample["answer"]}
        except Exception as e:
            traceback.print_exc()
            entry = {"question_id": sample["question_id"], "videoID": sample["videoID"],
                     "task_type": sample["task_type"], "condition": condition,
                     "ground_truth": sample["answer"], "prediction": None,
                     "correct": False, "error": str(e)}
        results.append(entry)
        completed.add(sample["question_id"])
        save_results(results, results_path)
        acc = sum(1 for r in results if r.get("correct")) / len(results)
        pbar.set_postfix_str(f"acc={acc:.3f}")

    overall_c = sum(1 for r in results if r.get("correct"))
    print(f"Overall: {overall_c}/{len(results)} = {overall_c/len(results):.3f}")
    return results


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    print(f"Questions: {len(samples)}")

    model, processor = load_model()

    # Run both conditions with one model load
    for condition in ["shuffled", "singleframe"]:
        run_condition(model, processor, samples, condition)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: 4-POINT DIAGNOSTIC LADDER (Qwen2-VL)")
    print("=" * 70)
    try:
        black = load_results(VDG_RESULTS_ROOT + "/full_study/qwen2vl_black_results.json")
        with open(VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json") as f:
            orig_all = json.load(f)
        orig = [r for r in orig_all if r["condition"] == "original"]
        shuf = load_results(RESULTS["shuffled"])
        sf = load_results(RESULTS["singleframe"])

        for label, data in [("Black", black), ("Single-frame", sf),
                            ("Shuffled", shuf), ("Original", orig)]:
            if data:
                acc = sum(1 for r in data if r.get("correct")) / len(data)
                print(f"  {label:15s}: {acc:.3f} ({len(data)} questions)")
    except Exception as e:
        print(f"  Error loading comparison data: {e}")


if __name__ == "__main__":
    main()
