"""
Step 4: Run Qwen2-VL-7B-Instruct inference on pilot videos across 3 compression conditions.
Saves results incrementally to results/pilot_results.json. Fully resumable.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
import os
import sys
import traceback
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme/pilot_sample.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/pilot_results.json"
VIDEO_DIRS = {
    "original": VDG_DATA_ROOT + "/videomme/videos",
    "crf18": VDG_DATA_ROOT + "/videomme/crf18",
    "crf38": VDG_DATA_ROOT + "/videomme/crf38",
}
CONDITIONS = ["original", "crf18", "crf38"]
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 256 * 256
MIN_PIXELS = 28 * 28
FPS = 0.5


def sanity_checks():
    """Verify CUDA and all video files exist before starting."""
    assert torch.cuda.is_available(), "CUDA not available!"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    assert "4090" in gpu_name, f"Expected RTX 4090, got {gpu_name}"

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    missing = []
    for s in samples:
        for cond in CONDITIONS:
            vpath = os.path.join(VIDEO_DIRS[cond], f"{s['videoID']}.mp4")
            if not os.path.exists(vpath):
                missing.append((cond, s["videoID"]))
    if missing:
        print(f"WARNING: {len(missing)} video files missing:")
        for cond, vid in missing[:10]:
            print(f"  {cond}/{vid}.mp4")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")
        sys.exit(1)

    print(f"All video files verified. {len(samples)} questions, {len(CONDITIONS)} conditions.")
    return samples


def make_video_path(video_dir, video_id):
    """Create a plain absolute path with forward slashes for Qwen2-VL + decord."""
    path = os.path.join(video_dir, f"{video_id}.mp4")
    path = os.path.abspath(path).replace("\\", "/")
    return path


def load_model():
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    print("Model loaded.")
    return model, processor


def run_single(model, processor, video_uri, question, options):
    """Run inference for a single question. Returns the model's answer letter."""
    options_text = "\n".join(options)
    prompt = (
        f"{question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_uri,
                    "max_pixels": MAX_PIXELS,
                    "min_pixels": MIN_PIXELS,
                    "fps": FPS,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    # Trim input tokens
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Extract answer letter
    for char in response:
        if char.upper() in "ABCD":
            return char.upper()
    return response[:5]  # fallback: return first few chars


def load_existing_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def get_completed_keys(results):
    """Return set of (question_id, condition) that are already done."""
    return {(r["question_id"], r["condition"]) for r in results}


def compute_accuracy(results, task_type, condition):
    relevant = [r for r in results if r["task_type"] == task_type and r["condition"] == condition]
    if not relevant:
        return 0.0
    correct = sum(1 for r in relevant if r["correct"])
    return correct / len(relevant)


def main():
    samples = sanity_checks()
    model, processor = load_model()

    results = load_existing_results()
    completed = get_completed_keys(results)
    print(f"Existing results: {len(results)} ({len(completed)} unique)")

    # Build work items
    work = []
    for s in samples:
        for cond in CONDITIONS:
            key = (s["question_id"], cond)
            if key not in completed:
                work.append((s, cond))

    print(f"Remaining work items: {len(work)}")
    if not work:
        print("All done!")
        print_summary(results)
        return

    task_types = sorted(set(s["task_type"] for s in samples))

    pbar = tqdm(work, desc="Inference")
    for sample, condition in pbar:
        video_path = make_video_path(VIDEO_DIRS[condition], sample["videoID"])
        try:
            pred = run_single(model, processor, video_path, sample["question"], sample["options"])
            gt = sample["answer"]
            is_correct = (pred == gt)

            result_entry = {
                "question_id": sample["question_id"],
                "video_id": sample["video_id"],
                "videoID": sample["videoID"],
                "task_type": sample["task_type"],
                "condition": condition,
                "question": sample["question"],
                "ground_truth": gt,
                "prediction": pred,
                "correct": is_correct,
            }
        except Exception as e:
            traceback.print_exc()
            result_entry = {
                "question_id": sample["question_id"],
                "video_id": sample["video_id"],
                "videoID": sample["videoID"],
                "task_type": sample["task_type"],
                "condition": condition,
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "prediction": None,
                "correct": False,
                "error": str(e),
            }

        results.append(result_entry)
        completed.add((sample["question_id"], condition))
        save_results(results)

        # Update progress bar with live accuracy
        acc_parts = []
        for tt in task_types:
            for cond in CONDITIONS:
                acc = compute_accuracy(results, tt, cond)
                short_tt = "OCR" if "OCR" in tt else "AR"
                acc_parts.append(f"{short_tt}/{cond[:3]}={acc:.0%}")
        pbar.set_postfix_str(" ".join(acc_parts))

    print("\n=== INFERENCE COMPLETE ===")
    print_summary(results)


def print_summary(results):
    task_types = sorted(set(r["task_type"] for r in results))
    print(f"\n{'task_type':<25} | {'original':>8} | {'crf18':>8} | {'crf38':>8}")
    print("-" * 60)
    for tt in task_types:
        accs = []
        for cond in CONDITIONS:
            acc = compute_accuracy(results, tt, cond)
            accs.append(f"{acc:.2f}")
        short_name = tt if len(tt) <= 20 else tt[:20]
        print(f"{short_name:<25} | {accs[0]:>8} | {accs[1]:>8} | {accs[2]:>8}")


if __name__ == "__main__":
    main()
