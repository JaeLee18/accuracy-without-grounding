"""
Full study inference: Qwen2-VL-7B-Instruct across 6 CRF conditions.
Saves results incrementally. Fully resumable.
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

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/full_study/qwen2vl_results.json"
VIDEO_BASE = VDG_DATA_ROOT + "/videomme_full"
CONDITIONS = ["original", "crf18", "crf23", "crf28", "crf33", "crf38"]
CONDITION_DIRS = {
    "original": os.path.join(VIDEO_BASE, "videos"),
    "crf18": os.path.join(VIDEO_BASE, "crf18"),
    "crf23": os.path.join(VIDEO_BASE, "crf23"),
    "crf28": os.path.join(VIDEO_BASE, "crf28"),
    "crf33": os.path.join(VIDEO_BASE, "crf33"),
    "crf38": os.path.join(VIDEO_BASE, "crf38"),
}
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 256 * 256
MIN_PIXELS = 28 * 28
FPS = 0.25


def sanity_checks():
    assert torch.cuda.is_available(), "CUDA not available!"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    missing = []
    for s in samples:
        for cond in CONDITIONS:
            vpath = os.path.join(CONDITION_DIRS[cond], f"{s['videoID']}.mp4")
            if not os.path.exists(vpath):
                missing.append((cond, s["videoID"]))
    if missing:
        unique_missing = set(missing)
        print(f"WARNING: {len(unique_missing)} missing video files")
        for cond, vid in list(unique_missing)[:10]:
            print(f"  {cond}/{vid}.mp4")
        sys.exit(1)

    print(f"All files verified. {len(samples)} questions x {len(CONDITIONS)} conditions = {len(samples)*len(CONDITIONS)} total.")
    return samples


def make_video_path(video_dir, video_id):
    path = os.path.join(video_dir, f"{video_id}.mp4")
    return os.path.abspath(path).replace("\\", "/")


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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path,
                 "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "fps": FPS},
                {"type": "text", "text": prompt},
            ],
        }
    ]

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
    samples = sanity_checks()
    model, processor = load_model()

    results = load_results()
    completed = {(r["question_id"], r["condition"]) for r in results}
    print(f"Existing results: {len(results)} ({len(completed)} unique)")

    work = []
    for s in samples:
        for cond in CONDITIONS:
            if (s["question_id"], cond) not in completed:
                work.append((s, cond))

    print(f"Remaining: {len(work)}")
    if not work:
        print("All done!")
        return

    task_types = sorted(set(s["task_type"] for s in samples))

    pbar = tqdm(work, desc="Qwen2-VL")
    for sample, condition in pbar:
        video_path = make_video_path(CONDITION_DIRS[condition], sample["videoID"])
        try:
            pred = run_single(model, processor, video_path, sample["question"], sample["options"])
            gt = sample["answer"]
            entry = {
                "question_id": sample["question_id"],
                "video_id": sample["video_id"],
                "videoID": sample["videoID"],
                "task_type": sample["task_type"],
                "condition": condition,
                "ground_truth": gt,
                "prediction": pred,
                "correct": pred == gt,
            }
        except Exception as e:
            traceback.print_exc()
            entry = {
                "question_id": sample["question_id"],
                "video_id": sample["video_id"],
                "videoID": sample["videoID"],
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

        # Live accuracy
        done_count = len([r for r in results if not r.get("error")])
        correct_count = len([r for r in results if r.get("correct")])
        pbar.set_postfix_str(f"acc={correct_count}/{done_count}")

    print("\n=== QWEN2-VL INFERENCE COMPLETE ===")
    for tt in task_types:
        accs = []
        for cond in CONDITIONS:
            rel = [r for r in results if r["task_type"] == tt and r["condition"] == cond and not r.get("error")]
            acc = sum(1 for r in rel if r["correct"]) / len(rel) if rel else 0
            accs.append(f"{acc:.3f}")
        print(f"  {tt:<25} " + " | ".join(accs))

if __name__ == "__main__":
    main()
