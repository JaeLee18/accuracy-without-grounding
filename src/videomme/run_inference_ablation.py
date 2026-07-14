"""
Exp D (blur/framedrop) + Exp E (FPS=1.0 on TR) — single model load.
Reads exp_d_sample.json for Exp D, full TR questions from full_sample.json for Exp E.
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

SAMPLE_PATH    = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
EXP_D_SAMPLE   = VDG_DATA_ROOT + "/videomme_full/exp_d_sample.json"
RESULTS_PATH   = VDG_RESULTS_ROOT + "/full_study/qwen2vl_ablation_results.json"
VIDEO_BASE     = VDG_DATA_ROOT + "/videomme_full"
ABLATION_BASE  = VDG_DATA_ROOT + "/videomme_full/ablation"
MODEL_ID       = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS     = 256 * 256
MIN_PIXELS     = 28 * 28
DEFAULT_FPS    = 0.25


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


def run_single(model, processor, video_path, question, options, fps=DEFAULT_FPS):
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
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(EXP_D_SAMPLE) as f:
        exp_d_samples = json.load(f)
    with open(SAMPLE_PATH) as f:
        all_samples = json.load(f)

    tr_samples = [s for s in all_samples if s["task_type"] == "Temporal Reasoning"]

    # Define all conditions
    ablation_conditions = [
        # Exp D — blur (original video dir not used; ablation dir used)
        ("blur_s3",    f"{ABLATION_BASE}/blur_s3",    exp_d_samples, DEFAULT_FPS),
        ("blur_s6",    f"{ABLATION_BASE}/blur_s6",    exp_d_samples, DEFAULT_FPS),
        ("blur_s10",   f"{ABLATION_BASE}/blur_s10",   exp_d_samples, DEFAULT_FPS),
        # Exp D — framedrop
        ("framedrop2", f"{ABLATION_BASE}/framedrop2", exp_d_samples, DEFAULT_FPS),
        ("framedrop3", f"{ABLATION_BASE}/framedrop3", exp_d_samples, DEFAULT_FPS),
        ("framedrop4", f"{ABLATION_BASE}/framedrop4", exp_d_samples, DEFAULT_FPS),
        # Exp E — FPS=1.0 on TR, original videos
        ("fps1_tr",    f"{VIDEO_BASE}/videos",        tr_samples,    1.0),
    ]

    model, processor = load_model()

    results = load_results()
    completed = {(r["question_id"], r["condition"]) for r in results}
    print(f"Existing results: {len(results)}")

    for cond_name, video_dir, samples, fps in ablation_conditions:
        # Verify video directory exists
        if not os.path.isdir(video_dir):
            print(f"\nSKIPPING {cond_name}: directory not found: {video_dir}")
            continue

        work = [s for s in samples if (s["question_id"], cond_name) not in completed]
        if not work:
            print(f"\n{cond_name}: already complete ({len(samples)} questions)")
            continue

        print(f"\n=== {cond_name} ({len(work)} remaining / {len(samples)}) ===")
        pbar = tqdm(work, desc=cond_name)
        for sample in pbar:
            video_path = f"{video_dir}/{sample['videoID']}.mp4"
            video_path_fwd = video_path.replace("\\", "/")
            try:
                pred = run_single(model, processor, video_path_fwd,
                                  sample["question"], sample["options"], fps=fps)
                entry = {
                    "question_id": sample["question_id"],
                    "videoID": sample["videoID"],
                    "task_type": sample["task_type"],
                    "condition": cond_name,
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
                    "condition": cond_name,
                    "ground_truth": sample["answer"],
                    "prediction": None,
                    "correct": False,
                    "error": str(e),
                }
            results.append(entry)
            completed.add((sample["question_id"], cond_name))
            save_results(results)
            done = len([r for r in results if r["condition"] == cond_name])
            acc_vals = [r for r in results if r["condition"] == cond_name and not r.get("error")]
            acc = sum(1 for r in acc_vals if r["correct"]) / len(acc_vals) if acc_vals else 0
            pbar.set_postfix_str(f"acc={acc:.3f}")

    # Summary
    print("\n=== ABLATION SUMMARY ===")
    from collections import defaultdict
    by_cond = defaultdict(lambda: [0, 0])
    for r in results:
        by_cond[r["condition"]][1] += 1
        if r.get("correct"):
            by_cond[r["condition"]][0] += 1
    for cond, (c, n) in sorted(by_cond.items()):
        print(f"  {cond:<15} {c}/{n} = {c/n:.3f}")


if __name__ == "__main__":
    main()
