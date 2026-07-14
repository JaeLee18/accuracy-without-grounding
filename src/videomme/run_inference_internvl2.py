"""
Full study inference: InternVL2-8B across 6 CRF conditions.
Saves results incrementally. Fully resumable.
16 frames per video, 448x448, bfloat16, flash attention.
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
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

SAMPLE_PATH  = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/full_study/internvl2_results.json"
VIDEO_BASE   = VDG_DATA_ROOT + "/videomme_full"
CONDITIONS   = ["original", "crf18", "crf23", "crf28", "crf33", "crf38"]
CONDITION_DIRS = {
    "original": os.path.join(VIDEO_BASE, "videos"),
    "crf18":    os.path.join(VIDEO_BASE, "crf18"),
    "crf23":    os.path.join(VIDEO_BASE, "crf23"),
    "crf28":    os.path.join(VIDEO_BASE, "crf28"),
    "crf33":    os.path.join(VIDEO_BASE, "crf33"),
    "crf38":    os.path.join(VIDEO_BASE, "crf38"),
}
MODEL_ID   = "OpenGVLab/InternVL2-8B"
NUM_FRAMES = 16
INPUT_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_video_frames(video_path, num_frames=NUM_FRAMES):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = [_transform(Image.fromarray(vr[i].asnumpy()).convert("RGB")) for i in indices]
    return torch.stack(frames)  # (N, 3, 448, 448)


def sanity_checks():
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    missing = []
    for s in samples:
        for cond in CONDITIONS:
            vpath = os.path.join(CONDITION_DIRS[cond], f"{s['videoID']}.mp4")
            if not os.path.exists(vpath):
                missing.append((cond, s["videoID"]))
    if missing:
        print(f"WARNING: {len(missing)} missing video files")
        for cond, vid in missing[:10]:
            print(f"  {cond}/{vid}.mp4")
        sys.exit(1)

    print(f"All files verified. {len(samples)} questions x {len(CONDITIONS)} conditions = {len(samples)*len(CONDITIONS)} total.")
    return samples


def load_model():
    print("Loading InternVL2-8B...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False,
    )
    print("Model loaded.")
    return model, tokenizer


def run_single(model, tokenizer, video_path, question, options):
    options_text = "\n".join(options)
    prompt_text = (
        f"{question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )

    pixel_values = load_video_frames(video_path, NUM_FRAMES).to(torch.bfloat16).cuda()
    num_patches_list = [1] * NUM_FRAMES

    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(NUM_FRAMES)])
    full_question = video_prefix + prompt_text

    generation_config = dict(do_sample=False, max_new_tokens=32)
    response = model.chat(
        tokenizer, pixel_values, full_question, generation_config,
        num_patches_list=num_patches_list,
    )

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
    model, tokenizer = load_model()

    results = load_results()
    completed = {(r["question_id"], r["condition"]) for r in results}
    print(f"Existing results: {len(results)} ({len(completed)} unique)")

    work = [
        (s, cond) for s in samples for cond in CONDITIONS
        if (s["question_id"], cond) not in completed
    ]
    print(f"Remaining: {len(work)}")
    if not work:
        print("All done!")
        return

    task_types = sorted(set(s["task_type"] for s in samples))

    pbar = tqdm(work, desc="InternVL2")
    for sample, condition in pbar:
        video_path = os.path.abspath(
            os.path.join(CONDITION_DIRS[condition], f"{sample['videoID']}.mp4")
        )
        try:
            pred = run_single(model, tokenizer, video_path, sample["question"], sample["options"])
            entry = {
                "question_id": sample["question_id"],
                "video_id":    sample["video_id"],
                "videoID":     sample["videoID"],
                "task_type":   sample["task_type"],
                "condition":   condition,
                "ground_truth": sample["answer"],
                "prediction":  pred,
                "correct":     pred == sample["answer"],
            }
        except Exception as e:
            traceback.print_exc()
            entry = {
                "question_id": sample["question_id"],
                "video_id":    sample["video_id"],
                "videoID":     sample["videoID"],
                "task_type":   sample["task_type"],
                "condition":   condition,
                "ground_truth": sample["answer"],
                "prediction":  None,
                "correct":     False,
                "error":       str(e),
            }

        results.append(entry)
        completed.add((sample["question_id"], condition))
        save_results(results)

        done_count    = sum(1 for r in results if not r.get("error"))
        correct_count = sum(1 for r in results if r.get("correct"))
        pbar.set_postfix_str(f"acc={correct_count}/{done_count}")

    print("\n=== INTERNVL2 INFERENCE COMPLETE ===")
    for tt in task_types:
        accs = []
        for cond in CONDITIONS:
            rel = [r for r in results if r["task_type"] == tt and r["condition"] == cond and not r.get("error")]
            acc = sum(1 for r in rel if r["correct"]) / len(rel) if rel else 0
            accs.append(f"{acc:.3f}")
        print(f"  {tt:<25} " + " | ".join(accs))


if __name__ == "__main__":
    main()
