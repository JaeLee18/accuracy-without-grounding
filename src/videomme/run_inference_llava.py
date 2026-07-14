"""
Full study inference: LLaVA-Video-7B-Qwen2 across 6 CRF conditions.
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
import copy
import traceback
import numpy as np
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/full_study/llava_results.json"
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
MODEL_ID = "lmms-lab/LLaVA-Video-7B-Qwen2"
MAX_FRAMES = 32


def load_video(video_path, max_frames_num=MAX_FRAMES, fps=1, force_sample=True):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = vr.get_avg_fps()
    frame_idx = list(range(0, total_frame_num, max(1, round(avg_fps / fps))))
    frame_time = [i / avg_fps for i in frame_idx]

    if len(frame_idx) > max_frames_num or force_sample:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / avg_fps for i in frame_idx]

    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_time_str, video_time


def sanity_checks():
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    missing = set()
    for s in samples:
        for cond in CONDITIONS:
            vpath = os.path.join(CONDITION_DIRS[cond], f"{s['videoID']}.mp4")
            if not os.path.exists(vpath):
                missing.add((cond, s["videoID"]))
    if missing:
        print(f"WARNING: {len(missing)} missing video files")
        sys.exit(1)

    print(f"All files verified. {len(samples)} questions x {len(CONDITIONS)} conditions.")
    return samples


def load_model():
    print("Loading LLaVA-Video-7B-Qwen2...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        MODEL_ID, None, "llava_qwen",
        torch_dtype="float16", device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print("Model loaded.")
    return tokenizer, model, image_processor


def run_single(tokenizer, model, image_processor, video_path, question, options):
    options_text = "\n".join(options)
    prompt_text = (
        f"{question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )

    video, frame_time, video_time = load_video(video_path, MAX_FRAMES, fps=1, force_sample=True)
    video_tensor = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
    video_tensor = [video_tensor]

    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {len(video)} frames "
        f"are uniformly sampled from it. These frames are located at {frame_time}. "
        "Please answer the following questions related to this video."
    )

    full_question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt_text}"
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], full_question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, images=video_tensor, modalities=["video"],
            do_sample=False, temperature=0, max_new_tokens=32,
        )

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

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
    tokenizer, model, image_processor = load_model()

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

    pbar = tqdm(work, desc="LLaVA-Video")
    for sample, condition in pbar:
        video_path = os.path.join(CONDITION_DIRS[condition], f"{sample['videoID']}.mp4")
        video_path = os.path.abspath(video_path)
        try:
            pred = run_single(tokenizer, model, image_processor, video_path, sample["question"], sample["options"])
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

        done_count = len([r for r in results if not r.get("error")])
        correct_count = len([r for r in results if r.get("correct")])
        pbar.set_postfix_str(f"acc={correct_count}/{done_count}")

    print("\n=== LLAVA-VIDEO INFERENCE COMPLETE ===")
    for tt in task_types:
        accs = []
        for cond in CONDITIONS:
            rel = [r for r in results if r["task_type"] == tt and r["condition"] == cond and not r.get("error")]
            acc = sum(1 for r in rel if r["correct"]) / len(rel) if rel else 0
            accs.append(f"{acc:.3f}")
        print(f"  {tt:<25} " + " | ".join(accs))

if __name__ == "__main__":
    main()
