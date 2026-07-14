"""
Shuffled-frames and single-frame ablation for LLaVA-Video-7B-Qwen2 on Video-MME.
Modifies frames in-memory before passing to model. Runs both conditions sequentially.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os, random, traceback, copy
import numpy as np
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
VIDEO_BASE  = VDG_DATA_ROOT + "/videomme_full/videos"
MODEL_ID    = "lmms-lab/LLaVA-Video-7B-Qwen2"
MAX_FRAMES  = 32
SEED = 42

RESULTS = {
    "shuffled":    VDG_RESULTS_ROOT + "/full_study/llava_shuffled_results.json",
    "singleframe": VDG_RESULTS_ROOT + "/full_study/llava_singleframe_results.json",
}


def load_video(video_path, max_frames_num=MAX_FRAMES, fps=1, force_sample=True):
    """Load video frames as numpy array. Returns (frames, frame_time_str, video_time)."""
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


def load_model():
    print(f"Loading {MODEL_ID}...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_ID, None, "llava_qwen", torch_dtype="float16", device_map="auto",
        attn_implementation="sdpa")
    model.eval()
    return tokenizer, model, image_processor


def run_inference(tokenizer, model, image_processor, frames_np, frame_time_str, video_time, question, options):
    """Run LLaVA inference with a numpy frame array."""
    options_text = "\n".join(options)
    prompt_text = (f"{question}\n\n{options_text}\n\n"
                   "Answer with only the letter (A, B, C, or D) of the correct option.")
    video_tensor = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"].cuda().half()
    video_tensor = [video_tensor]
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {len(frames_np)} frames "
        f"are uniformly sampled from it. These frames are located at {frame_time_str}. "
        "Please answer the following questions related to this video.")
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
            do_sample=False, temperature=0, max_new_tokens=32)
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    for char in response:
        if char.upper() in "ABCD":
            return char.upper()
    return response[:5]


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_condition(tokenizer, model, image_processor, samples, condition):
    results_path = RESULTS[condition]
    results = load_results(results_path)
    completed = {r["question_id"] for r in results}
    work = [s for s in samples if s["question_id"] not in completed]
    print(f"\n=== LLaVA {condition.upper()} === Remaining: {len(work)} / {len(samples)}")

    if not work:
        print("All done!")
        return results

    pbar = tqdm(work, desc=f"LLaVA {condition}")
    for sample in pbar:
        video_path = os.path.join(VIDEO_BASE, f"{sample['videoID']}.mp4")
        try:
            frames, frame_time_str, video_time = load_video(video_path)

            if condition == "shuffled":
                frame_seed = hash(sample["videoID"]) + SEED
                rng = random.Random(frame_seed)
                indices = list(range(len(frames)))
                rng.shuffle(indices)
                frames = frames[indices]
            elif condition == "singleframe":
                mid = len(frames) // 2
                frames = np.stack([frames[mid]] * len(frames))

            pred = run_inference(tokenizer, model, image_processor,
                                 frames, frame_time_str, video_time,
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
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    tokenizer, model, image_processor = load_model()
    for condition in ["shuffled", "singleframe"]:
        run_condition(tokenizer, model, image_processor, samples, condition)


if __name__ == "__main__":
    main()
