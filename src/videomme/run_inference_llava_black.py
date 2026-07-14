"""
Exp C: LLaVA-Video-7B-Qwen2 black screen baseline.
Mirrors run_inference_black.py but uses the LLaVA model and inference pipeline.
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
import numpy as np
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
from imageio_ffmpeg import get_ffmpeg_exe

SAMPLE_PATH     = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
RESULTS_PATH    = VDG_RESULTS_ROOT + "/full_study/llava_black_results.json"
VIDEO_BASE      = VDG_DATA_ROOT + "/videomme_full/videos"
MODEL_ID        = "lmms-lab/LLaVA-Video-7B-Qwen2"
MAX_FRAMES      = 32
BLACK_VIDEO_DIR = VDG_RESULTS_ROOT + "/full_study/black_videos"

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


def load_model():
    print("Loading LLaVA-Video-7B-Qwen2...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_ID, None, "llava_qwen", torch_dtype="float16", device_map="auto",
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
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    tokenizer, model, image_processor = load_model()

    results = load_results()
    completed = {r["question_id"] for r in results}
    work = [s for s in samples if s["question_id"] not in completed]
    print(f"Remaining: {len(work)} / {len(samples)}")

    if not work:
        print("All done!")
    else:
        pbar = tqdm(work, desc="LLaVA black screen")
        for sample in pbar:
            orig_path = f"{VIDEO_BASE}/{sample['videoID']}.mp4"
            try:
                duration = get_video_duration(orig_path)
                black_path = make_black_video(sample["videoID"], duration)
                pred = run_single(tokenizer, model, image_processor,
                                  black_path, sample["question"], sample["options"])
                entry = {
                    "question_id": sample["question_id"],
                    "videoID": sample["videoID"],
                    "task_type": sample["task_type"],
                    "condition": "black",
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
                    "condition": "black",
                    "ground_truth": sample["answer"],
                    "prediction": None,
                    "correct": False,
                    "error": str(e),
                }
            results.append(entry)
            completed.add(sample["question_id"])
            save_results(results)
            acc = sum(1 for r in results if r.get("correct")) / len(results)
            pbar.set_postfix_str(f"acc={acc:.3f}")

    # Summary with comparison
    print("\n=== LLAVA BLACK SCREEN BASELINE ===")
    from collections import defaultdict
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["task_type"]][1] += 1
        if r.get("correct"):
            by_type[r["task_type"]][0] += 1

    overall_c = sum(1 for r in results if r.get("correct"))
    print(f"Overall: {overall_c}/{len(results)} = {overall_c/len(results):.3f}  (chance=0.250)")
    for tt, (c, n) in sorted(by_type.items()):
        print(f"  {tt:<30} {c}/{n} = {c/n:.3f}")


if __name__ == "__main__":
    main()
