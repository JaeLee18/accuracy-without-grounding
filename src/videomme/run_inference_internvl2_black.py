"""
InternVL2-8B black screen baseline on Video-MME.
Reuses black videos from results/full_study/black_videos/ (created by llava_black run).
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
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import imageio_ffmpeg

SAMPLE_PATH     = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
RESULTS_PATH    = VDG_RESULTS_ROOT + "/full_study/internvl2_black_results.json"
VIDEO_BASE      = VDG_DATA_ROOT + "/videomme_full/videos"
BLACK_VIDEO_DIR = VDG_RESULTS_ROOT + "/full_study/black_videos"
MODEL_ID        = "OpenGVLab/InternVL2-8B"
NUM_FRAMES      = 16
INPUT_SIZE      = 448
IMAGENET_MEAN   = (0.485, 0.456, 0.406)
IMAGENET_STD    = (0.229, 0.224, 0.225)

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_video_duration(video_path):
    result = subprocess.run([ffmpeg_exe, "-i", video_path], capture_output=True, text=True)
    import re
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    if m:
        h, mn, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mn * 60 + sec
    return 10.0


def make_black_video(video_id, duration):
    os.makedirs(BLACK_VIDEO_DIR, exist_ok=True)
    out_path = os.path.join(BLACK_VIDEO_DIR, f"{video_id}.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    cmd = [ffmpeg_exe, "-y", "-f", "lavfi",
           "-i", f"color=c=black:size=224x224:rate=1:duration={duration:.1f}",
           "-c:v", "libx264", "-crf", "18", "-preset", "fast",
           "-loglevel", "error", out_path]
    subprocess.run(cmd, check=True)
    return out_path


def load_video_frames(video_path, num_frames=NUM_FRAMES):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = [_transform(Image.fromarray(vr[i].asnumpy()).convert("RGB")) for i in indices]
    return torch.stack(frames)


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
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    model, tokenizer = load_model()

    results = load_results()
    completed = {r["question_id"] for r in results}
    work = [s for s in samples if s["question_id"] not in completed]
    print(f"Remaining: {len(work)} / {len(samples)}")

    if not work:
        print("All done!")
    else:
        pbar = tqdm(work, desc="InternVL2 black")
        for sample in pbar:
            orig_path = os.path.join(VIDEO_BASE, f"{sample['videoID']}.mp4")
            try:
                duration   = get_video_duration(orig_path)
                black_path = make_black_video(sample["videoID"], duration)
                pred = run_single(model, tokenizer, black_path,
                                  sample["question"], sample["options"])
                entry = {
                    "question_id":  sample["question_id"],
                    "videoID":      sample["videoID"],
                    "task_type":    sample["task_type"],
                    "condition":    "black",
                    "ground_truth": sample["answer"],
                    "prediction":   pred,
                    "correct":      pred == sample["answer"],
                }
            except Exception as e:
                traceback.print_exc()
                entry = {
                    "question_id":  sample["question_id"],
                    "videoID":      sample["videoID"],
                    "task_type":    sample["task_type"],
                    "condition":    "black",
                    "ground_truth": sample["answer"],
                    "prediction":   None,
                    "correct":      False,
                    "error":        str(e),
                }
            results.append(entry)
            completed.add(sample["question_id"])
            save_results(results)
            acc = sum(1 for r in results if r.get("correct")) / len(results)
            pbar.set_postfix_str(f"acc={acc:.3f}")

    print("\n=== INTERNVL2 BLACK SCREEN BASELINE ===")
    from collections import defaultdict
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["task_type"]][1] += 1
        if r.get("correct"):
            by_type[r["task_type"]][0] += 1
    overall_c = sum(1 for r in results if r.get("correct"))
    print(f"Overall: {overall_c}/{len(results)} = {overall_c/len(results):.3f}")
    for tt, (c, n) in sorted(by_type.items()):
        print(f"  {tt:<30} {c}/{n} = {c/n:.3f}")


if __name__ == "__main__":
    main()
