"""
InternVL2-8B inference on EgoSchema: original and black screen.
All 2 conditions in one results file. Resume key: (question_id, condition).
Black condition: ffmpeg-generated black video of same duration, cached in BLACK_DIR.
16 frames per video, 448x448, bfloat16. 5 options (A-E). Fully resumable.
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
import re
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import imageio_ffmpeg

SAMPLE_PATH  = VDG_DATA_ROOT + "/egoschema/egoschema_subset.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/egoschema/internvl2_egoschema_results.json"
VIDEOS_DIR   = VDG_DATA_ROOT + "/egoschema/videos"
BLACK_DIR    = VDG_DATA_ROOT + "/egoschema/black"
MODEL_ID     = "OpenGVLab/InternVL2-8B"
CONDITIONS   = ["original", "black"]
NUM_FRAMES   = 16
INPUT_SIZE   = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(BLACK_DIR, exist_ok=True)
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_video_duration(video_path):
    result = subprocess.run([ffmpeg_exe, "-i", video_path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    if m:
        h, mn, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mn * 60 + sec
    return 180.0


def make_black_video(video_uid, duration):
    out = os.path.join(BLACK_DIR, f"{video_uid}.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    cmd = [ffmpeg_exe, "-y", "-f", "lavfi",
           "-i", f"color=c=black:s=320x240:d={duration:.2f}",
           "-c:v", "libx264", "-t", str(duration), out]
    subprocess.run(cmd, capture_output=True)
    return out


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
        "Answer with only the letter (A, B, C, D, or E) of the correct option."
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
        if char.upper() in "ABCDE":
            return char.upper()
    return response[:5]


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    model, tokenizer = load_model()
    results  = load_results()
    # Exclude error entries so missing videos can be retried after download
    completed = {(r["question_id"], r["condition"]) for r in results if not r.get("error")}
    print(f"Done (no-error): {len(completed)} | Total results: {len(results)} | Target: {len(samples) * len(CONDITIONS)}")

    work = []
    for s in samples:
        for cond in CONDITIONS:
            if (s["question_id"], cond) not in completed:
                if cond == "black" and (s["question_id"], "original") not in completed:
                    continue
                work.append((s, cond))
    print(f"Remaining: {len(work)}")

    pbar = tqdm(work, desc="InternVL2-EgoSchema")
    for sample, condition in pbar:
        video_uid = sample["video_uid"]
        if condition == "black":
            orig_path  = os.path.join(VIDEOS_DIR, f"{video_uid}.mp4")
            duration   = get_video_duration(orig_path)
            video_path = make_black_video(video_uid, duration)
        else:
            video_path = os.path.join(VIDEOS_DIR, f"{video_uid}.mp4")
        video_path = os.path.abspath(video_path).replace("\\", "/")

        try:
            pred = run_single(model, tokenizer, video_path,
                              sample["question"], sample["options"])
            entry = {
                "question_id":  sample["question_id"],
                "video_uid":    video_uid,
                "task_type":    sample["task_type"],
                "condition":    condition,
                "ground_truth": sample["answer"],
                "prediction":   pred,
                "correct":      pred == sample["answer"],
                "error":        None,
            }
        except Exception as e:
            traceback.print_exc()
            entry = {
                "question_id":  sample["question_id"],
                "video_uid":    video_uid,
                "task_type":    sample["task_type"],
                "condition":    condition,
                "ground_truth": sample["answer"],
                "prediction":   None,
                "correct":      False,
                "error":        str(e),
            }
        results.append(entry)
        completed.add((sample["question_id"], condition))
        save_results(results)
        n_correct = sum(1 for r in results if r.get("correct"))
        n_valid   = sum(1 for r in results if not r.get("error"))
        pbar.set_postfix_str(f"acc={n_correct}/{n_valid}")

    print("\n=== INTERNVL2 EgoSchema COMPLETE ===")
    from collections import defaultdict
    tt_accs = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r.get("error"):
            tt_accs[r["task_type"]][r["condition"]].append(r["correct"])
    for tt in sorted(tt_accs):
        accs = {c: sum(v) / len(v) for c, v in tt_accs[tt].items() if v}
        print(f"  {tt:<30} orig={accs.get('original', 0):.3f} "
              f"black={accs.get('black', 0):.3f}")


if __name__ == "__main__":
    main()
