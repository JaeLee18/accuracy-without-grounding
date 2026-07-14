"""
Scale test: InternVL2-26B, BLACK SCREEN ONLY on Video-MME.
Tests whether larger LM = higher black-screen accuracy (stronger language priors).

For A100/H100 (>=40GB): runs in bf16 natively.
For 24GB GPUs: uses 4-bit quantization (set USE_4BIT=1).

Usage:
  export DATA_ROOT=/path/to/cluster/data
  export RESULTS_ROOT=/path/to/cluster/results
  export USE_4BIT=0   # set to 1 for 24GB GPUs
  python run_internvl2_26b_black.py
"""
import json, os, subprocess, re, traceback
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import imageio_ffmpeg

DATA_ROOT = os.environ.get("DATA_ROOT", "../data")
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "../results")
USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"

SAMPLE_PATH = os.path.join(DATA_ROOT, "full_sample.json")
RESULTS_PATH = os.path.join(RESULTS_ROOT, "internvl2_26b_black_results.json")
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
BLACK_DIR = os.path.join(DATA_ROOT, "black_26b")
MODEL_ID = "OpenGVLab/InternVL2-26B"
NUM_FRAMES = 16
INPUT_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

os.makedirs(RESULTS_ROOT, exist_ok=True)
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
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 30.0


def make_black_video(video_id, duration):
    out = os.path.join(BLACK_DIR, f"{video_id}.mp4")
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
    print(f"Loading {MODEL_ID} {'(4-bit)' if USE_4BIT else '(bf16)'} ...")
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            MODEL_ID, quantization_config=bnb_config,
            low_cpu_mem_usage=True, trust_remote_code=True,
        ).eval()
    else:
        model = AutoModel.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False,
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM: {vram:.1f} GB")
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
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    print(f"Dataset: {len(samples)} questions (black screen only)")

    model, tokenizer = load_model()
    results = load_results()
    completed = {r["question_id"] for r in results if not r.get("error")}
    print(f"Already done: {len(completed)} | Target: {len(samples)}")

    work = [s for s in samples if s["question_id"] not in completed]
    print(f"Remaining: {len(work)}")
    if not work:
        _print_summary(results)
        return

    pbar = tqdm(work, desc="IV2-26B-black")
    for sample in pbar:
        video_id = sample["videoID"]
        orig_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        duration = get_video_duration(orig_path)
        video_path = make_black_video(video_id, duration)

        try:
            pred = run_single(model, tokenizer, video_path, sample["question"], sample["options"])
            entry = {
                "question_id": sample["question_id"],
                "videoID": video_id,
                "task_type": sample["task_type"],
                "condition": "black",
                "ground_truth": sample["answer"],
                "prediction": pred,
                "correct": pred == sample["answer"],
                "error": None,
            }
        except Exception as e:
            traceback.print_exc()
            entry = {
                "question_id": sample["question_id"],
                "videoID": video_id,
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
        torch.cuda.empty_cache()

        n_ok = sum(1 for r in results if not r.get("error"))
        n_correct = sum(1 for r in results if r.get("correct"))
        pbar.set_postfix_str(f"acc={n_correct}/{n_ok}")

    print("\n=== INTERNVL2-26B BLACK SCREEN COMPLETE ===")
    _print_summary(results)


def _print_summary(results):
    from collections import defaultdict
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if not r.get("error"):
            stats[r["task_type"]]["total"] += 1
            stats[r["task_type"]]["correct"] += int(r.get("correct", False))

    print(f"\n{'Task Type':<25} {'Black Acc':>10} {'n':>5}")
    print("-" * 42)
    tc, tn = 0, 0
    for tt in sorted(stats.keys()):
        s = stats[tt]
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"{tt:<25} {acc:>10.3f} {s['total']:>5}")
        tc += s["correct"]; tn += s["total"]
    if tn:
        print(f"{'OVERALL':<25} {tc/tn:>10.3f} {tn:>5}")


if __name__ == "__main__":
    main()
