"""
Single-frame ablation for InternVL2-8B on Video-MME.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os, traceback
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

SAMPLE_PATH  = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/full_study/internvl2_singleframe_results.json"
ABLATION_DIR = VDG_RESULTS_ROOT + "/full_study/singleframe_videos"
CONDITION    = "singleframe"
MODEL_ID     = "OpenGVLab/InternVL2-8B"
NUM_FRAMES   = 16
INPUT_SIZE   = 448
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
    return torch.stack(frames)

def load_model():
    print(f"Loading {MODEL_ID}...")
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def run_single(model, tokenizer, video_path, question, options):
    options_text = "\n".join(options)
    prompt_text = (f"{question}\n\n{options_text}\n\n"
                   "Answer with only the letter (A, B, C, or D) of the correct option.")
    pixel_values = load_video_frames(video_path, NUM_FRAMES).to(torch.bfloat16).cuda()
    num_patches_list = [1] * NUM_FRAMES
    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(NUM_FRAMES)])
    full_question = video_prefix + prompt_text
    generation_config = dict(do_sample=False, max_new_tokens=32)
    response = model.chat(tokenizer, pixel_values, full_question, generation_config,
                          num_patches_list=num_patches_list)
    for char in response:
        if char.upper() in "ABCD":
            return char.upper()
    return response[:5]

def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    model, tokenizer = load_model()
    results = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    completed = {r["question_id"] for r in results}
    work = [s for s in samples if s["question_id"] not in completed]
    print(f"Remaining: {len(work)} / {len(samples)}")
    if not work:
        print("All done!")
        return
    pbar = tqdm(work, desc=f"InternVL2 {CONDITION}")
    for sample in pbar:
        video_path = os.path.join(ABLATION_DIR, f"{sample['videoID']}.mp4")
        try:
            pred = run_single(model, tokenizer, video_path,
                              sample["question"], sample["options"])
            entry = {"question_id": sample["question_id"], "videoID": sample["videoID"],
                     "task_type": sample["task_type"], "condition": CONDITION,
                     "ground_truth": sample["answer"], "prediction": pred,
                     "correct": pred == sample["answer"]}
        except Exception as e:
            traceback.print_exc()
            entry = {"question_id": sample["question_id"], "videoID": sample["videoID"],
                     "task_type": sample["task_type"], "condition": CONDITION,
                     "ground_truth": sample["answer"], "prediction": None,
                     "correct": False, "error": str(e)}
        results.append(entry)
        completed.add(sample["question_id"])
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        acc = sum(1 for r in results if r.get("correct")) / len(results)
        pbar.set_postfix_str(f"acc={acc:.3f}")
    overall_c = sum(1 for r in results if r.get("correct"))
    print(f"\n=== INTERNVL2 {CONDITION.upper()} ===")
    print(f"Overall: {overall_c}/{len(results)} = {overall_c/len(results):.3f}")

if __name__ == "__main__":
    main()
