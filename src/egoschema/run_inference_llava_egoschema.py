"""
LLaVA-Video-7B-Qwen2 inference on EgoSchema: original and black screen.
All 2 conditions in one results file. Resume key: (question_id, condition).
Black condition: ffmpeg-generated black video of same duration, cached in BLACK_DIR.
5 options (A-E). Fully resumable.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os, copy, traceback, subprocess, re
import numpy as np
import torch
from tqdm import tqdm
import imageio_ffmpeg
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

SAMPLE_PATH  = VDG_DATA_ROOT + "/egoschema/egoschema_subset.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/egoschema/llava_egoschema_results.json"
VIDEOS_DIR   = VDG_DATA_ROOT + "/egoschema/videos"
BLACK_DIR    = VDG_DATA_ROOT + "/egoschema/black"
MODEL_ID     = "lmms-lab/LLaVA-Video-7B-Qwen2"
CONDITIONS   = ["original", "black"]
MAX_FRAMES   = 32

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(BLACK_DIR, exist_ok=True)
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()


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


def load_video(video_path, max_frames_num=MAX_FRAMES):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = vr.get_avg_fps()
    video_time = total_frame_num / avg_fps
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frame_time = [i / avg_fps for i in frame_idx]
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_time_str, video_time


def load_model():
    print("Loading LLaVA-Video-7B-Qwen2 ...")
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
        "Answer with only the letter (A, B, C, D, or E) of the correct option."
    )

    video, frame_time, video_time = load_video(video_path, MAX_FRAMES)
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

    tokenizer, model, image_processor = load_model()
    results = load_results()
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

    pbar = tqdm(work, desc="LLaVA-EgoSchema")
    for sample, condition in pbar:
        video_uid = sample["video_uid"]
        if condition == "black":
            orig_path = os.path.join(VIDEOS_DIR, f"{video_uid}.mp4")
            duration = get_video_duration(orig_path)
            video_path = make_black_video(video_uid, duration)
        else:
            video_path = os.path.join(VIDEOS_DIR, f"{video_uid}.mp4")
        video_path = os.path.abspath(video_path).replace("\\", "/")

        try:
            pred = run_single(tokenizer, model, image_processor, video_path,
                              sample["question"], sample["options"])
            entry = {
                "question_id": sample["question_id"],
                "video_uid":   video_uid,
                "task_type":   sample["task_type"],
                "condition":   condition,
                "ground_truth": sample["answer"],
                "prediction":  pred,
                "correct":     pred == sample["answer"],
                "error":       None,
            }
        except Exception as e:
            traceback.print_exc()
            entry = {
                "question_id": sample["question_id"],
                "video_uid":   video_uid,
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
        n_correct = sum(1 for r in results if r.get("correct"))
        n_valid   = sum(1 for r in results if not r.get("error"))
        pbar.set_postfix_str(f"acc={n_correct}/{n_valid}")

    print("\n=== LLAVA EgoSchema COMPLETE ===")
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
