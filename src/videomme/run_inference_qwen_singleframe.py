"""
Single-frame ablation for Qwen2-VL-7B-Instruct on Video-MME.
Uses pre-generated single-frame videos from prepare_ablation_videos.py.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os, traceback, torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SAMPLE_PATH  = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
RESULTS_PATH = VDG_RESULTS_ROOT + "/full_study/qwen2vl_singleframe_results.json"
ABLATION_DIR = VDG_RESULTS_ROOT + "/full_study/singleframe_videos"
CONDITION    = "singleframe"
MODEL_ID     = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS   = 256 * 256
MIN_PIXELS   = 28 * 28
FPS          = 0.25

def load_model():
    print(f"Loading {MODEL_ID}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return model, processor

def run_single(model, processor, video_path, question, options):
    options_text = "\n".join(options)
    prompt = (f"{question}\n\n{options_text}\n\n"
              "Answer with only the letter (A, B, C, or D) of the correct option.")
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path,
         "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "fps": FPS},
        {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    for char in response:
        if char.upper() in "ABCD":
            return char.upper()
    return response[:5]

def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    model, processor = load_model()
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
    pbar = tqdm(work, desc=f"Qwen {CONDITION}")
    for sample in pbar:
        video_path = os.path.join(ABLATION_DIR, f"{sample['videoID']}.mp4").replace("\\", "/")
        try:
            pred = run_single(model, processor, video_path,
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
    print(f"\n=== QWEN {CONDITION.upper()} ===")
    print(f"Overall: {overall_c}/{len(results)} = {overall_c/len(results):.3f}")

if __name__ == "__main__":
    main()
