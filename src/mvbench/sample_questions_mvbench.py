"""
Sample 50 questions per MVBench task type (seed=42).
Converts full-text answers to letter format (A/B/C/D).
Outputs mvbench_sample.json with same schema as full_sample.json.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json, random, os

RAW_DIR = VDG_DATA_ROOT + "/mvbench/raw"
OUT_PATH = VDG_DATA_ROOT + "/mvbench/mvbench_sample.json"
SEED = 42
N_PER_TASK = 50
MAX_SKIPS_PER_TASK = 2

TASK_TYPES = [
    "action_antonym", "action_count", "action_localization", "action_prediction",
    "action_sequence", "character_order", "counterfactual_inference",
    "egocentric_navigation", "episodic_reasoning", "fine_grained_action",
    "moving_attribute", "moving_count", "moving_direction", "object_existence",
    "object_interaction", "object_shuffle", "scene_transition", "state_change",
    "unexpected_action",
]

TASK_ZIP = {
    "action_antonym": "ssv2_video.zip",
    "action_count": "sta.zip",
    "action_localization": "star.zip",
    "action_prediction": "star.zip",
    "action_sequence": "data0613.zip",
    "character_order": "tvqa.zip",
    "counterfactual_inference": "clevrer.zip",
    "egocentric_navigation": "vlnqa.zip",
    "episodic_reasoning": "tvqa.zip",
    "fine_grained_action": "data0613.zip",
    "moving_attribute": "data0613.zip",
    "moving_count": "data0613.zip",
    "moving_direction": "data0613.zip",
    "object_existence": "clevrer.zip",
    "object_interaction": "clevrer.zip",
    "object_shuffle": "data0613.zip",
    "scene_transition": "scene_qa.zip",
    "state_change": "perception.zip",
    "unexpected_action": "FunQA_test.zip",
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
skips_log = []
all_samples = []

for task in TASK_TYPES:
    rng = random.Random(SEED)
    with open(os.path.join(RAW_DIR, f"{task}.json")) as f:
        data = json.load(f)

    rng.shuffle(data)
    collected = []
    skips = 0

    for entry in data:
        if len(collected) >= N_PER_TASK:
            break
        raw_answer = entry["answer"]
        candidates = entry["candidates"]
        if raw_answer not in candidates:
            skips += 1
            skips_log.append({"task": task, "video": entry["video"], "answer": raw_answer})
            continue
        idx = candidates.index(raw_answer)
        if idx >= 5:
            skips += 1
            skips_log.append({"task": task, "video": entry["video"], "answer": raw_answer, "reason": f"idx={idx} >= 5"})
            continue
        letter = "ABCDE"[idx]
        options = [f"{c}. {text}" for c, text in zip("ABCDE", candidates)]
        video_raw = entry["video"]
        video_id = os.path.splitext(video_raw.replace("/", "_").replace("\\", "_"))[0]
        sample = {
            "question_id": f"{task}_{len(collected):03d}",
            "task_type": task,
            "videoID": video_id,
            "question": entry["question"],
            "options": options,
            "answer": letter,
            "source_zip": TASK_ZIP[task],
            "video_raw": video_raw,
            "start": entry.get("start"),
            "end": entry.get("end"),
        }
        collected.append(sample)

    if skips > MAX_SKIPS_PER_TASK:
        print(f"ERROR: {task} has {skips} skips (>{MAX_SKIPS_PER_TASK}), aborting")
        raise SystemExit(1)

    all_samples.extend(collected)
    print(f"  {task}: {len(collected)} questions, {skips} skips")

with open(OUT_PATH, "w") as f:
    json.dump(all_samples, f, indent=2, ensure_ascii=False)

with open(VDG_DATA_ROOT + "/mvbench/sampling_skips.log", "w") as f:
    json.dump(skips_log, f, indent=2)

unique_videos = len(set(s["videoID"] for s in all_samples))
print(f"\nTotal: {len(all_samples)} questions, {unique_videos} unique videos")
print(f"Skips: {len(skips_log)} total")
print(f"Saved to {OUT_PATH}")
