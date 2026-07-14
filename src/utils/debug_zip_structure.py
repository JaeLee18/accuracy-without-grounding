
# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import json, zipfile, os

with open(VDG_DATA_ROOT + "/mvbench/mvbench_sample.json", encoding="latin-1") as f:
    samples = json.load(f)

def check_zip(zip_name, task_types):
    print(f"\n=== {zip_name} ===")
    zip_path = VDG_DATA_ROOT + f"/mvbench/raw/{zip_name}"
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
    mp4s = [m for m in members if m.lower().endswith('.mp4')]
    print(f"Total MP4s in zip: {len(mp4s)}")
    if mp4s:
        print("First 5:", mp4s[:5])

    task_samples = [s for s in samples if s["task_type"] in task_types]
    print(f"\nNeeded videos ({task_types}):")
    for s in task_samples[:3]:
        print(f"  video_raw='{s['video_raw']}' basename='{os.path.basename(s['video_raw'])}'")

    # Check matching
    basenames_in_zip = set(os.path.basename(m) for m in mp4s)
    found = sum(1 for s in task_samples if os.path.basename(s["video_raw"]) in basenames_in_zip)
    print(f"Matching by basename: {found}/{len(task_samples)}")

check_zip("data0613.zip", ["action_antonym", "action_sequence", "fine_grained_action",
                            "moving_attribute", "moving_count", "moving_direction", "object_shuffle"])
check_zip("sta.zip", ["action_count"])
check_zip("star.zip", ["action_localization", "action_prediction"])
check_zip("clevrer.zip", ["counterfactual_inference", "object_existence", "object_interaction"])
