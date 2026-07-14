"""
Download MVBench JSON annotations and video zips from HuggingFace.
Skips fine_grained_pose (requires licence-gated NTU RGB+D).
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import os
from huggingface_hub import hf_hub_download

RAW_DIR = VDG_DATA_ROOT + "/mvbench/raw"
REPO_ID = "OpenGVLab/MVBench"
REPO_TYPE = "dataset"

TASK_TYPES = [
    "action_antonym", "action_count", "action_localization", "action_prediction",
    "action_sequence", "character_order", "counterfactual_inference",
    "egocentric_navigation", "episodic_reasoning", "fine_grained_action",
    "moving_attribute", "moving_count", "moving_direction", "object_existence",
    "object_interaction", "object_shuffle", "scene_transition", "state_change",
    "unexpected_action",
]

VIDEO_ZIPS = [
    "video/FunQA_test.zip",
    "video/Moments_in_Time_Raw.zip",
    "video/clevrer.zip",
    "video/data0613.zip",
    "video/perception.zip",
    "video/scene_qa.zip",
    "video/ssv2_video.zip",
    "video/sta.zip",
    "video/star.zip",
    "video/tvqa.zip",
    "video/vlnqa.zip",
]

os.makedirs(RAW_DIR, exist_ok=True)

print("=== Downloading MVBench JSON annotations ===")
for task in TASK_TYPES:
    dest = os.path.join(RAW_DIR, f"{task}.json")
    if os.path.exists(dest):
        print(f"  [skip] {task}.json")
        continue
    print(f"  Downloading {task}.json ...")
    path = hf_hub_download(REPO_ID, repo_type=REPO_TYPE, filename=f"json/{task}.json")
    import shutil
    shutil.copy(path, dest)
    print(f"  -> {dest}")

print("\n=== Downloading video zips ===")
for zip_file in VIDEO_ZIPS:
    zip_name = os.path.basename(zip_file)
    dest = os.path.join(RAW_DIR, zip_name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 1024:
        print(f"  [skip] {zip_name} (already exists)")
        continue
    print(f"  Downloading {zip_name} ...")
    path = hf_hub_download(REPO_ID, repo_type=REPO_TYPE, filename=zip_file)
    import shutil
    shutil.copy(path, dest)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"  -> {dest} ({size_mb:.0f} MB)")

print("\n=== Download complete ===")
print(f"JSONs: {len([f for f in os.listdir(RAW_DIR) if f.endswith('.json')])}/19")
print(f"Zips:  {len([f for f in os.listdir(RAW_DIR) if f.endswith('.zip')])}/11")
