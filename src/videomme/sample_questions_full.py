"""
Full study: Sample 100 questions from each of 6 task types (600 total).
Prioritizes short-duration, then medium, then long.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_VIDEOMME_SRC = os.environ.get("VDG_VIDEOMME_SRC", "data/raw/Video-MME")
# ------------------------------------------------------------
import json
import os
import pandas as pd
import numpy as np

PARQUET_PATH = VDG_VIDEOMME_SRC + "/videomme/test-00000-of-00001.parquet"
OUTPUT_DIR = VDG_DATA_ROOT + "/videomme_full"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "full_sample.json")
SEED = 42
SAMPLES_PER_TYPE = 100

# 6 task types: 3 static + 3 temporal
TARGET_TYPES = [
    "OCR Problems",
    "Object Recognition",
    "Attribute Perception",
    "Action Recognition",
    "Action Reasoning",
    "Temporal Reasoning",
]

def sample_prioritized(subset, n, rng):
    """Sample n from subset, prioritizing short > medium > long."""
    chosen_parts = []
    remaining = n
    for dur in ["short", "medium", "long"]:
        pool = subset[subset["duration"] == dur]
        if len(pool) >= remaining:
            chosen_parts.append(pool.sample(n=remaining, random_state=rng))
            remaining = 0
            break
        else:
            chosen_parts.append(pool)
            remaining -= len(pool)
    if remaining > 0:
        print(f"  WARNING: could only get {n - remaining}/{n} samples")
    return pd.concat(chosen_parts, ignore_index=True)

def main():
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Total annotations: {len(df)}")

    rng = np.random.RandomState(SEED)
    sampled = []

    for task_type in TARGET_TYPES:
        subset = df[df["task_type"] == task_type]
        chosen = sample_prioritized(subset, SAMPLES_PER_TYPE, rng)
        sampled.append(chosen)
        print(f"{task_type:<25} sampled {len(chosen):>3}: "
              f"short={len(chosen[chosen.duration=='short']):>3}, "
              f"medium={len(chosen[chosen.duration=='medium']):>3}, "
              f"long={len(chosen[chosen.duration=='long']):>3}")

    result_df = pd.concat(sampled, ignore_index=True)

    # Convert to JSON
    records = []
    for _, row in result_df.iterrows():
        records.append({
            "video_id": row["video_id"],
            "duration": row["duration"],
            "domain": row["domain"],
            "sub_category": row["sub_category"],
            "videoID": row["videoID"],
            "question_id": row["question_id"],
            "task_type": row["task_type"],
            "question": row["question"],
            "options": list(row["options"]),
            "answer": row["answer"],
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(records)} questions to {OUTPUT_PATH}")
    print(f"Unique videos: {result_df['videoID'].nunique()}")

    print("\n=== SUMMARY ===")
    print(f"{'Task Type':<25} {'Count':<8} {'Short':<8} {'Medium':<8} {'Long':<8}")
    print("-" * 57)
    for tt in TARGET_TYPES:
        sub = result_df[result_df["task_type"] == tt]
        print(f"{tt:<25} {len(sub):<8} "
              f"{len(sub[sub.duration=='short']):<8} "
              f"{len(sub[sub.duration=='medium']):<8} "
              f"{len(sub[sub.duration=='long']):<8}")

if __name__ == "__main__":
    main()
