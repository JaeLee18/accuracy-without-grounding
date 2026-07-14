"""
Step 2: Sample 30 OCR + 30 Action Recognition questions from Video-MME.
Prioritizes short-duration videos. Saves to data/videomme/pilot_sample.json.
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
OUTPUT_DIR = VDG_DATA_ROOT + "/videomme"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pilot_sample.json")
SEED = 42
SAMPLES_PER_TYPE = 30
TARGET_TYPES = ["OCR Problems", "Action Recognition"]

def main():
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Total annotations: {len(df)}")
    print(f"Task types available: {sorted(df['task_type'].unique())}\n")

    # Filter to target task types
    df_filtered = df[df["task_type"].isin(TARGET_TYPES)].copy()
    print(f"After filtering to {TARGET_TYPES}:")
    for tt in TARGET_TYPES:
        subset = df_filtered[df_filtered["task_type"] == tt]
        print(f"  {tt}: {len(subset)} questions "
              f"(short={len(subset[subset.duration=='short'])}, "
              f"medium={len(subset[subset.duration=='medium'])}, "
              f"long={len(subset[subset.duration=='long'])})")
    print()

    rng = np.random.RandomState(SEED)
    sampled = []

    for task_type in TARGET_TYPES:
        subset = df_filtered[df_filtered["task_type"] == task_type]

        # Prioritize short-duration videos
        short = subset[subset["duration"] == "short"]
        if len(short) >= SAMPLES_PER_TYPE:
            chosen = short.sample(n=SAMPLES_PER_TYPE, random_state=rng)
        else:
            # Take all short, fill remainder from medium, then long
            chosen = short.copy()
            remaining = SAMPLES_PER_TYPE - len(chosen)
            medium = subset[subset["duration"] == "medium"]
            if len(medium) >= remaining:
                chosen = pd.concat([chosen, medium.sample(n=remaining, random_state=rng)])
            else:
                chosen = pd.concat([chosen, medium])
                remaining = SAMPLES_PER_TYPE - len(chosen)
                long_ = subset[subset["duration"] == "long"]
                chosen = pd.concat([chosen, long_.sample(n=min(remaining, len(long_)), random_state=rng)])

        sampled.append(chosen)
        print(f"Sampled {len(chosen)} from {task_type}:")
        for dur in ["short", "medium", "long"]:
            n = len(chosen[chosen["duration"] == dur])
            if n > 0:
                print(f"  {dur}: {n}")

    result_df = pd.concat(sampled, ignore_index=True)

    # Convert to JSON-serializable format
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
    print(f"\nUnique videos needed: {result_df['videoID'].nunique()}")

    # Summary
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
