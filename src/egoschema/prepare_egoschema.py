"""
Download and prepare the EgoSchema subset (500 questions) from HuggingFace.

Dataset: lmms-lab/EgoSchema, config='Subset', split='test'
Output:  data/egoschema/egoschema_subset.json

HuggingFace field names (actual):
  question_idx : str   sequential index '00000'–'00499'
  video_idx    : str   UUID (used as mp4 filename stem)
  question     : str
  option       : list[str]  5 option texts WITHOUT letter prefix
  answer       : str   '0'–'4' (0-based index)

Output schema per question:
  question_id : str  (video_idx — unique key matching video filename)
  video_uid   : str  (video_idx)
  question    : str
  options     : list[str]  ["A. ...", "B. ...", "C. ...", "D. ...", "E. ..."]
  answer      : str  "A" / "B" / "C" / "D" / "E"
  task_type   : str  "egoschema"

NOTE: Videos are NOT in the HuggingFace dataset. Download separately:
  https://github.com/egoschema/EgoSchema  (Google Drive link)
  Place as data/egoschema/videos/{video_idx}.mp4
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------

import json
import os

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_PATH   = VDG_DATA_ROOT + "/egoschema/egoschema_subset.json"
OPTION_LABELS = list("ABCDE")


def index_to_letter(idx_str):
    """Convert 0-based answer string ('0'–'4') to letter A-E."""
    try:
        idx = int(idx_str)
    except (ValueError, TypeError):
        raise ValueError(f"Unexpected answer value: {idx_str!r}")
    if 0 <= idx <= 4:
        return OPTION_LABELS[idx]
    raise ValueError(f"Answer index out of range: {idx!r}")


def format_options(option_list):
    """Return list of option strings.

    The HuggingFace lmms-lab/EgoSchema Subset dataset stores options already
    prefixed with 'A. ', 'B. ', etc., so we use them verbatim.
    """
    if len(option_list) != 5:
        raise ValueError(f"Expected 5 options, got {len(option_list)}")
    return list(option_list)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading lmms-lab/EgoSchema (config='Subset', split='test') from HuggingFace ...")
    ds = load_dataset("lmms-lab/EgoSchema", "Subset", split="test")
    print(f"  Loaded {len(ds)} rows.")

    samples = []
    errors  = []

    for i, row in enumerate(ds):
        try:
            video_idx = row["video_idx"]
            answer    = index_to_letter(row["answer"])
            options   = format_options(row["option"])

            sample = {
                "question_id": video_idx,       # UUID — unique, matches filename
                "video_uid":   video_idx,        # filename: {video_idx}.mp4
                "question":    row["question"],
                "options":     options,
                "answer":      answer,
                "task_type":   "egoschema",
            }
            samples.append(sample)
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
            print(f"  [WARNING] Row {i}: {e}")

    print(f"\nConverted {len(samples)} questions ({len(errors)} errors).")

    if errors:
        print("Errors:")
        for err in errors:
            print(f"  Row {err['index']}: {err['error']}")

    print(f"\nSaving to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print("Saved.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  Total questions : {len(samples)}")
    answer_counts = {}
    for s in samples:
        answer_counts[s["answer"]] = answer_counts.get(s["answer"], 0) + 1
    for letter in OPTION_LABELS:
        print(f"  Answer {letter}       : {answer_counts.get(letter, 0)}")

    # Sample output
    if samples:
        print("\n--- Sample question (index 0) ---")
        s = samples[0]
        print(f"  question_id : {s['question_id']}")
        print(f"  video_uid   : {s['video_uid']}  (mp4 must be at data/egoschema/videos/{s['video_uid']}.mp4)")
        print(f"  question    : {s['question'][:120]}")
        for opt in s["options"]:
            print(f"    {opt[:100]}")
        print(f"  answer      : {s['answer']}")
        print(f"  task_type   : {s['task_type']}")

    print(f"\nDone. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
