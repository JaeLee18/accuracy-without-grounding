"""
Extract only the videos needed for the full study from zip chunks.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_VIDEOMME_SRC = os.environ.get("VDG_VIDEOMME_SRC", "data/raw/Video-MME")
# ------------------------------------------------------------
import json
import os
import zipfile
from tqdm import tqdm

SAMPLE_PATH = VDG_DATA_ROOT + "/videomme_full/full_sample.json"
ZIP_DIR = VDG_VIDEOMME_SRC + ""
OUTPUT_DIR = VDG_DATA_ROOT + "/videomme_full/videos"

def main():
    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    needed_ids = set(s["videoID"] for s in samples)
    needed_files = {f"data/{vid}.mp4" for vid in needed_ids}
    print(f"Need {len(needed_ids)} unique videos")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    already = set(os.listdir(OUTPUT_DIR))
    still_needed = {f for f in needed_files if f.replace("data/", "") not in already}
    if len(still_needed) < len(needed_files):
        print(f"Already extracted: {len(needed_files) - len(still_needed)}")

    if not still_needed:
        print("All videos already extracted!")
        return

    zip_files = sorted([
        os.path.join(ZIP_DIR, f) for f in os.listdir(ZIP_DIR)
        if f.startswith("videos_chunked_") and f.endswith(".zip")
    ])
    print(f"Scanning {len(zip_files)} zip files for {len(still_needed)} videos...")

    extracted = 0
    for zf_path in tqdm(zip_files, desc="Zip files"):
        with zipfile.ZipFile(zf_path, 'r') as zf:
            names = set(zf.namelist())
            to_extract = still_needed & names
            if not to_extract:
                continue
            for member in to_extract:
                filename = os.path.basename(member)
                target = os.path.join(OUTPUT_DIR, filename)
                with zf.open(member) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
                extracted += 1
                still_needed.discard(member)

    print(f"\nExtracted {extracted} videos to {OUTPUT_DIR}")
    if still_needed:
        print(f"WARNING: {len(still_needed)} videos not found!")
        for f in sorted(still_needed):
            print(f"  Missing: {f}")
    else:
        print("All needed videos extracted successfully!")

    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f"{vid}.mp4"))
        for vid in needed_ids
        if os.path.exists(os.path.join(OUTPUT_DIR, f"{vid}.mp4"))
    )
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
