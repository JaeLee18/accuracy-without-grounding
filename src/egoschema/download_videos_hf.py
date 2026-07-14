"""
Download EgoSchema subset videos from HuggingFace (lmms-lab/EgoSchema).

The repo contains 5 chunked zip archives:
  videos_chunked_01.zip ... videos_chunked_05.zip

This script downloads and extracts all 5 zips into:
  data/egoschema/videos/

Resume-safe: skips already-downloaded and already-extracted files.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import os
import json
import zipfile

from huggingface_hub import hf_hub_download

VIDEOS_DIR   = VDG_DATA_ROOT + "/egoschema/videos"
SAMPLES_PATH = VDG_DATA_ROOT + "/egoschema/egoschema_subset.json"
CACHE_DIR    = VDG_DATA_ROOT + "/egoschema/hf_cache"
REPO_ID      = "lmms-lab/EgoSchema"
ZIP_NAMES    = [
    "videos_chunked_01.zip",
    "videos_chunked_02.zip",
    "videos_chunked_03.zip",
    "videos_chunked_04.zip",
    "videos_chunked_05.zip",
]

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)


def count_mp4(directory):
    return sum(1 for fn in os.listdir(directory) if fn.endswith(".mp4"))


def main():
    for zip_name in ZIP_NAMES:
        print(f"\n{'='*60}")
        print(f"Processing {zip_name} ...")

        # Download via huggingface_hub (handles auth + resuming)
        local_zip = os.path.join(CACHE_DIR, zip_name)
        if os.path.exists(local_zip) and os.path.getsize(local_zip) > 1_000_000:
            print(f"  Already downloaded: {local_zip} ({os.path.getsize(local_zip) / 1e9:.2f} GB)")
        else:
            print(f"  Downloading from HuggingFace ...")
            local_zip = hf_hub_download(
                repo_id=REPO_ID,
                filename=zip_name,
                repo_type="dataset",
                local_dir=CACHE_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"  Saved to: {local_zip} ({os.path.getsize(local_zip) / 1e9:.2f} GB)")

        # Extract
        print(f"  Extracting to {VIDEOS_DIR} ...")
        before = count_mp4(VIDEOS_DIR)
        with zipfile.ZipFile(local_zip, "r") as zf:
            members = zf.infolist()
            mp4_members = [m for m in members if m.filename.endswith(".mp4")]
            print(f"  Archive contains {len(mp4_members)} mp4 files")
            for m in mp4_members:
                # Extract to flat directory (strip any subdirectory structure)
                basename = os.path.basename(m.filename)
                target   = os.path.join(VIDEOS_DIR, basename)
                if os.path.exists(target) and os.path.getsize(target) > 0:
                    continue  # already extracted
                with zf.open(m) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        after = count_mp4(VIDEOS_DIR)
        print(f"  Extracted {after - before} new files (total now: {after})")

    # Final check
    print(f"\n{'='*60}")
    total = count_mp4(VIDEOS_DIR)
    print(f"Total mp4 files in {VIDEOS_DIR}: {total}")

    if os.path.exists(SAMPLES_PATH):
        with open(SAMPLES_PATH) as f:
            samples = json.load(f)
        needed  = {s["video_uid"] for s in samples}
        present = {fn.replace(".mp4", "") for fn in os.listdir(VIDEOS_DIR) if fn.endswith(".mp4")}
        missing = needed - present
        print(f"Subset questions: {len(needed)}  |  Videos present: {len(needed & present)}  |  Missing: {len(missing)}")
        if not missing:
            print("\nAll subset videos present. Ready to run inference!")
        else:
            print(f"\n{len(missing)} videos still missing.")
    print("Done.")


if __name__ == "__main__":
    main()
