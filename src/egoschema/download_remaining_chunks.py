"""
Finish EgoSchema video download: re-extract chunk 3, then get chunks 4+5.
Deletes each zip immediately after extraction to conserve disk space.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import os, json, zipfile
from huggingface_hub import hf_hub_download

VIDEOS_DIR = VDG_DATA_ROOT + "/egoschema/videos"
CACHE_DIR  = VDG_DATA_ROOT + "/egoschema/hf_cache"
REPO_ID    = "lmms-lab/EgoSchema"
SAMPLES    = VDG_DATA_ROOT + "/egoschema/egoschema_subset.json"

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)


def count_mp4():
    return sum(1 for fn in os.listdir(VIDEOS_DIR) if fn.endswith(".mp4"))


def extract_and_delete(local_zip):
    print(f"  Extracting {os.path.basename(local_zip)} ...")
    before = count_mp4()
    with zipfile.ZipFile(local_zip, "r") as zf:
        members = [m for m in zf.infolist() if m.filename.endswith(".mp4")]
        print(f"  Archive contains {len(members)} mp4 files")
        new_count = 0
        for m in members:
            basename = os.path.basename(m.filename)
            target   = os.path.join(VIDEOS_DIR, basename)
            if os.path.exists(target) and os.path.getsize(target) > 0:
                continue
            with zf.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            new_count += 1
    after = count_mp4()
    print(f"  Extracted {new_count} new files (total: {after})")
    print(f"  Deleting {os.path.basename(local_zip)} to free space ...")
    os.remove(local_zip)
    print(f"  Deleted.")
    return after


def main():
    # Step 1: Re-extract chunk 3 (already downloaded, partially extracted)
    chunk3 = os.path.join(CACHE_DIR, "videos_chunked_03.zip")
    if os.path.exists(chunk3):
        print(f"\n{'='*60}")
        print("Step 1: Finishing extraction of videos_chunked_03.zip ...")
        extract_and_delete(chunk3)
    else:
        print("Chunk 3 zip not found — may have been deleted already.")

    # Check coverage after chunk 3
    check_coverage()

    # Step 2 + 3: Download chunks 4 and 5
    for i, zip_name in enumerate(["videos_chunked_04.zip", "videos_chunked_05.zip"], start=4):
        print(f"\n{'='*60}")
        print(f"Step {i-1}: Downloading and extracting {zip_name} ...")
        local_zip = hf_hub_download(
            repo_id=REPO_ID,
            filename=zip_name,
            repo_type="dataset",
            local_dir=CACHE_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"  Downloaded: {local_zip} ({os.path.getsize(local_zip)/1e9:.2f} GB)")
        extract_and_delete(local_zip)
        check_coverage()

    print(f"\n{'='*60}")
    print("ALL CHUNKS COMPLETE")
    check_coverage(verbose=True)


def check_coverage(verbose=False):
    if not os.path.exists(SAMPLES):
        return
    with open(SAMPLES) as f:
        samples = json.load(f)
    needed  = {s["video_uid"] for s in samples}
    present = {fn.replace(".mp4", "") for fn in os.listdir(VIDEOS_DIR) if fn.endswith(".mp4")}
    found   = needed & present
    missing = needed - present
    print(f"  Subset coverage: {len(found)}/{len(needed)} ({len(found)/len(needed)*100:.1f}%)")
    if verbose and missing:
        print(f"  Still missing: {len(missing)} videos")


if __name__ == "__main__":
    main()
