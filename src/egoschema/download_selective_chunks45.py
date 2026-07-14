"""
Download chunks 4 and 5 but extract ONLY the missing subset videos.
This keeps peak disk usage to ~21GB (one zip at a time) instead of 42+GB.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import os, json, zipfile
from huggingface_hub import hf_hub_download

VIDEOS_DIR = VDG_DATA_ROOT + "/egoschema/videos"
CACHE_DIR  = VDG_DATA_ROOT + "/egoschema/hf_cache"
SAMPLES    = VDG_DATA_ROOT + "/egoschema/egoschema_subset.json"
REPO_ID    = "lmms-lab/EgoSchema"

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)


def load_missing_uids():
    with open(SAMPLES) as f:
        samples = json.load(f)
    needed  = {s["video_uid"] for s in samples}
    present = {fn.replace(".mp4", "") for fn in os.listdir(VIDEOS_DIR) if fn.endswith(".mp4")}
    missing = needed - present
    return missing


def selective_extract_and_delete(local_zip, target_uids):
    """Extract only mp4 files whose stem is in target_uids, then delete zip."""
    print(f"  Opening {os.path.basename(local_zip)} ...")
    found = 0
    extracted = 0
    with zipfile.ZipFile(local_zip, "r") as zf:
        members = zf.infolist()
        mp4s = [m for m in members if m.filename.endswith(".mp4")]
        print(f"  Archive has {len(mp4s)} mp4 files, looking for {len(target_uids)} needed.")
        for m in mp4s:
            basename = os.path.basename(m.filename)
            uid = basename.replace(".mp4", "")
            if uid not in target_uids:
                continue
            found += 1
            target = os.path.join(VIDEOS_DIR, basename)
            if os.path.exists(target) and os.path.getsize(target) > 0:
                continue
            with zf.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted += 1
    print(f"  Found {found} needed videos in archive, extracted {extracted} new ones.")
    print(f"  Deleting zip to free space ...")
    import time, gc
    gc.collect()
    for attempt in range(6):
        try:
            os.remove(local_zip)
            print(f"  Deleted.")
            break
        except PermissionError:
            print(f"  File locked (attempt {attempt+1}/6), waiting 3s ...")
            time.sleep(3)
    else:
        print(f"  WARNING: Could not delete {local_zip} — delete manually to free ~21 GB.")
    return extracted


def check_coverage():
    with open(SAMPLES) as f:
        samples = json.load(f)
    needed  = {s["video_uid"] for s in samples}
    present = {fn.replace(".mp4", "") for fn in os.listdir(VIDEOS_DIR) if fn.endswith(".mp4")}
    found   = needed & present
    missing = needed - present
    print(f"  Coverage: {len(found)}/{len(needed)} ({len(found)/len(needed)*100:.1f}%),  missing: {len(missing)}")
    return needed - present


def main():
    import shutil
    free_bytes = shutil.disk_usage(VDG_DATA_ROOT).free
    print(f"Current free disk space: {free_bytes/1e9:.1f} GB")

    for zip_name in ["videos_chunked_04.zip", "videos_chunked_05.zip"]:
        missing = load_missing_uids()
        if not missing:
            print(f"\nAll subset videos found! Skipping {zip_name}.")
            break

        print(f"\n{'='*60}")
        print(f"Processing {zip_name} ... need {len(missing)} more videos")

        local_zip = os.path.join(CACHE_DIR, zip_name)
        if not (os.path.exists(local_zip) and os.path.getsize(local_zip) > 1_000_000):
            free_before = shutil.disk_usage(VDG_DATA_ROOT).free
            print(f"  Downloading (free: {free_before/1e9:.1f} GB) ...")
            local_zip = hf_hub_download(
                repo_id=REPO_ID,
                filename=zip_name,
                repo_type="dataset",
                local_dir=CACHE_DIR,
                local_dir_use_symlinks=False,
            )
            free_after = shutil.disk_usage(VDG_DATA_ROOT).free
            print(f"  Downloaded: {os.path.getsize(local_zip)/1e9:.2f} GB  "
                  f"(free now: {free_after/1e9:.1f} GB)")
        else:
            print(f"  Already downloaded: {local_zip}")

        selective_extract_and_delete(local_zip, missing)
        missing_after = check_coverage()
        print(f"  Still missing after this chunk: {len(missing_after)}")

    print(f"\n{'='*60}")
    print("Final state:")
    missing_final = check_coverage()
    if not missing_final:
        print("\n  ALL 500 subset videos are present. Ready to run inference!")
    else:
        print(f"\n  {len(missing_final)} videos still missing.")
        print("  Possibly in a chunk not yet downloaded, or not in the dataset.")
        # Save missing list
        missing_path = VDG_DATA_ROOT + "/egoschema/still_missing.txt"
        with open(missing_path, "w") as f:
            f.write("\n".join(sorted(missing_final)))
        print(f"  Missing UIDs saved to: {missing_path}")


if __name__ == "__main__":
    main()
