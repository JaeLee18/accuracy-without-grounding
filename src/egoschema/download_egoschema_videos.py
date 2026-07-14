"""
Download EgoSchema subset videos (500 clips).

EgoSchema videos are clips from the Ego4D dataset. The official subset
videos (500 × ~3-min clips) are provided by the EgoSchema team.

DOWNLOAD METHODS (try in order):

Method 1 — HuggingFace Hub (lmms-lab/egoschema-videos, if available):
  huggingface-cli download lmms-lab/egoschema-videos --local-dir data/egoschema/videos/

Method 2 — EgoSchema official Google Drive (requires gdown):
  pip install gdown
  gdown "https://drive.google.com/uc?id=1_oyJ5rCiVgPMcKsd6KVsr_tenILtDUUT" -O egoschema_videos.zip
  unzip egoschema_videos.zip -d data/egoschema/videos/

Method 3 — Ego4D API (if you have Ego4D access):
  pip install ego4d
  ego4d --output_directory data/egoschema --datasets clips --video_uids <uid_list>

This script attempts Method 2 (gdown) and falls back to listing what's needed.
"""

# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
# ------------------------------------------------------------
import os
import json

VIDEOS_DIR   = VDG_DATA_ROOT + "/egoschema/videos"
SAMPLES_PATH = VDG_DATA_ROOT + "/egoschema/egoschema_subset.json"

def main():
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # Load sample metadata to know which video UIDs we need
    if not os.path.exists(SAMPLES_PATH):
        print(f"[ERROR] Run prepare_egoschema.py first to create {SAMPLES_PATH}")
        return

    with open(SAMPLES_PATH) as f:
        samples = json.load(f)

    needed = sorted({s["video_uid"] for s in samples})
    existing = {fn.replace(".mp4", "") for fn in os.listdir(VIDEOS_DIR) if fn.endswith(".mp4")}
    missing  = [uid for uid in needed if uid not in existing]

    print(f"Videos needed:   {len(needed)}")
    print(f"Videos present:  {len(existing)}")
    print(f"Videos missing:  {len(missing)}")

    if not missing:
        print("\nAll videos present. Ready to run inference.")
        return

    # Try Method 1: gdown
    try:
        import gdown
        print("\nAttempting gdown download (EgoSchema official Google Drive)...")
        # EgoSchema subset video archive — Google Drive file ID from project page
        GDRIVE_ID = "1_oyJ5rCiVgPMcKsd6KVsr_tenILtDUUT"
        out_zip = VDG_DATA_ROOT + "/egoschema/egoschema_videos.zip"
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_ID}",
            out_zip,
            quiet=False,
        )
        # Extract
        import zipfile
        print(f"Extracting {out_zip} ...")
        with zipfile.ZipFile(out_zip, "r") as zf:
            zf.extractall(VIDEOS_DIR)
        print("Extraction complete.")
    except ImportError:
        print("\n[INFO] gdown not installed. Install with: pip install gdown")
        print("Then re-run this script.")
    except Exception as e:
        print(f"\n[WARN] gdown download failed: {e}")
        print("Manual download instructions:")
        print("  1. Visit: https://github.com/egoschema/EgoSchema")
        print("  2. Download the subset videos (500 clips)")
        print("  3. Place mp4 files in:", VIDEOS_DIR)
        print("  4. Each filename must be {video_uid}.mp4")
        print(f"\nFirst 5 needed UIDs:")
        for uid in missing[:5]:
            print(f"  {uid}.mp4")

    # Re-check after download attempt
    existing_after = {fn.replace(".mp4", "") for fn in os.listdir(VIDEOS_DIR) if fn.endswith(".mp4")}
    still_missing  = [uid for uid in needed if uid not in existing_after]
    print(f"\nAfter download: {len(existing_after)} videos present, {len(still_missing)} still missing")

    if not still_missing:
        print("\nAll videos present. Ready to run inference!")
    else:
        # Save the list of missing UIDs for manual handling
        missing_path = VDG_DATA_ROOT + "/egoschema/missing_video_uids.txt"
        with open(missing_path, "w") as f:
            f.write("\n".join(still_missing))
        print(f"Missing UID list saved to: {missing_path}")


if __name__ == "__main__":
    main()
