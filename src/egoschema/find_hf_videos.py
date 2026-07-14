"""Search HuggingFace for EgoSchema video files."""
from huggingface_hub import list_repo_files, hf_hub_download, HfApi
import os

api = HfApi()

# Check lmms-lab/EgoSchema repo for video files
print("Listing files in lmms-lab/EgoSchema (dataset)...")
try:
    files = list(list_repo_files("lmms-lab/EgoSchema", repo_type="dataset"))
    mp4_files = [f for f in files if f.endswith(".mp4")]
    parquet_files = [f for f in files if f.endswith(".parquet")]
    print(f"Total files: {len(files)}")
    print(f"  .mp4 files: {len(mp4_files)}")
    print(f"  .parquet files: {len(parquet_files)}")
    if parquet_files:
        print("Parquet files:", parquet_files[:10])
    if mp4_files:
        print("Sample mp4:", mp4_files[:3])
    else:
        # Show all file types
        from collections import Counter
        ext_counter = Counter(os.path.splitext(f)[1] for f in files)
        print("File extensions:", dict(ext_counter.most_common(20)))
except Exception as e:
    print(f"lmms-lab/EgoSchema error: {e}")
