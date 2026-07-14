"""List and download EgoSchema zip files from HuggingFace."""
from huggingface_hub import list_repo_files, hf_hub_url, HfApi
import os

api = HfApi()

print("All files in lmms-lab/EgoSchema:")
files = list(list_repo_files("lmms-lab/EgoSchema", repo_type="dataset"))
for f in files:
    print(f"  {f}")

# Show URLs for zip files
zip_files = [f for f in files if f.endswith(".zip")]
print(f"\nZip files ({len(zip_files)}):")
for f in zip_files:
    url = hf_hub_url("lmms-lab/EgoSchema", f, repo_type="dataset")
    print(f"  {f}")
    print(f"    URL: {url}")
