"""Check if lmms-lab/EgoSchema GENERATION config has video data."""
from datasets import load_dataset

print("Checking GENERATION config structure...")
ds = load_dataset("lmms-lab/EgoSchema", "GENERATION", split="test")
print(f"Rows: {len(ds)}")
print("Features:", ds.features)
row = ds[0]
for k, v in row.items():
    if isinstance(v, bytes):
        print(f"  {k}: <bytes len={len(v)}>")
    elif hasattr(v, 'filename'):
        print(f"  {k}: <file: {v.filename}>")
    elif isinstance(v, dict) and 'bytes' in v:
        print(f"  {k}: <dict with bytes key, path={v.get('path','?')}, len={len(v.get('bytes', b''))}>")
    elif isinstance(v, (str, int, float)):
        print(f"  {k}: {repr(v)[:120]}")
    else:
        print(f"  {k}: {type(v).__name__}: {repr(v)[:200]}")
