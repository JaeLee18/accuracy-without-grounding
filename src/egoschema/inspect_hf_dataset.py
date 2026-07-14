"""Inspect lmms-lab/EgoSchema HuggingFace dataset structure."""
from datasets import load_dataset, get_dataset_config_names

configs = get_dataset_config_names("lmms-lab/EgoSchema")
print("Available configs:", configs)

# Try 'Subset' config
ds = load_dataset("lmms-lab/EgoSchema", "Subset")
print("Splits:", list(ds.keys()))
for split_name, split_ds in ds.items():
    print(f"\nSplit '{split_name}': {len(split_ds)} rows")
    print("Features:", split_ds.features)
    row = split_ds[0]
    for k, v in row.items():
        if isinstance(v, (str, int, float)):
            print(f"  {k}: {repr(v)[:120]}")
        elif isinstance(v, bytes):
            print(f"  {k}: <bytes len={len(v)}>")
        elif hasattr(v, '__len__'):
            print(f"  {k}: <{type(v).__name__} len={len(v)}>")
        else:
            print(f"  {k}: {type(v).__name__}")
