"""Check raw option text format in lmms-lab/EgoSchema Subset."""
from datasets import load_dataset

ds = load_dataset("lmms-lab/EgoSchema", "Subset", split="test")
row = ds[0]
print("Raw options for row 0:")
for i, opt in enumerate(row["option"]):
    print(f"  [{i}]: {repr(opt)}")
print(f"Answer: {repr(row['answer'])}")
