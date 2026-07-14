"""Compute three pairwise task-level Spearman r values and CLIP proxy justification note."""
import numpy as np
from scipy.stats import spearmanr

# Verified task-level VGG values from recompute_full_table1.py
tasks = ["Attr.Perc", "Obj.Rec", "OCR", "Act.Rec", "Act.Reas", "Temp.Reas"]
q_vgg  = [0.400, 0.250, 0.321, 0.210, 0.225, 0.000]
l_vgg  = [0.460, 0.480, 0.330, 0.320, 0.160, 0.080]
iv_vgg = [0.380, 0.380, 0.230, 0.250, 0.180, 0.090]

ql, pql   = spearmanr(q_vgg, l_vgg)
qi, pqi   = spearmanr(q_vgg, iv_vgg)
li, pli   = spearmanr(l_vgg, iv_vgg)
avg = np.mean([ql, qi, li])

print("Pairwise task-level Spearman r (6 task types):")
print(f"  Qwen–LLaVA:    r={ql:.3f}  p={pql:.3f}")
print(f"  Qwen–InternVL2: r={qi:.3f}  p={pqi:.3f}")
print(f"  LLaVA–InternVL2: r={li:.3f}  p={pli:.3f}")
print(f"  Mean r: {avg:.3f}")
print()

# Also check what CLIP model each of the 3 models uses
print("CLIP justification note:")
print("  Qwen2-VL: ViT-bigG/14 (custom)")
print("  LLaVA-Video: CLIP ViT-L/14")
print("  InternVL2: InternViT-300M (not CLIP)")
print()
print("CLIP ViT-B/32 proxy: the most widely used public CLIP checkpoint;")
print("LLaVA-Video uses ViT-L/14 which is a stronger CLIP variant.")
print("The proxy argument: if B/32 detects blur, a stronger L/14 encoder also would.")
