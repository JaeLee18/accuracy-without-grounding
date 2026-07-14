# Full Cluster Ablation Results — 2026-03-21

## 11-Model Cross-Summary

| Model | Family | Size | Quant | Orig | Black | VGG |
|-------|--------|------|-------|------|-------|-----|
| Qwen2-VL-2B | Qwen2-VL | 2B | bf16 | 0.462 | 0.317 | +0.145 |
| Qwen2-VL-72B | Qwen2-VL | 72B | 4-bit | 0.412 | 0.348 | +0.063 |
| Qwen2.5-VL-3B | Qwen2.5-VL | 3B | bf16 | 0.572 | 0.362 | +0.209 |
| Qwen2.5-VL-7B | Qwen2.5-VL | 7B | bf16 | 0.630 | 0.333 | +0.297 |
| Qwen2.5-VL-32B | Qwen2.5-VL | 32B | 4-bit | 0.635 | 0.369 | +0.266 |
| Qwen2.5-VL-72B | Qwen2.5-VL | 72B | 4-bit | 0.703 | 0.377 | +0.327 |
| InternVL2-2B | InternVL2 | 2B | bf16 | 0.513 | 0.295 | +0.218 |
| InternVL2-8B | InternVL2 | 8B | bf16 | 0.583 | 0.312 | +0.272 |
| InternVL2-26B | InternVL2 | 26B | bf16 | 0.582 | 0.320 | +0.262 |
| InternVL2-76B | InternVL2 | 76B | 4-bit | 0.308 | 0.357 | -0.048 |
| InternVL2.5-78B | InternVL2.5 | 78B | 4-bit | 0.452 | 0.452 | 0.000 |

## Temporal Reasoning VGG by model

| Model | Temp.Reas VGG |
|-------|---------------|
| Qwen2-VL-2B | +0.010 |
| Qwen2-VL-72B | +0.020 |
| Qwen2.5-VL-3B | +0.127 |
| Qwen2.5-VL-7B | +0.225 |
| Qwen2.5-VL-32B | +0.186 |
| Qwen2.5-VL-72B | +0.260 |
| InternVL2-2B | -0.030 |
| InternVL2-8B | +0.100 |
| InternVL2-26B | +0.100 |
| InternVL2-76B | -0.010 |
| InternVL2.5-78B | +0.030 |

## FPS ablation (Temporal Reasoning, 3 Qwen2-VL scales)

| Model | 0.5 FPS VGG | 1.0 FPS VGG | 2.0 FPS VGG |
|-------|-------------|-------------|-------------|
| Qwen2-VL-2B | +0.040 | +0.050 | +0.050 |
| Qwen2-VL-7B | -0.010 | -0.010 | -0.010 |
| Qwen2-VL-72B | +0.120 | +0.120 | +0.100 |
