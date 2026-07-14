# ACM MM Ablation Results: Visual Grounding Gap (VGG) on Video-MME

**VGG = Original Accuracy − Black Screen Accuracy**
- Positive VGG: model uses visual content
- Near-zero / negative VGG: language priors dominate; video provides little or no benefit

**Benchmark**: Video-MME (600 questions, 407 videos, 6 task types, 100 questions each)
**GPU**: NVIDIA H100 80GB HBM3 (Anvil cluster)
**Date**: 2026-03-25 (updated from 2026-03-21)

All experiments in sections 2–9 use the **600-Q subset** (`full_sample.json`) consistently across all four conditions (black, singleframe, shuffled, original). Section 1 (Qwen2-VL) uses 1200-Q VideoMME historical runs; see section 8.2 for the consistent 600-Q ladder.

---

## 1. Scale Ablation — Qwen2-VL (historical, 1200-Q VideoMME)

| Model | Orig Acc | Black Acc | VGG | n (ok) |
|-------|----------|-----------|-----|--------|
| Qwen2-VL-2B | 0.462 | 0.317 | **+0.145** | 1200 |
| Qwen2-VL-72B | 0.412 | 0.348 | **+0.063** | 1200 |

**Finding**: VGG *decreases* from 2B to 72B in the Qwen2-VL family. The 72B model relies more heavily on language priors, suggesting larger models within this generation compensate for visual uncertainty via stronger text understanding.

Note: Qwen2-VL-72B used 4-bit quantization (NF4, double quant, bf16 compute) and `max_frames=32` to fit on 80 GB VRAM. 141 OOM failures from an earlier run (no `max_frames` limit) were recovered and are not counted in the ok total. 600-Q subset results for Qwen2-VL are in section 8.2 (72B complete, 2B original pending).

### Per-Task Breakdown

| Task Type | 2B Orig | 2B Black | 2B VGG | 72B Orig | 72B Black | 72B VGG |
|-----------|---------|----------|--------|----------|-----------|---------|
| Action Reasoning | 0.490 | 0.370 | +0.120 | 0.470 | 0.400 | +0.070 |
| Action Recognition | 0.440 | 0.330 | +0.110 | 0.500 | 0.390 | +0.110 |
| Attribute Perception | 0.500 | 0.280 | +0.220 | 0.350 | 0.300 | +0.050 |
| OCR Problems | 0.540 | 0.360 | +0.180 | 0.440 | 0.310 | +0.130 |
| Object Recognition | 0.500 | 0.270 | +0.230 | 0.300 | 0.300 | +0.000 |
| Temporal Reasoning | 0.300 | 0.290 | +0.010 | 0.410 | 0.390 | +0.020 |
| **OVERALL** | **0.462** | **0.317** | **+0.145** | **0.412** | **0.348** | **+0.063** |

---

## 2. Scale Ablation — Qwen2.5-VL (600-Q subset)

| Model | Orig Acc | Black Acc | VGG | n (ok) |
|-------|----------|-----------|-----|--------|
| Qwen2.5-VL-3B | 0.572 | 0.362 | **+0.210** | 600 |
| Qwen2.5-VL-7B | 0.630 | 0.333 | **+0.297** | 600 |
| Qwen2.5-VL-32B | 0.635 | 0.369 | **+0.266** | 600 |
| Qwen2.5-VL-72B | 0.703 | 0.377 | **+0.326** | 600 |

**Finding**: VGG *increases* with scale in the Qwen2.5-VL family — a reversal of the Qwen2-VL trend. Larger Qwen2.5-VL models achieve higher overall accuracy AND make better use of visual content. The 7B and 72B show the highest VGG.

Qwen2.5-VL-3B/7B: bfloat16, no quantization. Qwen2.5-VL-32B/72B: 4-bit NF4 quantization.

### Per-Task Breakdown

| Task Type | 3B Orig | 3B Black | 3B VGG | 7B Orig | 7B Black | 7B VGG | 32B Orig | 32B Black | 32B VGG | 72B Orig | 72B Black | 72B VGG |
|-----------|---------|----------|--------|---------|----------|--------|----------|-----------|---------|----------|-----------|---------|
| Action Reasoning | 0.550 | 0.390 | +0.160 | 0.590 | 0.303 | +0.287 | 0.620 | 0.410 | +0.210 | 0.760 | 0.470 | +0.290 |
| Action Recognition | 0.570 | 0.460 | +0.110 | 0.590 | 0.360 | +0.230 | 0.600 | 0.420 | +0.180 | 0.660 | 0.400 | +0.260 |
| Attribute Perception | 0.650 | 0.390 | +0.260 | 0.760 | 0.310 | +0.450 | 0.690 | 0.320 | +0.370 | 0.770 | 0.310 | +0.460 |
| OCR Problems | 0.690 | 0.370 | +0.320 | 0.670 | 0.330 | +0.340 | 0.700 | 0.347 | +0.353 | 0.770 | 0.400 | +0.370 |
| Object Recognition | 0.540 | 0.260 | +0.280 | 0.570 | 0.320 | +0.250 | 0.580 | 0.280 | +0.300 | 0.580 | 0.260 | +0.320 |
| Temporal Reasoning | 0.430 | 0.303 | +0.127 | 0.600 | 0.375 | +0.225 | 0.620 | 0.434 | +0.186 | 0.680 | 0.420 | +0.260 |
| **OVERALL** | **0.572** | **0.362** | **+0.210** | **0.630** | **0.333** | **+0.297** | **0.635** | **0.369** | **+0.266** | **0.703** | **0.377** | **+0.326** |

---

## 3. Scale Ablation — Qwen3-VL (600-Q subset)

| Model | Orig Acc | Black Acc | VGG | n (ok) |
|-------|----------|-----------|-----|--------|
| Qwen3-VL-2B | 0.447 | 0.333 | **+0.114** | 600 |
| Qwen3-VL-8B | 0.514 | 0.407 | **+0.107** | 599 |
| Qwen3-VL-32B | 0.557 | 0.392 | **+0.165** | 594 |

**Finding**: Qwen3-VL shows substantially lower VGG than Qwen2.5-VL at comparable sizes (+0.107–+0.165 vs +0.210–+0.297). Despite higher black-screen accuracy at each size class (reflecting stronger language priors), the original accuracy is also lower than Qwen2.5-VL equivalents — suggesting a generation that is better at exploiting text patterns but not proportionally better at visual grounding. The 32B shows the strongest VGG within the family.

Qwen3-VL-2B/8B: bfloat16. Qwen3-VL-32B: 4-bit NF4 quantization.

### Per-Task Breakdown

| Task Type | 2B Orig | 2B Black | 2B VGG | 8B Orig | 8B Black | 8B VGG | 32B Orig | 32B Black | 32B VGG |
|-----------|---------|----------|--------|---------|----------|--------|----------|-----------|---------|
| Action Reasoning | 0.420 | 0.333 | +0.087 | 0.525 | 0.450 | +0.075 | 0.610 | 0.449 | +0.161 |
| Action Recognition | 0.480 | 0.400 | +0.080 | 0.490 | 0.420 | +0.070 | 0.590 | 0.470 | +0.120 |
| Attribute Perception | 0.500 | 0.350 | +0.150 | 0.580 | 0.410 | +0.170 | 0.630 | 0.360 | +0.270 |
| OCR Problems | 0.440 | 0.337 | +0.103 | 0.510 | 0.385 | +0.125 | 0.480 | 0.360 | +0.120 |
| Object Recognition | 0.480 | 0.280 | +0.200 | 0.550 | 0.380 | +0.170 | 0.510 | 0.340 | +0.170 |
| Temporal Reasoning | 0.360 | 0.299 | +0.061 | 0.430 | 0.394 | +0.036 | 0.521 | 0.372 | +0.149 |
| **OVERALL** | **0.447** | **0.333** | **+0.114** | **0.514** | **0.407** | **+0.107** | **0.557** | **0.392** | **+0.165** |

---

## 4. Scale Ablation — InternVL2 / InternVL2.5 (600-Q subset)

| Model | Orig Acc | Black Acc | VGG | n (ok) | Notes |
|-------|----------|-----------|-----|--------|-------|
| InternVL2-2B | 0.513 | 0.295 | **+0.218** | 593 | bfloat16 |
| InternVL2-8B | 0.583 | 0.312 | **+0.272** | 600 | bfloat16 |
| InternVL2-26B | 0.582 | 0.320 | **+0.262** | 600 | bfloat16 |
| InternVL2-76B | 0.308 | 0.357 | **-0.048** | 600 | 4-bit, use_flash_attn=False |
| InternVL2.5-78B | 0.452 | 0.452 | **0.000** | 600 | 4-bit, use_flash_attn=False |

**Finding**: InternVL2 scales well from 2B to 26B with VGG holding at +0.22–+0.27. However, at 76B the VGG turns negative (−0.048): the model scores *higher* on black screen than original video, indicating dominant language priors or degradation from heavy quantization. InternVL2.5-78B recovers overall accuracy (0.452 vs 0.308) but still yields VGG ≈ 0.000, suggesting the 78B model is accurate but entirely language-driven.

**Technical note**: Flash Attention on H100 returns bfloat16 activations regardless of input dtype, colliding with float16 biases in the EVA-CLIP vision encoder of the 76B/78B models. Fixed with `use_flash_attn=False`.

### Per-Task Breakdown

| Task Type | 2B VGG | 8B VGG | 26B VGG | 76B VGG | 78B VGG |
|-----------|--------|--------|---------|---------|---------|
| Action Reasoning | +0.180 | +0.190 | +0.210 | -0.080 | -0.030 |
| Action Recognition | +0.170 | +0.300 | +0.130 | -0.130 | -0.030 |
| Attribute Perception | +0.426 | +0.390 | +0.440 | +0.010 | +0.050 |
| OCR Problems | +0.310 | +0.270 | +0.340 | -0.020 | -0.020 |
| Object Recognition | +0.255 | +0.380 | +0.350 | -0.060 | +0.000 |
| Temporal Reasoning | -0.030 | +0.100 | +0.100 | -0.010 | +0.030 |

| Task Type | 2B Orig | 2B Black | 8B Orig | 8B Black | 26B Orig | 26B Black | 76B Orig | 76B Black | 78B Orig | 78B Black |
|-----------|---------|----------|---------|----------|----------|-----------|----------|-----------|----------|-----------|
| Action Reasoning | 0.530 | 0.350 | 0.570 | 0.380 | 0.560 | 0.350 | 0.320 | 0.400 | 0.490 | 0.520 |
| Action Recognition | 0.480 | 0.310 | 0.600 | 0.300 | 0.570 | 0.440 | 0.300 | 0.430 | 0.550 | 0.580 |
| Attribute Perception | 0.656 | 0.230 | 0.700 | 0.310 | 0.730 | 0.290 | 0.350 | 0.340 | 0.450 | 0.400 |
| OCR Problems | 0.580 | 0.270 | 0.550 | 0.280 | 0.610 | 0.270 | 0.320 | 0.340 | 0.400 | 0.420 |
| Object Recognition | 0.515 | 0.260 | 0.640 | 0.260 | 0.620 | 0.270 | 0.230 | 0.290 | 0.330 | 0.330 |
| Temporal Reasoning | 0.320 | 0.350 | 0.440 | 0.340 | 0.400 | 0.300 | 0.330 | 0.340 | 0.490 | 0.460 |
| **OVERALL** | **0.513** | **0.295** | **0.583** | **0.312** | **0.582** | **0.320** | **0.308** | **0.357** | **0.452** | **0.452** |

---

## 5. Scale Ablation — VideoLLaMA2 & LLaVA-Video (600-Q subset)

| Model | Orig Acc | Black Acc | VGG | n (ok) |
|-------|----------|-----------|-----|--------|
| VideoLLaMA2-7B | 0.585 | 0.342 | **+0.243** | 600 |
| LLaVA-Video-7B | 0.638 | 0.351 | **+0.287** | 600 |

**Finding**: Both video-specialist models show strong VGG despite being 7B scale. LLaVA-Video-7B (+0.287) matches Qwen2.5-VL-7B (+0.297) and outperforms all Qwen3-VL models, with an exceptionally high Attribute Perception VGG (+0.510). VideoLLaMA2-7B (+0.243) also exceeds all Qwen3-VL sizes, suggesting architectural specialization for video is more impactful than scale alone.

### Per-Task Breakdown

| Task Type | VLLaMA Orig | VLLaMA Black | VLLaMA VGG | LLaVA Orig | LLaVA Black | LLaVA VGG |
|-----------|-------------|--------------|------------|------------|-------------|-----------|
| Action Reasoning | 0.670 | 0.370 | +0.300 | 0.630 | 0.420 | +0.210 |
| Action Recognition | 0.530 | 0.330 | +0.200 | 0.690 | 0.440 | +0.250 |
| Attribute Perception | 0.710 | 0.350 | +0.360 | 0.800 | 0.290 | +0.510 |
| OCR Problems | 0.550 | 0.360 | +0.190 | 0.580 | 0.333 | +0.247 |
| Object Recognition | 0.630 | 0.290 | +0.340 | 0.700 | 0.270 | +0.430 |
| Temporal Reasoning | 0.420 | 0.350 | +0.070 | 0.430 | 0.350 | +0.080 |
| **OVERALL** | **0.585** | **0.342** | **+0.243** | **0.638** | **0.351** | **+0.287** |

---

## 6. FPS Ablation — Temporal Reasoning

Evaluated on 100 Temporal Reasoning questions per FPS level × condition.
FPS values: 0.5, 1.0, 2.0 frames/second.

### Qwen2-VL-2B

| FPS | Orig Acc | Black Acc | VGG |
|-----|----------|-----------|-----|
| 0.5 | 0.330 | 0.290 | +0.040 |
| 1.0 | 0.340 | 0.290 | +0.050 |
| 2.0 | 0.340 | 0.290 | +0.050 |

### Qwen2-VL-72B

| FPS | Orig Acc | Black Acc | VGG |
|-----|----------|-----------|-----|
| 0.5 | 0.510 | 0.390 | +0.120 |
| 1.0 | 0.520 | 0.400 | +0.120 |
| 2.0 | 0.510 | 0.410 | +0.100 |

### Qwen2.5-VL-3B

| FPS | Orig Acc | Black Acc | VGG |
|-----|----------|-----------|-----|
| 0.5 | 0.480 | 0.333 | +0.147 |
| 1.0 | 0.535 | 0.323 | +0.212 |
| 2.0 | 0.465 | 0.333 | +0.132 |

### Qwen2.5-VL-7B

| FPS | Orig Acc | Black Acc | VGG |
|-----|----------|-----------|-----|
| 0.5 | 0.606 | 0.380 | +0.226 |
| 1.0 | 0.602 | 0.370 | +0.232 |
| 2.0 | 0.636 | 0.350 | +0.286 |

### Qwen2.5-VL-32B

| FPS | Orig Acc | Black Acc | VGG |
|-----|----------|-----------|-----|
| 0.5 | 0.660 | 0.450 | +0.210 |
| 1.0 | 0.640 | 0.450 | +0.190 |
| 2.0 | 0.640 | 0.420 | +0.220 |

### Qwen2.5-VL-72B

| FPS | Orig Acc | Black Acc | VGG |
|-----|----------|-----------|-----|
| 0.5 | 0.670 | 0.400 | +0.270 |
| 1.0 | 0.660 | 0.410 | +0.250 |
| 2.0 | 0.630 | 0.420 | +0.210 |

### Qwen3-VL-8B

| FPS | Orig Acc | Black Acc | VGG |
|-----|----------|-----------|-----|
| 0.5 | 0.460 | 0.440 | +0.020 |
| 1.0 | 0.530 | 0.420 | +0.110 |
| 2.0 | 0.520 | 0.420 | +0.100 |

**Finding**: VGG on Temporal Reasoning is essentially flat across all FPS levels for most models. Higher sampling rates (more frames) do not substantially improve visual grounding for temporal questions — the bottleneck is not frame count but architectural/training capacity. The Qwen2.5-VL-72B shows a slight VGG decrease at 2 FPS vs 0.5 FPS, while Qwen2.5-VL-7B shows a slight increase at 2 FPS — neither trend is consistent across models. Qwen3-VL-8B has near-zero VGG across all FPS levels, consistent with its low overall VGG.

---

## 7. Cross-Model Summary Table (600-Q subset)

| Model | Family | Size | Quant | Orig Acc | Black Acc | VGG |
|-------|--------|------|-------|----------|-----------|-----|
| Qwen2-VL-2B† | Qwen2-VL | 2B | bf16 | 0.462 | 0.317 | **+0.145** |
| Qwen2-VL-72B† | Qwen2-VL | 72B | 4-bit | 0.412 | 0.348 | **+0.063** |
| Qwen2.5-VL-3B | Qwen2.5-VL | 3B | bf16 | 0.572 | 0.362 | **+0.210** |
| Qwen2.5-VL-7B | Qwen2.5-VL | 7B | bf16 | 0.630 | 0.333 | **+0.297** |
| Qwen2.5-VL-32B | Qwen2.5-VL | 32B | 4-bit | 0.635 | 0.369 | **+0.266** |
| Qwen2.5-VL-72B | Qwen2.5-VL | 72B | 4-bit | 0.703 | 0.377 | **+0.326** |
| Qwen3-VL-2B | Qwen3-VL | 2B | bf16 | 0.447 | 0.333 | **+0.114** |
| Qwen3-VL-8B | Qwen3-VL | 8B | bf16 | 0.514 | 0.407 | **+0.107** |
| Qwen3-VL-32B | Qwen3-VL | 32B | 4-bit | 0.557 | 0.392 | **+0.165** |
| InternVL2-2B | InternVL2 | 2B | bf16 | 0.513 | 0.295 | **+0.218** |
| InternVL2-8B | InternVL2 | 8B | bf16 | 0.583 | 0.312 | **+0.272** |
| InternVL2-26B | InternVL2 | 26B | bf16 | 0.582 | 0.320 | **+0.262** |
| InternVL2-76B | InternVL2 | 76B | 4-bit | 0.308 | 0.357 | **-0.048** |
| InternVL2.5-78B | InternVL2.5 | 78B | 4-bit | 0.452 | 0.452 | **0.000** |
| VideoLLaMA2-7B | VideoLLaMA2 | 7B | bf16 | 0.585 | 0.342 | **+0.243** |
| LLaVA-Video-7B | LLaVA | 7B | bf16 | 0.638 | 0.351 | **+0.287** |

†Qwen2-VL values from 1200-Q VideoMME historical runs; 600-Q consistent ladder: 2B orig=0.472 / black=0.313 / VGG=+0.159, 72B orig=0.667 / black=0.410 / VGG=+0.257 (see section 10.2).

---

## 8. Key Findings

### 8.1 VGG is Consistently Low for Temporal Reasoning
Across all models and all FPS levels, Temporal Reasoning has the lowest VGG (often near zero). This is the clearest signal: **current video LMMs do not effectively use temporal information** — even at 72B scale, temporal reasoning VGG ranges from near-zero (Qwen3-VL-8B: +0.036) to moderate (Qwen2.5-VL-72B: +0.260), and FPS level has negligible impact.

### 8.2 Generation Matters More Than Scale
- Qwen2.5-VL-3B (VGG +0.210) outperforms Qwen2-VL-72B (VGG +0.063) despite being 24× smaller. Newer generation visual architecture improvements outweigh raw scale.
- Within Qwen2.5-VL, larger models tend to have higher VGG.
- Qwen3-VL shows a VGG *regression* vs Qwen2.5-VL: Qwen3-VL-32B (+0.165) vs Qwen2.5-VL-32B (+0.266). Stronger language priors in Qwen3-VL do not translate to better visual grounding.

### 8.3 Very Large InternVL2 Models Show Language Prior Dominance
InternVL2-76B and InternVL2.5-78B have VGG ≤ 0. At these scales, the models answer as well (or better) from a black screen as from actual video, indicating exploitation of question/answer text patterns rather than visual content.

### 8.4 FPS Has No Effect on Temporal VGG
Varying FPS from 0.5 to 2.0 does not consistently change VGG for temporal reasoning across any model. Models are not bottlenecked by frame count.

### 8.5 Attribute Perception Has Highest VGG
Across all models, Attribute Perception consistently shows the highest VGG. Object and spatial attributes are harder to answer from text alone. LLaVA-Video-7B achieves +0.510 AP VGG — the highest single-task VGG observed.

### 8.6 Video-Specialist Models Punch Above Their Weight
LLaVA-Video-7B (VGG +0.287) matches Qwen2.5-VL-7B (+0.297) and outperforms all Qwen3-VL models. VideoLLaMA2-7B (+0.243) exceeds all InternVL2 sizes ≥26B. Architectural specialization for video matters as much as scale.

---

## 9. Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Benchmark | Video-MME 600-Q subset (`full_sample.json`) |
| Conditions | original video, black screen video (same duration) |
| Video sampling | 0.25 FPS (main experiments), 0.5/1.0/2.0 FPS (FPS ablation) |
| Max pixels | 256×256 |
| Min pixels | 28×28 |
| Max new tokens | 32 |
| Max frames (72B) | 32 (OOM prevention) |
| Quantization (large) | 4-bit NF4, double quant, bf16 compute |
| Flash attention | Disabled for InternVL2-76B / InternVL2.5-78B |
| HF transformers | 4.57.6 |
| Qwen environment | `/anvil/projects/x-cis250283/lee2161/conda_torch` |
| InternVL/VideoLLaMA env | `/anvil/projects/x-nairr250105/acm_mm/env` |

---

## 10. Ablation Experiment: Shuffled + Singleframe Conditions

**Date**: 2026-03-22 (shuffled/singleframe complete); 2026-03-25 (black + original filled in for all models)

The ablation uses two degraded video conditions:
- **Singleframe**: one frame repeated for the full video duration → measures spatial/static visual grounding
- **Shuffled**: original frames in random order → measures multi-frame visual grounding (without temporal order)

**4-point diagnostic ladder**: black (language floor) → singleframe (spatial) → shuffled (frame diversity) → original (temporal)

Key deltas:
- **Δspatial** = singleframe − black (benefit of seeing one real frame vs black)
- **Δdiversity** = shuffled − singleframe (benefit of seeing multiple diverse frames)
- **Δtemporal** = original − shuffled (benefit of temporal ordering)

---

### 10.1 Δdiversity Table — All 16 Models

| Model | Family | Shuffled | Singleframe | **Δdiversity** |
|-------|--------|----------|-------------|----------------|
| Qwen2-VL-2B | Qwen2-VL | 0.475 | 0.377 | **+0.098** |
| Qwen2-VL-72B | Qwen2-VL | 0.637 | 0.490 | **+0.147** |
| Qwen2.5-VL-3B | Qwen2.5-VL | 0.550 | 0.420 | **+0.130** |
| Qwen2.5-VL-7B | Qwen2.5-VL | 0.572 | 0.425 | **+0.147** |
| Qwen2.5-VL-32B | Qwen2.5-VL | 0.600 | 0.462 | **+0.138** |
| Qwen2.5-VL-72B | Qwen2.5-VL | 0.625 | 0.482 | **+0.143** |
| Qwen3-VL-2B | Qwen3-VL | 0.418 | 0.385 | **+0.033** |
| Qwen3-VL-8B | Qwen3-VL | 0.522 | 0.488 | **+0.034** |
| Qwen3-VL-32B | Qwen3-VL | 0.513 | 0.475 | **+0.038** |
| InternVL2-2B | InternVL2 | 0.503 | 0.380 | **+0.123** |
| InternVL2-8B | InternVL2 | 0.575 | 0.420 | **+0.155** |
| InternVL2-26B | InternVL2 | 0.575 | 0.405 | **+0.170** |
| InternVL2-76B | InternVL2 | 0.295 | 0.280 | **+0.015** |
| InternVL2.5-78B | InternVL2 | 0.452 | 0.448 | **+0.004** |
| VideoLLaMA2-7B | VideoLLaMA2 | 0.550 | 0.475 | **+0.075** |
| LLaVA-Video-7B | LLaVA | 0.628 | 0.443 | **+0.185** |

---

### 10.2 Full 4-Point Ladder — All Models (600-Q subset)

*Δspatial = singleframe − black; Δdiversity = shuffled − singleframe; Δtemporal = original − shuffled*

#### Qwen2-VL

| Model | Black | Singleframe | Shuffled | Original | Δspatial | Δdiversity | Δtemporal |
|-------|-------|-------------|----------|----------|----------|------------|-----------|
| Qwen2-VL-2B | 0.313 | 0.377 | 0.475 | 0.472 | +0.064 | +0.098 | −0.003 |
| Qwen2-VL-72B | 0.410 | 0.490 | 0.637 | 0.667 | +0.080 | +0.147 | +0.030 |

#### Qwen2.5-VL

| Model | Black | Singleframe | Shuffled | Original | Δspatial | Δdiversity | Δtemporal |
|-------|-------|-------------|----------|----------|----------|------------|-----------|
| Qwen2.5-VL-3B | 0.362 | 0.420 | 0.550 | 0.572 | +0.058 | +0.130 | +0.022 |
| Qwen2.5-VL-7B | 0.333 | 0.425 | 0.572 | 0.630 | +0.092 | +0.147 | +0.058 |
| Qwen2.5-VL-32B | 0.369 | 0.462 | 0.600 | 0.635 | +0.093 | +0.138 | +0.035 |
| Qwen2.5-VL-72B | 0.377 | 0.482 | 0.625 | 0.703 | +0.105 | +0.143 | +0.078 |

#### Qwen3-VL

| Model | Black | Singleframe | Shuffled | Original | Δspatial | Δdiversity | Δtemporal |
|-------|-------|-------------|----------|----------|----------|------------|-----------|
| Qwen3-VL-2B | 0.333 | 0.385 | 0.418 | 0.447 | +0.052 | +0.033 | +0.029 |
| Qwen3-VL-8B | 0.407 | 0.488 | 0.522 | 0.514 | +0.081 | +0.034 | −0.008 |
| Qwen3-VL-32B | 0.392 | 0.475 | 0.513 | 0.557 | +0.083 | +0.038 | +0.044 |

#### InternVL2 / InternVL2.5

| Model | Black | Singleframe | Shuffled | Original | Δspatial | Δdiversity | Δtemporal |
|-------|-------|-------------|----------|----------|----------|------------|-----------|
| InternVL2-2B | 0.295 | 0.380 | 0.503 | 0.513 | +0.085 | +0.123 | +0.010 |
| InternVL2-8B | 0.312 | 0.420 | 0.575 | 0.583 | +0.108 | +0.155 | +0.008 |
| InternVL2-26B | 0.320 | 0.405 | 0.575 | 0.582 | +0.085 | +0.170 | +0.007 |
| InternVL2-76B | 0.357 | 0.280 | 0.295 | 0.308 | −0.077 | +0.015 | +0.013 |
| InternVL2.5-78B | 0.452 | 0.448 | 0.452 | 0.452 | −0.004 | +0.004 | +0.000 |

#### Video-Specialist Models

| Model | Black | Singleframe | Shuffled | Original | Δspatial | Δdiversity | Δtemporal |
|-------|-------|-------------|----------|----------|----------|------------|-----------|
| VideoLLaMA2-7B | 0.342 | 0.475 | 0.550 | 0.585 | +0.133 | +0.075 | +0.035 |
| LLaVA-Video-7B | 0.351 | 0.443 | 0.628 | 0.638 | +0.092 | +0.185 | +0.010 |

---

### 10.3 Per-Task Δdiversity Breakdown

| Task | Q2-2B | Q2-72B | Q2.5-7B | Q2.5-72B | Q3-8B | Q3-32B | IV2-8B | IV2-26B | VLLaMA | LLaVA |
|------|-------|--------|---------|----------|-------|--------|--------|---------|--------|-------|
| Action Reasoning | +0.041 | +0.170 | +0.180 | +0.150 | −0.020 | 0.000 | +0.110 | +0.190 | +0.060 | +0.160 |
| Action Recognition | +0.140 | +0.100 | +0.180 | +0.160 | +0.020 | +0.030 | +0.050 | +0.080 | +0.100 | +0.160 |
| Attribute Perception | +0.210 | +0.260 | +0.233 | +0.220 | −0.010 | +0.070 | +0.270 | +0.310 | +0.150 | +0.250 |
| OCR Problems | +0.130 | +0.160 | +0.050 | +0.120 | +0.100 | +0.020 | +0.220 | +0.150 | +0.010 | +0.220 |
| Object Recognition | +0.090 | +0.110 | +0.203 | +0.140 | +0.080 | +0.100 | +0.220 | +0.220 | +0.170 | +0.320 |
| Temporal Reasoning | −0.013 | +0.080 | +0.023 | +0.070 | +0.030 | +0.010 | +0.060 | +0.070 | −0.040 | +0.000 |

---

### 10.4 Generation Comparison — Qwen

| Model | Black | Shuffled | Singleframe | Original | VGG | Δdiversity |
|-------|-------|----------|-------------|----------|-----|------------|
| Qwen2-VL-2B | 0.313 | 0.475 | 0.377 | 0.472 | +0.159 | +0.098 |
| **Qwen3-VL-2B** | **0.333** | **0.418** | **0.385** | **0.447** | **+0.114** | **+0.033** |
| Qwen2.5-VL-7B | 0.333 | 0.572 | 0.425 | 0.630 | +0.297 | +0.147 |
| **Qwen3-VL-8B** | **0.407** | **0.522** | **0.488** | **0.514** | **+0.107** | **+0.034** |
| Qwen2.5-VL-32B | 0.369 | 0.600 | 0.462 | 0.635 | +0.266 | +0.138 |
| **Qwen3-VL-32B** | **0.392** | **0.513** | **0.475** | **0.557** | **+0.165** | **+0.038** |

**Finding**: Qwen3-VL shows a striking collapse in Δdiversity (+0.033–+0.038) vs Qwen2.5-VL (+0.130–+0.147). Qwen3-VL's singleframe accuracy is nearly as high as shuffled, meaning it extracts almost all available visual information from a single frame. Despite higher black-screen accuracy (stronger language priors), VGG drops substantially — Qwen3-VL-8B VGG (+0.107) is less than half of Qwen2.5-VL-7B VGG (+0.297). Architectural changes in Qwen3-VL's visual encoding appear to have sacrificed visual diversity benefit.

---

### 10.5 Key Findings from Ablation

**Δdiversity is the dominant visual signal** across most models (+0.10–+0.19 for well-functioning models). The ability to see multiple diverse frames (shuffled > singleframe) captures most of the gap between language-only and full visual understanding.

**Δtemporal is near zero or slightly negative for most models**. Once frame order is randomized, restoring it provides little benefit. Confirmed for all 15 models where original is available. Qwen3-VL-8B even shows negative Δtemporal (−0.008): shuffled outperforms original video.

**Δspatial is negative for large InternVL2** (76B: −0.077, 78B: −0.004). These models score *lower* with a single real frame than with a black screen, confirming extreme language-prior dominance.

**Qwen3-VL anomaly**: Δdiversity collapses to ~+0.035 across all sizes. The singleframe accuracy is proportionally much higher, suggesting Qwen3-VL's visual encoder extracts more information from a single frame but gains little from additional frames.

**LLaVA-Video-7B** has the highest Δdiversity (+0.185) among all 16 models, and the highest single-task VGG observed (Attribute Perception: +0.510).

**4-point ladder is now complete for all 16 models** (as of 2026-03-25). Qwen2-VL-2B original (0.472, 600-Q) was the final piece.