# Ablation Experiments — GPU Cluster Instructions

## Overview

Three experiments needed to push paper from 8.5/10 to 9+/10.
All scripts are in `<repo-root>/experiments/`.

---

## Experiment 1: FPS Ablation (Temporal Reasoning)

**Purpose**: Resolve whether Temporal Reasoning VGG ≈ 0 is real (language-dominated questions) or an artifact of 0.25 FPS frame sparsity.

**What to run**: Qwen2-VL-7B-Instruct on 100 Temporal Reasoning questions at **0.5 FPS, 1.0 FPS, 2.0 FPS** — both original video and black screen at each FPS level.

**Script**: `run_fps_ablation.py`
- Change `FPS_LEVELS = [0.5, 1.0, 2.0]`
- Change `MAX_PIXELS = 256 * 256` (restore full resolution on big GPU)
- Total items: 100 questions × 3 FPS × 2 conditions = **600 inference calls**
- Estimated time: ~1-2 hours on A100 80GB

**Data needed**:
- `data/videomme_full/full_sample.json` — 600 Video-MME questions (script filters to Temporal Reasoning)
- `data/videomme_full/videos/` — original videos (only ~98 unique videos for Temporal Reasoning)

**Output**: `results/experiments/fps_ablation_results.json`

**What we're looking for**:
| FPS | Expected if real | Expected if artifact |
|-----|-----------------|---------------------|
| 0.25 (baseline) | VGG ≈ 0 | VGG ≈ 0 |
| 0.5 | VGG ≈ 0 | VGG ≈ 0.05 |
| 1.0 | VGG ≈ 0 | VGG ≈ 0.10+ |
| 2.0 | VGG ≈ 0 | VGG ≈ 0.15+ |

If VGG stays ≈ 0 across all FPS → **Temporal Reasoning is genuinely language-dominated** (paper's claim vindicated).
If VGG rises with FPS → **paper needs to revise the claim** (honest either way).

**Model download**:
```bash
# ~14GB
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct
```

---

## Experiment 2: Qwen2-VL-2B Scale Test (Downscale)

**Purpose**: Test whether the VGG task-type spectrum is preserved at 3.5× smaller model scale. If yes → VGG is a benchmark property, not a model-scale property.

**What to run**: Qwen2-VL-2B-Instruct on full Video-MME (600 questions), original + black screen.

**Script**: `run_qwen2b_videomme.py`
- Total items: 600 questions × 2 conditions = **1200 inference calls**
- Estimated time: ~1-2 hours (2B is fast)
- VRAM: ~4-5 GB in fp16 (runs on any GPU)

**Data needed**:
- `data/videomme_full/full_sample.json`
- `data/videomme_full/videos/` — all 407 unique videos (~13.6 GB)

**Output**: `results/experiments/qwen2b_videomme_results.json`

**What we're looking for**:
- Same VGG task-type ordering as 7B (Attr.Perc > Obj.Rec > OCR > Act.Rec > Act.Reas > Temp.Reas)
- Overall VGG magnitude may differ (2B likely lower accuracy, possibly lower VGG)
- Spearman rank correlation between 2B and 7B task-type VGG ordering

**Model download**:
```bash
# ~4GB
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct
```

---

## Experiment 3: InternVL2-26B Black-Screen Only (Upscale)

**Purpose**: Test the scaling prediction — larger LM backbone → higher black-screen accuracy → lower VGG. Only need black-screen condition (cheap).

**What to run**: InternVL2-26B on Video-MME black screen only (600 questions).

**Script**: `run_internvl2_26b_black.py`
- Uses 4-bit quantization (BitsAndBytesConfig) — ~13-15 GB VRAM
- On A100 80GB: can run in fp16/bf16 without quantization — change the script
- Total items: **600 inference calls** (black screen only)
- Estimated time: ~2-4 hours

**Data needed**:
- `data/videomme_full/full_sample.json`
- `data/videomme_full/videos/` — only needed for duration measurement (to create matching black videos)

**Output**: `results/experiments/internvl2_26b_black_results.json`

**What we're looking for**:
- Compare 26B black-screen accuracy to 8B black-screen accuracy per task type
- If 26B black > 8B black → confirms scaling prediction (stronger priors at scale)
- Expected: 26B black ≈ 45-55% vs 8B black ≈ 39%

**Model download**:
```bash
# ~52GB in fp16
huggingface-cli download OpenGVLab/InternVL2-26B --revision main
```

**For fp16 on A100** — modify `run_internvl2_26b_black.py` load_model():
```python
def load_model():
    # Remove BitsAndBytesConfig, load in bf16 directly
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False,
    )
    return model, tokenizer
```

---

## Data to Transfer to Cluster

```
data/videomme_full/
├── full_sample.json          (~1 MB)
└── videos/                   (~13.6 GB, 407 mp4 files)

<repo-root>/experiments/
├── run_fps_ablation.py
├── run_qwen2b_videomme.py
└── run_internvl2_26b_black.py
```

Total data transfer: **~13.6 GB** (videos + scripts + sample JSON).

---

## Python Dependencies

```bash
conda create -n ablation python=3.10
conda activate ablation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate qwen-vl-utils decord imageio-ffmpeg
pip install bitsandbytes  # only needed for 4-bit InternVL2-26B
pip install tqdm numpy scipy pillow
```

---

## After Running — What to Bring Back

1. `results/experiments/fps_ablation_results.json`
2. `results/experiments/qwen2b_videomme_results.json`
3. `results/experiments/internvl2_26b_black_results.json`

I will then:
- Analyze all three and compute VGG tables
- Integrate results into the paper
- Re-run the reviewer loop to push toward 9/10

---

## Priority Order

1. **FPS Ablation** (highest impact — resolves biggest confound, ~600 calls)
2. **Qwen2-VL-2B** (easiest — tiny model, 1200 calls)
3. **InternVL2-26B black** (most interesting — tests scaling, 600 calls)
