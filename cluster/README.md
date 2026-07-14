# Cluster ablation package

This directory is a standalone subset for the GPU-heavy FPS and scale experiments.

## Setup

```bash
cd cluster
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers accelerate qwen-vl-utils decord imageio-ffmpeg
python -m pip install tqdm numpy scipy pillow bitsandbytes
```

Place the Video-MME sample at `data/full_sample.json` and the referenced MP4 files under
`data/videos/`. The repository includes the sample JSON but not the videos.

## Run

```bash
export DATA_ROOT="$(pwd)/data"
export RESULTS_ROOT="$(pwd)/results"
bash run_all.sh
```

The runners checkpoint incrementally and can be restarted.

## Experiments

| Script | Model | Approximate bf16 VRAM | Purpose |
|---|---|---:|---|
| `scripts/run_fps_ablation.py` | Qwen2-VL-7B | 16 GB | 0.5, 1.0, and 2.0 FPS |
| `scripts/run_qwen2b_videomme.py` | Qwen2-VL-2B | 5 GB | small-scale Video-MME evaluation |
| `scripts/run_internvl2_26b_black.py` | InternVL2-26B | 52 GB | 26B black-screen baseline |

Set `USE_4BIT=1` for the InternVL2-26B runner when bf16 does not fit. Quantized results should not
be interpreted as matched-precision scaling results.

Outputs appear in `results/`. See `TODO_ABLATION_CLUSTER.md` for the original experiment record.
