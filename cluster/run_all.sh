#!/bin/bash
# Run all 3 ablation experiments sequentially.
# Override DATA_ROOT and RESULTS_ROOT when data lives outside this directory.
#
# Usage:
#   cd cluster
#   bash run_all.sh

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_ROOT="${DATA_ROOT:-$HERE/data}"
export RESULTS_ROOT="${RESULTS_ROOT:-$HERE/results}"
mkdir -p "$RESULTS_ROOT"

echo "============================================"
echo "DATA_ROOT:    $DATA_ROOT"
echo "RESULTS_ROOT: $RESULTS_ROOT"
echo "============================================"

# Check videos exist
n_videos=$(ls "$DATA_ROOT/videos/"*.mp4 2>/dev/null | wc -l)
echo "Found $n_videos videos in $DATA_ROOT/videos/"
if [ "$n_videos" -lt 50 ]; then
    echo "ERROR: Expected 407 videos. Did you copy the videos/ folder?"
    exit 1
fi

echo ""
echo "========== EXPERIMENT 1: FPS Ablation =========="
echo "Model: Qwen2-VL-7B | 100 questions x 3 FPS x 2 conditions = 600 items"
python "$HERE/scripts/run_fps_ablation.py"

echo ""
echo "========== EXPERIMENT 2: Qwen2-VL-2B Scale Test =========="
echo "Model: Qwen2-VL-2B | 600 questions x 2 conditions = 1200 items"
python "$HERE/scripts/run_qwen2b_videomme.py"

echo ""
echo "========== EXPERIMENT 3: InternVL2-26B Black Screen =========="
echo "Model: InternVL2-26B | 600 questions x 1 condition = 600 items"
# Set USE_4BIT=1 if your GPU has < 40GB VRAM
export USE_4BIT="${USE_4BIT:-0}"
python "$HERE/scripts/run_internvl2_26b_black.py"

echo ""
echo "========== ALL EXPERIMENTS COMPLETE =========="
echo "Results in: $RESULTS_ROOT/"
ls -la "$RESULTS_ROOT/"*.json
