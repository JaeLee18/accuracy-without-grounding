#!/bin/bash
# Wait for compression to finish, then run inference + analysis sequentially.
# Paths are overridable via environment variables (see repo README).
VDG_DATA_ROOT="${VDG_DATA_ROOT:-data}"
CRF33_DIR="$VDG_DATA_ROOT/videomme_full/crf33"
CRF38_DIR="$VDG_DATA_ROOT/videomme_full/crf38"
TARGET=407
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[wait_then_infer] Waiting for compression to finish (target: $TARGET files each)..."

while true; do
    n33=$(ls "$CRF33_DIR" 2>/dev/null | wc -l)
    n38=$(ls "$CRF38_DIR" 2>/dev/null | wc -l)
    echo "  crf33: $n33/$TARGET  crf38: $n38/$TARGET  ($(date '+%H:%M:%S'))"
    if [ "$n33" -ge "$TARGET" ] && [ "$n38" -ge "$TARGET" ]; then
        echo "[wait_then_infer] Compression complete. Starting inference pipeline..."
        break
    fi
    sleep 30
done

python "$HERE/run_all.py" qwen llava analyze
echo "[wait_then_infer] Pipeline finished at $(date)."
