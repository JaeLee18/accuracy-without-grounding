#!/bin/bash
# Wait for Qwen main inference to finish (3600 entries), then run black screen baseline.
# Paths are overridable via environment variables (see repo README).
VDG_RESULTS_ROOT="${VDG_RESULTS_ROOT:-results}"
QWEN_RESULTS="$VDG_RESULTS_ROOT/full_study/qwen2vl_results.json"
TARGET=3600
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[wait_then_black] Waiting for Qwen to reach $TARGET entries..."
while true; do
    n=$(python -c "import json; print(len(json.load(open('$QWEN_RESULTS'))))" 2>/dev/null || echo 0)
    echo "  Qwen entries: $n/$TARGET  ($(date '+%H:%M:%S'))"
    if [ "$n" -ge "$TARGET" ]; then
        echo "[wait_then_black] Qwen complete. Starting black screen baseline..."
        break
    fi
    sleep 60
done

python "$HERE/run_inference_black.py"
echo "[wait_then_black] Black screen baseline done at $(date)."
