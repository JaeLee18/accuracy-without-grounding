"""
Sequential inference pipeline: runs all 3 models on EgoSchema, one at a time.
Each model is run twice: once after the main pass (to catch any missing-video errors
that get resolved once all 500 videos are present).

Run this AFTER Qwen's first pass is complete.
Usage: conda run -n compression_pilot python run_all_models_pipeline.py
"""
import subprocess, sys, os, json

# --- VDG portable paths (override via environment variables) ---
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
HERE = os.path.dirname(os.path.abspath(__file__))  # this script's directory
# ------------------------------------------------------------

SCRIPTS = [
    os.path.join(HERE, "run_inference_qwen_egoschema.py"),
    os.path.join(HERE, "run_inference_llava_egoschema.py"),
    os.path.join(HERE, "run_inference_internvl2_egoschema.py"),
]
RESULTS = [
    VDG_RESULTS_ROOT + "/egoschema/qwen2vl_egoschema_results.json",
    VDG_RESULTS_ROOT + "/egoschema/llava_egoschema_results.json",
    VDG_RESULTS_ROOT + "/egoschema/internvl2_egoschema_results.json",
]
PYTHON = sys.executable


def count_errors(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        data = json.load(f)
    return sum(1 for r in data if r.get("error"))


def run(script):
    print(f"\n{'='*60}")
    print(f"Running: {os.path.basename(script)}")
    print(f"{'='*60}")
    result = subprocess.run([PYTHON, script], capture_output=False)
    return result.returncode


for script, result_path in zip(SCRIPTS, RESULTS):
    # Run once (handles all available videos)
    rc = run(script)
    # If there were errors, run again (fills in videos that arrived from chunk 5)
    err_count = count_errors(result_path)
    if err_count > 0:
        print(f"\n{err_count} errors in {os.path.basename(result_path)} — re-running to fill in missing videos ...")
        run(script)
    err_after = count_errors(result_path)
    if err_after > 0:
        print(f"  Still {err_after} errors after retry — check if all videos are downloaded.")

print("\n" + "="*60)
print("All models complete. Running analysis ...")
subprocess.run([PYTHON, os.path.join(HERE, "analyze_egoschema.py")])
print("\nDone. Check " + VDG_RESULTS_ROOT + "/egoschema/egoschema_summary.json for results.")
