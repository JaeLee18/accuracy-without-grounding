"""
Master runner for full study. Execute steps sequentially.
Usage: conda run -n compression_pilot python run_all.py [step]
Steps: sample, extract, compress, flow, qwen, llava, analyze, all
"""
import subprocess
import sys
import time
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

STEPS = {
    "sample": "sample_questions_full.py",
    "extract": "extract_videos_full.py",
    "compress": "compress_videos_full.py",
    "flow": "compute_optical_flow.py",
    "qwen": "run_inference_qwen.py",
    "llava": "run_inference_llava.py",
    "analyze": "analyze_full.py",
}

ALL_ORDER = ["sample", "extract", "compress", "flow", "qwen", "llava", "analyze"]

def run_step(name):
    script = os.path.join(SCRIPTS_DIR, STEPS[name])
    print(f"\n{'='*60}")
    print(f"  STEP: {name} — {STEPS[name]}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([PYTHON, script])
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {name} completed in {elapsed:.0f}s")
    return result.returncode == 0

def main():
    steps = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    if "all" in steps:
        steps = ALL_ORDER

    for step in steps:
        if step not in STEPS:
            print(f"Unknown step: {step}")
            print(f"Available: {', '.join(STEPS.keys())}, all")
            sys.exit(1)

    for step in steps:
        if not run_step(step):
            print(f"\nStep '{step}' failed. Stopping.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  ALL STEPS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
