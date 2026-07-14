
# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, os, subprocess, re
from collections import defaultdict, Counter
import imageio_ffmpeg

with open(VDG_RESULTS_ROOT + '/full_study/qwen2vl_results.json') as f:
    main = json.load(f)
with open(VDG_DATA_ROOT + '/videomme_full/full_sample.json') as f:
    samples = {s['question_id']: s for s in json.load(f)}

# Find questions that failed on all 6 conditions
qid_conds = defaultdict(dict)
for r in main:
    qid_conds[r['question_id']][r['condition']] = r

failed_all6 = []
for qid, conds in qid_conds.items():
    errors = {c: r for c, r in conds.items() if r.get('error')}
    if len(errors) == 6:
        failed_all6.append(samples[qid])

print(f'Total failed-all-6: {len(failed_all6)}')
dur_counts = Counter(s['duration'] for s in failed_all6)
print('Duration breakdown:', dict(dur_counts))
print()

video_dir = VDG_DATA_ROOT + '/videomme_full/videos'
rows = []
for s in failed_all6:
    vpath = os.path.join(video_dir, s['videoID'] + '.mp4')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([ffmpeg, '-i', vpath], capture_output=True, text=True)
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', result.stderr)
    if m:
        h, mn, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
        dur = h*3600 + mn*60 + sec
    else:
        dur = -1
    rows.append((s['videoID'], s['task_type'], s['duration'], dur))

rows.sort(key=lambda x: -x[3])
print(f"{'VideoID':<40} {'Task':<20} {'Cat':<8} {'Sec':>6}  frames@0.25  frames@0.1")
print('-'*90)
for vid, tt, cat, sec in rows:
    f25 = int(sec * 0.25) if sec > 0 else -1
    f10 = int(sec * 0.1) if sec > 0 else -1
    print(f'{vid:<40} {tt:<20} {cat:<8} {sec:>6.0f}  {f25:>11}  {f10:>9}')

secs = [x[3] for x in rows if x[3] > 0]
secs_sorted = sorted(secs)
print()
print(f'Duration (sec): min={min(secs):.0f}  median={secs_sorted[len(secs_sorted)//2]:.0f}  max={max(secs):.0f}')
print(f'Frames @0.25:   min={int(min(secs)*0.25)}  median={int(secs_sorted[len(secs_sorted)//2]*0.25)}  max={int(max(secs)*0.25)}')
print(f'Frames @0.10:   min={int(min(secs)*0.1)}   median={int(secs_sorted[len(secs_sorted)//2]*0.1)}   max={int(max(secs)*0.1)}')

# Task type breakdown
print()
tt_counts = Counter(s['task_type'] for s in failed_all6)
print('By task type:')
for tt, n in tt_counts.most_common():
    print(f'  {tt}: {n}')
