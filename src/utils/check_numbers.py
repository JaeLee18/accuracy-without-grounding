
# --- VDG portable paths (override via environment variables) ---
import os
VDG_DATA_ROOT = os.environ.get("VDG_DATA_ROOT", "data")
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json
from collections import defaultdict
from scipy import stats
import numpy as np

with open(VDG_RESULTS_ROOT + '/full_study/qwen2vl_results.json') as f:
    main = json.load(f)
with open(VDG_RESULTS_ROOT + '/full_study/qwen2vl_black_results.json') as f:
    black = json.load(f)

CONDITIONS = ['original','crf18','crf23','crf28','crf33','crf38']
TASK_TYPES = ['Attribute Perception','Object Recognition','Action Recognition','OCR Problems','Action Reasoning','Temporal Reasoning']

qid_results = defaultdict(dict)
for r in main:
    if not r.get('error'):
        qid_results[r['question_id']][r['condition']] = r['correct']

complete_qids = {qid for qid, conds in qid_results.items() if len(conds)==6}

with open(VDG_DATA_ROOT + '/videomme_full/full_sample.json') as f:
    samples = {s['question_id']: s for s in json.load(f)}

print(f'Complete questions (all 6 conditions): {len(complete_qids)}/600')
for tt in TASK_TYPES:
    n = sum(1 for qid in complete_qids if samples[qid]['task_type']==tt)
    print(f'  {tt}: {n}/100')

print()
print('='*80)
print('TABLE 1: ACCURACY BY TASK TYPE AND CONDITION')
print('='*80)
print(f'{"Task Type":<25}' + ''.join(f'{c:>9}' for c in CONDITIONS))
print('-'*79)

tt_accs = {}
for tt in TASK_TYPES:
    tt_qids = [qid for qid in complete_qids if samples[qid]['task_type']==tt]
    accs = []
    for cond in CONDITIONS:
        correct = sum(1 for qid in tt_qids if qid_results[qid][cond])
        accs.append(correct/len(tt_qids) if tt_qids else 0)
    tt_accs[tt] = accs
    row = f'{tt:<25}' + ''.join(f'{a:>9.3f}' for a in accs)
    print(row)

# Overall row
print('-'*79)
for cond_idx, cond in enumerate(CONDITIONS):
    vals = [tt_accs[tt][cond_idx] for tt in TASK_TYPES]
    tt_accs.setdefault('__overall__', [])
if '__overall__' not in tt_accs:
    tt_accs['__overall__'] = []
overall = [sum(tt_accs[tt][i] for tt in TASK_TYPES)/6 for i in range(6)]
print(f'{"Overall (macro-avg)":<25}' + ''.join(f'{a:>9.3f}' for a in overall))

print()
print('='*80)
print('TABLE 2: VGG (VISUAL GROUNDING GAP) — sorted by VGG desc')
print('='*80)
black_by_qid = {r['question_id']: r['correct'] for r in black}

print(f'{"Task Type":<25} {"Orig":>7} {"Black":>7} {"VGG":>7} {"CRF38":>7} {"Deg@38":>8}  n_comp')
print('-'*75)

vgg_data = {}
for tt in TASK_TYPES:
    tt_qids_all = [s['question_id'] for s in samples.values() if s['task_type']==tt]
    black_correct = sum(1 for qid in tt_qids_all if black_by_qid.get(qid, False))
    black_acc = black_correct / len(tt_qids_all)
    tt_cq = [qid for qid in complete_qids if samples[qid]['task_type']==tt]
    orig_acc = sum(1 for qid in tt_cq if qid_results[qid]['original']) / len(tt_cq)
    crf38_acc = sum(1 for qid in tt_cq if qid_results[qid]['crf38']) / len(tt_cq)
    vgg = orig_acc - black_acc
    deg38 = crf38_acc - orig_acc
    vgg_data[tt] = {'orig': orig_acc, 'black': black_acc, 'vgg': vgg, 'crf38': crf38_acc, 'deg38': deg38, 'n': len(tt_cq)}

for tt in sorted(TASK_TYPES, key=lambda t: -vgg_data[t]['vgg']):
    d = vgg_data[tt]
    print(f'{tt:<25} {d["orig"]:7.3f} {d["black"]:7.3f} {d["vgg"]:+7.3f} {d["crf38"]:7.3f} {d["deg38"]:+8.3f}  {d["n"]}')

print()
print('='*80)
print('CRF DEGRADATION SLOPES (linregress on CRF 0,18,23,28,33,38 vs accuracy)')
print('='*80)
CRF_VALS = [0, 18, 23, 28, 33, 38]
for tt in TASK_TYPES:
    orig = vgg_data[tt]['orig']
    rest = tt_accs[tt][1:]
    accs = [orig] + rest
    slope, intercept, r, p, se = stats.linregress(CRF_VALS, accs)
    print(f'{tt:<25} slope={slope:+.6f}  p={p:.4f}  R2={r**2:.3f}  CRF38_deg={accs[-1]-accs[0]:+.3f}')

print()
print('='*80)
print('OVERALL SUMMARY')
print('='*80)
orig_avg = sum(vgg_data[tt]['orig'] for tt in TASK_TYPES)/6
black_avg = sum(vgg_data[tt]['black'] for tt in TASK_TYPES)/6
crf38_avg = sum(vgg_data[tt]['crf38'] for tt in TASK_TYPES)/6
print(f'  Macro-avg original:     {orig_avg:.3f}')
print(f'  Macro-avg black screen: {black_avg:.3f}')
print(f'  Macro-avg VGG:          {orig_avg - black_avg:+.3f}')
print(f'  Macro-avg CRF38:        {crf38_avg:.3f}')
print(f'  Macro-avg deg@38:       {crf38_avg - orig_avg:+.3f}')
print()
print('KEY PAPER CLAIM CHECK:')
print(f'  TR orig acc:   {vgg_data["Temporal Reasoning"]["orig"]:.3f}')
print(f'  TR black acc:  {vgg_data["Temporal Reasoning"]["black"]:.3f}')
print(f'  TR VGG:        {vgg_data["Temporal Reasoning"]["vgg"]:+.3f}  (should be LOWEST)')
print(f'  AP VGG:        {vgg_data["Attribute Perception"]["vgg"]:+.3f}  (should be HIGHEST)')
print(f'  VGG rank order: ', end='')
ranked = sorted(TASK_TYPES, key=lambda t: vgg_data[t]['vgg'])
print(' < '.join(t[:4] for t in ranked))
