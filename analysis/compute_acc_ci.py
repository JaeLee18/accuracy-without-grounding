
# --- VDG portable paths (override via environment variables) ---
import os
VDG_RESULTS_ROOT = os.environ.get("VDG_RESULTS_ROOT", "results")
# ------------------------------------------------------------
import json, numpy as np

q_vme  = json.load(open(VDG_RESULTS_ROOT + '/full_study/qwen2vl_results.json'))
iv_vme = json.load(open(VDG_RESULTS_ROOT + '/full_study/internvl2_results.json'))

iv_vo = {r['question_id']: r for r in iv_vme if r['condition']=='original' and not r.get('error')}
q_vo  = {r['question_id']: r for r in q_vme  if r['condition']=='original' and not r.get('error')}

common = sorted(set(iv_vo) & set(q_vo))
n = len(common)

iv_acc = np.mean([iv_vo[q]['correct'] for q in common])
q_acc  = np.mean([q_vo[q]['correct']  for q in common])
diff   = iv_acc - q_acc

rng = np.random.default_rng(42)
boot_diffs = []
for _ in range(2000):
    idx = rng.integers(0, n, size=n)
    samp = [common[i] for i in idx]
    boot_diffs.append(np.mean([iv_vo[q]['correct'] for q in samp]) - np.mean([q_vo[q]['correct'] for q in samp]))

ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
print(f'n_common = {n}')
print(f'InternVL2 orig = {iv_acc:.4f} ({iv_acc*100:.1f}%)')
print(f'Qwen2-VL orig  = {q_acc:.4f} ({q_acc*100:.1f}%)')
print(f'Diff (IV2-Qwen) = {diff*100:.2f}pp')
print(f'95% bootstrap CI on diff: [{ci_lo*100:.2f}pp, {ci_hi*100:.2f}pp]')
print(f'CI includes zero: {ci_lo <= 0 <= ci_hi}')
