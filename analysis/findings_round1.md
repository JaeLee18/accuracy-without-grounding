# Statistical Findings — Round 1
Generated: 2026-03-18
Models: Qwen2-VL-7B, LLaVA-Video-7B, InternVL2-8B
Benchmarks: Video-MME (600q × 6 CRF), MVBench (462q × 3 conditions)

## Reviewer verdict: 14/25 findings rated 🌟 WOW

---

## 🌟 WOW Findings

**F1** — Median per-question VGG = 0; only 10.3% of questions show positive VGG across all 3 models simultaneously. The aggregate VGG is driven by a hard core minority.

**F3** — McNemar dissociation: two models with statistically identical original-video accuracy (InternVL2 vs Qwen2-VL, p=0.536) have significantly different black-screen profiles (p=0.0014). Language prior exploitation is separable from video capability.

**F5** — Destructive ratio: removing video from a correct answer is 4–7× more likely to break it than fix a wrong one. LLaVA 7.3:1, InternVL2 5.7:1, Qwen 4.25:1. LLaVA repels from C (−14.2 pp) on black screen.

**F8** — Language prior saturation: Qwen achieves 89.4% on text-solvable questions without video (+258% above chance). Visually necessary questions reach only 42–53% even with video. The benchmark conflates saturated and impossible.

**F11** — Shared wrong consensus: all 3 models agree on the same wrong letter on 19.8% of questions on black screen. Ground truth concentrates in C/D while models collectively pick A/B — dataset-level positional bias.

**F12** — Task-type VGG spectrum: Attribute Perception VGG=0.413; Temporal Reasoning VGG=0.050 (CI includes zero for all 3 models). Ranking consistent across all 3 architectures.

**F14** — Negative VGG: video hurts Qwen2-VL on episodic_reasoning (VGG=−0.060). LLaVA achieves +34 pp above 5-option chance (54%) on black screen for episodic questions.

**F15** — Calibration bomb: Qwen's black-screen distribution statistically matches GT letter distribution (p=0.230, not significant). InternVL2 strongly diverges (p<0.0001). Part of Qwen's language prior "strength" is accidental positional calibration.

**F16** — Temporal Reasoning = text comprehension: VGG indistinguishable from zero for all 3 models (CIs include zero, p=0.06–0.69), yet black screen significantly above chance (p<0.001) due to GT label skew (B=38%).

**F17** — Scale-dependent VGG dissociation: task-level VGG cross-model r=0.809; question-level VGG cross-model r=0.274–0.304. Models agree on which tasks need vision but disagree on which specific questions do.

**F19** — Cross-benchmark CRF: MVBench CRF degradation is 7–16× larger than Video-MME. action_antonym collapses 48 pp at CRF38. CRF robustness is benchmark-specific, not a universal property.

**F23** — Cancellation mechanism: flat CRF curves arise from bidirectional answer flips (~10% flip rate), not true robustness. InternVL2 gains +3 net correct answers at CRF38.

**F24** — Exact chance floor: InternVL2 Object Recognition on black screen = exactly 25.0% (theoretical chance). The only cell of 18 model×task combinations at true chance; all others significantly above.

**F13** (conditional WOW) — Qwen VGG stable ±0.4% across Video-MME and MVBench; LLaVA diverges 39%. Upgrades to WOW if mechanism is explained.

---

## ✅ GOOD Findings (supporting, not headline)
F2: Bootstrap CIs on VGG (all positive)
F4: Cohen's kappa on black screen (κ=0.32–0.43)
F7: 89–92% question stability across all CRF levels
F10: Cross-model error correlation r=0.42–0.44
F13: Cross-benchmark VGG stability (conditional)
F18: Black screen correct predicts original correct (monotone)
F20: Qwen null predictions suppress VGG by 4.5 pp

## ⚠️ MARGINAL
F6: Kendall tau CRF monotonicity
F9: Prediction entropy
F22: Four-way regime / language flukes
F25: Task-mode letter strategy (redundant with F15)

## Merged
F21 → merge into F5 (same destructive ratio data)

---

## Reviewer's 6 Directions for Round 2
1. Characterize the 10.3% hard-core questions — task concentration, fraction of total VGG
2. Explain LLaVA's 39% cross-benchmark VGG divergence mechanistically
3. Explain +48 pp action_antonym CRF anomaly
4. Predict question-level VGG agreement from question properties
5. GT label skew as predictor of above-chance black-screen performance
6. Bonferroni-corrected chance test for all 18 model×task cells
