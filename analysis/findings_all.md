# All Statistical Findings — Consolidated
Generated: 2026-03-18
Multi-agent loop: Stats agent × ACM MM reviewer × 3 rounds
Total: 20 🌟 WOW findings

---

# ROUND 1 — 14 WOW findings

## Core VGG Findings

**F1** 🌟 — Median per-question VGG = 0; only 10.3% of questions show positive VGG across all 3 models simultaneously. The aggregate VGG is driven by a hard-core minority.

**F2** ✅ — Bootstrap 95% CIs on VGG (all positive): Qwen [0.221,0.280], LLaVA [0.273,0.339], InternVL2 [0.138,0.197].

**F3** 🌟 — McNemar dissociation: two models with statistically identical original-video accuracy (InternVL2 vs Qwen2-VL, p=0.536) have significantly different black-screen profiles (p=0.0014). Language prior exploitation is separable from video capability.

**F4** ✅ — Cohen's kappa on black screen (κ=0.32–0.43): moderate agreement across model pairs.

**F5** 🌟 — Destructive ratio: removing video is 4–7× more likely to break a correct answer than fix a wrong one. LLaVA 7.3:1, InternVL2 5.7:1, Qwen 4.25:1. LLaVA repels from C (−14.2 pp) on black screen.

**F7** ✅ — 89–92% question stability across all CRF levels.

## Language Prior Saturation

**F8** 🌟 — Language prior saturation: Qwen achieves 89.4% on text-solvable questions without video (+258% above chance). Visually necessary questions reach only 42–53% even with video. The benchmark conflates saturated and impossible.

**F10** ✅ — Cross-model error correlation r=0.42–0.44.

**F11** 🌟 — Shared wrong consensus: all 3 models agree on the same wrong letter on 19.8% of questions on black screen. Ground truth concentrates in C/D while models collectively pick A/B — dataset-level positional bias.

## Task-Type VGG Spectrum

**F12** 🌟 — Task-type VGG spectrum: Attribute Perception VGG=0.413; Temporal Reasoning VGG=0.050 (CI includes zero for all 3 models). Ranking consistent across all 3 architectures.

**F13** ✅ — Cross-benchmark VGG stability (conditional WOW): Qwen VGG stable ±0.4% across Video-MME and MVBench. Upgrades to WOW if mechanism is explained (explained in Round 2).

**F14** 🌟 — Near-zero VGG for Qwen2-VL on episodic_reasoning (corrected: VGG=0.000, 95% CI [−0.161, 0.161]). LLaVA achieves +34 pp above 5-option chance (54%) on black screen for episodic questions. [Updated from R3-2: original −0.060 was unmatched estimate; corrected = 0.000]

**F15** 🌟 — Calibration: Qwen's black-screen distribution statistically matches GT letter distribution (p=0.230). InternVL2 strongly diverges (p<0.0001). Part of Qwen's language prior "strength" is accidental positional calibration.

**F16** 🌟 — Temporal Reasoning = text comprehension: VGG indistinguishable from zero for all 3 models (CIs include zero, p=0.06–0.69), yet black screen significantly above chance (p<0.001) due to GT label skew (B=38%).

**F17** 🌟 — Scale-dependent VGG dissociation: task-level VGG cross-model r=0.809; question-level VGG cross-model r=0.274–0.304. Models agree on which tasks need vision but disagree on which specific questions do.

## Cross-Benchmark and CRF

**F18** ✅ — Black screen correct predicts original correct (monotone across all 3 models).

**F19** 🌟 — Cross-benchmark CRF: MVBench CRF degradation is 7–16× larger than Video-MME. [action_antonym anomaly explained as pipeline artifact in Round 2/3]

**F20** ✅ — Qwen null predictions suppress VGG by 4.5 pp.

**F23** 🌟 — Cancellation mechanism: flat CRF curves arise from bidirectional answer flips (~10% flip rate), not true robustness. InternVL2 gains +3 net correct answers at CRF38.

**F24** 🌟 — Exact chance floor: InternVL2 Object Recognition on black screen = exactly 25.0% (theoretical chance). The only cell of 18 model×task combinations at true chance; all others significantly above.

---

# ROUND 2 — 3 WOW findings

**NEW-4** 🌟 — LLaVA's 39% cross-benchmark VGG divergence explained: two MVBench task types (scene_transition + unexpected_action, both at 86% black accuracy) inflate MVBench black-screen performance. Without them, MVBench VGG ≈ 0.25, nearly identical to Video-MME. Qwen shows only 4.3pp divergence.

**NEW-7** 🌟 — action_antonym +48pp CRF anomaly (F19) is a data artifact: 30/50 CRF38 videos have corrupted files (decoding failures), treated as incorrect predictions. With valid 20/50 samples: Qwen 0.84→0.90, InternVL2 0.88→0.95 — normal robustness. Error confound created spurious 48pp collapse.

**NEW-10** 🌟 — Semantic label-blindness: scene_transition and unexpected_action (86% black accuracy) are answerable from question text alone because the question structure reveals the answer. Not GT skew or letter guessing — genuine language-based scene inference.

---

# ROUND 3 — 3 WOW findings

**R3-5** 🌟 — Consensus correctness perfectly predicts HC membership: n_correct_original=3 → HC=25.4%; n_correct_original<3 → HC=0.0%. Point-biserial r=0.336 (p<0.000001). Hard-core questions are specifically those all models master with vision but all models fail without.

**R3-7** 🌟 — scene_transition mechanism (semantic leakage, not letter bias): GT distribution A:36% B:20% C:22% D:22% (near-uniform). LLaVA predictions: A:42% B:24% C:18% D:16% — near-ideal GT calibration from text alone. Question text inherently describes whether transition occurred.

**R3-8** 🌟 — state_change contrastive falsification: state_change has MORE GT skew (A:44%) yet LLaVA black = 28%. Simultaneously falsifies GT-skew AND letter-strategy alternative explanations for scene_transition's 86% black accuracy. Single best methodological control in the paper.

---

# Summary Table

| Category | Findings |
|----------|----------|
| 🌟 WOW (main text) | F1, F3, F5, F8, F11, F12, F14, F15, F16, F17, F19, F23, F24, F13(upgraded), NEW-4, NEW-7, NEW-10, R3-5, R3-7, R3-8 |
| ✅ GOOD (supporting) | F2, F4, F7, F10, F18, F20, NEW-1, NEW-3, NEW-5, NEW-6, NEW-8, NEW-9, NEW-12, NEW-13, R3-1, R3-3, R3-6 |
| ⚠️ MARGINAL | F6, F9, F22, F25, NEW-2, NEW-11, NEW-14, R3-2, R3-4 |
| Total WOW | **23** |

---

# ROUND 4 — 3 WOW findings (InternVL2 MVBench complete)

**R4-2** 🌟 — MVBench McNemar dissociation replication: InternVL2 significantly better on original (p=0.0003 vs Qwen, p=0.0006 vs LLaVA), but ALL pairs statistically identical on black screen (p=0.53–0.94). F3 dissociation from Video-MME replicated independently on MVBench.

**R4-6** 🌟 — action_prediction: only task where language priors universally fail for all 3 models (Qwen≈chance, LLaVA below chance, InternVL2 ns after Bonferroni). Empirical lower bound on visual dependency. Clearest measure of genuine video necessity in the benchmark.

**R4-8** 🌟 — object_existence black-screen convergence: 37pp spread in original accuracy (IV2=100%, Qwen=84.8%, LLaVA=63.0%) collapses to 45.7–47.8% on black screen for all 3 models. Black-screen floor is architecture-independent, determined by question structure. Strongest validity argument for VGG metric.
