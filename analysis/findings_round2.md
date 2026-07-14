# Statistical Findings — Round 2
Generated: 2026-03-18
Models: Qwen2-VL-7B, LLaVA-Video-7B, InternVL2-8B
Benchmarks: Video-MME (547 common Qs, 3 models), MVBench (462q × 3 conditions)
Targeting: Reviewer's 6 directions from Round 1

## Reviewer verdict: 3/14 findings rated 🌟 WOW

---

## 🌟 WOW Findings

**NEW-4** — LLaVA's 39% cross-benchmark VGG divergence is mechanistically explained: Video-MME VGG=0.305, MVBench VGG=0.186 (11.9pp absolute, 39% relative). Two MVBench task types — scene_transition (86% black-screen acc) and unexpected_action (86%) — inflate MVBench black-screen performance. Without these, MVBench VGG ≈ 0.25, nearly identical to Video-MME. Qwen2-VL shows only 4.3pp divergence.

**NEW-7** — action_antonym +48pp CRF anomaly (F19 from Round 1) is a data artifact. 30/50 CRF38 samples had inference errors (model failed, prediction=None), treated as incorrect. With valid-only predictions (20/50): Qwen CRF38=0.900 vs orig=0.840, InternVL2 CRF38=0.950 vs orig=0.880 — STABLE or slightly better. The artifact mechanism: error-rate confound (18/50=0.36 vs 18/20=0.90 = 54pp spurious collapse).

**NEW-10** — "Semantic label-blindness" explains above-chance black-screen performance better than GT skew: scene_transition and unexpected_action at 86% black accuracy because question text semantically implies the answer (e.g., yes/no questions where Yes dominates). This is a question-design flaw, not a statistical artifact. Directly actionable for benchmark construction.

---

## ✅ GOOD Findings

**NEW-1** — Hard-core questions (VGG>0 for all 3 models): 62/547 = 11.3%. Task-type distribution significantly non-uniform: chi2=18.21, p=0.0027. Attribute Perception: 19% HC rate, Object Recognition: 18%, Temporal Reasoning: 1.2%.

**NEW-3** — 48.3% of questions have MIXED VGG sign (at least 1 model positive, at least 1 non-positive). Models agree on task-level video necessity but disagree on which specific questions require vision. Question-level VGG is model-dependent, not a property of the question alone.

**NEW-5** — Qwen2-VL cross-benchmark VGG divergence is only 4.3pp (Video-MME=0.250, MVBench=0.208), confirming the LLaVA divergence is architecture/strategy-specific, not a general benchmark artifact.

**NEW-6** — LLaVA's black-screen letter distribution differs significantly between benchmarks (chi2=30.55, p<0.0001): Video-MME A=26.3%/B=29.2%/C=18.7%/D=25.8%; MVBench A=34.4%/B=26.2%/C=24.0%/D=13.2%. Benchmark-specific letter bias interacts with GT distribution to produce VGG divergence.

**NEW-8** — action_antonym artifact arithmetic: 18 correct / 50 total (with errors=wrong) = 0.36 vs orig=0.84 → spurious 48pp collapse. With 20 valid samples: 18/20=0.90. Error rate confound, not video quality effect.

**NEW-9** — GT label skew does NOT predict black-screen performance (Spearman r=0.257, p=0.623). All task types have near-maximum GT entropy (1.339–1.386 vs max 1.386). Temporal Reasoning's above-chance black performance is not caused by GT letter skew.

**NEW-12** — Model asymmetry in language prior exploitation (post-Bonferroni): Qwen 3/6 significant, LLaVA 3/6, InternVL2 only 1/6. InternVL2 uses language priors far less aggressively — consistent with lower aggregate black-screen accuracy.

**NEW-13** — Object Recognition is the only task type where ALL THREE models are non-significant after Bonferroni correction. Language priors provide essentially zero advantage for specific object identification.

## ⚠️ MARGINAL

**NEW-2** — HC questions account for 40.4% of total VGG. Near-tautological given HC definition (all 3 models flip correct→wrong). One-sentence mention sufficient.

**NEW-11** — 7/18 cells significant after Bonferroni. Useful rigor check but buries the lead (InternVL2 has 0 non-perceptual significant cells).

**NEW-14** — Task-type agreement hierarchy (63.7% down to 38.3%). Task type is a weak predictor of question-level VGG sign. GT letter D vs A HC-rate difference (17.5% vs 7.8%) underpowered without sample sizes.

---

## Reviewer's 4 Remaining Gaps

**Gap A (Critical):** Audit error rates across all model×task×CRF combinations — confirm action_antonym-style error confound doesn't affect other cells.

**Gap B (Moderate):** GT letter D vs A HC-rate — provide sample sizes (n_D=120, n_A=128) and chi-square test.

**Gap C (Moderate):** What predicts mixed-sign vs HC vs non-positive category at question level? Logistic regression on question properties.

**Gap D (Minor):** Directly verify GT label distribution for scene_transition and unexpected_action (the 76% Yes base rate claim).
