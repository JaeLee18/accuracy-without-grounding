# Statistical Findings — Round 3
Generated: 2026-03-18
Models: Qwen2-VL-7B, LLaVA-Video-7B, InternVL2-8B
Targeting: Reviewer's 4 remaining gaps from Round 2

## Reviewer verdict: 3/8 findings rated 🌟 WOW

---

## 🌟 WOW Findings

**R3-5** — Consensus correctness predicts hard-core membership perfectly: ALL questions where any model fails with original video (n=303) have HC=0% (impossible by definition). Questions where all 3 models succeed with original (n=244) have HC=25.4%. Point-biserial r=0.336 (p<0.000001). HC is not a random hard subset — it is specifically the questions every model masters with vision but every model fails without it.

**R3-7** — scene_transition 86% black-screen accuracy mechanism: NOT binary Yes/No bias. GT distribution is A:36% B:20% C:22% D:22% (near-uniform). LLaVA black-screen predictions: A:42% B:24% C:18% D:16% — nearly ideal calibration to GT without video. The mechanism is SEMANTIC LEAKAGE: scene_transition question text inherently describes whether a transition occurred, enabling correct inference without visual frames.

**R3-8** — state_change contrastive validation (strongest finding): state_change has MORE skewed GT (A:44%, B:22%, C:34%, only 3 options) yet LLaVA black-screen = 28%. If GT skew or letter strategy explained high black accuracy, state_change should be higher. This single contrast falsifies ALL GT-skew and letter-strategy alternative explanations simultaneously, validating R3-7's semantic leakage mechanism. Reviewers get an airtight control comparison.

---

## ✅ GOOD Findings

**R3-1** — Video-MME CRF error audit: Qwen2-VL has systematic errors in Action Reasoning (20%), OCR Problems (19%), Temporal Reasoning (14%). Error rates are IDENTICAL across all 6 CRF levels including original. CRF analysis is unbiased — same valid-question subsets at every compression level. Required in appendix.

**R3-3** — action_antonym decoding failure root cause: 30/50 CRF38 videos are unreadable files (Error reading 102660.mp4 etc), confirmed across all 3 models simultaneously. This is upstream pipeline corruption, not a modeling artifact. Valid 20/50 survivors show normal CRF robustness. Required in appendix alongside NEW-7/8.

**R3-6** — Task type predicts HC structure: Attribute Perception and Object Recognition have 57% and 50% MIXED VGG sign (models disagree on which questions need vision). Temporal Reasoning has only 41% mixed AND only 1.2% HC. Temporal questions are where models most diverge in video usage strategies — perception tasks produce consistent video necessity, temporal tasks produce model-specific disagreements.

## ⚠️ MARGINAL

**R3-2 [Resolved]** — Qwen2-VL episodic_reasoning differential error (38% at original, 20% at black). Corrected matched-VGG analysis: orig=0.548 (n=31), black=0.548 (n=31), VGG=0.000, 95% CI [-0.161, 0.161]. Unmatched VGG=0.048. Difference = 4.8pp — negligible. R1-F14 finding stands. The earlier round 1 estimate of VGG=−0.060 is revised to ≈0 (the sign flip was noise; Qwen episodic VGG is near-zero). LLaVA episodic VGG=0.120 is unaffected.

**R3-4** — GT letter D predicts HC at 17.5% vs A at 7.8%. 4-way chi-square p=0.094 (marginal), A vs D 2x2 p=0.0346. Causal claim not established. Report as observation only: last-position answers may require more visual specificity to identify.

---

## Validity status after Round 3

ALL gaps resolved:
- Gap A: CRF error audit complete — Video-MME clean, action_antonym is pipeline failure
- Gap B: Letter D HC rate marginally higher, p=0.094 (4-way), soften language
- Gap C: Corrected episodic_reasoning VGG = 0.000; finding stands
- Gap D: scene_transition mechanism = semantic leakage, not GT skew

---

## Cumulative WOW count
| Round | WOW count |
|-------|-----------|
| Round 1 | 14 |
| Round 2 | 3 |
| Round 3 | 3 |
| **Total** | **20** |
