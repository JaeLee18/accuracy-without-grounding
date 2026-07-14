# Statistical Findings — Round 4
Generated: 2026-03-18
Models: All 3 (Qwen2-VL-7B, LLaVA-Video-7B, InternVL2-8B) — InternVL2 MVBench complete
Analyses: Video-MME 3-model joint, MVBench McNemar, cross-benchmark divergence, destructive ratio, Bonferroni MVBench, Cohen's kappa

## Reviewer verdict: 3/8 findings rated 🌟 WOW

---

## 🌟 WOW Findings

**R4-2** — MVBench McNemar dissociation replication: InternVL2 significantly outperforms Qwen (p=0.0003) and LLaVA (p=0.0006) on MVBench ORIGINAL accuracy. On MVBench BLACK SCREEN: all three pairs statistically identical (p=0.53–0.94). Superior video understanding does not transfer to language-prior conditions. This is the F3 Video-MME dissociation replicated on an independent benchmark with different question types. Anchor paragraph for Results §3.2: "The dissociation is not benchmark-specific."

**R4-6** — action_prediction is the only MVBench task where language priors universally fail: Qwen=0.240 (≈chance), LLaVA=0.200 (below chance), InternVL2=0.380 (p_bonf=0.78, ns). All three architecturally distinct models fail to exploit language priors. This is the empirical lower bound on visual dependency — the benchmark's clearest measure of genuine video necessity. Actionable for benchmark designers. Place in Discussion as "what visually pure looks like."

**R4-8** — object_existence black-screen convergence: despite 37pp spread in original accuracy (InternVL2=100%, Qwen=84.8%, LLaVA=63.0%), all three models converge to 45.7–47.8% black-screen accuracy. The black-screen floor is determined by question structure (answer distribution / linguistic cues), not model capacity. Strongest validity argument for VGG metric as architecture-independent measurement. Place in Methods/Validity §.

---

## ✅ GOOD Findings

**R4-1** — VGG ranking across 3 models: LLaVA=0.305 > InternVL2=0.252 > Qwen=0.238. Task-type hierarchy (Attribute Perception highest, Temporal Reasoning lowest) consistent across all 3 architectures including InternVL2 (no CLIP encoder). 3-model confirmation of F12.

**R4-3** — Cross-benchmark VGG divergence 3-model picture: InternVL2=5.8%, Qwen=12.6%, LLaVA=39.0%. LLaVA confirmed outlier. The two other architecturally distinct models both show <13% divergence, attributing LLaVA's deviation to architecture-specific behavior rather than benchmark noise.

**R4-4** — Destructive ratio on MVBench: InternVL2 5.39:1, Qwen 4.07:1, LLaVA 3.46:1. Ratio ordering mirrors VGG ordering. LLaVA's lowest ratio: language priors partially substitute for video, reducing unique video signal. Validates destructive ratio as a VGG-consistent construct.

**R4-5** — MVBench Bonferroni 14/27 significant (52% vs Video-MME 39%). Key dissociations: InternVL2 action_antonym NOT significant (0.340) while Qwen (0.600***) and LLaVA (0.460*) are; state_change InternVL2 significant (0.540***) but not Qwen/LLaVA — opposite dissociation. Supports architecture-specific language prior strategies.

**R4-7** — Cohen's kappa preserved across benchmarks: MVBench Qwen-LLaVA=0.441, Qwen-IV2=0.307, LLaVA-IV2=0.280; Video-MME Qwen-LLaVA=0.422, Qwen-IV2=0.350, LLaVA-IV2=0.270. Qwen and LLaVA agree most (both language-prior-heavy); InternVL2 most idiosyncratic. Kappa ordering stable across both benchmarks.

---

## Cumulative WOW count
| Round | WOW count |
|-------|-----------|
| Round 1 | 14 |
| Round 2 | 3 |
| Round 3 | 3 |
| Round 4 | 3 |
| **Total** | **23** |
