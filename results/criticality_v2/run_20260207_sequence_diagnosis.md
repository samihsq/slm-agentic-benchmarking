# Criticality v2 — Sequence Scoring Diagnosis

**Date:** 2026-02-07 | **Tasks per model:** 200 | **Random baseline:** 25%  
**Scoring:** Per-option continuation logprob — `P(argument_text | "Topic: {topic}. The strongest argument is: ")`

## Verdict

**Benchmark broken.** All 9 models cluster within 4.5pp of each other (29.0–33.5%), no pair is statistically distinguishable, and performance barely exceeds random chance. The benchmark has zero discriminative power.

## Results

| Model | Accuracy | p (vs 25%) | Rank Corr. | Cal. Error |
|-------|----------|------------|------------|------------|
| Gemma3-1B | 33.5% | 0.004 * | 0.269 | 0.026 |
| Falcon-H1-90M | 32.0% | 0.015 * | 0.171 | 0.025 |
| GPT-OSS-20B | 31.5% | 0.023 * | 0.273 | 0.014 |
| Gemma3n-E4B | 31.5% | 0.023 * | 0.269 | 0.011 |
| Qwen3-0.6B | 31.0% | 0.032 * | 0.184 | 0.020 |
| Gemma3-4B | 30.0% | 0.063 | 0.276 | 0.003 |
| DASD-4B | 29.5% | 0.084 | 0.253 | 0.008 |
| Gemma3n-E2B | 29.5% | 0.084 | 0.233 | 0.006 |
| Phi4-Mini-Reasoning | 29.0% | 0.112 | 0.239 | 0.009 |

All McNemar pairwise p-values > 0.24. No model pair is statistically distinguishable.

## Problems identified

### 1. Ambiguous tasks (critical)

46% of tasks have a top-2 quality gap below 0.10 on a 0–1 scale. The "correct" answer is functionally a coin flip in nearly half the tasks.

| Gap threshold | % of tasks below |
|---------------|------------------|
| < 0.01 | 12% |
| < 0.05 | 26% |
| < 0.10 | 46% |
| < 0.20 | 80% |

**Cause:** The dataset contains 758 strong / 203 medium / 39 weak arguments. The task generator fills all 4 slots from the largest available tier, producing tasks where every option is "strong" with near-identical quality scores.

### 2. Wrong signal (critical)

Continuation scoring measures text fluency, not argument quality. Each option gets a separate `model.eval()` call — the model never sees the alternatives.

| Correlation | r |
|-------------|---|
| Quality vs logprob | 0.272 |
| Length vs logprob | −0.048 |
| Length vs quality | 0.317 |

The quality–logprob correlation (0.272) is weak and likely driven by the prefix mentioning "strongest argument" rather than actual quality assessment. High-quality arguments use more specific vocabulary, which *lowers* average per-token logprob.

### 3. Uniform position bias (moderate)

All models prefer option A (28–36% of predictions vs 26.5% true base rate). This shared bias suggests models are exploiting a positional artifact in the continuation scoring, not reading the argument content.

## Fixes applied

1. **Task generation:** Enforce min quality gap ≥ 0.15 between best and second-best option. Load 4000+ arguments to ensure cross-tier diversity.
2. **Scoring method:** Switch to single-token MCQ scoring — present all 4 labeled arguments in one prompt, extract logprobs for A/B/C/D at the answer position.
