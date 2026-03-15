# Criticality v2 — Fixed Scoring Results

**Date:** 2026-02-07 | **Tasks per model:** 200 | **Random baseline:** 25%

## Results

| Model | Accuracy | 95% CI | p (vs random) | Margin | Entropy |
|-------|----------|--------|----------------|--------|---------|
| Gemma3n-E4B | **54.0%** | [47.1, 60.8] | < 0.0001 *** | 10.80 | 0.098 |
| GPT-OSS-20B | 49.0% | [42.2, 55.9] | < 0.0001 *** | 0.80 | 1.625 |
| DASD-4B | 48.0% | [41.2, 54.9] | < 0.0001 *** | 1.30 | 1.311 |
| Gemma3-4B | 48.0% | [41.2, 54.9] | < 0.0001 *** | 4.91 | 0.336 |
| Gemma3n-E2B | 48.0% | [41.2, 54.9] | < 0.0001 *** | 7.97 | 0.170 |
| Phi4-Mini-Reasoning | 42.5% | [35.9, 49.4] | < 0.0001 *** | 3.05 | 0.644 |
| Gemma3-1B | 40.5% | [33.9, 47.4] | < 0.0001 *** | 2.18 | 0.883 |
| Qwen3-0.6B | 37.0% | [30.6, 43.9] | 0.0001 *** | 1.32 | 1.324 |
| Falcon-H1-90M | 25.0% | [19.5, 31.4] | 0.527 ns | 0.41 | 1.750 |

CIs: Wilson score. P-values: one-sided binomial test. Entropy max = 2.0 (uniform over 4).

## What changed

| | Old method | New method |
|---|---|---|
| **Scoring** | Per-option continuation logprob — each argument scored independently as P(text \| prefix) | Single-token MCQ — all 4 arguments in one prompt, logprob extracted for A/B/C/D at answer position |
| **Task quality** | No gap filter; 46% of tasks had top-2 quality gap < 0.10 | min gap ≥ 0.15 enforced; mean gap = 0.212 |
| **Accuracy spread** | 4.5pp (29.0–33.5%) | 29.0pp (25.0–54.0%) |
| **Significant pairs** | 0 / 36 | 17 / 36 |
| **Models > random** | 5/9 (marginal, p ∈ 0.01–0.04) | 8/9 (all p < 0.001) |

## Why this works

**The old method measured fluency, not judgment.** Continuation scoring asks "how probable is this text as a completion?" — it gives each argument a separate eval call, so the model never sees the alternatives. High-quality arguments tend to use more specific vocabulary and longer constructions, which *lowers* average per-token logprob. The signal weakly anti-correlates with what we want.

**The new method forces comparative reasoning.** The full MCQ prompt places all 4 arguments in the model's context window. At the "Answer:" position, the transformer's hidden states have attended to every argument via self-attention. The logit distribution over A/B/C/D at that single position reflects a comparative judgment, not a fluency estimate.

Three pieces of evidence confirm this:

1. **Confidence scales with ability.** Falcon-H1-90M (25%, entropy 1.75) is effectively guessing — its distribution over labels is near-uniform. Gemma3n-E4B (54%, entropy 0.098) is extremely confident and usually right. Under the old method all models had similar entropy (~1.6).

2. **Position biases are now model-specific.** Old method: all models preferred A. New method: Gemma3-1B prefers A, Qwen3-0.6B prefers D, Falcon prefers B/C. Each model is forming its own representation of relative quality rather than exploiting a shared positional artifact.

3. **Task quality amplifies but doesn't create the signal.** Enforcing min gap ≥ 0.15 ensures a genuinely distinguishable correct answer. But tasks with large gaps still scored ~33% under the old scoring — the scoring change was the primary fix; task quality widened the separation.

## Emerging tiers

| Tier | Models | Accuracy | Notes |
|------|--------|----------|-------|
| A | Gemma3n-E4B | 54% | Separable from all below E2B (McNemar p < 0.02) |
| B | GPT-OSS-20B, DASD-4B, Gemma3-4B, Gemma3n-E2B | 48–49% | Not separable from each other; clearly above tier C |
| C | Phi4-Mini-Reasoning, Gemma3-1B | 40–43% | Intermediate |
| D | Qwen3-0.6B | 37% | Separable from tier B (p < 0.03), above random |
| F | Falcon-H1-90M | 25% | At random — 90M model cannot perform this task |

## Sample size

**Current:** 200 tasks per model. **Dataset ceiling:** ~1000 tasks at gap ≥ 0.15 (10k arguments loaded).

| n | CI half-width | Detect vs random | Detect pairwise |
|---|---------------|------------------|-----------------|
| 200 | ±6.9pp | 8.6pp | ~14pp |
| 500 | ±4.4pp | 5.5pp | ~9pp |
| 1000 | ±3.1pp | 3.8pp | ~6pp |

n=200 is adequate for the current conclusions. The tier B models (48–49%) differ by ~1pp — separating them would require n > 5000, which exceeds the dataset. Increasing to n=500 would tighten CIs enough to firmly separate tier A from tier B (currently marginal at p ≈ 0.06).
