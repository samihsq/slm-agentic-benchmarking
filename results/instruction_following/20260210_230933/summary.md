# Instruction Following Benchmark Results

**Run Date:** 2026-02-10 23:09:33  
**Task:** Matrix transformation across 28 levels of increasing difficulty  
**Rollouts per model:** 10  
**Score:** Consecutive levels passed / 28

## Results

| Model | Agent | Mean Score | Success Rate | Max Level | Avg Latency (s) |
|---|---|---:|---:|---:|---:|
| DASD-4B-Thinking (Q4_K_M) | OllamaAgent | 0.7714 | 100% | 26 | 279.66 |
| gpt-oss-20b | OllamaAgent | 0.6857 | 100% | 26 | 115.06 |
| gpt-4o | OneShotAgent | 0.5357 | 100% | 15 | 17.49 |
| deepseek-v3.2 | OneShotAgent | 0.5107 | 100% | 26 | 335.51 |
| mistral-large-3 | OneShotAgent | 0.3321 | 100% | 26 | 18.40 |
| qwen3-0.6b | OllamaAgent | 0.1679 | 100% | 12 | 13.10 |
| llama-3.2-11b-vision | OneShotAgent | 0.1429 | 100% | 4 | 125.01 |
| mistral-nemo | OneShotAgent | 0.1429 | 100% | 4 | 118.82 |
| ministral-3b | OneShotAgent | 0.1393 | 100% | 4 | 4.73 |
| mistral-small | OneShotAgent | 0.1214 | 90% | 4 | 4.85 |
| phi-4-mini | OneShotAgent | 0.0679 | 70% | 4 | 94.85 |
| llama-3.3-70b | OneShotAgent | 0.0607 | 50% | 4 | 197.09 |
| gemma3n-e2b | OllamaAgent | 0.0500 | 50% | 4 | 5.13 |
| phi4-mini-reasoning (latest) | OllamaAgent | 0.0179 | 40% | 2 | 30.00 |
| phi-4 | OneShotAgent | 0.0036 | 10% | 1 | 29.92 |
| Falcon-H1-Tiny-R-90M (Q4_K_M) | OllamaAgent | 0.0000 | 0% | 0 | 13.66 |
| deepseek-v3 | OneShotAgent | 0.0000 | 0% | 0 | 329.37 |
| gemma3-1b | OllamaAgent | 0.0000 | 0% | 0 | 0.87 |
| gemma3-4b | OllamaAgent | 0.0000 | 0% | 0 | 1.67 |
| gemma3n-e4b | OllamaAgent | 0.0000 | 0% | 0 | 3.43 |
| phi-4-mini-reasoning | OneShotAgent | 0.0000 | 0% | 0 | 67.39 |

## Key Observations

- **Top performer:** DASD-4B-Thinking (0.77 mean score) consistently reached level 26 of 28.
- **Best latency/score trade-off:** gpt-4o scored 0.54 with only 17.5s average latency.
- **6 models scored 0.0**, unable to pass even level 1 — mostly small models (gemma3-1b/4b, Falcon-90M) or models with output formatting issues (deepseek-v3, phi-4-mini-reasoning).
- **Level 4→5 was a common cliff:** Many mid-tier models (ministral-3b, mistral-nemo, llama-3.2-11b-vision) topped out at level 4.
- **Model size ≠ score:** qwen3-0.6b (0.6B params) outperformed llama-3.3-70b, and DASD-4B beat gpt-4o.
