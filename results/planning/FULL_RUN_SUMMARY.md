# Planning Benchmark — Full Run Complete

**Completed:** 2026-02-25  
**Sample test:** Passed (gpt-4o, limit 2, no errors)  
**Full run:** 50 tasks per model, all Azure (10) and Ollama (9) models.

---

## Azure (OneShotAgent, limit 50)

| Model | Success rate | Avg score | Total cost | Num tasks |
|-------|-------------:|----------:|-----------:|----------:|
| gpt-4o | 80% | 0.833 | $0.27 | 50 |
| ministral-3b | 42% | 0.461 | $0.01 | 50 |
| mistral-large-3 | 46% | 0.338 | $0.03 | 50 |
| phi-4-mini-reasoning | 18% | 0.202 | $0.03 | 50 |
| phi-4-mini | 10% | 0.174 | $0.01 | 50 |
| llama-3.3-70b | 4% | 0.096 | $0.00 | 50 |
| mistral-small | 0% | 0.158 | $0.02 | 50 |
| mistral-nemo | 0% | 0.075 | $0.00 | 50 |
| phi-4 | 0% | 0.075 | $0.00 | 50 |
| llama-3.2-11b-vision | 0% | 0.075 | $0.00 | 50 |

**Artifacts:** `results/planning/all_models_summary.json`, `results/planning/<model>_20260225_042432/OneShotAgent/`

---

## Ollama (OllamaAgent, limit 50)

| Model | Success rate | Avg score | Num tasks |
|-------|-------------:|----------:|----------:|
| qwen3-0.6b | 64% | 0.517 | 50 |
| gpt-oss-20b | 52% | 0.611 | 50 |
| dasd-4b | 36% | 0.420 | 50 |
| falcon-h1-90m | 0% | 0.075 | 50 |
| gemma3-1b | 0% | 0.075 | 50 |
| gemma3-4b | 0% | 0.075 | 50 |
| gemma3n-e2b | 0% | 0.075 | 50 |
| gemma3n-e4b | 0% | 0.075 | 50 |
| phi4-mini-reasoning-ollama | 0% | 0.075 | 50 |

**Artifacts:** `results/ollama/20260225_042433/planning/<model>/OllamaAgent/`

---

## Notes

- Travel scoring uses explicit total (no line-item as total); shopping uses JSON + budget + completeness.
- Azure `max_tokens` set to 16384 to avoid API errors.
- Models with 0% success and ~0.075 score typically produced minimal or off-format output; no run-breaking errors observed.
