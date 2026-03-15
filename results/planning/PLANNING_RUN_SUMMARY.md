# Planning Benchmark Run Summary

**Date:** 2026-02-25  
**Limit:** 3 tasks per model (travel + shopping mix)  
**Agent:** one_shot (Azure), OllamaAgent (local)

---

## 1. Tests

All **43 pytest** tests in `tests/benchmarks/test_planning_runner.py` **passed** (scoring, load_tasks, format_task, E2E with MockAgent).

---

## 2. Azure (all serverless models)

Run: `python run_benchmark.py --all-models --benchmark planning --agent one_shot --limit 3`

| Model                  | Success rate | Avg score | Avg latency | Cost ($) |
|------------------------|-------------:|----------:|------------:|---------:|
| gpt-4o                 | 100%         | 1.00      | 4.8s       | 0.018    |
| phi-4-mini-reasoning   | 66.7%        | 0.65      | 69.7s      | 0.003    |
| mistral-large-3        | 66.7%        | 0.52      | 40.9s      | 0.002    |
| phi-4                  | 33.3%        | 0.38      | 105.6s     | 0.0003   |
| llama-3.2-11b-vision   | 33.3%        | 0.38      | 12.1s      | 0.001    |
| llama-3.3-70b          | 33.3%        | 0.36      | 4.9s       | 0.0006   |
| phi-4-mini             | 33.3%        | 0.25      | 14.6s      | 0.001    |
| mistral-small          | 0%           | 0.17      | 6.8s       | 0.001    |
| ministral-3b           | 0%           | 0.05      | 3.1s       | 0.001    |
| mistral-nemo           | 0%           | 0.05      | 18.3s      | 0        |

**Output quality (spot check):**
- **gpt-4o:** Full day-by-day itinerary (Day 1–3), budget summary (2698 CNY ≤ 3000), transport/hotel/meals/attractions; shopping JSON with items and final_total. All three tasks scored 1.0.
- **ministral-3b:** Truncated output ("**Day-by-Day Itinerary**" only); correctly scored 0 (structure/budget/constraint all low).

---

## 3. Ollama (local models)

Run: `python scripts/run_ollama_benchmarks.py --benchmarks planning --limit 3`

| Model               | Success rate | Mean composite | Mean latency |
|--------------------|-------------:|---------------:|-------------:|
| qwen3-0.6b         | 100%         | 0.92           | 4.9s         |
| gpt-oss-20b        | 66.7%        | 0.52           | 39.4s        |
| dasd-4b            | 33.3%        | 0.38           | 40.0s        |
| falcon-h1-90m      | 0%           | 0.05           | 17.4s        |
| gemma3-1b          | 0%           | 0.05           | 17.5s        |
| gemma3-4b          | 0%           | 0.05           | 17.1s        |
| gemma3n-e2b        | 0%           | 0.05           | 17.3s        |
| gemma3n-e4b        | 0%           | 0.05           | 17.0s        |

**Note:** `phi4-mini-reasoning-ollama` was not present in this run (may not have been in the default model list or run skipped).

**Output quality (spot check):**
- **qwen3-0.6b:** Shopping task 1.0 (valid JSON, budget, completeness); both travel tasks ~0.87 (structure 0.83, budget 1.0, constraint 0.8). Plans are coherent and within budget.
- **dasd-4b:** 1/3 success (shopping); travel tasks failed on structure/budget/constraint.
- **falcon-h1-90m:** Near-zero scores; outputs too short or off-format.

---

## 4. Verdict

- **Benchmark behavior:** Planning benchmark is working as intended on both Azure and Ollama: tasks load (synthetic when HF dataset is cached/unavailable), prompts are domain-specific, and heuristic scoring (structure, budget, constraints for travel; JSON, budget, completeness for shopping) differentiates strong vs weak outputs.
- **Progress:** Azure run completed for all 10 serverless models; Ollama run completed for 8 models. Results and traces are under `results/planning/` (Azure) and `results/ollama/<timestamp>/planning/` (Ollama).
- **Output quality:** Strong models (e.g. gpt-4o, qwen3-0.6b) produce on-budget, structured plans and score high; weak/small models (e.g. ministral-3b, falcon-h1-90m) produce minimal or off-format text and score low. No crashes or missing result files observed.
