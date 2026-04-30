# Maximizing Small Models: How Agentic Skill Profiles Predict Orchestration Performance

Code accompanying our submission to the [Agents in the Wild](https://agentwild-workshop.github.io/icml2026/) workshop at ICML 2026. We benchmark 11 small language models (SLMs, ~0.6B–20B parameters) across four agent architectures on BIG-bench Lite and a set of skill-specific probes, and study whether per-skill profiles predict performance under multi-agent orchestration.

## What this evaluates

**Models** — 11 SLMs across two backends:

- *Ollama (local)*: `qwen3-0.6b`, `gemma3-1b`, `gemma3n-e2b`, `gemma3n-e4b`, `gemma3-4b`, `dasd-4b` (DeepSeek-distilled), `phi4-mini-reasoning`, `gpt-oss-20b`.
- *Azure AI Foundry (serverless)*: `phi-4`, `ministral-3b`, `mistral-small`.

Both registries live in `src/config/azure_llm_config.py` (`OLLAMA_MODELS`, `AVAILABLE_MODELS`).

**Architectures** (`src/agents/`):
- `OneShotAgent` — single direct LLM call (non-agentic baseline).
- `SequentialAgent` — CrewAI Analyzer → Evaluator → Responder pipeline.
- `ConcurrentAgent` — multiple CrewAI agents on the same task in parallel, results merged.
- `GroupChatAgent` — CrewAI agents in a discussion loop with a manager.

**Benchmarks** (`src/benchmarks/skills/`):
- **BIG-bench Lite (BBL24)** — primary suite, 24 tasks loaded from [`tasksource/bigbench`](https://huggingface.co/datasets/tasksource/bigbench) on HuggingFace.
- Skill probes: `recall`, `episodic_memory`, `criticality`, `summarization`, `instruction_following`, `planning`, `plan_bench`, `matrix_recall`.

Full model and architecture details are in `src/config/azure_llm_config.py` and `src/agents/`.

## Setup

```bash
poetry install

# Ollama (local) — pull the models you want to evaluate (one-time)
ollama pull qwen3:0.6b gemma3:1b gemma3n:e2b gemma3:4b gpt-oss:20b
# ...etc; full list in src/config/azure_llm_config.py

# Azure AI Foundry — needed for phi-4 / ministral-3b / mistral-small
export AZURE_API_KEY="<your key>"
export AZURE_AI_ENDPOINT="<your endpoint>"     # optional; a default is set in code
```

The Ollama path needs no credentials. The Azure path requires an Azure AI Foundry deployment with the three serverless models above accessible via your `AZURE_API_KEY`. Reproducing the paper's full results requires both; the Ollama-only subset (8 of 11 models) runs without any API keys.

### Datasets

Datasets are pulled at runtime from their original sources — none are vendored:

| Benchmark | Source | Notes |
|---|---|---|
| BIG-bench Lite | HuggingFace `tasksource/bigbench` | auto-downloaded by `datasets.load_dataset` on first run |
| Recall, Episodic Memory | HuggingFace | auto-downloaded; cached under `data/` (gitignored) |
| Criticality | IBM ArgQ-30k | follow upstream instructions |
| PlanBench | Karthik Valmeekam's [PlanBench](https://github.com/karthikv792/LLMs-Planning) | vendored under `vendor/plan_bench/` |

The `data/` directory is gitignored; first invocation of a runner will populate it.

## Reproducing the paper

The paper's main results come from the BIG-bench Lite sweep across 11 models × 4 architectures (8 Ollama + 3 Azure).

**Local run:**
```bash
# Full sweep — Azure models run concurrently, Ollama models sequentially
poetry run python scripts/run_bigbench_lite_sweep.py

# Ollama-only subset (no Azure credentials needed)
poetry run python scripts/run_bigbench_lite_sweep.py --backends ollama

# Resume after interruption
poetry run python scripts/run_bigbench_lite_sweep.py --resume

# Pilot (4 examples/task) for a fast sanity check
poetry run python scripts/run_bigbench_lite_sweep.py --pilot
```

**Modal (4× L4 GPUs, the configuration we used for the Ollama half):**
```bash
modal token new                                   # one-time
poetry run modal run scripts/modal_ollama_bigbench.py
poetry run modal run scripts/modal_ollama_bigbench.py --sync-only   # pull results back
```
Worker → model assignments are at the top of `scripts/modal_ollama_bigbench.py`. The Azure models are run separately via the local sweep script.

**Skill probes** (recall, criticality, etc.) are run via:
```bash
poetry run python run_benchmark.py --model <name> --agent <arch> --benchmark <skill>
```

### Results layout

A successful sweep populates:

```
results/bigbench_lite/ollama/<model>/<architecture>/
    summary.json      # suite-level metrics
    results.jsonl     # per-question results
    <task_id>/trace.json   # full prompt/response/token/latency trace
```

The full `results/` tree from our sweep is not committed to git — it will be attached as a GitHub release alongside the workshop submission so figures and tables can be regenerated without re-running the sweep.

Aggregation/plotting helpers:
- `scripts/join_bigbench_skill_scores.py` — joins BBL scores with skill-probe scores.
- `scripts/plot_skill_vs_bigbench.py` — generates the skill-vs-BBL scatter plots.
- `scripts/generate_results_report.py` — assembles summary tables.

## Repo layout

```
src/
  agents/          # 4 architectures, Azure + Ollama variants
  benchmarks/skills/  # active runners (bigbench, recall, criticality, ...)
  config/          # model registries (Azure + Ollama)
  utils/trace.py   # TraceCapture context manager
scripts/           # sweep runners, Modal entrypoint, plotting
vendor/plan_bench/  # vendored PlanBench harness
```

## Citation

```bibtex
@inproceedings{slm-agentic-2026,
  title     = {Maximizing Small Models: How Agentic Skill Profiles Predict Orchestration Performance},
  author    = {Qureshi, Samih and Han, Fiona and Yang, Justin},
  booktitle = {ICML 2026 Workshop on Agents in the Wild},
  year      = {2026}
}
```

## License

MIT — see `LICENSE`.
