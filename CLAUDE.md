# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install

# Lint
poetry run ruff check .

# Format
poetry run black .

# Run all tests
poetry run pytest tests/

# Run a single test file
poetry run pytest tests/benchmarks/test_metrics.py -v

# Run the full BIG-bench Lite sweep (Azure + Ollama)
poetry run python scripts/run_bigbench_lite_sweep.py

# Pilot run (4 examples/task, fast iteration)
poetry run python scripts/run_bigbench_lite_sweep.py --pilot

# Resume an interrupted sweep
poetry run python scripts/run_bigbench_lite_sweep.py --resume

# Single model + agent via CLI
poetry run python run_benchmark.py --model phi-4 --agent sequential --benchmark medqa

# Multi-model concurrent runner
poetry run python benchmark_runner.py --models phi-4,gpt-4o --benchmarks recall,episodic_memory --limit 50

# Modal remote sweep (4x L4 GPU workers)
poetry run modal run scripts/modal_ollama_bigbench.py

# Sync results from Modal volume locally
poetry run modal run scripts/modal_ollama_bigbench.py --sync-only

# Check Modal app status
poetry run modal app list
```

## Architecture

### Agent Layer (`src/agents/`)

Four architectures, each implementing `BaseAgent.respond_to_task()`:

- **`OneShotAgent`** — Bypasses CrewAI entirely; makes a single direct `litellm.completion()` call. This is the true non-agentic baseline (1 task = 1 LLM call).
- **`SequentialAgent`** — Three-stage CrewAI pipeline: Analyzer → Evaluator → Responder running in sequence.
- **`ConcurrentAgent`** — Multiple CrewAI agents running in parallel on the same task, merging results.
- **`GroupChatAgent`** — CrewAI agents in a discussion loop with a manager.

Each architecture also has an **Ollama variant** (`OllamaAgent`, `OllamaSequentialAgent`, `OllamaConcurrentAgent`, `OllamaGroupChatAgent`) defined in `ollama_agentic_agent.py` and `ollama_agent.py`. These are identical in structure but use `LLM(model="ollama/<name>", base_url=...)` instead of Azure credentials.

**Important:** Only `OneShotAgent` avoids CrewAI's ReAct format loop. The three agentic architectures use CrewAI's `Process.sequential` / `Process.hierarchical` and are subject to repeated retries when SLMs fail to emit the expected `Action:` / `Thought:` format — this is the primary source of slowness with small models.

### Benchmark Layer (`src/benchmarks/skills/`)

Active benchmarks:
- **`bigbench/`** — BIG-bench Lite 24-task suite (`BBL24`). Primary benchmark for the sweep. Scoring: multiple-choice letter match, exact string, or BLEU, auto-selected per task.
- **`recall/`**, **`episodic_memory/`** — Long-context retrieval benchmarks.
- **`criticality/`** — Argument quality pairwise comparison (IBM 30k dataset).
- **`summarization/`**, **`planning/`**, **`instruction_following/`**, **`matrix_recall/`**, **`plan_bench/`** — Additional skill benchmarks.

`src/benchmarks/archive/` contains older medical/tool-calling runners (`MedQA`, `BFCL`, `MedAgentBench`) that are no longer actively used.

### Model Configuration (`src/config/azure_llm_config.py`)

Two model registries:
- **`AVAILABLE_MODELS`** — Azure AI Foundry serverless models (requires `AZURE_API_KEY`). All use `openai/<ModelName>` strings via LiteLLM.
- **`OLLAMA_MODELS`** — Local/remote Ollama models (no API key). Uses `ollama/<name>` strings. This is the registry the Modal sweep reads for shard assignments.

`get_llm(model_name)` returns a CrewAI `LLM` instance. `get_llm_config(model_name)` returns the raw dict. Both raise `ValueError` for unknown model names.

### Modal Sweep (`scripts/modal_ollama_bigbench.py`)

Launches 4 L4 GPU workers in parallel, each handling a fixed model shard:
- **worker-1**: `qwen3-0.6b`, `gemma3-1b`
- **worker-2**: `gemma3n-e2b`, `phi4-mini-reasoning-ollama`
- **worker-3**: `dasd-4b`, `gemma3-4b`, `gemma3n-e4b`
- **worker-4**: `gpt-oss-20b`

Each worker starts its own Ollama server, pre-pulls models into a persistent volume (`slm-ollama-model-cache`), then calls `run_bigbench_lite_sweep.py` sequentially per model. Results are written to the `slm-bigbench-results` volume and synced locally after all workers complete.

### Tracing (`src/utils/trace.py`)

`TraceCapture` context manager wraps a benchmark run and accumulates `QuestionTrace` objects (one per question), each containing a list of `LLMCall` records with full prompt/response text, token counts, and latency. Traces are written as `trace.json` files under `results/<benchmark>/<model>/<architecture>/<task_id>/`.

### Results Layout

```
results/bigbench_lite/
  ollama/<model>/<architecture>/
    summary.json      # suite-level metrics
    results.jsonl     # per-question results
    <task_id>/trace.json
```

`summary.json` presence is used by `--resume` to skip completed (model, architecture) pairs and by `--sync-only` validation to confirm what finished.

## Environment Variables

| Variable | Purpose |
|---|---|
| `AZURE_API_KEY` | Required for Azure AI Foundry models |
| `AZURE_AI_ENDPOINT` | Defaults to the CS199 endpoint if unset |

Ollama runs don't require any credentials. Modal auth uses the local `~/.modal.toml` token set by `modal token new`.
