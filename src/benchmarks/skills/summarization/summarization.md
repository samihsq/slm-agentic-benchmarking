## Goal

Add a **summarization skill benchmark** based on the **XSum** dataset so we can run it via `benchmark_runner.py` like the other skills (`recall`, `episodic_memory`, `criticality`).

The key constraint in this repo: benchmarks are designed to be **self-scoring** (no separate judge model). So for XSum we’ll score against the reference summary using **ROUGE-L** (or ROUGE-1/2/L).

## Why XSum (for this repo)

- **Clean task shape**: single document → single abstractive summary.
- **Fast**: articles are relatively short compared to GovReport/QMSum.
- **Easy to auto-score**: reference summaries exist.

## Implementation plan (matching existing benchmark patterns)

### 1) Add a runner like other skills

Create `src/benchmarks/skills/summarization/runner.py` implementing a `SummarizationRunner` similar to `RecallRunner`.

Recommended public API (keep it consistent with other runners):

- `load_tasks(limit: Optional[int]) -> List[Dict[str, Any]]`
- `format_task(task: Dict[str, Any]) -> tuple[str, Dict[str, Any]]`
- `_process_task(task: Dict[str, Any]) -> EvaluationResult`
- `run(limit: Optional[int] = None, save_results: bool = True) -> List[EvaluationResult]`

Use the same infra pieces the other runners use:

- `ThreadSafeAdaptiveLimiter` for concurrency
- `TraceCapture` for per-task trace saving
- optional `CostTracker` usage (token usage is already attached by agents like `OneShotAgent`)

### 2) Load XSum via Hugging Face `datasets`

XSum is available via HF datasets:

- dataset name: `xsum`
- fields you’ll typically use:
  - document/article: `document`
  - reference summary: `summary`
  - id: `id` (or construct your own task id)

Pseudo-shape:

```python
from datasets import load_dataset

ds = load_dataset("xsum", split="test")  # or "validation" for quicker dev
tasks = []
for row in ds.select(range(limit)):
    tasks.append({
        "task_id": f"xsum_{row['id']}",
        "document": row["document"],
        "reference": row["summary"],
    })
```

Notes:

- Prefer **`validation`** split for faster iteration while building.
- Add a small **local fallback** (like `RecallRunner` does) if dataset download isn’t available in some environments.

### 3) Prompt format (important: match `BaseAgent.parse_json_response`)

Agents parse JSON and treat `"response"` (or `"answer"`) as the output text.

So in your prompt, require JSON with `"response"` containing the model summary:

Example task prompt:

```text
You are given a news article. Write a single-sentence summary that captures the main point.

ARTICLE:
{document}

Return JSON:
{"response": "<one-sentence summary>", "confidence": 0.0-1.0}
```

Context object should mark this as a “general” benchmark so the system prompt doesn’t switch to medical/tool/memory styles:

```python
context = {
  "benchmark_type": "general",
  "task_type": "summarization",
  "dataset": "xsum",
  "split": split_name,
  "doc_tokens": len(document) // 4,  # rough estimate
}
```

### 4) Scoring: ROUGE-L (self-scoring, no judge model)

Pick one automatic metric and stick to it initially:

- **Primary**: ROUGE-L F1
- (Optional) Track ROUGE-1/2 as secondary stats

Implementation options:

- **Option A (recommended)**: use HF `evaluate`
  - `evaluate.load("rouge")`
  - returns ROUGE scores with stemming support
  - requires adding dependency: `evaluate` (and it will pull `rouge-score`)
- **Option B**: use `rouge-score` directly
  - requires adding dependency: `rouge-score`

Keep the output `score` in `EvaluationResult` as a **0–1 float** (ROUGE-L F1 is already in that range).

Suggested “success” criterion:

- `success = rougeL >= 0.15` (tune later; XSum is hard, so don’t set this too high)

### 5) Save per-task traces and a JSONL summary (copy existing conventions)

Follow `RecallRunner._save_result_incremental`:

- create `results/summarization/<model>_<timestamp>/<AgentClassName>/<task_id>/trace.json`
- append lightweight line to `results.jsonl` with:
  - task_id
  - rougeL (and rouge1/rouge2 optionally)
  - latency
  - cost
  - match/success

### 6) Wire it into the top-level runner

In `benchmark_runner.py`:

- Add `"summarization"` to `BENCHMARKS`
- Implement `run_summarization(model, agent_type, concurrency, limit)`
  - create agent the same way as other benchmarks
  - create `SummarizationRunner(...)`
  - patch `_process_task` to call `dashboard.update(...)` (same pattern as `run_recall`)
- Add to `BENCHMARK_RUNNERS` dict:
  - `"summarization": run_summarization`

Also add `SummarizationRunner` import in `src/benchmarks/skills/__init__.py` (optional but consistent with other skills).

## Recommended first version scope (to keep it shippable)

- **Split**: `validation`
- **Limit default**: respect `-n/--limit`
- **Metric**: ROUGE-L only (add ROUGE-1/2 later)
- **Prompt**: one-sentence summary (XSum style)
- **No extra judge model** (keep it aligned with current repo design)

## Later extensions (if we want “modern” beyond ROUGE)

- Add a second summarization benchmark for long context (e.g., **GovReport** from SCROLLS).
- Add a factuality probe (e.g., SummEdits-style inconsistency detection) as a *separate* benchmark, since it’s a different task shape than “generate summary”.