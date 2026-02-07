# Quick Reference: Running Benchmarks

## Run All Models on Recall + Episodic Memory

### Full Run (All 13 Models)
```bash
python benchmark_runner.py \
  --models all \
  --benchmarks recall,episodic_memory \
  --agents oneshot \
  --limit 100 \
  --concurrency 5
```
**Time**: 30-45 minutes | **Cost**: ~$5-10 | **Tasks**: 2,600

---

### Quick Test (4 Models)
```bash
python benchmark_runner.py \
  --models phi-4,gpt-4o,mistral-small,deepseek-v3.2 \
  --benchmarks recall,episodic_memory \
  --agents oneshot \
  --limit 50 \
  --concurrency 10
```
**Time**: 10-15 minutes | **Cost**: ~$2-3 | **Tasks**: 400

---

### Small Test (3 Models, Fast)
```bash
python benchmark_runner.py \
  --models phi-4,gpt-4o,mistral-small \
  --benchmarks recall,episodic_memory \
  --agents oneshot \
  --limit 25 \
  --concurrency 10
```
**Time**: 5-8 minutes | **Cost**: ~$1-2 | **Tasks**: 150

---

## Available Models (13 Total)

- **Microsoft**: phi-4, phi-4-mini, phi-4-mini-reasoning
- **Mistral**: mistral-small, mistral-large-3, mistral-nemo, ministral-3b
- **Meta**: llama-3.3-70b, llama-3.2-11b-vision
- **DeepSeek**: deepseek-v3.2, deepseek-v3, deepseek-r1
- **OpenAI**: gpt-4o

---

## Run Single Benchmark

### Recall Only
```bash
python benchmark_runner.py \
  --models all \
  --benchmarks recall \
  --agents oneshot \
  --limit 100 \
  --concurrency 5
```

### Episodic Memory Only
```bash
python benchmark_runner.py \
  --models all \
  --benchmarks episodic_memory \
  --agents oneshot \
  --limit 100 \
  --concurrency 5
```

---

## Run With Multiple Agent Types

```bash
python benchmark_runner.py \
  --models phi-4,gpt-4o,mistral-small \
  --benchmarks recall,episodic_memory \
  --agents oneshot,sequential,concurrent \
  --limit 100 \
  --concurrency 5
```
**Tasks**: 3 models × 2 benchmarks × 3 agents × 100 = 1,800 tasks

---

## Results Location

```
results/
├── recall/
│   └── {model}_{timestamp}/
│       └── OneShotAgent/
│           ├── summary.json
│           ├── results.jsonl
│           └── recall_0000/
│               └── trace.json
├── episodic_memory/
│   └── {model}_{timestamp}/
│       └── OneShotAgent/
│           ├── summary.json
│           └── ...
└── combined/
    └── combined_run_{timestamp}.json
```

---

## Live Dashboard

When running, you'll see a real-time dashboard:
```
┌──────────────────────────────────────────────┐
│ Model         | Bench      | Tasks  | Acc   │
├──────────────────────────────────────────────┤
│ phi-4         | recall     | 45/100 | 82.0% │
│ phi-4         | episodic   | 32/100 | 65.0% │
│ gpt-4o        | recall     | 67/100 | 88.0% │
│ mistral-small | recall     | 89/100 | 91.0% │
└──────────────────────────────────────────────┘
```

Press `Ctrl+C` to stop gracefully (saves partial results).

---

## Adjust Concurrency

- `--concurrency 1`: Slowest, safest (sequential)
- `--concurrency 5`: Balanced (default)
- `--concurrency 10`: Faster, more API load
- `--concurrency 20`: Very fast, may hit rate limits

---

## Troubleshooting

### Rate Limits
If you hit rate limits, the system will automatically back off and retry.

### Cost Tracking
Check cost at any time:
```bash
python -c "from src.evaluation import CostTracker; CostTracker().print_summary()"
```

### Resume Failed Run
Results are saved incrementally, so you can check partial results even if interrupted.

---

## Dataset Size Options

Edit `benchmark_runner.py` or `run_benchmark.py` to change dataset size:

**Recall**: Uses episodic memory narrative
**Episodic Memory**: 
- 20 chapters → ~13K tokens
- 200 chapters → ~103K tokens (edit `num_chapters=200`)
- 2000 chapters → ~1M tokens (edit `num_chapters=2000`)
