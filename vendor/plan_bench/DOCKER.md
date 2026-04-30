# PlanBench with VAL in Docker

This directory contains Docker configuration to run PlanBench with VAL (the plan validator) pre-installed. VAL is required for full plan correctness evaluation.

## Quick Start

### 1. Build the image

```bash
# From the repo root
docker build -f Dockerfile.planbench -t slm-planbench .
```

This builds an Ubuntu 22.04 image with:
- VAL compiled and installed at `/usr/local`
- Python 3.10 and project dependencies
- PlanBench vendored code and test suite

### 2. Run PlanBench with VAL

#### Option A: Using docker-compose (recommended)

```bash
# Set your API key
export LITELLM_API_KEY=your_key_here

# Run tests (validates VAL is working)
docker compose -f docker-compose.planbench.yml run --rm planbench python -m pytest tests/benchmarks/test_plan_bench_runner.py -v

# Run a single model with PlanBench
docker compose -f docker-compose.planbench.yml run --rm planbench python run_benchmark.py --benchmark plan_bench --plan-bench-task t1 --plan-bench-config blocksworld --limit 5 --model azure/gpt-4o
```

#### Option B: Using docker run directly

```bash
# Run tests
docker run --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/results:/workspace/results \
  -e VAL=/usr/local \
  -e LITELLM_API_KEY=your_key \
  -e USE_LITELLM=1 \
  slm-planbench \
  python -m pytest tests/benchmarks/test_plan_bench_runner.py -v

# Run PlanBench
docker run --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/results:/workspace/results \
  -e VAL=/usr/local \
  -e LITELLM_API_KEY=your_key \
  -e USE_LITELLM=1 \
  slm-planbench \
  python run_benchmark.py --benchmark plan_bench --plan-bench-task t1 --plan-bench-config blocksworld --limit 5 --model azure/gpt-4o
```

#### Option C: Using the helper script

```bash
# Set your API key
export LITELLM_API_KEY=your_key_here

# Run tests
./scripts/run_planbench_container.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VAL` | Path to VAL directory (contains `validate` binary) | `/usr/local` (set in Dockerfile) |
| `LITELLM_API_KEY` | Azure API key for LiteLLM | Required for Azure models |
| `LITELLM_API_BASE` | Azure endpoint URL | Optional |
| `USE_LITELLM` | Enable LiteLLM adapter | `1` (set in Dockerfile) |
| `LITELLM_MODEL` | Model ID for LiteLLM | Set automatically by runner |

## What VAL Does

VAL is the plan validator from KCL-Planning. It checks whether a generated plan satisfies the PDDL domain and problem constraints. When VAL is installed and `VAL` is set:

- PlanBench runs `response_evaluation.py` after `response_generation.py`.
- Each plan is extracted and validated against the domain/problem PDDL files.
- `llm_correct` is set to `True` or `False` based on plan validity.
- Summary `success_rate` reflects actual plan correctness.

Without VAL, PlanBench still generates plans and saves them, but:
- `llm_correct` is not set.
- Instances are marked as `evaluated=False`.
- Summary `success_rate` is `null`.

## Troubleshooting

### VAL not found

If you see errors about `validate` not being found:

```bash
# Verify VAL is installed in the container
docker run --rm slm-planbench ls -la /usr/local/bin/validate

# Check VAL env var
docker run --rm slm-planbench env | grep VAL
```

### API key errors

Make sure `LITELLM_API_KEY` is set:

```bash
# In docker-compose.yml
environment:
  - LITELLM_API_KEY=${LITELLM_API_KEY}

# Or pass at runtime
docker run -e LITELLM_API_KEY=your_key ...
```

### Permission errors on results

The container writes to `./results` (mounted volume). Ensure the directory is writable:

```bash
mkdir -p results
chmod 777 results
```

## Building VAL from Source (Manual)

If you need to rebuild VAL manually inside the container:

```bash
docker run --rm -it slm-planbench bash

# Inside container
cd /tmp
git clone https://github.com/KCL-Planning/VAL.git
cd VAL
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install
```

## Running All Benchmarks in Docker

To run the full benchmark suite with VAL:

```bash
docker compose -f docker-compose.planbench.yml run --rm planbench python scripts/run_ollama_benchmarks.py --benchmarks plan_bench --limit 5
```

Results will be saved to `./results/ollama/<timestamp>/plan_bench/`.

## See Also

- [PlanBench README](./README.md) — PlanBench tasks and usage
- [LLMs-Planning](https://github.com/karthikv792/LLMs-Planning) — Original PlanBench repository
- [VAL](https://github.com/KCL-Planning/VAL) — Plan validator