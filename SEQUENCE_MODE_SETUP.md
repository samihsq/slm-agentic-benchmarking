# Criticality v2: Sequence Scoring Mode Setup

## Overview

Criticality v2 now supports **sequence scoring mode** via llama-cpp-python. This allows evaluating ANY model (even tiny ones that can't follow MCQ instructions) by directly measuring conditional sequence likelihoods.

## Installation (Remote Machine: 10.27.102.240)

1. **Install llama-cpp-python with Metal acceleration:**

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

2. **Verify installation:**

```bash
python3 -c "from llama_cpp import Llama; print('✓ llama-cpp-python installed')"
```

## Usage

### Run with Sequence Mode

Use the `--model-path` flag to enable sequence mode:

```bash
# DASD-4B (4B params, previously worked in API mode)
python run_benchmark.py \
    --model DASD-4B \
    --agent one_shot \
    --benchmark criticality_v2 \
    --model-path /Users/samih_sully/.ollama/models/blobs/sha256-11e95b9455d84dff393e43d161c238ab65ded003bc85c3f29b628e2461c8879d \
    --limit 10

# Qwen3-0.6B (600M params, PREVIOUSLY FAILED in API mode - this is the real test!)
python run_benchmark.py \
    --model qwen3-0.6b \
    --agent one_shot \
    --benchmark criticality_v2 \
    --model-path /Users/samih_sully/.ollama/models/blobs/sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa \
    --limit 10

# Falcon-H1-90M (90M params, also previously failed)
python run_benchmark.py \
    --model falcon-h1-90m \
    --agent one_shot \
    --benchmark criticality_v2 \
    --model-path /Users/samih_sully/.ollama/models/blobs/sha256-c763e52b3902e5834d602b46c779ae54fa7b82cd94a0e893d818b804e081bdfd \
    --limit 10
```

### Run Test Script

A dedicated test script is provided:

```bash
cd /path/to/slm-agentic-benchmarking
python scripts/test_criticality_v2_sequence.py
```

This will test DASD-4B and Qwen3-0.6B with 5 tasks each.

## Model Blob Paths (Ollama on 10.27.102.240)

All models are stored as GGUF blobs in `/Users/samih_sully/.ollama/models/blobs/`:

| Model | Params | Blob SHA256 |
|-------|--------|-------------|
| **DASD-4B** | 4.0B | `sha256-11e95b9455d84dff393e43d161c238ab65ded003bc85c3f29b628e2461c8879d` |
| **Qwen3-0.6B** | 751M | `sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa` |
| **Falcon-H1-90M** | 108M | `sha256-c763e52b3902e5834d602b46c779ae54fa7b82cd94a0e893d818b804e081bdfd` |
| **phi4-mini-reasoning** | 3.8B | `sha256-f4dd2368e6c32725dc1c5c5548ae9ee2724d6a79052952eb50b65e26288022c4` |
| **gemma3n:e2b** | 4.5B | `sha256-3839a254cf2d00b208c6e2524c129e4438f9d106bba4c3fbc12b631f519d1de1` |
| **gemma3n:e4b** | 6.9B | `sha256-38e8dcc30df4eb0e29eaf5c74ba6ce3f2cd66badad50768fc14362acfb8b8cb6` |

## How It Works

### API Mode (default without --model-path)

- Asks model to output A/B/C/D via OpenAI-compatible API
- Extracts logprobs of choice tokens
- **Problem:** Only works for models that can follow MCQ instructions

### Sequence Mode (with --model-path)

- Loads GGUF directly via llama-cpp-python
- For each option, computes: `P(argument_text | prompt_prefix)`
- Scores by average token logprob (length-normalized)
- **Advantage:** Works on ANY model, no instruction-following needed

### Example Scoring

Given prefix: `Topic: "School uniforms". The strongest argument is: `

For each option A/B/C/D:
1. Tokenize: `prefix + option_text`
2. Run `model.eval()` to get logits
3. Extract logprobs for option tokens only (skip prefix)
4. Average logprob = mean of token logprobs
5. Rank by highest average logprob

## Expected Results

### Success Criteria

- **DASD-4B**: Should produce valid logprob extractions, non-zero accuracy
- **Qwen3-0.6B**: Should WORK in sequence mode (even though it failed in API mode)
- Both should show meaningful rank correlations (> 0.1)

### Previous Results (API Mode)

From `docs/criticality/PLAN.md`:

- **DASD-4B**: ✅ Worked (thinking model, 99.99% confidence on answer token)
- **Qwen3-0.6B**: ❌ Failed (always thinks, choice tokens barely appeared)
- **Falcon-H1-90M**: ❌ Failed (echoes prompt in `<think>` tags)

Sequence mode should fix Qwen3 and Falcon-H1 failures.

## Troubleshooting

### ImportError: No module named 'llama_cpp'

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

### FileNotFoundError: Model file not found

Check that the blob path exists:

```bash
ls -lh /Users/samih_sully/.ollama/models/blobs/sha256-11e95b...
```

### Metal GPU Not Used

Verify Metal is enabled:

```python
from llama_cpp import Llama
model = Llama(
    model_path="...",
    n_gpu_layers=-1,  # -1 = all layers to GPU
    verbose=True
)
# Check output for "Metal" mentions
```

### Low Memory / OOM

Reduce context size:

```bash
# Edit runner.py or pass via constructor
n_ctx=1024  # Default is 2048
```

Or use fewer GPU layers:

```bash
n_gpu_layers=20  # Instead of -1 (all)
```

## Next Steps

After verifying sequence mode works:

1. Run full benchmarks (limit=100+) on multiple models
2. Compare sequence mode vs API mode accuracy
3. Analyze rank correlations and calibration metrics
4. Test with larger models (phi4, gemma3n)
