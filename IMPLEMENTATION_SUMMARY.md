# Instruction Following and Summarization Benchmarks - Implementation Summary

## Completed Implementation

Successfully implemented and integrated two new benchmarks:

### 1. Instruction Following Benchmark
- **File**: `src/benchmarks/skills/instruction_following/runner.py`
- **Description**: Tests model's ability to follow matrix transformation rules across 28 levels of increasing difficulty
- **Features**:
  - Each task is a full 28-level rollout (model called 28 times per task)
  - Score = consecutive levels passed / 28
  - Exact matrix comparison for correctness
  - Level-by-level accuracy tracking
  - Default: 10 task rollouts

### 2. Summarization Benchmark
- **File**: `src/benchmarks/skills/summarization/runner.py`
- **Description**: Tests single-sentence news article summarization using XSum dataset
- **Features**:
  - ROUGE-L F1 scoring (primary metric)
  - ROUGE-1, ROUGE-2 as secondary metrics
  - Success threshold: ROUGE-L >= 0.15
  - Dataset: XSum validation split by default
  - Fallback to synthetic examples if dataset unavailable

## Files Created

1. `src/benchmarks/skills/instruction_following/runner.py` - Main runner
2. `src/benchmarks/skills/instruction_following/__init__.py` - Module init
3. `src/benchmarks/skills/summarization/runner.py` - Main runner
4. `src/benchmarks/skills/summarization/__init__.py` - Module init

## Files Modified

1. `pyproject.toml` - Added `rouge-score = "^0.1.2"` dependency
2. `src/benchmarks/skills/__init__.py` - Added imports for both runners
3. `src/benchmarks/__init__.py` - Added imports for both runners
4. `run_benchmark.py` - Registered both benchmarks (imports, runner dict, CLI choices, cost estimation)
5. `scripts/run_ollama_benchmarks.py` - Registered both benchmarks (imports, runners dict, BENCHMARKS list, DEFAULT_LIMITS)

## Usage Examples

### Azure API (via run_benchmark.py)
```bash
# Instruction Following
python run_benchmark.py --model phi-4 --agent one_shot --benchmark instruction_following --limit 5

# Summarization
python run_benchmark.py --model phi-4 --agent one_shot --benchmark summarization --limit 50
```

### Ollama (via run_ollama_benchmarks.py)
```bash
# Single model, both benchmarks
python scripts/run_ollama_benchmarks.py --models dasd-4b --benchmarks instruction_following,summarization --limit 50

# All models, specific benchmark
python scripts/run_ollama_benchmarks.py --models all --benchmarks instruction_following --limit 10
```

## Provider Support

Both benchmarks work with **Azure and Ollama** out of the box:
- They accept `agent: BaseAgent` parameter
- `OneShotAgent` (Azure/LiteLLM) and `OllamaAgent` (Ollama REST) both implement `BaseAgent`
- No provider-specific code in either runner
- Identical output formats regardless of provider

## Testing

All imports verified:
```bash
poetry run python3 -c "from src.benchmarks import InstructionFollowingRunner, SummarizationRunner; print('✓ Imports successful')"
# Output: ✓ Imports successful
```

CLI integration verified:
```bash
poetry run python run_benchmark.py --help
# Shows: instruction_following, summarization in benchmark choices
```

Dependencies installed:
- `rouge-score==0.1.2` (and dependencies: nltk, joblib, absl-py)
- `datasets` (already present)

## Implementation Notes

1. **Task Granularity**: Instruction following uses 28 sequential API calls per task rollout to mirror the progressive difficulty structure
2. **Scoring**: Both use exact/objective metrics (exact matrix match, ROUGE-L) - no LLM judge needed
3. **Concurrency**: Both support concurrent execution for faster benchmarking
4. **Trace Capture**: Full TraceCapture integration for debugging and analysis
5. **Cost Tracking**: Optional CostTracker support for Azure API usage

## Next Steps

The benchmarks are ready to use. To run them:
1. Ensure `poetry install` has been run to install dependencies
2. Set up Azure credentials (for Azure models) or Ollama instance (for local models)
3. Run using the CLI examples above
