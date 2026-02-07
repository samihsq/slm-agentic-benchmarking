# SLM Agentic Benchmarking Framework

Comprehensive benchmarking framework for evaluating Small Language Models (SLMs) in agentic architectures, deployed on Azure AI.

## Overview

This framework extends the CrewAI-based agent architectures to benchmark SOTA SLMs on:
- **Medical benchmarks**: MedAgentBench (agentic), MedQA/MedMCQA (knowledge)
- **Tool calling benchmarks**: MCP-Bench, BFCL v3
- **Reasoning & Memory benchmarks**: Criticality, Recall, Episodic Memory

## Features

- **4 Agent Architectures**: OneShot, Sequential, Concurrent, GroupChat
- **SOTA Baseline**: GPT-4o/GPT-4o-mini non-agentic comparison
- **Cost Tracking**: Real-time budget monitoring with alerts
- **Azure Integration**: Serverless models + Azure ML endpoints
- **Multiple Benchmarks**: Medical, tool-calling, reasoning, and memory evaluation

## Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Configure Azure Credentials

```bash
# For Azure OpenAI (GPT-4o baseline)
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# For Azure AI Foundry (serverless models)
export AZURE_AI_API_KEY="your-key"

# For Azure ML endpoints (custom models)
export AZURE_ML_ENDPOINT="https://your-endpoint.inference.ml.azure.com"
```

### 3. Estimate Costs

```bash
python run_benchmark.py --estimate --models all --benchmark all
```

### 4. Run Benchmarks

```bash
# Non-agentic baseline
python run_benchmark.py --model gpt-4o-mini --agent baseline --benchmark medqa

# Agentic architecture
python run_benchmark.py --model phi-4 --agent sequential --benchmark medagent

# Compare agentic vs baseline
python run_benchmark.py --compare-baseline --model phi-4 --agent sequential --benchmark medqa

# Run reasoning benchmarks
python run_benchmark.py --model phi-4 --agent one_shot --benchmark criticality --limit 100
python run_benchmark.py --model phi-4 --agent one_shot --benchmark recall --limit 100

# Run multiple models concurrently
python benchmark_runner.py --models phi-4,gpt-4o,mistral-small --benchmarks criticality,recall --limit 100
```

## Available Models

### Serverless (Pay-per-token)
- `phi-4` - Microsoft Phi-4 14B
- `llama-3.3-70b` - Meta Llama 3.3 70B
- `mistral-small` - Mistral Small 24B
- `mistral-large-3` - Mistral Large 3 123B
- `deepseek-v3.2` - DeepSeek V3.2 671B
- `gpt-4o` - OpenAI GPT-4o (baseline)
- `gpt-4o-mini` - OpenAI GPT-4o-mini (cost-efficient baseline)

### Azure ML Endpoints (Requires deployment)
- `glm-4.7-flash` - GLM-4.7-Flash 30B MoE
- `qwen3-30b-a3b` - Qwen3 30B MoE
- `lfm2.5-1.2b` - LFM2.5 1.2B Thinking

## Benchmarks

### Medical
- **MedAgentBench**: 100 clinical agentic tasks (FHIR-compliant)
- **MedQA**: 1,273 USMLE-style questions
- **MedMCQA**: Indian medical MCQs

### Tool Calling
- **MCP-Bench**: 250+ tool-calling tasks
- **BFCL v3**: Function calling evaluation

### Reasoning & Memory
- **Criticality**: Argument quality assessment (IBM 30k dataset)
  - Tests ability to judge argument strength and persuasiveness
  - Pairwise comparisons with crowd-sourced quality scores
- **Recall**: Keyword-based sentence retrieval
  - Tests information retrieval from passages (100-2000 tokens)
  - Difficulty levels: easy, medium, hard
- **Episodic Memory**: Long-context state tracking
  - Tests entity tracking across extended narratives (10K-1M tokens)
  - Simple recall and chronological awareness

## Cost Management

The framework includes built-in cost tracking:

```python
from src.evaluation import CostTracker

tracker = CostTracker(budget_limit=10000.0)
tracker.print_summary()  # View spending breakdown
```

Budget alerts trigger at 30%, 60%, and 90% thresholds.

## Project Structure

```
slm-agentic-benchmarking/
├── src/
│   ├── agents/          # Agent architectures (OneShot, Sequential, Concurrent, GroupChat)
│   ├── benchmarks/      # Benchmark runners (MedQA, BFCL, Criticality, Recall, Episodic Memory)
│   ├── config/          # Azure LLM configuration
│   ├── evaluation/      # Metrics and cost tracking
│   └── utils/           # Utilities (rate limiting, tracing)
├── docs/                # Documentation and project proposals
├── scripts/             # Utility scripts (e.g., reevaluate_bfcl.py)
├── data/                # Datasets (episodic_memory, etc.)
├── results/             # Benchmark results and traces
│   └── combined/        # Combined run results
├── benchmark_runner.py  # Concurrent multi-model runner with live dashboard
├── run_benchmark.py     # Single model CLI
└── pyproject.toml       # Dependencies
```

## Documentation

- **[docs/PLAN.md](docs/PLAN.md)** - Implementation plan and architecture
- **[docs/EXPLANATION.md](docs/EXPLANATION.md)** - Detailed system explanation
- **[docs/CS120_Final_Project.pdf](docs/CS120_Final_Project.pdf)** - Course project documentation
- **[docs/Project Proposal...pdf](docs/)** - Original project proposal

## License

MIT License
