# PlanBench

Vendored from LLMs-Planning/plan-bench (paper arxiv.org/abs/2206.10498). Evaluates LLMs on formal PDDL planning: plan generation, optimality, verification, reuse, replanning, execution reasoning, goal reformulation.

## Tasks (t1 to t8_3)

- t1: Plan generation
- t2: Plan optimality
- t3: Plan verification
- t4: Plan reuse
- t5: Plan generalization
- t6: Replanning
- t7: Plan execution
- t8_1: Goal shuffling
- t8_2: Full to partial goal
- t8_3: Partial to full goal

## Domains / configs

blocksworld, depots, logistics (IPC-style). See configs/*.yaml.

## Running via this repo

CLI: python run_benchmark.py --benchmark plan_bench --plan-bench-task t1 --plan-bench-config blocksworld --limit N --model MODEL

Ollama: python scripts/run_ollama_benchmarks.py --benchmarks plan_bench

The runner uses LiteLLM when USE_LITELLM=1 and LITELLM_MODEL are set (see utils/llm_utils.py).

## Full evaluation (plan correctness)

Full evaluation (validating plans with VAL) requires:

- VAL: plan validator. Set VAL=/path/to/val (directory containing the validate binary). Build instructions: see the PlanBench README on GitHub (karthikv792/LLMs-Planning).
- Fast Downward (optional): set FAST_DOWNWARD=/path/to/fast-downward.
- PR2 (optional): set PR2=/path/to/pr2.

VAL and optionally FD/PR2 are typically built on Linux. Without VAL, the pipeline still runs prompt generation and LLM response generation; only the validation step is skipped.

## Environment variables

USE_LITELLM: Set to 1 to use LiteLLM (this repo path).
LITELLM_MODEL: Model id for LiteLLM (e.g. azure/gpt-4o). Set automatically by the runner.
VAL: Path to VAL directory for plan validation.
FAST_DOWNWARD: Optional; path to Fast Downward.
PR2: Optional; path to PR2.
