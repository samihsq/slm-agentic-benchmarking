# Plan: Replace CrewAI with Direct API Call Chains for Ollama Agents

Status: IN PROGRESS
Created: 2026-03-11

## Problem Statement

CrewAI's ReAct format loop causes 92-100% timeout rates on SLMs (< 4B parameters).
The framework demands structured `Thought:/Action:/Observation:` tokens that small
models can't reliably produce within the 60-90 second timeout window.

Evidence from the March 11 sweep (BBL24, 308 questions per combo):
- gemma3-4b (dense 4B): 98.4% timeout, 0.5% accuracy in sequential
- gemma3n-e2b (2B MoE): 96.8% timeout, 0.5% accuracy in sequential
- qwen3-0.6b (0.6B): 100% timeout, 0.0% accuracy in sequential
- gpt-oss-20b (20B): 100% timeout, 0.0% accuracy in sequential

The benchmark only scores the final answer text, so the ReAct scaffolding adds
compute cost (23-43x latency overhead) with zero accuracy benefit.

## Scope

Replace **only the Ollama agentic agents** in `src/agents/ollama_agentic_agent.py`.
The Azure CrewAI agents (`sequential_agent.py`, `concurrent_agent.py`,
`group_chat_agent.py`) stay untouched -- they use larger models where CrewAI works.

## Architecture: Before vs After

### Before (CrewAI)
```
Task -> CrewAI Crew(Process.sequential) -> ReAct loop per agent
     -> Analyzer Agent (Thought/Action/Obs cycle, retries on format failure)
     -> Evaluator Agent (same cycle)
     -> Responder Agent (same cycle)
     -> kickoff_with_timeout(crew, 90s)
```
Small models get stuck in the ReAct retry loop, hit the 90s timeout, return empty.

### After (Direct API Chains)
```
Task -> _call_ollama(analyzer_prompt + task) -> analysis text (30s timeout)
     -> _call_ollama(evaluator_prompt + task + analysis) -> strategy text (30s timeout)
     -> _call_ollama(responder_prompt + task + analysis + strategy) -> final JSON (30s timeout)
```
Each call is a simple prompt-in/text-out HTTP request. No format constraints beyond
what the prompt asks for. If a call fails, the chain continues with empty input for
that stage rather than blocking the entire pipeline.

---

## Phase 1: New Implementation

### File: `src/agents/ollama_chained_agent.py` (~250 lines)

Three classes, all extending `BaseAgent`:

#### 1. OllamaSequentialAgent -- 3 chained calls

```python
class OllamaSequentialAgent(BaseAgent):
    def __init__(self, model, verbose, ollama_base_url, per_call_timeout=30):
        super().__init__(model=model, verbose=verbose, max_iterations=1)
        self._ollama = OllamaAgent(model=model, ollama_base_url=ollama_base_url)
        self.per_call_timeout = per_call_timeout
```

Call chain:
1. System = `get_system_prompt("bigbench", "sequential_analyzer")`
   User = `"TASK: {full_task}"`
   -> `analysis_text` (postprocessed to strip <think> tags)
2. System = `get_system_prompt("bigbench", "sequential_evaluator")`
   User = `"TASK: {full_task}\n\nANALYSIS:\n{analysis_text}"`
   -> `strategy_text`
3. System = `get_system_prompt("bigbench", "sequential_responder")`
   User = `"ORIGINAL TASK: {full_task}\n\nANALYSIS:\n{analysis_text}\n\nSTRATEGY:\n{strategy_text}"`
   -> final answer (parsed via `parse_json_response()`)

Each call records to `TraceCapture.record(role="Task Analyzer", ...)` etc.

#### 2. OllamaConcurrentAgent -- 3 parallel + 1 synthesis

```python
class OllamaConcurrentAgent(BaseAgent):
    # same __init__ pattern
```

Call structure:
1. Fire 3 calls in parallel using `concurrent.futures.ThreadPoolExecutor(max_workers=3)`:
   - analyst: `get_system_prompt("bigbench", "concurrent_analyst")` + task
   - researcher: `get_system_prompt("bigbench", "concurrent_researcher")` + task
   - critic: `get_system_prompt("bigbench", "concurrent_critic")` + task
2. Collect results (30s timeout each; failed calls -> empty string)
3. Synthesis call:
   System = `get_system_prompt("bigbench", "concurrent_synthesizer")`
   User = `"TASK: {full_task}\n\nANALYSIS:\n{analyst_text}\n\nRESEARCH:\n{researcher_text}\n\nCRITIQUE:\n{critic_text}"`
   -> final answer

Records all 4 calls to TraceCapture (analyst, researcher, critic, synthesizer).

#### 3. OllamaGroupChatAgent -- 4 chained calls

Call chain:
1. `groupchat_proposer` + task -> proposal
2. `groupchat_critic` + task + proposal -> critique
3. `groupchat_advisor` + task + proposal + critique -> advice
4. `groupchat_moderator` + task + proposal + critique + advice -> final JSON

Records all 4 calls to TraceCapture.

### Design Decisions

- **Composition over inheritance**: Each class composes an internal `OllamaAgent`
  instance to reuse `_call_ollama()`, `_postprocess()`, retry logic, and
  thinking-model quirks. Does NOT inherit from `OllamaAgent` (the interface
  contract is `BaseAgent.respond_to_task()`).

- **Same prompts**: Uses the existing `src/agents/prompts/bigbench/prompts.yaml`
  role prompts unchanged. The prompts define reasoning persona, not ReAct format.

- **Per-call timeout**: Default 30s per Ollama HTTP call (via `urllib` timeout
  parameter, already supported in `_call_ollama`). Total worst-case for
  sequential: 3 x 30s = 90s, same budget as before but each call can succeed
  independently.

- **Graceful degradation**: If an intermediate call fails (timeout or HTTP error),
  the chain continues with an empty string for that stage's output. The final
  responder/synthesizer/moderator still gets the task and whatever prior stages
  succeeded. This is strictly better than CrewAI's all-or-nothing timeout.

- **Intermediate postprocessing**: Each intermediate result goes through
  `OllamaAgent._postprocess()` to strip `<think>` tags and extract clean text
  before feeding to the next stage. This prevents thinking tokens from polluting
  downstream prompts.

---

## Phase 2: Wire Up

### `src/agents/__init__.py`
Change import source from `ollama_agentic_agent` to `ollama_chained_agent`:
```python
# Before:
from .ollama_agentic_agent import OllamaSequentialAgent, OllamaConcurrentAgent, OllamaGroupChatAgent
# After:
from .ollama_chained_agent import OllamaSequentialAgent, OllamaConcurrentAgent, OllamaGroupChatAgent
```
Class names stay identical -- nothing downstream changes.

### `src/agents/ollama_agentic_agent.py`
Rename to `ollama_agentic_agent_crewai.py`. Preserved for comparison testing, not
imported by default.

### Files that require NO changes:
- `scripts/run_bigbench_lite_sweep.py` -- dispatches by class name, unchanged
- `scripts/modal_ollama_bigbench.py` -- calls sweep script as subprocess
- `src/agents/ollama_agent.py` -- one-shot stays as-is
- `src/agents/sequential_agent.py` -- Azure CrewAI stays as-is
- `src/agents/concurrent_agent.py` -- Azure CrewAI stays as-is
- `src/agents/group_chat_agent.py` -- Azure CrewAI stays as-is
- `src/agents/base_agent.py` -- no changes
- `src/agents/prompts/bigbench/prompts.yaml` -- same prompts reused
- `src/benchmarks/skills/bigbench/runner.py` -- only sees `BaseAgent` interface
- `src/utils/trace.py` -- `TraceCapture.record()` API unchanged

---

## Phase 3: Comparison Test

### File: `tests/agents/test_ollama_chained_vs_crewai.py`

Integration test requiring a local Ollama instance with `dasd-4b` model pulled.

#### Setup
- 10 BBL24 questions from a fixed task (`conceptual_combinations` -- 10 MC questions,
  deterministic, good spread of difficulty)
- Load questions via `BigBenchRunner` task loading (same path as real sweep)
- Skip test if Ollama not reachable (`@pytest.mark.integration`)

#### Test Matrix
For each of the 10 questions, run:
1. `OllamaAgent` (one-shot) -- baseline, identical in both old and new
2. Old `OllamaSequentialAgent` (from `ollama_agentic_agent_crewai.py`)
3. New `OllamaSequentialAgent` (from `ollama_chained_agent.py`)
4. New `OllamaConcurrentAgent` (from `ollama_chained_agent.py`)
5. New `OllamaGroupChatAgent` (from `ollama_chained_agent.py`)

#### Captured per question per architecture:
```json
{
  "question_idx": 0,
  "task_text": "...",
  "architecture": "sequential_chained",
  "response": "B",
  "reasoning": "...",
  "confidence": 0.8,
  "latency_seconds": 12.3,
  "timed_out": false,
  "trace_calls": [
    {"role": "Task Analyzer", "input_len": 450, "output_len": 200, "latency": 3.1},
    {"role": "Approach Evaluator", "input_len": 650, "output_len": 180, "latency": 4.2},
    {"role": "Response Generator", "input_len": 830, "output_len": 150, "latency": 5.0}
  ],
  "metadata": { ... }
}
```

#### Assertions

1. **Completion**: All 4 chained architectures complete 10/10 questions without
   timeout (the primary improvement over CrewAI).

2. **Latency**: Chained sequential average latency < 30s per question
   (vs ~90s with CrewAI). Chained concurrent should be faster than sequential
   due to parallel first stage.

3. **Trace integrity**: Each architecture produces the expected number of
   intermediate trace records:
   - one_shot: 0 intermediate calls (single call)
   - sequential: 3 calls (analyzer, evaluator, responder)
   - concurrent: 4 calls (analyst, researcher, critic, synthesizer)
   - group_chat: 4 calls (proposer, critic, advisor, moderator)

4. **One-shot parity**: One-shot results are identical between old and new
   (sanity check -- one-shot implementation doesn't change).

5. **No accuracy assertion** between old CrewAI and new chained: the answers
   will differ because the prompt pipeline is fundamentally different (no ReAct
   formatting). The point is that new chained actually *produces* answers instead
   of timing out.

#### Output
Results saved to `tests/agents/comparison_results.json` for manual inspection.
Console output shows a summary table:

```
Architecture          Completed  Avg Lat  Timeouts  Accuracy
one_shot              10/10      8.0s     0         7/10
sequential_crewai     2/10       87.3s    8         0/10
sequential_chained    10/10      14.2s    0         6/10
concurrent_chained    10/10      11.8s    0         7/10
group_chat_chained    10/10      18.5s    0         5/10
```

#### Running
```bash
# Requires: ollama running locally with dasd-4b pulled
poetry run pytest tests/agents/test_ollama_chained_vs_crewai.py -v -s --timeout=600

# Skip if no ollama:
poetry run pytest tests/ -m "not integration"
```

---

## Phase 4: Cleanup

1. Run `poetry run ruff check .` and `poetry run black .`
2. Verify `poetry run pytest tests/` passes (non-integration tests)
3. Update CLAUDE.md architecture section to note Ollama agents no longer use CrewAI

---

## Risk Assessment

- **Low risk**: The new implementation is strictly simpler (direct HTTP calls vs
  CrewAI orchestration). The Ollama API call logic is already proven in `OllamaAgent`.
- **Behavior change**: Intermediate reasoning will differ because we're not using
  CrewAI's ReAct format. This is intentional -- the ReAct loop is what causes timeouts.
- **No data loss**: Old results in `results/bigbench_lite/ollama/` stay valid.
  New runs produce the same file layout (summary.json, results.jsonl, trace.json).
- **Reversible**: The old CrewAI implementation is preserved as
  `ollama_agentic_agent_crewai.py`. To revert, change one import line.
