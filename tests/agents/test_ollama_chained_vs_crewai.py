"""
Integration test: compare chained Ollama agents vs old CrewAI agents.

Requires a local Ollama instance with dasd-4b pulled.
Run:  poetry run pytest tests/agents/test_ollama_chained_vs_crewai.py -v -s --timeout=600
Skip: poetry run pytest tests/ -m "not integration"
"""

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OLLAMA_URL = "http://localhost:11434"
MODEL = "dasd-4b"
N_QUESTIONS = 10

# 10 fixed multiple-choice questions (conceptual_combinations style)
# These are simple enough to get answers but diverse enough to test the pipeline.
FIXED_QUESTIONS = [
    {
        "task": 'A "megafauna cat" would be a cat that is: (A) small (B) large (C) fast (D) furry',
        "expected": "B",
        "context": {"benchmark_type": "bigbench"},
    },
    {
        "task": 'An "ice knife" would be a knife made of: (A) metal (B) wood (C) ice (D) stone',
        "expected": "C",
        "context": {"benchmark_type": "bigbench"},
    },
    {
        "task": 'A "fire ant" is an ant that: (A) is on fire (B) lives in fire (C) stings painfully (D) is red',
        "expected": "C",
        "context": {"benchmark_type": "bigbench"},
    },
    {
        "task": 'Which is larger? (A) a mouse (B) an elephant (C) a cat (D) a dog',
        "expected": "B",
        "context": {"benchmark_type": "bigbench"},
    },
    {
        "task": 'What is 7 + 5? (A) 10 (B) 11 (C) 12 (D) 13',
        "expected": "C",
        "context": {"benchmark_type": "bigbench"},
    },
    {
        "task": 'The capital of France is: (A) London (B) Berlin (C) Madrid (D) Paris',
        "expected": "D",
        "context": {"benchmark_type": "bigbench"},
    },
    {
        "task": 'Water boils at what temperature in Celsius? (A) 50 (B) 75 (C) 100 (D) 150',
        "expected": "C",
        "context": {"benchmark_type": "bigbench"},
    },
    {
        "task": 'Which planet is closest to the Sun? (A) Venus (B) Mercury (C) Earth (D) Mars',
        "expected": "B",
        "context": {"benchmark_type": "bigbench"},
    },
    {
        "task": 'A synonym for "happy" is: (A) sad (B) angry (C) joyful (D) tired',
        "expected": "C",
        "context": {"benchmark_type": "bigbench"},
    },
    {
        "task": 'How many legs does a spider have? (A) 4 (B) 6 (C) 8 (D) 10',
        "expected": "C",
        "context": {"benchmark_type": "bigbench"},
    },
]


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _model_available() -> bool:
    try:
        # Resolve config key to actual Ollama model name
        from src.config.azure_llm_config import OLLAMA_MODELS
        actual_name = OLLAMA_MODELS.get(MODEL, {}).get("model", MODEL)

        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
            names = [m.get("name", "") for m in data.get("models", [])]
            return any(actual_name in n or MODEL in n for n in names)
    except Exception:
        return False


pytestmark = pytest.mark.integration
skip_no_ollama = pytest.mark.skipif(
    not _ollama_reachable(), reason="Ollama not reachable at localhost:11434"
)
skip_no_model = pytest.mark.skipif(
    _ollama_reachable() and not _model_available(),
    reason=f"Model {MODEL} not pulled in Ollama",
)


def _run_architecture(agent, questions: List[Dict]) -> List[Dict[str, Any]]:
    """Run an agent on a list of questions, return per-question results."""
    results = []
    for i, q in enumerate(questions):
        start = time.time()
        try:
            resp = agent.respond_to_task(q["task"], q.get("context"))
            latency = time.time() - start
            timed_out = (resp.metadata or {}).get("timed_out", False)
            results.append({
                "idx": i,
                "response": resp.response,
                "reasoning": resp.reasoning[:200],
                "confidence": resp.confidence,
                "latency": latency,
                "timed_out": timed_out,
                "expected": q["expected"],
                "correct": q["expected"].lower() in resp.response.lower(),
                "metadata": resp.metadata,
            })
        except Exception as e:
            latency = time.time() - start
            results.append({
                "idx": i,
                "response": "",
                "reasoning": f"ERROR: {e}",
                "confidence": 0,
                "latency": latency,
                "timed_out": True,
                "expected": q["expected"],
                "correct": False,
                "error": str(e)[:200],
            })
    return results


def _print_summary(name: str, results: List[Dict]):
    completed = sum(1 for r in results if not r.get("timed_out"))
    correct = sum(1 for r in results if r.get("correct"))
    avg_lat = sum(r["latency"] for r in results) / max(len(results), 1)
    timeouts = sum(1 for r in results if r.get("timed_out"))
    print(f"  {name:<25} {completed}/{len(results)} done  "
          f"{correct}/{len(results)} correct  "
          f"avg={avg_lat:.1f}s  timeouts={timeouts}")


@skip_no_ollama
@skip_no_model
class TestOllamaChainedAgents:
    """Test the new chained Ollama agents against the old CrewAI ones."""

    def test_one_shot_baseline(self):
        """One-shot should work identically (implementation unchanged)."""
        from src.agents.ollama_agent import OllamaAgent

        agent = OllamaAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
        results = _run_architecture(agent, FIXED_QUESTIONS)
        _print_summary("one_shot", results)

        completed = sum(1 for r in results if not r.get("timed_out"))
        assert completed == N_QUESTIONS, f"One-shot should complete all {N_QUESTIONS}, got {completed}"

    def test_chained_sequential_completes(self):
        """Chained sequential should complete all questions without timeout."""
        from src.agents.ollama_chained_agent import OllamaSequentialAgent

        agent = OllamaSequentialAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
        results = _run_architecture(agent, FIXED_QUESTIONS)
        _print_summary("sequential_chained", results)

        completed = sum(1 for r in results if not r.get("timed_out"))
        assert completed == N_QUESTIONS, f"Chained sequential should complete all {N_QUESTIONS}, got {completed}"

        avg_lat = sum(r["latency"] for r in results) / N_QUESTIONS
        assert avg_lat < 60, f"Chained sequential avg latency {avg_lat:.1f}s should be < 60s"

    def test_chained_concurrent_completes(self):
        """Chained concurrent should complete all questions without timeout."""
        from src.agents.ollama_chained_agent import OllamaConcurrentAgent

        agent = OllamaConcurrentAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
        results = _run_architecture(agent, FIXED_QUESTIONS)
        _print_summary("concurrent_chained", results)

        completed = sum(1 for r in results if not r.get("timed_out"))
        assert completed == N_QUESTIONS, f"Chained concurrent should complete all {N_QUESTIONS}, got {completed}"

    def test_chained_group_chat_completes(self):
        """Chained group chat should complete all questions without timeout."""
        from src.agents.ollama_chained_agent import OllamaGroupChatAgent

        agent = OllamaGroupChatAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
        results = _run_architecture(agent, FIXED_QUESTIONS)
        _print_summary("group_chat_chained", results)

        completed = sum(1 for r in results if not r.get("timed_out"))
        assert completed == N_QUESTIONS, f"Chained group_chat should complete all {N_QUESTIONS}, got {completed}"

    def test_chained_sequential_has_traces(self):
        """Chained sequential should record 3 trace calls per question."""
        from src.agents.ollama_chained_agent import OllamaSequentialAgent
        from src.utils.trace import TraceCapture

        agent = OllamaSequentialAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
        q = FIXED_QUESTIONS[0]

        with TraceCapture(task_id="test_q0", agent_type="sequential", model=MODEL, input_question=q["task"]) as tc:
            agent.respond_to_task(q["task"], q.get("context"))

        assert len(tc.trace.calls) == 3, f"Sequential should have 3 trace calls, got {len(tc.trace.calls)}"
        roles = [c.role for c in tc.trace.calls]
        assert "Task Analyzer" in roles
        assert "Approach Evaluator" in roles
        assert "Response Generator" in roles

    def test_chained_concurrent_has_traces(self):
        """Chained concurrent should record 4 trace calls per question."""
        from src.agents.ollama_chained_agent import OllamaConcurrentAgent
        from src.utils.trace import TraceCapture

        agent = OllamaConcurrentAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
        q = FIXED_QUESTIONS[0]

        with TraceCapture(task_id="test_q0", agent_type="concurrent", model=MODEL, input_question=q["task"]) as tc:
            agent.respond_to_task(q["task"], q.get("context"))

        assert len(tc.trace.calls) == 4, f"Concurrent should have 4 trace calls, got {len(tc.trace.calls)}"

    def test_chained_group_chat_has_traces(self):
        """Chained group chat should record 4 trace calls per question."""
        from src.agents.ollama_chained_agent import OllamaGroupChatAgent
        from src.utils.trace import TraceCapture

        agent = OllamaGroupChatAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
        q = FIXED_QUESTIONS[0]

        with TraceCapture(task_id="test_q0", agent_type="group_chat", model=MODEL, input_question=q["task"]) as tc:
            agent.respond_to_task(q["task"], q.get("context"))

        assert len(tc.trace.calls) == 4, f"Group chat should have 4 trace calls, got {len(tc.trace.calls)}"

    def test_crewai_sequential_timeout_rate(self):
        """Old CrewAI sequential should show high timeout rate (demonstrating the problem)."""
        from src.agents.ollama_agentic_agent_crewai import OllamaSequentialAgent as CrewAISeq

        agent = CrewAISeq(model=MODEL, ollama_base_url=OLLAMA_URL)
        # Only run 3 questions to avoid wasting time on timeouts
        results = _run_architecture(agent, FIXED_QUESTIONS[:3])
        _print_summary("sequential_crewai (3q)", results)

        timeouts = sum(1 for r in results if r.get("timed_out"))
        # We expect most/all to timeout, but don't hard-assert — just report
        print(f"  CrewAI timeout rate: {timeouts}/3 ({100*timeouts/3:.0f}%)")


@skip_no_ollama
@skip_no_model
def test_full_comparison():
    """
    Run all architectures on all 10 questions and save comparison results.
    This is the main comparison test — run with -s to see the summary table.
    """
    from src.agents.ollama_agent import OllamaAgent
    from src.agents.ollama_chained_agent import (
        OllamaSequentialAgent,
        OllamaConcurrentAgent,
        OllamaGroupChatAgent,
    )

    all_results = {}

    print("\n=== Full Comparison: 4 architectures x 10 questions ===\n")

    # One-shot
    agent = OllamaAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
    all_results["one_shot"] = _run_architecture(agent, FIXED_QUESTIONS)
    _print_summary("one_shot", all_results["one_shot"])

    # Chained sequential
    agent = OllamaSequentialAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
    all_results["sequential_chained"] = _run_architecture(agent, FIXED_QUESTIONS)
    _print_summary("sequential_chained", all_results["sequential_chained"])

    # Chained concurrent
    agent = OllamaConcurrentAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
    all_results["concurrent_chained"] = _run_architecture(agent, FIXED_QUESTIONS)
    _print_summary("concurrent_chained", all_results["concurrent_chained"])

    # Chained group chat
    agent = OllamaGroupChatAgent(model=MODEL, ollama_base_url=OLLAMA_URL)
    all_results["group_chat_chained"] = _run_architecture(agent, FIXED_QUESTIONS)
    _print_summary("group_chat_chained", all_results["group_chat_chained"])

    # Save results
    out_path = Path(__file__).parent / "comparison_results.json"
    # Strip non-serializable metadata
    serializable = {}
    for arch, results in all_results.items():
        serializable[arch] = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "metadata"}
            serializable[arch].append(row)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Summary table
    print(f"\n{'Architecture':<25} {'Done':>5} {'Correct':>8} {'Avg Lat':>8} {'Timeouts':>9}")
    print("-" * 60)
    for arch, results in all_results.items():
        done = sum(1 for r in results if not r.get("timed_out"))
        correct = sum(1 for r in results if r.get("correct"))
        avg_lat = sum(r["latency"] for r in results) / max(len(results), 1)
        tos = sum(1 for r in results if r.get("timed_out"))
        print(f"{arch:<25} {done:>4}/{len(results)} {correct:>4}/{len(results)} {avg_lat:>7.1f}s {tos:>9}")

    # Assert all chained complete
    for arch in ["sequential_chained", "concurrent_chained", "group_chat_chained"]:
        done = sum(1 for r in all_results[arch] if not r.get("timed_out"))
        assert done == N_QUESTIONS, f"{arch} should complete all {N_QUESTIONS}, got {done}"
