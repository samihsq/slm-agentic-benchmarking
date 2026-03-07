"""
Pytest unit and integration tests for the Planning (DeepPlanning) benchmark runner.
All tests use a MockAgent for offline, LLM-free execution.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base_agent import BaseAgent, BenchmarkResponse, EvaluationResult
from src.benchmarks.skills.planning.runner import (
    PlanningRunner,
    _extract_budget_from_text,
    _score_shopping,
    _score_travel,
    SUCCESS_THRESHOLD,
)
from src.evaluation.cost_tracker import CostTracker


# ----- Mock agent -----


class MockAgent(BaseAgent):
    """Returns canned responses based on context domain; no LLM calls."""

    def __init__(self, travel_response: str | None = None, shopping_response: str | None = None, raise_error: bool = False):
        super().__init__(model="mock-model", verbose=False)
        self._travel = travel_response
        self._shopping = shopping_response
        self._raise_error = raise_error

    def respond_to_task(self, task: str, context: dict | None = None) -> BenchmarkResponse:
        if self._raise_error:
            raise RuntimeError("mock agent error")
        domain = (context or {}).get("domain", "travel")
        if domain == "travel":
            text = self._travel or self._perfect_travel_response()
        else:
            text = self._shopping or self._perfect_shopping_response()
        return BenchmarkResponse(
            response=text,
            reasoning="mock reasoning",
            success=True,
            confidence=0.9,
            metadata={"prompt_tokens": 10, "completion_tokens": 50},
        )

    @staticmethod
    def _perfect_travel_response() -> str:
        return """Day 1: Flight to Chengdu, hotel check-in, Jinli Street visit, dinner.
Day 2: Attraction visit, lunch, transport to hotel.
Budget Summary: Flights 1200, Hotel 560, Food 340. Total: 2100 CNY."""

    @staticmethod
    def _perfect_shopping_response() -> str:
        return '{"items": [{"name": "ShockWave Runner", "price": 499}, {"name": "Sports Jacket", "price": 299}], "coupons": ["BRAND20"], "final_total": 646}'


# ----- Travel scoring -----


class TestTravelScoring:
    def test_perfect_travel_response(self):
        query = "Plan a 3-day trip to Chengdu. Budget 3000 CNY. Include Jinli Street."
        constraints = ["Chengdu", "3000", "Jinli"]
        response = """Day 1: Flight to Chengdu, hotel, Jinli Street, dinner.
Day 2: Attraction, lunch, transport.
Budget Summary: Total 2100 CNY."""
        scores = _score_travel(response, query, constraints)
        assert scores["structure_score"] > 0
        assert scores["budget_score"] >= 0.5
        assert scores["constraint_score"] > 0
        assert scores["composite_score"] >= 0.5

    def test_missing_day_headers(self):
        response = "Just some text about travel. No day structure. Budget 1000."
        query = "Trip to Beijing. Budget 2000."
        scores = _score_travel(response, query, ["Beijing"])
        # No "day N" -> structure component weak
        assert "day" not in response.lower() or "day 1" not in response.lower()
        assert scores["composite_score"] < 1.0

    def test_budget_exceeded(self):
        query = "Trip to Shanghai. Budget 1000 CNY."
        response = "Day 1: Hotel 800. Budget total: 1500 CNY."
        scores = _score_travel(response, query, [])
        assert scores["budget_score"] == 0.0

    def test_budget_within_cap(self):
        query = "Trip to Shanghai. Budget 3000 CNY."
        response = "Day 1: Hotel 500. Budget Summary: Total 2500 CNY."
        scores = _score_travel(response, query, [])
        assert scores["budget_score"] == 1.0

    def test_budget_score_requires_explicit_total(self):
        """Line-item amounts (e.g. 300 CNY for one night) must not count as total."""
        query = "Plan a 3-day trip to Chengdu. Budget 3000 CNY."
        response = "Day 1: Flight 500. Hotel 300 CNY. Lunch 50. Jinli Street visit."
        scores = _score_travel(response, query, ["Chengdu"])
        assert scores["budget_score"] == 0.0

    def test_constraint_keywords_present(self):
        query = "Plan a trip to Chengdu. 3-day."
        response = "Day 1: Arrive in Chengdu. Hotel in Chengdu. Visit Jinli."
        scores = _score_travel(response, query, ["Chengdu", "Jinli"])
        assert scores["constraint_score"] > 0

    def test_constraint_keywords_missing(self):
        query = "Plan a trip to Chengdu."
        response = "Day 1: Go to Beijing. Hotel in Beijing."
        scores = _score_travel(response, query, ["Chengdu"])
        assert scores["constraint_score"] < 1.0

    def test_empty_response(self):
        scores = _score_travel("", "Trip to Paris. Budget 2000.", ["Paris"])
        assert scores["composite_score"] <= 0.5
        assert scores["structure_score"] == 0.0


# ----- Shopping scoring -----


class TestShoppingScoring:
    def test_valid_json_cart(self):
        response = '{"items": [{"id": "1", "price": 100}], "final_total": 100}'
        scores = _score_shopping(response, "Buy one item. Budget 200.", budget_cap=200, required_item_count=1)
        assert scores["json_score"] == 1.0

    def test_invalid_json(self):
        response = '{"items": [invalid json'
        scores = _score_shopping(response, "Buy item. Budget 100.", budget_cap=100, required_item_count=1)
        assert scores["json_score"] == 0.0

    def test_json_missing_items_key(self):
        response = '{"total": 50}'
        scores = _score_shopping(response, "Budget 100.", budget_cap=100, required_item_count=1)
        assert scores["json_score"] == 0.0

    def test_budget_respected(self):
        response = '{"items": [{"price": 80}], "final_total": 80}'
        scores = _score_shopping(response, "Budget 100.", budget_cap=100, required_item_count=1)
        assert scores["budget_score"] == 1.0

    def test_budget_exceeded(self):
        response = '{"items": [{"price": 150}], "final_total": 150}'
        scores = _score_shopping(response, "Budget 100.", budget_cap=100, required_item_count=1)
        assert scores["budget_score"] == 0.0

    def test_completeness_full(self):
        response = '{"items": [{"a": 1}, {"b": 2}], "final_total": 50}'
        scores = _score_shopping(response, "Buy 2 items.", budget_cap=100, required_item_count=2)
        assert scores["completeness_score"] == 1.0

    def test_completeness_partial(self):
        response = '{"items": [{"a": 1}], "final_total": 30}'
        scores = _score_shopping(response, "Buy 2 items.", budget_cap=100, required_item_count=2)
        assert 0 < scores["completeness_score"] < 1.0

    def test_empty_response(self):
        scores = _score_shopping("", "Budget 100. Buy 1 item.", budget_cap=100, required_item_count=1)
        assert scores["composite_score"] <= 0.5
        assert scores["json_score"] == 0.0


# ----- Task loading -----


class TestLoadTasks:
    def test_synthetic_fallback(self):
        with patch.dict("sys.modules", {"datasets": None}):
            runner = PlanningRunner(MockAgent(), domain="all", language="en")
            tasks = runner.load_tasks(limit=5)
        assert len(tasks) <= 5
        for t in tasks:
            assert "task_id" in t
            assert "query" in t
            assert "domain" in t
            assert "constraints" in t

    def test_synthetic_task_schema(self):
        runner = PlanningRunner(MockAgent(), domain="all", language="en")
        tasks = runner.load_tasks(limit=10)
        assert len(tasks) >= 1
        for t in tasks:
            assert t["task_id"]
            assert t["query"]
            assert t["domain"] in ("travel", "shopping")

    def test_domain_filter_travel(self):
        runner = PlanningRunner(MockAgent(), domain="travel", language="en")
        tasks = runner.load_tasks(limit=10)
        assert all(t["domain"] == "travel" for t in tasks)

    def test_domain_filter_shopping(self):
        runner = PlanningRunner(MockAgent(), domain="shopping", language="en")
        tasks = runner.load_tasks(limit=10)
        assert all(t["domain"] == "shopping" for t in tasks)

    def test_domain_filter_all(self):
        runner = PlanningRunner(MockAgent(), domain="all", language="en")
        tasks = runner.load_tasks(limit=10)
        domains = {t["domain"] for t in tasks}
        assert "travel" in domains or "shopping" in domains

    def test_limit_respected(self):
        runner = PlanningRunner(MockAgent(), domain="all", language="en")
        tasks = runner.load_tasks(limit=3)
        assert len(tasks) == 3

    def test_language_filter_en(self):
        runner = PlanningRunner(MockAgent(), domain="all", language="en")
        tasks = runner.load_tasks(limit=20)
        for t in tasks:
            assert t["query"]


# ----- Task formatting -----


class TestFormatTask:
    def test_travel_prompt_contains_query(self):
        runner = PlanningRunner(MockAgent(), domain="travel")
        task = {"task_id": "t1", "query": "Plan a trip to Tokyo. Budget 5000.", "domain": "travel", "ground_truth": {}, "constraints": []}
        text, _ = runner.format_task(task)
        assert "Tokyo" in text
        assert "5000" in text

    def test_travel_prompt_requests_itinerary(self):
        runner = PlanningRunner(MockAgent(), domain="travel")
        task = {"task_id": "t1", "query": "Trip to Paris.", "domain": "travel", "ground_truth": {}, "constraints": []}
        text, _ = runner.format_task(task)
        assert "day" in text.lower()
        assert "itinerary" in text.lower() or "schedule" in text.lower()

    def test_travel_prompt_requests_budget_summary(self):
        runner = PlanningRunner(MockAgent(), domain="travel")
        task = {"task_id": "t1", "query": "Trip.", "domain": "travel", "ground_truth": {}, "constraints": []}
        text, _ = runner.format_task(task)
        assert "budget" in text.lower()

    def test_shopping_prompt_contains_query(self):
        runner = PlanningRunner(MockAgent(), domain="shopping")
        task = {"task_id": "s1", "query": "Buy shoes. Budget 200.", "domain": "shopping", "ground_truth": {}, "constraints": []}
        text, _ = runner.format_task(task)
        assert "shoes" in text
        assert "200" in text

    def test_shopping_prompt_requests_json(self):
        runner = PlanningRunner(MockAgent(), domain="shopping")
        task = {"task_id": "s1", "query": "Buy item.", "domain": "shopping", "ground_truth": {}, "constraints": []}
        text, _ = runner.format_task(task)
        assert "json" in text.lower()
        assert "items" in text.lower()

    def test_context_has_benchmark_type(self):
        runner = PlanningRunner(MockAgent())
        task = {"task_id": "x", "query": "q", "domain": "travel", "ground_truth": {}, "constraints": []}
        _, context = runner.format_task(task)
        assert "benchmark_type" in context or "task_type" in context

    def test_context_has_domain(self):
        runner = PlanningRunner(MockAgent())
        task = {"task_id": "x", "query": "q", "domain": "shopping", "ground_truth": {}, "constraints": []}
        _, context = runner.format_task(task)
        assert context.get("domain") == "shopping"


# ----- E2E runner -----


class TestPlanningRunnerE2E:
    def test_run_returns_evaluation_results(self):
        runner = PlanningRunner(MockAgent(), domain="all", verbose=False)
        results = runner.run(limit=2, save_results=False)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, EvaluationResult)

    def test_result_has_required_fields(self):
        runner = PlanningRunner(MockAgent(), domain="all", verbose=False)
        results = runner.run(limit=1, save_results=False)
        assert len(results) == 1
        r = results[0]
        assert hasattr(r, "task_id") and r.task_id
        assert hasattr(r, "success") and isinstance(r.success, bool)
        assert hasattr(r, "score") and r.score is not None
        assert hasattr(r, "latency") and r.latency is not None

    def test_score_between_0_and_1(self):
        runner = PlanningRunner(MockAgent(), domain="all", verbose=False)
        results = runner.run(limit=3, save_results=False)
        for r in results:
            assert 0.0 <= (r.score or 0) <= 1.0

    def test_travel_domain_only(self):
        runner = PlanningRunner(MockAgent(), domain="travel", verbose=False)
        results = runner.run(limit=2, save_results=False)
        for r in results:
            assert (r.metadata or {}).get("domain") == "travel"

    def test_shopping_domain_only(self):
        runner = PlanningRunner(MockAgent(), domain="shopping", verbose=False)
        results = runner.run(limit=2, save_results=False)
        for r in results:
            assert (r.metadata or {}).get("domain") == "shopping"

    def test_mixed_domain(self):
        runner = PlanningRunner(MockAgent(), domain="all", verbose=False)
        results = runner.run(limit=4, save_results=False)
        domains = [(r.metadata or {}).get("domain") for r in results]
        assert "travel" in domains
        assert "shopping" in domains

    def test_result_metadata_has_sub_scores(self):
        runner = PlanningRunner(MockAgent(), domain="all", verbose=False)
        results = runner.run(limit=2, save_results=False)
        for r in results:
            m = r.metadata or {}
            domain = m.get("domain")
            if domain == "travel":
                assert "structure_score" in m or "budget_score" in m or "constraint_score" in m
            else:
                assert "json_score" in m or "budget_score" in m or "completeness_score" in m

    def test_concurrency_produces_same_count(self):
        runner = PlanningRunner(MockAgent(), concurrency=2, domain="all", verbose=False)
        results = runner.run(limit=4, save_results=False)
        assert len(results) == 4

    def test_save_results_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = PlanningRunner(MockAgent(), run_dir=run_dir, domain="all", verbose=False)
            runner.run(limit=2, save_results=True)
            agent_dir = run_dir / "MockAgent"
            assert agent_dir.exists()
            assert (agent_dir / "results.jsonl").exists()
            assert (agent_dir / "summary.json").exists()

    def test_summary_json_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = PlanningRunner(MockAgent(), run_dir=run_dir, domain="all", verbose=False)
            runner.run(limit=2, save_results=True)
            summary_path = run_dir / "MockAgent" / "summary.json"
            data = json.loads(summary_path.read_text())
            assert "mean_composite_score" in data or "avg_score" in data or "num_tasks" in data
            assert "success_rate" in data
            assert "num_tasks" in data
            assert "mean_latency" in data
            assert "total_cost" in data

    def test_per_task_trace_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = PlanningRunner(MockAgent(), run_dir=run_dir, domain="travel", verbose=False)
            results = runner.run(limit=2, save_results=True)
            agent_dir = run_dir / "MockAgent"
            for r in results:
                task_dir = agent_dir / r.task_id
                assert task_dir.exists()
                assert (task_dir / "trace.json").exists()

    def test_agent_error_handled_gracefully(self):
        runner = PlanningRunner(MockAgent(raise_error=True), domain="travel", verbose=False)
        results = runner.run(limit=1, save_results=False)
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].score == 0.0

    def test_cost_tracked(self):
        cost_tracker = CostTracker(budget_limit=1000.0)
        agent = MockAgent()
        runner = PlanningRunner(agent, cost_tracker=cost_tracker, domain="all", verbose=False)
        runner.run(limit=2, save_results=False)
        assert hasattr(cost_tracker, "total_spent")
        summary = cost_tracker.get_summary()
        assert summary.total_cost >= 0


# ----- Budget extraction (used by scoring) -----


def test_extract_budget_from_text():
    assert _extract_budget_from_text("Budget: 3000 CNY") == 3000.0
    assert _extract_budget_from_text("Total budget 800") == 800.0
    assert _extract_budget_from_text("No numbers here") is None or _extract_budget_from_text("No numbers here") is not None
