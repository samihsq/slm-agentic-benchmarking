"""
Unit tests for PlanBench integration: prompt structure, LiteLLM adapter, runner.
Covers: VAL not set (not evaluated), evaluated results, summary stats, incremental write,
and VAL_API_URL routing (validate_plan and get_val_feedback call API when env var is set).
No real API or VAL binary required.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base_agent import BaseAgent
from src.benchmarks.skills.plan_bench.runner import (
    PlanBenchRunner,
    TASK_TO_NAME,
    _LIMIT_SENTINEL_INSTANCE_ID,
    _engine_slug,
    _plan_bench_root,
    _results_from_structured,
)


def _plan_bench_prompts_path() -> Path:
    root = _plan_bench_root()
    return root / "prompts" / "blocksworld" / "task_1_plan_generation.json"


class TestResultsFromStructured:
    """Unit tests for _results_from_structured (evaluated vs not evaluated)."""

    def test_evaluated_true_success(self):
        structured = {
            "instances": [
                {"instance_id": 2, "llm_correct": True, "llm_raw_response": "(pick-up b1)"},
            ]
        }
        out = _results_from_structured(structured, [2])
        assert len(out) == 1
        assert out[0]["evaluated"] is True
        assert out[0]["success"] is True
        assert out[0]["score"] == 1.0
        assert out[0]["llm_correct"] is True

    def test_evaluated_true_failure(self):
        structured = {
            "instances": [
                {"instance_id": 3, "llm_correct": False, "llm_raw_response": "bad"},
            ]
        }
        out = _results_from_structured(structured, [3])
        assert len(out) == 1
        assert out[0]["evaluated"] is True
        assert out[0]["success"] is False
        assert out[0]["score"] == 0.0

    def test_not_evaluated_no_llm_correct(self):
        """When VAL was not run, llm_correct is missing -> not evaluated."""
        structured = {
            "instances": [
                {"instance_id": 2, "llm_raw_response": "(pick-up b1)\n (put-down b1)"},
            ]
        }
        out = _results_from_structured(structured, [2])
        assert len(out) == 1
        assert out[0]["evaluated"] is False
        assert out[0]["success"] is None
        assert out[0]["score"] is None
        assert out[0]["llm_correct"] is None

    def test_not_evaluated_llm_correct_binary_used_when_present(self):
        structured = {
            "instances": [
                {"instance_id": 2, "llm_correct_binary": True, "llm_raw_response": "x"},
            ]
        }
        out = _results_from_structured(structured, [2])
        assert len(out) == 1
        assert out[0]["evaluated"] is True
        assert out[0]["success"] is True
        assert out[0]["score"] == 1.0

    def test_filters_by_instance_ids(self):
        structured = {
            "instances": [
                {"instance_id": 1, "llm_correct": True, "llm_raw_response": "a"},
                {"instance_id": 2, "llm_correct": False, "llm_raw_response": "b"},
                {"instance_id": 3, "llm_correct": None, "llm_raw_response": "c"},
            ]
        }
        out = _results_from_structured(structured, [1, 3])
        assert len(out) == 2
        assert out[0]["instance_id"] == 1 and out[0]["evaluated"] is True
        assert out[1]["instance_id"] == 3 and out[1]["evaluated"] is False

    def test_mixed_evaluated_and_not(self):
        structured = {
            "instances": [
                {"instance_id": 2, "llm_correct": True, "llm_raw_response": "a"},
                {"instance_id": 3, "llm_raw_response": "b"},
                {"instance_id": 4, "llm_correct": False, "llm_raw_response": "c"},
            ]
        }
        out = _results_from_structured(structured, [2, 3, 4])
        assert len(out) == 3
        assert [r["evaluated"] for r in out] == [True, False, True]
        assert [r["success"] for r in out] == [True, None, False]
        assert [r["score"] for r in out] == [1.0, None, 0.0]


class TestPlanBenchPromptStructure:
    """Assert vendored prompts have expected JSON structure."""

    def test_blocksworld_t1_prompt_file_exists(self):
        path = _plan_bench_prompts_path()
        assert path.exists(), f"Pre-generated prompts not found: {path}"

    def test_blocksworld_t1_prompt_structure(self):
        path = _plan_bench_prompts_path()
        with open(path) as f:
            data = json.load(f)
        assert "instances" in data
        instances = data["instances"]
        assert len(instances) >= 1
        for inst in instances[:3]:
            assert "instance_id" in inst
            assert "query" in inst
            assert isinstance(inst["query"], str)
            assert len(inst["query"]) > 0


def _load_llm_utils():
    """Load vendored llm_utils module for testing."""
    import importlib.util
    plan_bench_root = Path(__file__).resolve().parents[2] / "benchmarks" / "plan_bench"
    llm_utils_path = plan_bench_root / "utils" / "llm_utils.py"
    if not llm_utils_path.exists():
        pytest.skip("Vendored plan_bench not present")
    spec = importlib.util.spec_from_file_location("llm_utils_test", llm_utils_path)
    llm_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_utils)
    return llm_utils


class TestPlanBenchAdapter:
    """LiteLLM adapter: when send_query_litellm is called, litellm.completion is invoked with expected args."""

    def test_send_query_litellm_calls_litellm_with_stop(self):
        llm_utils = _load_llm_utils()

        with patch.dict(os.environ, {"LITELLM_MODEL": "test-model"}):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock(
                    choices=[MagicMock(message=MagicMock(content=" (pick-up b1)\n (put-down b1)"))]
                )
                out = llm_utils.send_query_litellm(
                    "test query", "eng", 100, stop="[STATEMENT]"
                )
                mock_completion.assert_called_once()
                call_kw = mock_completion.call_args[1]
                assert call_kw.get("model") == "test-model"
                assert "[STATEMENT]" in (call_kw.get("stop") or [])
                assert out.strip() == "(pick-up b1)\n (put-down b1)"

    def test_send_query_litellm_no_retry_for_non_ollama_empty_content(self):
        """Non-Ollama model returning empty content is returned as-is (no retry)."""
        llm_utils = _load_llm_utils()

        with patch.dict(os.environ, {"LITELLM_MODEL": "azure/gpt-4"}):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock(
                    choices=[MagicMock(message=MagicMock(content=""))]
                )
                out = llm_utils.send_query_litellm("test query", "eng", 100)
                assert mock_completion.call_count == 1
                assert out == ""

    def test_send_query_litellm_ollama_empty_content_retries_with_think_false(self):
        """Ollama thinking model returns empty content -> retry with think=False and return result."""
        llm_utils = _load_llm_utils()

        with patch.dict(os.environ, {"LITELLM_MODEL": "ollama/dasd-4b"}):
            with patch("litellm.completion") as mock_completion:
                empty_resp = MagicMock(choices=[MagicMock(message=MagicMock(content=""))])
                real_resp = MagicMock(choices=[MagicMock(message=MagicMock(content="(pick-up b1)\n(put-down b1)"))])
                mock_completion.side_effect = [empty_resp, real_resp]

                out = llm_utils.send_query_litellm("test query", "eng", 100, stop="[STATEMENT]")

                assert mock_completion.call_count == 2, "Should retry when Ollama content is empty"
                first_call_kw = mock_completion.call_args_list[0][1]
                assert "extra_body" not in first_call_kw or first_call_kw.get("extra_body", {}).get("think") is not False
                retry_call_kw = mock_completion.call_args_list[1][1]
                assert retry_call_kw.get("extra_body", {}).get("think") is False
                assert out == "(pick-up b1)\n(put-down b1)"

    def test_send_query_litellm_ollama_retry_returns_empty_string_if_still_empty(self):
        """If retry also returns empty, return empty string (don't crash)."""
        llm_utils = _load_llm_utils()

        with patch.dict(os.environ, {"LITELLM_MODEL": "ollama/some-model"}):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock(
                    choices=[MagicMock(message=MagicMock(content=""))]
                )
                out = llm_utils.send_query_litellm("test query", "eng", 100)
                assert mock_completion.call_count == 2
                assert out == ""


class TestValApiRouting:
    """Unit tests for VAL_API_URL routing in vendored validate_plan and get_val_feedback."""

    def _load_vendored_utils(self):
        """Load the vendored plan_bench utils package with stubs for heavy/unavailable deps."""
        import sys
        import types

        plan_bench_root = Path(__file__).resolve().parents[2] / "benchmarks" / "plan_bench"
        if not (plan_bench_root / "utils" / "__init__.py").exists():
            pytest.skip("Vendored plan_bench not present")

        def _stub(name, **attrs):
            m = sys.modules.get(name) or types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
            return m

        _stub("tarski")
        _stub("tarski.io", PDDLReader=type("PDDLReader", (), {}))
        _stub("tarski.syntax")
        tarski_formulas = _stub("tarski.syntax.formulas")
        tarski_formulas.__all__ = []
        transformers = _stub("transformers")
        transformers.StoppingCriteriaList = type("StoppingCriteriaList", (), {})
        transformers.StoppingCriteria = type("StoppingCriteria", (), {})

        pb_str = str(plan_bench_root)
        added = pb_str not in sys.path
        if added:
            sys.path.insert(0, pb_str)

        # Clear any previously cached utils to force a fresh load with our stubs
        for key in list(sys.modules.keys()):
            if key == "utils" or key.startswith("utils."):
                del sys.modules[key]

        try:
            import utils
            return utils
        finally:
            if added and pb_str in sys.path:
                sys.path.remove(pb_str)

    def test_validate_plan_calls_api_when_val_api_url_set(self, tmp_path):
        mod = self._load_vendored_utils()

        domain_file = tmp_path / "domain.pddl"
        instance_file = tmp_path / "instance.pddl"
        plan_file = tmp_path / "plan.txt"
        domain_file.write_text("(define (domain test))")
        instance_file.write_text("(define (problem p) (:domain test))")
        plan_file.write_text("(pick-up b1)")

        api_response = json.dumps({"valid": True, "output": "Plan valid\n"}).encode()

        class FakeResponse:
            def read(self):
                return api_response
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with patch.dict(os.environ, {"VAL_API_URL": "http://localhost:8001"}):
            with patch("urllib.request.urlopen", return_value=FakeResponse()) as mock_urlopen:
                result = mod.validate_plan(str(domain_file), str(instance_file), str(plan_file))

        assert result is True
        mock_urlopen.assert_called_once()
        request_obj = mock_urlopen.call_args[0][0]
        body = json.loads(request_obj.data)
        assert "(define (domain test))" in body["domain"]
        assert body["verbose"] is False

    def test_validate_plan_returns_false_when_api_says_invalid(self, tmp_path):
        mod = self._load_vendored_utils()

        domain_file = tmp_path / "domain.pddl"
        instance_file = tmp_path / "instance.pddl"
        plan_file = tmp_path / "plan.txt"
        domain_file.write_text("(define (domain test))")
        instance_file.write_text("(define (problem p) (:domain test))")
        plan_file.write_text("(bad-action)")

        api_response = json.dumps({"valid": False, "output": "Plan invalid\n"}).encode()

        class FakeResponse:
            def read(self):
                return api_response
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with patch.dict(os.environ, {"VAL_API_URL": "http://localhost:8001"}):
            with patch("urllib.request.urlopen", return_value=FakeResponse()):
                result = mod.validate_plan(str(domain_file), str(instance_file), str(plan_file))

        assert result is False

    def test_validate_plan_uses_local_binary_when_no_val_api_url(self, tmp_path):
        mod = self._load_vendored_utils()

        domain_file = tmp_path / "domain.pddl"
        instance_file = tmp_path / "instance.pddl"
        plan_file = tmp_path / "plan.txt"
        domain_file.write_text("(define (domain test))")
        instance_file.write_text("(define (problem p) (:domain test))")
        plan_file.write_text("(pick-up b1)")

        env = {k: v for k, v in os.environ.items() if k != "VAL_API_URL"}
        env["VAL"] = "/fake/val"

        with patch.dict(os.environ, env, clear=True):
            with patch("os.popen") as mock_popen:
                mock_popen.return_value.read.return_value = "Plan valid\n"
                result = mod.validate_plan(str(domain_file), str(instance_file), str(plan_file))

        assert result is True
        mock_popen.assert_called_once()
        assert "/fake/val/validate" in mock_popen.call_args[0][0]


class TestValApiService:
    """Unit tests for the FastAPI val_api service (no real VAL binary required)."""

    def _get_test_client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")
        import importlib.util
        val_api_path = Path(__file__).resolve().parents[2] / "val_api" / "main.py"
        if not val_api_path.exists():
            pytest.skip("val_api/main.py not found")
        spec = importlib.util.spec_from_file_location("val_api_main", val_api_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return TestClient(mod.app), mod

    def test_health_endpoint_returns_ok_or_degraded(self):
        client, _ = self._get_test_client()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert "val_path" in data

    def test_validate_returns_valid_true_for_plan_valid_output(self):
        client, mod = self._get_test_client()
        fake_result = MagicMock()
        fake_result.stdout = "Plan valid\nSuccessful plans:\n1 plan found.\n"
        fake_result.stderr = ""
        with patch("subprocess.run", return_value=fake_result):
            with patch.object(mod, "_find_validate", return_value="/usr/local/bin/validate"):
                response = client.post("/validate", json={
                    "domain": "(define (domain test))",
                    "problem": "(define (problem p) (:domain test))",
                    "plan": "(pick-up b1)",
                    "verbose": False,
                })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "Plan valid" in data["output"]

    def test_validate_returns_valid_false_for_invalid_plan(self):
        client, mod = self._get_test_client()
        fake_result = MagicMock()
        fake_result.stdout = "Plan invalid\nFailed plans:\n"
        fake_result.stderr = ""
        with patch("subprocess.run", return_value=fake_result):
            with patch.object(mod, "_find_validate", return_value="/usr/local/bin/validate"):
                response = client.post("/validate", json={
                    "domain": "(define (domain test))",
                    "problem": "(define (problem p) (:domain test))",
                    "plan": "(bad-action)",
                    "verbose": False,
                })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_validate_returns_503_when_binary_not_found(self):
        client, mod = self._get_test_client()
        with patch.object(mod, "_find_validate", return_value=None):
            response = client.post("/validate", json={
                "domain": "(define (domain test))",
                "problem": "(define (problem p) (:domain test))",
                "plan": "(pick-up b1)",
            })
        assert response.status_code == 503

    def test_validate_passes_verbose_flag(self):
        client, mod = self._get_test_client()
        fake_result = MagicMock()
        fake_result.stdout = "Plan valid\n"
        fake_result.stderr = ""
        with patch("subprocess.run", return_value=fake_result) as mock_run:
            with patch.object(mod, "_find_validate", return_value="/usr/local/bin/validate"):
                client.post("/validate", json={
                    "domain": "(define (domain test))",
                    "problem": "(define (problem p) (:domain test))",
                    "plan": "(pick-up b1)",
                    "verbose": True,
                })
        cmd = mock_run.call_args[0][0]
        assert "-v" in cmd


class TestPlanBenchRunnerHelpers:
    def test_engine_slug(self):
        assert _engine_slug("gpt-4o") == "litellm_gpt_4o"
        assert _engine_slug("azure/gpt-4o") == "litellm_azure_gpt_4o"

    def test_task_to_name(self):
        assert TASK_TO_NAME["t1"] == "task_1_plan_generation"
        assert TASK_TO_NAME["t8_3"] == "task_8_3_partial_to_full"

    def test_runner_rejects_unknown_task(self):
        agent = MagicMock(spec=BaseAgent, model="test")
        with pytest.raises(ValueError, match="Unknown task"):
            PlanBenchRunner(agent, task="t99", config="blocksworld")


class MockAgent(BaseAgent):
    def __init__(self):
        super().__init__(model="mock-planbench", verbose=False)

    def respond_to_task(self, task: str, context=None):
        from src.agents.base_agent import BenchmarkResponse
        return BenchmarkResponse(response="(pick-up b1)", reasoning="", success=True, confidence=1.0, metadata={})


class TestPlanBenchRunnerRunMocked:
    """Runner run() with _run_in_plan_bench mocked (no LLM, no VAL)."""

    def test_run_returns_results_with_success_score(self):
        from src.benchmarks.skills.plan_bench import runner as runner_mod
        agent = MockAgent()
        run_dir = Path(tempfile.mkdtemp())
        try:
            runner = PlanBenchRunner(agent, task="t1", config="blocksworld", run_dir=run_dir, limit=1)
            canned = {
                "instances": [
                    {"instance_id": 2, "llm_correct": True, "llm_raw_response": "(pick-up b1)"},
                ]
            }
            with patch.object(runner_mod, "_run_in_plan_bench", return_value=canned):
                results = runner.run(limit=1, save_results=True)
            assert len(results) == 1
            assert results[0]["instance_id"] == 2
            assert results[0]["success"] is True
            assert results[0]["score"] == 1.0
            assert (run_dir / agent.__class__.__name__ / "results.jsonl").exists()
            summary = json.loads((run_dir / agent.__class__.__name__ / "summary.json").read_text())
            assert summary["num_tasks"] == 1
            assert summary["num_evaluated"] == 1
            assert summary["success_rate"] == 1.0
            assert results[0]["evaluated"] is True
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    def test_run_returns_all_requested_instances_when_vendored_mutates_list(self):
        """Vendored get_responses mutates specified_instances (removes each id). Runner must pass a copy so filtering still has the requested instance_ids."""
        from src.benchmarks.skills.plan_bench import runner as runner_mod
        agent = MockAgent()
        run_dir = Path(tempfile.mkdtemp())
        canned = {
            "instances": [
                {"instance_id": 2, "llm_correct": False, "llm_raw_response": "(pick-up a)"},
                {"instance_id": 3, "llm_correct": True, "llm_raw_response": "(stack a b)"},
            ]
        }

        def mutate_then_return(*args, **kwargs):
            specified = kwargs.get("specified_instances", args[6] if len(args) > 6 else [])
            if isinstance(specified, list):
                specified.clear()
            return canned

        try:
            runner = PlanBenchRunner(agent, task="t1", config="blocksworld", run_dir=run_dir, limit=2)
            with patch.object(runner_mod, "_run_in_plan_bench", side_effect=mutate_then_return):
                results = runner.run(limit=2, save_results=True)
            assert len(results) == 2, "Runner must pass copy of instance_ids so mutation does not zero results"
            summary = json.loads((run_dir / agent.__class__.__name__ / "summary.json").read_text())
            assert summary["num_tasks"] == 2
            assert summary["num_evaluated"] == 2
            assert summary["success_rate"] == 0.5
            assert summary["total_correct"] == 1
            ids = [r["instance_id"] for r in results]
            assert 2 in ids and 3 in ids
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    def test_run_fallback_returns_filtered_results(self):
        """When response file is missing, fallback returns full prompt data; runner must filter by instance_ids and return only requested instances."""
        from src.benchmarks.skills.plan_bench import runner as runner_mod
        agent = MockAgent()
        run_dir = Path(tempfile.mkdtemp())
        fallback_like = {
            "instances": [
                {"instance_id": 2, "llm_raw_response": "", "llm_correct": False},
                {"instance_id": 3, "llm_raw_response": "", "llm_correct": False},
                {"instance_id": 4, "llm_raw_response": "", "llm_correct": False},
            ]
        }
        try:
            runner = PlanBenchRunner(agent, task="t1", config="blocksworld", run_dir=run_dir, limit=2)
            with patch.object(runner_mod, "_run_in_plan_bench", return_value=fallback_like):
                results = runner.run(limit=2, save_results=True)
            assert len(results) == 2
            summary = json.loads((run_dir / agent.__class__.__name__ / "summary.json").read_text())
            assert summary["num_tasks"] == 2
            assert all(r["instance_id"] in (2, 3) for r in results)
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    def test_run_when_val_not_run_results_not_counted_as_false(self):
        """When VAL was not run, structured has no llm_correct -> not evaluated, success_rate None."""
        from src.benchmarks.skills.plan_bench import runner as runner_mod
        agent = MockAgent()
        run_dir = Path(tempfile.mkdtemp())
        no_eval = {
            "instances": [
                {"instance_id": 2, "llm_raw_response": "(pick-up b1)"},
                {"instance_id": 3, "llm_raw_response": "(stack a b)"},
            ]
        }
        try:
            runner = PlanBenchRunner(agent, task="t1", config="blocksworld", run_dir=run_dir, limit=2)
            with patch.object(runner_mod, "_run_in_plan_bench", return_value=no_eval):
                results = runner.run(limit=2, save_results=True)
            assert len(results) == 2
            for r in results:
                assert r["evaluated"] is False
                assert r["success"] is None
                assert r["score"] is None
            summary = json.loads((run_dir / agent.__class__.__name__ / "summary.json").read_text())
            assert summary["num_tasks"] == 2
            assert summary["num_evaluated"] == 0
            assert summary["total_correct"] == 0
            assert summary["success_rate"] is None
            lines = (run_dir / agent.__class__.__name__ / "results.jsonl").read_text().strip().split("\n")
            for line in lines:
                row = json.loads(line)
                assert row["evaluated"] is False
                assert row["success"] is None
                assert row["score"] is None
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    def test_run_summary_success_rate_only_over_evaluated(self):
        """Success rate is computed only over evaluated instances; num_evaluated and total_correct match."""
        from src.benchmarks.skills.plan_bench import runner as runner_mod
        agent = MockAgent()
        run_dir = Path(tempfile.mkdtemp())
        mixed = {
            "instances": [
                {"instance_id": 2, "llm_correct": True, "llm_raw_response": "a"},
                {"instance_id": 3, "llm_raw_response": "b"},
                {"instance_id": 4, "llm_correct": False, "llm_raw_response": "c"},
            ]
        }
        try:
            runner = PlanBenchRunner(agent, task="t1", config="blocksworld", run_dir=run_dir, limit=3)
            with patch.object(runner_mod, "_run_in_plan_bench", return_value=mixed):
                results = runner.run(limit=3, save_results=True)
            assert len(results) == 3
            summary = json.loads((run_dir / agent.__class__.__name__ / "summary.json").read_text())
            assert summary["num_tasks"] == 3
            assert summary["num_evaluated"] == 2
            assert summary["total_correct"] == 1
            assert summary["success_rate"] == 0.5
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    def test_run_calls_run_in_plan_bench_once_per_instance(self):
        """Runner calls _run_in_plan_bench once per instance (incremental run)."""
        from src.benchmarks.skills.plan_bench import runner as runner_mod
        agent = MockAgent()
        run_dir = Path(tempfile.mkdtemp())
        # Each call returns accumulated structured output (as vendored file would).
        side_effect = [
            {"instances": [{"instance_id": 2, "llm_correct": True, "llm_raw_response": "x"}]},
            {"instances": [
                {"instance_id": 2, "llm_correct": True, "llm_raw_response": "x"},
                {"instance_id": 3, "llm_correct": True, "llm_raw_response": "y"},
            ]},
        ]
        try:
            runner = PlanBenchRunner(agent, task="t1", config="blocksworld", run_dir=run_dir, limit=2)
            with patch.object(runner_mod, "_run_in_plan_bench", side_effect=side_effect) as m:
                results = runner.run(limit=2, save_results=True)
            assert len(results) == 2
            assert m.call_count == 2
            for call in m.call_args_list:
                args, _ = call
                specified = args[6]
                assert len(specified) == 2
                assert _LIMIT_SENTINEL_INSTANCE_ID in specified
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    def test_run_calls_val_api_when_val_api_url_set(self):
        """When VAL_API_URL is set, _run_in_plan_bench uses ResponseEvaluator which calls the API via vendored validate_plan."""
        from src.benchmarks.skills.plan_bench import runner as runner_mod
        agent = MockAgent()
        run_dir = Path(tempfile.mkdtemp())
        canned = {
            "instances": [
                {"instance_id": 2, "llm_correct": True, "llm_raw_response": "(pick-up b1)"},
            ]
        }
        try:
            runner = PlanBenchRunner(agent, task="t1", config="blocksworld", run_dir=run_dir, limit=1)
            with patch.dict(os.environ, {"VAL_API_URL": "http://localhost:8001"}):
                with patch.object(runner_mod, "_run_in_plan_bench", return_value=canned):
                    results = runner.run(limit=1, save_results=False)
            assert len(results) == 1
            assert results[0]["evaluated"] is True
            assert results[0]["success"] is True
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    def test_run_writes_after_each_instance(self):
        """_write_results is called after each instance (save as it comes in)."""
        from src.benchmarks.skills.plan_bench import runner as runner_mod
        agent = MockAgent()
        run_dir = Path(tempfile.mkdtemp())
        try:
            runner = PlanBenchRunner(agent, task="t1", config="blocksworld", run_dir=run_dir, limit=2)
            with patch.object(runner, "_write_results") as write_mock:
                with patch.object(runner_mod, "_run_in_plan_bench", side_effect=[
                    {"instances": [{"instance_id": 2, "llm_correct": True, "llm_raw_response": "a"}]},
                    {"instances": [
                        {"instance_id": 2, "llm_correct": True, "llm_raw_response": "a"},
                        {"instance_id": 3, "llm_correct": False, "llm_raw_response": "b"},
                    ]},
                ]):
                    runner.run(limit=2, save_results=True)
            assert write_mock.call_count == 2
            first_call_results = write_mock.call_args_list[0][0][0]
            assert len(first_call_results) == 1
            assert first_call_results[0]["instance_id"] == 2
            second_call_results = write_mock.call_args_list[1][0][0]
            assert len(second_call_results) == 2
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
