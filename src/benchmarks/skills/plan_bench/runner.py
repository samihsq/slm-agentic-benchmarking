"""
PlanBench runner: runs the vendored PlanBench pipeline with LiteLLM (Azure/Ollama).
Uses USE_LITELLM=1 and LITELLM_MODEL so the vendored response_generation
calls our adapter in vendor/plan_bench/utils/llm_utils.py.
Evaluation requires VAL: set VAL env to the directory containing the validate binary (KCL-Planning/VAL).
If VAL is missing, instances are not evaluated (evaluated=False, success=None).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ....agents.base_agent import BaseAgent
from ....evaluation.cost_tracker import CostTracker
from ....config import get_llm_config

TASK_TO_NAME = {
    "t1": "task_1_plan_generation",
    "t2": "task_2_plan_optimality",
    "t3": "task_3_plan_verification",
    "t4": "task_4_plan_reuse",
    "t5": "task_5_plan_generalization",
    "t6": "task_6_replanning",
    "t7": "task_7_plan_execution",
    "t8_1": "task_8_1_goal_shuffling",
    "t8_2": "task_8_2_full_to_partial",
    "t8_3": "task_8_3_partial_to_full",
}

_LIMIT_SENTINEL_INSTANCE_ID = -1


def _engine_slug(model: str) -> str:
    return "litellm_" + re.sub(r"[^a-z0-9_]", "_", model.lower())


def _plan_bench_root() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "vendor" / "plan_bench"


def _results_from_structured(
    structured: Dict[str, Any], instance_ids: List[int]
) -> List[Dict[str, Any]]:
    """Build result dicts from PlanBench structured output for requested instance IDs.
    When VAL was not run, llm_correct is missing -> evaluated=False, success=None, score=None.
    """
    results = []
    for inst in structured.get("instances", []):
        if inst.get("instance_id") not in instance_ids:
            continue
        correct = inst.get("llm_correct")
        if correct is None:
            correct = inst.get("llm_correct_binary")
        evaluated = correct is not None
        if evaluated:
            success = bool(correct)
            score = 1.0 if correct else 0.0
        else:
            success = None
            score = None
        results.append({
            "instance_id": inst["instance_id"],
            "llm_correct": correct,
            "llm_raw_response": inst.get("llm_raw_response", ""),
            "evaluated": evaluated,
            "success": success,
            "score": score,
            "latency": 0.0,
            "cost": 0.0,
        })
    return results


def _run_in_plan_bench(
    plan_bench_root: Path,
    config: str,
    task_id: str,
    task_name: str,
    engine_slug: str,
    litellm_model: str,
    specified_instances: List[int],
    ignore_existing: bool,
    verbose: bool,
    litellm_api_key: Optional[str] = None,
    litellm_api_base: Optional[str] = None,
) -> Dict[str, Any]:
    import yaml

    config_file = plan_bench_root / "configs" / f"{config}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"PlanBench config not found: {config_file}")
    with open(config_file) as f:
        data = yaml.safe_load(f)
    domain_name = data["domain_name"]
    prompt_path = plan_bench_root / "prompts" / domain_name / f"{task_name}.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"PlanBench prompts not found: {prompt_path}")

    cwd = os.getcwd()
    plan_bench_str = str(plan_bench_root)
    try:
        os.chdir(plan_bench_str)
        if plan_bench_str not in sys.path:
            sys.path.insert(0, plan_bench_str)
        os.environ["USE_LITELLM"] = "1"
        os.environ["LITELLM_MODEL"] = litellm_model
        if litellm_api_key:
            os.environ["LITELLM_API_KEY"] = litellm_api_key
        if litellm_api_base:
            os.environ["LITELLM_API_BASE"] = litellm_api_base

        from response_generation import ResponseGenerator

        response_generator = ResponseGenerator(
            str(config_file), engine_slug, verbose=verbose, ignore_existing=ignore_existing
        )
        response_generator.get_responses(
            task_name, specified_instances=specified_instances, run_till_completion=False
        )

        val_path = os.environ.get("VAL")
        if val_path and os.path.exists(os.path.join(val_path, "validate")):
            try:
                from response_evaluation import ResponseEvaluator

                evaluator = ResponseEvaluator(
                    str(config_file), engine_slug, specified_instances=[], verbose=verbose,
                    # Always re-read from responses/ so incremental runs pick up new responses
                    ignore_existing=True,
                )
                if task_id in ("t1", "t2", "t4", "t5", "t6", "t8_1", "t8_2", "t8_3"):
                    evaluator.evaluate_plan(task_name)
                elif task_id == "t7":
                    evaluator.evaluate_state(task_name)
                elif task_id == "t3":
                    evaluator.evaluate_verification(task_name)
            except Exception as eval_exc:
                print(f"Warning: evaluation failed for {task_name}: {eval_exc}")

        results_dir = plan_bench_root / "results" / domain_name / engine_slug
        response_dir = plan_bench_root / "responses" / domain_name / engine_slug
        if (results_dir / f"{task_name}.json").exists():
            with open(results_dir / f"{task_name}.json") as f:
                return json.load(f)
        response_file = response_dir / f"{task_name}.json"
        if response_file.exists():
            with open(response_file) as f:
                return json.load(f)
        with open(prompt_path) as f:
            fallback = json.load(f)
        for inst in fallback.get("instances", []):
            inst.setdefault("llm_raw_response", "")
            # Do not set llm_correct so instances are treated as not evaluated (VAL not run)
        return fallback
    finally:
        os.chdir(cwd)
        if plan_bench_str in sys.path:
            sys.path.remove(plan_bench_str)


class PlanBenchRunner:
    def __init__(
        self,
        agent: BaseAgent,
        task: str = "t1",
        config: str = "blocksworld",
        limit: Optional[int] = None,
        ignore_existing: bool = False,
        verbose: bool = False,
        run_dir: Optional[Path] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        if task not in TASK_TO_NAME:
            raise ValueError(f"Unknown task {task}. Valid: {list(TASK_TO_NAME.keys())}")
        self.agent = agent
        self.task = task
        self.task_name = TASK_TO_NAME[task]
        self.config = config
        self.limit = limit
        self.ignore_existing = ignore_existing
        self.verbose = verbose
        self.run_dir = run_dir or (Path("results") / "plan_bench" / time.strftime("%Y%m%d_%H%M%S"))
        self.cost_tracker = cost_tracker
        self._plan_bench_root = _plan_bench_root()

    def run(self, limit: Optional[int] = None, save_results: bool = True) -> List[Dict[str, Any]]:
        limit = limit or self.limit
        litellm_api_key = None
        litellm_api_base = None
        try:
            llm_config = get_llm_config(self.agent.model)
            litellm_model = llm_config.get("model") or self.agent.model
            litellm_api_key = llm_config.get("azure_api_key")
            litellm_api_base = llm_config.get("azure_endpoint")
            if llm_config.get("provider") == "ollama" and not litellm_model.startswith("ollama/"):
                litellm_model = "ollama/" + litellm_model
        except Exception:
            litellm_model = self.agent.model
            if litellm_model and not litellm_model.startswith("ollama/") and ":" in litellm_model:
                litellm_model = "ollama/" + litellm_model
        engine_slug = _engine_slug(self.agent.model)

        import yaml
        config_file = self._plan_bench_root / "configs" / f"{self.config}.yaml"
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        domain_name = config_data["domain_name"]
        with open(self._plan_bench_root / "prompts" / domain_name / f"{self.task_name}.json") as f:
            prompt_data = json.load(f)
        instances = prompt_data.get("instances", [])
        if not instances:
            return []
        instance_ids = [inst["instance_id"] for inst in instances]
        if limit:
            instance_ids = instance_ids[:limit]

        # Run one instance at a time so we can write results after each (save as it comes in).
        # Vendored get_responses mutates specified_instances (removes each id). If the list
        # becomes empty, it stops filtering and may run all instances. Keep a sentinel id so
        # filtering remains active after removal of the target instance.
        results = []
        for iid in instance_ids:
            structured = _run_in_plan_bench(
                self._plan_bench_root,
                self.config,
                self.task,
                self.task_name,
                engine_slug,
                litellm_model,
                [iid, _LIMIT_SENTINEL_INSTANCE_ID],
                self.ignore_existing,
                self.verbose,
                litellm_api_key=litellm_api_key,
                litellm_api_base=litellm_api_base,
            )
            results = _results_from_structured(structured, instance_ids)
            if save_results and self.run_dir:
                self._write_results(results)
        return results

    def _write_results(self, results: List[Dict[str, Any]]) -> None:
        agent_name = self.agent.__class__.__name__
        out_dir = Path(self.run_dir) / agent_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.jsonl", "w") as f:
            for r in results:
                ev = r.get("evaluated", False)
                row = {
                    "task_id": "instance_{}".format(r["instance_id"]),
                    "instance_id": r["instance_id"],
                    "evaluated": ev,
                }
                if ev:
                    row["success"] = r.get("success", False)
                    row["score"] = r.get("score", 0.0)
                else:
                    row["success"] = None
                    row["score"] = None
                f.write(json.dumps(row) + "\n")
        n = len(results)
        num_evaluated = sum(1 for r in results if r.get("evaluated"))
        correct = sum(1 for r in results if r.get("llm_correct"))
        summary = {
            "agent": agent_name,
            "model": self.agent.model,
            "benchmark": "PlanBench",
            "task": self.task,
            "config": self.config,
            "num_tasks": n,
            "num_evaluated": num_evaluated,
            "total_correct": correct,
            "success_rate": round(correct / num_evaluated, 4) if num_evaluated else None,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
