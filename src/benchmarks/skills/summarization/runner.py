"""
Summarization (XSum) Benchmark Runner.

Scores summaries with a configurable automatic metric:
  - rougeL (reference-based)
  - bertscore (reference-based)
  - bartscore (source-based; conditional likelihood)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ....agents.base_agent import BaseAgent, EvaluationResult
from ....evaluation.cost_tracker import CostTracker
from ....utils.adaptive_limiter import ThreadSafeAdaptiveLimiter
from ....utils.trace import TraceCapture, QuestionTrace

from .metrics import MetricUnavailableError, SummarizationScorer


class SummarizationRunner:
    """
    Runner for summarization evaluation on XSum.

    Dataset fields:
      - document: source article
      - summary: reference summary
      - id: row id
    """

    def __init__(
        self,
        agent: BaseAgent,
        *,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
        split: str = "validation",
        metric: str = "rougeL",
        success_threshold: Optional[float] = None,
        bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
        bartscore_model: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
    ):
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self.split = split

        self.metric = metric
        self.scorer = SummarizationScorer(
            metric=metric,
            bertscore_model_type=bertscore_model_type,
            bartscore_model=bartscore_model,
            device=device,
        )

        # Metric-specific default thresholds (tune later).
        if success_threshold is not None:
            self.success_threshold = float(success_threshold)
        else:
            m = metric.lower()
            if m in ("rouge", "rougel", "rouge-l", "rouge_l"):
                self.success_threshold = 0.15
            elif m in ("bertscore", "bert-score", "bert_score"):
                self.success_threshold = 0.85
            elif m in ("bartscore", "bart-score", "bart_score"):
                # exp(-NLL) tends to be small; treat this as a weak filter.
                self.success_threshold = 0.01
            else:
                self.success_threshold = 0.0

        self._lock = Lock()
        self._output_dir: Optional[Path] = None

        self._total = 0
        self._success = 0
        self._scores: List[float] = []
        self._error_counts = defaultdict(int)

    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Summarization runner requires HuggingFace 'datasets'. Install with: poetry add datasets"
            ) from e

        # XSum: prefer validation for quicker iteration.
        ds = load_dataset("xsum", split=self.split)
        if limit:
            ds = ds.select(range(min(int(limit), len(ds))))

        tasks: List[Dict[str, Any]] = []
        for row in ds:
            rid = row.get("id")
            task_id = f"xsum_{rid}" if rid is not None else f"xsum_{len(tasks)}"
            tasks.append(
                {
                    "task_id": task_id,
                    "document": row.get("document", ""),
                    "reference": row.get("summary", ""),
                    "dataset": "xsum",
                    "split": self.split,
                }
            )
        return tasks

    def format_task(self, task: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        doc = task.get("document", "") or ""

        # Truncate document to stay within small-context models (e.g. phi-4 = 16384 tokens).
        # Reserve 512 tokens for template overhead + 512 for completion = ~1024 tokens.
        # At ~4 chars/token, a 16384-token window gives ~60k chars; cap at 56000 to be safe.
        MAX_DOC_CHARS = 56_000
        truncated = len(doc) > MAX_DOC_CHARS
        if truncated:
            doc = doc[:MAX_DOC_CHARS]

        task_text = f"""You are given a news article. Write a single-sentence summary that captures the main point.

ARTICLE:
{doc}

Return JSON:
{{"response": "<one-sentence summary>", "confidence": 0.0-1.0}}"""

        context = {
            "benchmark_type": "general",
            "task_type": "summarization",
            "dataset": task.get("dataset", "xsum"),
            "split": task.get("split", self.split),
            "doc_tokens": len(doc) // 4,
            # One sentence is never more than ~100 tokens; 512 is very generous.
            # This prevents phi-4 (16384-token context) from crashing when the
            # agent hardcodes max_tokens=65536, which alone exceeds phi-4's window.
            "max_completion_tokens": 512,
        }
        return task_text, context

    def _process_task(self, task: Dict[str, Any]) -> EvaluationResult:
        start_time = time.time()
        task_id = task["task_id"]

        task_text, context = self.format_task(task)
        source = str(task.get("document", "") or "")
        reference = str(task.get("reference", "") or "")

        with TraceCapture(
            task_id=task_id,
            agent_type=self.agent.__class__.__name__,
            model=self.agent.model,
            input_question=task_text[:1000] + "..." if len(task_text) > 1000 else task_text,
        ) as trace_ctx:
            response = self.agent.respond_to_task(task_text, context)
            latency = time.time() - start_time

            prediction = (response.response or "").strip()

            try:
                metric_result = self.scorer.score_pair(
                    source=source, prediction=prediction, reference=reference
                )
                score = float(metric_result.score)
                metric_details = dict(metric_result.details)
            except MetricUnavailableError as e:
                score = 0.0
                metric_details = {"error": str(e), "metric": self.metric}
            except Exception as e:
                score = 0.0
                metric_details = {"error": f"scoring_failed: {e}", "metric": self.metric}

            success = score >= self.success_threshold

            # Cost (if agent provided token metadata)
            cost = 0.0
            meta = response.metadata if isinstance(response.metadata, dict) else {}
            if self.cost_tracker and meta:
                prompt_tokens = meta.get("prompt_tokens", 0)
                completion_tokens = meta.get("completion_tokens", 0)
                if prompt_tokens > 0:
                    with self._lock:
                        cost = self.cost_tracker.log_usage(
                            model=self.agent.model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            task_id=task_id,
                            benchmark="summarization",
                            agent_type=self.agent.__class__.__name__,
                        )

            # Trace
            trace_ctx.trace.final_output = prediction[:5000]
            trace_ctx.trace.predicted = prediction[:500]
            trace_ctx.trace.correct = reference[:500]
            trace_ctx.trace.match = success
            trace_ctx.trace.confidence = score
            trace_ctx.trace.total_latency = latency
            trace_ctx.trace.total_cost = cost
            trace_ctx.trace.reasoning = response.reasoning or ""

        return EvaluationResult(
            task_id=task_id,
            prompt=task_text[:1000] + "..." if len(task_text) > 1000 else task_text,
            agent_response=prediction,
            success=success,
            score=score,
            latency=latency,
            cost=cost,
            metadata={
                "metric": self.metric,
                "threshold": self.success_threshold,
                "reference": reference,
                "source_preview": source[:500],
                "prediction": prediction,
                "metric_details": metric_details,
                "reasoning": response.reasoning,
                "trace": trace_ctx.trace,
            },
        )

    def _save_result_incremental(self, result: EvaluationResult):
        if not self._output_dir:
            return

        with self._lock:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}

            task_dir = self._output_dir / result.task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            trace = metadata.get("trace")
            trace_path = task_dir / "trace.json"
            if trace and isinstance(trace, QuestionTrace):
                trace_path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
            else:
                trace_path.write_text(
                    json.dumps(
                        {
                            "task_id": result.task_id,
                            "agent_type": self.agent.__class__.__name__,
                            "model": self.agent.model,
                            "predicted": (result.agent_response or "")[:500],
                            "correct": (metadata.get("reference") or "")[:500],
                            "match": bool(result.success),
                            "score": float(result.score or 0.0),
                            "total_latency": round(float(result.latency or 0.0), 2),
                            "total_cost": round(float(result.cost or 0.0), 6),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            metric_details = metadata.get("metric_details") or {}
            if not isinstance(metric_details, dict):
                metric_details = {}

            summary_line = {
                "task_id": result.task_id,
                "metric": metadata.get("metric"),
                "score": round(float(result.score or 0.0), 6),
                "success": bool(result.success),
                "latency": round(float(result.latency or 0.0), 2),
                "cost": round(float(result.cost or 0.0), 6),
                # Convenience keys for review scripts
                **{k: v for k, v in metric_details.items() if isinstance(v, (int, float, str, bool))},
            }
            with (self._output_dir / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary_line) + "\n")

    def run(self, *, limit: Optional[int] = None, save_results: bool = True) -> List[EvaluationResult]:
        tasks = self.load_tasks(limit)

        self._total = 0
        self._success = 0
        self._scores = []
        self._error_counts.clear()

        print(f"\nRunning Summarization benchmark ({self.split}) with {len(tasks)} tasks...")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Model: {self.agent.model}")
        print(f"Metric: {self.metric} (threshold={self.success_threshold})")
        print(f"Concurrency: {self.concurrency}")

        if save_results:
            if self.run_dir:
                base_dir = self.run_dir
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_dir = Path("results") / "summarization" / f"{self.agent.model}_{timestamp}"
            self._output_dir = base_dir / self.agent.__class__.__name__
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving to: {self._output_dir}/\n")

        if self.concurrency > 1:
            results = self._run_concurrent(tasks)
        else:
            results = self._run_sequential(tasks)

        avg_score = sum(self._scores) / len(self._scores) if self._scores else 0.0
        success_rate = self._success / max(self._total, 1)
        print(f"\nAvg score: {avg_score:.3f}")
        print(f"Success rate: {success_rate*100:.1f}% ({self._success}/{self._total})")

        if save_results and self._output_dir:
            summary = {
                "agent": self.agent.__class__.__name__,
                "model": self.agent.model,
                "benchmark": "Summarization (XSum)",
                "split": self.split,
                "metric": self.metric,
                "threshold": self.success_threshold,
                "num_tasks": len(results),
                "avg_score": avg_score,
                "success_rate": success_rate,
                "results_dir": str(self._output_dir),
            }
            (self._output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(f"\nSummary saved to: {self._output_dir / 'summary.json'}")

        return results

    def _run_sequential(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        pbar = tqdm(
            tasks,
            desc="Running",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Avg: {postfix}",
            postfix="---",
        )

        for task in pbar:
            result = self._process_task(task)
            results.append(result)

            with self._lock:
                self._total += 1
                self._success += 1 if result.success else 0
                self._scores.append(float(result.score or 0.0))
                avg = sum(self._scores) / len(self._scores) if self._scores else 0.0

            self._save_result_incremental(result)
            pbar.set_postfix_str(f"{avg:.3f}")

            if self.verbose:
                status = "✓" if result.success else "✗"
                tqdm.write(f"  {result.task_id}: {status} score={result.score:.3f} ({result.latency:.1f}s)")

        return results

    def _run_concurrent(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        results: List[Optional[EvaluationResult]] = [None] * len(tasks)

        limiter = ThreadSafeAdaptiveLimiter(
            max_concurrency=self.concurrency,
            min_concurrency=1,
            backoff_factor=0.5,
            recovery_threshold=10,
        )

        def process_with_limiter(task: Dict[str, Any]) -> EvaluationResult:
            wait_time = limiter.wait_if_needed()
            if wait_time > 0 and self.verbose:
                tqdm.write(f"  ⏳ Rate limit backoff: {wait_time:.1f}s")
            try:
                r = self._process_task(task)
                limiter.record_success()
                return r
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str:
                    limiter.record_error(backoff_seconds=5.0)
                raise

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {
                executor.submit(process_with_limiter, task): i for i, task in enumerate(tasks)
            }

            pbar = tqdm(
                total=len(tasks),
                desc=f"Running ({self.concurrency}x)",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Avg: {postfix}",
                postfix="---",
            )

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = EvaluationResult(
                        task_id=tasks[idx].get("task_id", f"t_{idx}"),
                        prompt="",
                        agent_response=f"Error: {e}",
                        success=False,
                        score=0.0,
                        latency=0.0,
                        cost=0.0,
                        metadata={"error": str(e), "metric": self.metric},
                    )

                results[idx] = result

                with self._lock:
                    self._total += 1
                    self._success += 1 if result.success else 0
                    self._scores.append(float(result.score or 0.0))
                    avg = sum(self._scores) / len(self._scores) if self._scores else 0.0

                self._save_result_incremental(result)
                pbar.set_description(f"Running ({limiter.current_concurrency}x)")
                pbar.set_postfix_str(f"{avg:.3f}")
                pbar.update(1)

            pbar.close()

        return [r for r in results if r is not None]

