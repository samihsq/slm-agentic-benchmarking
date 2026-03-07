"""
Evaluation metrics for benchmarking.

Calculates performance metrics across different benchmarks.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class BenchmarkMetrics:
    """Metrics for a benchmark run."""
    accuracy: float  # Percentage correct (0-1)
    success_rate: Optional[float]  # Task completion rate (0-1), None when no tasks evaluated (e.g. PlanBench without VAL)
    avg_confidence: float  # Average model confidence (0-1)
    avg_latency: float  # Average time per task (seconds)
    total_cost: float  # Total cost in USD
    num_tasks: int  # Number of tasks completed
    num_evaluated: Optional[int] = None  # When set, only these tasks had success/score (e.g. PlanBench with VAL)

    # Optional detailed metrics
    per_task_metrics: Optional[List[Dict[str, Any]]] = None


def _get_attr(obj, key: str, default: Any = None) -> Any:
    """Get attribute from object or dict."""
    if hasattr(obj, key):
        return getattr(obj, key, default)
    elif isinstance(obj, dict):
        return obj.get(key, default)
    return default


def calculate_metrics(
    results: List[Any],
    ground_truth: Optional[List[Any]] = None
) -> BenchmarkMetrics:
    """
    Calculate metrics from benchmark results.
    
    Args:
        results: List of result objects or dicts with: success, score, latency, cost
        ground_truth: Optional ground truth labels for accuracy calculation
    
    Returns:
        BenchmarkMetrics object
    """
    if not results:
        return BenchmarkMetrics(
            accuracy=0.0,
            success_rate=0.0,
            avg_confidence=0.0,
            avg_latency=0.0,
            total_cost=0.0,
            num_tasks=0,
        )

    num_tasks = len(results)
    has_evaluated_key = any(_get_attr(r, "evaluated") is not None for r in results)
    if has_evaluated_key:
        evaluated_results = [r for r in results if bool(_get_attr(r, "evaluated", False))]
        num_evaluated = len(evaluated_results)
        success_count = sum(1 for r in evaluated_results if _get_attr(r, "success", False))
        total_confidence = sum((_get_attr(r, "score", 0.0) or 0.0) for r in evaluated_results)
        success_rate = (success_count / num_evaluated) if num_evaluated else None
        avg_confidence = total_confidence / num_evaluated if num_evaluated else 0.0
    else:
        num_evaluated = None
        success_count = sum(1 for r in results if _get_attr(r, "success", False))
        total_confidence = sum(_get_attr(r, "score", 0.0) or 0.0 for r in results)
        success_rate = success_count / num_tasks
        avg_confidence = total_confidence / num_tasks

    total_latency = sum(_get_attr(r, "latency", 0.0) or 0.0 for r in results)
    total_cost = sum(_get_attr(r, "cost", 0.0) or 0.0 for r in results)

    # Calculate accuracy if ground truth provided
    accuracy = 0.0
    if ground_truth and len(ground_truth) == num_tasks:
        correct = sum(
            1 for r, gt in zip(results, ground_truth)
            if _get_attr(r, "prediction") == gt
        )
        accuracy = correct / num_tasks

    return BenchmarkMetrics(
        accuracy=accuracy,
        success_rate=success_rate,
        avg_confidence=avg_confidence,
        avg_latency=total_latency / num_tasks if total_latency > 0 else 0.0,
        total_cost=total_cost,
        num_tasks=num_tasks,
        num_evaluated=num_evaluated,
        per_task_metrics=results if len(results) < 100 else None,
    )


def compare_metrics(
    baseline: BenchmarkMetrics,
    comparison: BenchmarkMetrics
) -> Dict[str, float]:
    """
    Compare two sets of metrics.
    
    Returns dict with percentage changes.
    """
    return {
        "accuracy_change": (comparison.accuracy - baseline.accuracy) * 100,
        "success_rate_change": (
            ((comparison.success_rate or 0.0) - (baseline.success_rate or 0.0)) * 100
        ),
        "latency_change": (
            ((comparison.avg_latency - baseline.avg_latency) / baseline.avg_latency * 100)
            if baseline.avg_latency > 0 else 0.0
        ),
        "cost_ratio": (
            comparison.total_cost / baseline.total_cost
            if baseline.total_cost > 0 else 0.0
        ),
    }
