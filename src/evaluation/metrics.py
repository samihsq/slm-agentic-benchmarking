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
    success_rate: float  # Task completion rate (0-1)
    avg_confidence: float  # Average model confidence (0-1)
    avg_latency: float  # Average time per task (seconds)
    total_cost: float  # Total cost in USD
    num_tasks: int  # Number of tasks completed
    
    # Optional detailed metrics
    per_task_metrics: Optional[List[Dict[str, Any]]] = None


def calculate_metrics(
    results: List[Dict[str, Any]],
    ground_truth: Optional[List[Any]] = None
) -> BenchmarkMetrics:
    """
    Calculate metrics from benchmark results.
    
    Args:
        results: List of result dicts with keys: success, confidence, latency, cost
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
    success_count = sum(1 for r in results if r.get("success", False))
    total_confidence = sum(r.get("confidence", 0.0) for r in results)
    total_latency = sum(r.get("latency", 0.0) for r in results)
    total_cost = sum(r.get("cost", 0.0) for r in results)
    
    # Calculate accuracy if ground truth provided
    accuracy = 0.0
    if ground_truth and len(ground_truth) == num_tasks:
        correct = sum(
            1 for r, gt in zip(results, ground_truth)
            if r.get("prediction") == gt
        )
        accuracy = correct / num_tasks
    
    return BenchmarkMetrics(
        accuracy=accuracy,
        success_rate=success_count / num_tasks,
        avg_confidence=total_confidence / num_tasks,
        avg_latency=total_latency / num_tasks if total_latency > 0 else 0.0,
        total_cost=total_cost,
        num_tasks=num_tasks,
        per_task_metrics=results if len(results) < 100 else None,  # Save only for small runs
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
        "success_rate_change": (comparison.success_rate - baseline.success_rate) * 100,
        "latency_change": (
            ((comparison.avg_latency - baseline.avg_latency) / baseline.avg_latency * 100)
            if baseline.avg_latency > 0 else 0.0
        ),
        "cost_ratio": (
            comparison.total_cost / baseline.total_cost
            if baseline.total_cost > 0 else 0.0
        ),
    }
