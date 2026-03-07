from src.evaluation.metrics import BenchmarkMetrics, calculate_metrics, compare_metrics


def test_calculate_metrics_without_evaluated_flag_uses_all_tasks():
    results = [
        {"success": True, "score": 1.0, "latency": 1.0, "cost": 0.1},
        {"success": False, "score": 0.0, "latency": 2.0, "cost": 0.2},
    ]
    m = calculate_metrics(results)
    assert m.num_tasks == 2
    assert m.num_evaluated is None
    assert m.success_rate == 0.5
    assert m.avg_confidence == 0.5


def test_calculate_metrics_with_evaluated_flag_uses_only_evaluated_tasks():
    results = [
        {"evaluated": True, "success": True, "score": 1.0, "latency": 1.0, "cost": 0.1},
        {"evaluated": False, "success": None, "score": None, "latency": 2.0, "cost": 0.2},
        {"evaluated": True, "success": False, "score": 0.0, "latency": 3.0, "cost": 0.3},
    ]
    m = calculate_metrics(results)
    assert m.num_tasks == 3
    assert m.num_evaluated == 2
    assert m.success_rate == 0.5
    assert m.avg_confidence == 0.5


def test_calculate_metrics_with_zero_evaluated_returns_none_success_rate():
    results = [
        {"evaluated": False, "success": None, "score": None, "latency": 1.0, "cost": 0.1},
        {"evaluated": False, "success": None, "score": None, "latency": 2.0, "cost": 0.2},
    ]
    m = calculate_metrics(results)
    assert m.num_tasks == 2
    assert m.num_evaluated == 0
    assert m.success_rate is None
    assert m.avg_confidence == 0.0


def test_compare_metrics_handles_none_success_rate():
    baseline = BenchmarkMetrics(
        accuracy=0.0,
        success_rate=None,
        avg_confidence=0.0,
        avg_latency=1.0,
        total_cost=1.0,
        num_tasks=10,
    )
    comparison = BenchmarkMetrics(
        accuracy=0.0,
        success_rate=0.5,
        avg_confidence=0.0,
        avg_latency=1.0,
        total_cost=2.0,
        num_tasks=10,
    )
    c = compare_metrics(baseline, comparison)
    assert c["success_rate_change"] == 50.0
