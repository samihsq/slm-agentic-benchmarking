"""
Cost tracking and budget management for benchmarking experiments.

Tracks token usage and costs across multiple models and experiments,
provides budget alerts, and estimates costs before running experiments.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class UsageRecord:
    """Record of a single API call usage."""
    timestamp: str
    model: str
    task_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    benchmark: str = ""
    agent_type: str = ""


@dataclass
class CostSummary:
    """Summary of costs across experiments."""
    total_cost: float
    total_tokens: int
    by_model: Dict[str, float] = field(default_factory=dict)
    by_benchmark: Dict[str, float] = field(default_factory=dict)
    by_agent: Dict[str, float] = field(default_factory=dict)
    num_calls: int = 0


class CostTracker:
    """
    Track costs and manage budget for benchmarking experiments.
    
    Features:
    - Real-time cost tracking across models
    - Budget alerts at configurable thresholds
    - Pre-run cost estimation
    - Detailed usage logs
    - Experiment cost summaries
    """
    
    # Pricing per 1M tokens (from azure_llm_config.py)
    PRICING_PER_1M = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "phi-4": {"input": 0.07, "output": 0.14},
        "llama-3.3-70b": {"input": 0.27, "output": 0.27},
        "mistral-small-3.1": {"input": 0.10, "output": 0.30},
        "glm-4.7-flash": {"input": 0.08, "output": 0.08},
        "qwen3-30b-a3b": {"input": 0.08, "output": 0.08},
        "lfm2.5-1.2b": {"input": 0.01, "output": 0.02},
    }
    
    def __init__(
        self,
        budget_limit: float = 10000.0,
        alert_thresholds: Optional[List[float]] = None,
        log_file: Optional[str] = None,
    ):
        """
        Initialize cost tracker.
        
        Args:
            budget_limit: Maximum budget in USD
            alert_thresholds: Budget percentages to alert at (e.g., [0.3, 0.6, 0.9])
            log_file: Path to save usage logs
        """
        self.budget_limit = budget_limit
        self.alert_thresholds = alert_thresholds or [0.3, 0.6, 0.9]
        self.log_file = log_file or "cost_tracking.json"
        
        self.total_spent = 0.0
        self.total_tokens = 0
        self.usage_records: List[UsageRecord] = []
        self.alerted_thresholds = set()
        
        # Load existing records if available
        self._load_existing_records()
    
    def _load_existing_records(self):
        """Load existing usage records from log file."""
        log_path = Path(self.log_file)
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    data = json.load(f)
                    self.total_spent = data.get("total_spent", 0.0)
                    self.total_tokens = data.get("total_tokens", 0)
                    records = data.get("records", [])
                    self.usage_records = [
                        UsageRecord(**r) for r in records
                    ]
            except Exception as e:
                print(f"Warning: Could not load existing cost records: {e}")
    
    def _save_records(self):
        """Save usage records to log file."""
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "total_spent": self.total_spent,
                "total_tokens": self.total_tokens,
                "records": [asdict(r) for r in self.usage_records],
                "last_updated": datetime.now().isoformat(),
            }
            
            with open(log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cost records: {e}")
    
    def log_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        task_id: str = "",
        benchmark: str = "",
        agent_type: str = "",
    ) -> float:
        """
        Log API usage and calculate cost.
        
        Args:
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            task_id: Task identifier
            benchmark: Benchmark name
            agent_type: Agent architecture type
        
        Returns:
            Cost in USD for this call
        """
        # Calculate cost
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Create usage record
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            task_id=task_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            benchmark=benchmark,
            agent_type=agent_type,
        )
        
        # Update totals
        self.usage_records.append(record)
        self.total_spent += cost
        self.total_tokens += record.total_tokens
        
        # Check for budget alerts
        self._check_budget_alerts()
        
        # Save to file
        self._save_records()
        
        return cost
    
    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate cost for token usage."""
        if model not in self.PRICING_PER_1M:
            print(f"Warning: Unknown model pricing for {model}, using average")
            return (prompt_tokens + completion_tokens) / 1_000_000 * 0.50
        
        pricing = self.PRICING_PER_1M[model]
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _check_budget_alerts(self):
        """Check if budget thresholds have been crossed and alert."""
        percent_used = self.total_spent / self.budget_limit
        
        for threshold in self.alert_thresholds:
            if percent_used >= threshold and threshold not in self.alerted_thresholds:
                self.alerted_thresholds.add(threshold)
                self._alert(
                    f"⚠️  BUDGET ALERT: {threshold * 100:.0f}% "
                    f"(${self.total_spent:.2f} / ${self.budget_limit:.2f})"
                )
    
    def _alert(self, message: str):
        """Send budget alert."""
        print("\n" + "=" * 70)
        print(message)
        print("=" * 70 + "\n")
    
    def estimate_cost(
        self,
        model: str,
        num_tasks: int,
        avg_prompt_tokens: int,
        avg_completion_tokens: int,
    ) -> float:
        """
        Estimate cost before running experiments.
        
        Args:
            model: Model name
            num_tasks: Number of tasks to run
            avg_prompt_tokens: Average input tokens per task
            avg_completion_tokens: Average output tokens per task
        
        Returns:
            Estimated total cost in USD
        """
        cost_per_task = self._calculate_cost(
            model, avg_prompt_tokens, avg_completion_tokens
        )
        return cost_per_task * num_tasks
    
    def can_afford(self, estimated_cost: float) -> bool:
        """Check if estimated cost fits within remaining budget."""
        remaining = self.budget_limit - self.total_spent
        return estimated_cost <= remaining
    
    def get_summary(self) -> CostSummary:
        """Get cost summary across experiments."""
        by_model: Dict[str, float] = {}
        by_benchmark: Dict[str, float] = {}
        by_agent: Dict[str, float] = {}
        
        for record in self.usage_records:
            # By model
            by_model[record.model] = by_model.get(record.model, 0.0) + record.cost
            
            # By benchmark
            if record.benchmark:
                by_benchmark[record.benchmark] = (
                    by_benchmark.get(record.benchmark, 0.0) + record.cost
                )
            
            # By agent type
            if record.agent_type:
                by_agent[record.agent_type] = (
                    by_agent.get(record.agent_type, 0.0) + record.cost
                )
        
        return CostSummary(
            total_cost=self.total_spent,
            total_tokens=self.total_tokens,
            by_model=by_model,
            by_benchmark=by_benchmark,
            by_agent=by_agent,
            num_calls=len(self.usage_records),
        )
    
    def print_summary(self):
        """Print formatted cost summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("COST SUMMARY")
        print("=" * 70)
        print(f"Total Spent: ${summary.total_cost:.2f} / ${self.budget_limit:.2f}")
        print(f"Total Tokens: {summary.total_tokens:,}")
        print(f"API Calls: {summary.num_calls}")
        print(f"Budget Used: {(summary.total_cost / self.budget_limit) * 100:.1f}%")
        
        if summary.by_model:
            print("\nBy Model:")
            for model, cost in sorted(summary.by_model.items(), key=lambda x: -x[1]):
                print(f"  {model}: ${cost:.2f}")
        
        if summary.by_benchmark:
            print("\nBy Benchmark:")
            for bench, cost in sorted(summary.by_benchmark.items(), key=lambda x: -x[1]):
                print(f"  {bench}: ${cost:.2f}")
        
        if summary.by_agent:
            print("\nBy Agent Type:")
            for agent, cost in sorted(summary.by_agent.items(), key=lambda x: -x[1]):
                print(f"  {agent}: ${cost:.2f}")
        
        print("=" * 70 + "\n")
    
    def reset(self):
        """Reset tracker (use with caution!)."""
        self.total_spent = 0.0
        self.total_tokens = 0
        self.usage_records = []
        self.alerted_thresholds = set()
        self._save_records()


def estimate_experiment_cost(
    models: List[str],
    benchmarks: Dict[str, int],  # benchmark_name -> num_tasks
    avg_tokens_per_task: int = 6000,
    agent_multiplier: float = 3.0,  # Agentic calls use ~3x more tokens
) -> Dict[str, float]:
    """
    Estimate cost for a full experiment.
    
    Args:
        models: List of model names to test
        benchmarks: Dict of benchmark names to number of tasks
        avg_tokens_per_task: Average tokens per task (input + output)
        agent_multiplier: Token multiplier for agentic architectures
    
    Returns:
        Dict with cost breakdown
    """
    tracker = CostTracker()
    
    # Assume 60/40 split for input/output tokens
    avg_input = int(avg_tokens_per_task * 0.6)
    avg_output = int(avg_tokens_per_task * 0.4)
    
    costs = {}
    total = 0.0
    
    for model in models:
        model_cost = 0.0
        
        for benchmark, num_tasks in benchmarks.items():
            # Non-agentic baseline
            baseline_cost = tracker.estimate_cost(
                model, num_tasks, avg_input, avg_output
            )
            
            # Agentic architectures (4 types: sequential, concurrent, group, one-shot)
            # One-shot is similar to baseline, others use ~3x tokens
            agentic_cost = (
                tracker.estimate_cost(model, num_tasks, avg_input, avg_output) +  # one-shot
                3 * tracker.estimate_cost(
                    model, num_tasks,
                    int(avg_input * agent_multiplier),
                    int(avg_output * agent_multiplier)
                )  # other 3 architectures
            )
            
            model_cost += baseline_cost + agentic_cost
        
        costs[model] = model_cost
        total += model_cost
    
    costs["TOTAL"] = total
    return costs
