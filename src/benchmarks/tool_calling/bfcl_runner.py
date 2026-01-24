"""
BFCL (Berkeley Function Calling Leaderboard) Runner.

BFCL v3 evaluates function calling capabilities with:
- Simple function calls
- Parallel function calls
- Nested function calls

This provides a standardized baseline for tool-calling evaluation.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from ...agents.base_agent import BaseAgent, EvaluationResult
from ...evaluation.cost_tracker import CostTracker


class BFCLRunner:
    """
    Runner for BFCL v3 evaluation.
    
    BFCL tests function calling in various scenarios:
    - Single function calls
    - Multiple parallel calls
    - Nested/sequential calls
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
    ):
        """
        Initialize BFCL runner.
        
        Args:
            agent: Agent to evaluate
            cost_tracker: Optional cost tracker
            verbose: Enable verbose output
        """
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
    
    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load BFCL tasks.
        
        Args:
            limit: Maximum number of tasks
        
        Returns:
            List of task dicts
        """
        # Sample BFCL-style tasks
        sample_tasks = [
            {
                "task_id": "bfcl_001",
                "type": "simple",
                "description": "Call the get_weather function for San Francisco",
                "functions": [
                    {
                        "name": "get_weather",
                        "description": "Get current weather for a location",
                        "parameters": {
                            "location": "string",
                        },
                    },
                ],
                "expected_call": {
                    "name": "get_weather",
                    "arguments": {"location": "San Francisco"},
                },
            },
            {
                "task_id": "bfcl_002",
                "type": "parallel",
                "description": "Get weather for both New York and Los Angeles",
                "functions": [
                    {
                        "name": "get_weather",
                        "description": "Get current weather for a location",
                        "parameters": {
                            "location": "string",
                        },
                    },
                ],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"location": "New York"}},
                    {"name": "get_weather", "arguments": {"location": "Los Angeles"}},
                ],
            },
            {
                "task_id": "bfcl_003",
                "type": "nested",
                "description": "Search for restaurants in San Francisco, then get details for the first result",
                "functions": [
                    {
                        "name": "search_restaurants",
                        "description": "Search for restaurants",
                        "parameters": {
                            "location": "string",
                        },
                    },
                    {
                        "name": "get_restaurant_details",
                        "description": "Get details for a restaurant",
                        "parameters": {
                            "restaurant_id": "string",
                        },
                    },
                ],
                "expected_sequence": [
                    {"name": "search_restaurants", "arguments": {"location": "San Francisco"}},
                    {"name": "get_restaurant_details", "arguments": {"restaurant_id": "<from_first_call>"}},
                ],
            },
        ]
        
        return sample_tasks[:limit] if limit else sample_tasks
    
    def format_task(self, task: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Format task with function descriptions."""
        funcs_desc = "\n".join([
            f"- {f['name']}: {f['description']} (params: {f.get('parameters', {})})"
            for f in task.get("functions", [])
        ])
        
        task_text = f"{task['description']}\n\nAvailable functions:\n{funcs_desc}"
        
        context = {
            "benchmark_type": "tool_calling",
            "tools": task.get("functions", []),
            "task_type": task.get("type", "simple"),
        }
        
        return task_text, context
    
    def check_function_call(self, response: BenchmarkResponse, expected: Dict[str, Any]) -> tuple[bool, float]:
        """Check if function was called correctly."""
        tool_calls = response.metadata.get("tool_calls", [])
        response_text = response.response.lower()
        
        func_name = expected.get("name", "").lower()
        
        # Check metadata first
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("name", "").lower() == func_name:
                    # Check arguments match
                    args = tc.get("arguments", {})
                    expected_args = expected.get("arguments", {})
                    matches = sum(1 for k, v in expected_args.items() if args.get(k) == v)
                    score = matches / len(expected_args) if expected_args else 1.0
                    return matches == len(expected_args), score
        
        # Fallback: check if function name appears
        if func_name in response_text:
            return True, 0.5
        
        return False, 0.0
    
    def run(
        self,
        limit: Optional[int] = None,
        save_results: bool = True,
    ) -> List[EvaluationResult]:
        """
        Run BFCL evaluation.
        
        Args:
            limit: Maximum number of tasks
            save_results: Whether to save results
        
        Returns:
            List of evaluation results
        """
        tasks = self.load_tasks(limit)
        results = []
        
        print(f"\nRunning BFCL v3 with {len(tasks)} tasks...")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Model: {self.agent.model}\n")
        
        successful = 0
        
        for i, task in enumerate(tasks, 1):
            if self.verbose:
                print(f"Task {i}/{len(tasks)}: {task['task_id']} ({task.get('type', 'simple')})")
            
            start_time = time.time()
            
            # Format task
            task_text, context = self.format_task(task)
            
            # Run agent
            response = self.agent.respond_to_task(task_text, context)
            
            latency = time.time() - start_time
            
            # Check function call based on task type
            if task.get("type") == "parallel":
                expected_calls = task.get("expected_calls", [])
                success = all(
                    self.check_function_call(response, exp)[0]
                    for exp in expected_calls
                )
                score = sum(
                    self.check_function_call(response, exp)[1]
                    for exp in expected_calls
                ) / len(expected_calls) if expected_calls else 0.0
            elif task.get("type") == "nested":
                # For nested, check if first call happened
                expected_seq = task.get("expected_sequence", [])
                if expected_seq:
                    success, score = self.check_function_call(response, expected_seq[0])
                else:
                    success, score = False, 0.0
            else:
                # Simple call
                expected = task.get("expected_call", {})
                success, score = self.check_function_call(response, expected)
            
            if success:
                successful += 1
            
            # Calculate cost
            cost = 0.0
            if self.cost_tracker and response.metadata:
                prompt_tokens = response.metadata.get("prompt_tokens", 0)
                completion_tokens = response.metadata.get("completion_tokens", 0)
                if prompt_tokens > 0:
                    cost = self.cost_tracker.log_usage(
                        model=self.agent.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        task_id=task["task_id"],
                        benchmark="bfcl",
                        agent_type=self.agent.__class__.__name__,
                    )
            
            result = EvaluationResult(
                task_id=task["task_id"],
                prompt=task_text,
                agent_response=response.response,
                success=success,
                score=score,
                latency=latency,
                cost=cost,
                metadata={
                    "task_type": task.get("type", "simple"),
                    "tool_calls": response.metadata.get("tool_calls", []),
                    "reasoning": response.reasoning,
                },
            )
            
            results.append(result)
            
            if self.verbose:
                status = "✓" if success else "✗"
                print(f"  {status} Score: {score:.2f}, Latency: {latency:.2f}s")
        
        success_rate = successful / len(tasks) if tasks else 0.0
        print(f"\nSuccess Rate: {success_rate * 100:.1f}% ({successful}/{len(tasks)})")
        
        if save_results:
            self._save_results(results, success_rate)
        
        return results
    
    def _save_results(self, results: List[EvaluationResult], success_rate: float):
        """Save results to file."""
        output_dir = Path("results") / "bfcl"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent.model}_{self.agent.__class__.__name__}_{timestamp}.json"
        output_file = output_dir / filename
        
        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": "BFCL v3",
            "success_rate": success_rate,
            "num_tasks": len(results),
            "results": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "score": r.score,
                    "latency": r.latency,
                    "cost": r.cost,
                    "response": r.agent_response[:300],
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to: {output_file}")
