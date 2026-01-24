"""
MCP-Bench Integration.

MCP-Bench evaluates tool-calling capabilities with 250+ tools across 28 MCP servers.
Tests multi-hop planning, parameter precision, and cross-domain coordination.

Reference: arXiv:2508.20453
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from ...agents.base_agent import BaseAgent, EvaluationResult
from ...evaluation.cost_tracker import CostTracker


class MCPBenchRunner:
    """
    Runner for MCP-Bench evaluation.
    
    MCP-Bench tests agents on realistic tool-use tasks requiring:
    - Tool schema understanding
    - Multi-step planning
    - Parameter control
    - Cross-domain coordination
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
    ):
        """
        Initialize MCP-Bench runner.
        
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
        Load MCP-Bench tasks.
        
        Args:
            limit: Maximum number of tasks
        
        Returns:
            List of task dicts
        """
        # Sample tool-calling tasks
        # In production, load from MCP-Bench dataset
        sample_tasks = [
            {
                "task_id": "mcp_001",
                "description": "Search for papers about 'machine learning in healthcare' and get the top 3 results",
                "tools": [
                    {
                        "name": "search_papers",
                        "description": "Search academic papers",
                        "parameters": {
                            "query": "string",
                            "max_results": "integer",
                        },
                    },
                    {
                        "name": "get_paper_details",
                        "description": "Get details for a specific paper",
                        "parameters": {
                            "paper_id": "string",
                        },
                    },
                ],
                "expected_tool_calls": ["search_papers", "get_paper_details"],
            },
            {
                "task_id": "mcp_002",
                "description": "Convert 100 USD to EUR using current exchange rates",
                "tools": [
                    {
                        "name": "get_exchange_rate",
                        "description": "Get current exchange rate between currencies",
                        "parameters": {
                            "from_currency": "string",
                            "to_currency": "string",
                        },
                    },
                    {
                        "name": "convert_currency",
                        "description": "Convert amount between currencies",
                        "parameters": {
                            "amount": "number",
                            "from_currency": "string",
                            "to_currency": "string",
                        },
                    },
                ],
                "expected_tool_calls": ["get_exchange_rate", "convert_currency"],
            },
            {
                "task_id": "mcp_003",
                "description": "Calculate the square root of 144 and then multiply by 5",
                "tools": [
                    {
                        "name": "calculate",
                        "description": "Perform mathematical calculations",
                        "parameters": {
                            "expression": "string",
                        },
                    },
                ],
                "expected_tool_calls": ["calculate"],
            },
        ]
        
        return sample_tasks[:limit] if limit else sample_tasks
    
    def format_task(self, task: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Format task with tool descriptions."""
        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']} (params: {t.get('parameters', {})})"
            for t in task.get("tools", [])
        ])
        
        task_text = f"{task['description']}\n\nAvailable tools:\n{tools_desc}"
        
        context = {
            "benchmark_type": "tool_calling",
            "tools": task.get("tools", []),
            "expected_tool_calls": task.get("expected_tool_calls", []),
        }
        
        return task_text, context
    
    def check_tool_usage(self, response: BenchmarkResponse, expected_tools: List[str]) -> tuple[bool, float]:
        """
        Check if agent used the expected tools.
        
        Returns:
            (success, score)
        """
        tool_calls = response.metadata.get("tool_calls", [])
        response_text = response.response.lower()
        
        # Check if tool calls are in metadata
        if tool_calls:
            used_tools = [tc.get("name", "") for tc in tool_calls if isinstance(tc, dict)]
            matches = sum(1 for tool in expected_tools if tool in used_tools)
            score = matches / len(expected_tools) if expected_tools else 0.0
            return matches == len(expected_tools), score
        
        # Fallback: check if tool names appear in response
        matches = sum(1 for tool in expected_tools if tool.lower() in response_text)
        score = matches / len(expected_tools) if expected_tools else 0.0
        
        return matches == len(expected_tools), score
    
    def run(
        self,
        limit: Optional[int] = None,
        save_results: bool = True,
    ) -> List[EvaluationResult]:
        """
        Run MCP-Bench evaluation.
        
        Args:
            limit: Maximum number of tasks
            save_results: Whether to save results
        
        Returns:
            List of evaluation results
        """
        tasks = self.load_tasks(limit)
        results = []
        
        print(f"\nRunning MCP-Bench with {len(tasks)} tasks...")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Model: {self.agent.model}\n")
        
        successful = 0
        
        for i, task in enumerate(tasks, 1):
            if self.verbose:
                print(f"Task {i}/{len(tasks)}: {task['task_id']}")
            
            start_time = time.time()
            
            # Format task
            task_text, context = self.format_task(task)
            
            # Run agent
            response = self.agent.respond_to_task(task_text, context)
            
            latency = time.time() - start_time
            
            # Check tool usage
            success, score = self.check_tool_usage(
                response,
                task.get("expected_tool_calls", [])
            )
            
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
                        benchmark="mcp_bench",
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
                    "expected_tools": task.get("expected_tool_calls", []),
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
        output_dir = Path("results") / "mcp_bench"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent.model}_{self.agent.__class__.__name__}_{timestamp}.json"
        output_file = output_dir / filename
        
        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": "MCP-Bench",
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
