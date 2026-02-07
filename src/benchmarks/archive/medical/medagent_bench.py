"""
MedAgentBench Integration.

MedAgentBench provides 100 clinically-derived agentic tasks with a FHIR-compliant
interactive environment for evaluating LLM agents in medical applications.

GitHub: https://github.com/stanfordmlgroup/MedAgentBench

To use:
1. Clone the repository: git clone https://github.com/stanfordmlgroup/MedAgentBench
2. Install dependencies from their requirements.txt
3. Set MEDAGENT_BENCH_PATH environment variable
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from ....agents.base_agent import BaseAgent, BenchmarkResponse, EvaluationResult
from ....evaluation.cost_tracker import CostTracker


class MedAgentBenchRunner:
    """
    Runner for MedAgentBench evaluation.
    
    MedAgentBench tests agents on 100 clinical tasks across 10 categories:
    - Diagnosis
    - Treatment planning
    - Medication management
    - Lab result interpretation
    - Patient history analysis
    - Etc.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
    ):
        """
        Initialize MedAgentBench runner.
        
        Args:
            agent: Agent to evaluate
            cost_tracker: Optional cost tracker
            verbose: Enable verbose output
        """
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        
        # Look for MedAgentBench installation
        self.bench_path = self._find_medagent_bench()
    
    def _find_medagent_bench(self) -> Optional[Path]:
        """Find MedAgentBench installation."""
        # Check environment variable
        env_path = os.getenv("MEDAGENT_BENCH_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path
        
        # Check common locations
        common_paths = [
            Path.cwd() / "MedAgentBench",
            Path.cwd() / "benchmarks" / "MedAgentBench",
            Path.home() / "MedAgentBench",
        ]
        
        for path in common_paths:
            if path.exists():
                return path
        
        print(
            "Warning: MedAgentBench not found. "
            "Please clone it from https://github.com/stanfordmlgroup/MedAgentBench "
            "and set MEDAGENT_BENCH_PATH environment variable."
        )
        return None
    
    def load_tasks(self, subset: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load MedAgentBench tasks.
        
        Args:
            subset: Optional category subset (e.g., "diagnosis", "treatment")
            limit: Maximum number of tasks to load
        
        Returns:
            List of task dicts
        """
        if not self.bench_path:
            # Return sample tasks for demonstration
            return self._get_sample_tasks(limit or 5)
        
        # TODO: Load actual tasks from MedAgentBench
        # tasks_file = self.bench_path / "data" / "tasks.json"
        # with open(tasks_file) as f:
        #     all_tasks = json.load(f)
        
        # For now, return sample tasks
        return self._get_sample_tasks(limit or 10)
    
    def _get_sample_tasks(self, num: int) -> List[Dict[str, Any]]:
        """Get sample medical tasks for demonstration."""
        samples = [
            {
                "task_id": "medagent_001",
                "category": "diagnosis",
                "task": "A 45-year-old patient presents with chest pain, shortness of breath, and elevated troponin levels. What is the most likely diagnosis?",
                "patient_data": {
                    "age": 45,
                    "symptoms": ["chest pain", "shortness of breath"],
                    "lab_results": {"troponin": "elevated"},
                },
                "ground_truth": "acute myocardial infarction",
            },
            {
                "task_id": "medagent_002",
                "category": "medication",
                "task": "Patient on warfarin with INR of 5.2. What is the appropriate management?",
                "patient_data": {
                    "medications": ["warfarin 5mg daily"],
                    "lab_results": {"INR": 5.2},
                },
                "ground_truth": "hold warfarin and give vitamin K",
            },
            {
                "task_id": "medagent_003",
                "category": "lab_interpretation",
                "task": "Interpret the following CBC results: WBC 15,000, Hgb 10.2, Platelets 450,000. What is the most concerning finding?",
                "patient_data": {
                    "lab_results": {
                        "WBC": 15000,
                        "hemoglobin": 10.2,
                        "platelets": 450000,
                    },
                },
                "ground_truth": "leukocytosis with anemia",
            },
        ]
        return samples[:num]
    
    def run(
        self,
        subset: Optional[str] = None,
        limit: Optional[int] = None,
        save_results: bool = True,
    ) -> List[EvaluationResult]:
        """
        Run MedAgentBench evaluation.
        
        Args:
            subset: Optional category subset
            limit: Maximum number of tasks
            save_results: Whether to save results to file
        
        Returns:
            List of evaluation results
        """
        tasks = self.load_tasks(subset, limit)
        results = []
        
        print(f"\nRunning MedAgentBench with {len(tasks)} tasks...")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Model: {self.agent.model}\n")
        
        for i, task in enumerate(tasks, 1):
            if self.verbose:
                print(f"Task {i}/{len(tasks)}: {task['task_id']}")
            
            start_time = time.time()
            
            # Prepare context
            context = {
                "benchmark_type": "medical",
                "patient_data": json.dumps(task.get("patient_data", {})),
                "category": task.get("category", ""),
            }
            
            # Run agent
            response = self.agent.respond_to_task(task["task"], context)
            
            latency = time.time() - start_time
            
            # Calculate cost (if tokens available in response metadata)
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
                        benchmark="medagent",
                        agent_type=self.agent.__class__.__name__,
                    )
            
            # Create evaluation result
            result = EvaluationResult(
                task_id=task["task_id"],
                prompt=task["task"],
                agent_response=response.response,
                success=response.success,
                score=response.confidence,
                latency=latency,
                cost=cost,
                metadata={
                    "category": task.get("category"),
                    "ground_truth": task.get("ground_truth"),
                    "reasoning": response.reasoning,
                },
            )
            
            results.append(result)
            
            if self.verbose:
                print(f"  Success: {response.success}, Confidence: {response.confidence:.2f}, Latency: {latency:.2f}s")
        
        # Save results
        if save_results:
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: List[EvaluationResult]):
        """Save results to file."""
        output_dir = Path("results") / "medagent"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent.model}_{self.agent.__class__.__name__}_{timestamp}.json"
        
        output_file = output_dir / filename
        
        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": "MedAgentBench",
            "num_tasks": len(results),
            "results": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "score": r.score,
                    "latency": r.latency,
                    "cost": r.cost,
                    "response": r.agent_response[:500],  # Truncate for file size
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
