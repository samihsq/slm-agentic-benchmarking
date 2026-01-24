"""
MedQA/MedMCQA Benchmark Runner.

Traditional medical question-answering benchmarks:
- MedQA: USMLE-style questions (1,273 questions)
- MedMCQA: Indian medical MCQs (194,000+ questions)

These test medical knowledge rather than agentic capabilities.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from ...agents.base_agent import BaseAgent, EvaluationResult
from ...evaluation.cost_tracker import CostTracker


class MedQARunner:
    """
    Runner for MedQA and MedMCQA benchmarks.
    
    These are traditional multiple-choice question benchmarks
    that test medical knowledge.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        dataset: str = "medqa",  # "medqa" or "medmcqa"
    ):
        """
        Initialize MedQA runner.
        
        Args:
            agent: Agent to evaluate
            cost_tracker: Optional cost tracker
            verbose: Enable verbose output
            dataset: Which dataset to use ("medqa" or "medmcqa")
        """
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.dataset = dataset
    
    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load MedQA/MedMCQA tasks.
        
        Args:
            limit: Maximum number of questions to load
        
        Returns:
            List of question dicts
        """
        # Sample questions for demonstration
        # In production, load from HuggingFace datasets:
        # from datasets import load_dataset
        # dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options")
        
        sample_questions = [
            {
                "question_id": "medqa_001",
                "question": "A 65-year-old man presents with progressive memory loss and difficulty with daily activities. MRI shows hippocampal atrophy. What is the most likely diagnosis?",
                "options": {
                    "A": "Alzheimer's disease",
                    "B": "Vascular dementia",
                    "C": "Frontotemporal dementia",
                    "D": "Lewy body dementia",
                },
                "answer": "A",
                "explanation": "Hippocampal atrophy is characteristic of Alzheimer's disease.",
            },
            {
                "question_id": "medqa_002",
                "question": "A patient with type 2 diabetes presents with polyuria, polydipsia, and weight loss despite increased appetite. Blood glucose is 350 mg/dL. What is the most appropriate initial treatment?",
                "options": {
                    "A": "Metformin",
                    "B": "Insulin",
                    "C": "Sulfonylurea",
                    "D": "DPP-4 inhibitor",
                },
                "answer": "B",
                "explanation": "High glucose with symptoms suggests need for insulin initially.",
            },
            {
                "question_id": "medqa_003",
                "question": "A 30-year-old woman presents with chest pain that worsens with inspiration. ECG shows ST elevation in leads II, III, aVF. What is the most likely diagnosis?",
                "options": {
                    "A": "Acute MI",
                    "B": "Pericarditis",
                    "C": "Pulmonary embolism",
                    "D": "Aortic dissection",
                },
                "answer": "B",
                "explanation": "ST elevation in inferior leads with pleuritic pain suggests pericarditis.",
            },
        ]
        
        return sample_questions[:limit] if limit else sample_questions
    
    def format_question(self, question: Dict[str, Any]) -> str:
        """Format question with options for the agent."""
        options_str = "\n".join([f"{k}. {v}" for k, v in question["options"].items()])
        return f"{question['question']}\n\nOptions:\n{options_str}\n\nAnswer with the letter (A, B, C, or D) and brief explanation."
    
    def parse_answer(self, response: str, correct_answer: str) -> tuple[bool, float]:
        """
        Parse agent's answer and check correctness.
        
        Returns:
            (is_correct, confidence)
        """
        response_upper = response.upper()
        
        # Look for answer letter
        for letter in ["A", "B", "C", "D"]:
            if letter in response_upper and correct_answer.upper() == letter:
                # Check if it's clearly the answer (not just mentioned)
                if f"answer is {letter}" in response_upper or f"{letter}." in response_upper[:50]:
                    return True, 1.0
        
        # Check if correct answer is mentioned
        if correct_answer.upper() in response_upper:
            return True, 0.8
        
        return False, 0.0
    
    def run(
        self,
        limit: Optional[int] = None,
        save_results: bool = True,
    ) -> List[EvaluationResult]:
        """
        Run MedQA evaluation.
        
        Args:
            limit: Maximum number of questions
            save_results: Whether to save results
        
        Returns:
            List of evaluation results
        """
        questions = self.load_tasks(limit)
        results = []
        
        print(f"\nRunning {self.dataset.upper()} with {len(questions)} questions...")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Model: {self.agent.model}\n")
        
        correct = 0
        
        for i, question in enumerate(questions, 1):
            if self.verbose:
                print(f"Question {i}/{len(questions)}: {question['question_id']}")
            
            start_time = time.time()
            
            # Format question
            formatted_task = self.format_question(question)
            
            # Run agent
            context = {"benchmark_type": "medical"}
            response = self.agent.respond_to_task(formatted_task, context)
            
            latency = time.time() - start_time
            
            # Check answer
            is_correct, confidence = self.parse_answer(
                response.response,
                question["answer"]
            )
            
            if is_correct:
                correct += 1
            
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
                        task_id=question["question_id"],
                        benchmark=self.dataset,
                        agent_type=self.agent.__class__.__name__,
                    )
            
            result = EvaluationResult(
                task_id=question["question_id"],
                prompt=formatted_task,
                agent_response=response.response,
                success=is_correct,
                score=confidence,
                latency=latency,
                cost=cost,
                metadata={
                    "correct_answer": question["answer"],
                    "explanation": question.get("explanation"),
                    "reasoning": response.reasoning,
                },
            )
            
            results.append(result)
            
            if self.verbose:
                status = "✓" if is_correct else "✗"
                print(f"  {status} Correct: {question['answer']}, Latency: {latency:.2f}s")
        
        accuracy = correct / len(questions) if questions else 0.0
        print(f"\nAccuracy: {accuracy * 100:.1f}% ({correct}/{len(questions)})")
        
        if save_results:
            self._save_results(results, accuracy)
        
        return results
    
    def _save_results(self, results: List[EvaluationResult], accuracy: float):
        """Save results to file."""
        output_dir = Path("results") / self.dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent.model}_{self.agent.__class__.__name__}_{timestamp}.json"
        output_file = output_dir / filename
        
        data = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": self.dataset.upper(),
            "accuracy": accuracy,
            "num_questions": len(results),
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
