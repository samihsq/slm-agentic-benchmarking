"""
Base agent interface for benchmarking evaluation.
All agent types must implement this interface.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class BenchmarkResponse:
    """Standard response format for all benchmark agents."""

    response: str  # The agent's response to the task
    reasoning: str  # Explanation of the reasoning process
    success: bool = False  # Whether the task was completed successfully
    confidence: float = 0.0  # Confidence score (0-1)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Result of a benchmark evaluation."""

    task_id: str
    prompt: str
    agent_response: str
    success: bool
    score: Optional[float] = None  # Normalized score (0-1)
    latency: Optional[float] = None  # Time taken in seconds
    cost: Optional[float] = None  # Estimated cost in USD
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """
    Base class for benchmarking agents.

    All agent implementations (one-shot, sequential, concurrent, group chat)
    must inherit from this class and implement the respond_to_task method.

    The goal is to evaluate how LLMs perform on various benchmarking tasks,
    comparing how different agent architectures affect performance.
    """

    def __init__(
        self,
        model: str = "phi-4",
        verbose: bool = False,
        max_iterations: int = 1,
    ):
        """
        Initialize the base agent.

        Args:
            model: LLM model to use for generating responses
            verbose: Enable verbose output
            max_iterations: Maximum reasoning iterations
        """
        self.model = model
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.response_history: List[Dict[str, Any]] = []
        self.iteration_count = 0

    @abstractmethod
    def respond_to_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> BenchmarkResponse:
        """
        Generate a response to a benchmark task.

        Args:
            task: The task instruction or question
            context: Additional context for the task (e.g., tools, patient data)

        Returns:
            BenchmarkResponse with the response and metadata
        """
        pass

    def add_to_history(
        self,
        task: str,
        response: str,
        reasoning: str,
        success: bool = False,
    ):
        """Add a response attempt to the agent's history."""
        self.response_history.append(
            {
                "iteration": self.iteration_count,
                "task": task,
                "response": response,
                "reasoning": reasoning,
                "success": success,
            }
        )
        self.iteration_count += 1

    def get_response_history(self, max_entries: Optional[int] = None) -> str:
        """Get formatted response history for context."""
        entries = self.response_history[-max_entries:] if max_entries else self.response_history
        if not entries:
            return "No previous tasks processed."

        history_parts = []
        for entry in entries:
            status = "SUCCESS" if entry["success"] else "FAILED"
            history_parts.append(
                f"[{entry['iteration']}] ({status})\n"
                f"Task: {entry['task'][:100]}...\n"
                f"Reasoning: {entry['reasoning'][:150]}...\n"
                f"Response: {entry['response'][:200]}..."
            )

        return "\n\n".join(history_parts)

    def reset(self):
        """Reset the agent state."""
        self.response_history = []
        self.iteration_count = 0

    def get_system_prompt(self, benchmark_type: str = "general", role: str = "oneshot") -> str:
        """
        Get the system prompt for task response generation.
        
        Args:
            benchmark_type: The benchmark type (medical, tool_calling, general)
            role: The agent role (oneshot, sequential_analyzer, etc.)
        
        Returns:
            The appropriate prompt string
        """
        try:
            from .prompts import get_prompt
            return get_prompt(benchmark_type, role)
        except (ImportError, FileNotFoundError, KeyError):
            # Fallback to basic prompts if prompt system not available
            return self._get_fallback_prompt(benchmark_type)
    
    def _get_fallback_prompt(self, benchmark_type: str) -> str:
        """Fallback prompts if the prompt system is not available."""
        prompts = {
            "medical": """You are a medical AI assistant. Answer the multiple-choice question by selecting the best answer.

OUTPUT FORMAT (required JSON):
{"reasoning": "<your analysis>", "answer": "<A, B, C, D, or E>", "confidence": <0.0-1.0>}""",
            
            "tool_calling": """You are an AI assistant for function calling. Determine the correct function call(s).

OUTPUT FORMAT (required JSON):
{"reasoning": "<your analysis>", "tool_calls": [{"name": "<func>", "arguments": {...}}], "confidence": <0.0-1.0>}""",
            
            "general": """You are an AI assistant. Answer the question accurately.

OUTPUT FORMAT (required JSON):
{"reasoning": "<your analysis>", "answer": "<your answer>", "confidence": <0.0-1.0>}"""
        }
        return prompts.get(benchmark_type, prompts["general"])

    def parse_json_response(self, result: str) -> BenchmarkResponse:
        """
        Robustly parse a JSON response from the LLM.
        
        Handles nested braces, markdown code blocks, and various edge cases.
        Falls back gracefully if JSON parsing fails.
        """
        # Try to find JSON in markdown code blocks first
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', result)
        if code_block_match:
            json_str = code_block_match.group(1)
            try:
                data = json.loads(json_str)
                # Handle both "response" and "answer" fields (prompts use "answer")
                response_text = data.get("response") or data.get("answer", "")
                return BenchmarkResponse(
                    response=str(response_text),
                    reasoning=data.get("reasoning", ""),
                    success=data.get("success", True),
                    confidence=float(data.get("confidence", 0.5)),
                    metadata={"raw_result": result[:500], "tool_calls": data.get("tool_calls", [])},
                )
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Try to find standalone JSON object with balanced braces
        for match in re.finditer(r'\{', result):
            start = match.start()
            depth = 0
            for i, char in enumerate(result[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        json_candidate = result[start:i+1]
                        try:
                            data = json.loads(json_candidate)
                            # Accept if has response, answer, or reasoning field
                            if "response" in data or "answer" in data or "reasoning" in data:
                                response_text = data.get("response") or data.get("answer", "")
                                return BenchmarkResponse(
                                    response=str(response_text),
                                    reasoning=data.get("reasoning", ""),
                                    success=data.get("success", True),
                                    confidence=float(data.get("confidence", 0.5)),
                                    metadata={"raw_result": result[:500], "tool_calls": data.get("tool_calls", [])},
                                )
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
        
        # Fallback: extract response from text patterns
        response = result
        reasoning = ""
        
        # Look for "Response:" pattern
        response_match = re.search(r'(?:^|\n)(?:Response|RESPONSE)[:\s]*(.+?)(?:\n\n|$)', result, re.DOTALL | re.IGNORECASE)
        if response_match:
            response = response_match.group(1).strip()
        
        # Look for "Reasoning:" pattern  
        reasoning_match = re.search(r'(?:^|\n)(?:Reasoning|REASONING)[:\s]*(.+?)(?:\n\n|Response|$)', result, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        # If no patterns found, use the raw result
        if response == result:
            paragraphs = [p.strip() for p in result.split('\n\n') if p.strip() and len(p.strip()) > 20]
            if paragraphs:
                response = paragraphs[0][:2000]
        
        return BenchmarkResponse(
            response=response[:2000],
            reasoning=reasoning if reasoning else "Extracted from unstructured response",
            success=True,
            confidence=0.5,
            metadata={"raw_result": result[:500], "parse_fallback": True},
        )
