"""
SOTA Baseline Agent for Benchmarking.

A non-agentic baseline using frontier models (GPT-4o, GPT-4o-mini)
for direct comparison against SLM agentic architectures.

This provides the upper-bound performance baseline.
"""

import litellm
import time
import random
from typing import Optional, Dict, Any

from .base_agent import BaseAgent, BenchmarkResponse
from ..config import get_llm_config

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds


class BaselineAgent(BaseAgent):
    """
    SOTA non-agentic baseline using frontier models.

    Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                  SOTA Baseline (GPT-4o)                   │
    │                                                          │
    │  Task → [Single Frontier Model Call] → Response          │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

    Provides upper-bound performance for comparison.
    Identical to OneShotAgent but defaults to GPT-4o/GPT-4o-mini.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b",  # Default to Llama 3.3 70B for SOTA baseline
        verbose: bool = False,
    ):
        """
        Initialize baseline agent for SOTA comparison.
        
        Args:
            model: Model to use (llama-3.3-70b for SOTA, or any other model)
            verbose: Enable verbose output
        """
        super().__init__(
            model=model,
            verbose=verbose,
            max_iterations=1,
        )
        
        # Get model config for direct LiteLLM calls
        self.config = get_llm_config(model)
        self.model_id = self.config["model"]

    def respond_to_task(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResponse:
        """
        Generate a response with a single frontier model call.

        Args:
            task: The benchmark task or question
            context: Additional context (tools, data, etc.)

        Returns:
            BenchmarkResponse with the response and reasoning
        """
        
        # Determine benchmark type from context
        benchmark_type = (context or {}).get("benchmark_type", "general")
        system_prompt = self.get_system_prompt(benchmark_type)

        # Add context information if provided
        context_str = ""
        if context:
            if "tools" in context:
                context_str += f"\n\nAvailable tools: {context['tools']}"
            if "patient_data" in context:
                context_str += f"\n\nPatient data: {context['patient_data']}"
            if "additional_info" in context:
                context_str += f"\n\n{context['additional_info']}"

        user_message = f"TASK: {task}{context_str}"

        # Direct LiteLLM call with exponential backoff retry
        llm_kwargs = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "api_key": self.config["azure_api_key"],
            "api_base": self.config["azure_endpoint"],
        }
        
        result_text = None
        usage = None
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                response = litellm.completion(**llm_kwargs)
                result_text = response.choices[0].message.content
                usage = response.usage
                
                if self.verbose:
                    print(f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")
                break  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff with jitter
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                    if self.verbose:
                        print(f"Retry {attempt + 1}/{MAX_RETRIES} after {delay:.1f}s: {str(e)[:100]}")
                    time.sleep(delay)
                else:
                    if self.verbose:
                        print(f"All {MAX_RETRIES} retries failed: {e}")
        
        if result_text is None:
            # All retries failed
            result_text = f'{{"reasoning": "Error after {MAX_RETRIES} retries: {str(last_error)[:150]}", "confidence": 0.0, "response": "Error: {str(last_error)[:100]}"}}'

        # Parse the response
        parsed = self.parse_json_response(result_text)

        # Add token usage to metadata if available
        if usage is not None:
            parsed.metadata = parsed.metadata or {}
            parsed.metadata.update({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            })

        # Add to history
        self.add_to_history(
            task=task,
            response=parsed.response,
            reasoning=parsed.reasoning,
            success=parsed.success,
        )

        return parsed


def get_baseline_agent(model: str = "llama-3.3-70b", verbose: bool = False) -> BaselineAgent:
    """
    Convenience function to get a SOTA baseline agent.
    
    Args:
        model: Model to use (llama-3.3-70b for SOTA comparison)
        verbose: Enable verbose output
    
    Returns:
        Configured BaselineAgent
    """
    return BaselineAgent(model=model, verbose=verbose)
