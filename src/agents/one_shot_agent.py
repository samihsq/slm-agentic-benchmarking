"""
One-Shot Agent for Benchmarking.

A simple single-agent baseline that directly calls the LLM once
without CrewAI's ReAct loop overhead.

This is the true non-agentic baseline - 1 task = 1 LLM call.
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


class OneShotAgent(BaseAgent):
    """
    One-shot benchmark agent using a single direct LLM call.

    Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                    One-Shot Response                      │
    │                                                          │
    │  Task → [Direct LLM Call] → Response                     │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

    Bypasses CrewAI entirely to avoid ReAct loop overhead.
    True 1:1 ratio of tasks to LLM calls - the non-agentic baseline.
    """

    def __init__(
        self,
        model: str = "phi-4",
        verbose: bool = False,
    ):
        super().__init__(
            model=model,
            verbose=verbose,
            max_iterations=1,
        )

        # Get model config for direct LiteLLM calls
        self.config = get_llm_config(model)
        self.model_id = self.config["model"]

    def respond_to_task(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResponse:
        """
        Generate a response with a single LLM call.

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

        # Azure/gpt-4o support at most 16384 completion tokens; allow per-task override
        max_tokens = (context or {}).get("max_completion_tokens", 16384)

        # Prepare LiteLLM kwargs
        llm_kwargs = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "api_key": self.config["azure_api_key"],
            "api_base": self.config["azure_endpoint"],
        }

        result_text = None
        last_error = None
        usage = None

        # Retry with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                response = litellm.completion(**llm_kwargs)
                result_text = response.choices[0].message.content
                usage = response.usage
                break  # Success

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                    if self.verbose:
                        print(
                            f"Retry {attempt + 1}/{MAX_RETRIES} after {delay:.1f}s: {str(e)[:100]}"
                        )
                    time.sleep(delay)
                else:
                    if self.verbose:
                        print(f"All {MAX_RETRIES} retries failed: {e}")

        if result_text is None:
            result_text = f'{{"reasoning": "Error after {MAX_RETRIES} retries: {str(last_error)[:150]}", "confidence": 0.0, "response": "Error: {str(last_error)[:100]}"}}'

        # Parse the response using robust parser from base class
        parsed = self.parse_json_response(result_text)

        # Add token usage to metadata if available
        if usage is not None:
            parsed.metadata = parsed.metadata or {}
            parsed.metadata.update(
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            )

        # Add to history
        self.add_to_history(
            task=task,
            response=parsed.response,
            reasoning=parsed.reasoning,
            success=parsed.success,
        )

        return parsed
