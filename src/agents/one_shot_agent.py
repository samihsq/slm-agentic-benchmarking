"""
One-Shot Agent for Benchmarking.

A simple single-agent baseline that directly calls the LLM once
without CrewAI's ReAct loop overhead.

This is the true non-agentic baseline - 1 task = 1 LLM call.
"""

import litellm
from typing import Optional, Dict, Any

from .base_agent import BaseAgent, BenchmarkResponse
from ..config import get_llm_config


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
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
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

        try:
            # Prepare LiteLLM kwargs based on provider
            llm_kwargs = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.7,
                "max_tokens": 2048,
            }
            
            # Add provider-specific configuration
            if self.config["provider"] == "openai":
                llm_kwargs["api_key"] = self.config["azure_api_key"]
                llm_kwargs["api_base"] = self.config["azure_endpoint"]
                llm_kwargs["api_version"] = "2024-08-01-preview"
            elif self.config.get("requires_endpoint"):
                llm_kwargs["api_base"] = self.config["endpoint_url"]
                llm_kwargs["api_key"] = "dummy"  # Azure ML endpoints may not need key
            else:
                llm_kwargs["api_key"] = self.config.get("azure_api_key", "dummy")
            
            # Direct LiteLLM call - bypasses CrewAI overhead
            response = litellm.completion(**llm_kwargs)
            
            result_text = response.choices[0].message.content
            
        except Exception as e:
            # Fallback response on error
            result_text = f'{{"reasoning": "Error calling LLM: {str(e)[:200]}", "confidence": 0.0, "response": "I encountered an error processing this task."}}'
            if self.verbose:
                print(f"Error in OneShotAgent: {e}")

        # Parse the response using robust parser from base class
        parsed = self.parse_json_response(result_text)

        # Add to history
        self.add_to_history(
            task=task,
            response=parsed.response,
            reasoning=parsed.reasoning,
            success=parsed.success,
        )

        return parsed
