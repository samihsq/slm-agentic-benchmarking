"""
SOTA Baseline Agent for Benchmarking.

A non-agentic baseline using frontier models (GPT-4o, GPT-4o-mini)
for direct comparison against SLM agentic architectures.

This provides the upper-bound performance baseline.
"""

import litellm
from typing import Optional, Dict, Any

from .base_agent import BaseAgent, BenchmarkResponse
from ..config import get_llm_config


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
        model: str = "gpt-4o-mini",  # Default to cost-efficient SOTA
        verbose: bool = False,
    ):
        """
        Initialize baseline agent.
        
        Args:
            model: SOTA model to use (gpt-4o or gpt-4o-mini)
            verbose: Enable verbose output
        """
        super().__init__(
            model=model,
            verbose=verbose,
            max_iterations=1,
        )
        
        # Validate it's a SOTA model
        if not model.startswith("gpt-4o"):
            print(f"Warning: BaselineAgent is designed for GPT-4o models, got {model}")
        
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

        try:
            # Direct LiteLLM call to Azure OpenAI
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
                "api_version": "2024-08-01-preview",
            }
            
            response = litellm.completion(**llm_kwargs)
            
            result_text = response.choices[0].message.content
            
            # Track token usage for cost estimation
            usage = response.usage
            if self.verbose:
                print(f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")
            
        except Exception as e:
            # Fallback response on error
            result_text = f'{{"reasoning": "Error calling frontier model: {str(e)[:200]}", "confidence": 0.0, "response": "I encountered an error processing this task."}}'
            if self.verbose:
                print(f"Error in BaselineAgent: {e}")

        # Parse the response
        parsed = self.parse_json_response(result_text)

        # Add token usage to metadata if available
        if 'usage' in locals():
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


def get_baseline_agent(cost_efficient: bool = True, verbose: bool = False) -> BaselineAgent:
    """
    Convenience function to get a baseline agent.
    
    Args:
        cost_efficient: If True, use gpt-4o-mini. If False, use gpt-4o.
        verbose: Enable verbose output
    
    Returns:
        Configured BaselineAgent
    """
    model = "gpt-4o-mini" if cost_efficient else "gpt-4o"
    return BaselineAgent(model=model, verbose=verbose)
