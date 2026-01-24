"""
LLM Configuration for Azure AI Foundry via LiteLLM.

Supports both Azure OpenAI (serverless) and Azure ML endpoints.
Pricing as of January 2026.
"""

import os
from typing import Dict, Any, Optional
from crewai import LLM


# Available models on Azure AI
AVAILABLE_MODELS = {
    # ==========================================================================
    # AZURE OPENAI MODELS (Serverless - Pre-deployed)
    # ==========================================================================
    "gpt-4o": {
        "model": "azure/gpt-4o",
        "description": "GPT-4o - SOTA frontier baseline",
        "context_window": 128000,
        "cost_per_1m_input": 2.50,
        "cost_per_1m_output": 10.00,
        "provider": "openai",
        "serverless": True,
    },
    "gpt-4o-mini": {
        "model": "azure/gpt-4o-mini",
        "description": "GPT-4o-mini - Cost-efficient SOTA",
        "context_window": 128000,
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
        "provider": "openai",
        "serverless": True,
    },
    
    # ==========================================================================
    # AZURE AI FOUNDRY MODELS (Serverless)
    # ==========================================================================
    "phi-4": {
        "model": "azure_ai/Phi-4",
        "description": "Phi-4 14B - Microsoft native, Azure optimized",
        "context_window": 16384,
        "cost_per_1m_input": 0.07,
        "cost_per_1m_output": 0.14,
        "provider": "microsoft",
        "serverless": True,
    },
    "llama-3.3-70b": {
        "model": "azure_ai/Meta-Llama-3.3-70B-Instruct",
        "description": "Llama 3.3 70B - Strong baseline",
        "context_window": 128000,
        "cost_per_1m_input": 0.27,
        "cost_per_1m_output": 0.27,
        "provider": "meta",
        "serverless": True,
    },
    "mistral-small-3.1": {
        "model": "azure_ai/Mistral-Small-3.1",
        "description": "Mistral Small 3.1 24B - Fast and efficient",
        "context_window": 128000,
        "cost_per_1m_input": 0.10,
        "cost_per_1m_output": 0.30,
        "provider": "mistral",
        "serverless": True,
    },
    
    # ==========================================================================
    # AZURE ML ENDPOINTS (Custom Deployments from HuggingFace)
    # ==========================================================================
    "glm-4.7-flash": {
        "model": "azure_ml/glm-4-7-flash",
        "description": "GLM-4.7-Flash 30B MoE - SOTA reasoning/coding",
        "context_window": 128000,
        "cost_per_1m_input": 0.08,
        "cost_per_1m_output": 0.08,
        "provider": "zhipu",
        "serverless": False,
        "requires_endpoint": True,
    },
    "qwen3-30b-a3b": {
        "model": "azure_ml/qwen3-30b-a3b-thinking",
        "description": "Qwen3 30B MoE - Thinking/CoT mode",
        "context_window": 128000,
        "cost_per_1m_input": 0.08,
        "cost_per_1m_output": 0.08,
        "provider": "qwen",
        "serverless": False,
        "requires_endpoint": True,
    },
    "lfm2.5-1.2b": {
        "model": "azure_ml/lfm2-5-1-2b-thinking",
        "description": "LFM2.5 1.2B - Ultra-small edge model",
        "context_window": 32768,
        "cost_per_1m_input": 0.01,
        "cost_per_1m_output": 0.02,
        "provider": "liquid",
        "serverless": False,
        "requires_endpoint": True,
    },
}

# Default model
DEFAULT_MODEL = "phi-4"


def get_llm_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get LLM configuration for a given model name.

    Args:
        model_name: Name of the model (e.g., 'phi-4')

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If model is not found or required credentials are missing
    """
    model_name = model_name or os.getenv("DEFAULT_MODEL", DEFAULT_MODEL)

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    config = AVAILABLE_MODELS[model_name].copy()

    # Add Azure configuration based on model type
    if config["provider"] == "openai":
        config["azure_api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        config["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not config["azure_api_key"] or not config["azure_endpoint"]:
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables required. "
                "Set them to use Azure OpenAI models."
            )
    elif config.get("requires_endpoint"):
        # Azure ML endpoint
        endpoint_env = f"AZURE_ML_ENDPOINT_{model_name.upper().replace('-', '_').replace('.', '_')}"
        config["endpoint_url"] = os.getenv(endpoint_env) or os.getenv("AZURE_ML_ENDPOINT")
        
        if not config["endpoint_url"]:
            raise ValueError(
                f"{endpoint_env} or AZURE_ML_ENDPOINT environment variable required. "
                f"Deploy the model to Azure ML first, then set the endpoint URL."
            )
    else:
        # Azure AI Foundry serverless
        config["azure_api_key"] = os.getenv("AZURE_AI_API_KEY")
        
        if not config["azure_api_key"]:
            raise ValueError(
                "AZURE_AI_API_KEY environment variable required. "
                "Get it from Azure AI Foundry console."
            )

    return config


def get_llm(
    model_name: Optional[str] = None, 
    temperature: float = 0.7, 
    max_tokens: int = 4096
) -> LLM:
    """
    Create a CrewAI LLM instance configured for Azure AI.

    Supports Azure OpenAI, Azure AI Foundry serverless, and Azure ML endpoints.

    Args:
        model_name: Name of the model
        temperature: Sampling temperature
        max_tokens: Maximum output tokens

    Returns:
        Configured LLM instance
    """
    config = get_llm_config(model_name)

    # LLM configuration - LiteLLM handles routing based on model string
    llm_kwargs = {
        "model": config["model"],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Add provider-specific configuration
    if config["provider"] == "openai":
        llm_kwargs["api_key"] = config["azure_api_key"]
        llm_kwargs["api_base"] = config["azure_endpoint"]
        llm_kwargs["api_version"] = "2024-08-01-preview"
    elif config.get("requires_endpoint"):
        llm_kwargs["api_base"] = config["endpoint_url"]
        llm_kwargs["api_key"] = os.getenv("AZURE_ML_API_KEY", "dummy")
    else:
        llm_kwargs["api_key"] = config["azure_api_key"]

    return LLM(**llm_kwargs)


def estimate_cost(
    input_tokens: int, 
    output_tokens: int, 
    model_name: Optional[str] = None
) -> float:
    """
    Estimate the cost for a given number of tokens.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_name: Model to estimate for

    Returns:
        Estimated cost in USD
    """
    config = get_llm_config(model_name)

    input_cost = (input_tokens / 1_000_000) * config["cost_per_1m_input"]
    output_cost = (output_tokens / 1_000_000) * config["cost_per_1m_output"]

    return input_cost + output_cost


def list_models(
    provider: Optional[str] = None, 
    serverless_only: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    List available models, optionally filtered.

    Args:
        provider: Filter by provider (openai, microsoft, meta, mistral, etc.)
        serverless_only: If True, only return serverless models

    Returns:
        Dictionary of model configurations
    """
    models = AVAILABLE_MODELS

    if provider:
        models = {
            name: config
            for name, config in models.items()
            if config.get("provider") == provider
        }

    if serverless_only:
        models = {
            name: config
            for name, config in models.items()
            if config.get("serverless", False)
        }

    return models


def print_model_info():
    """Print a formatted table of available models."""
    print("\n" + "=" * 100)
    print("Available Models on Azure AI")
    print("=" * 100)

    # Group by deployment type
    serverless = list_models(serverless_only=True)
    custom = {k: v for k, v in AVAILABLE_MODELS.items() if k not in serverless}

    print(f"\n{'─' * 100}")
    print("  SERVERLESS MODELS (Pay-per-token, no infrastructure)")
    print(f"{'─' * 100}")
    for name, config in serverless.items():
        print(f"\n  {name}")
        print(f"    {config['description']}")
        print(f"    Context: {config['context_window']:,} tokens")
        print(
            f"    Cost: ${config['cost_per_1m_input']:.2f}/1M in, "
            f"${config['cost_per_1m_output']:.2f}/1M out"
        )

    if custom:
        print(f"\n{'─' * 100}")
        print("  AZURE ML ENDPOINTS (Requires deployment)")
        print(f"{'─' * 100}")
        for name, config in custom.items():
            print(f"\n  {name}")
            print(f"    {config['description']}")
            print(f"    Context: {config['context_window']:,} tokens")
            print(
                f"    Est. Cost: ${config['cost_per_1m_input']:.2f}/1M in, "
                f"${config['cost_per_1m_output']:.2f}/1M out (+ endpoint hosting)"
            )

    print("\n" + "=" * 100)
    print("\nTo use serverless models: Just set AZURE_OPENAI_API_KEY or AZURE_AI_API_KEY")
    print("To use Azure ML models: Deploy to endpoint first, then set AZURE_ML_ENDPOINT_<MODEL>")
    print("=" * 100 + "\n")
