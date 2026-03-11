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
    # SMALL LANGUAGE MODELS (SLMs) - Serverless, Pay-per-token
    # These are confirmed working on your Azure endpoint
    # ==========================================================================
    "phi-4": {
        "model": "openai/Phi-4",
        "description": "Phi-4 14B - Microsoft's flagship SLM, excellent reasoning",
        "size": "14B",
        "context_window": 16384,
        "cost_per_1m_input": 0.07,
        "cost_per_1m_output": 0.14,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "phi-4-mini": {
        "model": "openai/Phi-4-mini-instruct",
        "description": "Phi-4-mini 3.8B - Ultra-efficient, great for cost-sensitive tasks",
        "size": "3.8B",
        "context_window": 128000,
        "cost_per_1m_input": 0.013,
        "cost_per_1m_output": 0.05,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "phi-4-mini-reasoning": {
        "model": "openai/Phi-4-mini-reasoning",
        "description": "Phi-4-mini-reasoning 3.8B - Chain-of-thought optimized",
        "size": "3.8B",
        "context_window": 128000,
        "cost_per_1m_input": 0.013,
        "cost_per_1m_output": 0.05,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "mistral-nemo": {
        "model": "openai/Mistral-Nemo",
        "description": "Mistral Nemo 12B - Efficient multilingual model",
        "size": "12B",
        "context_window": 128000,
        "cost_per_1m_input": 0.10,
        "cost_per_1m_output": 0.10,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "ministral-3b": {
        "model": "openai/Ministral-3B",
        "description": "Ministral 3B - Ultra-compact Mistral model",
        "size": "3B",
        "context_window": 128000,
        "cost_per_1m_input": 0.04,
        "cost_per_1m_output": 0.04,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "mistral-small": {
        "model": "openai/Mistral-Small-2503",
        "description": "Mistral Small 2503 - 24B efficient model",
        "size": "24B",
        "context_window": 32000,
        "cost_per_1m_input": 0.10,
        "cost_per_1m_output": 0.30,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "mistral-large-3": {
        "model": "openai/Mistral-Large-3",
        "description": "Mistral Large 3 - Mistral's flagship 123B model",
        "size": "123B",
        "context_window": 128000,
        "cost_per_1m_input": 2.00,
        "cost_per_1m_output": 6.00,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "llama-3.2-11b-vision": {
        "model": "openai/Llama-3.2-11B-Vision-Instruct",
        "description": "Llama 3.2 11B Vision - Multimodal SLM",
        "size": "11B",
        "context_window": 128000,
        "cost_per_1m_input": 0.037,
        "cost_per_1m_output": 0.037,
        "provider": "azure_foundry",
        "serverless": True,
    },
    # ==========================================================================
    # LARGE MODELS - For baseline comparisons
    # ==========================================================================
    "gpt-4o": {
        "model": "openai/gpt-4o",
        "description": "GPT-4o - OpenAI's flagship multimodal model",
        "size": "~200B",
        "context_window": 128000,
        "cost_per_1m_input": 2.50,
        "cost_per_1m_output": 10.00,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "llama-3.3-70b": {
        "model": "openai/Llama-3.3-70B-Instruct",
        "description": "Llama 3.3 70B - Strong open-weight baseline",
        "size": "70B",
        "context_window": 128000,
        "cost_per_1m_input": 0.27,
        "cost_per_1m_output": 0.27,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "deepseek-r1": {
        "model": "openai/DeepSeek-R1",
        "description": "DeepSeek R1 - Strong reasoning model",
        "size": "671B MoE",
        "context_window": 128000,
        "cost_per_1m_input": 0.55,
        "cost_per_1m_output": 2.19,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "deepseek-v3": {
        "model": "openai/DeepSeek-V3-0324",
        "description": "DeepSeek V3 - 685B MoE, excellent general performance",
        "size": "685B MoE",
        "context_window": 128000,
        "cost_per_1m_input": 0.27,
        "cost_per_1m_output": 1.10,
        "provider": "azure_foundry",
        "serverless": True,
    },
    "deepseek-v3.2": {
        "model": "openai/DeepSeek-V3.2",
        "description": "DeepSeek V3.2 - Latest DeepSeek model",
        "size": "685B MoE",
        "context_window": 128000,
        "cost_per_1m_input": 0.27,
        "cost_per_1m_output": 1.10,
        "provider": "azure_foundry",
        "serverless": True,
    },
}

# Ollama models (local/remote, no API key needed)
OLLAMA_MODELS = {
    "dasd-4b": {
        "model": "hf.co/mradermacher/DASD-4B-Thinking-GGUF:Q4_K_M",
        "description": "DASD 4B Thinking - Qwen3-family reasoning model",
        "size": "4B",
        "context_window": 32768,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
        "provider": "ollama",
        "serverless": False,
    },
    "falcon-h1-90m": {
        "model": "hf.co/tiiuae/Falcon-H1-Tiny-R-90M-GGUF:Q4_K_M",
        "description": "Falcon H1 Tiny 90M - Ultra-small reasoning model",
        "size": "90M",
        "context_window": 8192,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
        "provider": "ollama",
        "serverless": False,
    },
    "qwen3-0.6b": {
        "model": "qwen3:0.6b",
        "description": "Qwen3 0.6B - Sub-billion thinking model",
        "size": "0.6B",
        "context_window": 32768,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
        "provider": "ollama",
        "serverless": False,
    },
    "gemma3-1b": {
        "model": "gemma3:1b",
        "description": "Gemma 3 1B - Google ultra-small model",
        "size": "1B",
        "context_window": 32768,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
        "provider": "ollama",
        "serverless": False,
    },
    "gemma3-4b": {
        "model": "gemma3:4b",
        "description": "Gemma 3 4B - Google small model",
        "size": "4B",
        "context_window": 32768,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
        "provider": "ollama",
        "serverless": False,
    },
    "gemma3n-e2b": {
        "model": "gemma3n:e2b",
        "description": "Gemma 3N E2B - Google edge 2B model",
        "size": "2B",
        "context_window": 32768,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
        "provider": "ollama",
        "serverless": False,
    },
    "gemma3n-e4b": {
        "model": "gemma3n:e4b",
        "description": "Gemma 3N E4B - Google edge 4B model",
        "size": "4B",
        "context_window": 32768,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
        "provider": "ollama",
        "serverless": False,
    },
    "gpt-oss-20b": {
        "model": "gpt-oss:20b",
        "description": "GPT-OSS 20B - Open-source GPT 20B model",
        "size": "20B",
        "context_window": 32768,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
        "provider": "ollama",
        "serverless": False,
    },
    "phi4-mini-reasoning-ollama": {
        "model": "phi4-mini-reasoning:latest",
        "description": "Phi-4 Mini Reasoning (Ollama) - Local reasoning model",
        "size": "3.8B",
        "context_window": 16384,
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
        "provider": "ollama",
        "serverless": False,
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

    # Add Azure AI Foundry configuration
    config["azure_api_key"] = os.getenv("AZURE_API_KEY")
    # Per-model endpoint takes priority (for dedicated managed online endpoints).
    # Falls back to the shared serverless endpoint env var.
    config["azure_endpoint"] = config.get("azure_endpoint") or os.getenv(
        "AZURE_AI_ENDPOINT",
        "https://SLM-Bench-CS199-Winter-26.openai.azure.com/openai/v1"
    )
    
    if not config["azure_api_key"]:
        raise ValueError(
            "AZURE_API_KEY environment variable required. "
            "Get it from Azure AI Foundry console."
        )

    return config


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> LLM:
    """
    Create a CrewAI LLM instance configured for Azure AI.

    Supports Azure OpenAI, Azure AI Foundry serverless, and Azure ML endpoints.

    Args:
        model_name: Name of the model
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        timeout: HTTP timeout in seconds (prevents hung Azure calls)

    Returns:
        Configured LLM instance
    """
    config = get_llm_config(model_name)

    # LLM configuration - LiteLLM handles routing based on model string
    llm_kwargs = {
        "model": config["model"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }

    # Azure AI Foundry (OpenAI-compatible endpoint)
    llm_kwargs["api_key"] = config["azure_api_key"]
    llm_kwargs["api_base"] = config["azure_endpoint"]

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
    print("\nTo use serverless models: Set AZURE_API_KEY")
    print("Default endpoint: https://SLM-Bench-CS199-Winter-26.openai.azure.com/openai/v1")
    print("=" * 100 + "\n")
