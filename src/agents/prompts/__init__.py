"""
Prompt management system for benchmark-specific agent prompts.

Usage:
    from src.agents.prompts import get_prompt
    
    # Get a prompt for a specific benchmark and agent role
    prompt = get_prompt("medical", "oneshot")
    prompt = get_prompt("tool_calling", "sequential_responder")
    prompt = get_prompt("tool_calling", "concurrent_synthesizer")
"""

from .loader import get_prompt, get_agent_prompts, list_available_prompts

__all__ = ["get_prompt", "get_agent_prompts", "list_available_prompts"]
