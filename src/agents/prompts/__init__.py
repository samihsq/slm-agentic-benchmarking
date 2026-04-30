"""
Prompt management system for benchmark-specific agent prompts.

Usage:
    from src.agents.prompts import get_prompt

    prompt = get_prompt("bigbench", "oneshot")
    prompt = get_prompt("general", "sequential_responder")
"""

from .loader import get_prompt, get_agent_prompts, list_available_prompts

__all__ = ["get_prompt", "get_agent_prompts", "list_available_prompts"]
