"""
Prompt loader utility for benchmark-specific prompts.

Prompts are organized by benchmark type (e.g. bigbench, general) and
agent role (oneshot, sequential_*, concurrent_*, groupchat_*).
"""

from pathlib import Path
from typing import Dict, Optional
import yaml


# Cache for loaded prompts
_prompt_cache: Dict[str, Dict[str, str]] = {}


def _load_prompts_for_benchmark(benchmark: str) -> Dict[str, str]:
    """Load all prompts for a benchmark type from YAML file."""
    if benchmark in _prompt_cache:
        return _prompt_cache[benchmark]
    
    prompts_dir = Path(__file__).parent / benchmark
    prompts_file = prompts_dir / "prompts.yaml"
    
    if not prompts_file.exists():
        raise FileNotFoundError(f"No prompts found for benchmark: {benchmark}")
    
    with open(prompts_file) as f:
        prompts = yaml.safe_load(f)
    
    _prompt_cache[benchmark] = prompts
    return prompts


def get_prompt(benchmark: str, role: str) -> str:
    """
    Get a prompt for a specific benchmark and agent role.
    
    Args:
        benchmark: The benchmark type (e.g. "bigbench", "general").
        role: The agent role (oneshot, sequential_analyzer, concurrent_synthesizer, etc.).

    Returns:
        The prompt string.

    Examples:
        >>> get_prompt("bigbench", "oneshot")
        >>> get_prompt("general", "sequential_responder")
    """
    prompts = _load_prompts_for_benchmark(benchmark)
    
    if role not in prompts:
        # Try to find a default or fallback
        if "default" in prompts:
            return prompts["default"]
        raise KeyError(f"No prompt found for role '{role}' in benchmark '{benchmark}'")
    
    return prompts[role]


def get_agent_prompts(benchmark: str, agent_type: str) -> Dict[str, str]:
    """
    Get all prompts for a specific agent type in a benchmark.
    
    Args:
        benchmark: The benchmark type
        agent_type: The agent type (oneshot, sequential, concurrent, groupchat)
    
    Returns:
        Dictionary of role -> prompt for that agent type
    """
    prompts = _load_prompts_for_benchmark(benchmark)
    
    # Filter prompts that start with agent_type prefix
    prefix = agent_type.lower() + "_"
    agent_prompts = {}
    
    for role, prompt in prompts.items():
        if role.lower().startswith(prefix) or role.lower() == agent_type.lower():
            # Remove prefix for cleaner access
            clean_role = role[len(prefix):] if role.lower().startswith(prefix) else role
            agent_prompts[clean_role] = prompt
    
    # Also include the base prompt if it exists
    if agent_type.lower() in prompts:
        agent_prompts["system"] = prompts[agent_type.lower()]
    
    return agent_prompts


def list_available_prompts() -> Dict[str, list]:
    """List all available benchmark types and their roles."""
    prompts_dir = Path(__file__).parent
    available = {}
    
    for subdir in prompts_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("_"):
            prompts_file = subdir / "prompts.yaml"
            if prompts_file.exists():
                with open(prompts_file) as f:
                    prompts = yaml.safe_load(f)
                available[subdir.name] = list(prompts.keys())
    
    return available
