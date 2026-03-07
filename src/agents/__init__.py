"""Agent architectures for benchmarking."""

from .base_agent import BaseAgent, BenchmarkResponse
from .one_shot_agent import OneShotAgent
from .sequential_agent import SequentialAgent
from .concurrent_agent import ConcurrentAgent
from .group_chat_agent import GroupChatAgent
from .baseline_agent import BaselineAgent, get_baseline_agent
from .ollama_agent import OllamaAgent
from .ollama_agentic_agent import OllamaSequentialAgent, OllamaConcurrentAgent, OllamaGroupChatAgent

__all__ = [
    "BaseAgent",
    "BenchmarkResponse",
    "OneShotAgent",
    "SequentialAgent",
    "ConcurrentAgent",
    "GroupChatAgent",
    "BaselineAgent",
    "get_baseline_agent",
    "OllamaAgent",
    "OllamaSequentialAgent",
    "OllamaConcurrentAgent",
    "OllamaGroupChatAgent",
]
