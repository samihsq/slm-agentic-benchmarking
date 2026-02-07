"""Benchmark integration modules."""

# Active skill benchmarks
from .skills.criticality.v1.runner import CriticalityRunner
from .skills.criticality.v2.runner import CriticalityV2Runner
from .skills.recall.runner import RecallRunner
from .skills.episodic_memory.runner import EpisodicMemoryRunner

# Archived benchmarks (still importable for backward compatibility)
from .archive.medical.medagent_bench import MedAgentBenchRunner
from .archive.medical.medqa_runner import MedQARunner
from .archive.tool_calling.mcp_bench import MCPBenchRunner
from .archive.tool_calling.bfcl_runner import BFCLRunner

__all__ = [
    # Active skills
    "CriticalityRunner",
    "CriticalityV2Runner",
    "RecallRunner",
    "EpisodicMemoryRunner",
    # Archived
    "MedAgentBenchRunner",
    "MedQARunner",
    "MCPBenchRunner",
    "BFCLRunner",
]
