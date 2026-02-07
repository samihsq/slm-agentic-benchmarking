"""Criticality (Argument Quality) benchmark - evaluates critical thinking and argument assessment."""

from .v1.runner import CriticalityRunner
from .v2.runner import CriticalityV2Runner

__all__ = [
    "CriticalityRunner",
    "CriticalityV2Runner",
]
