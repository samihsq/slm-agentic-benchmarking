"""BIG-bench Lite benchmark (tasksource/bigbench on HuggingFace)."""

from .runner import BigBenchRunner, DEFAULT_TASK_CONFIGS
from .task_sets import (
    BIGBENCH_LITE_24,
    BIGBENCH_CORE_6,
    BIGBENCH_LITE_BY_NAME,
    BBL24_TASK_NAMES,
)

__all__ = [
    "BigBenchRunner",
    "DEFAULT_TASK_CONFIGS",
    "BIGBENCH_LITE_24",
    "BIGBENCH_CORE_6",
    "BIGBENCH_LITE_BY_NAME",
    "BBL24_TASK_NAMES",
]
