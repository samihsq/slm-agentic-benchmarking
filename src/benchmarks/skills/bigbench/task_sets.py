"""
BIG-bench Lite task manifest and per-task metadata.

The official 24-task BIG-bench Lite (BBL24) list, confirmed against
tasksource/bigbench on HuggingFace and the upstream keywords_to_tasks.md.

Scoring methods:
  "mc"    - multiple-choice letter match (multiple_choice_scores)
  "exact" - case-insensitive exact string match against targets[0]
  "bleu"  - sentence-level BLEU against targets[0]

Undersized tasks have fewer than 20 examples in the train split alone;
load_tasks() combines train + validation splits for them.
"""

from typing import List, Dict, Any

# Per-task metadata: name, scoring method, and whether the task is undersized.
# Undersized tasks will have train+validation combined; all available examples used.
BIGBENCH_LITE_24: List[Dict[str, Any]] = [
    {"name": "auto_debugging",                      "scoring": "exact", "undersized": True},
    {"name": "bbq_lite_json",                       "scoring": "mc",    "undersized": False},
    {"name": "code_line_description",               "scoring": "mc",    "undersized": False},
    {"name": "conceptual_combinations",             "scoring": "mc",    "undersized": False},
    {"name": "conlang_translation",                 "scoring": "bleu",  "undersized": False},
    {"name": "emoji_movie",                         "scoring": "mc",    "undersized": False},
    {"name": "formal_fallacies_syllogisms_negation","scoring": "mc",    "undersized": False},
    {"name": "hindu_knowledge",                     "scoring": "mc",    "undersized": False},
    {"name": "known_unknowns",                      "scoring": "mc",    "undersized": False},
    {"name": "language_identification",             "scoring": "mc",    "undersized": False},
    {"name": "linguistics_puzzles",                 "scoring": "exact", "undersized": False},
    {"name": "logic_grid_puzzle",                   "scoring": "mc",    "undersized": False},
    {"name": "logical_deduction",                   "scoring": "mc",    "undersized": False},
    {"name": "misconceptions_russian",              "scoring": "mc",    "undersized": False},
    {"name": "novel_concepts",                      "scoring": "mc",    "undersized": True},
    {"name": "operators",                           "scoring": "exact", "undersized": False},
    {"name": "parsinlu_reading_comprehension",      "scoring": "exact", "undersized": False},
    {"name": "play_dialog_same_or_different",       "scoring": "mc",    "undersized": False},
    {"name": "repeat_copy_logic",                   "scoring": "exact", "undersized": True},
    {"name": "strange_stories",                     "scoring": "mc",    "undersized": False},
    {"name": "strategyqa",                          "scoring": "mc",    "undersized": False},
    {"name": "symbol_interpretation",               "scoring": "mc",    "undersized": False},
    {"name": "vitaminc_fact_verification",          "scoring": "mc",    "undersized": False},
    {"name": "winowhy",                             "scoring": "mc",    "undersized": False},
]

# Legacy 6-task set retained for backwards compatibility.
BIGBENCH_CORE_6 = [
    "logical_deduction",
    "navigate",
    "penguins_in_a_table",
    "causal_judgment",
    "date_understanding",
    "arithmetic",
]

# Convenient name-keyed lookup
BIGBENCH_LITE_BY_NAME: Dict[str, Dict[str, Any]] = {
    t["name"]: t for t in BIGBENCH_LITE_24
}

BBL24_TASK_NAMES: List[str] = [t["name"] for t in BIGBENCH_LITE_24]
