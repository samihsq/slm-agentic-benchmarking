#!/usr/bin/env python3
"""
Derive best model per skill from existing benchmark summary.json files.

Walks results_root (default: results/), collects accuracy/success_rate per
(model, benchmark), maps benchmarks to skills, and returns the model with
highest score per skill. Used to build role_models for skill-routed agents.
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure repo root is on path so src.config imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Benchmark directory or label -> skill name (for aggregation)
BENCHMARK_TO_SKILL: Dict[str, str] = {
    "planning": "planning",
    "plan_bench": "planning",
    "criticality_v2": "criticality",
    "criticality": "criticality",
    "recall": "recall",
    "matrix_recall": "recall",
    "summarization": "summarization",
    "instruction_following": "instruction_following",
    "word_instruction_following": "instruction_following",
}

# Skills we care about for skill-routed agents (must have at least one benchmark)
SKILLS = ["planning", "criticality", "recall", "summarization", "instruction_following"]


def _extract_score(obj: Dict[str, Any]) -> Optional[float]:
    """Extract a single numeric score from a summary dict."""
    if "accuracy" in obj and obj["accuracy"] is not None:
        return float(obj["accuracy"])
    if "success_rate" in obj and obj["success_rate"] is not None:
        return float(obj["success_rate"])
    metrics = obj.get("metrics") or {}
    if "top1_accuracy" in metrics:
        return float(metrics["top1_accuracy"])
    if "mean_composite_score" in obj:
        return float(obj["mean_composite_score"])
    return None


def _extract_model(obj: Dict[str, Any], path: Path) -> Optional[str]:
    """Normalize model name to lowercase key (e.g. gpt-oss-20b)."""
    model = obj.get("model")
    if not model:
        # Infer from path: .../planning/dasd-4b/OllamaAgent/summary.json -> dasd-4b (parent of Agent dir)
        # path = .../model/AgentName/summary.json -> path.parent = AgentName, path.parent.parent = model
        p = path.parent
        if p.name in ("OllamaAgent", "OneShotAgent", "SequentialAgent", "ConcurrentAgent", "GroupChatAgent", "BaselineAgent"):
            model = p.parent.name
        else:
            return None
    raw = str(model).strip()
    return _normalize_model_key(raw) if raw else None


def _benchmark_from_path(path: Path) -> Optional[str]:
    """Infer benchmark name from path (e.g. results/planning/... or results/ollama/ts/planning/model/...)."""
    parts = path.parts
    if "results" not in parts:
        return None
    idx = parts.index("results")
    # results/planning/... or results/criticality_v2/...
    if idx + 1 < len(parts) and parts[idx + 1] in BENCHMARK_TO_SKILL:
        return parts[idx + 1]
    # results/ollama/timestamp/planning/model/Agent/summary.json
    if idx + 1 < len(parts) and parts[idx + 1] == "ollama" and idx + 3 < len(parts):
        candidate = parts[idx + 3]
        if candidate in BENCHMARK_TO_SKILL:
            return candidate
    # results/plan_bench/model_ts/Agent/summary.json
    if "plan_bench" in parts:
        return "plan_bench"
    if "planning" in parts:
        return "planning"
    if "criticality_v2" in parts:
        return "criticality_v2"
    if "criticality" in parts:
        return "criticality"
    if "summarization" in parts:
        return "summarization"
    if "matrix_recall" in parts:
        return "matrix_recall"
    if "recall" in parts:
        return "recall"
    if "word_instruction_following" in parts:
        return "word_instruction_following"
    if "instruction_following" in parts:
        return "instruction_following"
    return None


def _build_model_key_lookup() -> Dict[str, str]:
    """Build reverse map: model-string-lowercase → config_key for all known models."""
    lookup: Dict[str, str] = {}
    try:
        from src.config.azure_llm_config import AVAILABLE_MODELS, OLLAMA_MODELS
        for key, cfg in {**AVAILABLE_MODELS, **OLLAMA_MODELS}.items():
            model_str = cfg.get("model", "")
            if model_str:
                lookup[model_str.lower()] = key
                # also map without vendor prefix (openai/Phi-4 → phi-4 style key)
                bare = model_str.lower().split("/")[-1]
                if bare not in lookup:
                    lookup[bare] = key
        # also map keys directly to themselves
        for key in {**AVAILABLE_MODELS, **OLLAMA_MODELS}:
            lookup[key.lower()] = key
    except ImportError:
        pass
    return lookup


_MODEL_KEY_LOOKUP: Optional[Dict[str, str]] = None


def _normalize_model_key(raw: str) -> str:
    """Map a raw model string (e.g. 'qwen3:0.6b', 'openai/Phi-4') to a config key."""
    global _MODEL_KEY_LOOKUP
    if _MODEL_KEY_LOOKUP is None:
        _MODEL_KEY_LOOKUP = _build_model_key_lookup()
    key = _MODEL_KEY_LOOKUP.get(raw.lower())
    if key:
        return key
    # partial match: strip ':latest', version suffixes and retry
    stripped = raw.lower().split(":")[0].replace("/", "-").replace("_", "-")
    return _MODEL_KEY_LOOKUP.get(stripped, raw.lower())


def collect_scores(results_root: Path) -> Dict[str, Dict[str, float]]:
    """
    Collect (model -> score) per skill from all summary.json under results_root.
    Returns skill -> { model_key: best_score_seen_for_that_model }.
    """
    skill_scores: Dict[str, Dict[str, float]] = {s: {} for s in SKILLS}

    for path in results_root.rglob("summary.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        benchmark = _benchmark_from_path(path)
        if not benchmark or benchmark not in BENCHMARK_TO_SKILL:
            continue
        skill = BENCHMARK_TO_SKILL[benchmark]
        if skill not in skill_scores:
            skill_scores[skill] = {}

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                model = _extract_model(item, path)
                score = _extract_score(item)
                if model and score is not None:
                    skill_scores[skill][model] = max(skill_scores[skill].get(model, 0), score)
        else:
            model = _extract_model(data, path)
            score = _extract_score(data)
            if model and score is not None:
                skill_scores[skill][model] = max(skill_scores[skill].get(model, 0), score)

    return skill_scores


def _azure_model_keys() -> set:
    try:
        from src.config.azure_llm_config import AVAILABLE_MODELS
        return set(AVAILABLE_MODELS.keys())
    except ImportError:
        return set()


def get_best_models_per_skill(
    results_root: Optional[Path] = None,
    default_model: str = "phi-4",
    azure_only: bool = False,
) -> Dict[str, str]:
    """
    Return { skill: model_key } where model_key is the best-performing model
    for that skill according to existing summary.json files.
    """
    root = results_root or Path("results")
    if not root.is_dir():
        return {s: default_model for s in SKILLS}

    skill_scores = collect_scores(root)
    azure_keys = _azure_model_keys() if azure_only else None
    out: Dict[str, str] = {}
    for skill in SKILLS:
        candidates = skill_scores.get(skill) or {}
        if azure_keys is not None:
            candidates = {k: v for k, v in candidates.items() if k in azure_keys}
        if not candidates:
            out[skill] = default_model
            continue
        best_model = max(candidates, key=candidates.get)  # type: ignore[arg-type]
        out[skill] = best_model
    return out


def main():
    parser = argparse.ArgumentParser(description="Print best model per skill from benchmark results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Results root to scan")
    parser.add_argument("--default", default="phi-4", help="Default model when no results for a skill")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    parser.add_argument("--azure-only", action="store_true", help="Only return models that are valid Azure config keys")
    args = parser.parse_args()

    mapping = get_best_models_per_skill(results_root=args.results_dir, default_model=args.default, azure_only=args.azure_only)
    if args.json:
        print(json.dumps(mapping, indent=2))
    else:
        print("Best model per skill (from existing results):")
        for skill, model in mapping.items():
            print(f"  {skill}: {model}")


if __name__ == "__main__":
    main()
