"""
MCQ Task Generator for Criticality v2.

Constructs multiple-choice argument quality tasks from the IBM Argument
Quality Ranking 30k dataset. Each task presents 4 arguments of known quality
tiers and asks the model to select the strongest.

Quality tiers:
  - strong: WA/MACE-P > 0.7
  - medium: 0.4 - 0.7
  - weak:   < 0.4
"""

import random
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from datasets import load_dataset


# ── Quality tier boundaries ────────────────────────────────────────────────
TIER_STRONG = 0.7
TIER_MEDIUM = 0.4

CHOICE_LABELS = ["A", "B", "C", "D"]


def _quality_tier(score: float) -> str:
    """Map a quality score to a tier label."""
    if score >= TIER_STRONG:
        return "strong"
    elif score >= TIER_MEDIUM:
        return "medium"
    return "weak"


class MCQTaskGenerator:
    """
    Generate MCQ argument-quality tasks from IBM Argument Quality 30k.

    Each task has 4 options drawn from different quality tiers with a known
    ground-truth quality ranking so logprob distributions can be compared to
    the true ordering.
    """

    def __init__(
        self,
        quality_score: str = "WA",
        seed: int = 42,
        verbose: bool = False,
    ):
        """
        Args:
            quality_score: Which column to use for quality – "WA" or "MACE-P".
            seed: RNG seed for reproducibility.
            verbose: Print progress info.
        """
        self.quality_score = quality_score
        self.verbose = verbose
        self._rng = random.Random(seed)
        self._arguments_by_topic: Dict[str, Dict[str, List[Dict]]] = {}

    # ── Data loading ───────────────────────────────────────────────────────

    def load_arguments(self, limit: Optional[int] = None) -> int:
        """
        Load arguments from HuggingFace and group by topic → tier.

        Returns:
            Total number of arguments loaded.
        """
        if self.verbose:
            print("Loading IBM Argument Quality 30k dataset...")

        ds = load_dataset(
            "ibm-research/argument_quality_ranking_30k",
            "argument_quality_ranking",
            split="train",
            streaming=True,
        )

        self._arguments_by_topic = {}
        loaded = 0

        for item in ds:
            if limit and loaded >= limit:
                break

            topic = item.get("topic", "unknown")
            argument_text = item.get("argument", "")
            quality = float(item.get(self.quality_score, 0.0))

            if not argument_text or quality <= 0:
                continue

            tier = _quality_tier(quality)

            if topic not in self._arguments_by_topic:
                self._arguments_by_topic[topic] = defaultdict(list)

            self._arguments_by_topic[topic][tier].append(
                {
                    "argument": argument_text,
                    "quality": quality,
                    "tier": tier,
                    "topic": topic,
                    "stance": item.get("stance_WA", 0),
                }
            )
            loaded += 1

        if self.verbose:
            n_topics = len(self._arguments_by_topic)
            tier_counts = defaultdict(int)
            for tiers in self._arguments_by_topic.values():
                for tier, args in tiers.items():
                    tier_counts[tier] += len(args)
            print(f"Loaded {loaded} arguments across {n_topics} topics")
            print(f"  strong: {tier_counts['strong']}, medium: {tier_counts['medium']}, weak: {tier_counts['weak']}")

        return loaded

    # ── Task generation ────────────────────────────────────────────────────

    def _pick_one(self, pool: List[Dict]) -> Optional[Dict]:
        """Pick a random argument from a pool (without removing it)."""
        if not pool:
            return None
        return self._rng.choice(pool)

    def _build_mcq(
        self,
        topic: str,
        tiers: Dict[str, List[Dict]],
        task_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a single 4-option MCQ task.

        Strategy:
          1. Sample one strong, one medium, one weak argument.
          2. Add a 4th from whichever tier has the most remaining.
          3. Shuffle the options and record the ground-truth ranking.
        """
        strong = self._pick_one(tiers.get("strong", []))
        medium = self._pick_one(tiers.get("medium", []))
        weak = self._pick_one(tiers.get("weak", []))

        # Need at least 3 distinct tiers for a meaningful task
        available = [a for a in [strong, medium, weak] if a is not None]
        if len(available) < 2:
            return None

        # Fill to 4 options
        while len(available) < 4:
            # Pick from the tier with the most arguments
            best_tier = max(tiers.keys(), key=lambda t: len(tiers[t]))
            extra = self._pick_one(tiers[best_tier])
            if extra and extra not in available:
                available.append(extra)
            else:
                break  # Can't get to 4 unique options

        if len(available) < 3:
            return None

        # Shuffle for presentation
        self._rng.shuffle(available)

        # Build the ground-truth quality ranking (1 = best)
        sorted_by_quality = sorted(available, key=lambda a: a["quality"], reverse=True)
        quality_rank = {id(a): rank + 1 for rank, a in enumerate(sorted_by_quality)}

        # The correct answer is the option with rank 1 (highest quality)
        best_arg = sorted_by_quality[0]
        correct_label = CHOICE_LABELS[available.index(best_arg)]

        options = []
        for i, arg in enumerate(available):
            label = CHOICE_LABELS[i]
            options.append(
                {
                    "label": label,
                    "argument": arg["argument"],
                    "quality": arg["quality"],
                    "tier": arg["tier"],
                    "rank": quality_rank[id(arg)],
                }
            )

        # Ground-truth ranking: label → rank
        ground_truth_ranking = {opt["label"]: opt["rank"] for opt in options}
        # Quality scores for correlation: label → quality
        ground_truth_qualities = {opt["label"]: opt["quality"] for opt in options}

        return {
            "task_id": f"crit_v2_{task_idx:04d}",
            "topic": topic,
            "options": options,
            "correct_label": correct_label,
            "ground_truth_ranking": ground_truth_ranking,
            "ground_truth_qualities": ground_truth_qualities,
            "num_options": len(options),
        }

    def generate_tasks(self, num_tasks: int = 200) -> List[Dict[str, Any]]:
        """
        Generate MCQ tasks across all loaded topics.

        Args:
            num_tasks: Target number of tasks.

        Returns:
            List of MCQ task dictionaries.
        """
        if not self._arguments_by_topic:
            raise RuntimeError("Call load_arguments() before generate_tasks().")

        tasks: List[Dict[str, Any]] = []
        topics = list(self._arguments_by_topic.keys())
        self._rng.shuffle(topics)

        task_idx = 0
        # Round-robin through topics until we have enough tasks
        round_num = 0
        while len(tasks) < num_tasks and round_num < 50:
            for topic in topics:
                if len(tasks) >= num_tasks:
                    break
                tiers = self._arguments_by_topic[topic]
                task = self._build_mcq(topic, tiers, task_idx)
                if task:
                    tasks.append(task)
                    task_idx += 1
            round_num += 1

        if self.verbose:
            print(f"Generated {len(tasks)} MCQ tasks from {len(set(t['topic'] for t in tasks))} topics")

        return tasks

    # ── Prompt formatting ──────────────────────────────────────────────────

    @staticmethod
    def format_prompt(task: Dict[str, Any], system_constrained: bool = True) -> str:
        """
        Format an MCQ task into a prompt string.

        Args:
            task: MCQ task dict from generate_tasks().
            system_constrained: If True, use strict "output ONLY a single letter" framing.

        Returns:
            Formatted prompt string.
        """
        options_block = "\n".join(
            f"{opt['label']}) {opt['argument']}" for opt in task["options"]
        )

        if system_constrained:
            prompt = (
                f'Topic: "{task["topic"]}"\n\n'
                f"Which of the following arguments is the strongest?\n\n"
                f"{options_block}\n\n"
                f"Answer:"
            )
        else:
            prompt = (
                f'Topic: "{task["topic"]}"\n\n'
                f"Read the following arguments and identify which one is the strongest "
                f"(most persuasive, best evidenced, most compelling).\n\n"
                f"{options_block}\n\n"
                f"Which argument is the strongest? Respond with the letter of your choice."
            )

        return prompt

    @staticmethod
    def get_system_prompt() -> str:
        """System prompt for MCQ logprob probing."""
        return (
            "You are a multiple choice answering machine. "
            "Read the arguments and output ONLY a single letter (A, B, C, or D) "
            "corresponding to the strongest argument. Do not explain."
        )

    # ── Perturbation (Phase 3) ─────────────────────────────────────────────

    def generate_shuffled_variant(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a shuffled-order variant of a task (for Phase 3 consistency testing).

        The options are reordered but the ground-truth mapping is updated accordingly.
        """
        new_task = dict(task)
        options = list(task["options"])
        self._rng.shuffle(options)

        # Reassign labels
        for i, opt in enumerate(options):
            opt = dict(opt)
            opt["label"] = CHOICE_LABELS[i]
            options[i] = opt

        # Find new correct label
        best_rank = min(opt["rank"] for opt in options)
        correct_label = next(opt["label"] for opt in options if opt["rank"] == best_rank)

        new_task["options"] = options
        new_task["correct_label"] = correct_label
        new_task["ground_truth_ranking"] = {opt["label"]: opt["rank"] for opt in options}
        new_task["ground_truth_qualities"] = {opt["label"]: opt["quality"] for opt in options}
        new_task["task_id"] = task["task_id"] + "_shuffled"

        return new_task

    # ── Refutation pairs (Phase 2) ─────────────────────────────────────────

    def generate_refutation_pairs(
        self, num_pairs: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate claim + candidate refutation sets for Phase 2 (freeform probing).

        Uses opposite-stance arguments on the same topic as refutation candidates.

        Returns:
            List of refutation task dicts.
        """
        pairs: List[Dict[str, Any]] = []

        for topic, tiers in self._arguments_by_topic.items():
            if len(pairs) >= num_pairs:
                break

            # Collect all arguments for this topic
            all_args = []
            for tier_args in tiers.values():
                all_args.extend(tier_args)

            # Group by stance
            stances: Dict[int, List[Dict]] = defaultdict(list)
            for arg in all_args:
                stances[arg.get("stance", 0)].append(arg)

            # Need at least 2 stances
            stance_keys = list(stances.keys())
            if len(stance_keys) < 2:
                continue

            # Use one stance as claim, opposite as refutations
            claim_stance = stance_keys[0]
            refutation_stance = stance_keys[1]

            claim_args = stances[claim_stance]
            refutation_args = stances[refutation_stance]

            if len(refutation_args) < 3:
                continue

            # Pick a claim
            claim = self._rng.choice(claim_args)

            # Pick 4 refutations of varying quality
            self._rng.shuffle(refutation_args)
            candidates = refutation_args[:4]

            # Sort by quality for ground truth ranking
            candidates.sort(key=lambda a: a["quality"], reverse=True)

            pairs.append(
                {
                    "task_id": f"refutation_{len(pairs):04d}",
                    "topic": topic,
                    "claim": claim["argument"],
                    "claim_quality": claim["quality"],
                    "refutations": [
                        {
                            "label": CHOICE_LABELS[i],
                            "argument": c["argument"],
                            "quality": c["quality"],
                            "tier": c["tier"],
                            "rank": i + 1,
                        }
                        for i, c in enumerate(candidates)
                    ],
                    "best_refutation_label": "A",  # Sorted by quality, A is best
                }
            )

        if self.verbose:
            print(f"Generated {len(pairs)} refutation pairs")

        return pairs
