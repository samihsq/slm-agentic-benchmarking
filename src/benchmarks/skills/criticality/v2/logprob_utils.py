"""
Logprob extraction and calibration utilities for Criticality v2.

Handles:
  - Extracting choice-token logprobs from OpenAI-compatible API responses
  - Scanning thinking-model token sequences for the answer pattern (DASD-4B)
  - Converting logprobs to probability distributions via softmax
  - Computing calibration metrics (rank correlation, calibration error, margin)
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


CHOICE_TOKENS = {"A", "B", "C", "D"}


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class ChoiceLogprobs:
    """Logprob distribution over choice tokens at a single position."""

    logprobs: Dict[str, float]  # e.g. {"A": -0.01, "B": -3.5, "C": -5.2, "D": -7.1}
    probabilities: Dict[str, float] = field(default_factory=dict)
    position: int = -1  # Token position where these were extracted
    source: str = "direct"  # "direct" | "thinking_scan" | "top_logprobs" | "sequence"

    def __post_init__(self):
        if self.logprobs and not self.probabilities:
            self.probabilities = softmax_dict(self.logprobs)

    @property
    def top_choice(self) -> Optional[str]:
        """Return the choice with the highest probability."""
        if not self.probabilities:
            return None
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def margin(self) -> float:
        """Logprob gap between top and second choice."""
        if len(self.logprobs) < 2:
            return 0.0
        sorted_lp = sorted(self.logprobs.values(), reverse=True)
        return sorted_lp[0] - sorted_lp[1]

    @property
    def entropy(self) -> float:
        """Shannon entropy of the probability distribution."""
        if not self.probabilities:
            return 0.0
        return -sum(
            p * math.log2(p) for p in self.probabilities.values() if p > 0
        )


@dataclass
class CalibrationMetrics:
    """Aggregate calibration metrics over a set of MCQ evaluations."""

    rank_correlation: float = 0.0  # Spearman rho: model ranking vs true ranking
    top1_accuracy: float = 0.0  # Fraction where highest-logprob == best argument
    calibration_error: float = 0.0  # Mean |confidence - accuracy|
    avg_margin: float = 0.0  # Average logprob gap between top and 2nd choice
    avg_entropy: float = 0.0  # Average entropy of choice distributions
    num_tasks: int = 0

    # Breakdown by difficulty bucket
    easy_accuracy: float = 0.0  # Tasks where quality spread is large
    hard_accuracy: float = 0.0  # Tasks where quality spread is small

    def to_dict(self) -> Dict:
        return {
            "rank_correlation": round(self.rank_correlation, 4),
            "top1_accuracy": round(self.top1_accuracy, 4),
            "calibration_error": round(self.calibration_error, 4),
            "avg_margin": round(self.avg_margin, 4),
            "avg_entropy": round(self.avg_entropy, 4),
            "num_tasks": self.num_tasks,
            "easy_accuracy": round(self.easy_accuracy, 4),
            "hard_accuracy": round(self.hard_accuracy, 4),
        }


# ── Math helpers ───────────────────────────────────────────────────────────

def softmax_dict(logprobs: Dict[str, float]) -> Dict[str, float]:
    """Convert a dict of logprobs to a probability distribution via softmax."""
    if not logprobs:
        return {}
    max_lp = max(logprobs.values())
    exps = {k: math.exp(v - max_lp) for k, v in logprobs.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def spearman_rank_correlation(
    predicted_ranking: Dict[str, int],
    true_ranking: Dict[str, int],
) -> float:
    """
    Compute Spearman rank correlation between two rankings.

    Args:
        predicted_ranking: label -> rank (1 = best) from model logprobs
        true_ranking: label -> rank (1 = best) from ground truth quality

    Returns:
        Spearman rho in [-1, 1].
    """
    common = sorted(set(predicted_ranking.keys()) & set(true_ranking.keys()))
    if len(common) < 2:
        return 0.0

    pred = [predicted_ranking[k] for k in common]
    true = [true_ranking[k] for k in common]

    rho, _ = scipy_stats.spearmanr(pred, true)
    return float(rho) if not math.isnan(rho) else 0.0


# ── Logprob extraction ─────────────────────────────────────────────────────

class LogprobExtractor:
    """
    Extract choice-token logprobs from OpenAI-compatible chat completion responses.

    Handles two modes:
      1. **Direct mode**: Model outputs a single token (A/B/C/D) — read logprobs
         at position 0 of the content.
      2. **Thinking-model mode** (DASD-4B via Ollama): The content field is empty
         but logprobs span thinking + answer tokens. Scan the full sequence for
         the answer pattern ("answer:" or a bare choice token near the end).
    """

    def __init__(
        self,
        choice_tokens: Optional[set] = None,
        thinking_model: bool = False,
    ):
        """
        Args:
            choice_tokens: Set of valid choice tokens (default: {A, B, C, D}).
            thinking_model: If True, scan full token sequence for answer pattern.
        """
        self.choice_tokens = choice_tokens or CHOICE_TOKENS
        self.thinking_model = thinking_model

    def extract_from_response(self, response) -> Optional[ChoiceLogprobs]:
        """
        Extract choice logprobs from an OpenAI ChatCompletion response object.

        Args:
            response: OpenAI ChatCompletion response (with logprobs=True).

        Returns:
            ChoiceLogprobs or None if extraction fails.
        """
        try:
            choice = response.choices[0]
            logprobs_data = choice.logprobs

            if logprobs_data is None:
                return None

            content_logprobs = logprobs_data.content
            if not content_logprobs:
                return None

            if self.thinking_model:
                return self._extract_thinking_model(content_logprobs)
            else:
                return self._extract_direct(content_logprobs)

        except (AttributeError, IndexError, TypeError):
            return None

    def _extract_direct(self, content_logprobs: list) -> Optional[ChoiceLogprobs]:
        """
        Direct extraction: look at the first token and its top_logprobs
        for choice tokens.
        """
        if not content_logprobs:
            return None

        token_info = content_logprobs[0]
        choice_lps = {}

        # Check the main token
        main_token = token_info.token.strip().upper()
        if main_token in self.choice_tokens:
            choice_lps[main_token] = token_info.logprob

        # Check top_logprobs for alternative choices
        if hasattr(token_info, "top_logprobs") and token_info.top_logprobs:
            for alt in token_info.top_logprobs:
                alt_token = alt.token.strip().upper()
                if alt_token in self.choice_tokens and alt_token not in choice_lps:
                    choice_lps[alt_token] = alt.logprob

        if not choice_lps:
            return None

        return ChoiceLogprobs(
            logprobs=choice_lps,
            position=0,
            source="direct",
        )

    def _extract_thinking_model(self, content_logprobs: list) -> Optional[ChoiceLogprobs]:
        """
        Thinking-model extraction: scan the full token sequence for the
        answer pattern. DASD-4B outputs thinking tokens followed by the answer,
        and the "answer:" position gives the cleanest signal.
        """
        best_choice_lps: Optional[Dict[str, float]] = None
        best_position: int = -1
        best_score: float = -float("inf")

        for pos, token_info in enumerate(content_logprobs):
            token_text = token_info.token.strip().lower()

            # Look for "answer:" or similar patterns that precede the choice
            is_answer_position = token_text in ("answer", "answer:", ":")
            is_choice_token = token_text.upper() in self.choice_tokens

            if not is_answer_position and not is_choice_token:
                continue

            # If this is an "answer:" token, look at the NEXT token for choices
            if is_answer_position and pos + 1 < len(content_logprobs):
                next_info = content_logprobs[pos + 1]
                choice_lps = self._collect_choices(next_info)
                if choice_lps:
                    # Score: prefer positions later in the sequence (closer to final answer)
                    score = pos
                    if score > best_score:
                        best_choice_lps = choice_lps
                        best_position = pos + 1
                        best_score = score

            # If this IS a choice token, collect from its own top_logprobs
            if is_choice_token:
                choice_lps = self._collect_choices(token_info)
                if choice_lps:
                    score = pos
                    if score > best_score:
                        best_choice_lps = choice_lps
                        best_position = pos
                        best_score = score

        if best_choice_lps is None:
            # Fallback: check the very last token
            if content_logprobs:
                last_info = content_logprobs[-1]
                choice_lps = self._collect_choices(last_info)
                if choice_lps:
                    best_choice_lps = choice_lps
                    best_position = len(content_logprobs) - 1

        if best_choice_lps is None:
            return None

        return ChoiceLogprobs(
            logprobs=best_choice_lps,
            position=best_position,
            source="thinking_scan",
        )

    def _collect_choices(self, token_info) -> Optional[Dict[str, float]]:
        """Collect choice token logprobs from a single token's top_logprobs."""
        choice_lps = {}

        # Main token
        main = token_info.token.strip().upper()
        if main in self.choice_tokens:
            choice_lps[main] = token_info.logprob

        # Alternatives
        if hasattr(token_info, "top_logprobs") and token_info.top_logprobs:
            for alt in token_info.top_logprobs:
                alt_token = alt.token.strip().upper()
                if alt_token in self.choice_tokens and alt_token not in choice_lps:
                    choice_lps[alt_token] = alt.logprob

        return choice_lps if choice_lps else None

    # ── Raw logprob dict extraction ────────────────────────────────────────

    def extract_from_raw(
        self,
        logprobs_content: List[Dict[str, Any]],
    ) -> Optional[ChoiceLogprobs]:
        """
        Extract from raw logprobs dicts (for use when not using the OpenAI SDK
        objects directly, e.g. from Ollama raw JSON).

        Args:
            logprobs_content: List of token logprob dicts, each with keys
                'token', 'logprob', and optionally 'top_logprobs' (list of
                dicts with 'token' and 'logprob').
        """
        if not logprobs_content:
            return None

        if self.thinking_model:
            return self._extract_thinking_raw(logprobs_content)

        # Direct mode: first token
        first = logprobs_content[0]
        choice_lps = {}

        main = first.get("token", "").strip().upper()
        if main in self.choice_tokens:
            choice_lps[main] = first.get("logprob", 0.0)

        for alt in first.get("top_logprobs", []):
            alt_token = alt.get("token", "").strip().upper()
            if alt_token in self.choice_tokens and alt_token not in choice_lps:
                choice_lps[alt_token] = alt.get("logprob", 0.0)

        if not choice_lps:
            return None

        return ChoiceLogprobs(logprobs=choice_lps, position=0, source="direct")

    def _extract_thinking_raw(
        self, logprobs_content: List[Dict]
    ) -> Optional[ChoiceLogprobs]:
        """Thinking-model extraction from raw dict format."""
        best_lps = None
        best_pos = -1
        best_score = -float("inf")

        for pos, tok_info in enumerate(logprobs_content):
            token_text = tok_info.get("token", "").strip().lower()
            is_answer = token_text in ("answer", "answer:", ":")
            is_choice = token_text.upper() in self.choice_tokens

            if not is_answer and not is_choice:
                continue

            if is_answer and pos + 1 < len(logprobs_content):
                nxt = logprobs_content[pos + 1]
                lps = self._collect_choices_raw(nxt)
                if lps and pos > best_score:
                    best_lps, best_pos, best_score = lps, pos + 1, pos

            if is_choice:
                lps = self._collect_choices_raw(tok_info)
                if lps and pos > best_score:
                    best_lps, best_pos, best_score = lps, pos, pos

        # Fallback: last token
        if best_lps is None and logprobs_content:
            lps = self._collect_choices_raw(logprobs_content[-1])
            if lps:
                best_lps = lps
                best_pos = len(logprobs_content) - 1

        if best_lps is None:
            return None

        return ChoiceLogprobs(logprobs=best_lps, position=best_pos, source="thinking_scan")

    @staticmethod
    def _collect_choices_raw(tok_info: Dict) -> Optional[Dict[str, float]]:
        lps = {}
        main = tok_info.get("token", "").strip().upper()
        if main in CHOICE_TOKENS:
            lps[main] = tok_info.get("logprob", 0.0)
        for alt in tok_info.get("top_logprobs", []):
            t = alt.get("token", "").strip().upper()
            if t in CHOICE_TOKENS and t not in lps:
                lps[t] = alt.get("logprob", 0.0)
        return lps if lps else None


# ── Aggregate calibration computation ──────────────────────────────────────

def compute_calibration_metrics(
    results: List[Dict[str, Any]],
) -> CalibrationMetrics:
    """
    Compute aggregate calibration metrics from a list of per-task results.

    Each result dict should have:
        - "choice_logprobs": ChoiceLogprobs instance
        - "ground_truth_ranking": {label: rank}
        - "ground_truth_qualities": {label: quality}
        - "correct_label": str
        - "quality_spread": float (max_quality - min_quality)

    Returns:
        CalibrationMetrics instance.
    """
    if not results:
        return CalibrationMetrics()

    rank_correlations = []
    top1_correct = 0
    margins = []
    entropies = []
    confidences = []
    easy_correct = 0
    easy_total = 0
    hard_correct = 0
    hard_total = 0

    for r in results:
        cl: ChoiceLogprobs = r.get("choice_logprobs")
        if cl is None:
            continue

        gt_ranking = r.get("ground_truth_ranking", {})
        correct_label = r.get("correct_label", "")
        quality_spread = r.get("quality_spread", 0.0)

        # Predicted ranking from logprobs (higher logprob = better rank)
        sorted_by_prob = sorted(cl.probabilities.items(), key=lambda x: x[1], reverse=True)
        pred_ranking = {label: rank + 1 for rank, (label, _) in enumerate(sorted_by_prob)}

        # Rank correlation
        rho = spearman_rank_correlation(pred_ranking, gt_ranking)
        rank_correlations.append(rho)

        # Top-1 accuracy
        if cl.top_choice == correct_label:
            top1_correct += 1

        # Margin and entropy
        margins.append(cl.margin)
        entropies.append(cl.entropy)

        # Confidence = probability assigned to correct label
        conf = cl.probabilities.get(correct_label, 0.0)
        confidences.append(conf)

        # Difficulty buckets
        if quality_spread > 0.3:
            easy_total += 1
            if cl.top_choice == correct_label:
                easy_correct += 1
        else:
            hard_total += 1
            if cl.top_choice == correct_label:
                hard_correct += 1

    n = len(rank_correlations)
    if n == 0:
        return CalibrationMetrics()

    # Calibration error: mean |confidence - accuracy|
    accuracy = top1_correct / n
    avg_confidence = sum(confidences) / n if confidences else 0.0
    calibration_error = abs(avg_confidence - accuracy)

    return CalibrationMetrics(
        rank_correlation=sum(rank_correlations) / n,
        top1_accuracy=accuracy,
        calibration_error=calibration_error,
        avg_margin=sum(margins) / n,
        avg_entropy=sum(entropies) / n,
        num_tasks=n,
        easy_accuracy=easy_correct / easy_total if easy_total > 0 else 0.0,
        hard_accuracy=hard_correct / hard_total if hard_total > 0 else 0.0,
    )
