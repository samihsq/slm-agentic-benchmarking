"""
Sequence scoring for Criticality v2 — Single-Token MCQ Mode.

Presents the full MCQ prompt (all 4 arguments labeled A-D) to the model,
then extracts logprobs for the label tokens (A, B, C, D) at the answer
position. This lets the model see all options in context and make a
comparative judgment, rather than scoring each option independently.

Falls back to per-option continuation scoring if label tokens can't be
resolved (e.g., unusual tokenizer).

Uses llama-cpp-python to load GGUF models and extract token-level logprobs.
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

from scipy.special import log_softmax

from .logprob_utils import ChoiceLogprobs


CHOICE_LABELS = ["A", "B", "C", "D"]


class SequenceScorer:
    """
    Score argument options via single-token MCQ logprob extraction.

    Builds a full MCQ prompt with all options visible, evaluates it, then
    reads the logprob for each label token (A/B/C/D) at the final position.
    The model sees all arguments in context and can compare them.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for sequence scoring. "
                "Install with: pip install llama-cpp-python"
            )

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.verbose = verbose

        if verbose:
            print(f"Loading GGUF model: {model_path}")

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,
            verbose=verbose,
        )

        # Pre-resolve label token IDs for the choice letters
        self._label_token_ids: Dict[str, int] = {}
        for label in CHOICE_LABELS:
            # Try multiple token representations (with/without space prefix)
            for variant in [label, f" {label}", label.lower(), f" {label.lower()}"]:
                tokens = self.model.tokenize(
                    variant.encode("utf-8"),
                    add_bos=False,
                    special=False,
                )
                if len(tokens) == 1:
                    self._label_token_ids[label] = tokens[0]
                    break

        if verbose:
            resolved = list(self._label_token_ids.keys())
            print(f"Model loaded. Context: {n_ctx}, GPU layers: {n_gpu_layers}")
            print(f"Resolved label tokens: {resolved} ({len(resolved)}/4)")

    def score_options(
        self,
        prefix: str,
        options: List[str],
        labels: Optional[List[str]] = None,
        task: Optional[Dict] = None,
    ) -> ChoiceLogprobs:
        """
        Score options using single-token MCQ logprob extraction.

        Builds a full MCQ prompt, evaluates it, and reads logprobs for
        the label tokens at the final position.

        Args:
            prefix: Ignored for MCQ mode (kept for API compat). The prompt
                is built from the task dict if provided.
            options: List of argument texts.
            labels: Choice labels (default: A, B, C, D).
            task: Full task dict (used to build the MCQ prompt).

        Returns:
            ChoiceLogprobs with per-label logprobs.
        """
        if labels is None:
            labels = CHOICE_LABELS[:len(options)]

        if len(labels) != len(options):
            raise ValueError(
                f"Labels ({len(labels)}) must match options ({len(options)})"
            )

        # If we have resolved label tokens and enough of them, use MCQ mode
        resolved_labels = [l for l in labels if l in self._label_token_ids]
        if len(resolved_labels) >= 3:
            return self._score_mcq(options, labels, task)
        else:
            # Fallback: per-option continuation scoring (legacy)
            return self._score_continuation(prefix, options, labels)

    def _score_mcq(
        self,
        options: List[str],
        labels: List[str],
        task: Optional[Dict] = None,
    ) -> ChoiceLogprobs:
        """
        Single-token MCQ scoring: present all options, read label logprobs.
        """
        # Build the MCQ prompt
        if task:
            topic = task.get("topic", "")
        else:
            topic = ""

        options_block = "\n".join(
            f"{label}) {text}" for label, text in zip(labels, options)
        )

        prompt = (
            f'Topic: "{topic}"\n\n'
            f"Which of the following arguments is the strongest?\n\n"
            f"{options_block}\n\n"
            f"Answer:"
        )

        # Tokenize and evaluate
        tokens = self.model.tokenize(
            prompt.encode("utf-8"),
            add_bos=True,
            special=False,
        )

        # Truncate if needed
        max_tokens = self.model.n_ctx() - 1
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        self.model.reset()
        self.model.eval(tokens)

        # Extract logprobs at the last position (predicting the next token)
        last_pos = len(tokens) - 1
        if last_pos >= self.model.n_tokens:
            last_pos = self.model.n_tokens - 1

        logits = self.model.scores[last_pos, :]
        all_logprobs = log_softmax(logits)

        # Read logprobs for each label token
        option_scores = {}
        for label in labels:
            if label in self._label_token_ids:
                token_id = self._label_token_ids[label]
                option_scores[label] = float(all_logprobs[token_id])
            else:
                # Missing label — assign very low score
                option_scores[label] = -100.0

        return ChoiceLogprobs(
            logprobs=option_scores,
            position=last_pos,
            source="sequence_mcq",
        )

    def _score_continuation(
        self,
        prefix: str,
        options: List[str],
        labels: List[str],
    ) -> ChoiceLogprobs:
        """
        Legacy fallback: score each option independently by continuation logprob.
        """
        prefix_tokens = self.model.tokenize(
            prefix.encode("utf-8"),
            add_bos=True,
            special=False,
        )
        prefix_len = len(prefix_tokens)

        option_scores = {}

        for label, option_text in zip(labels, options):
            full_text = prefix + option_text
            full_tokens = self.model.tokenize(
                full_text.encode("utf-8"),
                add_bos=True,
                special=False,
            )

            self.model.reset()
            self.model.eval(full_tokens)

            option_logprobs = []
            for i in range(prefix_len, len(full_tokens)):
                token_id = full_tokens[i]
                if i - 1 < self.model.n_tokens:
                    prev_logits = self.model.scores[i - 1, :]
                    token_logprobs = log_softmax(prev_logits)
                    log_prob = float(token_logprobs[token_id])
                    option_logprobs.append(log_prob)

            if option_logprobs:
                avg_logprob = np.mean(option_logprobs)
            else:
                avg_logprob = -100.0

            option_scores[label] = avg_logprob

        return ChoiceLogprobs(
            logprobs=option_scores,
            position=-1,
            source="sequence_continuation",
        )

    def close(self):
        """Release model memory."""
        if hasattr(self, 'model'):
            del self.model
            if self.verbose:
                print("Model unloaded")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
