"""
Sequence likelihood scoring for Criticality v2.

Instead of asking models to output A/B/C/D (which requires instruction-following),
we directly compute P(argument_text | prompt_context) for each candidate argument.
This works on ANY model, including tiny ones that can't follow MCQ formatting.

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
    Score argument options by computing conditional sequence likelihood.
    
    For each option, we measure the average log-probability of the argument's
    tokens given the shared prefix. Higher average logprob = model finds that
    argument more probable/natural in context.
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        """
        Load a GGUF model for sequence scoring.
        
        Args:
            model_path: Path to GGUF file (e.g., Ollama blob).
            n_ctx: Maximum context window size.
            n_gpu_layers: Number of layers to offload to GPU (-1 = all, 0 = CPU only).
            verbose: Print model info on load.
        """
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
            logits_all=True,  # CRITICAL: enables logits for all tokens
            verbose=verbose,
        )
        
        if verbose:
            print(f"Model loaded. Context: {n_ctx}, GPU layers: {n_gpu_layers}")
    
    def score_options(
        self,
        prefix: str,
        options: List[str],
        labels: Optional[List[str]] = None,
    ) -> ChoiceLogprobs:
        """
        Score a list of text options by their conditional likelihood given a prefix.
        
        Args:
            prefix: Shared context (e.g., "Topic: X. The strongest argument is: ")
            options: List of candidate texts to score (e.g., 4 arguments)
            labels: Optional list of choice labels (default: ["A", "B", "C", "D"])
        
        Returns:
            ChoiceLogprobs with logprobs and probabilities for each option.
        """
        if labels is None:
            labels = CHOICE_LABELS[:len(options)]
        
        if len(labels) != len(options):
            raise ValueError(f"Number of labels ({len(labels)}) must match options ({len(options)})")
        
        # Tokenize prefix once
        prefix_tokens = self.model.tokenize(
            prefix.encode("utf-8"),
            add_bos=True,
            special=False,
        )
        prefix_len = len(prefix_tokens)
        
        # Score each option
        option_scores = {}
        
        for label, option_text in zip(labels, options):
            # Tokenize full sequence: prefix + option
            full_text = prefix + option_text
            full_tokens = self.model.tokenize(
                full_text.encode("utf-8"),
                add_bos=True,
                special=False,
            )
            
            # Eval the full sequence to get logits
            self.model.reset()
            self.model.eval(full_tokens)
            
            # Extract logprobs for option tokens only (skip prefix)
            option_logprobs = []
            for i in range(prefix_len, len(full_tokens)):
                token_id = full_tokens[i]
                
                # Logits at position i-1 predict token at position i
                if i - 1 < self.model.n_tokens:
                    prev_logits = self.model.scores[i - 1, :]
                    token_logprobs = log_softmax(prev_logits)
                    log_prob = float(token_logprobs[token_id])
                    option_logprobs.append(log_prob)
            
            # Average logprob (normalize by token count to avoid length bias)
            if option_logprobs:
                avg_logprob = np.mean(option_logprobs)
            else:
                # Fallback: option was empty or only 1 token
                avg_logprob = -100.0  # Very low score
            
            option_scores[label] = avg_logprob
        
        # Return as ChoiceLogprobs
        return ChoiceLogprobs(
            logprobs=option_scores,
            position=-1,  # Not applicable for sequence scoring
            source="sequence",
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
