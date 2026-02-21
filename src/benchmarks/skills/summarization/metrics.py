from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional, Tuple


class MetricUnavailableError(RuntimeError):
    pass


@dataclass
class MetricResult:
    score: float  # normalized to 0..1
    details: Dict[str, Any]


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class SummarizationScorer:
    """
    Computes a single scalar score in [0, 1] for a (prediction, reference) pair.

    Supported metrics:
      - rougeL: ROUGE-L F1 via HF evaluate
      - bertscore: BERTScore F1 via HF evaluate (uses bert-score under the hood)
      - bartscore: exp(-NLL) of reference given source, using a seq2seq LM (BART/T5/etc.)
    """

    def __init__(
        self,
        metric: str = "rougeL",
        *,
        bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
        bertscore_lang: str = "en",
        bartscore_model: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
        rouge_use_stemmer: bool = True,
    ):
        self.metric = metric.lower()
        self.bertscore_model_type = bertscore_model_type
        self.bertscore_lang = bertscore_lang
        self.bartscore_model = bartscore_model
        self.device = device
        self.rouge_use_stemmer = rouge_use_stemmer

        self._lock = Lock()
        self._rouge_metric = None
        self._bertscore_metric = None
        self._bart = None  # (tokenizer, model, device)

    def score_pair(self, *, source: str, prediction: str, reference: str) -> MetricResult:
        prediction = (prediction or "").strip()
        reference = (reference or "").strip()
        source = (source or "").strip()

        if not prediction or not reference:
            return MetricResult(score=0.0, details={"error": "empty prediction/reference"})

        if self.metric in ("rouge", "rougel", "rouge-l", "rouge_l"):
            return self._score_rougeL(prediction=prediction, reference=reference)

        if self.metric in ("bertscore", "bert-score", "bert_score"):
            return self._score_bertscore(prediction=prediction, reference=reference)

        if self.metric in ("bartscore", "bart-score", "bart_score"):
            return self._score_bartscore(source=source, prediction=prediction)

        raise ValueError(f"Unknown summarization metric: {self.metric}")

    def _score_rougeL(self, *, prediction: str, reference: str) -> MetricResult:
        try:
            import evaluate  # type: ignore
        except Exception as e:
            raise MetricUnavailableError(
                "ROUGE scoring requires the 'evaluate' package (and its rouge deps). "
                "Install with: poetry add evaluate rouge-score"
            ) from e

        with self._lock:
            if self._rouge_metric is None:
                self._rouge_metric = evaluate.load("rouge")

            out = self._rouge_metric.compute(
                predictions=[prediction],
                references=[reference],
                use_stemmer=self.rouge_use_stemmer,
            )

        rouge_l = float(out.get("rougeL", 0.0) or 0.0)
        return MetricResult(score=_clamp01(rouge_l), details={"rougeL": rouge_l, **out})

    def _score_bertscore(self, *, prediction: str, reference: str) -> MetricResult:
        try:
            import evaluate  # type: ignore
        except Exception as e:
            raise MetricUnavailableError(
                "BERTScore scoring requires 'evaluate' + 'bert-score' (+ torch/transformers). "
                "Install with: poetry add evaluate bert-score torch transformers"
            ) from e

        with self._lock:
            if self._bertscore_metric is None:
                self._bertscore_metric = evaluate.load("bertscore")

            out = self._bertscore_metric.compute(
                predictions=[prediction],
                references=[reference],
                lang=self.bertscore_lang,
                model_type=self.bertscore_model_type,
                # rescale_with_baseline makes scores more comparable across models/datasets,
                # but requires baseline files; keep off by default.
                rescale_with_baseline=False,
            )

        # HF evaluate returns lists per example
        f1 = out.get("f1", [0.0])[0] if isinstance(out.get("f1"), list) else out.get("f1", 0.0)
        precision = (
            out.get("precision", [0.0])[0]
            if isinstance(out.get("precision"), list)
            else out.get("precision", 0.0)
        )
        recall = (
            out.get("recall", [0.0])[0]
            if isinstance(out.get("recall"), list)
            else out.get("recall", 0.0)
        )

        f1f = float(f1 or 0.0)
        return MetricResult(
            score=_clamp01(f1f),
            details={
                "bertscore_f1": f1f,
                "bertscore_precision": float(precision or 0.0),
                "bertscore_recall": float(recall or 0.0),
                "bertscore_model_type": self.bertscore_model_type,
            },
        )

    def _get_bart(self) -> Tuple[Any, Any, str]:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise MetricUnavailableError(
                "BARTScore requires 'torch' + 'transformers'. "
                "Install with: poetry add torch transformers"
            ) from e

        if self._bart is not None:
            return self._bart

        device = self.device
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tok = AutoTokenizer.from_pretrained(self.bartscore_model, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.bartscore_model)
        model.eval()
        model.to(device)
        self._bart = (tok, model, device)
        return self._bart

    def _score_bartscore(self, *, source: str, prediction: str) -> MetricResult:
        """
        Approximates BARTScore as exp(-NLL) of generating the *prediction* given the source.

        This yields a number in (0, 1] (geometric mean token probability), which fits the repo's
        normalized `EvaluationResult.score` contract.
        """
        # BARTScore is conditional on source -> summary.
        if not source:
            return MetricResult(score=0.0, details={"error": "empty source"})

        with self._lock:
            tok, model, device = self._get_bart()

            import torch  # type: ignore

            # Keep lengths conservative for CPU.
            src = tok(
                source,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            tgt = tok(
                prediction,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )

            input_ids = src["input_ids"].to(device)
            attention_mask = src.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            labels = tgt["input_ids"].to(device)
            # Mask padding for loss
            pad_id = tok.pad_token_id
            if pad_id is not None:
                labels = labels.clone()
                labels[labels == pad_id] = -100

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss  # mean NLL over non-masked tokens

            nll = float(loss.item()) if loss is not None else 1e9
            score = float(torch.exp(torch.tensor(-nll)).item())

        # Numerical guard
        score = 0.0 if score < 0.0 or score != score else score
        score = 1.0 if score > 1.0 else score

        return MetricResult(
            score=score,
            details={
                "bartscore_geomean_prob": score,
                "bartscore_nll": nll,
                "bartscore_model": self.bartscore_model,
            },
        )

