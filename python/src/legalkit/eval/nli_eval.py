"""NLI model evaluation on legal-specific edge cases.

Compares DeBERTa-NLI (off-the-shelf) against LLM-as-judge on a curated
set of hand-labelled legal entailment pairs covering patterns where general
NLI models are known to struggle:

- notwithstanding clauses
- shall vs. may distinctions
- knowledge-qualified representations
- Material Adverse Effect carve-outs
- quantitative precision (up to vs. exactly)
- temporal qualifiers
- conditions precedent
- defined-term substitution

Results include accuracy (overall and per-category), confusion matrix,
latency, and estimated cost — enabling a direct capability/cost tradeoff
analysis between local models and LLM judges.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from legalkit.models.citation import EntailmentClass


# Path to the curated test set
_EDGE_CASES_PATH = Path(__file__).parent.parent.parent.parent / "data/eval/legal_nli_edge_cases.json"

# 3-class grouping used for cross-model comparison
# (DeBERTa only does 3 classes; partially_entailed maps to entailed for this metric)
_BINARY_POSITIVE = {"entailed", "partially_entailed"}
_BINARY_NEGATIVE = {"contradicted", "neutral"}

# Cost estimates (USD per 1K tokens, as of 2025)
_LLM_COST_PER_1K = {
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}
# Average token counts for entailment check
_AVG_INPUT_TOKENS = 350
_AVG_OUTPUT_TOKENS = 60


@dataclass
class NLITestCase:
    """A single legal entailment test case."""

    id: str
    category: str
    passage: str
    claim: str
    label: str          # ground truth: entailed | partially_entailed | contradicted | neutral
    explanation: str = ""

    @property
    def binary_label(self) -> str:
        """Positive (entailed/partially) or negative (contradicted/neutral)."""
        return "positive" if self.label in _BINARY_POSITIVE else "negative"


@dataclass
class NLIJudgment:
    """A single model judgment on a test case."""

    case_id: str
    predicted: str      # model prediction (one of the 4 classes)
    confidence: float   # 0–1 confidence score
    latency_ms: float
    raw_response: str = ""  # raw model output for debugging


@dataclass
class NLIEvalResult:
    """Aggregate results for one model on the full test set."""

    model_name: str
    judgments: list[NLIJudgment]
    test_cases: list[NLITestCase]
    total_latency_ms: float

    # ---------------------------------------------------------------------------
    # Core metrics
    # ---------------------------------------------------------------------------

    @property
    def accuracy_4class(self) -> float:
        """Exact 4-class match accuracy."""
        if not self.judgments:
            return 0.0
        correct = sum(
            1 for j, c in zip(self.judgments, self.test_cases)
            if j.predicted == c.label
        )
        return correct / len(self.judgments)

    @property
    def accuracy_binary(self) -> float:
        """Binary accuracy: (entailed ∪ partially_entailed) vs (contradicted ∪ neutral)."""
        if not self.judgments:
            return 0.0
        correct = sum(
            1 for j, c in zip(self.judgments, self.test_cases)
            if _to_binary(j.predicted) == c.binary_label
        )
        return correct / len(self.judgments)

    @property
    def per_category_accuracy(self) -> dict[str, dict]:
        """Accuracy broken down by edge-case category."""
        from collections import defaultdict
        by_cat: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for j, c in zip(self.judgments, self.test_cases):
            by_cat[c.category].append((j.predicted, c.label))

        result = {}
        for cat, pairs in sorted(by_cat.items()):
            exact = sum(1 for p, gt in pairs if p == gt) / len(pairs)
            binary = sum(
                1 for p, gt in pairs
                if _to_binary(p) == ("positive" if gt in _BINARY_POSITIVE else "negative")
            ) / len(pairs)
            result[cat] = {
                "n": len(pairs),
                "accuracy_4class": round(exact, 3),
                "accuracy_binary": round(binary, 3),
            }
        return result

    @property
    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        """4-class confusion matrix: {true_label: {predicted_label: count}}."""
        labels = ["entailed", "partially_entailed", "contradicted", "neutral"]
        matrix: dict[str, dict[str, int]] = {
            gt: {pred: 0 for pred in labels} for gt in labels
        }
        for j, c in zip(self.judgments, self.test_cases):
            pred = j.predicted if j.predicted in labels else "neutral"
            matrix[c.label][pred] += 1
        return matrix

    @property
    def avg_latency_ms(self) -> float:
        if not self.judgments:
            return 0.0
        return self.total_latency_ms / len(self.judgments)

    @property
    def estimated_cost_usd(self) -> float | None:
        """Estimated LLM cost for this run, or None for local models."""
        rates = _LLM_COST_PER_1K.get(self.model_name)
        if rates is None:
            return None
        n = len(self.judgments)
        input_cost = (n * _AVG_INPUT_TOKENS / 1000) * rates["input"]
        output_cost = (n * _AVG_OUTPUT_TOKENS / 1000) * rates["output"]
        return round(input_cost + output_cost, 4)

    def calibration(self, score_fn=None):
        """Compute confidence calibration for this result.

        Returns a :class:`~legalkit.eval.calibration.CalibrationResult`
        measuring how well the model's confidence scores align with actual
        accuracy on the test cases.
        """
        from legalkit.eval.calibration import calibrate_nli_result
        return calibrate_nli_result(self, score_fn=score_fn)

    def summary(self) -> str:
        """One-line summary string."""
        cost = self.estimated_cost_usd
        cost_str = f"  est. cost=${cost:.4f}" if cost is not None else "  (local model)"
        return (
            f"{self.model_name:<50} "
            f"binary={self.accuracy_binary:.1%}  "
            f"4-class={self.accuracy_4class:.1%}  "
            f"avg_latency={self.avg_latency_ms:.0f}ms"
            f"{cost_str}"
        )


# ---------------------------------------------------------------------------
# Test case loading
# ---------------------------------------------------------------------------


def load_edge_cases(path: Path | None = None) -> list[NLITestCase]:
    """Load the curated legal NLI edge cases.

    Args:
        path: Override the default path to the JSON file.

    Returns:
        List of :class:`NLITestCase`.
    """
    p = path or _EDGE_CASES_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Edge cases file not found: {p}\n"
            "Expected at data/eval/legal_nli_edge_cases.json"
        )
    with open(p) as f:
        data = json.load(f)

    return [
        NLITestCase(
            id=c["id"],
            category=c["category"],
            passage=c["passage"],
            claim=c["claim"],
            label=c["label"],
            explanation=c.get("explanation", ""),
        )
        for c in data["cases"]
    ]


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


def run_llm_judge(
    test_cases: list[NLITestCase],
    llm,  # LLMClient
    progress_callback: Callable[[int, int], None] | None = None,
) -> NLIEvalResult:
    """Run LLM-as-judge entailment on all test cases.

    Uses the existing :func:`~legalkit.pipeline.verify._check_entailment_llm`
    pipeline so results are directly comparable to production behaviour.

    Args:
        test_cases: Cases to evaluate.
        llm: :class:`~legalkit.llm.client.LLMClient` instance.
        progress_callback: Optional ``(completed, total) -> None``.

    Returns:
        :class:`NLIEvalResult`.
    """
    from legalkit.pipeline.verify import _check_entailment_llm

    judgments: list[NLIJudgment] = []
    total_latency = 0.0

    for i, case in enumerate(test_cases):
        t0 = time.perf_counter()
        try:
            entailment_class, score, explanation = _check_entailment_llm(
                section_id=case.id,
                passage=case.passage,
                claim=case.claim,
                llm=llm,
            )
            predicted = entailment_class.value
        except Exception as exc:
            predicted = "neutral"
            score = 0.0
            explanation = str(exc)

        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency += latency_ms

        judgments.append(NLIJudgment(
            case_id=case.id,
            predicted=predicted,
            confidence=score,
            latency_ms=latency_ms,
            raw_response=explanation,
        ))

        if progress_callback:
            progress_callback(i + 1, len(test_cases))

    return NLIEvalResult(
        model_name=llm.config.generation_model,
        judgments=judgments,
        test_cases=test_cases,
        total_latency_ms=total_latency,
    )


# ---------------------------------------------------------------------------
# DeBERTa NLI judge
# ---------------------------------------------------------------------------

# HuggingFace models to support
DEBERTA_MODELS = {
    "deberta-base": "cross-encoder/nli-deberta-v3-base",
    "deberta-large": "cross-encoder/nli-deberta-v3-large",
    "deberta-small": "cross-encoder/nli-deberta-v3-small",
}


def run_deberta_nli(
    test_cases: list[NLITestCase],
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    device: str = "cpu",
    progress_callback: Callable[[int, int], None] | None = None,
) -> NLIEvalResult:
    """Run a DeBERTa NLI cross-encoder on all test cases.

    Maps the 3-class NLI output (entailment / neutral / contradiction) to
    the 4-class legal schema:
    - entailment score ≥ 0.7  → entailed
    - 0.35 ≤ entailment score < 0.7 → partially_entailed
    - neutral score highest    → neutral
    - contradiction score highest → contradicted

    Args:
        test_cases: Cases to evaluate.
        model_name: HuggingFace model identifier or short alias from
            :data:`DEBERTA_MODELS`.
        device: ``'cpu'`` or ``'cuda'``.
        progress_callback: Optional ``(completed, total) -> None``.

    Returns:
        :class:`NLIEvalResult`.
    """
    resolved_model = DEBERTA_MODELS.get(model_name, model_name)

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for DeBERTa evaluation. "
            "Install with: uv add transformers"
        ) from exc

    nli = hf_pipeline(
        "zero-shot-classification",
        model=resolved_model,
        device=0 if device == "cuda" else -1,
    )

    candidate_labels = ["entailment", "neutral", "contradiction"]
    judgments: list[NLIJudgment] = []
    total_latency = 0.0

    for i, case in enumerate(test_cases):
        t0 = time.perf_counter()
        try:
            result = nli(
                sequences=case.claim,
                candidate_labels=candidate_labels,
                hypothesis_template="{}",
                multi_label=False,
            )
            scores = dict(zip(result["labels"], result["scores"]))
            predicted, confidence = _map_nli_to_legal(scores)
            raw = json.dumps(scores)
        except Exception as exc:
            predicted = "neutral"
            confidence = 0.0
            raw = str(exc)

        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency += latency_ms

        judgments.append(NLIJudgment(
            case_id=case.id,
            predicted=predicted,
            confidence=confidence,
            latency_ms=latency_ms,
            raw_response=raw,
        ))

        if progress_callback:
            progress_callback(i + 1, len(test_cases))

    return NLIEvalResult(
        model_name=resolved_model,
        judgments=judgments,
        test_cases=test_cases,
        total_latency_ms=total_latency,
    )


def _map_nli_to_legal(scores: dict[str, float]) -> tuple[str, float]:
    """Map 3-class NLI scores to 4-class legal entailment labels.

    Uses entailment probability to distinguish entailed from partially_entailed:
    - entailment ≥ 0.70 → entailed
    - 0.35 ≤ entailment < 0.70 → partially_entailed
    - contradiction is highest → contradicted
    - neutral is highest (or entailment < 0.35) → neutral

    Args:
        scores: Dict mapping NLI class names to probabilities.

    Returns:
        ``(predicted_label, confidence)`` tuple.
    """
    ent = scores.get("entailment", 0.0)
    neu = scores.get("neutral", 0.0)
    con = scores.get("contradiction", 0.0)

    # Contradiction wins if it's the top class
    if con > ent and con > neu:
        return "contradicted", con

    # Entailment range
    if ent >= 0.70:
        return "entailed", ent
    elif ent >= 0.35:
        return "partially_entailed", ent

    # Neutral
    return "neutral", neu


# ---------------------------------------------------------------------------
# Fine-tuned legal entailment classifier
# ---------------------------------------------------------------------------


def run_finetuned_nli(
    test_cases: list[NLITestCase],
    model_path: str = "models/legal-entailment/model",
    device: str = "cpu",
    progress_callback: Callable[[int, int], None] | None = None,
) -> NLIEvalResult:
    """Run the fine-tuned legal entailment classifier on test cases.
    
    The fine-tuned model is a 4-class AutoModelForSequenceClassification:
      0 = entailed, 1 = partially_entailed, 2 = contradicted, 3 = neutral
    
    Unlike run_deberta_nli() which uses zero-shot classification with NLI
    candidate labels, this model directly classifies (premise, hypothesis)
    pairs into 4 legal entailment classes.
    
    Args:
        test_cases: Cases to evaluate (uses passage as premise, claim as hypothesis)
        model_path: Local path or HF model ID for the fine-tuned model.
        device: 'cpu' or 'cuda'.
        progress_callback: Optional (completed, total) -> None.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "transformers and torch are required. Install with: uv add transformers torch"
        ) from exc

    # Load model
    resolved_path = str(Path(model_path)) if Path(model_path).exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(resolved_path)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_path)
    device_obj = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device_obj)
    model.eval()

    # Label mapping — determine from model's num_labels
    # (config.id2label is unreliable — it retains the base model's labels
    #  when ignore_mismatched_sizes=True replaces the classification head)
    num_labels = model.config.num_labels
    if num_labels == 3:
        LABEL_NAMES = ["entailed", "contradicted", "neutral"]
    else:
        LABEL_NAMES = ["entailed", "partially_entailed", "contradicted", "neutral"]
    
    judgments: list[NLIJudgment] = []
    total_latency = 0.0
    
    with torch.no_grad():
        for i, case in enumerate(test_cases):
            t0 = time.perf_counter()
            try:
                inputs = tokenizer(
                    case.passage,
                    case.claim,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device_obj)
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                predicted = LABEL_NAMES[pred_idx]
                confidence = float(probs[pred_idx])
                raw = json.dumps({l: round(p, 4) for l, p in zip(LABEL_NAMES, probs)})
            except Exception as exc:
                predicted = "neutral"
                confidence = 0.0
                raw = str(exc)
            
            latency_ms = (time.perf_counter() - t0) * 1000
            total_latency += latency_ms
            
            judgments.append(NLIJudgment(
                case_id=case.id,
                predicted=predicted,
                confidence=confidence,
                latency_ms=latency_ms,
                raw_response=raw,
            ))
            
            if progress_callback:
                progress_callback(i + 1, len(test_cases))
    
    return NLIEvalResult(
        model_name=f"legal-deberta-large:{Path(model_path).name}",
        judgments=judgments,
        test_cases=test_cases,
        total_latency_ms=total_latency,
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def compare_results(
    results: list[NLIEvalResult],
    k: int | None = None,
) -> dict:
    """Produce a structured comparison of multiple eval results.

    Args:
        results: List of :class:`NLIEvalResult` to compare.
        k: If set, only use the first ``k`` test cases.

    Returns:
        Dict with per-model metrics and category breakdown.
    """
    comparison: dict = {"models": [], "category_breakdown": {}}

    categories = sorted({c.category for r in results for c in r.test_cases})

    for result in results:
        cases = result.test_cases[:k] if k else result.test_cases
        judgments = result.judgments[:k] if k else result.judgments

        # Rebuild a trimmed result for metrics
        trimmed = NLIEvalResult(
            model_name=result.model_name,
            judgments=judgments,
            test_cases=cases,
            total_latency_ms=sum(j.latency_ms for j in judgments),
        )

        comparison["models"].append({
            "name": result.model_name,
            "accuracy_binary": round(trimmed.accuracy_binary, 4),
            "accuracy_4class": round(trimmed.accuracy_4class, 4),
            "avg_latency_ms": round(trimmed.avg_latency_ms, 1),
            "estimated_cost_usd": trimmed.estimated_cost_usd,
            "per_category": trimmed.per_category_accuracy,
        })

    # Per-category, per-model breakdown
    for cat in categories:
        comparison["category_breakdown"][cat] = {}
        for m in comparison["models"]:
            cat_data = m["per_category"].get(cat, {})
            comparison["category_breakdown"][cat][m["name"]] = {
                "binary": cat_data.get("accuracy_binary", 0),
                "4class": cat_data.get("accuracy_4class", 0),
            }

    return comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_binary(label: str) -> str:
    return "positive" if label in _BINARY_POSITIVE else "negative"
