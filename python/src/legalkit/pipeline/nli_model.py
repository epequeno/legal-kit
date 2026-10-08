"""Local NLI model wrapper for legal entailment checking.

Provides a lazily-loaded, cached wrapper around the fine-tuned
DeBERTa-large legal entailment model. Thread-safe via module-level
singleton pattern. Falls back to None if the model path doesn't exist,
allowing callers to fall back to LLM-as-judge.

Usage:
    model = LocalNLIModel.load("models/legal-entailment/model")
    if model:
        cls, score, explanation = model.predict(passage, claim)
    else:
        # fall back to LLM
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LocalNLIModel:
    """Loaded fine-tuned legal entailment classifier."""

    model_path: str
    _model: object = None  # AutoModelForSequenceClassification
    _tokenizer: object = None  # AutoTokenizer
    _device: str = "cpu"


# Class-level constants and cache
LocalNLIModel.LABEL_NAMES = ["entailed", "partially_entailed", "contradicted", "neutral"]
LocalNLIModel._cache = {}


@classmethod
def _load_model(
    cls, model_path: str | None, device: str = "cpu"
) -> "LocalNLIModel | None":
    """Load model from path, return None if path doesn't exist or load fails.

    Caches by path — second call with same path returns the already-loaded model.
    Returns None (not an exception) if the model isn't available, so callers
    can gracefully fall back to LLM.
    """
    if not model_path:
        return None
    if not Path(model_path).exists():
        return None
    if model_path in cls._cache:
        return cls._cache[model_path]
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tok = AutoTokenizer.from_pretrained(model_path)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_path)
        device_obj = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        mdl = mdl.to(device_obj)
        mdl.eval()
        instance = cls(
            model_path=model_path,
            _model=mdl,
            _tokenizer=tok,
            _device=str(device_obj),
        )
        cls._cache[model_path] = instance
        return instance
    except Exception:
        return None


def _predict(
    self,
    passage: str,
    claim: str,
) -> tuple[str, float, str]:
    """Predict entailment class for (passage, claim).

    Returns:
        (label, confidence, explanation_json)
        label is one of: entailed, partially_entailed, contradicted, neutral
    """
    import torch

    inputs = self._tokenizer(
        passage[:2000],
        claim,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to(self._device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = self._model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    pred_idx = int(torch.argmax(logits).item())
    label = self.LABEL_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    explanation = json.dumps(
        {l: round(p, 4) for l, p in zip(self.LABEL_NAMES, probs)}
    )
    return label, confidence, explanation


@classmethod
def _clear_cache(cls) -> None:
    """Clear the model cache (useful for testing)."""
    cls._cache.clear()


# Attach methods to class
LocalNLIModel.load = _load_model
LocalNLIModel.predict = _predict
LocalNLIModel.clear_cache = _clear_cache
