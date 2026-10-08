"""Prepare entailment training data for fine-tuning.

Takes raw logged samples and produces balanced train/val/test splits
in formats suitable for DeBERTa (NLI) and decoder LLMs.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from legalkit.eval.entailment_log import EntailmentLogger, EntailmentSample


# Class label mapping for NLI models
LABEL_MAP = {
    "entailed": 0,
    "partially_entailed": 1,
    "contradicted": 2,
    "neutral": 3,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def prepare_dataset(
    log_dir: Path | None = None,
    output_dir: Path | None = None,
    balance_strategy: str = "downsample",
    max_samples_per_class: int | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    hard_negatives_path: Path | None = None,
    cross_passage_cap: float = 0.15,
) -> dict:
    """Prepare balanced train/val/test splits from logged entailment samples.

    Args:
        log_dir: Directory containing entailment JSONL logs.
        output_dir: Where to write the prepared datasets.
        balance_strategy: "downsample" (reduce majority) or "none" (keep as-is).
        max_samples_per_class: Cap per class (overrides balance_strategy).
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        hard_negatives_path: Path to hard_negatives.jsonl. If provided, these
                             are merged into the pool before balancing.
        cross_passage_cap: Max fraction of neutral class from cross_passage source
                           (default 0.15 = 15%). Prevents trivial negatives.

    Returns:
        Stats dict with sample counts.
    """
    output_dir = output_dir or Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # Load all samples
    samples = EntailmentLogger.load_all(log_dir)
    if not samples:
        raise ValueError("No entailment samples found. Run pipeline or generate-data first.")

    # Filter out very short passages/claims
    samples = [
        s for s in samples
        if len(s.passage.strip()) >= 30 and len(s.claim.strip()) >= 10
    ]

    # Load hard negatives if provided
    if hard_negatives_path and hard_negatives_path.exists():
        with open(hard_negatives_path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                # Convert back to EntailmentSample format
                samples.append(EntailmentSample(
                    passage=record["premise"],  # Map premise → passage
                    claim=record["hypothesis"],  # Map hypothesis → claim
                    entailment_class=record["label_name"],
                    confidence=1.0,
                    explanation="",
                    section_id=record.get("section_id", ""),
                    doc_id=record.get("doc_id", ""),
                    query="",
                    judge_model="",
                    timestamp="",
                    source=record.get("source", "hard_negative"),
                    reviewed=False,
                    corrected_class=None,
                ))

        # When hard negatives are provided, default to 1000 per class if not set
        if max_samples_per_class is None:
            max_samples_per_class = 1000

    # Group by class
    by_class: dict[str, list[EntailmentSample]] = {}
    for s in samples:
        by_class.setdefault(s.entailment_class, []).append(s)

    # Balance
    if max_samples_per_class:
        target_per_class = max_samples_per_class
    elif balance_strategy == "downsample":
        # Downsample to the size of the smallest non-trivial class
        class_sizes = {k: len(v) for k, v in by_class.items() if len(v) > 0}
        target_per_class = min(class_sizes.values())
    else:
        target_per_class = max(len(v) for v in by_class.values())

    balanced: list[EntailmentSample] = []
    for cls, cls_samples in by_class.items():
        random.shuffle(cls_samples)

        # Special handling for neutral class: cap cross_passage samples
        if cls == "neutral" and hard_negatives_path:
            # Separate cross_passage from other neutrals
            cross_passage = [s for s in cls_samples if "cross_passage" in s.source]
            other_neutrals = [s for s in cls_samples if "cross_passage" not in s.source]

            # Cap cross_passage to cross_passage_cap * target
            max_cross_passage = int(target_per_class * cross_passage_cap)
            cross_passage_sample = cross_passage[:max_cross_passage]

            # Fill remainder with other neutrals
            remaining_slots = target_per_class - len(cross_passage_sample)
            other_sample = other_neutrals[:remaining_slots]

            balanced.extend(cross_passage_sample + other_sample)
        else:
            balanced.extend(cls_samples[:target_per_class])

    random.shuffle(balanced)

    # Split
    n = len(balanced)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    test_set = balanced[:n_test]
    val_set = balanced[n_test:n_test + n_val]
    train_set = balanced[n_test + n_val:]

    # Write outputs
    _write_nli_format(train_set, output_dir / "train.jsonl")
    _write_nli_format(val_set, output_dir / "val.jsonl")
    _write_nli_format(test_set, output_dir / "test.jsonl")

    # Also write a HuggingFace-compatible dataset
    _write_hf_format(train_set, val_set, test_set, output_dir)

    # Stats
    stats = {
        "total_raw": len(samples),
        "total_balanced": len(balanced),
        "target_per_class": target_per_class,
        "balance_strategy": balance_strategy,
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
        "train_distribution": dict(Counter(s.entailment_class for s in train_set)),
        "val_distribution": dict(Counter(s.entailment_class for s in val_set)),
        "test_distribution": dict(Counter(s.entailment_class for s in test_set)),
        "source_distribution": dict(Counter(s.source for s in balanced)),
    }

    # Write stats
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    return stats


def _write_nli_format(samples: list[EntailmentSample], path: Path) -> None:
    """Write samples in NLI format (premise/hypothesis/label)."""
    with open(path, "w") as f:
        for s in samples:
            record = {
                "premise": s.passage,
                "hypothesis": s.claim,
                "label": LABEL_MAP.get(s.entailment_class, 3),
                "label_name": s.entailment_class,
                "section_id": s.section_id,
                "doc_id": s.doc_id,
                "source": s.source,
            }
            f.write(json.dumps(record) + "\n")


def _write_hf_format(
    train: list[EntailmentSample],
    val: list[EntailmentSample],
    test: list[EntailmentSample],
    output_dir: Path,
) -> None:
    """Write in HuggingFace datasets format (dataset_dict with arrow-compatible JSONL)."""
    hf_dir = output_dir / "hf_dataset"
    hf_dir.mkdir(exist_ok=True)

    for split_name, split_data in [("train", train), ("validation", val), ("test", test)]:
        split_file = hf_dir / f"{split_name}.jsonl"
        with open(split_file, "w") as f:
            for s in split_data:
                record = {
                    "premise": s.passage,
                    "hypothesis": s.claim,
                    "label": LABEL_MAP.get(s.entailment_class, 3),
                }
                f.write(json.dumps(record) + "\n")

    # Write dataset card
    card = """---
dataset_info:
  features:
    - name: premise
      dtype: string
    - name: hypothesis
      dtype: string
    - name: label
      dtype:
        class_label:
          names:
            '0': entailed
            '1': partially_entailed
            '2': contradicted
            '3': neutral
  splits:
    - name: train
    - name: validation
    - name: test
---

# Legal Entailment Dataset

Training data for legal-domain Natural Language Inference.

Generated from the LegalBench-RAG corpus using:
- LLM-generated claim pairs from real legal passages
- Perturbation-based negatives (amount, party, negation changes)
- Cross-passage negative mining
- Pipeline verification judgments

## Label Definitions

- **entailed** (0): The passage directly supports the claim
- **partially_entailed** (1): The passage somewhat supports the claim but the claim makes inferences
- **contradicted** (2): The passage contradicts the claim
- **neutral** (3): The passage is unrelated to the claim
"""
    (hf_dir / "README.md").write_text(card)
