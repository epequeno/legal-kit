"""
Generate hard-negative training pairs for legal entailment fine-tuning.

Replaces the cross-passage random negatives that caused v0 fine-tuning
failure (26.4% accuracy) with topically-related hard negatives. Takes
existing entailed pairs from the entailment log, applies pattern-based
perturbations, and merges with the curated catalog.

Usage:
    python -m legalkit.eval.generate_hard_negatives
    python -m legalkit.eval.generate_hard_negatives --output data/training/hard_negatives.jsonl
    legalkit generate-hard-negatives  # via CLI
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

from legalkit.eval.entailment_patterns import (
    detect_patterns,
    generate_negatives_from_perturbation,
    load_pattern_catalog,
)


# Default paths
_DEFAULT_LOG_PATH = Path("data/entailment_logs/entailment_2026-03-17.jsonl")
_DEFAULT_CATALOG_PATH = Path("data/eval/legal_entailment_patterns.json")
_DEFAULT_OUTPUT_PATH = Path("data/training/hard_negatives.jsonl")

# Class label mapping
LABEL_MAP = {
    "entailed": 0,
    "partially_entailed": 1,
    "contradicted": 2,
    "neutral": 3,
}


def load_entailed_pairs(
    log_path: Path | None = None,
    sources: list[str] | None = None,
    max_n: int | None = None,
) -> list[dict]:
    """Load entailed+partially_entailed pairs from entailment log.

    Default log_path: data/entailment_logs/entailment_2026-03-17.jsonl
    Returns pairs with keys: passage, claim, label, section_id, doc_id, source
    Filters to entailed and partially_entailed only (these are the positive
    pairs we perturb to create hard negatives).

    Args:
        log_path: Path to entailment log JSONL file. Defaults to latest log.
        sources: Filter to specific sources (e.g. ['llm_generated', 'pipeline']).
                 None means all sources.
        max_n: Maximum number of pairs to load. None means unlimited.

    Returns:
        List of dicts with keys: passage, claim, label, section_id, doc_id, source
    """
    path = log_path or _DEFAULT_LOG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Entailment log not found: {path}")

    pairs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            # Filter to entailed/partially_entailed
            if record.get("entailment_class") not in {"entailed", "partially_entailed"}:
                continue

            # Filter by source if specified
            if sources is not None and record.get("source") not in sources:
                continue

            pairs.append({
                "passage": record["passage"],
                "claim": record["claim"],
                "label": record["entailment_class"],
                "section_id": record.get("section_id", ""),
                "doc_id": record.get("doc_id", ""),
                "source": record.get("source", "unknown"),
            })

            if max_n and len(pairs) >= max_n:
                break

    return pairs


def generate_perturbation_negatives(
    pairs: list[dict],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Run detect_patterns() + generate_negatives_from_perturbation() on each pair.

    For each pair:
    1. Run detect_patterns(pair['passage']) to find which patterns are present
    2. For each detected pattern, call generate_negatives_from_perturbation()
    3. Convert PatternExample results to training-data JSONL dicts

    Returns list of dicts in training JSONL format with source='perturbation:pattern_name'

    Args:
        pairs: List of entailed pairs with keys: passage, claim, label, ...
        progress_callback: Optional callback(completed, total) for progress updates.

    Returns:
        List of training dicts with keys: premise, hypothesis, label, label_name,
        section_id, doc_id, source
    """
    negatives = []
    total = len(pairs)

    for idx, pair in enumerate(pairs, 1):
        passage = pair["passage"]
        claim = pair["claim"]

        # Detect which patterns are present
        detected = detect_patterns(passage)

        # Generate perturbations for each detected pattern
        for pattern_name in detected:
            perturbations = generate_negatives_from_perturbation(
                passage=passage,
                entailed_claim=claim,
                category=pattern_name,
            )

            # Convert PatternExample to training format
            for pex in perturbations:
                negatives.append({
                    "premise": pex.passage,  # Map passage → premise
                    "hypothesis": pex.claim,  # Map claim → hypothesis
                    "label": LABEL_MAP.get(pex.label, 3),
                    "label_name": pex.label,
                    "section_id": pair.get("section_id", ""),
                    "doc_id": pair.get("doc_id", ""),
                    "source": f"perturbation:{pattern_name}",
                })

        if progress_callback:
            progress_callback(idx, total)

    return negatives


def load_catalog_as_training_pairs(
    catalog_path: Path | None = None,
) -> list[dict]:
    """Convert curated pattern catalog to training JSONL format.

    Maps PatternExample.label ('entailed'/'contradicted'/'neutral')
    to integer labels. Sets source='curated:catalog'.

    Args:
        catalog_path: Path to catalog JSON. Defaults to data/eval/legal_entailment_patterns.json

    Returns:
        List of training dicts with keys: premise, hypothesis, label, label_name,
        section_id, doc_id, source
    """
    path = catalog_path or _DEFAULT_CATALOG_PATH
    catalog = load_pattern_catalog(path)

    training_pairs = []
    for example in catalog.examples:
        training_pairs.append({
            "premise": example.passage,  # Map passage → premise
            "hypothesis": example.claim,  # Map claim → hypothesis
            "label": LABEL_MAP.get(example.label, 3),
            "label_name": example.label,
            "section_id": "",  # Catalog examples don't have section IDs
            "doc_id": "",
            "source": "curated:catalog",
        })

    return training_pairs


def deduplicate(
    pairs: list[dict],
    key_fields: tuple[str, ...] = ("premise", "hypothesis"),
) -> list[dict]:
    """Remove duplicate (premise, hypothesis) pairs. Keeps first occurrence.

    Args:
        pairs: List of training dicts.
        key_fields: Tuple of keys to use for deduplication.

    Returns:
        Deduplicated list of dicts.
    """
    seen = set()
    unique = []

    for pair in pairs:
        key = tuple(pair.get(field, "") for field in key_fields)
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    return unique


def generate_hard_negatives(
    log_path: Path | None = None,
    catalog_path: Path | None = None,
    output_path: Path | None = None,
    max_from_log: int | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Main generation function. Returns stats dict.

    Pipeline:
    1. Load entailed pairs from log (filter to llm_generated + pipeline + perturbation sources)
    2. Generate perturbation negatives from those pairs
    3. Load curated catalog examples
    4. Combine, deduplicate, save

    Args:
        log_path: Path to entailment log. Defaults to data/entailment_logs/entailment_2026-03-17.jsonl
        catalog_path: Path to pattern catalog. Defaults to data/eval/legal_entailment_patterns.json
        output_path: Where to save. Defaults to data/training/hard_negatives.jsonl
        max_from_log: Cap on pairs loaded from log (for testing). None means unlimited.
        progress_callback: Optional callback(stage, completed, total) for progress updates.

    Returns:
        Stats dict with:
        {
            'total': int,
            'by_label': {'entailed': n, 'partially_entailed': n, 'contradicted': n, 'neutral': n},
            'by_source': {'perturbation:shall_vs_may': n, 'curated:catalog': n, ...},
            'output_path': str,
        }
    """
    output_path = output_path or _DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage 1: Load entailed pairs from log
    # Filter to high-quality sources (exclude cross_passage which is the problem we're fixing)
    # Note: sources in the log are prefixed with "synthetic:" for generated data
    quality_sources = [
        "synthetic:llm_generated",
        "pipeline",
        "synthetic:perturbation:negated",
        "synthetic:perturbation:period_changed",
        "synthetic:perturbation:amount_changed",
        "synthetic:perturbation:parties_swapped",
    ]
    entailed_pairs = load_entailed_pairs(
        log_path=log_path,
        sources=quality_sources,
        max_n=max_from_log,
    )

    if progress_callback:
        progress_callback("load", len(entailed_pairs), len(entailed_pairs))

    # Stage 2: Generate perturbation negatives
    def _perturb_progress(completed: int, total: int) -> None:
        if progress_callback:
            progress_callback("perturb", completed, total)

    perturbation_negatives = generate_perturbation_negatives(
        entailed_pairs,
        progress_callback=_perturb_progress,
    )

    # Stage 3: Load curated catalog
    catalog_pairs = load_catalog_as_training_pairs(catalog_path=catalog_path)

    if progress_callback:
        progress_callback("catalog", len(catalog_pairs), len(catalog_pairs))

    # Stage 4: Combine and deduplicate
    all_pairs = perturbation_negatives + catalog_pairs
    unique_pairs = deduplicate(all_pairs)

    # Write to output
    with open(output_path, "w") as f:
        for pair in unique_pairs:
            f.write(json.dumps(pair) + "\n")

    # Compute stats
    label_counter = Counter(p["label_name"] for p in unique_pairs)
    source_counter = Counter(p["source"] for p in unique_pairs)

    stats = {
        "total": len(unique_pairs),
        "by_label": dict(label_counter),
        "by_source": dict(source_counter),
        "output_path": str(output_path),
        "input_entailed_pairs": len(entailed_pairs),
        "perturbation_generated": len(perturbation_negatives),
        "catalog_examples": len(catalog_pairs),
        "duplicates_removed": len(all_pairs) - len(unique_pairs),
    }

    return stats


if __name__ == "__main__":
    # Simple CLI when run directly
    import sys

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUTPUT_PATH
    stats = generate_hard_negatives(output_path=output)

    print(f"\n✓ Generated {stats['total']} hard-negative training pairs")
    print(f"  Output: {stats['output_path']}")
    print(f"\nBy label:")
    for label, count in sorted(stats["by_label"].items()):
        print(f"  {label}: {count}")
    print(f"\nBy source:")
    for source, count in sorted(stats["by_source"].items()):
        print(f"  {source}: {count}")
