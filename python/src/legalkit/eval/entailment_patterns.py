"""Legal entailment pattern analysis and hard-negative generation.

Catalogs the 8 legal NLI patterns where general models fail and provides
utilities for: pattern detection in passages, loading the curated pattern
catalog, and generating additional hard negatives from real clause text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Path to the curated pattern catalog
_PATTERNS_PATH = Path(__file__).parent.parent.parent.parent / "data/eval/legal_entailment_patterns.json"


@dataclass
class PatternMeta:
    """Metadata for a legal entailment pattern category."""

    description: str
    linguistic_signature: str
    why_general_nli_fails: str
    training_priority: int
    regex_hints: list[str] = field(default_factory=list)


# Pattern taxonomy: core linguistic patterns where general NLI models fail
PATTERN_TAXONOMY: dict[str, PatternMeta] = {
    "notwithstanding": PatternMeta(
        description="Clauses using 'notwithstanding' to override or carve out exceptions to general provisions.",
        linguistic_signature="Notwithstanding [general provision], [specific exception or override]",
        why_general_nli_fails="Models see keyword overlap between the notwithstanding clause and a claim about the general provision, predicting entailment without recognizing the override relationship.",
        training_priority=1,
        regex_hints=[
            r"\bnotwithstanding\b",
            r"\bnotwithstanding\s+(?:any|the|section)",
        ],
    ),
    "shall_vs_may": PatternMeta(
        description="Modal verb distinction between mandatory obligations ('shall') and discretionary rights ('may').",
        linguistic_signature="shall [action] vs. may [action]",
        why_general_nli_fails="Models trained on general text do not encode the legal-specific semantic difference between obligation and permission.",
        training_priority=1,
        regex_hints=[
            r"\bshall\b",
            r"\bmay\b(?!\s+not)",
            r"\bshall\s+(?:be|have|provide|deliver|pay|maintain)",
            r"\bmay\s+(?:terminate|assign|request)",
        ],
    ),
    "mae_carveout": PatternMeta(
        description="Material Adverse Effect definitions with exclusion lists using 'shall not constitute' or 'provided that'.",
        linguistic_signature="X shall NOT constitute MAE: (i) Y, (ii) Z...",
        why_general_nli_fails="Models see topical match between excluded items and the MAE definition, missing the explicit exclusion logic.",
        training_priority=2,
        regex_hints=[
            r"\bmaterial\s+adverse\s+effect\b",
            r"\bMAE\b",
            r"\bshall\s+not\s+constitute",
            r"\bexclude[sd]?\b.*\bmaterial\s+adverse",
            r"\bprovided\s+(?:that|however)",
        ],
    ),
    "quantitative_precision": PatternMeta(
        description="Exact numerical amounts, thresholds, caps, and ranges.",
        linguistic_signature="up to X, at least X, exceeding X, not less than X, between X and Y",
        why_general_nli_fails="Models compress numeric tokens into approximate semantic representations, losing precision required for legal interpretation.",
        training_priority=1,
        regex_hints=[
            r"\bup\s+to\s+\$?\d",
            r"\bat\s+least\s+\$?\d",
            r"\bnot\s+(?:less|more)\s+than\s+\$?\d",
            r"\bexceed(?:ing|s)?\s+\$?\d",
            r"\bminimum\s+of\s+\$?\d",
            r"\bmaximum\s+of\s+\$?\d",
            r"\$[\d,]+(?:\.\d+)?(?:\s+million)?",
        ],
    ),
    "temporal_qualifier": PatternMeta(
        description="Time periods, survival clauses, deadlines, and temporal conditions.",
        linguistic_signature="within X days, for a period of X months, prior to Y, following Z",
        why_general_nli_fails="Models attend to events and actions but discount temporal modifiers, treating different time periods as interchangeable.",
        training_priority=2,
        regex_hints=[
            r"\bwithin\s+\d+\s+(?:days?|months?|years?)",
            r"\bfor\s+a\s+period\s+of\s+\d+",
            r"\b\d+\s+(?:days?|months?|years?)\s+(?:following|after|prior to|before)",
            r"\bno\s+(?:later|earlier)\s+than\s+\d+\s+(?:days?|months?)",
            r"\bsurvive\s+(?:termination|closing)",
        ],
    ),
    "condition_precedent": PatternMeta(
        description="Conditional obligations using 'if', 'subject to', 'provided that', or 'unless'.",
        linguistic_signature="if [condition], then [consequence]; subject to [condition], [consequence]",
        why_general_nli_fails="Models see the consequent stated and predict it is entailed, ignoring the conditional structure that gates the obligation.",
        training_priority=1,
        regex_hints=[
            r"\bif\b(?!\s+and\s+only\s+if)",  # Match "if" (excluding "if and only if")
            r"\bsubject\s+to\b",
            r"\bconditioned?\s+(?:on|upon)\b",
            r"\bprovided\s+that\b",
            r"\bunless\b",
            r"\bcontingent\s+(?:on|upon)\b",
        ],
    ),
    "defined_term": PatternMeta(
        description="Defined terms with contractual meanings narrower or broader than ordinary usage.",
        linguistic_signature='"Capitalized Term" means [specific definition]',
        why_general_nli_fails="Models use pre-trained semantic associations for the term's ordinary meaning, failing to substitute the contract-specific definition.",
        training_priority=2,
        regex_hints=[
            r'"[A-Z][^"]*"\s+means\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b.*\bmeans\b',
            r"\bshall\s+have\s+the\s+meaning\b",
            r"\bas\s+defined\s+in\b",
        ],
    ),
    "knowledge_qualifier": PatternMeta(
        description="Representations and warranties qualified by 'to the Knowledge of', 'in all material respects', or similar epistemic/scope limitations.",
        linguistic_signature="to [Party's] Knowledge, in all material respects, except as set forth in Schedule X",
        why_general_nli_fails="Models focus on the semantic content of the representation and ignore the epistemic or materiality qualifier.",
        training_priority=1,
        regex_hints=[
            r"\bto\s+(?:the\s+)?(?:knowledge|best\s+knowledge)\s+of\b",
            r"\bin\s+all\s+material\s+respects\b",
            r"\bexcept\s+as\s+(?:set\s+forth|disclosed)\s+in\s+schedule\b",
            r"\bto\s+\w+(?:'s)?\s+knowledge\b",
        ],
    ),
}


@dataclass
class PatternExample:
    """A single pattern-based entailment example."""

    id: str
    category: str
    passage: str
    claim: str
    label: str  # entailed | contradicted | neutral
    explanation: str = ""
    difficulty: str = "medium"  # easy | medium | hard
    source: str = "curated"  # curated | generated | perturbation


@dataclass
class PatternCatalog:
    """A catalog of pattern-based entailment examples."""

    examples: list[PatternExample]
    version: str = "1.0"
    description: str = ""

    def by_category(self, category: str) -> list[PatternExample]:
        """Return all examples for a given category.

        Args:
            category: Pattern category name (e.g. 'notwithstanding').

        Returns:
            List of examples in that category.
        """
        return [ex for ex in self.examples if ex.category == category]

    def hard_negatives(self) -> list[PatternExample]:
        """Return all hard negative examples (contradicted + neutral).

        These are the examples most valuable for fine-tuning: they are
        topically related to the passage but logically incorrect or
        unaddressed, requiring careful legal reading to classify correctly.

        Returns:
            List of contradicted and neutral examples.
        """
        return [ex for ex in self.examples if ex.label in {"contradicted", "neutral"}]

    def by_difficulty(self, difficulty: str) -> list[PatternExample]:
        """Return examples filtered by difficulty level.

        Args:
            difficulty: One of 'easy', 'medium', 'hard'.

        Returns:
            List of examples at that difficulty level.
        """
        return [ex for ex in self.examples if ex.difficulty == difficulty]

    def to_training_pairs(self) -> list[dict[str, str]]:
        """Convert examples to training pair format.

        Returns:
            List of dicts with keys: passage, claim, label.
        """
        return [
            {
                "passage": ex.passage,
                "claim": ex.claim,
                "label": ex.label,
            }
            for ex in self.examples
        ]

    def stats(self) -> dict[str, Any]:
        """Compute catalog statistics.

        Returns:
            Dict with counts by category and label, plus overall stats.
        """
        from collections import defaultdict

        by_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        by_label: dict[str, int] = defaultdict(int)

        for ex in self.examples:
            by_category[ex.category][ex.label] += 1
            by_label[ex.label] += 1

        category_stats = {}
        for cat, label_counts in sorted(by_category.items()):
            total = sum(label_counts.values())
            category_stats[cat] = {
                "total": total,
                "entailed": label_counts.get("entailed", 0),
                "contradicted": label_counts.get("contradicted", 0),
                "neutral": label_counts.get("neutral", 0),
                "hard_negative_pct": round(
                    (label_counts.get("contradicted", 0) + label_counts.get("neutral", 0)) / total * 100,
                    1,
                ),
            }

        return {
            "total_examples": len(self.examples),
            "by_label": dict(by_label),
            "by_category": category_stats,
            "categories": sorted(by_category.keys()),
        }


def load_pattern_catalog(path: Path | None = None) -> PatternCatalog:
    """Load the curated legal entailment pattern catalog.

    Args:
        path: Override the default path to the JSON file.

    Returns:
        PatternCatalog instance.

    Raises:
        FileNotFoundError: If the catalog file does not exist.
    """
    p = path or _PATTERNS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Pattern catalog not found: {p}\n"
            "Expected at data/eval/legal_entailment_patterns.json"
        )

    with open(p) as f:
        data = json.load(f)

    examples = [
        PatternExample(
            id=ex["id"],
            category=ex["category"],
            passage=ex["passage"],
            claim=ex["claim"],
            label=ex["label"],
            explanation=ex.get("explanation", ""),
            difficulty=ex.get("difficulty", "medium"),
            source=ex.get("source", "curated"),
        )
        for ex in data["examples"]
    ]

    return PatternCatalog(
        examples=examples,
        version=data.get("version", "1.0"),
        description=data.get("description", ""),
    )


def detect_patterns(passage: str) -> list[str]:
    """Detect which pattern categories are likely present in a passage.

    Uses regex hints from the pattern taxonomy to identify linguistic
    signatures. Useful for targeted hard-negative generation or filtering
    training data by pattern type.

    Args:
        passage: Legal clause text.

    Returns:
        List of pattern category names detected (may be empty).
    """
    passage_lower = passage.lower()
    detected: set[str] = set()

    for category, meta in PATTERN_TAXONOMY.items():
        for regex_hint in meta.regex_hints:
            if re.search(regex_hint, passage, re.IGNORECASE):
                detected.add(category)
                break  # One match per category is sufficient

    return sorted(detected)


def generate_negatives_from_perturbation(
    passage: str,
    entailed_claim: str,
    category: str,
) -> list[PatternExample]:
    """Generate hard negatives by perturbing an entailed claim.

    Applies category-specific perturbation strategies to create
    contradicted claims that are topically similar but logically incorrect.

    Currently supports:
    - shall_vs_may: swap shall ↔ may
    - quantitative_precision: change amounts by ±50%, invert thresholds
    - temporal_qualifier: change time periods, swap prior to/following
    - knowledge_qualifier: drop 'to X's Knowledge' qualifier

    Args:
        passage: Original legal clause.
        entailed_claim: A claim that is entailed by the passage.
        category: Pattern category for perturbation strategy.

    Returns:
        List of PatternExample with contradicted labels (may be empty
        if no perturbations are applicable).
    """
    perturbations: list[PatternExample] = []

    if category == "shall_vs_may":
        # Swap shall → may (weakens obligation to discretion)
        if "shall" in entailed_claim.lower():
            perturbed = re.sub(r"\bshall\b", "may", entailed_claim, count=1, flags=re.IGNORECASE)
            perturbations.append(
                PatternExample(
                    id=f"gen-sm-{hash(perturbed) % 10000:04d}",
                    category="shall_vs_may",
                    passage=passage,
                    claim=perturbed,
                    label="contradicted",
                    explanation="Perturbation: changed 'shall' to 'may', weakening mandatory obligation to discretionary right.",
                    difficulty="hard",
                    source="perturbation",
                )
            )
        # Swap may → shall (strengthens discretion to obligation)
        if "may" in entailed_claim.lower() and "may not" not in entailed_claim.lower():
            perturbed = re.sub(r"\bmay\b", "shall", entailed_claim, count=1, flags=re.IGNORECASE)
            perturbations.append(
                PatternExample(
                    id=f"gen-sm-{hash(perturbed) % 10000:04d}",
                    category="shall_vs_may",
                    passage=passage,
                    claim=perturbed,
                    label="contradicted",
                    explanation="Perturbation: changed 'may' to 'shall', upgrading discretionary right to mandatory obligation.",
                    difficulty="hard",
                    source="perturbation",
                )
            )

    elif category == "quantitative_precision":
        # Change dollar amounts by ±50%
        def alter_amount(match: re.Match) -> str:
            amount_str = match.group(1).replace(",", "")
            try:
                amount = float(amount_str)
                # Randomly choose +50% or -50% (use hash for determinism)
                if hash(passage) % 2 == 0:
                    new_amount = amount * 1.5
                else:
                    new_amount = amount * 0.5
                # Format with commas
                if new_amount >= 1000:
                    return f"${new_amount:,.0f}"
                else:
                    return f"${new_amount:.2f}"
            except ValueError:
                return match.group(0)

        perturbed = re.sub(r"\$([\d,]+(?:\.\d+)?)", alter_amount, entailed_claim)
        if perturbed != entailed_claim:
            perturbations.append(
                PatternExample(
                    id=f"gen-qp-{hash(perturbed) % 10000:04d}",
                    category="quantitative_precision",
                    passage=passage,
                    claim=perturbed,
                    label="contradicted",
                    explanation="Perturbation: changed dollar amount by ±50% to test numeric precision.",
                    difficulty="hard",
                    source="perturbation",
                )
            )

        # Invert threshold direction (at least → up to, etc.)
        if "at least" in entailed_claim.lower():
            perturbed = re.sub(r"\bat\s+least\b", "up to", entailed_claim, count=1, flags=re.IGNORECASE)
            perturbations.append(
                PatternExample(
                    id=f"gen-qp-{hash(perturbed) % 10000:04d}",
                    category="quantitative_precision",
                    passage=passage,
                    claim=perturbed,
                    label="contradicted",
                    explanation="Perturbation: inverted threshold from 'at least' (floor) to 'up to' (ceiling).",
                    difficulty="hard",
                    source="perturbation",
                )
            )
        elif "up to" in entailed_claim.lower():
            perturbed = re.sub(r"\bup\s+to\b", "at least", entailed_claim, count=1, flags=re.IGNORECASE)
            perturbations.append(
                PatternExample(
                    id=f"gen-qp-{hash(perturbed) % 10000:04d}",
                    category="quantitative_precision",
                    passage=passage,
                    claim=perturbed,
                    label="contradicted",
                    explanation="Perturbation: inverted threshold from 'up to' (ceiling) to 'at least' (floor).",
                    difficulty="hard",
                    source="perturbation",
                )
            )

    elif category == "temporal_qualifier":
        # Change time periods (30 days → 60 days, etc.)
        def alter_period(match: re.Match) -> str:
            num = int(match.group(1))
            unit = match.group(2)
            # Double or halve the period
            new_num = num * 2 if hash(passage) % 2 == 0 else max(1, num // 2)
            return f"{new_num} {unit}"

        perturbed = re.sub(
            r"\b(\d+)\s+(days?|months?|years?)\b",
            alter_period,
            entailed_claim,
            count=1,
            flags=re.IGNORECASE,
        )
        if perturbed != entailed_claim:
            perturbations.append(
                PatternExample(
                    id=f"gen-tq-{hash(perturbed) % 10000:04d}",
                    category="temporal_qualifier",
                    passage=passage,
                    claim=perturbed,
                    label="contradicted",
                    explanation="Perturbation: changed time period (doubled or halved) to test temporal precision.",
                    difficulty="hard",
                    source="perturbation",
                )
            )

        # Swap prior to ↔ following
        if "prior to" in entailed_claim.lower():
            perturbed = re.sub(r"\bprior\s+to\b", "following", entailed_claim, count=1, flags=re.IGNORECASE)
            perturbations.append(
                PatternExample(
                    id=f"gen-tq-{hash(perturbed) % 10000:04d}",
                    category="temporal_qualifier",
                    passage=passage,
                    claim=perturbed,
                    label="contradicted",
                    explanation="Perturbation: swapped 'prior to' → 'following', inverting temporal direction.",
                    difficulty="hard",
                    source="perturbation",
                )
            )
        elif "following" in entailed_claim.lower():
            perturbed = re.sub(r"\bfollowing\b", "prior to", entailed_claim, count=1, flags=re.IGNORECASE)
            perturbations.append(
                PatternExample(
                    id=f"gen-tq-{hash(perturbed) % 10000:04d}",
                    category="temporal_qualifier",
                    passage=passage,
                    claim=perturbed,
                    label="contradicted",
                    explanation="Perturbation: swapped 'following' → 'prior to', inverting temporal direction.",
                    difficulty="hard",
                    source="perturbation",
                )
            )

    elif category == "knowledge_qualifier":
        # Drop "to X's Knowledge" qualifier
        perturbed = re.sub(
            r"\bto\s+(?:the\s+)?(?:\w+(?:'s)?)\s+(?:knowledge|best\s+knowledge),?\s*",
            "",
            entailed_claim,
            count=1,
            flags=re.IGNORECASE,
        )
        perturbed = re.sub(r"\s{2,}", " ", perturbed).strip()  # Clean up spacing
        if perturbed != entailed_claim:
            perturbations.append(
                PatternExample(
                    id=f"gen-kq-{hash(perturbed) % 10000:04d}",
                    category="knowledge_qualifier",
                    passage=passage,
                    claim=perturbed,
                    label="contradicted",
                    explanation="Perturbation: removed knowledge qualifier, upgrading limited representation to absolute assertion.",
                    difficulty="hard",
                    source="perturbation",
                )
            )

        # Drop "in all material respects"
        if "in all material respects" in entailed_claim.lower():
            perturbed = re.sub(
                r"\bin\s+all\s+material\s+respects,?\s*",
                "",
                entailed_claim,
                count=1,
                flags=re.IGNORECASE,
            )
            perturbed = re.sub(r"\s{2,}", " ", perturbed).strip()
            if perturbed != entailed_claim:
                perturbations.append(
                    PatternExample(
                        id=f"gen-kq-{hash(perturbed) % 10000:04d}",
                        category="knowledge_qualifier",
                        passage=passage,
                        claim=perturbed,
                        label="contradicted",
                        explanation="Perturbation: removed 'in all material respects', upgrading qualified statement to absolute.",
                        difficulty="hard",
                        source="perturbation",
                    )
                )

    return perturbations


def augment_training_data(
    base_pairs: list[dict[str, str]],
    catalog: PatternCatalog,
) -> list[dict[str, str]]:
    """Augment existing training data with pattern catalog examples.

    Merges the catalog's curated hard negatives into a base set of
    training pairs, removing duplicates based on (passage, claim) tuples.

    Args:
        base_pairs: Existing training pairs (each dict has keys: passage, claim, label).
        catalog: PatternCatalog with curated examples.

    Returns:
        Merged list of training pairs with duplicates removed.
    """
    # Use set to track (passage, claim) tuples for deduplication
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, str]] = []

    # Add base pairs first
    for pair in base_pairs:
        key = (pair["passage"], pair["claim"])
        if key not in seen:
            seen.add(key)
            merged.append(pair)

    # Add catalog examples
    for pair in catalog.to_training_pairs():
        key = (pair["passage"], pair["claim"])
        if key not in seen:
            seen.add(key)
            merged.append(pair)

    return merged
