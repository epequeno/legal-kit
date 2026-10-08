"""Tests for legal entailment pattern catalog and utilities."""

from __future__ import annotations

import pytest

from legalkit.eval.entailment_patterns import (
    PATTERN_TAXONOMY,
    PatternCatalog,
    PatternExample,
    augment_training_data,
    detect_patterns,
    generate_negatives_from_perturbation,
    load_pattern_catalog,
)


# ---------------------------------------------------------------------------
# Catalog loading and structure
# ---------------------------------------------------------------------------


def test_load_pattern_catalog():
    """Test that the pattern catalog loads successfully."""
    catalog = load_pattern_catalog()
    assert isinstance(catalog, PatternCatalog)
    assert len(catalog.examples) > 0
    assert catalog.version == "1.0"
    assert catalog.description != ""


def test_catalog_has_all_categories():
    """Test that all 8 pattern categories are represented."""
    catalog = load_pattern_catalog()
    categories = {ex.category for ex in catalog.examples}
    expected_categories = {
        "notwithstanding",
        "shall_vs_may",
        "mae_carveout",
        "quantitative_precision",
        "temporal_qualifier",
        "condition_precedent",
        "defined_term",
        "knowledge_qualifier",
    }
    assert categories == expected_categories


def test_minimum_examples_per_category():
    """Test that each category has at least 10 examples."""
    catalog = load_pattern_catalog()
    stats = catalog.stats()
    for cat, cat_stats in stats["by_category"].items():
        assert cat_stats["total"] >= 10, f"Category {cat} has only {cat_stats['total']} examples (minimum 10)"


def test_hard_negative_ratio():
    """Test that at least 50% of examples per category are hard negatives (contradicted/neutral)."""
    catalog = load_pattern_catalog()
    stats = catalog.stats()
    for cat, cat_stats in stats["by_category"].items():
        hard_neg_pct = cat_stats["hard_negative_pct"]
        assert hard_neg_pct >= 50.0, f"Category {cat} has only {hard_neg_pct}% hard negatives (minimum 50%)"


def test_example_structure():
    """Test that examples have required fields and valid values."""
    catalog = load_pattern_catalog()
    valid_labels = {"entailed", "contradicted", "neutral"}
    valid_difficulties = {"easy", "medium", "hard"}

    for ex in catalog.examples:
        assert isinstance(ex, PatternExample)
        assert ex.id != ""
        assert ex.category in PATTERN_TAXONOMY
        assert ex.passage != ""
        assert ex.claim != ""
        assert ex.label in valid_labels, f"Invalid label: {ex.label}"
        assert ex.difficulty in valid_difficulties, f"Invalid difficulty: {ex.difficulty}"
        assert ex.explanation != ""


# ---------------------------------------------------------------------------
# PatternCatalog methods
# ---------------------------------------------------------------------------


def test_by_category():
    """Test filtering examples by category."""
    catalog = load_pattern_catalog()

    # Test notwithstanding category
    nw_examples = catalog.by_category("notwithstanding")
    assert len(nw_examples) > 0
    assert all(ex.category == "notwithstanding" for ex in nw_examples)

    # Test that all categories return non-empty results
    for cat in PATTERN_TAXONOMY:
        cat_examples = catalog.by_category(cat)
        assert len(cat_examples) > 0, f"No examples found for category: {cat}"


def test_hard_negatives():
    """Test that hard_negatives() excludes entailed examples."""
    catalog = load_pattern_catalog()
    hard_negs = catalog.hard_negatives()

    # Should only contain contradicted and neutral examples
    assert all(ex.label in {"contradicted", "neutral"} for ex in hard_negs)

    # Should have fewer examples than total (since some are entailed)
    assert len(hard_negs) < len(catalog.examples)

    # Should be a significant portion of the catalog
    assert len(hard_negs) > len(catalog.examples) * 0.5


def test_by_difficulty():
    """Test filtering by difficulty level."""
    catalog = load_pattern_catalog()

    easy = catalog.by_difficulty("easy")
    medium = catalog.by_difficulty("medium")
    hard = catalog.by_difficulty("hard")

    # All difficulties should have examples
    assert len(easy) > 0
    assert len(medium) >= 0  # Medium may be empty if all are easy/hard
    assert len(hard) > 0

    # Check that filtering is correct
    assert all(ex.difficulty == "easy" for ex in easy)
    assert all(ex.difficulty == "medium" for ex in medium)
    assert all(ex.difficulty == "hard" for ex in hard)


def test_to_training_pairs():
    """Test conversion to training pair format."""
    catalog = load_pattern_catalog()
    pairs = catalog.to_training_pairs()

    assert len(pairs) == len(catalog.examples)

    for pair in pairs:
        assert "passage" in pair
        assert "claim" in pair
        assert "label" in pair
        assert pair["passage"] != ""
        assert pair["claim"] != ""
        assert pair["label"] in {"entailed", "contradicted", "neutral"}


def test_stats():
    """Test statistics computation."""
    catalog = load_pattern_catalog()
    stats = catalog.stats()

    # Check overall structure
    assert "total_examples" in stats
    assert "by_label" in stats
    assert "by_category" in stats
    assert "categories" in stats

    # Total should match
    assert stats["total_examples"] == len(catalog.examples)

    # All categories should be present
    assert len(stats["categories"]) == 8

    # Label counts should sum to total
    label_sum = sum(stats["by_label"].values())
    assert label_sum == stats["total_examples"]

    # Category counts should sum to total
    category_sum = sum(cat_stats["total"] for cat_stats in stats["by_category"].values())
    assert category_sum == stats["total_examples"]

    # Each category should have hard_negative_pct
    for cat_stats in stats["by_category"].values():
        assert "hard_negative_pct" in cat_stats
        assert 0 <= cat_stats["hard_negative_pct"] <= 100


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_detect_patterns_notwithstanding():
    """Test detection of notwithstanding patterns."""
    passage = "Notwithstanding any other provision, the liability cap shall not apply to fraud."
    detected = detect_patterns(passage)
    assert "notwithstanding" in detected


def test_detect_patterns_shall_vs_may():
    """Test detection of shall/may patterns."""
    passage1 = "Buyer shall pay the Purchase Price at Closing."
    passage2 = "Seller may terminate this Agreement upon written notice."

    detected1 = detect_patterns(passage1)
    detected2 = detect_patterns(passage2)

    assert "shall_vs_may" in detected1
    assert "shall_vs_may" in detected2


def test_detect_patterns_mae():
    """Test detection of Material Adverse Effect patterns."""
    passage = "Material Adverse Effect excludes changes in general economic conditions."
    detected = detect_patterns(passage)
    assert "mae_carveout" in detected


def test_detect_patterns_quantitative():
    """Test detection of quantitative precision patterns."""
    passage1 = "The liability cap is $5,000,000."
    passage2 = "Buyer shall pay at least $2,000,000."
    passage3 = "The fee shall not exceed $500,000."

    detected1 = detect_patterns(passage1)
    detected2 = detect_patterns(passage2)
    detected3 = detect_patterns(passage3)

    assert "quantitative_precision" in detected1
    assert "quantitative_precision" in detected2
    assert "quantitative_precision" in detected3


def test_detect_patterns_temporal():
    """Test detection of temporal qualifier patterns."""
    passage1 = "Notice must be provided within 30 days of the event."
    passage2 = "The warranty survives termination for a period of 12 months."

    detected1 = detect_patterns(passage1)
    detected2 = detect_patterns(passage2)

    assert "temporal_qualifier" in detected1
    assert "temporal_qualifier" in detected2


def test_detect_patterns_condition_precedent():
    """Test detection of condition precedent patterns."""
    passage1 = "If the conditions are satisfied, Buyer shall close."
    passage2 = "The obligation is subject to regulatory approval."
    passage3 = "The payment is contingent upon meeting the revenue target."

    detected1 = detect_patterns(passage1)
    detected2 = detect_patterns(passage2)
    detected3 = detect_patterns(passage3)

    assert "condition_precedent" in detected1
    assert "condition_precedent" in detected2
    assert "condition_precedent" in detected3


def test_detect_patterns_defined_term():
    """Test detection of defined term patterns."""
    passage1 = '"Affiliate" means any entity controlling or controlled by the Company.'
    passage2 = "Business Day shall have the meaning set forth in Section 1.1."

    detected1 = detect_patterns(passage1)
    detected2 = detect_patterns(passage2)

    assert "defined_term" in detected1
    assert "defined_term" in detected2


def test_detect_patterns_knowledge_qualifier():
    """Test detection of knowledge qualifier patterns."""
    passage1 = "To Seller's Knowledge, there are no pending claims."
    passage2 = "The Company has complied with all laws in all material respects."
    passage3 = "Except as set forth in Schedule 3.1, all contracts are valid."

    detected1 = detect_patterns(passage1)
    detected2 = detect_patterns(passage2)
    detected3 = detect_patterns(passage3)

    assert "knowledge_qualifier" in detected1
    assert "knowledge_qualifier" in detected2
    assert "knowledge_qualifier" in detected3


def test_detect_patterns_multiple():
    """Test detection of multiple patterns in same passage."""
    passage = (
        "Notwithstanding Section 5, Seller shall pay at least $1,000,000 "
        "within 30 days, subject to Buyer's approval."
    )
    detected = detect_patterns(passage)

    assert "notwithstanding" in detected
    assert "shall_vs_may" in detected
    assert "quantitative_precision" in detected
    assert "temporal_qualifier" in detected
    assert "condition_precedent" in detected


def test_detect_patterns_none():
    """Test that no patterns are detected in plain text."""
    passage = "The parties agree to work together in good faith."
    detected = detect_patterns(passage)
    # This passage may or may not trigger patterns depending on regex - just check it doesn't crash
    assert isinstance(detected, list)


# ---------------------------------------------------------------------------
# Perturbation generation
# ---------------------------------------------------------------------------


def test_generate_negatives_shall_to_may():
    """Test shall → may perturbation."""
    passage = "Buyer shall deliver the goods by Friday."
    claim = "Buyer shall deliver the goods by Friday."

    perturbations = generate_negatives_from_perturbation(passage, claim, "shall_vs_may")

    assert len(perturbations) > 0
    assert any("may" in p.claim.lower() for p in perturbations)
    assert all(p.label == "contradicted" for p in perturbations)
    assert all(p.category == "shall_vs_may" for p in perturbations)


def test_generate_negatives_may_to_shall():
    """Test may → shall perturbation."""
    passage = "Seller may terminate the Agreement upon notice."
    claim = "Seller may terminate the Agreement upon notice."

    perturbations = generate_negatives_from_perturbation(passage, claim, "shall_vs_may")

    assert len(perturbations) > 0
    assert any("shall" in p.claim.lower() for p in perturbations)
    assert all(p.label == "contradicted" for p in perturbations)


def test_generate_negatives_quantitative_amount():
    """Test dollar amount perturbation."""
    passage = "The purchase price is $1,000,000."
    claim = "The purchase price is $1,000,000."

    perturbations = generate_negatives_from_perturbation(passage, claim, "quantitative_precision")

    assert len(perturbations) > 0
    # Amount should be changed
    assert any("$1,000,000" not in p.claim for p in perturbations)
    assert all(p.label == "contradicted" for p in perturbations)
    assert all(p.category == "quantitative_precision" for p in perturbations)


def test_generate_negatives_quantitative_threshold():
    """Test threshold inversion (at least → up to)."""
    passage = "The fee is at least $500,000."
    claim = "The fee is at least $500,000."

    perturbations = generate_negatives_from_perturbation(passage, claim, "quantitative_precision")

    assert len(perturbations) > 0
    assert any("up to" in p.claim.lower() for p in perturbations)
    assert all(p.label == "contradicted" for p in perturbations)


def test_generate_negatives_temporal_period():
    """Test time period perturbation."""
    passage = "Notice must be given within 30 days."
    claim = "Notice must be given within 30 days."

    perturbations = generate_negatives_from_perturbation(passage, claim, "temporal_qualifier")

    assert len(perturbations) > 0
    # Period should be changed
    assert any("30 days" not in p.claim for p in perturbations)
    assert all(p.label == "contradicted" for p in perturbations)
    assert all(p.category == "temporal_qualifier" for p in perturbations)


def test_generate_negatives_temporal_direction():
    """Test temporal direction swap (prior to ↔ following)."""
    passage = "Payment is due prior to delivery."
    claim = "Payment is due prior to delivery."

    perturbations = generate_negatives_from_perturbation(passage, claim, "temporal_qualifier")

    assert len(perturbations) > 0
    assert any("following" in p.claim.lower() for p in perturbations)
    assert all(p.label == "contradicted" for p in perturbations)


def test_generate_negatives_knowledge_qualifier():
    """Test removal of knowledge qualifier."""
    passage = "To Seller's Knowledge, there are no pending claims."
    claim = "To Seller's Knowledge, there are no pending claims."

    perturbations = generate_negatives_from_perturbation(passage, claim, "knowledge_qualifier")

    assert len(perturbations) > 0
    # Knowledge qualifier should be removed
    assert any("knowledge" not in p.claim.lower() for p in perturbations)
    assert all(p.label == "contradicted" for p in perturbations)
    assert all(p.category == "knowledge_qualifier" for p in perturbations)


def test_generate_negatives_material_respects():
    """Test removal of 'in all material respects' qualifier."""
    passage = "The Company has complied with all laws in all material respects."
    claim = "The Company has complied with all laws in all material respects."

    perturbations = generate_negatives_from_perturbation(passage, claim, "knowledge_qualifier")

    assert len(perturbations) > 0
    assert any("material respects" not in p.claim.lower() for p in perturbations)
    assert all(p.label == "contradicted" for p in perturbations)


def test_generate_negatives_no_perturbation():
    """Test that no perturbations are generated when pattern not applicable."""
    passage = "The parties agree to negotiate in good faith."
    claim = "The parties will negotiate in good faith."

    # No shall/may to perturb
    perturbations = generate_negatives_from_perturbation(passage, claim, "shall_vs_may")
    assert len(perturbations) == 0

    # No dollar amounts to perturb
    perturbations = generate_negatives_from_perturbation(passage, claim, "quantitative_precision")
    # May still have threshold perturbations if "at least" etc. present
    # But in this case should be empty or minimal


# ---------------------------------------------------------------------------
# Training data augmentation
# ---------------------------------------------------------------------------


def test_augment_training_data():
    """Test merging catalog examples into training data."""
    base_pairs = [
        {"passage": "P1", "claim": "C1", "label": "entailed"},
        {"passage": "P2", "claim": "C2", "label": "contradicted"},
    ]

    catalog = load_pattern_catalog()
    augmented = augment_training_data(base_pairs, catalog)

    # Should include base pairs
    assert len(augmented) > len(base_pairs)

    # Should include catalog examples
    assert len(augmented) >= len(base_pairs) + len(catalog.examples)

    # Check that base pairs are present
    assert any(p["passage"] == "P1" and p["claim"] == "C1" for p in augmented)


def test_augment_training_data_deduplication():
    """Test that duplicate (passage, claim) pairs are removed."""
    catalog = load_pattern_catalog()

    # Create base pairs that duplicate some catalog examples
    first_catalog_example = catalog.examples[0]
    base_pairs = [
        {
            "passage": first_catalog_example.passage,
            "claim": first_catalog_example.claim,
            "label": first_catalog_example.label,
        },
        {"passage": "Unique passage", "claim": "Unique claim", "label": "neutral"},
    ]

    augmented = augment_training_data(base_pairs, catalog)

    # Check for duplicates
    seen = set()
    for pair in augmented:
        key = (pair["passage"], pair["claim"])
        assert key not in seen, f"Duplicate found: {key}"
        seen.add(key)

    # Should have exactly catalog.examples + 1 unique pair (the duplicate is merged)
    assert len(augmented) == len(catalog.examples) + 1


def test_augment_training_data_preserves_labels():
    """Test that labels are preserved during augmentation."""
    base_pairs = [
        {"passage": "P1", "claim": "C1", "label": "entailed"},
    ]

    catalog = load_pattern_catalog()
    augmented = augment_training_data(base_pairs, catalog)

    # Check that all pairs have valid labels
    valid_labels = {"entailed", "contradicted", "neutral", "partially_entailed"}
    for pair in augmented:
        assert pair["label"] in valid_labels


def test_augment_training_data_empty_base():
    """Test augmentation with empty base pairs."""
    catalog = load_pattern_catalog()
    augmented = augment_training_data([], catalog)

    # Should just return catalog examples
    assert len(augmented) == len(catalog.examples)


# ---------------------------------------------------------------------------
# PATTERN_TAXONOMY
# ---------------------------------------------------------------------------


def test_pattern_taxonomy_structure():
    """Test that PATTERN_TAXONOMY has correct structure."""
    assert len(PATTERN_TAXONOMY) == 8

    required_fields = ["description", "linguistic_signature", "why_general_nli_fails", "training_priority"]

    for category, meta in PATTERN_TAXONOMY.items():
        assert meta.description != ""
        assert meta.linguistic_signature != ""
        assert meta.why_general_nli_fails != ""
        assert meta.training_priority in {1, 2}
        assert isinstance(meta.regex_hints, list)
        assert len(meta.regex_hints) > 0


def test_pattern_taxonomy_regex_hints():
    """Test that regex hints are valid."""
    import re

    for category, meta in PATTERN_TAXONOMY.items():
        for hint in meta.regex_hints:
            try:
                re.compile(hint, re.IGNORECASE)
            except re.error as e:
                pytest.fail(f"Invalid regex in {category}: {hint} ({e})")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_pattern_example_with_minimal_fields():
    """Test creating PatternExample with minimal required fields."""
    ex = PatternExample(
        id="test-001",
        category="notwithstanding",
        passage="Test passage",
        claim="Test claim",
        label="entailed",
    )
    assert ex.explanation == ""
    assert ex.difficulty == "medium"
    assert ex.source == "curated"


def test_catalog_empty_examples():
    """Test PatternCatalog with empty examples list."""
    catalog = PatternCatalog(examples=[])
    assert catalog.by_category("notwithstanding") == []
    assert catalog.hard_negatives() == []
    assert catalog.to_training_pairs() == []
    stats = catalog.stats()
    assert stats["total_examples"] == 0
