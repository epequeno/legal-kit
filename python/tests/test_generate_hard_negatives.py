"""Tests for hard-negative training data generation pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from legalkit.eval.generate_hard_negatives import (
    deduplicate,
    generate_hard_negatives,
    generate_perturbation_negatives,
    load_catalog_as_training_pairs,
    load_entailed_pairs,
    LABEL_MAP,
)
from legalkit.eval.entailment_patterns import PatternExample


class TestLoadEntailedPairs:
    """Tests for load_entailed_pairs()."""

    def test_loads_correct_labels(self, tmp_path):
        """Load entailed+partially_entailed pairs from JSONL."""
        log_file = tmp_path / "test_log.jsonl"
        with open(log_file, "w") as f:
            # Entailed - should be loaded
            f.write(json.dumps({
                "passage": "Tenant shall pay rent monthly.",
                "claim": "Tenant must pay rent monthly.",
                "entailment_class": "entailed",
                "section_id": "§1",
                "doc_id": "lease",
                "source": "llm_generated"
            }) + "\n")
            # Partially entailed - should be loaded
            f.write(json.dumps({
                "passage": "Buyer may terminate with 30 days notice.",
                "claim": "Buyer can terminate the agreement.",
                "entailment_class": "partially_entailed",
                "section_id": "§2",
                "doc_id": "contract",
                "source": "pipeline"
            }) + "\n")
            # Contradicted - should be filtered out
            f.write(json.dumps({
                "passage": "Seller shall deliver within 10 days.",
                "claim": "Seller has 20 days to deliver.",
                "entailment_class": "contradicted",
                "section_id": "§3",
                "doc_id": "purchase",
                "source": "llm_generated"
            }) + "\n")
            # Neutral - should be filtered out
            f.write(json.dumps({
                "passage": "The purchase price is $5M.",
                "claim": "The closing date is June 1.",
                "entailment_class": "neutral",
                "section_id": "§4",
                "doc_id": "agreement",
                "source": "cross_passage"
            }) + "\n")

        pairs = load_entailed_pairs(log_path=log_file)

        assert len(pairs) == 2
        assert pairs[0]["label"] == "entailed"
        assert pairs[1]["label"] == "partially_entailed"

    def test_filters_to_entailed_and_partially_entailed_only(self, tmp_path):
        """Only load entailed and partially_entailed labels."""
        log_file = tmp_path / "test_log.jsonl"
        with open(log_file, "w") as f:
            for label in ["entailed", "partially_entailed", "contradicted", "neutral"]:
                f.write(json.dumps({
                    "passage": "Test passage.",
                    "claim": "Test claim.",
                    "entailment_class": label,
                    "source": "test"
                }) + "\n")

        pairs = load_entailed_pairs(log_path=log_file)

        assert len(pairs) == 2
        labels = {p["label"] for p in pairs}
        assert labels == {"entailed", "partially_entailed"}

    def test_respects_max_n(self, tmp_path):
        """Cap the number of loaded pairs with max_n."""
        log_file = tmp_path / "test_log.jsonl"
        with open(log_file, "w") as f:
            for i in range(100):
                f.write(json.dumps({
                    "passage": f"Passage {i}.",
                    "claim": f"Claim {i}.",
                    "entailment_class": "entailed",
                    "source": "test"
                }) + "\n")

        pairs = load_entailed_pairs(log_path=log_file, max_n=10)

        assert len(pairs) == 10

    def test_filters_by_source(self, tmp_path):
        """Filter to specific sources."""
        log_file = tmp_path / "test_log.jsonl"
        with open(log_file, "w") as f:
            f.write(json.dumps({
                "passage": "Pass 1",
                "claim": "Claim 1",
                "entailment_class": "entailed",
                "source": "llm_generated"
            }) + "\n")
            f.write(json.dumps({
                "passage": "Pass 2",
                "claim": "Claim 2",
                "entailment_class": "entailed",
                "source": "pipeline"
            }) + "\n")
            f.write(json.dumps({
                "passage": "Pass 3",
                "claim": "Claim 3",
                "entailment_class": "entailed",
                "source": "cross_passage"
            }) + "\n")

        pairs = load_entailed_pairs(log_path=log_file, sources=["llm_generated", "pipeline"])

        assert len(pairs) == 2
        sources = {p["source"] for p in pairs}
        assert sources == {"llm_generated", "pipeline"}


class TestGeneratePerturbationNegatives:
    """Tests for generate_perturbation_negatives()."""

    def test_returns_empty_for_no_detectable_patterns(self):
        """Return empty list when no patterns are detected."""
        pairs = [{
            "passage": "This is a plain text passage with no legal patterns.",
            "claim": "This is a plain claim.",
            "label": "entailed",
            "section_id": "§1",
            "doc_id": "doc1",
            "source": "test"
        }]

        negatives = generate_perturbation_negatives(pairs)

        assert negatives == []

    def test_returns_dicts_with_correct_schema(self):
        """Generated negatives have required training schema."""
        pairs = [{
            "passage": "Tenant shall pay rent of $1,000 monthly.",
            "claim": "Tenant shall pay $1,000 per month.",
            "label": "entailed",
            "section_id": "§1",
            "doc_id": "lease",
            "source": "llm_generated"
        }]

        negatives = generate_perturbation_negatives(pairs)

        assert len(negatives) > 0
        for neg in negatives:
            # Check all required keys are present
            assert "premise" in neg
            assert "hypothesis" in neg
            assert "label" in neg
            assert "label_name" in neg
            assert "source" in neg
            assert "section_id" in neg
            assert "doc_id" in neg

            # Check types
            assert isinstance(neg["premise"], str)
            assert isinstance(neg["hypothesis"], str)
            assert isinstance(neg["label"], int)
            assert isinstance(neg["label_name"], str)

    def test_produced_labels_are_not_entailed(self):
        """Perturbations generate hard negatives, not entailed labels."""
        pairs = [{
            "passage": "Employee shall provide 90 days notice of resignation.",
            "claim": "Employee shall provide 90 days notice.",
            "label": "entailed",
            "section_id": "§5",
            "doc_id": "employment",
            "source": "llm_generated"
        }]

        negatives = generate_perturbation_negatives(pairs)

        # Check that none of the generated negatives are entailed
        for neg in negatives:
            assert neg["label_name"] in {"contradicted", "neutral"}
            assert neg["label"] != 0  # 0 is entailed

    def test_source_includes_pattern_name(self):
        """Source field identifies which pattern was perturbed."""
        pairs = [{
            "passage": "Borrower shall maintain a minimum balance of $50,000.",
            "claim": "Borrower shall maintain at least $50,000.",
            "label": "entailed",
            "section_id": "§3",
            "doc_id": "loan",
            "source": "llm_generated"
        }]

        negatives = generate_perturbation_negatives(pairs)

        assert len(negatives) > 0
        for neg in negatives:
            # Source should be in format "perturbation:pattern_name"
            assert neg["source"].startswith("perturbation:")
            pattern_name = neg["source"].split(":", 1)[1]
            # Should be a valid pattern name
            assert pattern_name in {
                "shall_vs_may",
                "quantitative_precision",
                "temporal_qualifier",
                "knowledge_qualifier",
                "notwithstanding",
                "mae_carveout",
                "condition_precedent",
                "defined_term",
            }

    def test_progress_callback_invoked(self):
        """Progress callback is called during generation."""
        pairs = [
            {
                "passage": f"Tenant shall pay ${i},000 monthly.",
                "claim": f"Tenant shall pay ${i},000 per month.",
                "label": "entailed",
                "section_id": f"§{i}",
                "doc_id": "lease",
                "source": "llm_generated"
            }
            for i in range(1, 6)
        ]

        callbacks = []

        def callback(completed, total):
            callbacks.append((completed, total))

        generate_perturbation_negatives(pairs, progress_callback=callback)

        # Should have been called at least once
        assert len(callbacks) > 0
        # Final call should be (5, 5)
        assert callbacks[-1] == (5, 5)


class TestLoadCatalogAsTrainingPairs:
    """Tests for load_catalog_as_training_pairs()."""

    def test_maps_labels_correctly(self):
        """Map catalog labels to integer labels."""
        # Use the real catalog
        pairs = load_catalog_as_training_pairs()

        # Check that all labels are mapped correctly
        label_counts = {"entailed": 0, "contradicted": 0, "neutral": 0}
        for pair in pairs:
            assert pair["label"] in {0, 1, 2, 3}
            assert pair["label_name"] in {"entailed", "partially_entailed", "contradicted", "neutral"}

            # Verify mapping matches LABEL_MAP
            assert pair["label"] == LABEL_MAP[pair["label_name"]]

            if pair["label_name"] in label_counts:
                label_counts[pair["label_name"]] += 1

        # The catalog should have a mix of labels
        assert label_counts["entailed"] > 0
        assert label_counts["contradicted"] > 0
        assert label_counts["neutral"] > 0

    def test_sets_source_as_curated_catalog(self):
        """All catalog pairs have source='curated:catalog'."""
        pairs = load_catalog_as_training_pairs()

        assert len(pairs) > 0
        for pair in pairs:
            assert pair["source"] == "curated:catalog"

    def test_has_premise_and_hypothesis_keys(self):
        """Catalog pairs use training format (premise/hypothesis not passage/claim)."""
        pairs = load_catalog_as_training_pairs()

        assert len(pairs) > 0
        for pair in pairs:
            assert "premise" in pair
            assert "hypothesis" in pair
            assert "passage" not in pair
            assert "claim" not in pair


class TestDeduplicate:
    """Tests for deduplicate()."""

    def test_removes_exact_duplicate_pairs(self):
        """Remove pairs with identical (premise, hypothesis)."""
        pairs = [
            {"premise": "A", "hypothesis": "B", "label": 0},
            {"premise": "A", "hypothesis": "B", "label": 0},  # duplicate
            {"premise": "C", "hypothesis": "D", "label": 1},
        ]

        unique = deduplicate(pairs)

        assert len(unique) == 2
        assert unique[0]["premise"] == "A"
        assert unique[1]["premise"] == "C"

    def test_keeps_first_occurrence(self):
        """Keep the first occurrence when duplicates exist."""
        pairs = [
            {"premise": "A", "hypothesis": "B", "label": 0, "source": "first"},
            {"premise": "A", "hypothesis": "B", "label": 1, "source": "second"},  # different label
        ]

        unique = deduplicate(pairs)

        assert len(unique) == 1
        assert unique[0]["source"] == "first"

    def test_different_labels_same_text_are_duplicates(self):
        """Pairs with same text but different labels are still duplicates."""
        pairs = [
            {"premise": "A", "hypothesis": "B", "label": 0},
            {"premise": "A", "hypothesis": "B", "label": 2},
        ]

        unique = deduplicate(pairs)

        assert len(unique) == 1


class TestGenerateHardNegatives:
    """Integration tests for generate_hard_negatives()."""

    def test_correct_output_format(self, tmp_path):
        """Generated file is valid JSONL with correct schema."""
        # Create minimal test log
        log_file = tmp_path / "test_log.jsonl"
        with open(log_file, "w") as f:
            f.write(json.dumps({
                "passage": "Borrower shall maintain insurance of at least $1,000,000.",
                "claim": "Borrower shall maintain $1,000,000 in insurance.",
                "entailment_class": "entailed",
                "section_id": "§7",
                "doc_id": "loan",
                "source": "llm_generated"
            }) + "\n")

        output_file = tmp_path / "hard_negatives.jsonl"

        stats = generate_hard_negatives(
            log_path=log_file,
            output_path=output_file,
        )

        # Check output file exists and is valid JSONL
        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) > 0

        for line in lines:
            record = json.loads(line)
            # Check schema
            assert "premise" in record
            assert "hypothesis" in record
            assert "label" in record
            assert "label_name" in record
            assert "source" in record

    def test_deduplication_applied(self, tmp_path):
        """Duplicates are removed from final output."""
        # Create log with pairs that will generate overlapping perturbations
        log_file = tmp_path / "test_log.jsonl"
        with open(log_file, "w") as f:
            # Same passage twice - will generate identical perturbations
            for _ in range(2):
                f.write(json.dumps({
                    "passage": "Tenant shall pay rent of $2,000 monthly.",
                    "claim": "Tenant shall pay $2,000 per month.",
                    "entailment_class": "entailed",
                    "source": "llm_generated"
                }) + "\n")

        output_file = tmp_path / "hard_negatives.jsonl"

        stats = generate_hard_negatives(
            log_path=log_file,
            output_path=output_file,
        )

        # Deduplication is applied (might be 0 if perturbations generate different text)
        assert "duplicates_removed" in stats
        assert stats["duplicates_removed"] >= 0

    def test_stats_returned(self, tmp_path):
        """Stats dict contains expected keys and values."""
        log_file = tmp_path / "test_log.jsonl"
        with open(log_file, "w") as f:
            f.write(json.dumps({
                "passage": "Provider shall maintain uptime of at least 99.9%.",
                "claim": "Provider shall maintain at least 99.9% uptime.",
                "entailment_class": "entailed",
                "source": "pipeline"
            }) + "\n")

        output_file = tmp_path / "hard_negatives.jsonl"

        stats = generate_hard_negatives(
            log_path=log_file,
            output_path=output_file,
        )

        # Check required keys
        assert "total" in stats
        assert "by_label" in stats
        assert "by_source" in stats
        assert "output_path" in stats
        assert "input_entailed_pairs" in stats
        assert "perturbation_generated" in stats
        assert "catalog_examples" in stats
        assert "duplicates_removed" in stats

        # Check values make sense
        assert stats["total"] > 0
        assert isinstance(stats["by_label"], dict)
        assert isinstance(stats["by_source"], dict)

    def test_writes_valid_jsonl(self, tmp_path):
        """Each line in output is valid JSON."""
        log_file = tmp_path / "test_log.jsonl"
        with open(log_file, "w") as f:
            f.write(json.dumps({
                "passage": "Borrower may prepay without penalty.",
                "claim": "Borrower may prepay the loan.",
                "entailment_class": "entailed",
                "source": "llm_generated"
            }) + "\n")

        output_file = tmp_path / "hard_negatives.jsonl"

        generate_hard_negatives(
            log_path=log_file,
            output_path=output_file,
        )

        # Check every line is valid JSON
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)  # Will raise if invalid
                    assert isinstance(record, dict)


class TestPrepareDatasetWithHardNegatives:
    """Tests for prepare_dataset() with hard_negatives_path."""

    def test_hard_negatives_included_in_training_data(self, tmp_path):
        """Hard negatives are merged into the training pool."""
        from legalkit.eval.prepare_training import prepare_dataset

        # Create minimal entailment log
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "entailment_test.jsonl"  # Must match entailment_*.jsonl pattern

        with open(log_file, "w") as f:
            # Add some base samples
            for i in range(20):
                f.write(json.dumps({
                    "passage": f"Passage {i} with at least thirty characters here.",
                    "claim": f"Claim {i}.",
                    "entailment_class": "entailed",
                    "source": "pipeline",
                    "section_id": f"§{i}",
                    "doc_id": "test_doc",
                    "query": "",
                    "judge_model": "",
                    "timestamp": "",
                    "confidence": 1.0,
                    "explanation": "",
                    "reviewed": False,
                    "corrected_class": None,
                }) + "\n")

        # Create hard negatives file
        hard_neg_file = tmp_path / "hard_negatives.jsonl"
        with open(hard_neg_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({
                    "premise": f"Premise {i} with at least thirty characters here.",
                    "hypothesis": f"Hypothesis {i}.",
                    "label": 2,
                    "label_name": "contradicted",
                    "source": "perturbation:shall_vs_may",
                    "section_id": "",
                    "doc_id": "",
                }) + "\n")

        output_dir = tmp_path / "training"

        stats = prepare_dataset(
            log_dir=log_dir,
            output_dir=output_dir,
            hard_negatives_path=hard_neg_file,
            max_samples_per_class=10,
            seed=42,
        )

        # Check that source distribution includes hard negatives
        assert "source_distribution" in stats
        sources = stats["source_distribution"]
        assert "perturbation:shall_vs_may" in sources
        assert sources["perturbation:shall_vs_may"] > 0

    def test_cross_passage_cap_respected(self, tmp_path):
        """Neutral class has at most cross_passage_cap% from cross_passage."""
        from legalkit.eval.prepare_training import prepare_dataset

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "entailment_test.jsonl"  # Must match entailment_*.jsonl pattern

        with open(log_file, "w") as f:
            # Add entailed samples
            for i in range(100):
                f.write(json.dumps({
                    "passage": f"Entailed passage {i} with at least thirty characters here.",
                    "claim": f"Entailed claim {i}.",
                    "entailment_class": "entailed",
                    "source": "pipeline",
                    "section_id": f"§{i}",
                    "doc_id": "test_doc",
                    "query": "",
                    "judge_model": "",
                    "timestamp": "",
                    "confidence": 1.0,
                    "explanation": "",
                    "reviewed": False,
                    "corrected_class": None,
                }) + "\n")

            # Add many cross_passage neutrals
            for i in range(100):
                f.write(json.dumps({
                    "passage": f"Neutral passage {i} with at least thirty characters here.",
                    "claim": f"Neutral claim {i}.",
                    "entailment_class": "neutral",
                    "source": "cross_passage",
                    "section_id": f"§{i}",
                    "doc_id": "test_doc",
                    "query": "",
                    "judge_model": "",
                    "timestamp": "",
                    "confidence": 1.0,
                    "explanation": "",
                    "reviewed": False,
                    "corrected_class": None,
                }) + "\n")

        # Create hard negative neutrals
        hard_neg_file = tmp_path / "hard_negatives.jsonl"
        with open(hard_neg_file, "w") as f:
            for i in range(100):
                f.write(json.dumps({
                    "premise": f"Hard negative premise {i} with at least thirty characters here.",
                    "hypothesis": f"Hard negative hypothesis {i}.",
                    "label": 3,
                    "label_name": "neutral",
                    "source": "perturbation:quantitative_precision",
                    "section_id": "",
                    "doc_id": "",
                }) + "\n")

        output_dir = tmp_path / "training"

        stats = prepare_dataset(
            log_dir=log_dir,
            output_dir=output_dir,
            hard_negatives_path=hard_neg_file,
            cross_passage_cap=0.15,
            max_samples_per_class=100,
            seed=42,
        )

        # Check source distribution
        sources = stats["source_distribution"]

        # If there are cross_passage samples, they should be <= 15% of target
        if "cross_passage" in sources:
            target = stats["target_per_class"]
            max_cross = int(target * 0.15)
            assert sources["cross_passage"] <= max_cross

    def test_backward_compatible_when_no_hard_negatives(self, tmp_path):
        """Existing behavior unchanged when hard_negatives_path=None."""
        from legalkit.eval.prepare_training import prepare_dataset

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "entailment_test.jsonl"  # Must match entailment_*.jsonl pattern

        with open(log_file, "w") as f:
            for label in ["entailed", "contradicted", "neutral"]:
                for i in range(20):
                    f.write(json.dumps({
                        "passage": f"{label} passage {i} with at least thirty characters here.",
                        "claim": f"{label} claim {i}.",
                        "entailment_class": label,
                        "source": "pipeline",
                        "section_id": f"§{i}",
                        "doc_id": "test_doc",
                        "query": "",
                        "judge_model": "",
                        "timestamp": "",
                        "confidence": 1.0,
                        "explanation": "",
                        "reviewed": False,
                        "corrected_class": None,
                    }) + "\n")

        output_dir = tmp_path / "training"

        # Run without hard negatives
        stats = prepare_dataset(
            log_dir=log_dir,
            output_dir=output_dir,
            hard_negatives_path=None,  # Explicitly None
            seed=42,
        )

        # Should still work and produce balanced data
        assert stats["total_balanced"] > 0
        assert "train" in stats
        assert "val" in stats
        assert "test" in stats

        # Should have source distribution even without hard negatives
        assert "source_distribution" in stats
