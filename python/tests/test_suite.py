"""Tests for the internal test suite schema and tooling."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from legalkit.eval.test_suite import (
    CitationRef,
    ClaimLabel,
    Difficulty,
    EntailmentLabel,
    GroundTruth,
    QueryType,
    SuiteStats,
    Triple,
    TripleMetadata,
    TripleStatus,
    Verification,
    ValidationResult,
    load_suite,
    save_suite,
    validate_suite,
)

SUITE_PATH = Path(__file__).parent.parent / "data/test_suite/triples.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_triple(
    id: str = "t-001",
    status: str = "approved",
    difficulty: str = "easy",
    query_type: str = "factual",
    answer: str = "The answer.",
    citations: list | None = None,
) -> Triple:
    return Triple(
        id=id,
        query="What is the governing law?",
        document={"file_path": "doc.txt", "doc_id": "doc"},
        ground_truth=GroundTruth(
            answer=answer,
            citations=citations or [CitationRef(section_id="§10.1", entailment="entailed")],
        ),
        verification=Verification(claims=[
            ClaimLabel(text="The answer.", is_correct=True, citations=["§10.1"], entailment="entailed"),
        ]),
        metadata=TripleMetadata(
            difficulty=difficulty,
            query_type=query_type,
            clause_type="governing_law",
            status=status,
        ),
    )


# ---------------------------------------------------------------------------
# Seed file integrity
# ---------------------------------------------------------------------------


class TestSeedFile:
    def test_seed_file_exists(self):
        assert SUITE_PATH.exists(), f"Seed file not found: {SUITE_PATH}"

    def test_seed_has_50_triples(self):
        with open(SUITE_PATH) as f:
            data = json.load(f)
        assert len(data["triples"]) == 50

    def test_seed_loads_cleanly(self):
        triples = load_suite(SUITE_PATH)
        assert len(triples) == 50

    def test_all_ids_unique(self):
        triples = load_suite(SUITE_PATH)
        ids = [t.id for t in triples]
        assert len(ids) == len(set(ids))

    def test_all_queries_non_empty(self):
        triples = load_suite(SUITE_PATH)
        for t in triples:
            assert t.query.strip(), f"{t.id} has empty query"

    def test_approved_triples_have_answers(self):
        triples = load_suite(SUITE_PATH)
        approved = [t for t in triples if t.is_approved]
        for t in approved:
            assert t.ground_truth.answer.strip(), f"{t.id} is approved but has no answer"

    def test_approved_triples_have_citations(self):
        triples = load_suite(SUITE_PATH)
        approved = [t for t in triples if t.is_approved]
        for t in approved:
            assert t.ground_truth.citations, f"{t.id} is approved but has no citations"

    def test_at_least_36_approved(self):
        """Seed should have at least 36 approved triples."""
        triples = load_suite(SUITE_PATH)
        approved = sum(1 for t in triples if t.is_approved)
        assert approved >= 36, f"Only {approved} approved triples"

    def test_all_difficulties_represented(self):
        triples = load_suite(SUITE_PATH)
        difficulties = {t.metadata.difficulty for t in triples}
        assert {"easy", "medium", "hard"}.issubset(difficulties)

    def test_all_query_types_represented(self):
        triples = load_suite(SUITE_PATH)
        qtypes = {t.metadata.query_type for t in triples}
        assert len(qtypes) >= 4

    def test_seed_passes_validation(self):
        triples = load_suite(SUITE_PATH)
        result = validate_suite(triples)
        assert result.is_valid, f"Seed validation errors: {result.errors}"

    def test_covers_key_clause_types(self):
        triples = load_suite(SUITE_PATH)
        clause_types = {t.metadata.clause_type for t in triples}
        expected = {"governing_law", "purchase_price", "indemnification",
                    "non_compete", "termination", "survival"}
        found = expected & clause_types
        assert len(found) >= 5, f"Missing clause types: {expected - clause_types}"

    def test_entailment_labels_valid(self):
        triples = load_suite(SUITE_PATH)
        valid = {e.value for e in EntailmentLabel}
        for t in triples:
            for cl in t.verification.claims:
                assert cl.entailment in valid, f"{t.id}: invalid entailment {cl.entailment!r}"


# ---------------------------------------------------------------------------
# Triple model
# ---------------------------------------------------------------------------


class TestTriple:
    def test_is_approved(self):
        t = _make_triple(status="approved")
        assert t.is_approved

    def test_not_approved(self):
        t = _make_triple(status="needs_review")
        assert not t.is_approved

    def test_needs_review(self):
        t = _make_triple(status="needs_review")
        assert t.needs_review

    def test_draft_needs_review(self):
        t = _make_triple(status="draft")
        assert t.needs_review

    def test_reviewed_not_needs_review(self):
        t = _make_triple(status="reviewed")
        assert not t.needs_review

    def test_to_dict_round_trip(self):
        t = _make_triple()
        d = t.to_dict()
        assert d["id"] == t.id
        assert d["query"] == t.query
        assert d["ground_truth"]["answer"] == t.ground_truth.answer
        assert d["metadata"]["status"] == t.metadata.status

    def test_status_property(self):
        t = _make_triple(status="approved")
        assert t.status == TripleStatus.APPROVED

    def test_difficulty_property(self):
        t = _make_triple(difficulty="hard")
        assert t.difficulty == Difficulty.HARD

    def test_query_type_property(self):
        t = _make_triple(query_type="nuance")
        assert t.query_type == QueryType.NUANCE


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        triples = [_make_triple(id=f"t-{i:03d}") for i in range(5)]
        p = tmp_path / "suite.json"
        save_suite(triples, p)
        loaded = load_suite(p)
        assert len(loaded) == 5
        assert loaded[0].id == "t-000"

    def test_round_trip_preserves_answer(self, tmp_path):
        t = _make_triple(answer="The governing law is Delaware.")
        p = tmp_path / "suite.json"
        save_suite([t], p)
        loaded = load_suite(p)
        assert loaded[0].ground_truth.answer == "The governing law is Delaware."

    def test_round_trip_preserves_citations(self, tmp_path):
        t = _make_triple(citations=[CitationRef("§10.1", "some quote", "entailed")])
        p = tmp_path / "suite.json"
        save_suite([t], p)
        loaded = load_suite(p)
        assert loaded[0].ground_truth.citations[0].section_id == "§10.1"
        assert loaded[0].ground_truth.citations[0].quote == "some quote"

    def test_round_trip_preserves_metadata(self, tmp_path):
        t = _make_triple(status="reviewed", difficulty="hard")
        p = tmp_path / "suite.json"
        save_suite([t], p)
        loaded = load_suite(p)
        assert loaded[0].metadata.status == "reviewed"
        assert loaded[0].metadata.difficulty == "hard"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_suite(tmp_path / "missing.json")

    def test_save_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "nested/dir/suite.json"
        save_suite([_make_triple()], p)
        assert p.exists()


# ---------------------------------------------------------------------------
# SuiteStats
# ---------------------------------------------------------------------------


class TestSuiteStats:
    def _make_triples(self) -> list[Triple]:
        return [
            _make_triple("t1", status="approved", difficulty="easy", query_type="factual"),
            _make_triple("t2", status="approved", difficulty="medium", query_type="nuance"),
            _make_triple("t3", status="needs_review", difficulty="hard", query_type="synthesis"),
            _make_triple("t4", status="reviewed", difficulty="easy", query_type="factual"),
        ]

    def test_total_count(self):
        stats = SuiteStats.from_triples(self._make_triples())
        assert stats.total == 4

    def test_approved_count(self):
        stats = SuiteStats.from_triples(self._make_triples())
        assert stats.approved_count == 2

    def test_needs_review_count(self):
        stats = SuiteStats.from_triples(self._make_triples())
        assert stats.needs_review_count == 1

    def test_by_difficulty(self):
        stats = SuiteStats.from_triples(self._make_triples())
        assert stats.by_difficulty["easy"] == 2
        assert stats.by_difficulty["hard"] == 1

    def test_by_query_type(self):
        stats = SuiteStats.from_triples(self._make_triples())
        assert stats.by_query_type["factual"] == 2

    def test_summary_returns_string(self):
        stats = SuiteStats.from_triples(self._make_triples())
        s = stats.summary()
        assert isinstance(s, str)
        assert "Total" in s

    def test_empty_suite(self):
        stats = SuiteStats.from_triples([])
        assert stats.total == 0
        assert stats.approved_count == 0


# ---------------------------------------------------------------------------
# validate_suite
# ---------------------------------------------------------------------------


class TestValidateSuite:
    def test_valid_suite_passes(self):
        triples = [_make_triple(f"t-{i}") for i in range(3)]
        result = validate_suite(triples)
        assert result.is_valid

    def test_duplicate_ids_fails(self):
        triples = [_make_triple("same-id"), _make_triple("same-id")]
        result = validate_suite(triples)
        assert not result.is_valid
        assert any("Duplicate" in e for e in result.errors)

    def test_invalid_status_fails(self):
        t = _make_triple(status="approved")
        t.metadata.status = "invalid_status"
        result = validate_suite([t])
        assert not result.is_valid

    def test_invalid_difficulty_fails(self):
        t = _make_triple()
        t.metadata.difficulty = "extreme"
        result = validate_suite([t])
        assert not result.is_valid

    def test_approved_with_empty_answer_fails(self):
        t = _make_triple(status="approved", answer="")
        t.ground_truth.answer = ""
        result = validate_suite([t])
        assert not result.is_valid

    def test_small_suite_warns(self):
        triples = [_make_triple(f"t-{i}") for i in range(3)]
        result = validate_suite(triples)
        assert any("50+" in w or "triples" in w.lower() for w in result.warnings)

    def test_few_approved_warns(self):
        triples = [_make_triple(f"t-{i}", status="needs_review") for i in range(55)]
        result = validate_suite(triples)
        assert any("approved" in w.lower() for w in result.warnings)

    def test_summary_pass(self):
        triples = [_make_triple(f"t-{i}") for i in range(55)]
        result = validate_suite(triples)
        assert "PASS" in result.summary() or "FAIL" in result.summary()
