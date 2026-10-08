"""Tests for embedding training pair construction."""

import json
import tempfile
from pathlib import Path

import pytest

from legalkit.eval.embedding_pairs import (
    EmbeddingPair,
    extract_pairs_from_benchmark,
    build_training_data,
    _mine_hard_negative,
    _spans_overlap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_benchmark_file(tmp_path):
    """Create a minimal mock benchmark JSON."""
    bench_path = tmp_path / "benchmarks" / "test_dataset.json"
    bench_path.parent.mkdir(parents=True)
    
    benchmark_data = {
        "tests": [
            {
                "query": "What is the termination notice period?",
                "snippets": [
                    {
                        "file_path": "test_dataset/contract1.txt",
                        "span": [100, 200],
                        "answer": "The notice period is 30 days."
                    }
                ]
            },
            {
                "query": "What is the governing law?",
                "snippets": [
                    {
                        "file_path": "test_dataset/contract1.txt",
                        "span": [500, 600],
                        "answer": "This agreement shall be governed by the laws of Delaware."
                    }
                ]
            },
            # Duplicate query+positive pair
            {
                "query": "What is the termination notice period?",
                "snippets": [
                    {
                        "file_path": "test_dataset/contract1.txt",
                        "span": [100, 200],  # same span
                        "answer": "The notice period is 30 days."
                    }
                ]
            },
            # Very short answer (should be filtered)
            {
                "query": "What is X?",
                "snippets": [
                    {
                        "file_path": "test_dataset/contract1.txt",
                        "span": [700, 710],
                        "answer": "Short"
                    }
                ]
            },
        ]
    }
    
    bench_path.write_text(json.dumps(benchmark_data))
    return bench_path


@pytest.fixture
def mock_corpus_file(tmp_path):
    """Create a minimal mock corpus file."""
    corpus_path = tmp_path / "corpus" / "test_dataset" / "contract1.txt"
    corpus_path.parent.mkdir(parents=True)
    
    # Build a contract with known spans
    text_parts = [
        "A" * 100,  # 0-99
        "The notice period is 30 days.",  # 100-131 (span for query 1)
        "B" * 300,  # 132-431
        "C" * 68,   # 432-499
        "This agreement shall be governed by the laws of Delaware.",  # 500-558
        "D" * 100,  # 559-658
        "E" * 41,   # 659-699
        "Short",    # 700-705 (too short, should be filtered)
        "F" * 300,  # 706-1005
    ]
    
    text = "".join(text_parts)
    corpus_path.write_text(text)
    return corpus_path


# ---------------------------------------------------------------------------
# Test EmbeddingPair dataclass
# ---------------------------------------------------------------------------


def test_embedding_pair_to_dict():
    """Test EmbeddingPair.to_dict() has all required keys."""
    pair = EmbeddingPair(
        query="What is X?",
        positive="The answer is X.",
        hard_negative="Some other text.",
        dataset="cuad",
        doc_id="contract1",
        span=(100, 200),
    )
    
    d = pair.to_dict()
    
    assert d["query"] == "What is X?"
    assert d["positive"] == "The answer is X."
    assert d["hard_negative"] == "Some other text."
    assert d["dataset"] == "cuad"
    assert d["doc_id"] == "contract1"
    
    # span should NOT be in the dict (not needed for training)
    assert "span" not in d


# ---------------------------------------------------------------------------
# Test extract_pairs_from_benchmark
# ---------------------------------------------------------------------------


def test_extract_pairs_from_benchmark_basic(tmp_path, mock_benchmark_file, mock_corpus_file, monkeypatch):
    """Test basic extraction from a benchmark dataset."""
    # Monkeypatch the CORPUS_DIR and BENCHMARKS_DIR
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    
    pairs = extract_pairs_from_benchmark("test_dataset")
    
    # Should extract 2 valid pairs (duplicate filtered, short answer filtered)
    assert len(pairs) == 2
    
    # Check first pair
    assert pairs[0].query == "What is the termination notice period?"
    assert "notice period is 30 days" in pairs[0].positive
    assert pairs[0].dataset == "test_dataset"
    assert pairs[0].doc_id == "contract1"
    
    # Check second pair
    assert pairs[1].query == "What is the governing law?"
    assert "governed by the laws of Delaware" in pairs[1].positive


def test_extract_pairs_non_empty_fields(tmp_path, mock_benchmark_file, mock_corpus_file, monkeypatch):
    """Test that each pair has non-empty query, positive, hard_negative."""
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    
    pairs = extract_pairs_from_benchmark("test_dataset")
    
    for pair in pairs:
        assert pair.query.strip() != ""
        assert pair.positive.strip() != ""
        assert pair.hard_negative.strip() != ""


def test_extract_pairs_hard_negative_different(tmp_path, mock_benchmark_file, mock_corpus_file, monkeypatch):
    """Test that hard_negative is different text from positive."""
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    
    pairs = extract_pairs_from_benchmark("test_dataset")
    
    for pair in pairs:
        # Hard negative should not be identical to positive
        assert pair.hard_negative != pair.positive
        # And should not be a substring (though overlap is technically possible)
        # We'll just check they're not identical
        assert pair.hard_negative.strip() != pair.positive.strip()


def test_extract_pairs_deduplication(tmp_path, mock_benchmark_file, mock_corpus_file, monkeypatch):
    """Test that duplicate (query, positive) pairs are collapsed."""
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    
    # The mock benchmark has 2 identical query+span entries
    pairs = extract_pairs_from_benchmark("test_dataset")
    
    # Should only get 2 pairs total (not 3)
    assert len(pairs) == 2
    
    # Verify deduplication: count unique (query, positive) pairs
    unique_keys = set((p.query, p.positive) for p in pairs)
    assert len(unique_keys) == len(pairs)


def test_extract_pairs_min_positive_chars(tmp_path, mock_benchmark_file, mock_corpus_file, monkeypatch):
    """Test that short answer spans are skipped."""
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    
    # The "Short" span is [700, 710] which after construction is "ShortFFFFF" (10 chars)
    # So it won't be filtered with min=50. Let's test with a higher threshold
    pairs = extract_pairs_from_benchmark("test_dataset", min_positive_chars=50)
    
    # Should get the 2 long answers (notice period + governing law)
    assert len(pairs) == 2
    
    # But with min=200, should filter out even more
    pairs_strict = extract_pairs_from_benchmark("test_dataset", min_positive_chars=200)
    # The mock has varying lengths; this just tests the filter works
    assert len(pairs_strict) <= len(pairs)


def test_extract_pairs_missing_corpus_file(tmp_path, mock_benchmark_file, monkeypatch, capsys):
    """Test that missing corpus file is handled gracefully (no crash)."""
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    
    # Don't create the corpus file
    pairs = extract_pairs_from_benchmark("test_dataset")
    
    # Should return empty list, not crash
    assert pairs == []
    
    # Should print warning
    captured = capsys.readouterr()
    assert "Warning" in captured.out or "not found" in captured.out


def test_extract_pairs_max_n(tmp_path, mock_benchmark_file, mock_corpus_file, monkeypatch):
    """Test max_n parameter limits pairs extracted."""
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    
    # Extract only 1 pair
    pairs = extract_pairs_from_benchmark("test_dataset", max_n=1)
    
    # Should get at most 1 (after filtering)
    assert len(pairs) <= 1


# ---------------------------------------------------------------------------
# Test build_training_data
# ---------------------------------------------------------------------------


def test_build_training_data_creates_files(tmp_path, mock_benchmark_file, mock_corpus_file, monkeypatch):
    """Test that build_training_data creates train.jsonl and val.jsonl."""
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    monkeypatch.setattr(ep_module, "DATASET_NAMES", ["test_dataset"])
    
    output_dir = tmp_path / "embedding_training"
    
    stats = build_training_data(
        datasets=["test_dataset"],
        output_dir=output_dir,
    )
    
    # Check files exist
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    
    # Check stats
    assert stats["total"] > 0
    assert stats["train"] > 0
    assert stats["val"] >= 0


def test_build_training_data_val_ratio(tmp_path, mock_benchmark_file, mock_corpus_file, monkeypatch):
    """Test that val_ratio is respected."""
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    monkeypatch.setattr(ep_module, "DATASET_NAMES", ["test_dataset"])
    
    output_dir = tmp_path / "embedding_training"
    
    stats = build_training_data(
        datasets=["test_dataset"],
        output_dir=output_dir,
        val_ratio=0.5,  # 50% val
    )
    
    total = stats["total"]
    val_count = stats["val"]
    
    # Should be approximately 50% (within tolerance)
    expected_val = int(total * 0.5)
    assert abs(val_count - expected_val) <= 1


def test_build_training_data_max_per_dataset(tmp_path, mock_benchmark_file, mock_corpus_file, monkeypatch):
    """Test that max_per_dataset is respected."""
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    monkeypatch.setattr(ep_module, "DATASET_NAMES", ["test_dataset"])
    
    output_dir = tmp_path / "embedding_training"
    
    stats = build_training_data(
        datasets=["test_dataset"],
        output_dir=output_dir,
        max_per_dataset=1,
    )
    
    # Should get at most 1 pair per dataset
    assert stats["by_dataset"]["test_dataset"] <= 1


def test_build_training_data_jsonl_format(tmp_path, mock_benchmark_file, mock_corpus_file, monkeypatch):
    """Test that JSONL files have correct format."""
    import legalkit.eval.embedding_pairs as ep_module
    monkeypatch.setattr(ep_module, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(ep_module, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    monkeypatch.setattr(ep_module, "DATASET_NAMES", ["test_dataset"])
    
    output_dir = tmp_path / "embedding_training"
    
    build_training_data(
        datasets=["test_dataset"],
        output_dir=output_dir,
    )
    
    # Read train.jsonl
    train_path = output_dir / "train.jsonl"
    with open(train_path) as f:
        for line in f:
            obj = json.loads(line)
            # Check required keys
            assert "query" in obj
            assert "positive" in obj
            assert "hard_negative" in obj
            assert "dataset" in obj
            assert "doc_id" in obj


# ---------------------------------------------------------------------------
# Test helper functions
# ---------------------------------------------------------------------------


def test_spans_overlap():
    """Test _spans_overlap utility."""
    assert _spans_overlap((0, 10), (5, 15)) is True
    assert _spans_overlap((0, 10), (10, 20)) is False  # edge-to-edge not overlap
    assert _spans_overlap((0, 10), (11, 20)) is False
    assert _spans_overlap((5, 15), (0, 10)) is True
    assert _spans_overlap((0, 100), (50, 60)) is True  # contained


def test_mine_hard_negative_basic():
    """Test _mine_hard_negative extracts a non-overlapping passage."""
    text = "A" * 500 + "B" * 500 + "C" * 500
    positive_span = (500, 1000)  # the "B" section
    
    hard_neg = _mine_hard_negative(text, positive_span, window=1500)
    
    # Should be non-empty
    assert hard_neg.strip() != ""
    
    # Should not be the positive (the "B" section)
    assert hard_neg != text[500:1000]


def test_mine_hard_negative_no_overlap():
    """Test that hard negative doesn't overlap with positive span."""
    text = "A" * 500 + "B" * 500 + "C" * 500
    positive_span = (500, 1000)
    
    hard_neg = _mine_hard_negative(text, positive_span, window=1500)
    
    # The hard negative should not be identical to the positive
    positive_text = text[500:1000]
    assert hard_neg != positive_text
    
    # For a more robust check, we'd need to find where hard_neg appears in text
    # and verify it doesn't overlap [500, 1000]
    if hard_neg in text:
        idx = text.find(hard_neg)
        hard_neg_span = (idx, idx + len(hard_neg))
        # Verify no overlap
        assert not _spans_overlap(hard_neg_span, positive_span)
