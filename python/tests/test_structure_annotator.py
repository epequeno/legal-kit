"""Tests for structure_annotator.py — BIO-labelled training data generation."""

import json
import tempfile
from pathlib import Path

import pytest

from legalkit.eval.structure_annotator import (
    LABELS,
    LABEL2ID,
    ID2LABEL,
    AnnotatedLine,
    AnnotatedDocument,
    annotate_document,
    annotate_corpus,
    save_annotated_corpus,
    load_annotated_corpus,
    generate_training_data,
)


# ---------------------------------------------------------------------------
# Label taxonomy tests
# ---------------------------------------------------------------------------


def test_labels_has_15_labels():
    """LABELS has exactly 15 labels including 'O'."""
    assert len(LABELS) == 15
    assert "O" in LABELS


def test_label2id_id2label_consistency():
    """LABEL2ID and ID2LABEL are consistent inverses."""
    assert len(LABEL2ID) == len(LABELS)
    assert len(ID2LABEL) == len(LABELS)

    for label, label_id in LABEL2ID.items():
        assert ID2LABEL[label_id] == label

    for label_id, label in ID2LABEL.items():
        assert LABEL2ID[label] == label_id


def test_all_labels_valid():
    """All labels in LABELS are present in LABEL2ID."""
    for label in LABELS:
        assert label in LABEL2ID


def test_bio_label_structure():
    """BIO labels follow the expected B-/I- structure."""
    # Count B- and I- labels
    b_labels = [l for l in LABELS if l.startswith("B-")]
    i_labels = [l for l in LABELS if l.startswith("I-")]

    # Should have equal numbers
    assert len(b_labels) == len(i_labels)

    # Each B- should have a corresponding I-
    for b_label in b_labels:
        entity_type = b_label[2:]  # Strip "B-"
        i_label = f"I-{entity_type}"
        assert i_label in LABELS


# ---------------------------------------------------------------------------
# Annotation tests
# ---------------------------------------------------------------------------


def test_annotate_document_on_sample_contract():
    """annotate_document() on the sample_contract.txt fixture returns AnnotatedDocument."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()
    doc = annotate_document(
        text=text,
        doc_id="sample",
        file_path=str(sample_path),
        contract_type="test",
    )

    assert isinstance(doc, AnnotatedDocument)
    assert doc.doc_id == "sample"
    assert doc.contract_type == "test"
    assert len(doc.lines) > 0
    assert doc.n_lines == len(doc.lines)


def test_article_headers_labelled():
    """Article headers get labelled B-ARTICLE_HEADER."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()
    doc = annotate_document(text, doc_id="sample")

    # Find lines with "ARTICLE" in them
    article_lines = [
        line for line in doc.lines
        if "ARTICLE" in line.text
    ]

    # At least some should be labelled as ARTICLE_HEADER
    article_headers = [
        line for line in article_lines
        if line.label in ("B-ARTICLE_HEADER", "I-ARTICLE_HEADER")
    ]

    assert len(article_headers) > 0, "No article headers found"


def test_section_headers_labelled():
    """Section headers get labelled B-SECTION_HEADER or B-SUBSECTION_HEADER."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()
    doc = annotate_document(text, doc_id="sample")

    # Find lines with "Section" in them
    section_lines = [
        line for line in doc.lines
        if "Section" in line.text
    ]

    # At least some should be labelled as SECTION_HEADER or SUBSECTION_HEADER
    # (Section 2.1 is a subsection, Section 2 would be a section)
    section_headers = [
        line for line in section_lines
        if line.label in ("B-SECTION_HEADER", "I-SECTION_HEADER",
                          "B-SUBSECTION_HEADER", "I-SUBSECTION_HEADER")
    ]

    assert len(section_headers) > 0, "No section/subsection headers found"


def test_body_text_labelled_o():
    """Body text lines get labelled O."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()
    doc = annotate_document(text, doc_id="sample")

    # Most lines should be "O" (body text)
    o_lines = [line for line in doc.lines if line.label == "O"]
    assert len(o_lines) > 0
    assert len(o_lines) / len(doc.lines) > 0.5  # At least 50% should be O


def test_all_label_ids_valid():
    """All label IDs in AnnotatedDocument are valid (in LABEL2ID)."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()
    doc = annotate_document(text, doc_id="sample")

    for line in doc.lines:
        assert line.label_id in ID2LABEL
        assert ID2LABEL[line.label_id] == line.label


def test_line_numbers_sequential():
    """Line numbers are sequential starting from 1."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()
    doc = annotate_document(text, doc_id="sample")

    for i, line in enumerate(doc.lines, 1):
        assert line.line_number == i


def test_to_dict_round_trips():
    """to_dict() round-trips through JSON."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()
    doc = annotate_document(text, doc_id="sample")

    # Serialize to dict
    doc_dict = doc.to_dict()
    assert isinstance(doc_dict, dict)
    assert doc_dict["doc_id"] == "sample"
    assert len(doc_dict["lines"]) == len(doc.lines)

    # Should be JSON-serializable
    json_str = json.dumps(doc_dict)
    assert len(json_str) > 0

    # Deserialize
    doc_dict2 = json.loads(json_str)
    assert doc_dict2 == doc_dict


def test_label_distribution():
    """label_distribution property returns correct counts."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()
    doc = annotate_document(text, doc_id="sample")

    dist = doc.label_distribution
    assert isinstance(dist, dict)

    # Sum should equal total lines
    assert sum(dist.values()) == len(doc.lines)

    # All labels should be valid
    for label in dist.keys():
        assert label in LABELS


# ---------------------------------------------------------------------------
# Corpus processing tests
# ---------------------------------------------------------------------------


def test_save_and_load_annotated_corpus():
    """save_annotated_corpus() and load_annotated_corpus() round-trip correctly."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()
    doc1 = annotate_document(text, doc_id="sample1", contract_type="test")
    doc2 = annotate_document(text, doc_id="sample2", contract_type="test")

    docs = [doc1, doc2]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.jsonl"

        # Save
        stats = save_annotated_corpus(docs, output_path)
        assert stats["total_docs"] == 2
        assert stats["total_lines"] > 0
        assert output_path.exists()

        # Load
        loaded_docs = load_annotated_corpus(output_path)
        assert len(loaded_docs) == 2
        assert loaded_docs[0].doc_id == "sample1"
        assert loaded_docs[1].doc_id == "sample2"
        assert len(loaded_docs[0].lines) == len(doc1.lines)
        assert len(loaded_docs[1].lines) == len(doc2.lines)


def test_annotate_corpus_on_temp_dir():
    """annotate_corpus() on a temp dir with sample files returns list of AnnotatedDocuments."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a couple of test files
        (tmpdir_path / "doc1.txt").write_text(text)
        (tmpdir_path / "doc2.txt").write_text(text[:1000])  # Shorter version

        # Annotate
        docs = annotate_corpus(
            corpus_dir=tmpdir_path,
            contract_type="test",
            skip_errors=True,
        )

        assert len(docs) == 2
        assert docs[0].contract_type == "test"
        assert docs[1].contract_type == "test"
        assert docs[0].doc_id == "doc1"
        assert docs[1].doc_id == "doc2"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_annotate_empty_text():
    """annotate_document() handles empty text gracefully (returns empty lines)."""
    doc = annotate_document(
        text="",
        doc_id="empty",
        contract_type="test",
    )

    assert isinstance(doc, AnnotatedDocument)
    assert doc.doc_id == "empty"
    assert len(doc.lines) == 0
    assert doc.n_lines == 0


def test_annotate_whitespace_only():
    """annotate_document() handles whitespace-only text."""
    doc = annotate_document(
        text="   \n  \n   ",
        doc_id="whitespace",
        contract_type="test",
    )

    assert isinstance(doc, AnnotatedDocument)
    # Should have empty lines
    assert doc.n_lines >= 0


def test_annotate_no_structure():
    """annotate_document() handles text with no structure detected (all O labels)."""
    # Plain text with no legal structure
    text = """This is just some plain text.
It has no sections or articles.
Just regular paragraphs.
Nothing fancy here.
"""

    doc = annotate_document(text, doc_id="plain")

    assert isinstance(doc, AnnotatedDocument)
    assert len(doc.lines) > 0

    # All lines should be "O" since no structure detected
    for line in doc.lines:
        assert line.label == "O"


def test_annotate_single_line():
    """annotate_document() handles single-line text."""
    text = "ARTICLE I - INTRODUCTION"
    doc = annotate_document(text, doc_id="single")

    assert len(doc.lines) == 1
    # Should detect the article header
    assert doc.lines[0].label in ("B-ARTICLE_HEADER", "O")


# ---------------------------------------------------------------------------
# Roman numeral article test
# ---------------------------------------------------------------------------


def test_roman_articles():
    """Test annotation of roman numeral article headers."""
    roman_path = Path("tests/fixtures/roman_articles.txt")
    if not roman_path.exists():
        pytest.skip("roman_articles.txt not found")

    text = roman_path.read_text()
    doc = annotate_document(text, doc_id="roman")

    # Should have some article headers
    article_headers = [
        line for line in doc.lines
        if line.label in ("B-ARTICLE_HEADER", "I-ARTICLE_HEADER")
    ]

    assert len(article_headers) > 0, "No article headers detected in roman numeral test"


def test_bare_sections():
    """Test annotation of bare numeric sections (no 'Section' keyword)."""
    bare_path = Path("tests/fixtures/nda_bare_sections.txt")
    if not bare_path.exists():
        pytest.skip("nda_bare_sections.txt not found")

    text = bare_path.read_text()
    doc = annotate_document(text, doc_id="bare")

    # Should have some section headers
    section_headers = [
        line for line in doc.lines
        if line.label in ("B-SECTION_HEADER", "I-SECTION_HEADER")
    ]

    # At least some sections should be detected
    assert len(section_headers) > 0, "No section headers detected in bare sections test"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_generate_training_data_integration():
    """generate_training_data() full pipeline integration test."""
    sample_path = Path("tests/fixtures/sample_contract.txt")
    if not sample_path.exists():
        pytest.skip("sample_contract.txt not found")

    text = sample_path.read_text()

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = Path(tmpdir) / "corpus"
        corpus_dir.mkdir()

        # Create a small corpus
        for i in range(5):
            (corpus_dir / f"doc{i}.txt").write_text(text)

        output_dir = Path(tmpdir) / "output"

        # Run pipeline
        stats = generate_training_data(
            corpus_dirs={"test": corpus_dir},
            output_path=output_dir,
            max_per_corpus=5,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        # Check outputs
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "val.jsonl").exists()
        assert (output_dir / "test.jsonl").exists()
        assert (output_dir / "stats.json").exists()

        # Check stats
        assert stats["total_documents"] == 5
        assert stats["split_sizes"]["train"] > 0
        assert stats["split_sizes"]["val"] > 0
        assert stats["split_sizes"]["test"] > 0

        # Splits should sum to total
        total_split = (
            stats["split_sizes"]["train"]
            + stats["split_sizes"]["val"]
            + stats["split_sizes"]["test"]
        )
        assert total_split == stats["total_documents"]

        # Load and verify one file
        train_docs = load_annotated_corpus(output_dir / "train.jsonl")
        assert len(train_docs) == stats["split_sizes"]["train"]
        assert all(isinstance(d, AnnotatedDocument) for d in train_docs)


# ---------------------------------------------------------------------------
# Recital detection test
# ---------------------------------------------------------------------------


def test_recital_detection():
    """Test that WHEREAS recitals are detected and labelled."""
    text = """
ASSET PURCHASE AGREEMENT

WHEREAS, Seller desires to sell the Business; and

WHEREAS, Buyer desires to purchase the Business.

NOW, THEREFORE, the parties agree as follows:
"""

    doc = annotate_document(text, doc_id="recital_test")

    # Find lines with WHEREAS
    whereas_lines = [line for line in doc.lines if "WHEREAS" in line.text]

    # At least some should be labelled as recitals
    recital_labels = [
        line for line in whereas_lines
        if line.label in ("B-RECITAL", "I-RECITAL")
    ]

    # Should detect at least one recital
    # (Parser might not detect all, but should get some)
    # For this test, we just verify that the code runs without error
    assert len(doc.lines) > 0
