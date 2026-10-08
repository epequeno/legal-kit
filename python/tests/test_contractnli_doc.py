"""Tests for ContractNLI document-level NLI evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from legalkit.eval.contractnli_doc import (
    ContractNLIExample,
    ContractNLIPrediction,
    ContractNLIReport,
    load_contractnli,
    run_contractnli_eval,
    _predict_label,
    _compute_retrieval_recall,
    _normalize_label,
    _parse_hf_row,
)
from legalkit.models.citation import EntailmentClass


# ---------------------------------------------------------------------------
# Test data models
# ---------------------------------------------------------------------------


def test_contractnli_example_instantiation():
    """ContractNLIExample instantiates correctly."""
    ex = ContractNLIExample(
        doc_id="test_doc",
        document_text="This is a test NDA.",
        hypothesis="The agreement is governed by Delaware law.",
        gold_label="Entailed",
        gold_spans=[(10, 20)],
        hypothesis_id="hyp_1",
    )
    assert ex.doc_id == "test_doc"
    assert ex.gold_label == "Entailed"
    assert len(ex.gold_spans) == 1


def test_contractnli_prediction_instantiation():
    """ContractNLIPrediction instantiates correctly."""
    ex = ContractNLIExample(
        doc_id="test",
        document_text="text",
        hypothesis="hyp",
        gold_label="Entailed",
        gold_spans=[],
    )
    pred = ContractNLIPrediction(
        example=ex,
        predicted_label="Entailed",
        confidence=0.9,
        retrieved_spans=[(0, 10)],
        entailment_score=0.85,
        retrieval_recall=0.8,
        latency_ms=150.0,
    )
    assert pred.predicted_label == "Entailed"
    assert pred.confidence == 0.9


# ---------------------------------------------------------------------------
# Test ContractNLIReport metrics
# ---------------------------------------------------------------------------


def test_report_accuracy():
    """ContractNLIReport.accuracy() correct with known data."""
    ex1 = ContractNLIExample("d1", "text", "h1", "Entailed", [])
    ex2 = ContractNLIExample("d2", "text", "h2", "NotMentioned", [])
    ex3 = ContractNLIExample("d3", "text", "h3", "Contradicted", [])

    predictions = [
        ContractNLIPrediction(ex1, "Entailed", 0.9, [], 0.9, 0.0, 100),  # correct
        ContractNLIPrediction(ex2, "NotMentioned", 0.8, [], 0.8, 0.0, 100),  # correct
        ContractNLIPrediction(ex3, "Entailed", 0.7, [], 0.7, 0.0, 100),  # wrong
    ]

    report = ContractNLIReport(model_name="test", predictions=predictions)
    assert report.accuracy() == 2 / 3


def test_report_per_label_accuracy():
    """ContractNLIReport.per_label_accuracy() correct per-class."""
    ex1 = ContractNLIExample("d1", "text", "h1", "Entailed", [])
    ex2 = ContractNLIExample("d2", "text", "h2", "Entailed", [])
    ex3 = ContractNLIExample("d3", "text", "h3", "NotMentioned", [])

    predictions = [
        ContractNLIPrediction(ex1, "Entailed", 0.9, [], 0.9, 0.0, 100),  # correct
        ContractNLIPrediction(ex2, "NotMentioned", 0.8, [], 0.8, 0.0, 100),  # wrong
        ContractNLIPrediction(ex3, "NotMentioned", 0.8, [], 0.8, 0.0, 100),  # correct
    ]

    report = ContractNLIReport(model_name="test", predictions=predictions)
    per_label = report.per_label_accuracy()

    assert per_label["Entailed"] == 0.5  # 1/2 correct
    assert per_label["NotMentioned"] == 1.0  # 1/1 correct


def test_report_confusion_matrix():
    """ContractNLIReport.confusion_matrix() has correct structure and counts."""
    ex1 = ContractNLIExample("d1", "text", "h1", "Entailed", [])
    ex2 = ContractNLIExample("d2", "text", "h2", "NotMentioned", [])
    ex3 = ContractNLIExample("d3", "text", "h3", "Contradicted", [])

    predictions = [
        ContractNLIPrediction(ex1, "NotMentioned", 0.9, [], 0.9, 0.0, 100),
        ContractNLIPrediction(ex2, "NotMentioned", 0.8, [], 0.8, 0.0, 100),
        ContractNLIPrediction(ex3, "Contradicted", 0.7, [], 0.7, 0.0, 100),
    ]

    report = ContractNLIReport(model_name="test", predictions=predictions)
    cm = report.confusion_matrix()

    # Check structure
    assert "Entailed" in cm
    assert "NotMentioned" in cm
    assert "Contradicted" in cm

    # Check counts
    assert cm["Entailed"]["NotMentioned"] == 1
    assert cm["NotMentioned"]["NotMentioned"] == 1
    assert cm["Contradicted"]["Contradicted"] == 1
    assert cm["Entailed"]["Entailed"] == 0


def test_report_retrieval_recall():
    """ContractNLIReport.retrieval_recall() only counts Entailed/Contradicted examples."""
    ex1 = ContractNLIExample("d1", "text", "h1", "Entailed", [(0, 10)])
    ex2 = ContractNLIExample("d2", "text", "h2", "Contradicted", [(5, 15)])
    ex3 = ContractNLIExample("d3", "text", "h3", "NotMentioned", [])

    predictions = [
        ContractNLIPrediction(
            ex1, "Entailed", 0.9, [(0, 10)], 0.9, retrieval_recall=1.0, latency_ms=100
        ),
        ContractNLIPrediction(
            ex2, "Contradicted", 0.8, [(5, 10)], 0.8, retrieval_recall=0.5, latency_ms=100
        ),
        ContractNLIPrediction(
            ex3, "NotMentioned", 0.7, [], 0.7, retrieval_recall=0.0, latency_ms=100
        ),
    ]

    report = ContractNLIReport(model_name="test", predictions=predictions)
    # Average of 1.0 and 0.5 (NotMentioned is excluded)
    assert report.retrieval_recall() == 0.75


def test_report_not_mentioned_precision():
    """ContractNLIReport.not_mentioned_precision() computes correctly."""
    ex1 = ContractNLIExample("d1", "text", "h1", "NotMentioned", [])
    ex2 = ContractNLIExample("d2", "text", "h2", "NotMentioned", [])
    ex3 = ContractNLIExample("d3", "text", "h3", "NotMentioned", [])

    predictions = [
        ContractNLIPrediction(ex1, "NotMentioned", 0.9, [], 0.9, 0.0, 100),  # correct
        ContractNLIPrediction(ex2, "Entailed", 0.8, [(0, 10)], 0.8, 0.0, 100),  # wrong
        ContractNLIPrediction(ex3, "NotMentioned", 0.7, [], 0.7, 0.0, 100),  # correct
    ]

    report = ContractNLIReport(model_name="test", predictions=predictions)
    assert report.not_mentioned_precision() == 2 / 3


def test_report_to_dict():
    """ContractNLIReport.to_dict() is JSON-serialisable."""
    ex = ContractNLIExample("d1", "text", "h1", "Entailed", [(0, 10)])
    pred = ContractNLIPrediction(ex, "Entailed", 0.9, [(0, 10)], 0.9, 1.0, 100)

    report = ContractNLIReport(model_name="test-model", predictions=[pred])
    data = report.to_dict()

    # Check structure
    assert data["model_name"] == "test-model"
    assert data["n_predictions"] == 1
    assert "accuracy" in data
    assert "per_label_accuracy" in data
    assert "confusion_matrix" in data

    # Ensure it's JSON-serializable
    json_str = json.dumps(data)
    assert len(json_str) > 0


# ---------------------------------------------------------------------------
# Test label prediction
# ---------------------------------------------------------------------------


def test_predict_label_entailed_high_conf():
    """_predict_label maps entailed + high conf → Entailed."""
    label, conf = _predict_label("entailed", 0.85, 0.6, 0.3)
    assert label == "Entailed"
    assert conf == 0.85


def test_predict_label_partially_entailed():
    """_predict_label maps partially_entailed + high conf → Entailed."""
    label, conf = _predict_label("partially_entailed", 0.75, 0.6, 0.3)
    assert label == "Entailed"
    assert conf == 0.75


def test_predict_label_contradicted():
    """_predict_label maps contradicted + high conf → Contradicted."""
    label, conf = _predict_label("contradicted", 0.8, 0.6, 0.3)
    assert label == "Contradicted"
    assert conf == 0.8


def test_predict_label_neutral_low_conf():
    """_predict_label maps neutral or low conf → NotMentioned."""
    label, conf = _predict_label("neutral", 0.5, 0.6, 0.3)
    assert label == "NotMentioned"

    label2, conf2 = _predict_label("entailed", 0.2, 0.6, 0.3)
    assert label2 == "NotMentioned"


def test_predict_label_enum():
    """_predict_label handles EntailmentClass enum."""
    label, conf = _predict_label(EntailmentClass.ENTAILED, 0.85, 0.6, 0.3)
    assert label == "Entailed"


# ---------------------------------------------------------------------------
# Test retrieval recall computation
# ---------------------------------------------------------------------------


def test_compute_retrieval_recall_perfect():
    """_compute_retrieval_recall returns 1.0 for perfect coverage."""
    retrieved = [(0, 10), (20, 30)]
    gold = [(0, 10), (20, 30)]
    assert _compute_retrieval_recall(retrieved, gold) == 1.0


def test_compute_retrieval_recall_partial():
    """_compute_retrieval_recall computes partial overlap correctly."""
    retrieved = [(0, 5)]  # only covers half of first gold span
    gold = [(0, 10)]
    assert _compute_retrieval_recall(retrieved, gold) == 0.5


def test_compute_retrieval_recall_no_gold():
    """_compute_retrieval_recall returns 0 when no gold spans."""
    retrieved = [(0, 10)]
    gold = []
    assert _compute_retrieval_recall(retrieved, gold) == 0.0


def test_compute_retrieval_recall_no_overlap():
    """_compute_retrieval_recall returns 0 when no overlap."""
    retrieved = [(100, 110)]
    gold = [(0, 10)]
    assert _compute_retrieval_recall(retrieved, gold) == 0.0


# ---------------------------------------------------------------------------
# Test data loading
# ---------------------------------------------------------------------------


def test_load_contractnli_from_cache(tmp_path):
    """load_contractnli loads from cache when present."""
    cache_dir = tmp_path / "contractnli"
    cache_dir.mkdir()
    cache_file = cache_dir / "test.json"

    # Write cache file
    data = [
        {
            "doc_id": "doc1",
            "document_text": "This is an NDA.",
            "hypothesis": "The agreement is confidential.",
            "gold_label": "Entailed",
            "gold_spans": [[0, 10]],
            "hypothesis_id": "hyp1",
        }
    ]
    cache_file.write_text(json.dumps(data))

    # Load
    examples = load_contractnli(split="test", data_dir=cache_dir)

    assert len(examples) == 1
    assert examples[0].doc_id == "doc1"
    assert examples[0].gold_label == "Entailed"
    assert examples[0].gold_spans == [(0, 10)]


def test_load_contractnli_fallback_to_hf(tmp_path):
    """load_contractnli falls back to HF when no cache (mock the HF call)."""
    cache_dir = tmp_path / "contractnli"

    mock_data = [
        {
            "document": "NDA text",
            "hypothesis": "Hyp text",
            "annotations": [{"type": "Entailed", "spans": [[5, 15]]}],
            "doc_id": "doc1",
        }
    ]

    # Mock dataset needs to have select() method and be iterable
    # Use MagicMock to support __iter__
    mock_dataset = MagicMock()
    mock_dataset.select.return_value = mock_data
    mock_dataset.__iter__.return_value = iter(mock_data)
    mock_dataset.__len__.return_value = len(mock_data)

    with patch("datasets.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset

        examples = load_contractnli(split="test", max_n=1, data_dir=cache_dir)

        assert len(examples) == 1
        assert examples[0].doc_id == "doc1"
        assert examples[0].gold_label == "Entailed"


def test_load_contractnli_raises_when_both_fail(tmp_path):
    """load_contractnli raises FileNotFoundError when both fail."""
    cache_dir = tmp_path / "contractnli"

    with patch("datasets.load_dataset") as mock_load:
        mock_load.side_effect = Exception("HF error")

        with pytest.raises(FileNotFoundError) as exc_info:
            load_contractnli(split="test", data_dir=cache_dir)

        assert "contractnli-eval --download" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test HF row parsing
# ---------------------------------------------------------------------------


def test_parse_hf_row_format1():
    """_parse_hf_row handles document + annotations format."""
    row = {
        "document": "This is the NDA text.",
        "hypothesis": "The agreement is mutual.",
        "annotations": [{"type": "Contradicted", "spans": [[0, 10], [15, 20]]}],
        "doc_id": "nda_001",
    }
    ex = _parse_hf_row(row, 0)

    assert ex.doc_id == "nda_001"
    assert ex.document_text == "This is the NDA text."
    assert ex.hypothesis == "The agreement is mutual."
    assert ex.gold_label == "Contradicted"
    assert ex.gold_spans == [(0, 10), (15, 20)]


def test_parse_hf_row_format2():
    """_parse_hf_row handles text + hypothesis + label format."""
    row = {
        "text": "Contract text here",
        "hypothesis": "Governed by NY law",
        "label": "NotMentioned",
        "id": "test_123",
    }
    ex = _parse_hf_row(row, 5)

    assert ex.doc_id == "test_123"
    assert ex.document_text == "Contract text here"
    assert ex.hypothesis == "Governed by NY law"
    assert ex.gold_label == "NotMentioned"


def test_normalize_label():
    """_normalize_label handles various formats."""
    assert _normalize_label("entailed") == "Entailed"
    assert _normalize_label("ENTAILED") == "Entailed"
    assert _normalize_label("Contradicted") == "Contradicted"
    assert _normalize_label("not_mentioned") == "NotMentioned"
    assert _normalize_label(0) == "Entailed"
    assert _normalize_label(1) == "NotMentioned"
    assert _normalize_label(2) == "Contradicted"


# ---------------------------------------------------------------------------
# Test run_contractnli_eval
# ---------------------------------------------------------------------------


def test_run_contractnli_eval_with_mocks():
    """run_contractnli_eval with fully mocked pipeline."""
    # Create test examples
    ex1 = ContractNLIExample(
        doc_id="doc1",
        document_text="The parties agree to confidentiality.",
        hypothesis="The agreement requires confidentiality.",
        gold_label="Entailed",
        gold_spans=[(0, 20)],
    )

    # Mock LLM
    mock_llm = Mock()
    mock_llm.config.generation_model = "test-model"

    # Mock pipeline functions
    with patch("legalkit.pipeline.parse.parse_document") as mock_parse, \
         patch("legalkit.pipeline.chunk.chunk_document") as mock_chunk, \
         patch("legalkit.pipeline.embed.embed_texts") as mock_embed, \
         patch("legalkit.pipeline.index.DocumentIndex") as mock_index_class, \
         patch("legalkit.pipeline.verify._check_entailment_llm") as mock_entail:

        # Setup mocks
        mock_root = Mock()
        mock_parse.return_value = mock_root

        mock_chunk_obj = Mock()
        mock_chunk_obj.id = "chunk1"
        mock_chunk_obj.node_id = "doc1_sec1"
        mock_chunk_obj.text = "The parties agree to confidentiality."
        mock_chunk_obj.char_start = 0
        mock_chunk_obj.char_end = 20
        mock_chunk_obj.embedding_text = "The parties agree to confidentiality."
        mock_chunk.return_value = [mock_chunk_obj]

        mock_embed.return_value = [[0.1] * 384]  # fake embeddings

        mock_index = Mock()
        mock_index.search_hybrid.return_value = [(0, 0.9, 0.0, 0.0)]
        mock_index.chunks = [mock_chunk_obj]
        mock_index_class.return_value = mock_index

        # Mock entailment check
        mock_entail.return_value = (EntailmentClass.ENTAILED, 0.85, "Entailed")

        # Run
        report = run_contractnli_eval(
            examples=[ex1],
            llm=mock_llm,
            top_k=10,
        )

        # Verify
        assert len(report.predictions) == 1
        assert report.predictions[0].predicted_label == "Entailed"
        assert report.predictions[0].confidence == 0.85


def test_run_contractnli_eval_handles_errors():
    """run_contractnli_eval handles errors gracefully."""
    ex1 = ContractNLIExample(
        doc_id="doc1",
        document_text="text",
        hypothesis="hyp",
        gold_label="Entailed",
        gold_spans=[],
    )

    mock_llm = Mock()
    mock_llm.config.generation_model = "test-model"

    # Mock parse to raise error
    with patch("legalkit.pipeline.parse.parse_document") as mock_parse, \
         patch("legalkit.pipeline.chunk.chunk_document") as mock_chunk:

        mock_parse.side_effect = Exception("Parse error")
        mock_chunk.return_value = []  # fallback to empty chunks

        # This should not crash, but return NotMentioned predictions
        report = run_contractnli_eval(
            examples=[ex1],
            llm=mock_llm,
            top_k=10,
        )

        assert len(report.predictions) == 1
        assert report.predictions[0].predicted_label == "NotMentioned"


def test_run_contractnli_eval_caches_documents():
    """run_contractnli_eval parses each doc once even for multiple hypotheses."""
    # Same document, different hypotheses
    ex1 = ContractNLIExample("doc1", "text", "hyp1", "Entailed", [])
    ex2 = ContractNLIExample("doc1", "text", "hyp2", "NotMentioned", [])

    mock_llm = Mock()
    mock_llm.config.generation_model = "test-model"

    with patch("legalkit.pipeline.parse.parse_document") as mock_parse, \
         patch("legalkit.pipeline.chunk.chunk_document") as mock_chunk, \
         patch("legalkit.pipeline.embed.embed_texts") as mock_embed, \
         patch("legalkit.pipeline.index.DocumentIndex") as mock_index_class, \
         patch("legalkit.pipeline.verify._check_entailment_llm") as mock_entail:

        mock_parse.return_value = Mock()

        mock_chunk_obj = Mock()
        mock_chunk_obj.id = "chunk1"
        mock_chunk_obj.node_id = "doc1_sec1"
        mock_chunk_obj.text = "text"
        mock_chunk_obj.char_start = 0
        mock_chunk_obj.char_end = 4
        mock_chunk_obj.embedding_text = "text"
        mock_chunk.return_value = [mock_chunk_obj]

        mock_embed.return_value = [[0.1] * 384]

        mock_index = Mock()
        mock_index.search_hybrid.return_value = [(0, 0.9, 0.0, 0.0)]
        mock_index.chunks = [mock_chunk_obj]
        mock_index_class.return_value = mock_index

        mock_entail.return_value = (EntailmentClass.ENTAILED, 0.85, "")

        report = run_contractnli_eval(
            examples=[ex1, ex2],
            llm=mock_llm,
            top_k=10,
        )

        # parse_document should be called only ONCE (document caching)
        assert mock_parse.call_count == 1
        # But we should have 2 predictions
        assert len(report.predictions) == 2
