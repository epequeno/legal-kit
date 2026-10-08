"""Tests for ML-based document structure parser."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from legalkit.pipeline.parse_ml import (
    StructureClassifier,
    parse_document_ml,
    parse_document_hybrid,
    LABELS,
    LABEL_TO_NODE_TYPE,
)
from legalkit.models.document import NodeType
from legalkit.pipeline.parse import parse_document_with_ml


class TestStructureClassifier:
    """Tests for StructureClassifier model wrapper."""
    
    def test_load_none_returns_none(self):
        """Loading with None path returns None."""
        result = StructureClassifier.load(None)
        assert result is None
    
    def test_load_nonexistent_returns_none(self):
        """Loading non-existent path returns None."""
        StructureClassifier.clear_cache()
        result = StructureClassifier.load("/nonexistent/path/to/model")
        assert result is None
    
    def test_clear_cache(self):
        """clear_cache() clears the cache."""
        StructureClassifier._cache["test:cpu"] = None
        StructureClassifier.clear_cache()
        assert len(StructureClassifier._cache) == 0
    
    def test_classify_lines_empty_input(self):
        """classify_lines returns empty list for empty input."""
        # Create a mock classifier manually
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        classifier = StructureClassifier(mock_model, mock_tokenizer, "cpu")
        
        result = classifier.classify_lines([])
        assert result == []


class TestParseDocumentML:
    """Tests for parse_document_ml function."""
    
    def test_with_none_classifier_and_path_returns_empty_doc(self):
        """parse_document_ml with no classifier returns empty document."""
        text = "This is a test document."
        result = parse_document_ml(text, doc_id="test", classifier=None, model_path=None)
        
        assert result.id == "test"
        assert result.type == NodeType.DOCUMENT
    
    @patch.object(StructureClassifier, 'load')
    def test_auto_loads_classifier_from_path(self, mock_load):
        """parse_document_ml auto-loads classifier from model_path."""
        mock_classifier = MagicMock()
        mock_classifier.classify_lines.return_value = ["O"]
        mock_load.return_value = mock_classifier
        
        text = "Test"
        result = parse_document_ml(text, doc_id="test", model_path="/fake/path")
        
        mock_load.assert_called_once_with("/fake/path")
    
    def test_returns_document_node(self):
        """parse_document_ml returns DocumentNode with type=DOCUMENT."""
        mock_classifier = MagicMock()
        mock_classifier.classify_lines.return_value = ["O"]
        
        text = "Test document"
        result = parse_document_ml(text, doc_id="doc1", classifier=mock_classifier)
        
        assert result.type == NodeType.DOCUMENT
        assert result.id == "doc1"
    
    def test_detects_article_headers(self):
        """parse_document_ml detects B-ARTICLE_HEADER labels as ARTICLE nodes."""
        mock_classifier = MagicMock()
        mock_classifier.classify_lines.return_value = [
            "B-ARTICLE_HEADER",
            "O",
        ]
        
        text = "ARTICLE I - DEFINITIONS\nSome body text here."
        result = parse_document_ml(text, doc_id="doc1", classifier=mock_classifier)
        
        # Check that an ARTICLE node was created
        assert len(result.children) > 0
        article_nodes = [n for n in result.children if n.type == NodeType.ARTICLE]
        assert len(article_nodes) == 1
    
    def test_detects_section_headers(self):
        """parse_document_ml detects B-SECTION_HEADER labels as SECTION nodes."""
        mock_classifier = MagicMock()
        mock_classifier.classify_lines.return_value = [
            "B-SECTION_HEADER",
            "O",
        ]
        
        text = "Section 1.1 Introduction\nBody text."
        result = parse_document_ml(text, doc_id="doc1", classifier=mock_classifier)
        
        # Check that a SECTION node was created
        section_nodes = [n for n in result.all_nodes() if n.type == NodeType.SECTION]
        assert len(section_nodes) == 1
    
    def test_empty_text_returns_empty_document(self):
        """parse_document_ml with empty text returns empty document."""
        mock_classifier = MagicMock()
        
        result = parse_document_ml("", doc_id="empty", classifier=mock_classifier)
        
        assert result.type == NodeType.DOCUMENT
        assert result.id == "empty"
        assert len(result.children) == 0
    
    def test_multiple_sections_with_hierarchy(self):
        """parse_document_ml builds proper hierarchy for multiple sections."""
        mock_classifier = MagicMock()
        mock_classifier.classify_lines.return_value = [
            "B-ARTICLE_HEADER",
            "O",
            "B-SECTION_HEADER",
            "O",
            "B-SECTION_HEADER",
            "O",
        ]
        
        text = "ARTICLE I\nIntro\nSection 1.1\nBody 1\nSection 1.2\nBody 2"
        result = parse_document_ml(text, doc_id="doc1", classifier=mock_classifier)
        
        # Should have one article with two sections
        articles = [n for n in result.children if n.type == NodeType.ARTICLE]
        assert len(articles) == 1
        sections = [n for n in articles[0].children if n.type == NodeType.SECTION]
        assert len(sections) == 2


class TestParseDocumentHybrid:
    """Tests for parse_document_hybrid function."""
    
    def test_returns_regex_when_enough_structure(self):
        """Hybrid parser returns 'regex' when regex finds enough nodes."""
        # This contract should parse well with regex
        text = """
ARTICLE I - DEFINITIONS

Section 1.1 Affiliate
The term "Affiliate" means...

Section 1.2 Agreement
This Agreement...

ARTICLE II - OBLIGATIONS

Section 2.1 General
The Company shall...
"""
        result, method = parse_document_hybrid(
            text, doc_id="test", model_path=None, min_nodes_threshold=3
        )
        
        assert method == "regex"
        assert result.type == NodeType.DOCUMENT
        # Should have detected multiple structural nodes
        structural_nodes = [
            n for n in result.all_nodes()
            if n.type not in (NodeType.PARAGRAPH, NodeType.DOCUMENT)
        ]
        assert len(structural_nodes) >= 3
    
    def test_returns_regex_when_model_path_none(self):
        """Hybrid parser returns 'regex' even if sparse when model_path=None."""
        # Sparse text with no standard markers
        text = "Just some plain text with no structure."
        
        result, method = parse_document_hybrid(
            text, doc_id="test", model_path=None, min_nodes_threshold=10
        )
        
        # Should stick with regex since no ML fallback available
        assert method == "regex"
    
    def test_returns_regex_when_model_doesnt_exist(self):
        """Hybrid parser returns 'regex' when model path doesn't exist."""
        # Use a path that definitely doesn't exist
        text = "Plain text with no structure"
        
        result, method = parse_document_hybrid(
            text, doc_id="test", model_path="/this/path/does/not/exist", min_nodes_threshold=10
        )
        
        assert method == "regex"
        assert result.type == NodeType.DOCUMENT


class TestParseDocumentWithML:
    """Tests for parse_document_with_ml convenience wrapper."""
    
    def test_returns_document_node(self):
        """parse_document_with_ml returns DocumentNode."""
        # Use a well-structured doc that regex will parse successfully
        text = """
ARTICLE I - DEFINITIONS

Section 1.1 Terms
Various terms...
"""
        result = parse_document_with_ml(text, doc_id="test")
        
        assert result.type == NodeType.DOCUMENT
        assert result.id == "test"
    
    def test_handles_sparse_document(self):
        """parse_document_with_ml handles documents with little structure."""
        text = "This is just plain text."
        
        result = parse_document_with_ml(text, doc_id="test")
        
        # Should return a valid document even if sparse
        assert result.type == NodeType.DOCUMENT
        assert result.id == "test"


class TestLabelMapping:
    """Tests for label-to-node-type mapping."""
    
    def test_all_b_labels_have_mapping(self):
        """All B-* labels in LABELS have a mapping to NodeType."""
        for label in LABELS:
            if label.startswith("B-"):
                # All B- labels should have a mapping
                assert label in LABEL_TO_NODE_TYPE, f"Label {label} missing from LABEL_TO_NODE_TYPE"
    
    def test_mapped_types_are_valid(self):
        """All mapped NodeTypes are valid enum values."""
        for label, node_type in LABEL_TO_NODE_TYPE.items():
            assert isinstance(node_type, NodeType), f"Invalid NodeType for {label}"


class TestIntegration:
    """Integration tests that use the actual model if available."""
    
    def test_can_load_model_if_exists(self):
        """If model exists at default path, it should load successfully."""
        default_path = Path(__file__).parent.parent / "models/structure-parser/model-dir"
        
        if not default_path.exists():
            pytest.skip("Model not available at default path")
        
        classifier = StructureClassifier.load(str(default_path))
        
        if classifier is not None:
            # Model loaded successfully
            assert classifier.model is not None
            assert classifier.tokenizer is not None
            
            # Try classifying a simple line
            result = classifier.classify_lines(["Section 1.1 Introduction"])
            assert len(result) == 1
            assert result[0] in LABELS
    
    def test_parse_with_real_model_if_exists(self):
        """If model exists, parse_document_ml should work with real text."""
        default_path = Path(__file__).parent.parent / "models/structure-parser/model-dir"
        
        if not default_path.exists():
            pytest.skip("Model not available at default path")
        
        classifier = StructureClassifier.load(str(default_path))
        
        if classifier is None:
            pytest.skip("Could not load model")
        
        text = """
ARTICLE I - DEFINITIONS

Section 1.1 General Terms
The following terms shall have the meanings set forth below.

Section 1.2 Specific Terms
Additional definitions apply.
"""
        
        result = parse_document_ml(text, doc_id="test", classifier=classifier)
        
        assert result.type == NodeType.DOCUMENT
        # Model should detect some structure
        all_nodes = result.all_nodes()
        assert len(all_nodes) > 1  # More than just root


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
