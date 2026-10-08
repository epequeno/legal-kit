"""Tests for embedding evaluation and local model support."""

import sys
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock


# Mock sentence_transformers module for testing
sys.modules['sentence_transformers'] = MagicMock()


def test_load_local_embedding_model_caches_correctly():
    """Test that load_local_embedding_model caches the model correctly."""
    from legalkit.pipeline import embed
    
    # Reset module state
    embed._local_model = None
    embed._local_model_path = None
    
    mock_model = Mock()
    mock_st = Mock(return_value=mock_model)
    
    with patch("sentence_transformers.SentenceTransformer", mock_st):
        embed.load_local_embedding_model("models/test-model", device="cpu")
        
        # Verify model was loaded
        mock_st.assert_called_once_with("models/test-model", device="cpu")
        assert embed._local_model is mock_model
        assert embed._local_model_path == "models/test-model"


def test_embed_texts_local_falls_back_to_api_when_no_model():
    """Test that embed_texts_local falls back to API when no model path and no cached model."""
    from legalkit.pipeline import embed
    
    # Reset module state
    embed._local_model = None
    embed._local_model_path = None
    
    mock_embed_texts = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    
    with patch.object(embed, "embed_texts", mock_embed_texts):
        result = embed.embed_texts_local(["test query"])
        
        # Should have fallen back to API
        mock_embed_texts.assert_called_once_with(["test query"])
        np.testing.assert_array_equal(result, [[0.1, 0.2, 0.3]])


def test_embed_texts_local_uses_cached_model_on_second_call():
    """Test that embed_texts_local uses cached model on second call (verify no double-load)."""
    from legalkit.pipeline import embed
    
    # Reset module state
    embed._local_model = None
    embed._local_model_path = None
    
    # Mock model and encode
    mock_model = Mock()
    mock_model.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]], dtype=np.float64))
    mock_st = Mock(return_value=mock_model)
    
    with patch("sentence_transformers.SentenceTransformer", mock_st):
        # First call - should load model
        result1 = embed.embed_texts_local(["query 1"], model_path="models/test")
        mock_st.assert_called_once()
        
        # Second call - should use cached model
        result2 = embed.embed_texts_local(["query 2"], model_path="models/test")
        mock_st.assert_called_once()  # Still only called once!
        
        # Both calls should have used the same model
        assert mock_model.encode.call_count == 2


def test_embed_texts_local_handles_empty_string_input():
    """Test that embed_texts_local handles empty string input."""
    from legalkit.pipeline import embed
    
    # Reset module state
    embed._local_model = None
    embed._local_model_path = None
    
    # Mock model
    mock_model = Mock()
    mock_model.encode = Mock(return_value=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64))
    mock_st = Mock(return_value=mock_model)
    
    with patch("sentence_transformers.SentenceTransformer", mock_st):
        result = embed.embed_texts_local(["normal text", "   ", ""], model_path="models/test")
        
        # Verify empty strings were replaced with "empty"
        call_args = mock_model.encode.call_args[0][0]
        assert call_args[0] == "normal text"
        assert call_args[1] == "empty"
        assert call_args[2] == "empty"


def test_run_benchmark_accepts_embedding_fn_parameter():
    """Test that run_benchmark accepts embedding_fn parameter and uses it."""
    from legalkit.eval.benchmark import run_benchmark, BenchmarkQuery, GroundTruthSnippet
    
    # Mock corpus file
    mock_corpus_text = "ARTICLE I\nSection 1.1 Purpose\nThis is a test document."
    
    # Create mock query
    query = BenchmarkQuery(
        query="What is the purpose?",
        snippets=[
            GroundTruthSnippet(file_path="test.txt", span=(10, 40))
        ],
        tags=["test"],
    )
    
    # Mock embedding function
    embed_call_count = {"count": 0}
    def mock_embed_fn(texts):
        embed_call_count["count"] += 1
        # Return random embeddings
        return np.random.rand(len(texts), 384).astype(np.float32)
    
    # Mock file reading
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", return_value=mock_corpus_text):
        
        results = run_benchmark(
            queries=[query],
            max_chunks_per_query=10,
            embedding_fn=mock_embed_fn,
            progress_callback=None,
        )
        
        # Verify our custom embedding function was called
        assert embed_call_count["count"] > 0
        
        # Verify results were returned
        assert len(results.results) == 1
        assert results.results[0].query == "What is the purpose?"


def test_run_embedding_comparison_returns_correct_dict_structure():
    """Test that run_embedding_comparison returns correct dict structure."""
    from legalkit.eval.embedding_eval import run_embedding_comparison
    
    # Mock everything
    mock_queries = [
        Mock(query="test query 1", snippets=[Mock(file_path="test.txt", span=(0, 10))], tags=["test"]),
        Mock(query="test query 2", snippets=[Mock(file_path="test.txt", span=(10, 20))], tags=["test"]),
    ]
    
    mock_results = Mock()
    mock_results.recall_at_k = Mock(side_effect=lambda k, tag=None: 0.5 if k <= 64 else 0.0)
    mock_results.precision_at_k = Mock(side_effect=lambda k, tag=None: 0.4 if k <= 64 else 0.0)
    mock_results.tags = ["test"]
    
    with patch("legalkit.eval.embedding_eval.load_benchmark", return_value=mock_queries), \
         patch("legalkit.eval.embedding_eval.run_benchmark", return_value=mock_results), \
         patch("legalkit.pipeline.embed.embed_texts_local") as mock_local, \
         patch("legalkit.pipeline.embed.embed_texts") as mock_api:
        
        mock_local.return_value = np.random.rand(10, 384).astype(np.float32)
        mock_api.return_value = np.random.rand(10, 384).astype(np.float32)
        
        result = run_embedding_comparison(
            datasets=["test"],
            max_per_dataset=2,
            local_model_path="models/test",
            top_k=64,
        )
        
        # Check structure
        assert "local_model" in result
        assert "api_model" in result
        assert "improvement" in result
        assert "queries_evaluated" in result
        assert "datasets" in result
        
        # Check nested structure
        assert "recall@k" in result["local_model"]
        assert "precision@k" in result["local_model"]
        assert "per_dataset" in result["local_model"]
        
        assert "recall@k" in result["api_model"]
        assert "precision@k" in result["api_model"]
        
        assert "overall_recall@64" in result["improvement"]
        assert "overall_precision@64" in result["improvement"]
        assert "per_k" in result["improvement"]


def test_embed_query_local():
    """Test embed_query_local returns 1-D array."""
    from legalkit.pipeline import embed
    
    # Reset module state
    embed._local_model = None
    embed._local_model_path = None
    
    # Mock model
    mock_model = Mock()
    mock_model.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]], dtype=np.float64))
    mock_st = Mock(return_value=mock_model)
    
    with patch("sentence_transformers.SentenceTransformer", mock_st):
        result = embed.embed_query_local("test query", model_path="models/test")
        
        # Should be 1-D array
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])


def test_load_local_embedding_model_raises_on_failure():
    """Test that load_local_embedding_model raises RuntimeError on failure."""
    from legalkit.pipeline import embed
    
    # Reset module state
    embed._local_model = None
    embed._local_model_path = None
    
    mock_st = Mock(side_effect=Exception("Model not found"))
    
    with patch("sentence_transformers.SentenceTransformer", mock_st):
        with pytest.raises(RuntimeError, match="Failed to load local embedding model"):
            embed.load_local_embedding_model("invalid/path")


def test_embed_texts_local_reloads_when_different_model_path():
    """Test that embed_texts_local reloads when a different model path is provided."""
    from legalkit.pipeline import embed
    
    # Reset module state
    embed._local_model = None
    embed._local_model_path = None
    
    # Mock models
    mock_model1 = Mock()
    mock_model1.encode = Mock(return_value=np.array([[0.1, 0.2]], dtype=np.float64))
    
    mock_model2 = Mock()
    mock_model2.encode = Mock(return_value=np.array([[0.3, 0.4]], dtype=np.float64))
    
    models = [mock_model1, mock_model2]
    mock_st = Mock(side_effect=models)
    
    with patch("sentence_transformers.SentenceTransformer", mock_st):
        # First call with model A
        result1 = embed.embed_texts_local(["query"], model_path="models/A")
        assert mock_st.call_count == 1
        
        # Second call with model B - should reload
        result2 = embed.embed_texts_local(["query"], model_path="models/B")
        assert mock_st.call_count == 2
        
        # Verify different models were used
        np.testing.assert_array_almost_equal(result1[0], [0.1, 0.2])
        np.testing.assert_array_almost_equal(result2[0], [0.3, 0.4])


def test_embed_texts_local_truncates_long_texts():
    """Test that embed_texts_local truncates very long texts."""
    from legalkit.pipeline import embed
    
    # Reset module state
    embed._local_model = None
    embed._local_model_path = None
    
    # Mock model
    mock_model = Mock()
    mock_model.encode = Mock(return_value=np.array([[0.1, 0.2]], dtype=np.float64))
    mock_st = Mock(return_value=mock_model)
    
    long_text = "x" * 20000  # Longer than 10000 char limit
    
    with patch("sentence_transformers.SentenceTransformer", mock_st):
        result = embed.embed_texts_local([long_text], model_path="models/test")
        
        # Verify text was truncated
        call_args = mock_model.encode.call_args[0][0]
        assert len(call_args[0]) == 10000
