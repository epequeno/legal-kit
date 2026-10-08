"""Tests for the LegalKit FastAPI backend."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from legalkit.api.app import app
from legalkit.api.models import AskRequest, DocumentInfo
from legalkit.api.routes.query import sse_event
from legalkit.api.store import DocumentStore


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def temp_store(tmp_path):
    """Temporary document store for testing."""
    return DocumentStore(cache_dir=tmp_path)


# ───────────────────────────────────────────────────────────────────
# Health & Basic Routes
# ───────────────────────────────────────────────────────────────────


def test_health_endpoint(client):
    """GET /api/health returns 200 with status=ok."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_list_documents_empty(client):
    """GET /api/documents returns empty list initially."""
    response = client.get("/api/documents/")
    assert response.status_code == 200
    assert response.json() == []


def test_get_nonexistent_document(client):
    """GET /api/documents/{id} returns 404 for nonexistent doc."""
    response = client.get("/api/documents/nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_nonexistent_document(client):
    """DELETE /api/documents/{id} returns 404 for nonexistent doc."""
    response = client.delete("/api/documents/nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ───────────────────────────────────────────────────────────────────
# Document Upload & Ingestion
# ───────────────────────────────────────────────────────────────────


def test_upload_invalid_file_type(client):
    """POST /api/documents with invalid file type returns 400."""
    files = {"file": ("test.xyz", b"content", "application/octet-stream")}
    response = client.post("/api/documents/", files=files)
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_upload_no_filename(client):
    """POST /api/documents with no filename returns 422 (validation error)."""
    # Create a file upload without a filename
    files = {"file": (None, b"content", "text/plain")}
    response = client.post("/api/documents/", files=files)
    # FastAPI returns 422 for validation errors
    assert response.status_code == 422


@patch("legalkit.api.routes.documents.store")
@patch("legalkit.pipeline.definitions.DefinitionIndex")
@patch("legalkit.pipeline.index.DocumentIndex")
@patch("legalkit.pipeline.embed.embed_texts")
@patch("legalkit.pipeline.chunk.chunk_document")
@patch("legalkit.pipeline.parse.parse_document")
@patch("legalkit.pipeline.extract.extract_text")
def test_upload_valid_txt_file(
    mock_extract,
    mock_parse,
    mock_chunk,
    mock_embed,
    mock_doc_index,
    mock_def_index,
    mock_store_instance,
    client,
):
    """POST /api/documents with valid .txt file returns DocumentInfo."""
    import numpy as np

    from legalkit.models.retrieval import Chunk

    # Mock the pipeline
    mock_extract.return_value = "Sample legal text content"
    mock_root = MagicMock()
    mock_parse.return_value = mock_root

    # Create mock chunks
    mock_chunks = [
        Chunk(
            id="chunk_0",
            doc_id="test_doc",
            node_id="§1",
            text="Sample text",
        )
    ]
    mock_chunk.return_value = mock_chunks

    # Mock embeddings
    mock_embed.return_value = np.random.rand(1, 768)

    # Mock index
    mock_index_instance = MagicMock()
    mock_doc_index.return_value = mock_index_instance

    # Mock definition index
    mock_def_instance = MagicMock()
    mock_def_index.return_value = mock_def_instance

    # Mock store.save to avoid pickle issues
    mock_store_instance.save.return_value = None

    # Upload file
    content = b"Sample legal document content"
    files = {"file": ("test_doc.txt", content, "text/plain")}
    response = client.post("/api/documents/", files=files)

    assert response.status_code == 201
    data = response.json()

    # Validate DocumentInfo fields
    assert data["id"] == "test_doc"
    assert data["filename"] == "test_doc.txt"
    assert data["size_bytes"] == len(content)
    assert data["n_chunks"] == 1
    assert "ingested_at" in data

    # Verify ISO timestamp
    datetime.fromisoformat(data["ingested_at"])

    # Verify store.save was called
    assert mock_store_instance.save.called


# ───────────────────────────────────────────────────────────────────
# DocumentStore Tests
# ───────────────────────────────────────────────────────────────────


def test_store_save_and_list(temp_store):
    """DocumentStore.save() + list() round-trip."""
    doc_id = "test_doc"
    # Use pickleable data instead of MagicMock
    pipeline_data = {
        "root": "mock_root",
        "chunks": [],
        "index": "mock_index",
        "def_index": "mock_def_index",
        "text": "Sample text",
        "file": "test.txt",
    }
    meta = DocumentInfo(
        id=doc_id,
        filename="test.txt",
        size_bytes=1024,
        n_chunks=10,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )

    temp_store.save(doc_id, pipeline_data, meta)

    # List should return the document
    docs = temp_store.list()
    assert len(docs) == 1
    assert docs[0].id == doc_id
    assert docs[0].filename == "test.txt"


def test_store_get(temp_store):
    """DocumentStore.get() returns correct metadata."""
    doc_id = "test_doc"
    meta = DocumentInfo(
        id=doc_id,
        filename="test.txt",
        size_bytes=1024,
        n_chunks=10,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )
    pipeline_data = {"root": "mock_root"}

    temp_store.save(doc_id, pipeline_data, meta)

    # Get should return the same metadata
    retrieved = temp_store.get(doc_id)
    assert retrieved is not None
    assert retrieved.id == doc_id
    assert retrieved.filename == "test.txt"


def test_store_delete(temp_store):
    """DocumentStore.delete() removes document."""
    doc_id = "test_doc"
    meta = DocumentInfo(
        id=doc_id,
        filename="test.txt",
        size_bytes=1024,
        n_chunks=10,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )
    pipeline_data = {"root": "mock_root"}

    temp_store.save(doc_id, pipeline_data, meta)
    assert temp_store.exists(doc_id)

    # Delete
    deleted = temp_store.delete(doc_id)
    assert deleted is True
    assert not temp_store.exists(doc_id)

    # Delete again should return False
    deleted = temp_store.delete(doc_id)
    assert deleted is False


def test_store_load_pipeline(temp_store):
    """DocumentStore.load_pipeline() returns pipeline data."""
    doc_id = "test_doc"
    pipeline_data = {
        "root": "mock_root",
        "chunks": ["chunk1", "chunk2"],
        "index": "mock_index",
    }
    meta = DocumentInfo(
        id=doc_id,
        filename="test.txt",
        size_bytes=1024,
        n_chunks=2,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )

    temp_store.save(doc_id, pipeline_data, meta)

    # Load pipeline
    loaded = temp_store.load_pipeline(doc_id)
    assert loaded is not None
    assert loaded["root"] == "mock_root"
    assert loaded["chunks"] == ["chunk1", "chunk2"]


def test_store_load_nonexistent(temp_store):
    """DocumentStore.load_pipeline() returns None for nonexistent doc."""
    loaded = temp_store.load_pipeline("nonexistent")
    assert loaded is None


# ───────────────────────────────────────────────────────────────────
# SSE Event Formatting
# ───────────────────────────────────────────────────────────────────


def test_sse_event_format():
    """SSE event format helper produces correct string."""
    result = sse_event("progress", {"stage": "retrieving", "message": "Test"})
    assert result == 'event: progress\ndata: {"stage": "retrieving", "message": "Test"}\n\n'

    result = sse_event("result", {"answer": "The answer is 42"})
    assert result == 'event: result\ndata: {"answer": "The answer is 42"}\n\n'


# ───────────────────────────────────────────────────────────────────
# Request Model Validation
# ───────────────────────────────────────────────────────────────────


def test_ask_request_validation():
    """AskRequest validates question is required."""
    # Valid request
    req = AskRequest(question="What is the cap?")
    assert req.question == "What is the cap?"
    assert req.top_k == 20  # default

    # Custom top_k
    req = AskRequest(question="Test", top_k=50)
    assert req.top_k == 50

    # Missing question should fail
    with pytest.raises(Exception):  # Pydantic validation error
        AskRequest()


def test_ask_request_top_k_default():
    """AskRequest top_k defaults to 20."""
    req = AskRequest(question="Test question")
    assert req.top_k == 20


# ───────────────────────────────────────────────────────────────────
# Query Endpoint (mocked pipeline)
# ───────────────────────────────────────────────────────────────────


@patch("legalkit.pipeline.report.build_evidence_report")
@patch("legalkit.llm.client.LLMClient")
@patch("legalkit.pipeline.retrieve.retrieve")
@patch("legalkit.pipeline.embed.embed_query")
@patch("legalkit.api.routes.query.store")
def test_ask_endpoint_sse_stream(
    mock_store_instance,
    mock_embed_query,
    mock_retrieve,
    mock_llm_client,
    mock_build_report,
    client,
):
    """POST /api/documents/{id}/ask streams SSE events."""
    import numpy as np

    from legalkit.models.report import (
        ConfidenceLevel,
        ConfidenceScore,
        EvidenceReport,
    )
    from legalkit.models.retrieval import RetrievalResult

    # Mock store
    mock_store_instance.exists.return_value = True
    mock_store_instance.load_pipeline.return_value = {
        "root": MagicMock(),
        "index": MagicMock(),
        "def_index": None,
    }

    # Mock pipeline components
    mock_embed_query.return_value = np.random.rand(768)
    mock_retrieve.return_value = RetrievalResult(
        query="What is the cap?",
        ranked_chunks=[],
        cross_ref_chunks=[],
    )

    # Mock report
    mock_report = EvidenceReport(
        question="What is the cap?",
        answer="The cap is $5M.",
        claims=[],
        overall_confidence=ConfidenceScore(level=ConfidenceLevel.HIGH, score=0.9),
        doc_id="test_doc",
        model="gpt-4",
        elapsed_seconds=1.5,
    )
    mock_build_report.return_value = mock_report

    # Make request
    response = client.post(
        "/api/documents/test_doc/ask",
        json={"question": "What is the cap?", "top_k": 20},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Parse SSE events
    events = []
    for line in response.text.strip().split("\n\n"):
        if not line:
            continue
        lines = line.split("\n")
        event_type = None
        event_data = None
        for l in lines:
            if l.startswith("event: "):
                event_type = l[7:]
            elif l.startswith("data: "):
                event_data = json.loads(l[6:])
        if event_type and event_data:
            events.append((event_type, event_data))

    # Verify events
    assert len(events) >= 3  # At least progress events + result

    # Check for progress events
    progress_stages = [e[1]["stage"] for e in events if e[0] == "progress"]
    assert "retrieving" in progress_stages
    assert "generating" in progress_stages
    assert "verifying" in progress_stages

    # Check for result event
    result_events = [e for e in events if e[0] == "result"]
    assert len(result_events) == 1
    result_data = result_events[0][1]
    assert result_data["answer"] == "The cap is $5M."
    assert result_data["doc_id"] == "test_doc"


def test_ask_nonexistent_document(client):
    """POST /api/documents/{id}/ask returns 404 for nonexistent doc."""
    response = client.post(
        "/api/documents/nonexistent/ask",
        json={"question": "Test", "top_k": 20},
    )
    assert response.status_code == 404
