"""Document management endpoints."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from legalkit.api.models import DocumentInfo
from legalkit.api.store import DocumentStore

router = APIRouter(prefix="/api/documents", tags=["documents"])
store = DocumentStore()

# Supported file extensions (from extract.py)
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


@router.get("/", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    """List all ingested documents."""
    return store.list()


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str) -> DocumentInfo:
    """Get metadata for a specific document."""
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )
    return doc


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str):
    """Delete a document."""
    if not store.delete(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )


@router.post("/", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED)
def ingest_document(file: UploadFile = File(...)) -> DocumentInfo:
    """Ingest a new document.

    Accepts .txt, .pdf, or .docx files. Runs the full ingestion pipeline:
    - Extract text
    - Parse document structure
    - Chunk content
    - Generate embeddings
    - Build search index
    - Build definition index

    Skips contextual enrichment for speed (--no-context mode).
    """
    from legalkit.pipeline.chunk import chunk_document
    from legalkit.pipeline.definitions import DefinitionIndex
    from legalkit.pipeline.embed import embed_texts
    from legalkit.pipeline.extract import extract_text
    from legalkit.pipeline.index import DocumentIndex
    from legalkit.pipeline.parse import parse_document

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type {suffix!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = file.file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Extract text
        try:
            text = extract_text(tmp_path)
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to extract text: {e}",
            )

        doc_id = Path(file.filename).stem
        t0 = time.time()

        # Parse document structure
        root = parse_document(text, doc_id=doc_id)

        # Chunk
        chunks = chunk_document(root, doc_id=doc_id)

        # Embed (skip contextual enrichment for speed)
        embedding_texts = [chunk.embedding_text for chunk in chunks]
        embeddings = embed_texts(embedding_texts)

        # Build index
        index = DocumentIndex()
        index.build(chunks, embeddings)

        # Build definition index
        def_index = DefinitionIndex()
        def_index.build(root)

        elapsed = time.time() - t0

        # Prepare metadata
        ingested_at = datetime.now(timezone.utc).isoformat()
        meta = DocumentInfo(
            id=doc_id,
            filename=file.filename,
            size_bytes=len(content),
            n_chunks=len(chunks),
            ingested_at=ingested_at,
        )

        # Save to store
        pipeline_data = {
            "root": root,
            "chunks": chunks,
            "index": index,
            "def_index": def_index,
            "text": text,
            "file": file.filename,
        }
        store.save(doc_id, pipeline_data, meta)

        return meta

    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)
