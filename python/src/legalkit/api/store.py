"""Filesystem document store.

Wraps the existing .legalkit_cache/ pickle mechanism from cli.py,
adding document metadata (filename, size, ingested_at) alongside
the pipeline state (index, root, chunks).

Meta is stored as JSON alongside each pickle:
  .legalkit_cache/{doc_id}.pkl  — pipeline state
  .legalkit_cache/{doc_id}.json — DocumentInfo metadata
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from legalkit.api.models import DocumentInfo

CACHE_DIR = Path(".legalkit_cache")


class DocumentStore:
    """Filesystem document store using pickle + JSON metadata."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def list(self) -> list[DocumentInfo]:
        """List all ingested documents."""
        docs = []
        for meta_file in self.cache_dir.glob("*.json"):
            try:
                with open(meta_file) as f:
                    data = json.load(f)
                    docs.append(DocumentInfo(**data))
            except Exception:
                # Skip corrupted metadata files
                pass
        return sorted(docs, key=lambda d: d.ingested_at, reverse=True)

    def get(self, doc_id: str) -> DocumentInfo | None:
        """Get metadata for a specific document."""
        meta_file = self.cache_dir / f"{doc_id}.json"
        if not meta_file.exists():
            return None
        try:
            with open(meta_file) as f:
                data = json.load(f)
                return DocumentInfo(**data)
        except Exception:
            return None

    def save(self, doc_id: str, pipeline_data: dict, meta: DocumentInfo) -> None:
        """Save pipeline state and metadata to disk.

        Args:
            doc_id: Document identifier.
            pipeline_data: Dict containing index, root, chunks, def_index, etc.
            meta: DocumentInfo metadata.
        """
        # Save pipeline state as pickle
        pkl_file = self.cache_dir / f"{doc_id}.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(pipeline_data, f)

        # Save metadata as JSON
        meta_file = self.cache_dir / f"{doc_id}.json"
        with open(meta_file, "w") as f:
            json.dump(meta.model_dump(), f, indent=2)

    def load_pipeline(self, doc_id: str) -> dict | None:
        """Load pipeline state from disk.

        Returns:
            Dict with keys: index, root, chunks, def_index, text, file
            or None if not found.
        """
        pkl_file = self.cache_dir / f"{doc_id}.pkl"
        if not pkl_file.exists():
            return None
        try:
            with open(pkl_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def delete(self, doc_id: str) -> bool:
        """Delete a document and its metadata.

        Returns:
            True if deleted, False if not found.
        """
        pkl_file = self.cache_dir / f"{doc_id}.pkl"
        meta_file = self.cache_dir / f"{doc_id}.json"

        found = False
        if pkl_file.exists():
            pkl_file.unlink()
            found = True
        if meta_file.exists():
            meta_file.unlink()
            found = True

        return found

    def exists(self, doc_id: str) -> bool:
        """Check if a document exists."""
        pkl_file = self.cache_dir / f"{doc_id}.pkl"
        return pkl_file.exists()
