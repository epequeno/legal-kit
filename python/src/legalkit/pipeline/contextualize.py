"""Stage 2.5: Contextual enrichment.

Generates LLM-produced context for each chunk following Anthropic's
contextual retrieval technique:

  https://www.anthropic.com/news/contextual-retrieval

A truncated view of the full document is passed to a fast model
(claude-3-haiku) for each chunk. The model writes 2–3 sentences that
situate the chunk within the document — capturing document type, parties,
and section purpose. This context is prepended to the chunk text before
embedding, dramatically improving retrieval recall for out-of-context
clauses.

Throughput: chunks are processed concurrently via ThreadPoolExecutor so
a 100-chunk document takes roughly the same wall-clock time as 10 sequential
calls.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from legalkit.models.document import DocumentNode
from legalkit.models.retrieval import Chunk

logger = logging.getLogger(__name__)

# Anthropic contextual retrieval recommends feeding the whole document.
# We cap at 80 K chars (~20 K tokens) to stay within Haiku's context window
# while keeping API latency reasonable for very large contracts.
_MAX_DOC_CHARS = 80_000

# Chunk text sent with the contextualization prompt (the full chunk is
# already in the document; this is just the focal excerpt).
_MAX_CHUNK_CHARS = 1_200

# Concurrent API workers — empirically safe against OpenRouter/Anthropic
# rate limits for most tier levels.
DEFAULT_WORKERS = 10


def build_document_context(
    root: DocumentNode,
    full_text: str,
    max_chars: int = _MAX_DOC_CHARS,
) -> str:
    """Build the document context string passed to each contextualization call.

    Strategy:
    - If the full document text fits within max_chars, use it directly.
    - Otherwise, compose a compact structural summary: the first 3 K chars
      (preamble / parties block) plus every section heading and its first
      sentence. This gives the model enough signal to understand document
      type, parties, and structure without exceeding the context window.

    Args:
        root: Parsed document tree root.
        full_text: Raw document text.
        max_chars: Character cap for the returned context string.

    Returns:
        Document context string ready for the LLM prompt.
    """
    if len(full_text) <= max_chars:
        return full_text

    # Document too large — build a structural synopsis.
    lines: list[str] = []

    # 1. Opening block (parties, recitals, definitions intro)
    lines.append(full_text[:3_000].strip())
    lines.append("\n\n[... document continues — structural outline below ...]\n")

    # 2. Section headings with first sentence of each section
    for node in root.all_nodes():
        if not node.title:
            continue
        heading = f"{node.id}: {node.title}" if node.title else node.id
        # First sentence of the node's own text (not children)
        snippet = ""
        if node.text:
            first_line = node.text.strip().split("\n")[0][:200]
            snippet = f" — {first_line}" if first_line else ""
        lines.append(f"  {heading}{snippet}")

    synopsis = "\n".join(lines)
    # Final safety truncation
    return synopsis[:max_chars]


def contextualize_chunks(
    chunks: list[Chunk],
    doc_context: str,
    llm: object,  # LLMClient — avoid circular import
    *,
    max_workers: int = DEFAULT_WORKERS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Chunk]:
    """Generate contextual descriptions for all chunks and attach them.

    Each chunk's ``context_text`` field is populated in-place. The returned
    list is the same object as the input list (mutated).

    Args:
        chunks: Chunks to contextualize (mutated in-place).
        doc_context: Document context string from :func:`build_document_context`.
        llm: :class:`~legalkit.llm.client.LLMClient` instance.
        max_workers: Thread pool size for concurrent API calls.
        progress_callback: Optional ``(completed, total) -> None`` callback
            called after each chunk completes.

    Returns:
        The same ``chunks`` list, with ``context_text`` set on each element.
    """
    total = len(chunks)
    if total == 0:
        return chunks

    completed = 0
    errors = 0

    def _contextualize_one(chunk: Chunk) -> tuple[Chunk, str | None]:
        chunk_path = " > ".join(chunk.node_path) if chunk.node_path else chunk.node_id
        chunk_text = chunk.text[:_MAX_CHUNK_CHARS]
        try:
            ctx = llm.contextualize_chunk(
                document=doc_context,
                chunk_path=chunk_path,
                chunk_text=chunk_text,
            )
            return chunk, ctx.strip()
        except Exception as exc:
            logger.warning("Failed to contextualize chunk %s: %s", chunk.id, exc)
            return chunk, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_contextualize_one, chunk): chunk for chunk in chunks}

        for future in as_completed(futures):
            nonlocal_completed = None  # workaround: use list for nonlocal in 3.10
            chunk, ctx = future.result()
            if ctx is not None:
                chunk.context_text = ctx
            else:
                errors += 1
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    if errors:
        logger.warning(
            "Contextual enrichment: %d/%d chunks failed (will embed without context).",
            errors,
            total,
        )

    return chunks
