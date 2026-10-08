"""Query endpoints with SSE streaming."""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from legalkit.api.models import AskRequest
from legalkit.api.store import DocumentStore

router = APIRouter(prefix="/api/documents", tags=["query"])
store = DocumentStore()


def sse_event(event: str, data: dict) -> str:
    """Format an SSE event.

    Args:
        event: Event name (e.g., "progress", "result", "error").
        data: Event data (will be JSON-serialized).

    Returns:
        SSE-formatted string: "event: {event}\\ndata: {json}\\n\\n"
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/{doc_id}/ask")
async def ask(doc_id: str, req: AskRequest):
    """Ask a question about a document (SSE streaming).

    Streams progress events and final result:
    - event: progress → {stage: "retrieving"|"generating"|"verifying", message: "..."}
    - event: result → {answer: "...", claims: [...], overall_confidence: {...}, ...}
    - event: error → {message: "..."}

    The result data follows the EvidenceReport.to_dict() schema.
    """
    # Verify document exists
    if not store.exists(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for the query pipeline."""
        try:
            t0 = time.time()

            # Stage 1: Retrieve
            yield sse_event(
                "progress",
                {"stage": "retrieving", "message": "Finding relevant passages..."},
            )

            # Load pipeline data and run retrieval (sync, so use executor)
            loop = asyncio.get_event_loop()

            def _retrieve():
                from legalkit.pipeline.embed import embed_query
                from legalkit.pipeline.retrieve import retrieve

                data = store.load_pipeline(doc_id)
                if data is None:
                    raise ValueError(f"Failed to load pipeline data for {doc_id}")

                index = data["index"]
                root = data["root"]
                def_index = data.get("def_index")

                # Embed query
                query_emb = embed_query(req.question)

                # Retrieve
                result = retrieve(
                    query=req.question,
                    query_embedding=query_emb,
                    index=index,
                    doc_tree=root,
                    top_k=req.top_k,
                    definition_index=def_index,
                )

                return result, root

            retrieval_result, root = await loop.run_in_executor(None, _retrieve)

            # Stage 2: Generate + Verify
            yield sse_event(
                "progress",
                {"stage": "generating", "message": "Generating answer..."},
            )

            yield sse_event(
                "progress",
                {"stage": "verifying", "message": "Verifying citations..."},
            )

            def _generate_and_verify():
                from legalkit.llm.client import LLMClient
                from legalkit.pipeline.report import build_evidence_report

                llm = LLMClient()
                report = build_evidence_report(
                    question=req.question,
                    retrieval_result=retrieval_result,
                    doc_tree=root,
                    llm=llm,
                    doc_id=doc_id,
                )
                return report

            report = await loop.run_in_executor(None, _generate_and_verify)

            # Stage 3: Return result
            elapsed = time.time() - t0
            result_data = report.to_dict()
            result_data["elapsed_seconds"] = round(elapsed, 2)
            result_data["doc_id"] = doc_id

            yield sse_event("result", result_data)

        except Exception as e:
            # Send error event
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
