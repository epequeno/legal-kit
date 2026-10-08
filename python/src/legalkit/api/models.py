"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Document metadata returned by the API."""

    id: str
    filename: str
    size_bytes: int
    n_chunks: int
    ingested_at: str


class AskRequest(BaseModel):
    """Request body for /api/documents/{id}/ask endpoint."""

    question: str
    top_k: int = Field(default=20, ge=1, le=100)


class HealthResponse(BaseModel):
    """Response for /api/health endpoint."""

    status: str
    version: str
