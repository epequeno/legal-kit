# LegalKit FastAPI Backend - Implementation Summary

## ✅ Completed Implementation

### API Contract (Exact Implementation)
- `GET /api/health` → `{status: "ok", version: "0.1.0"}`
- `GET /api/documents` → `DocumentInfo[]`
- `POST /api/documents` (multipart file) → `DocumentInfo` (201 Created)
- `GET /api/documents/{id}` → `DocumentInfo`
- `DELETE /api/documents/{id}` → 204 No Content
- `POST /api/documents/{id}/ask` (JSON body) → SSE stream

### Files Created

#### Core API Files
- **src/legalkit/api/__init__.py** - Package init
- **src/legalkit/api/models.py** - Pydantic v2 models (DocumentInfo, AskRequest, HealthResponse)
- **src/legalkit/api/store.py** - Filesystem document store with pickle + JSON metadata
- **src/legalkit/api/app.py** - Main FastAPI application with CORS and frontend serving
- **src/legalkit/api/routes/__init__.py** - Routes package init
- **src/legalkit/api/routes/documents.py** - Document CRUD endpoints
- **src/legalkit/api/routes/query.py** - SSE streaming query endpoint

#### Scripts
- **scripts/serve.py** - Server launcher script with CLI args

#### Tests
- **tests/test_api.py** - Comprehensive test suite (17 tests)

### Dependencies Added (pyproject.toml)
- `fastapi>=0.115`
- `uvicorn[standard]>=0.30`
- `python-multipart>=0.0.9`
- `httpx>=0.27` (dev dependency for testing)

## Test Results

✅ **All 17 API tests passing**
✅ **All 820 total tests passing** (no regressions)

### API Test Coverage
1. Health endpoint
2. List documents (empty state)
3. Get/delete nonexistent document (404 handling)
4. Upload invalid file type (400 validation)
5. Upload without filename (422 validation)
6. Upload valid .txt file (full pipeline mocked)
7. DocumentStore save/list/get/delete round-trip
8. DocumentStore load pipeline data
9. SSE event formatting
10. AskRequest validation
11. SSE streaming query endpoint (full pipeline mocked)

## Key Features

### Document Store
- Wraps existing `.legalkit_cache/` pickle mechanism
- Adds JSON metadata files alongside pickles
- Supports list, get, save, load, delete, exists operations
- Thread-safe file operations

### Document Ingestion
- Supports .txt, .pdf, .docx files
- Full pipeline: extract → parse → chunk → embed → index
- Skips contextual enrichment for MVP speed (--no-context mode)
- Returns DocumentInfo with metadata

### SSE Streaming Query
- Three progress stages: retrieving, generating, verifying
- Runs sync pipeline code in async executor
- Streams final result as EvidenceReport.to_dict()
- Error handling with error events

### Frontend Serving
- Serves static Elm frontend from frontend/dist/
- SPA routing (all routes serve index.html)
- CORS enabled for local dev (ports 1234, 3000)

## Usage

### Start Server
```bash
# Default (port 8000, auto-reload)
uv run python scripts/serve.py

# Custom port
uv run python scripts/serve.py --port 9000

# No auto-reload
uv run python scripts/serve.py --no-reload

# Bind to specific host
uv run python scripts/serve.py --host 127.0.0.1
```

### API Documentation
Once started, visit:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

### Test API
```bash
# Run API tests only
uv run pytest tests/test_api.py -v

# Run all tests
uv run pytest tests/ -v
```

## Next Steps (Frontend Integration)

The API is ready for the Elm frontend to integrate:
1. Document upload form → `POST /api/documents`
2. Document list → `GET /api/documents`
3. Query interface → `POST /api/documents/{id}/ask` (SSE)
4. Result rendering from EvidenceReport.to_dict() schema

All endpoints follow the exact contract specified in the task requirements.
