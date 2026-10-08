"""FastAPI application for LegalKit.

Serves:
  /api/*         — REST + SSE API
  /*             — Static Elm frontend (from frontend/dist/)

Run with:
  uvicorn legalkit.api.app:app --reload --port 8000
  or
  uv run python scripts/serve.py
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from legalkit.api.routes import documents, query

app = FastAPI(title="LegalKit API", version="0.1.0")

# CORS for local development (Elm dev server on port 1234 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1234", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Include API routes
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/api/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


FRONTEND_DIST = Path("frontend/dist")


@app.get("/")
async def serve_index():
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"error": "Frontend not built"}


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve the Elm SPA.

    If the requested path matches an actual file in frontend/dist/ (e.g.
    elm.js, main.js), serve that file with the correct MIME type. Otherwise
    fall back to index.html so client-side routing works.
    """
    _MIME = {
        ".js":   "application/javascript",
        ".css":  "text/css",
        ".html": "text/html",
        ".json": "application/json",
        ".png":  "image/png",
        ".svg":  "image/svg+xml",
        ".ico":  "image/x-icon",
    }

    candidate = FRONTEND_DIST / full_path
    if candidate.exists() and candidate.is_file():
        suffix = candidate.suffix.lower()
        media_type = _MIME.get(suffix, "application/octet-stream")
        return FileResponse(str(candidate), media_type=media_type)

    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")

    return {"error": "Frontend not built. Run: elm make src/Main.elm --output=dist/elm.js"}
