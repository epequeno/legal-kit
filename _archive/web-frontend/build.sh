#!/bin/bash
# Build script for LegalKit frontend

set -e

echo "🔨 Building LegalKit frontend..."

# Check if elm is installed
if ! command -v elm &> /dev/null; then
    echo "❌ Error: elm is not installed"
    echo "Install with: npm install -g elm"
    exit 1
fi

# Compile Elm
echo "📦 Compiling Elm code..."
elm make src/Main.elm --output=elm.js

echo "✅ Build complete!"
echo ""
echo "To run the app:"
echo "  1. Start the FastAPI backend on http://localhost:8000"
echo "  2. Serve this directory with: python -m http.server 8080"
echo "  3. Open http://localhost:8080"
