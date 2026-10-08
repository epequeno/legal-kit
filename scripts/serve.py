"""Start the LegalKit API server.

Usage:
    uv run python scripts/serve.py
    uv run python scripts/serve.py --port 8000 --reload
    uv run python scripts/serve.py --host 0.0.0.0 --no-reload
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start the LegalKit API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--reload",
        action="store_true",
        default=True,
        help="Enable auto-reload on code changes",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload",
    )
    args = parser.parse_args()

    # Build uvicorn command
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "legalkit.api.app:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    # Add reload flag if not explicitly disabled
    if args.reload and not args.no_reload:
        cmd.append("--reload")

    print(f"Starting LegalKit API server on http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
