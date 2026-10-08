"""Publish the fine-tuned legal embeddings model to HuggingFace Hub.

Usage:
    uv run python scripts/publish_embeddings_to_hf.py
    uv run python scripts/publish_embeddings_to_hf.py --repo epequeno/legal-embeddings-bge-base
    uv run python scripts/publish_embeddings_to_hf.py --private
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = "epequeno/legal-embeddings-bge-base"
MODEL_DIR = Path("models/legal-embeddings/model")


def publish(repo_id: str, private: bool = False) -> None:
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Download from Modal first:\n"
            "  modal volume get legalkit-data models/legal-embeddings/model "
            "models/legal-embeddings/model"
        )

    required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    present = {f.name for f in MODEL_DIR.iterdir()}
    missing = required - present
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    print(f"Publishing {MODEL_DIR} → {repo_id}")
    print(f"  Private: {private}")

    url = create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    print(f"  Repo URL: {url}")

    print("  Uploading model files...")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add fine-tuned legal embeddings model (LegalKit v0.1)",
    )

    print(f"\n✓ Published to https://huggingface.co/{repo_id}")
    print(f"\nLoad it with:")
    print(f'  from sentence_transformers import SentenceTransformer')
    print(f'  model = SentenceTransformer("{repo_id}")')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish embeddings model to HuggingFace Hub")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    publish(args.repo, args.private)
