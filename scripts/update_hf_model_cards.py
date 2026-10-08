"""Update model card (README.md) on HuggingFace without re-uploading weights.

Usage:
    python scripts/update_hf_model_cards.py              # Update both
    python scripts/update_hf_model_cards.py --embeddings  # Embeddings only
    python scripts/update_hf_model_cards.py --entailment  # Entailment only
    python scripts/update_hf_model_cards.py --dry-run     # Show what would be uploaded
"""

import argparse
from pathlib import Path


MODELS = {
    "embeddings": {
        "repo": "epequeno/legal-embeddings-bge-base",
        "readme": Path("models/legal-embeddings/model/README.md"),
    },
    "entailment": {
        "repo": "epequeno/legal-entailment-deberta-v3-large",
        "readme": Path("models/legal-entailment/model/README.md"),
    },
}


def update_model_card(name: str, repo: str, readme_path: Path, dry_run: bool = False):
    """Upload updated README.md to HuggingFace."""
    if not readme_path.exists():
        print(f"  ❌ README not found: {readme_path}")
        return

    content = readme_path.read_text()
    print(f"\n{'='*60}")
    print(f"  Model: {name}")
    print(f"  Repo:  {repo}")
    print(f"  File:  {readme_path}")
    print(f"  Size:  {len(content):,} bytes")
    print(f"{'='*60}")

    if dry_run:
        # Show first 20 lines
        lines = content.split("\n")[:20]
        for line in lines:
            print(f"  | {line}")
        print(f"  | ... ({len(content.split(chr(10)))} total lines)")
        print(f"  [DRY RUN — would upload to {repo}]")
        return

    from huggingface_hub import HfApi
    api = HfApi()

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo,
        commit_message=f"Update model card with honest evaluation results (LODO, expanded NLI, label audit)",
    )
    print(f"  ✓ Uploaded to https://huggingface.co/{repo}")


def main():
    parser = argparse.ArgumentParser(description="Update HuggingFace model cards")
    parser.add_argument("--embeddings", action="store_true", help="Update embeddings model only")
    parser.add_argument("--entailment", action="store_true", help="Update entailment model only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    # If neither flag, update both
    if not args.embeddings and not args.entailment:
        targets = ["embeddings", "entailment"]
    else:
        targets = []
        if args.embeddings:
            targets.append("embeddings")
        if args.entailment:
            targets.append("entailment")

    for name in targets:
        info = MODELS[name]
        update_model_card(name, info["repo"], info["readme"], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
