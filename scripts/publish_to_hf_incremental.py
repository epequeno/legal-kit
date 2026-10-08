"""Upload model files individually to HuggingFace Hub with retry logic."""

from __future__ import annotations

import time
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "epequeno/legal-entailment-deberta-v3-large"
MODEL_DIR = Path("models/legal-entailment/model")


def upload_file_with_retry(api: HfApi, file_path: Path, retries: int = 3) -> bool:
    """Upload a single file with retry logic."""
    for attempt in range(retries):
        try:
            print(f"  Uploading {file_path.name}... (attempt {attempt + 1}/{retries})")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  ✓ {file_path.name} uploaded")
            return True
        except Exception as e:
            print(f"  ✗ Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    return False


def main():
    api = HfApi()
    
    # Get all files to upload
    files = sorted(MODEL_DIR.glob("*"))
    files = [f for f in files if f.is_file()]
    
    print(f"Found {len(files)} files to upload:")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name:40s} {size_mb:8.1f} MB")
    
    print(f"\nUploading to {REPO_ID}...")
    
    success = []
    failed = []
    
    for file_path in files:
        if upload_file_with_retry(api, file_path):
            success.append(file_path.name)
        else:
            failed.append(file_path.name)
            print(f"\n⚠ Failed to upload {file_path.name} after retries")
    
    print(f"\n{'='*60}")
    print(f"Upload complete!")
    print(f"  Success: {len(success)}/{len(files)}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"\nModel available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
