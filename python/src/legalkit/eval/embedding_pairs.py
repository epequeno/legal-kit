"""Build training pairs for legal retrieval embedding fine-tuning.

Extracts (query, positive_passage) pairs from the LegalBench-RAG benchmark
ground-truth span annotations, then mines hard negatives from the same
documents. This gives a training set grounded in real legal retrieval
annotations rather than synthetic data.

Usage:
    legalkit build-embedding-pairs
    legalkit build-embedding-pairs --max-per-dataset 500
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path("data/corpus")
BENCHMARKS_DIR = Path("data/benchmarks")
DATASET_NAMES = ["cuad", "maud", "contractnli", "privacy_qa"]


@dataclass
class EmbeddingPair:
    query: str
    positive: str          # the ground-truth relevant passage
    hard_negative: str     # a passage from the same doc that is NOT the answer
    dataset: str           # 'cuad', 'maud', 'contractnli', 'privacy_qa'
    doc_id: str
    span: tuple[int, int]  # char span of positive in source doc

    def to_dict(self) -> dict:
        return {
            'query': self.query,
            'positive': self.positive,
            'hard_negative': self.hard_negative,
            'dataset': self.dataset,
            'doc_id': self.doc_id,
        }


def extract_pairs_from_benchmark(
    dataset_name: str,          # 'cuad', 'maud', 'contractnli', 'privacy_qa'
    max_n: int | None = None,
    min_positive_chars: int = 50,   # skip very short answer spans
    hard_neg_window: int = 3000,    # chars to sample hard negative from
) -> list[EmbeddingPair]:
    """Extract (query, positive, hard_negative) triples from a benchmark dataset.
    
    Positive: text of the ground-truth span (answer) from the corpus file.
    Hard negative: text from the same document, same approximate position,
    but NOT overlapping with the positive span. Specifically: take the
    window of `hard_neg_window` chars centered on the positive span, 
    then sample a passage NOT overlapping the positive.
    
    Args:
        dataset_name: One of the 4 benchmark datasets.
        max_n: Cap on number of pairs to extract.
        min_positive_chars: Skip entries where answer span is too short.
        hard_neg_window: Window around span for mining hard negatives.
    
    Returns:
        List of EmbeddingPair objects.
    """
    bench_path = BENCHMARKS_DIR / f"{dataset_name}.json"
    if not bench_path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {bench_path}. "
            f"Run 'legalkit bench --download' first."
        )
    
    with open(bench_path) as f:
        data = json.load(f)
    
    # Handle both dict with "tests" key and plain list
    tests = data.get("tests", data) if isinstance(data, dict) else data
    if isinstance(tests, dict):
        tests = tests.get("tests", [])
    
    pairs: list[EmbeddingPair] = []
    seen_keys: set[tuple[str, str]] = set()  # (query, positive) deduplication
    
    for test in tests[:max_n] if max_n else tests:
        query = test["query"]
        
        for snippet in test.get("snippets", []):
            file_path = snippet["file_path"]
            span = snippet["span"]
            if isinstance(span, list):
                span = tuple(span)
            
            # Load corpus file
            corpus_file = CORPUS_DIR / file_path
            if not corpus_file.exists():
                # Skip with warning, don't crash
                print(f"Warning: corpus file not found: {corpus_file}")
                continue
            
            try:
                with open(corpus_file, encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Warning: failed to read {corpus_file}: {e}")
                continue
            
            # Extract positive
            span_start, span_end = span
            positive = text[span_start:span_end].strip()
            
            # Skip if too short
            if len(positive) < min_positive_chars:
                continue
            
            # Deduplicate
            key = (query, positive)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            # Mine hard negative
            hard_negative = _mine_hard_negative(
                text=text,
                positive_span=(span_start, span_end),
                window=hard_neg_window,
            )
            
            doc_id = Path(file_path).stem
            
            pairs.append(EmbeddingPair(
                query=query,
                positive=positive,
                hard_negative=hard_negative,
                dataset=dataset_name,
                doc_id=doc_id,
                span=(span_start, span_end),
            ))
    
    return pairs


def _mine_hard_negative(
    text: str,
    positive_span: tuple[int, int],
    window: int = 3000,
    target_chunk_size: int = 500,
) -> str:
    """Mine a hard negative passage from the same document.
    
    Strategy:
    - Take a window around the positive span
    - Split into ~500-char chunks
    - Pick a chunk that doesn't overlap with the positive span
    - If no suitable chunk, take the first paragraph
    """
    span_start, span_end = positive_span
    
    # Define window boundaries
    window_start = max(0, span_start - window)
    window_end = min(len(text), span_end + window)
    window_text = text[window_start:window_end]
    
    # Split into chunks
    chunks: list[tuple[int, int, str]] = []  # (abs_start, abs_end, text)
    pos = 0
    while pos < len(window_text):
        chunk_end = min(pos + target_chunk_size, len(window_text))
        
        # Try to break at sentence boundary
        if chunk_end < len(window_text):
            # Look for sentence enders
            for punct in ['. ', '.\n', '! ', '?\n', '? ']:
                idx = window_text.rfind(punct, pos, chunk_end)
                if idx > pos + target_chunk_size // 2:
                    chunk_end = idx + 1
                    break
        
        chunk_text = window_text[pos:chunk_end].strip()
        abs_start = window_start + pos
        abs_end = window_start + chunk_end
        
        if chunk_text:
            chunks.append((abs_start, abs_end, chunk_text))
        
        pos = chunk_end
    
    # Filter out chunks that overlap with positive span
    non_overlapping = [
        chunk for chunk in chunks
        if not _spans_overlap(
            (chunk[0], chunk[1]),
            positive_span
        )
    ]
    
    if non_overlapping:
        # Pick a random non-overlapping chunk
        return random.choice(non_overlapping)[2]
    
    # Fallback: take first paragraph from document
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        para = para.strip()
        if len(para) >= 200:  # reasonable length
            # Make sure it's not the positive
            para_start = text.find(para)
            para_end = para_start + len(para)
            if not _spans_overlap((para_start, para_end), positive_span):
                return para
    
    # Last resort: take the very beginning of the document
    fallback = text[:target_chunk_size].strip()
    return fallback if fallback else "No suitable hard negative found."


def _spans_overlap(span1: tuple[int, int], span2: tuple[int, int]) -> bool:
    """Check if two char spans overlap."""
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])


def build_training_data(
    datasets: list[str] | None = None,
    output_dir: Path | None = None,   # default: data/embedding_training/
    max_per_dataset: int | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Build full train/val split from all benchmark datasets.
    
    Returns stats dict:
    {
        'total': int,
        'train': int,
        'val': int,
        'by_dataset': {'cuad': n, 'maud': n, ...},
        'avg_positive_chars': float,
        'avg_hard_neg_chars': float,
    }
    """
    datasets = datasets or DATASET_NAMES
    output_dir = output_dir or Path("data/embedding_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(seed)
    
    # Collect all pairs
    all_pairs: list[EmbeddingPair] = []
    by_dataset: dict[str, int] = {}
    
    for ds_name in datasets:
        pairs = extract_pairs_from_benchmark(
            dataset_name=ds_name,
            max_n=max_per_dataset,
        )
        all_pairs.extend(pairs)
        by_dataset[ds_name] = len(pairs)
    
    # Shuffle
    random.shuffle(all_pairs)
    
    # Split train/val
    n_val = int(len(all_pairs) * val_ratio)
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]
    
    # Write JSONL files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    with open(train_path, 'w') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair.to_dict()) + '\n')
    
    with open(val_path, 'w') as f:
        for pair in val_pairs:
            f.write(json.dumps(pair.to_dict()) + '\n')
    
    # Compute stats
    avg_positive_chars = sum(len(p.positive) for p in all_pairs) / len(all_pairs)
    avg_hard_neg_chars = sum(len(p.hard_negative) for p in all_pairs) / len(all_pairs)
    
    return {
        'total': len(all_pairs),
        'train': len(train_pairs),
        'val': len(val_pairs),
        'by_dataset': by_dataset,
        'avg_positive_chars': round(avg_positive_chars, 1),
        'avg_hard_neg_chars': round(avg_hard_neg_chars, 1),
        'output_dir': str(output_dir),
        'train_path': str(train_path),
        'val_path': str(val_path),
    }
