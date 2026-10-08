"""ContractNLI document-level NLI evaluation.

Evaluates the full pipeline (retrieval → entailment) on the ContractNLI
benchmark (Koreeda & Manning, EMNLP 2021). Each example is a full NDA
document + a natural language hypothesis; the pipeline must predict
Entailed / NotMentioned / Contradicted.

This benchmarks different aspects than LegalBench-RAG (retrieval recall)
or the NLI edge cases (per-pattern accuracy) — it measures end-to-end
pipeline performance on realistic document-level NLI at scale.

Dataset: https://stanfordnlp.github.io/contract-nli/
HuggingFace: koreeda/ContractNLI (or stanfordnlp/contract_nli)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path("data")
_CONTRACTNLI_CACHE_DIR = _DATA_DIR / "contractnli"

# HuggingFace dataset identifiers to try in order
_HF_DATASET_IDS = [
    "koreeda/ContractNLI",
    "stanfordnlp/contract_nli",
]

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ContractNLIExample:
    """A single ContractNLI document-hypothesis pair."""

    doc_id: str
    document_text: str
    hypothesis: str
    gold_label: str  # "Entailed" | "NotMentioned" | "Contradicted"
    gold_spans: list[tuple[int, int]]  # character spans supporting the label
    hypothesis_id: str = ""


@dataclass
class ContractNLIPrediction:
    """Model prediction for one example."""

    example: ContractNLIExample
    predicted_label: str  # "Entailed" | "NotMentioned" | "Contradicted"
    confidence: float  # 0-1
    retrieved_spans: list[tuple[int, int]]  # which spans were retrieved
    entailment_score: float  # raw entailment score from verifier
    retrieval_recall: float  # overlap of retrieved spans with gold spans
    latency_ms: float


@dataclass
class ContractNLIReport:
    """Aggregate results across all evaluated examples."""

    model_name: str
    predictions: list[ContractNLIPrediction] = field(default_factory=list)

    def accuracy(self) -> float:
        """3-class accuracy."""
        if not self.predictions:
            return 0.0
        correct = sum(
            1
            for p in self.predictions
            if p.predicted_label == p.example.gold_label
        )
        return correct / len(self.predictions)

    def per_label_accuracy(self) -> dict[str, float]:
        """Accuracy per gold label."""
        from collections import defaultdict

        by_label: dict[str, list[bool]] = defaultdict(list)
        for p in self.predictions:
            correct = p.predicted_label == p.example.gold_label
            by_label[p.example.gold_label].append(correct)

        return {
            label: sum(results) / len(results) if results else 0.0
            for label, results in sorted(by_label.items())
        }

    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        """3×3 confusion matrix: {true_label: {predicted_label: count}}."""
        labels = ["Entailed", "NotMentioned", "Contradicted"]
        matrix: dict[str, dict[str, int]] = {
            true_label: {pred_label: 0 for pred_label in labels}
            for true_label in labels
        }
        for p in self.predictions:
            pred = (
                p.predicted_label
                if p.predicted_label in labels
                else "NotMentioned"
            )
            matrix[p.example.gold_label][pred] += 1
        return matrix

    def retrieval_recall(self) -> float:
        """Average span recall for Entailed+Contradicted examples only."""
        relevant = [
            p
            for p in self.predictions
            if p.example.gold_label in ("Entailed", "Contradicted")
        ]
        if not relevant:
            return 0.0
        return sum(p.retrieval_recall for p in relevant) / len(relevant)

    def not_mentioned_precision(self) -> float:
        """Fraction of NotMentioned correctly not over-retrieved."""
        nm_examples = [
            p for p in self.predictions if p.example.gold_label == "NotMentioned"
        ]
        if not nm_examples:
            return 0.0
        correct = sum(1 for p in nm_examples if p.predicted_label == "NotMentioned")
        return correct / len(nm_examples)

    def to_dict(self) -> dict:
        """Full serialisable report."""
        return {
            "model_name": self.model_name,
            "n_predictions": len(self.predictions),
            "accuracy": round(self.accuracy(), 4),
            "per_label_accuracy": {
                k: round(v, 4) for k, v in self.per_label_accuracy().items()
            },
            "confusion_matrix": self.confusion_matrix(),
            "retrieval_recall": round(self.retrieval_recall(), 4),
            "not_mentioned_precision": round(self.not_mentioned_precision(), 4),
            "predictions": [
                {
                    "doc_id": p.example.doc_id,
                    "hypothesis_id": p.example.hypothesis_id,
                    "hypothesis": p.example.hypothesis,
                    "gold_label": p.example.gold_label,
                    "predicted_label": p.predicted_label,
                    "confidence": round(p.confidence, 4),
                    "entailment_score": round(p.entailment_score, 4),
                    "retrieval_recall": round(p.retrieval_recall, 4),
                    "latency_ms": round(p.latency_ms, 2),
                }
                for p in self.predictions
            ],
        }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_contractnli(
    split: str = "test",
    max_n: int | None = None,
    data_dir: Path | None = None,
) -> list[ContractNLIExample]:
    """Load ContractNLI examples.

    Tries in order:
    1. Local cache: data/contractnli/<split>.json
    2. HuggingFace: koreeda/ContractNLI (with fallback to stanfordnlp/contract_nli)
    3. FileNotFoundError with download instructions

    Args:
        split: Dataset split (usually 'test' or 'train').
        max_n: Maximum examples to return. None = all.
        data_dir: Override default data directory.

    Returns:
        List of ContractNLIExample.

    Raises:
        FileNotFoundError: If neither local cache nor HF dataset is available.
    """
    cache_dir = Path(data_dir) if data_dir else _CONTRACTNLI_CACHE_DIR
    cache_file = cache_dir / f"{split}.json"

    # ── 1. Local cache ──────────────────────────────────────────────────
    if cache_file.exists():
        return _load_from_cache(cache_file, max_n)

    # ── 2. HuggingFace ──────────────────────────────────────────────────
    try:
        examples = _load_from_huggingface(split, max_n)
        # Save to cache for future runs
        _save_to_cache(examples, cache_file)
        return examples
    except Exception as hf_error:
        raise FileNotFoundError(
            f"ContractNLI dataset not found locally or on HuggingFace.\n"
            f"  HuggingFace error: {hf_error}\n"
            f"  Run 'legalkit contractnli-eval --download' to fetch the dataset.\n"
            f"  Or check: https://huggingface.co/datasets/koreeda/ContractNLI"
        ) from hf_error


def _load_from_cache(path: Path, max_n: int | None) -> list[ContractNLIExample]:
    """Load examples from a local JSON cache file."""
    with open(path) as f:
        data = json.load(f)

    examples = [
        ContractNLIExample(
            doc_id=ex["doc_id"],
            document_text=ex["document_text"],
            hypothesis=ex["hypothesis"],
            gold_label=ex["gold_label"],
            gold_spans=[tuple(span) for span in ex["gold_spans"]],
            hypothesis_id=ex.get("hypothesis_id", ""),
        )
        for ex in data
    ]
    return examples[:max_n] if max_n else examples


def _save_to_cache(examples: list[ContractNLIExample], path: Path) -> None:
    """Persist examples to local JSON cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "doc_id": ex.doc_id,
            "document_text": ex.document_text,
            "hypothesis": ex.hypothesis,
            "gold_label": ex.gold_label,
            "gold_spans": list(ex.gold_spans),
            "hypothesis_id": ex.hypothesis_id,
        }
        for ex in examples
    ]
    path.write_text(json.dumps(data, indent=2))


def _load_from_huggingface(
    split: str,
    max_n: int | None,
) -> list[ContractNLIExample]:
    """Load ContractNLI from HuggingFace datasets.

    Tries each dataset ID in _HF_DATASET_IDS in order.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required to load from HuggingFace.\n"
            "Install with: uv add datasets"
        ) from exc

    last_error: Exception | None = None
    for dataset_id in _HF_DATASET_IDS:
        try:
            ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
            if max_n:
                ds = ds.select(range(min(max_n, len(ds))))
            return [_parse_hf_row(row, i) for i, row in enumerate(ds)]
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not load ContractNLI from any known HF dataset. "
        f"Last error: {last_error}"
    )


def _parse_hf_row(row: dict, idx: int) -> ContractNLIExample:
    """Normalise a HuggingFace dataset row into a ContractNLIExample.

    ContractNLI HF rows can have multiple formats:
    - {"document": "...", "annotations": [{...}], "hypothesis": "..."}
    - {"text": "...", "hypothesis": "...", "label": "Entailed"}
    """
    # Format 1: document + annotations
    if "document" in row and "annotations" in row:
        document = row["document"]
        hypothesis = row.get("hypothesis", "")
        annotations = row["annotations"]

        # Extract label and spans from first annotation
        if annotations and len(annotations) > 0:
            anno = annotations[0]
            label = anno.get("type", "NotMentioned")
            # Normalize label format
            label = _normalize_label(label)
            spans = [tuple(span) for span in anno.get("spans", [])]
        else:
            label = "NotMentioned"
            spans = []

        # Generate doc_id if not present
        doc_id = row.get("doc_id") or row.get("id") or f"doc_{idx}"

        return ContractNLIExample(
            doc_id=doc_id,
            document_text=document,
            hypothesis=hypothesis,
            gold_label=label,
            gold_spans=spans,
            hypothesis_id=row.get("hypothesis_id", f"hyp_{idx}"),
        )

    # Format 2: text + hypothesis + label
    if "text" in row and "hypothesis" in row:
        label = _normalize_label(row.get("label", "NotMentioned"))
        return ContractNLIExample(
            doc_id=row.get("doc_id") or row.get("id") or f"doc_{idx}",
            document_text=row["text"],
            hypothesis=row["hypothesis"],
            gold_label=label,
            gold_spans=[],
            hypothesis_id=row.get("hypothesis_id", f"hyp_{idx}"),
        )

    # Fallback: use whatever we have
    doc_text = row.get("document") or row.get("text") or str(row)
    return ContractNLIExample(
        doc_id=f"doc_{idx}",
        document_text=doc_text,
        hypothesis=row.get("hypothesis", ""),
        gold_label="NotMentioned",
        gold_spans=[],
        hypothesis_id=f"hyp_{idx}",
    )


def _normalize_label(label: str | int) -> str:
    """Normalize label to standard format."""
    if isinstance(label, int):
        # Some datasets use 0=Entailed, 1=NotMentioned, 2=Contradicted
        mapping = {0: "Entailed", 1: "NotMentioned", 2: "Contradicted"}
        return mapping.get(label, "NotMentioned")

    label_str = str(label).strip()
    # Handle various capitalizations
    lower = label_str.lower()
    if "entail" in lower:
        return "Entailed"
    if "contradict" in lower:
        return "Contradicted"
    if "not" in lower or "neutral" in lower:
        return "NotMentioned"
    return label_str


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def download_contractnli(
    split: str = "test",
    data_dir: Path | None = None,
) -> int:
    """Download and cache ContractNLI dataset from HuggingFace.

    Args:
        split: Which split to cache (usually 'test').
        data_dir: Override default data directory.

    Returns:
        Number of examples cached.
    """
    try:
        examples = _load_from_huggingface(split, max_n=None)
        cache_dir = Path(data_dir) if data_dir else _CONTRACTNLI_CACHE_DIR
        cache_file = cache_dir / f"{split}.json"
        _save_to_cache(examples, cache_file)
        return len(examples)
    except Exception as e:
        raise RuntimeError(f"Failed to download ContractNLI: {e}") from e


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def _predict_label(
    entailment_class: str,
    confidence: float,
    entailment_threshold: float,
    not_mentioned_threshold: float,
) -> tuple[str, float]:
    """Map internal 4-class output to ContractNLI 3-class labels.

    Returns (predicted_label, confidence) where label is
    "Entailed" | "NotMentioned" | "Contradicted"

    Mapping:
    - "entailed" + high conf → "Entailed"
    - "partially_entailed" + high conf → "Entailed" (conservative)
    - "contradicted" + high conf → "Contradicted"
    - "neutral" or low conf → "NotMentioned"
    """
    # Handle EntailmentClass enum if passed
    if hasattr(entailment_class, "value"):
        entailment_class = entailment_class.value

    ec = entailment_class.lower()

    if ec == "entailed" and confidence >= entailment_threshold:
        return "Entailed", confidence
    elif ec == "partially_entailed" and confidence >= entailment_threshold:
        return "Entailed", confidence
    elif ec == "contradicted" and confidence >= entailment_threshold:
        return "Contradicted", confidence
    elif confidence < not_mentioned_threshold:
        return "NotMentioned", confidence
    else:
        # Default to NotMentioned for neutral or ambiguous cases
        return "NotMentioned", confidence


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_contractnli_eval(
    examples: list[ContractNLIExample],
    llm,  # LLMClient
    top_k: int = 10,
    min_chars: int = 100,
    max_chars: int = 1500,
    entailment_threshold: float = 0.6,
    not_mentioned_threshold: float = 0.3,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ContractNLIReport:
    """Run pipeline on each example.

    For each example:
    1. Parse + chunk the document (using pipeline.parse + pipeline.chunk)
    2. Embed chunks (pipeline.embed)
    3. Retrieve top_k chunks for the hypothesis (pipeline.index search)
    4. Run LLM entailment check on top retrieved chunk (pipeline.verify._check_entailment_llm)
    5. Map score → label
    6. Compute retrieval_recall (span overlap with gold spans)

    Args:
        examples: List of ContractNLIExample to evaluate.
        llm: LLMClient instance.
        top_k: Number of chunks to retrieve per hypothesis.
        min_chars: Minimum chunk size in characters.
        max_chars: Maximum chunk size in characters.
        entailment_threshold: Score threshold for Entailed/Contradicted.
        not_mentioned_threshold: Score threshold below which → NotMentioned.
        progress_callback: Optional (completed, total) -> None.

    Returns:
        ContractNLIReport with all predictions.
    """
    from legalkit.pipeline.parse import parse_document
    from legalkit.pipeline.chunk import chunk_document
    from legalkit.pipeline.embed import embed_texts
    from legalkit.pipeline.index import DocumentIndex
    from legalkit.pipeline.verify import _check_entailment_llm

    # Group examples by document to cache parsed/chunked documents
    from collections import defaultdict

    examples_by_doc: dict[str, list[ContractNLIExample]] = defaultdict(list)
    for ex in examples:
        examples_by_doc[ex.doc_id].append(ex)

    predictions: list[ContractNLIPrediction] = []
    total_processed = 0

    for doc_id, doc_examples in examples_by_doc.items():
        # Get document text (same for all examples with this doc_id)
        doc_text = doc_examples[0].document_text

        # ── 1-2. Parse + chunk ─────────────────────────────────────────
        try:
            root = parse_document(doc_text, doc_id=doc_id)
            chunks = chunk_document(
                root, doc_id=doc_id, min_chars=min_chars, max_chars=max_chars
            )
        except Exception:
            # If parsing fails, treat as one big chunk
            chunks = []

        if not chunks:
            # Fallback: create a single chunk from the whole document
            from legalkit.models.retrieval import Chunk

            chunks = [
                Chunk(
                    id=f"{doc_id}_full",
                    node_id=doc_id,
                    text=doc_text[:max_chars * 3],  # cap at 3x max
                    char_start=0,
                    char_end=min(len(doc_text), max_chars * 3),
                    node_path=[doc_id],
                )
            ]

        # ── 3. Embed ───────────────────────────────────────────────────
        try:
            embedding_texts = [c.embedding_text for c in chunks]
            embeddings = embed_texts(embedding_texts)

            # Build index
            index = DocumentIndex()
            index.build(chunks, embeddings)
        except Exception:
            # If embedding fails, skip this document
            for ex in doc_examples:
                predictions.append(
                    ContractNLIPrediction(
                        example=ex,
                        predicted_label="NotMentioned",
                        confidence=0.0,
                        retrieved_spans=[],
                        entailment_score=0.0,
                        retrieval_recall=0.0,
                        latency_ms=0.0,
                    )
                )
                total_processed += 1
                if progress_callback:
                    progress_callback(total_processed, len(examples))
            continue

        # ── 4-6. Retrieve + verify for each hypothesis ────────────────
        for ex in doc_examples:
            t0 = time.perf_counter()

            try:
                # Embed hypothesis
                hyp_embeddings = embed_texts([ex.hypothesis])
                hyp_emb = hyp_embeddings[0]

                # Retrieve top-k chunks
                results = index.search_hybrid(
                    query_embedding=hyp_emb,
                    query_text=ex.hypothesis,
                    top_k=top_k,
                )

                if not results:
                    # No chunks retrieved
                    predictions.append(
                        ContractNLIPrediction(
                            example=ex,
                            predicted_label="NotMentioned",
                            confidence=0.0,
                            retrieved_spans=[],
                            entailment_score=0.0,
                            retrieval_recall=0.0,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                        )
                    )
                    total_processed += 1
                    if progress_callback:
                        progress_callback(total_processed, len(examples))
                    continue

                # Get top chunk
                top_chunk_idx = results[0][0]
                top_chunk = chunks[top_chunk_idx]
                retrieved_spans = [(top_chunk.char_start, top_chunk.char_end)]

                # Run entailment check
                entailment_class, entailment_score, explanation = (
                    _check_entailment_llm(
                        section_id=top_chunk.node_id,
                        passage=top_chunk.text,
                        claim=ex.hypothesis,
                        llm=llm,
                    )
                )

                # Map to 3-class label
                predicted_label, conf = _predict_label(
                    entailment_class=entailment_class,
                    confidence=entailment_score,
                    entailment_threshold=entailment_threshold,
                    not_mentioned_threshold=not_mentioned_threshold,
                )

                # Compute retrieval recall
                ret_recall = _compute_retrieval_recall(
                    retrieved_spans=retrieved_spans, gold_spans=ex.gold_spans
                )

                predictions.append(
                    ContractNLIPrediction(
                        example=ex,
                        predicted_label=predicted_label,
                        confidence=conf,
                        retrieved_spans=retrieved_spans,
                        entailment_score=entailment_score,
                        retrieval_recall=ret_recall,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                )

            except Exception:
                # Error during processing
                predictions.append(
                    ContractNLIPrediction(
                        example=ex,
                        predicted_label="NotMentioned",
                        confidence=0.0,
                        retrieved_spans=[],
                        entailment_score=0.0,
                        retrieval_recall=0.0,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                )

            total_processed += 1
            if progress_callback:
                progress_callback(total_processed, len(examples))

    return ContractNLIReport(
        model_name=llm.config.generation_model, predictions=predictions
    )


def _compute_retrieval_recall(
    retrieved_spans: list[tuple[int, int]],
    gold_spans: list[tuple[int, int]],
) -> float:
    """Compute character-level recall of retrieved spans vs gold spans."""
    if not gold_spans:
        return 0.0

    # Total gold chars
    total_gold = sum(end - start for start, end in gold_spans)
    if total_gold == 0:
        return 0.0

    # Compute overlap
    covered = 0
    for g_start, g_end in gold_spans:
        for r_start, r_end in retrieved_spans:
            overlap_start = max(g_start, r_start)
            overlap_end = min(g_end, r_end)
            if overlap_end > overlap_start:
                covered += overlap_end - overlap_start

    return covered / total_gold
