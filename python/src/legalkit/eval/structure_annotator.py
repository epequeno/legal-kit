"""Generate BIO-labelled training data for the document structure model.

Uses the existing regex parser as a weak supervisor: run it over the CUAD/MAUD/
ContractNLI corpus, map the resulting DocumentNode types to BIO labels, and
produce a JSONL training file where each line is one document's worth of
line-level annotations.

This is the 'distant/weak supervision' strategy validated by TocBERT (Baghdadi
et al.) and used in CUAD itself — auto-label from a reliable heuristic source,
then fine-tune a transformer that learns to generalise beyond the heuristic.

Usage:
    python -m legalkit.eval.structure_annotator
    legalkit structure-annotate --corpus data/corpus/cuad
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from legalkit.models.document import DocumentNode, NodeType
from legalkit.pipeline.parse import parse_document


# ---------------------------------------------------------------------------
# BIO Label Taxonomy
# ---------------------------------------------------------------------------

LABELS = [
    "O",                    # outside (body text, blank lines)
    "B-ARTICLE_HEADER",     # "ARTICLE IV - INDEMNIFICATION"
    "I-ARTICLE_HEADER",
    "B-SECTION_HEADER",     # "Section 4.2 Cap on Liability"
    "I-SECTION_HEADER",
    "B-SUBSECTION_HEADER",  # "4.2.1 General Rule" or "(a)" at start of line
    "I-SUBSECTION_HEADER",
    "B-PREAMBLE",           # Agreement preamble / parties block
    "I-PREAMBLE",
    "B-RECITAL",            # "WHEREAS, ..."
    "I-RECITAL",
    "B-EXHIBIT_HEADER",     # "EXHIBIT A — BILL OF SALE"
    "I-EXHIBIT_HEADER",
    "B-DEFINITION",         # "'Affiliate' means..."
    "I-DEFINITION",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedLine:
    """One line of a contract with its BIO label."""

    text: str           # raw line text
    label: str          # BIO label string e.g. "B-SECTION_HEADER"
    label_id: int       # integer index
    line_number: int    # 1-indexed line number in document
    char_start: int     # character offset in original document
    char_end: int


@dataclass
class AnnotatedDocument:
    """A fully annotated document."""

    doc_id: str
    file_path: str
    contract_type: str   # "cuad", "maud", "contractnli", "other"
    lines: list[AnnotatedLine]

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def label_distribution(self) -> dict[str, int]:
        return dict(Counter(l.label for l in self.lines))

    def to_dict(self) -> dict:
        """Serialise to JSONL-compatible dict."""
        return {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "contract_type": self.contract_type,
            "lines": [
                {
                    "text": l.text,
                    "label": l.label,
                    "label_id": l.label_id,
                    "line_number": l.line_number,
                }
                for l in self.lines
            ],
        }


# ---------------------------------------------------------------------------
# Annotation Logic
# ---------------------------------------------------------------------------


def _node_type_to_bio_prefix(node_type: NodeType) -> str | None:
    """Map a NodeType to its BIO label prefix (without B- or I-)."""
    mapping = {
        NodeType.ARTICLE: "ARTICLE_HEADER",
        NodeType.SECTION: "SECTION_HEADER",
        NodeType.SUBSECTION: "SUBSECTION_HEADER",
        NodeType.CLAUSE: "SUBSECTION_HEADER",
        NodeType.SUBCLAUSE: "SUBSECTION_HEADER",
        NodeType.PREAMBLE: "PREAMBLE",
        NodeType.RECITAL: "RECITAL",
        NodeType.EXHIBIT: "EXHIBIT_HEADER",
        NodeType.SCHEDULE: "EXHIBIT_HEADER",
        NodeType.DEFINITION: "DEFINITION",
    }
    return mapping.get(node_type)


def annotate_document(
    text: str,
    doc_id: str = "",
    file_path: str = "",
    contract_type: str = "other",
) -> AnnotatedDocument:
    """Parse a contract and produce BIO-labelled lines.

    Algorithm:
    1. Run the existing parse_document() to get a DocumentNode tree
    2. For each DocumentNode, find which lines of the original text it covers
    3. Assign BIO labels to those lines based on NodeType:
       - NodeType.ARTICLE -> B-ARTICLE_HEADER / I-ARTICLE_HEADER
       - NodeType.SECTION -> B-SECTION_HEADER / I-SECTION_HEADER
       - NodeType.SUBSECTION / NodeType.CLAUSE -> B-SUBSECTION_HEADER / I-SUBSECTION_HEADER
       - NodeType.PREAMBLE -> B-PREAMBLE / I-PREAMBLE
       - NodeType.RECITAL -> B-RECITAL / I-RECITAL
       - NodeType.EXHIBIT / NodeType.SCHEDULE -> B-EXHIBIT_HEADER / I-EXHIBIT_HEADER
       - NodeType.DEFINITION -> B-DEFINITION / I-DEFINITION
       - Everything else -> O
    4. Lines that the parser missed entirely -> O

    Important: Only label the HEADING LINE of each node (the first line of
    title text), not the body content. Body text -> O. This makes the task
    a heading-detection problem, not a full semantic labelling problem.
    """
    # Handle empty text
    if not text.strip():
        return AnnotatedDocument(
            doc_id=doc_id or "empty",
            file_path=file_path,
            contract_type=contract_type,
            lines=[],
        )

    # Parse the document
    try:
        doc = parse_document(text, doc_id=doc_id or "doc")
    except Exception:
        # If parsing fails, label everything as O
        lines = []
        char_offset = 0
        for line_num, line_text in enumerate(text.splitlines(), 1):
            lines.append(
                AnnotatedLine(
                    text=line_text,
                    label="O",
                    label_id=LABEL2ID["O"],
                    line_number=line_num,
                    char_start=char_offset,
                    char_end=char_offset + len(line_text),
                )
            )
            char_offset += len(line_text) + 1  # +1 for newline
        return AnnotatedDocument(
            doc_id=doc_id or "error",
            file_path=file_path,
            contract_type=contract_type,
            lines=lines,
        )

    # Split text into lines with their char ranges
    line_info = []
    char_offset = 0
    for line_text in text.splitlines():
        line_info.append({
            "text": line_text,
            "char_start": char_offset,
            "char_end": char_offset + len(line_text),
        })
        char_offset += len(line_text) + 1  # +1 for newline

    # Initialize all lines as "O"
    line_labels = ["O"] * len(line_info)

    # Get all nodes from the tree (excluding the root DOCUMENT node)
    all_nodes = [n for n in doc.all_nodes() if n.type != NodeType.DOCUMENT]

    # For each node, find the line(s) that contain its heading/title
    for node in all_nodes:
        bio_prefix = _node_type_to_bio_prefix(node.type)
        if bio_prefix is None:
            continue

        # Find lines that overlap with this node's heading
        # For most nodes, the heading is at the char_start position
        heading_start = node.char_start
        heading_end = node.char_start  # Will be extended

        # The heading typically spans from char_start to the first newline
        # Or to the end of the title if title is captured
        if node.title:
            # Find where the title appears in the text
            title_pos = text.find(node.title, heading_start, heading_start + 500)
            if title_pos >= 0:
                heading_end = title_pos + len(node.title)
            else:
                # Title might be split across lines or reformatted
                # Use the first line of the node as heading
                first_newline = text.find("\n", heading_start)
                if first_newline >= 0 and first_newline < node.char_end:
                    heading_end = first_newline
                else:
                    heading_end = min(heading_start + 200, node.char_end)
        else:
            # No title — use the first line
            first_newline = text.find("\n", heading_start)
            if first_newline >= 0 and first_newline < node.char_end:
                heading_end = first_newline
            else:
                heading_end = min(heading_start + 200, node.char_end)

        # Find all lines that overlap with [heading_start, heading_end]
        affected_lines = []
        for line_idx, line in enumerate(line_info):
            # Check if line overlaps with heading range
            if line["char_end"] >= heading_start and line["char_start"] <= heading_end:
                affected_lines.append(line_idx)

        # Label these lines with B- and I- tags
        for i, line_idx in enumerate(affected_lines):
            if i == 0:
                line_labels[line_idx] = f"B-{bio_prefix}"
            else:
                line_labels[line_idx] = f"I-{bio_prefix}"

    # Build AnnotatedLine objects
    annotated_lines = []
    for line_num, (line, label) in enumerate(zip(line_info, line_labels), 1):
        annotated_lines.append(
            AnnotatedLine(
                text=line["text"],
                label=label,
                label_id=LABEL2ID[label],
                line_number=line_num,
                char_start=line["char_start"],
                char_end=line["char_end"],
            )
        )

    return AnnotatedDocument(
        doc_id=doc_id or "doc",
        file_path=file_path,
        contract_type=contract_type,
        lines=annotated_lines,
    )


# ---------------------------------------------------------------------------
# Corpus Processing
# ---------------------------------------------------------------------------


def annotate_corpus(
    corpus_dir: Path,
    contract_type: str = "other",
    max_docs: int | None = None,
    skip_errors: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> list[AnnotatedDocument]:
    """Annotate all .txt files in a corpus directory."""
    corpus_path = Path(corpus_dir)
    txt_files = sorted(corpus_path.glob("*.txt"))

    if max_docs is not None:
        txt_files = txt_files[:max_docs]

    documents = []
    skipped = 0

    for txt_file in txt_files:
        if progress_callback:
            progress_callback(f"Processing {txt_file.name}")

        try:
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            doc_id = txt_file.stem
            annotated = annotate_document(
                text=text,
                doc_id=doc_id,
                file_path=str(txt_file),
                contract_type=contract_type,
            )
            documents.append(annotated)
        except Exception as e:
            if skip_errors:
                skipped += 1
                if progress_callback:
                    progress_callback(f"Skipped {txt_file.name}: {e}")
            else:
                raise

    return documents


def save_annotated_corpus(
    documents: list[AnnotatedDocument],
    output_path: Path,
) -> dict:
    """Save to JSONL, return stats dict.

    Returns: {total_docs, total_lines, label_distribution, skipped}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    all_labels: list[str] = []

    with output_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict()) + "\n")
            total_lines += doc.n_lines
            all_labels.extend(l.label for l in doc.lines)

    label_dist = dict(Counter(all_labels))

    return {
        "total_docs": len(documents),
        "total_lines": total_lines,
        "label_distribution": label_dist,
    }


def load_annotated_corpus(path: Path) -> list[AnnotatedDocument]:
    """Load from JSONL."""
    documents = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            lines = [
                AnnotatedLine(
                    text=l["text"],
                    label=l["label"],
                    label_id=l["label_id"],
                    line_number=l["line_number"],
                    char_start=0,  # Not stored in JSONL
                    char_end=0,
                )
                for l in data["lines"]
            ]
            documents.append(
                AnnotatedDocument(
                    doc_id=data["doc_id"],
                    file_path=data["file_path"],
                    contract_type=data["contract_type"],
                    lines=lines,
                )
            )
    return documents


def generate_training_data(
    corpus_dirs: dict[str, Path],  # {contract_type: path}
    output_path: Path,
    max_per_corpus: int | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Full pipeline: annotate all corpora, split, save train/val/test.

    Returns stats dict with counts and label distributions.
    """
    import random

    random.seed(seed)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Annotate all corpora
    all_documents = []
    corpus_stats = {}

    for contract_type, corpus_dir in corpus_dirs.items():
        print(f"\nAnnotating {contract_type} corpus at {corpus_dir}...")
        docs = annotate_corpus(
            corpus_dir=corpus_dir,
            contract_type=contract_type,
            max_docs=max_per_corpus,
            skip_errors=True,
        )
        all_documents.extend(docs)
        corpus_stats[contract_type] = len(docs)
        print(f"  → {len(docs)} documents annotated")

    # Shuffle and split
    random.shuffle(all_documents)
    n_docs = len(all_documents)
    n_test = int(n_docs * test_ratio)
    n_val = int(n_docs * val_ratio)
    n_train = n_docs - n_test - n_val

    train_docs = all_documents[:n_train]
    val_docs = all_documents[n_train : n_train + n_val]
    test_docs = all_documents[n_train + n_val :]

    # Save splits
    train_stats = save_annotated_corpus(train_docs, output_path / "train.jsonl")
    val_stats = save_annotated_corpus(val_docs, output_path / "val.jsonl")
    test_stats = save_annotated_corpus(test_docs, output_path / "test.jsonl")

    # Aggregate stats
    stats = {
        "corpus_stats": corpus_stats,
        "split_sizes": {
            "train": len(train_docs),
            "val": len(val_docs),
            "test": len(test_docs),
        },
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
        "total_documents": n_docs,
        "total_lines": train_stats["total_lines"]
        + val_stats["total_lines"]
        + test_stats["total_lines"],
    }

    # Save stats
    with (output_path / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    return stats
