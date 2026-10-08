"""Entailment logging — captures every verification judgment for training data.

Every time the pipeline checks whether a passage supports a claim,
we log the full context: passage text, claim text, entailment class,
confidence score, section IDs, and metadata. This data becomes the
training set for the legal entailment model (Phase 2).

Storage format: JSONL (one JSON object per line), append-only.
Each file covers one day, stored in data/entailment_logs/.
"""

from __future__ import annotations

import json
import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

LOG_DIR = Path("data/entailment_logs")


@dataclass
class EntailmentSample:
    """A single entailment training sample."""

    # Core pair
    passage: str
    """Source text from the document (the premise)."""

    claim: str
    """Claim made about the passage (the hypothesis)."""

    # Labels
    entailment_class: str
    """One of: entailed, partially_entailed, contradicted, neutral."""

    confidence: float
    """Model confidence in the classification. 0.0-1.0."""

    explanation: str
    """Brief explanation of the judgment."""

    # Provenance
    section_id: str
    """Which section the passage came from."""

    doc_id: str
    """Which document."""

    query: str
    """The user's original question that triggered this."""

    # Metadata
    judge_model: str
    """Which model produced this judgment (e.g., 'claude-sonnet-4')."""

    timestamp: str
    """ISO 8601 timestamp."""

    source: str = "pipeline"
    """Where this sample came from: 'pipeline', 'benchmark', 'manual_review'."""

    reviewed: bool = False
    """Whether a human has reviewed and confirmed/corrected this label."""

    corrected_class: str | None = None
    """If reviewed, the human-corrected entailment class (or None if original was correct)."""


class EntailmentLogger:
    """Append-only logger for entailment samples.

    Writes to daily JSONL files in data/entailment_logs/.
    Thread-safe via append mode.
    """

    def __init__(self, log_dir: Path | None = None):
        self.log_dir = log_dir or LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def log(
        self,
        passage: str,
        claim: str,
        entailment_class: str,
        confidence: float,
        explanation: str,
        section_id: str = "",
        doc_id: str = "",
        query: str = "",
        judge_model: str = "",
        source: str = "pipeline",
    ) -> None:
        """Log a single entailment judgment."""
        sample = EntailmentSample(
            passage=passage,
            claim=claim,
            entailment_class=entailment_class,
            confidence=confidence,
            explanation=explanation,
            section_id=section_id,
            doc_id=doc_id,
            query=query,
            judge_model=judge_model,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source=source,
        )

        log_file = self._current_file()
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(sample)) + "\n")

        self._count += 1

    def _current_file(self) -> Path:
        """Get today's log file path."""
        date_str = datetime.date.today().isoformat()
        return self.log_dir / f"entailment_{date_str}.jsonl"

    @property
    def session_count(self) -> int:
        """Number of samples logged in this session."""
        return self._count

    @staticmethod
    def load_all(log_dir: Path | None = None) -> list[EntailmentSample]:
        """Load all logged samples across all days."""
        log_dir = log_dir or LOG_DIR
        samples = []
        for f in sorted(log_dir.glob("entailment_*.jsonl")):
            for line in f.read_text().splitlines():
                line = line.strip()
                if line:
                    data = json.loads(line)
                    samples.append(EntailmentSample(**data))
        return samples

    @staticmethod
    def stats(log_dir: Path | None = None) -> dict:
        """Get summary statistics of logged samples."""
        samples = EntailmentLogger.load_all(log_dir)
        if not samples:
            return {"total": 0}

        from collections import Counter
        classes = Counter(s.entailment_class for s in samples)
        sources = Counter(s.source for s in samples)
        docs = Counter(s.doc_id for s in samples)
        reviewed = sum(1 for s in samples if s.reviewed)

        return {
            "total": len(samples),
            "by_class": dict(classes),
            "by_source": dict(sources),
            "unique_docs": len(docs),
            "reviewed": reviewed,
            "unreviewed": len(samples) - reviewed,
            "date_range": (samples[0].timestamp[:10], samples[-1].timestamp[:10]),
        }


# Global logger instance
_logger: EntailmentLogger | None = None


def get_logger() -> EntailmentLogger:
    """Get or create the global entailment logger."""
    global _logger
    if _logger is None:
        _logger = EntailmentLogger()
    return _logger
