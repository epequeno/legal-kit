"""Synthetic entailment dataset generation.

Generates training pairs for the legal entailment model using four strategies:
1. LLM-generated claims from real legal passages
2. Perturbation-based negatives (swap amounts, parties, negate conditions)
3. Cross-passage negative mining (claim from section A, passage from section B)
4. ContractNLI adaptation (existing legal NLI annotations)

All generated samples are logged through the entailment logger with
source="synthetic:{strategy}" for traceability.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

from legalkit.eval.entailment_log import EntailmentLogger, get_logger
from legalkit.llm.client import LLMClient
from legalkit.pipeline.parse import parse_document
from legalkit.pipeline.chunk import chunk_document
from legalkit.eval.benchmark import CORPUS_DIR


# ---------------------------------------------------------------------------
# Strategy 1: LLM-generated claim pairs from real passages
# ---------------------------------------------------------------------------

GENERATE_CLAIMS_SYSTEM = """\
You are a legal document analyst generating training data for an entailment \
model. Given a passage from a legal document, generate exactly 4 claims:

1. ENTAILED — a claim that is directly and clearly supported by the passage
2. PARTIALLY_ENTAILED — a claim that is related and somewhat supported, but \
makes an inference or uses language not directly in the passage
3. CONTRADICTED — a claim that directly contradicts something stated in the passage
4. NEUTRAL — a claim about a legal topic that is unrelated to this passage

Rules:
- Claims should be realistic legal statements a lawyer might make
- ENTAILED claims should be paraphrases, not exact quotes
- CONTRADICTED claims should be subtle — change a key fact, not obviously wrong
- NEUTRAL claims should be topically adjacent but about different provisions
- Each claim should be 1-2 sentences

Respond in exactly this format (no other text):
ENTAILED: <claim>
PARTIALLY_ENTAILED: <claim>
CONTRADICTED: <claim>
NEUTRAL: <claim>"""

GENERATE_CLAIMS_USER = """\
Legal passage (from {section_id} of {doc_type}):
\"\"\"{passage}\"\"\"

Generate 4 claims:"""


def generate_llm_pairs(
    passages: list[dict],
    llm: LLMClient,
    logger: EntailmentLogger | None = None,
    progress_callback=None,
) -> int:
    """Generate entailment pairs using LLM from real legal passages.

    Args:
        passages: List of dicts with keys: text, section_id, doc_id, doc_type
        llm: LLM client for generation.
        logger: Entailment logger (uses global if None).
        progress_callback: Optional callback(current, total).

    Returns:
        Number of samples generated.
    """
    logger = logger or get_logger()
    count = 0

    for i, p in enumerate(passages):
        if progress_callback and i % 5 == 0:
            progress_callback(i, len(passages))

        passage = p["text"][:2000]
        if len(passage.strip()) < 50:
            continue

        prompt = GENERATE_CLAIMS_USER.format(
            section_id=p.get("section_id", "unknown"),
            doc_type=p.get("doc_type", "legal document"),
            passage=passage,
        )

        try:
            response = llm.generate(
                system=GENERATE_CLAIMS_SYSTEM,
                user=prompt,
                model=llm.config.context_model,  # use cheap model
                max_tokens=500,
            )

            pairs = _parse_generated_claims(response)

            for entailment_class, claim in pairs:
                logger.log(
                    passage=passage,
                    claim=claim,
                    entailment_class=entailment_class,
                    confidence=1.0,  # LLM-generated ground truth
                    explanation=f"Synthetically generated ({entailment_class})",
                    section_id=p.get("section_id", ""),
                    doc_id=p.get("doc_id", ""),
                    query="",
                    judge_model=llm.config.context_model,
                    source=f"synthetic:llm_generated",
                )
                count += 1

            # Rate limiting
            time.sleep(0.2)

        except Exception as e:
            continue

    if progress_callback:
        progress_callback(len(passages), len(passages))

    return count


def _parse_generated_claims(response: str) -> list[tuple[str, str]]:
    """Parse LLM response into (entailment_class, claim) pairs."""
    pairs = []
    for label in ["ENTAILED", "PARTIALLY_ENTAILED", "CONTRADICTED", "NEUTRAL"]:
        m = re.search(rf"{label}:\s*(.+?)(?:\n|$)", response)
        if m:
            claim = m.group(1).strip()
            if claim:
                pairs.append((label.lower(), claim))
    return pairs


# ---------------------------------------------------------------------------
# Strategy 2: Perturbation-based negatives
# ---------------------------------------------------------------------------

# Patterns for systematic perturbation of legal claims
_AMOUNT_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|Dollars?)\b")
_PARTY_RE = re.compile(r"\b(Seller|Buyer|Licensee|Licensor|Landlord|Tenant|Borrower|Lender|Company|Parent|Acquirer)\b")
_PERIOD_RE = re.compile(r"\b(\d+)\s*(?:days?|months?|years?|business days?)\b", re.IGNORECASE)
_NEGATION_PHRASES = [
    (r"\bshall\b", "shall not"),
    (r"\bshall not\b", "shall"),
    (r"\bmay\b", "may not"),
    (r"\bmay not\b", "may"),
    (r"\bis required\b", "is not required"),
    (r"\bis not required\b", "is required"),
    (r"\bincludes?\b", "excludes"),
    (r"\bexcludes?\b", "includes"),
]


def generate_perturbation_pairs(
    entailed_samples: list[dict],
    logger: EntailmentLogger | None = None,
) -> int:
    """Generate contradicted samples by perturbing entailed claims.

    Takes existing entailed (passage, claim) pairs and creates contradictions
    by systematically altering key facts.

    Args:
        entailed_samples: List of dicts with keys: passage, claim, section_id, doc_id
        logger: Entailment logger (uses global if None).

    Returns:
        Number of samples generated.
    """
    logger = logger or get_logger()
    count = 0

    for sample in entailed_samples:
        claim = sample["claim"]
        passage = sample["passage"]
        perturbations = []

        # Try amount perturbation
        amounts = _AMOUNT_RE.findall(claim)
        if amounts:
            for amt in amounts:
                # Double or halve the amount
                perturbed = claim.replace(amt, _perturb_amount(amt))
                if perturbed != claim:
                    perturbations.append(("Amount changed", perturbed))
                    break

        # Try party swap
        parties = _PARTY_RE.findall(claim)
        if len(set(parties)) >= 2:
            party_list = list(set(parties))
            perturbed = claim.replace(party_list[0], "__TEMP__")
            perturbed = perturbed.replace(party_list[1], party_list[0])
            perturbed = perturbed.replace("__TEMP__", party_list[1])
            perturbations.append(("Parties swapped", perturbed))

        # Try period change
        periods = _PERIOD_RE.findall(claim)
        if periods:
            old_num = periods[0]
            new_num = str(int(old_num) * 2) if int(old_num) < 100 else str(int(old_num) // 2)
            perturbed = _PERIOD_RE.sub(
                lambda m: m.group(0).replace(m.group(1), new_num),
                claim,
                count=1,
            )
            if perturbed != claim:
                perturbations.append(("Period changed", perturbed))

        # Try negation
        for pattern, replacement in _NEGATION_PHRASES:
            if re.search(pattern, claim, re.IGNORECASE):
                perturbed = re.sub(pattern, replacement, claim, count=1, flags=re.IGNORECASE)
                if perturbed != claim:
                    perturbations.append(("Negated", perturbed))
                    break

        # Log each perturbation as contradicted
        for method, perturbed_claim in perturbations:
            logger.log(
                passage=passage,
                claim=perturbed_claim,
                entailment_class="contradicted",
                confidence=1.0,
                explanation=f"Perturbation: {method}",
                section_id=sample.get("section_id", ""),
                doc_id=sample.get("doc_id", ""),
                query="",
                judge_model="perturbation",
                source=f"synthetic:perturbation:{method.lower().replace(' ', '_')}",
            )
            count += 1

    return count


def _perturb_amount(amount_str: str) -> str:
    """Change a dollar amount to create a contradiction."""
    # Extract numeric part
    nums = re.findall(r"[\d,]+", amount_str)
    if not nums:
        return amount_str
    num_str = nums[0].replace(",", "")
    try:
        num = int(num_str)
        # Double it
        new_num = num * 2
        new_str = f"{new_num:,}"
        return amount_str.replace(nums[0], new_str)
    except ValueError:
        return amount_str


# ---------------------------------------------------------------------------
# Strategy 3: Cross-passage negative mining
# ---------------------------------------------------------------------------


def generate_cross_passage_pairs(
    chunks: list[dict],
    sample_size: int = 500,
    logger: EntailmentLogger | None = None,
) -> int:
    """Generate neutral pairs by combining claims from one section with passages from another.

    Takes a list of chunks from the same document and pairs each chunk's
    text (as a pseudo-claim) with a passage from a different chunk.
    These should be neutral — topically related but not entailing.

    Args:
        chunks: List of dicts with keys: text, section_id, doc_id
        sample_size: Maximum number of pairs to generate.
        logger: Entailment logger (uses global if None).

    Returns:
        Number of samples generated.
    """
    logger = logger or get_logger()
    count = 0

    if len(chunks) < 2:
        return 0

    # Generate random pairs from different sections
    pairs = []
    for _ in range(sample_size * 3):  # oversample then filter
        a, b = random.sample(range(len(chunks)), 2)
        if chunks[a]["section_id"] != chunks[b]["section_id"]:
            pairs.append((a, b))
        if len(pairs) >= sample_size:
            break

    for a_idx, b_idx in pairs:
        passage = chunks[a_idx]["text"][:2000]
        # Use the first sentence of the other chunk as a "claim"
        other_text = chunks[b_idx]["text"]
        claim = _extract_first_sentence(other_text)

        if not claim or len(claim) < 20:
            continue

        logger.log(
            passage=passage,
            claim=claim,
            entailment_class="neutral",
            confidence=0.9,  # high but not 1.0 — some pairs might actually be related
            explanation="Cross-passage negative: passage and claim from different sections",
            section_id=chunks[a_idx].get("section_id", ""),
            doc_id=chunks[a_idx].get("doc_id", ""),
            query="",
            judge_model="cross_passage",
            source="synthetic:cross_passage",
        )
        count += 1

    return count


def _extract_first_sentence(text: str) -> str:
    """Extract the first substantive sentence from text."""
    text = text.strip()
    # Split on sentence boundaries
    m = re.match(r"(.+?[.!?])\s", text)
    if m:
        return m.group(1).strip()
    # Fallback: first 200 chars
    return text[:200].strip()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(
    corpus_dir: Path | None = None,
    llm: LLMClient | None = None,
    max_docs: int = 20,
    passages_per_doc: int = 5,
    cross_passage_per_doc: int = 20,
    strategies: list[str] | None = None,
    progress_callback=None,
) -> dict[str, int]:
    """Generate a synthetic entailment dataset from the legal corpus.

    Args:
        corpus_dir: Path to corpus files. Defaults to LegalBench-RAG corpus.
        llm: LLM client (required for llm_generated strategy).
        max_docs: Maximum corpus documents to process.
        passages_per_doc: Passages to send to LLM per document.
        cross_passage_per_doc: Cross-passage pairs per document.
        strategies: Which strategies to use. Default: all available.
        progress_callback: Optional callback(stage, current, total).

    Returns:
        Dict of strategy -> sample count.
    """
    corpus_dir = corpus_dir or CORPUS_DIR
    strategies = strategies or ["llm_generated", "perturbation", "cross_passage"]

    results: dict[str, int] = {}
    logger = get_logger()

    # Collect corpus documents
    doc_files = []
    for ext in ("*.txt", "*.md"):
        doc_files.extend(corpus_dir.rglob(ext))
    doc_files = sorted(doc_files)[:max_docs]

    if progress_callback:
        progress_callback("loading", 0, len(doc_files))

    # Process each document
    all_passages: list[dict] = []
    all_chunks: list[dict] = []

    for di, doc_file in enumerate(doc_files):
        text = doc_file.read_text()
        doc_id = doc_file.stem
        rel_path = doc_file.relative_to(corpus_dir)
        doc_type = rel_path.parts[0] if len(rel_path.parts) > 1 else "legal_document"

        root = parse_document(text, doc_id=doc_id)
        chunks = chunk_document(root, doc_id=doc_id)

        for chunk in chunks:
            chunk_dict = {
                "text": chunk.text,
                "section_id": chunk.node_id,
                "doc_id": doc_id,
                "doc_type": doc_type,
            }
            all_chunks.append(chunk_dict)

            # Sample passages for LLM generation
            if len(chunk.text) >= 100:
                all_passages.append(chunk_dict)

        if progress_callback:
            progress_callback("loading", di + 1, len(doc_files))

    # Subsample passages for LLM generation
    random.shuffle(all_passages)
    llm_passages = all_passages[:max_docs * passages_per_doc]

    # Strategy 1: LLM-generated pairs
    if "llm_generated" in strategies and llm is not None:
        if progress_callback:
            progress_callback("llm_generating", 0, len(llm_passages))

        count = generate_llm_pairs(
            passages=llm_passages,
            llm=llm,
            logger=logger,
            progress_callback=lambda c, t: progress_callback("llm_generating", c, t) if progress_callback else None,
        )
        results["llm_generated"] = count

    # Strategy 3: Cross-passage negatives (before perturbation so we have data)
    if "cross_passage" in strategies:
        # Group chunks by document for intra-document pairing
        doc_chunks: dict[str, list[dict]] = {}
        for c in all_chunks:
            doc_chunks.setdefault(c["doc_id"], []).append(c)

        total_cross = 0
        for doc_id, chunks_list in doc_chunks.items():
            total_cross += generate_cross_passage_pairs(
                chunks=chunks_list,
                sample_size=cross_passage_per_doc,
                logger=logger,
            )
        results["cross_passage"] = total_cross

    # Strategy 2: Perturbation-based negatives (uses LLM-generated entailed pairs)
    if "perturbation" in strategies:
        # Load entailed samples from the log
        all_samples = EntailmentLogger.load_all()
        entailed = [
            {"passage": s.passage, "claim": s.claim, "section_id": s.section_id, "doc_id": s.doc_id}
            for s in all_samples
            if s.entailment_class == "entailed"
        ]
        count = generate_perturbation_pairs(
            entailed_samples=entailed,
            logger=logger,
        )
        results["perturbation"] = count

    return results
