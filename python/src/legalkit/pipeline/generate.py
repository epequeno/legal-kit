"""Stage 7: Grounded generation with citation prompting.

Takes retrieved chunks and generates an answer with inline citations,
then parses the response into structured claims.
"""

from __future__ import annotations

import re

from legalkit.llm.client import LLMClient
from legalkit.llm.prompts import (
    GROUNDED_GENERATION_SYSTEM,
    GROUNDED_GENERATION_USER,
    CHUNK_FORMAT,
)
from legalkit.models.citation import Claim, Citation
from legalkit.models.retrieval import RankedChunk


def generate_answer(
    question: str,
    ranked_chunks: list[RankedChunk],
    llm: LLMClient,
) -> tuple[str, list[Claim]]:
    """Generate a grounded answer with inline citations.

    Args:
        question: The user's question.
        ranked_chunks: Retrieved chunks with scores.
        llm: LLM client for generation.

    Returns:
        (raw_answer_text, parsed_claims)
    """
    # Format chunks for the prompt
    chunks_text = _format_chunks(ranked_chunks)

    prompt = GROUNDED_GENERATION_USER.format(
        question=question,
        chunks=chunks_text,
    )

    raw_answer = llm.generate(
        system=GROUNDED_GENERATION_SYSTEM,
        user=prompt,
    )

    # Parse the response into claims
    claims = parse_claims(raw_answer)

    return raw_answer, claims


def _format_chunks(ranked_chunks: list[RankedChunk]) -> str:
    """Format ranked chunks for inclusion in the generation prompt."""
    parts = []
    for rc in ranked_chunks:
        chunk = rc.chunk
        section_path = " > ".join(chunk.node_path) if chunk.node_path else chunk.node_id
        parts.append(CHUNK_FORMAT.format(
            section_id=chunk.node_id,
            section_path=section_path,
            text=chunk.text,
        ))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing — extract claims and citations
# ---------------------------------------------------------------------------

# Match citations like [§4.2(a)], [§8.3], [Article-IV]
_CITATION_RE = re.compile(r"\[(§[\d.]+(?:\([a-zA-Z0-9]+\))?|Article-[IVXLCDM]+|\w+-[\w.]+)\]")

# Match quoted text: "some text here"
_QUOTE_RE = re.compile(r'"([^"]{10,})"')

# Hedging language patterns
_HEDGING_PATTERNS = [
    re.compile(r"\b(?:appears? to|seems? to|may |might |could |possibly|arguably|unclear|ambiguous)\b", re.IGNORECASE),
]

# Gap acknowledgment patterns
_GAP_PATTERNS = [
    re.compile(r"\b(?:not (?:contain|include|address|provide|specify|mention)|no information|insufficient|missing|unclear from the excerpts|cannot determine)\b", re.IGNORECASE),
]


def parse_claims(answer: str) -> list[Claim]:
    """Parse an LLM answer into individual claims with citations.

    Strategy:
    - Split answer into sentences
    - Each sentence is a potential claim
    - Extract citations from each sentence
    - Detect hedging language and gap acknowledgments
    """
    sentences = _split_sentences(answer)
    claims: list[Claim] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue

        # Skip markdown headers and non-substantive lines
        if sentence.startswith("#") or sentence.startswith("---"):
            continue
        # Strip markdown header prefix for display but keep if there's substance
        if re.match(r"^#{1,4}\s+\w", sentence):
            sentence = re.sub(r"^#{1,4}\s+", "", sentence)
            if len(sentence) < 15:
                continue

        # Skip transitional/introductory phrases that aren't factual claims
        skip_patterns = [
            r"^Based on the provided",
            r"^According to the excerpts",
            r"^The (?:following|answer|excerpts?)",
            r"^Here (?:is|are) ",
            r"^In summary",
            r"^To summarize",
            r"^Specifically[,:]",
            r"^\*\*Missing",
            r"^\*\*Note",
            r"^However, (?:it is|it should be) (?:important to )?note",
        ]
        if any(re.match(p, sentence, re.IGNORECASE) for p in skip_patterns):
            # Keep the sentence but mark it as non-substantive if it has no citations
            pass  # will be caught by no_citations flag downstream

        # Extract citations
        citation_matches = _CITATION_RE.findall(sentence)
        citations = [
            Citation(section_id=ref)
            for ref in citation_matches
        ]

        # Extract quoted text and associate with nearest citation
        quotes = _QUOTE_RE.findall(sentence)
        if quotes and citations:
            # Associate first quote with first citation
            citations[0].quoted_text = quotes[0]

        # Set claim text on all citations
        # Strip citation markers from the claim text for cleaner NLI
        clean_text = _CITATION_RE.sub("", sentence).strip()
        clean_text = re.sub(r"\s{2,}", " ", clean_text)
        for cit in citations:
            cit.claim_text = clean_text

        # Detect hedging
        has_hedging = any(p.search(sentence) for p in _HEDGING_PATTERNS)

        # Detect gap acknowledgment
        is_gap = any(p.search(sentence) for p in _GAP_PATTERNS)

        claims.append(Claim(
            text=clean_text,
            citations=citations,
            has_hedging=has_hedging,
            is_gap=is_gap,
        ))

    return claims


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling legal citation patterns."""
    # Split on sentence-ending punctuation followed by space and capital
    # But be careful not to split on "§4.2" or "e.g." or "i.e."
    # Simple approach: split on ". " followed by uppercase, but not after §/numbers
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z](?![.)]\d))",
        text,
    )

    # Also split on newlines that separate paragraphs
    result = []
    for part in parts:
        sub_parts = part.split("\n\n")
        for sp in sub_parts:
            sp = sp.strip()
            if sp:
                result.append(sp)

    return result
