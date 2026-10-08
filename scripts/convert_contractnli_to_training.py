"""Convert ContractNLI dataset to entailment training pairs.

ContractNLI has 607 NDAs × 17 hypotheses with expert (lawyer) annotations.
Each (document, hypothesis) pair has a label (Entailment/Contradiction/NotMentioned)
and evidence spans showing which parts of the document support the label.

We extract (evidence_passage, hypothesis, label) triples for entailment training.
This gives us ~10K expert-labeled pairs — higher quality than LLM-generated data.

Usage:
    python scripts/convert_contractnli_to_training.py
    python scripts/convert_contractnli_to_training.py --max-passage-chars 500
    python scripts/convert_contractnli_to_training.py --include-not-mentioned
"""

import json
import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
CNLI_DIR = ROOT / "data" / "contractnli_raw" / "contract-nli"

# Map ContractNLI labels to our 3-class schema
LABEL_MAP = {
    "Entailment": {"label": 0, "label_name": "entailed"},
    "Contradiction": {"label": 1, "label_name": "contradicted"},
    "NotMentioned": {"label": 2, "label_name": "neutral"},
}


def load_split(split: str) -> tuple[list[dict], dict]:
    """Load a ContractNLI split. Returns (documents, hypotheses)."""
    path = CNLI_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download ContractNLI first:\n"
            "  curl -sL https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip -o /tmp/contract-nli.zip\n"
            "  unzip /tmp/contract-nli.zip -d data/contractnli_raw/"
        )
    data = json.load(open(path))
    return data["documents"], data["labels"]


def extract_evidence_passage(doc: dict, span_indices: list[int], 
                              max_chars: int = 500, context_chars: int = 100) -> str:
    """Extract the evidence passage from a document given span indices.
    
    Concatenates the evidence spans with context. If total length exceeds
    max_chars, takes the most relevant spans.
    """
    text = doc["text"]
    spans = doc["spans"]
    
    if not span_indices:
        return ""
    
    # Get actual char ranges and sort by position
    char_ranges = []
    for si in span_indices:
        if si < len(spans):
            start, end = spans[si]
            char_ranges.append((start, end))
    
    if not char_ranges:
        return ""
    
    char_ranges.sort()
    
    # Merge overlapping/adjacent spans with context
    merged = []
    for start, end in char_ranges:
        # Add context
        ctx_start = max(0, start - context_chars)
        ctx_end = min(len(text), end + context_chars)
        
        if merged and ctx_start <= merged[-1][1]:
            # Merge with previous
            merged[-1] = (merged[-1][0], max(merged[-1][1], ctx_end))
        else:
            merged.append((ctx_start, ctx_end))
    
    # Extract and join passages
    passages = []
    total_chars = 0
    for start, end in merged:
        passage = text[start:end].strip()
        if total_chars + len(passage) > max_chars and passages:
            break  # Don't exceed max
        passages.append(passage)
        total_chars += len(passage)
    
    return " [...] ".join(passages)


def extract_random_passage(doc: dict, exclude_spans: list[int],
                           target_chars: int = 300) -> str:
    """Extract a random non-evidence passage for NotMentioned examples."""
    import random
    
    text = doc["text"]
    spans = doc["spans"]
    exclude_set = set(exclude_spans)
    
    # Get non-evidence spans
    available = [i for i in range(len(spans)) if i not in exclude_set]
    if not available:
        # Fallback: take middle of document
        mid = len(text) // 2
        return text[mid:mid + target_chars].strip()
    
    # Pick a contiguous block of spans
    random.shuffle(available)
    passage_spans = sorted(available[:5])  # Take up to 5 spans
    
    parts = []
    total = 0
    for si in passage_spans:
        start, end = spans[si]
        part = text[start:end].strip()
        if total + len(part) > target_chars and parts:
            break
        parts.append(part)
        total += len(part)
    
    return " ".join(parts)


def convert_split(split: str, max_passage_chars: int = 500,
                  include_not_mentioned: bool = True) -> list[dict]:
    """Convert one ContractNLI split to entailment training pairs."""
    docs, hypotheses = load_split(split)
    pairs = []
    
    for doc in docs:
        doc_id = doc.get("file_name", str(doc.get("id", "")))
        annotations = doc["annotation_sets"][0]["annotations"]
        
        for hyp_id, ann in annotations.items():
            choice = ann["choice"]
            evidence_spans = ann.get("spans", [])
            hypothesis = hypotheses[hyp_id]["hypothesis"]
            
            label_info = LABEL_MAP[choice]
            
            if choice in ("Entailment", "Contradiction"):
                # Use evidence spans as the passage
                passage = extract_evidence_passage(
                    doc, evidence_spans, max_chars=max_passage_chars
                )
                if not passage or len(passage) < 30:
                    continue
                    
                pairs.append({
                    "premise": passage,
                    "hypothesis": hypothesis,
                    "label": label_info["label"],
                    "label_name": label_info["label_name"],
                    "source": f"contractnli:{split}",
                    "doc_id": doc_id,
                    "section_id": hyp_id,
                })
                
            elif choice == "NotMentioned" and include_not_mentioned:
                # For neutral: use a random non-evidence passage
                # This is legitimate — the hypothesis is NOT mentioned in the doc,
                # and we pick a passage that doesn't contain evidence for it
                passage = extract_random_passage(doc, evidence_spans)
                if not passage or len(passage) < 30:
                    continue
                
                pairs.append({
                    "premise": passage,
                    "hypothesis": hypothesis,
                    "label": label_info["label"],
                    "label_name": label_info["label_name"],
                    "source": f"contractnli:{split}",
                    "doc_id": doc_id,
                    "section_id": hyp_id,
                })
    
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-passage-chars", type=int, default=500)
    parser.add_argument("--include-not-mentioned", action="store_true", default=True)
    parser.add_argument("--no-not-mentioned", action="store_true",
                        help="Exclude NotMentioned (neutral) examples")
    args = parser.parse_args()
    
    include_neutral = not args.no_not_mentioned
    
    print("=" * 70)
    print("Converting ContractNLI to Entailment Training Data")
    print("=" * 70)
    
    all_pairs = []
    
    for split in ["train", "dev", "test"]:
        pairs = convert_split(
            split, 
            max_passage_chars=args.max_passage_chars,
            include_not_mentioned=include_neutral,
        )
        print(f"\n{split}: {len(pairs)} pairs extracted")
        
        label_dist = Counter(p["label_name"] for p in pairs)
        for l, c in sorted(label_dist.items()):
            print(f"  {l}: {c}")
        
        all_pairs.extend(pairs)
    
    print(f"\n{'='*70}")
    print(f"Total: {len(all_pairs)} pairs from ContractNLI")
    
    total_dist = Counter(p["label_name"] for p in all_pairs)
    print(f"\nOverall distribution:")
    for l, c in sorted(total_dist.items()):
        print(f"  {l:<20} {c:>5} ({c/len(all_pairs):.1%})")
    
    # Unique docs
    unique_docs = len(set(p["doc_id"] for p in all_pairs))
    print(f"\nUnique documents: {unique_docs}")
    
    # Save
    out_path = ROOT / "data" / "training_contractnli" / "contractnli_pairs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    
    print(f"\nSaved to: {out_path}")
    print(f"\nTo merge with v3 training data:")
    print(f"  python scripts/prepare_entailment_v3.py --audit-corrections \\")
    print(f"    --extra-data {out_path} data/training_targeted/targeted_negatives.jsonl")


if __name__ == "__main__":
    main()
