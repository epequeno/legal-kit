#!/usr/bin/env python3
"""Build a diverse GRPO prompt pool from multiple sources.

Sources:
  1. Existing 243 eval cases (synthetic, 14 categories)
  2. ContractNLI (expert labels, real NDAs, 17 hypothesis types)
  3. CUAD passages + LLM-generated claims (real contracts, diverse clauses)

Usage:
    cd python && uv run python ../scripts/build_grpo_prompts.py --stats
    cd python && uv run python ../scripts/build_grpo_prompts.py
    cd python && uv run python ../scripts/build_grpo_prompts.py --generate-cuad  # needs OPENROUTER_API_KEY
"""

import json
import random
import argparse
import asyncio
import os
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent

LABEL_MAP = {
    "Entailment": "entailed",
    "Contradiction": "contradicted",
    "NotMentioned": "neutral",
    "entailed": "entailed",
    "partially_entailed": "entailed",
    "contradicted": "contradicted",
    "neutral": "neutral",
}


def load_eval_cases():
    """Source 1: Our existing 243 eval cases."""
    path = ROOT / "data" / "eval" / "legal_nli_eval_expanded.json"
    raw = json.load(open(path))
    cases = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw

    prompts = []
    for c in cases:
        prompts.append({
            "passage": c.get("passage", c.get("premise", "")),
            "claim": c.get("claim", c.get("hypothesis", "")),
            "label": LABEL_MAP.get(c["label"], c["label"]),
            "source": "eval_expanded",
            "category": c.get("category", "unknown"),
        })
    return prompts


def load_contractnli(max_per_hypothesis=25):
    """Source 2: ContractNLI expert-labeled triples.
    
    Each triple: real NDA passage + expert hypothesis + expert label.
    We extract the relevant passage spans for Entailment/Contradiction,
    or a random passage section for NotMentioned.
    """
    path = ROOT / "data" / "contractnli_raw" / "contract-nli" / "train.json"
    if not path.exists():
        print(f"  ContractNLI not found at {path}")
        return []

    cnli = json.load(open(path))
    labels_meta = cnli["labels"]

    prompts = []
    by_hypothesis = Counter()

    for doc in cnli["documents"]:
        text = doc["text"]
        spans = doc["spans"]

        for ann_set in doc.get("annotation_sets", []):
            for hyp_key, ann in ann_set.get("annotations", {}).items():
                choice = ann["choice"]
                label = LABEL_MAP.get(choice)
                if not label:
                    continue

                # Rate limit per hypothesis to keep diversity
                if by_hypothesis[hyp_key] >= max_per_hypothesis:
                    continue

                # Get hypothesis text
                hyp_meta = labels_meta[hyp_key]
                if isinstance(hyp_meta, str):
                    hyp_meta = eval(hyp_meta)  # it's stored as a string repr of dict
                hypothesis = hyp_meta["hypothesis"]

                # Get passage text
                evidence_spans = ann.get("spans", [])
                if evidence_spans and choice != "NotMentioned":
                    # Use evidence spans + surrounding context
                    span_texts = []
                    for si in evidence_spans:
                        if si < len(spans):
                            s, e = spans[si]
                            # Add context: 200 chars before and after
                            ctx_start = max(0, s - 200)
                            ctx_end = min(len(text), e + 200)
                            span_texts.append(text[ctx_start:ctx_end])
                    passage = "\n...\n".join(span_texts)
                else:
                    # NotMentioned: use a random section of the document
                    # Pick a ~1000 char window
                    if len(text) > 1500:
                        start = random.randint(0, len(text) - 1500)
                        passage = text[start:start + 1500]
                    else:
                        passage = text

                # Skip very short passages
                if len(passage) < 100:
                    continue

                # Truncate very long passages
                if len(passage) > 3000:
                    passage = passage[:3000]

                prompts.append({
                    "passage": passage.strip(),
                    "claim": hypothesis,
                    "label": label,
                    "source": "contractnli_expert",
                    "category": hyp_meta.get("short_description", hyp_key),
                })
                by_hypothesis[hyp_key] += 1

    return prompts


def load_cuad_passages(n=200):
    """Source 3: Real CUAD contract passages for claim generation."""
    corpus_dir = ROOT / "data" / "corpus" / "cuad"
    if not corpus_dir.exists():
        print(f"  CUAD corpus not found at {corpus_dir}")
        return []

    files = [f for f in corpus_dir.iterdir() if f.suffix == ".txt"]
    random.shuffle(files)

    passages = []
    for f in files[:50]:  # sample from 50 contracts
        text = f.read_text(errors="ignore")
        # Extract ~1000 char sections that look like substantive clauses
        # Skip headers and boilerplate
        sections = re.split(r'\n\s*\n', text)
        for section in sections:
            section = section.strip()
            if 200 < len(section) < 2000 and any(kw in section.lower() for kw in [
                'shall', 'may', 'agrees to', 'warrants', 'represents',
                'indemnif', 'terminat', 'confidential', 'liability',
                'notwithstanding', 'provided that', 'subject to',
            ]):
                passages.append({
                    "passage": section[:2000],
                    "source_file": f.name,
                })
                if len(passages) >= n:
                    break
        if len(passages) >= n:
            break

    return passages


async def generate_cuad_claims(passages, api_key):
    """Generate (claim, label) pairs for CUAD passages via OpenRouter."""
    import httpx

    model = "qwen/qwen3-235b-a22b-2507"
    semaphore = asyncio.Semaphore(20)

    system = """Given a passage from a legal contract, generate exactly 3 claim-label pairs:
1. One claim that is ENTAILED by the passage (directly supported)
2. One claim that is CONTRADICTED by the passage (directly conflicts)
3. One claim that is NEUTRAL (not addressed by the passage)

Each claim should be specific and reference concrete provisions from the passage.

Output as JSON array:
[
  {"claim": "...", "label": "entailed"},
  {"claim": "...", "label": "contradicted"},
  {"claim": "...", "label": "neutral"}
]"""

    async def gen_one(client, passage_info, sem):
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": model,
                            "max_tokens": 500,
                            "temperature": 0.7,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": passage_info["passage"]},
                            ],
                        },
                        timeout=45,
                    )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]

                    # Parse JSON from response
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        claims = json.loads(json_match.group())
                        results = []
                        for c in claims:
                            if "claim" in c and "label" in c:
                                results.append({
                                    "passage": passage_info["passage"],
                                    "claim": c["claim"],
                                    "label": c["label"],
                                    "source": "cuad_generated",
                                    "category": "cuad_" + passage_info["source_file"][:20],
                                })
                        return results
                    return []
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(1)
            return []

    results = []
    async with httpx.AsyncClient(timeout=45) as client:
        tasks = [gen_one(client, p, semaphore) for p in passages]
        done = 0
        for coro in asyncio.as_completed(tasks):
            batch = await coro
            results.extend(batch)
            done += 1
            if done % 20 == 0:
                print(f"    [{done}/{len(passages)}] {len(results)} claims generated")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--generate-cuad", action="store_true",
                        help="Generate claims for CUAD passages (needs OPENROUTER_API_KEY)")
    parser.add_argument("--max-contractnli", type=int, default=25,
                        help="Max samples per ContractNLI hypothesis (default: 25)")
    parser.add_argument("--cuad-passages", type=int, default=100,
                        help="Number of CUAD passages to generate claims for")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("Building GRPO Prompt Pool")
    print("=" * 60)

    # Source 1: Eval cases
    print("\n[1] Loading eval cases...")
    eval_prompts = load_eval_cases()
    print(f"    {len(eval_prompts)} prompts")

    # Source 2: ContractNLI
    print("\n[2] Loading ContractNLI expert labels...")
    cnli_prompts = load_contractnli(max_per_hypothesis=args.max_contractnli)
    print(f"    {len(cnli_prompts)} prompts")

    # Source 3: CUAD
    cuad_prompts = []
    if args.generate_cuad:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("\n[3] CUAD: OPENROUTER_API_KEY not set, skipping")
        else:
            print(f"\n[3] Loading CUAD passages and generating claims...")
            passages = load_cuad_passages(n=args.cuad_passages)
            print(f"    {len(passages)} passages extracted")
            cuad_prompts = asyncio.run(generate_cuad_claims(passages, api_key))
            print(f"    {len(cuad_prompts)} claims generated")
    else:
        # Check if we already generated CUAD prompts
        cuad_cache = ROOT / "data" / "grpo_prompts" / "cuad_generated.jsonl"
        if cuad_cache.exists():
            cuad_prompts = [json.loads(l) for l in open(cuad_cache)]
            print(f"\n[3] Loaded {len(cuad_prompts)} cached CUAD prompts")
        else:
            print(f"\n[3] CUAD: Run with --generate-cuad to create (costs ~$0.01)")

    # Combine
    all_prompts = eval_prompts + cnli_prompts + cuad_prompts
    random.shuffle(all_prompts)

    # Stats
    print(f"\n{'='*60}")
    print(f"Total GRPO prompts: {len(all_prompts)}")
    print(f"{'='*60}")

    by_source = Counter(p["source"] for p in all_prompts)
    print(f"\nBy source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")

    by_label = Counter(p["label"] for p in all_prompts)
    print(f"\nBy label:")
    for label, count in sorted(by_label.items()):
        print(f"  {label}: {count}")

    if args.stats:
        return

    # Save
    out_dir = ROOT / "data" / "grpo_prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save combined
    out_path = out_dir / "grpo_prompts.jsonl"
    with open(out_path, "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved to {out_path}")

    # Save CUAD separately for caching
    if cuad_prompts:
        cuad_cache = out_dir / "cuad_generated.jsonl"
        with open(cuad_cache, "w") as f:
            for p in cuad_prompts:
                f.write(json.dumps(p) + "\n")
        print(f"CUAD cache: {cuad_cache}")

    # Show samples from each source
    print(f"\nSamples:")
    for src in sorted(by_source.keys()):
        sample = next(p for p in all_prompts if p["source"] == src)
        print(f"\n  [{src}] label={sample['label']}")
        print(f"  Passage: {sample['passage'][:150]}...")
        print(f"  Claim: {sample['claim'][:120]}")


if __name__ == "__main__":
    main()
