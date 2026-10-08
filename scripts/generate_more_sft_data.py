#!/usr/bin/env python3
"""Generate more SFT reasoning chains from unused training pairs.

Pulls unused pairs from data/training_v2/, generates reasoning chains
via OpenRouter, and appends to the SFT dataset.

Usage:
    cd python && uv run python ../scripts/generate_more_sft_data.py              # Generate 100 more
    cd python && uv run python ../scripts/generate_more_sft_data.py --n 200      # Generate 200 more
    cd python && uv run python ../scripts/generate_more_sft_data.py --category temporal  # Bias toward temporal
    cd python && uv run python ../scripts/generate_more_sft_data.py --stats      # Just show stats
"""

import json
import re
import asyncio
import argparse
import os
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

import httpx

ROOT = Path(__file__).parent.parent
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "qwen/qwen3-235b-a22b-2507"
RESULTS_DIR = ROOT / "data" / "sft_reasoning" / "incremental"

LABEL_COLLAPSE = {
    "entailed": "entailed",
    "partially_entailed": "entailed",
    "contradicted": "contradicted",
    "neutral": "neutral",
}

SYSTEM_PROMPT = """You are a legal reasoning assistant specializing in contract interpretation. Given a passage from a legally binding agreement and a claim about its meaning or implications, carefully analyze the text with strict attention to contractual language, including modal verbs (e.g., "may" vs. "must"), defined terms, exceptions, exclusions, and temporal or conditional clauses.

Begin by restating the key provisions in the passage, then assess whether the claim is fully supported, logically contradicted, or neither. Your reasoning must be precise, logically structured, and grounded exclusively in the text—do not infer intent or rely on external knowledge.

Construct a step-by-step analysis (50–150 words) that traces the legal logic, highlights critical linguistic nuances, and resolves any hierarchical qualifications. If the claim overgeneralizes, adds obligations, or misreads negations or exceptions, treat it as contradicted. When in doubt, default to neutral.

DO NOT use vague template phrases. Every reasoning chain must reference specific content from the passage.

Output format (exactly):
<think>
[Your reasoning here]
</think>
<answer>[entailed|contradicted|neutral]</answer>"""


def load_already_used():
    """Find which pairs we've already generated chains for."""
    used = set()
    # Check batch files
    batch_dir = ROOT / "data" / "sft_reasoning" / "batches"
    if batch_dir.exists():
        for f in batch_dir.glob("batch_*.json"):
            try:
                for item in json.load(open(f)):
                    used.add(item['hypothesis'][:80])
            except:
                pass
    # Check incremental results
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob("*.jsonl"):
            for line in open(f):
                try:
                    d = json.loads(line)
                    # Extract hypothesis from input
                    m = re.search(r'<claim>\n(.*?)\n</claim>', d.get('input', ''), re.DOTALL)
                    if m:
                        used.add(m.group(1)[:80])
                except:
                    pass
    return used


def load_unused_pairs(used, category_bias=None):
    """Load training pairs we haven't generated chains for yet."""
    rows = []
    for split in ('train', 'val', 'test'):
        path = ROOT / f'data/training_v2/{split}.jsonl'
        if path.exists():
            rows.extend(json.loads(l) for l in open(path))

    unused = []
    for r in rows:
        if r['hypothesis'][:80] not in used:
            unused.append(r)

    # If category bias, look for relevant content
    if category_bias:
        keywords = {
            'temporal': ['days', 'months', 'years', 'period', 'expire', 'survival',
                        'termination date', 'notice period', 'within', 'after the date',
                        'prior to', 'no later than', 'calendar days', 'business days'],
            'liability': ['liability', 'indemnif', 'cap', 'limitation', 'aggregate',
                         'consequential', 'incidental', 'punitive', 'damages'],
            'mae': ['material adverse', 'MAE', 'Material Adverse Effect', 'carve-out',
                    'provided however', 'notwithstanding', 'except for', 'excluding'],
            'condition': ['condition precedent', 'subject to', 'provided that',
                         'contingent upon', 'if and only if', 'upon satisfaction'],
        }
        bias_kw = keywords.get(category_bias, [])
        if bias_kw:
            biased = [r for r in unused if any(kw.lower() in r['premise'].lower() for kw in bias_kw)]
            other = [r for r in unused if r not in biased]
            # 70% biased, 30% other
            return biased, other

    return unused, []


async def generate_one(client, premise, hypothesis, label, semaphore):
    async with semaphore:
        user_msg = f"""<passage>
{premise[:3000]}
</passage>

<claim>
{hypothesis}
</claim>

The correct classification is: {label}. Explain why."""

        for attempt in range(3):
            try:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": MODEL,
                        "max_tokens": 300,
                        "temperature": 0.7,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                    },
                    timeout=45,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]

                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                else:
                    reasoning = re.sub(r'</?(?:think|answer|output|nothink)>.*', '', content, flags=re.DOTALL).strip()
                    if not reasoning:
                        reasoning = content[:500].strip()

                word_count = len(reasoning.split())
                if word_count < 20 and attempt < 2:
                    continue

                return {
                    "input": f'<passage>\n{premise}\n</passage>\n\n<claim>\n{hypothesis}\n</claim>',
                    "output": f'<think>\n{reasoning}\n</think>\n<answer>{label}</answer>',
                    "label": label,
                    "reasoning": reasoning,
                    "reasoning_word_count": word_count,
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                else:
                    return None


async def run_generation(pairs, n):
    semaphore = asyncio.Semaphore(30)
    results = []

    async with httpx.AsyncClient(timeout=45) as client:
        tasks = []
        for r in pairs[:n]:
            label = LABEL_COLLAPSE[r['label_name']]
            tasks.append(generate_one(client, r['premise'], r['hypothesis'], label, semaphore))

        done = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(tasks)}] {len(results)} succeeded")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate more SFT reasoning chains")
    parser.add_argument("--n", type=int, default=100, help="Number of chains to generate")
    parser.add_argument("--category", type=str, default=None,
                        choices=['temporal', 'liability', 'mae', 'condition'],
                        help="Bias generation toward a specific category")
    parser.add_argument("--stats", action="store_true", help="Just show stats, don't generate")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
    else:
        random.seed(int(datetime.now().timestamp()))

    used = load_already_used()
    biased, other = load_unused_pairs(used, args.category)

    # Count existing incremental results
    existing = 0
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob("*.jsonl"):
            existing += sum(1 for _ in open(f))

    print("=" * 60)
    print("SFT Reasoning Chain Generator")
    print("=" * 60)
    print(f"Already generated:  {len(used)} chains")
    print(f"Incremental so far: {existing} chains")

    if args.category:
        print(f"Category bias:      {args.category} ({len(biased)} matching, {len(other)} other)")
    else:
        total_unused = len(biased) + len(other) if args.category else len(biased)
        print(f"Unused pairs:       {total_unused}")

    unused_all = biased + other if args.category else biased
    dist = Counter(LABEL_COLLAPSE[r['label_name']] for r in unused_all)
    print(f"Unused by label:    {dict(sorted(dist.items()))}")

    if args.stats:
        return

    if not API_KEY:
        print("\nERROR: Set OPENROUTER_API_KEY")
        return

    # Select pairs to generate
    if args.category and biased:
        # 70% from biased category, 30% from other
        n_biased = min(int(args.n * 0.7), len(biased))
        n_other = min(args.n - n_biased, len(other))
        random.shuffle(biased)
        random.shuffle(other)
        selected = biased[:n_biased] + other[:n_other]
    else:
        random.shuffle(unused_all)
        selected = unused_all[:args.n]

    if not selected:
        print("\nNo unused pairs available!")
        return

    actual_n = len(selected)
    est_cost = actual_n * 750 * 0.1 / 1e6
    print(f"\nGenerating {actual_n} chains via {MODEL}")
    print(f"Estimated cost: ~${est_cost:.3f}")
    print()

    results = asyncio.run(run_generation(selected, actual_n))

    print(f"\nGenerated {len(results)}/{actual_n} chains")

    if results:
        avg_words = sum(r['reasoning_word_count'] for r in results) / len(results)
        label_dist = Counter(r['label'] for r in results)
        print(f"Avg reasoning: {avg_words:.0f} words")
        print(f"Labels: {dict(sorted(label_dist.items()))}")

        # Save
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.category}" if args.category else ""
        out_path = RESULTS_DIR / f"gen_{len(results)}{tag}_{timestamp}.jsonl"

        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        print(f"Saved to {out_path}")

        # Show samples
        print("\nSamples:")
        for r in results[:2]:
            print(f"  [{r['label']}] {r['reasoning'][:180]}...")
            print()

        # Show updated totals
        new_total = existing + len(results)
        print(f"Total incremental chains: {new_total}")
        print(f"\nRun again to generate more, or with --category temporal to target weak areas")


if __name__ == "__main__":
    main()
