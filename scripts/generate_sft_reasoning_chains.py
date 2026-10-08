"""Generate SFT reasoning chains for legal entailment training.

Takes existing (passage, claim, label) pairs and generates chain-of-thought
reasoning chains using Claude. Output format matches what we'll train the
reasoning model on:

  Input:  <passage>{passage}</passage>\n<claim>{claim}</claim>
  Output: <think>{reasoning}</think>\n<answer>{label}</answer>

Usage:
    python scripts/generate_sft_reasoning_chains.py --n 100        # Validation batch
    python scripts/generate_sft_reasoning_chains.py --n 5000       # Full run
    python scripts/generate_sft_reasoning_chains.py --n 50 --dry-run
"""

import json
import os
import re
import asyncio
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. pip install httpx")
    exit(1)

ROOT = Path(__file__).parent.parent

# Map 4-class to 3-class for output
LABEL_COLLAPSE = {
    "entailed": "entailed",
    "partially_entailed": "entailed",
    "contradicted": "contradicted",
    "neutral": "neutral",
}

SYSTEM_PROMPT = """You are a legal reasoning expert. Given a passage from a legal contract and a claim about it, explain your reasoning step-by-step, then classify the relationship.

Your reasoning should:
- Identify the relevant legal structure (carve-out, exception, modal verb, defined term, temporal provision, etc.)
- Apply it to the specific claim
- Note any qualifiers, conditions, or exceptions that affect the analysis
- Be concise: 50-150 words. Focus on the key legal logic, not restating the passage.
- Use specific legal terminology where appropriate

Output format (exactly):
<think>
[Your reasoning here]
</think>
<answer>[entailed|contradicted|neutral]</answer>"""

USER_TEMPLATE = """<passage>
{passage}
</passage>

<claim>
{claim}
</claim>

The correct classification is: {label}. Explain why."""


async def generate_chain(client: httpx.AsyncClient, api_key: str,
                          premise: str, hypothesis: str, label: str,
                          max_retries: int = 3) -> dict | None:
    """Generate a reasoning chain for one (passage, claim, label) triple."""
    
    # Truncate very long premises to save tokens
    if len(premise) > 3000:
        premise = premise[:3000] + "..."
    
    user_msg = USER_TEMPLATE.format(
        passage=premise,
        claim=hypothesis,
        label=label,
    )
    
    for attempt in range(max_retries):
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-sonnet-4",
                    "max_tokens": 400,
                    "temperature": 0.7,  # Some variation for diversity
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_msg}],
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            
            # Parse the response
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            
            if not think_match or not answer_match:
                if attempt < max_retries - 1:
                    continue
                return None
            
            reasoning = think_match.group(1).strip()
            predicted_label = answer_match.group(1).strip().lower()
            
            # Validate
            if predicted_label not in ("entailed", "contradicted", "neutral"):
                if attempt < max_retries - 1:
                    continue
                return None
            
            # Check reasoning quality: must be 20+ words
            word_count = len(reasoning.split())
            if word_count < 20:
                if attempt < max_retries - 1:
                    continue
                return None
            
            return {
                "reasoning": reasoning,
                "predicted_label": predicted_label,
                "word_count": word_count,
                "raw_response": content,
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            print(f"    Failed after {max_retries} attempts: {e}")
            return None


async def generate_batch(rows: list[dict], api_key: str, 
                          batch_size: int = 10) -> list[dict]:
    """Generate reasoning chains for a batch of rows."""
    results = []
    total = len(rows)
    
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            print(f"  Batch {batch_num}/{total_batches} "
                  f"(rows {i+1}-{min(i+batch_size, total)}/{total})...")
            
            tasks = [
                generate_chain(
                    client, api_key,
                    row["premise"],
                    row["hypothesis"],
                    LABEL_COLLAPSE[row["label_name"]],
                )
                for row in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            
            for row, result in zip(batch, batch_results):
                if result is None:
                    continue
                
                label_3class = LABEL_COLLAPSE[row["label_name"]]
                
                results.append({
                    # SFT training fields
                    "input": f"<passage>\n{row['premise']}\n</passage>\n\n<claim>\n{row['hypothesis']}\n</claim>",
                    "output": f"<think>\n{result['reasoning']}\n</think>\n<answer>{label_3class}</answer>",
                    
                    # Metadata
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": label_3class,
                    "reasoning": result["reasoning"],
                    "reasoning_word_count": result["word_count"],
                    "predicted_label": result["predicted_label"],
                    "label_match": result["predicted_label"] == label_3class,
                    "source": row.get("source", "unknown"),
                    "doc_id": row.get("doc_id", ""),
                })
            
            # Rate limiting
            if i + batch_size < total:
                await asyncio.sleep(0.5)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="Number of pairs to generate chains for (default: 100)")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-dir", type=str, default="data/training_v2",
                        help="Source training data directory")
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: Set OPENROUTER_API_KEY")
        return
    
    print("=" * 70)
    print("Generating SFT Reasoning Chains")
    print("=" * 70)
    
    # Load source data
    print(f"\nLoading source data from {args.input_dir}...")
    all_rows = []
    for split in ("train", "val", "test"):
        path = Path(args.input_dir) / f"{split}.jsonl"
        if path.exists():
            rows = [json.loads(l) for l in open(path)]
            all_rows.extend(rows)
    
    print(f"  Total source pairs: {len(all_rows)}")
    
    # Sample N pairs, stratified by label
    random.seed(args.seed)
    n = min(args.n, len(all_rows))
    
    # Stratified sampling: proportional to label distribution
    by_label = {}
    for row in all_rows:
        label = LABEL_COLLAPSE[row["label_name"]]
        by_label.setdefault(label, []).append(row)
    
    sampled = []
    per_label = max(1, n // len(by_label))
    for label, rows in by_label.items():
        random.shuffle(rows)
        count = min(per_label, len(rows))
        # Adjust last label to hit exactly n
        sampled.extend(rows[:count])
    
    # Trim or pad to exactly n
    random.shuffle(sampled)
    sampled = sampled[:n]
    
    sample_dist = Counter(LABEL_COLLAPSE[r["label_name"]] for r in sampled)
    print(f"  Sampled {len(sampled)} pairs:")
    for l, c in sorted(sample_dist.items()):
        print(f"    {l}: {c}")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would generate {len(sampled)} reasoning chains")
        print(f"Estimated cost: ~${len(sampled) * 0.004:.2f}")
        for row in sampled[:3]:
            label = LABEL_COLLAPSE[row["label_name"]]
            print(f"\n  [{label}] {row['hypothesis'][:100]}")
            print(f"    Passage: {row['premise'][:100]}...")
        return
    
    # Generate
    print(f"\nGenerating reasoning chains for {len(sampled)} pairs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Estimated cost: ~${len(sampled) * 0.004:.2f}")
    
    results = asyncio.run(generate_batch(sampled, api_key, args.batch_size))
    
    # Stats
    print(f"\n{'='*70}")
    print(f"Generated {len(results)} reasoning chains ({len(results)/len(sampled):.0%} success rate)")
    
    label_match = sum(1 for r in results if r["label_match"])
    print(f"  Label agreement: {label_match}/{len(results)} ({label_match/len(results):.0%})")
    
    avg_words = sum(r["reasoning_word_count"] for r in results) / len(results)
    print(f"  Avg reasoning length: {avg_words:.0f} words")
    
    result_dist = Counter(r["label"] for r in results)
    print(f"  Label distribution:")
    for l, c in sorted(result_dist.items()):
        print(f"    {l}: {c}")
    
    # Save
    out_dir = ROOT / "data" / "sft_reasoning"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"reasoning_chains_{len(results)}_{timestamp}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Also save as latest
    latest_path = out_dir / "reasoning_chains_latest.jsonl"
    with open(latest_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\n  Saved to: {out_path}")
    print(f"  Latest: {latest_path}")
    
    # Show samples
    print(f"\n{'='*70}")
    print("SAMPLE OUTPUTS")
    print(f"{'='*70}")
    for r in results[:3]:
        print(f"\n[{r['label']}] (match={r['label_match']}, {r['reasoning_word_count']} words)")
        print(f"  Claim: {r['hypothesis'][:120]}")
        print(f"  Reasoning: {r['reasoning'][:200]}...")
    
    if len(results) < 200:
        print(f"\nThis was a validation batch. To generate the full dataset:")
        print(f"  python scripts/generate_sft_reasoning_chains.py --n 5000")


if __name__ == "__main__":
    main()
