"""Regenerate neutral reasoning chains that were too generic.

Uses qwen/qwen3-235b-a22b-2507 via OpenRouter (~$0.05 for 771 examples).

Usage:
    cd python && uv run python ../scripts/regen_neutral_reasoning.py
"""

import json
import re
import asyncio
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "qwen/qwen3-235b-a22b-2507"

SYSTEM_PROMPT = """You are a legal reasoning expert. Given a passage from a legal contract and a claim about it, explain your reasoning step-by-step, then classify the relationship.

Your reasoning MUST be specific:
- State specifically what legal provisions the passage covers (e.g. "The passage defines indemnification obligations under Section 8.2")
- State specifically what the claim is about (e.g. "The claim concerns the cap on liability")
- Explain specifically why the passage doesn't support or contradict the claim
- If the passage is tangentially related, explain precisely where the logical gap is

DO NOT use generic phrases like:
- "neither supports nor contradicts"
- "different aspects of the contractual relationship"
- "the premise does not provide sufficient information"

Be concise: 50-150 words.

Output format:
<think>
[Your specific reasoning here]
</think>
<answer>[entailed|contradicted|neutral]</answer>"""


async def generate_one(client, idx, item, semaphore):
    async with semaphore:
        user_msg = f"""<passage>
{item["premise"][:3000]}
</passage>

<claim>
{item["hypothesis"]}
</claim>

The correct classification is: {item["label"]}. Explain why."""

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
                        reasoning = content.strip()

                word_count = len(reasoning.split())
                if word_count < 20 and attempt < 2:
                    continue

                sft_input = f'<passage>\n{item["premise"]}\n</passage>\n\n<claim>\n{item["hypothesis"]}\n</claim>'
                sft_output = f'<think>\n{reasoning}\n</think>\n<answer>{item["label"]}</answer>'

                return idx, {
                    "input": sft_input,
                    "output": sft_output,
                    "label": item["label"],
                    "reasoning": reasoning,
                    "reasoning_word_count": word_count,
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                else:
                    return idx, None


async def main():
    regen_dir = ROOT / "data" / "sft_reasoning" / "regen_batches"
    all_items = []
    for f in sorted(regen_dir.glob("regen_*.json")):
        all_items.extend(json.load(open(f)))

    print(f"Regenerating {len(all_items)} neutral reasoning chains")
    print(f"Model: {MODEL}")
    print()

    semaphore = asyncio.Semaphore(30)  # 30 concurrent
    results = [None] * len(all_items)
    done = 0

    async with httpx.AsyncClient(timeout=45) as client:
        tasks = [generate_one(client, i, item, semaphore) for i, item in enumerate(all_items)]

        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            results[idx] = result
            done += 1
            if done % 100 == 0:
                success = sum(1 for r in results if r is not None)
                print(f"  [{done}/{len(all_items)}] {success} succeeded")

    # Filter out failures
    good = [r for r in results if r is not None]
    print(f"\nGenerated {len(good)}/{len(all_items)} chains")

    # Quality check
    generic_phrases = ['neither supports', 'different aspects', 'address different',
                       'neutral with respect', 'does not provide sufficient']
    still_generic = sum(1 for r in good if any(p in r['reasoning'].lower() for p in generic_phrases))
    print(f"Still generic: {still_generic}/{len(good)} ({still_generic/len(good):.0%})")

    avg_words = sum(r['reasoning_word_count'] for r in good) / len(good)
    print(f"Avg reasoning length: {avg_words:.0f} words")

    # Sample
    print("\nSamples:")
    for r in good[:3]:
        print(f"  {r['reasoning'][:200]}...")
        print()

    out_path = ROOT / "data" / "sft_reasoning" / "regen_neutral_results.jsonl"
    with open(out_path, "w") as f:
        for r in good:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
