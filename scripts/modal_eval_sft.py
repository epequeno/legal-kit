"""Evaluate the SFT reasoning model on the expanded NLI eval set via Modal.

Usage:
    cd python && uv run modal run ../scripts/modal_eval_sft.py
"""

import modal
import json

app = modal.App("legalkit-eval-sft")
vol = modal.Volume.from_name("legalkit-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.51",
        "peft>=0.14",
        "accelerate>=1.2",
        "bitsandbytes>=0.45",
    )
)

VOLUME_PATH = "/data"
MODEL_ID = "Qwen/Qwen2.5-3B"
ADAPTER_PATH = f"{VOLUME_PATH}/models/legal-reasoning-sft"
EVAL_PATH = "data/eval/legal_nli_eval_expanded.json"


@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: vol},
    timeout=7200,
)
def evaluate():
    import re
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from collections import Counter

    vol.reload()

    print("=" * 70)
    print("Legal Reasoning SFT — Evaluation")
    print("=" * 70)

    # Load eval data from the mounted volume
    # First, upload it if needed
    raw = json.loads(open(f"{VOLUME_PATH}/eval/legal_nli_eval_expanded.json").read())
    eval_data = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
    print(f"\nEval set: {len(eval_data)} cases")

    # Load model
    print(f"\nLoading {MODEL_ID} + LoRA adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("  Model loaded.")

    # Must match the system prompt used during SFT training (modal_sft_qwen.py)
    SYSTEM_MSG = (
        "You are a legal reasoning expert. Given a passage from a legal contract "
        "and a claim about it, reason step-by-step about the relationship, then classify it. "
        "Output your reasoning in <think> tags and your answer in <answer> tags."
    )

    # Map eval labels to 3-class
    LABEL_MAP = {
        "entailed": "entailed",
        "partially_entailed": "entailed",
        "contradicted": "contradicted",
        "neutral": "neutral",
    }

    results = []
    correct_binary = 0
    correct_3class = 0
    total = 0

    for i, case in enumerate(eval_data):
        premise = case.get("premise", case.get("passage", ""))
        hypothesis = case.get("hypothesis", case.get("claim", ""))
        gold_label = case["label"]
        gold_3class = LABEL_MAP.get(gold_label, gold_label)

        # Format as chat
        user_msg = f"<passage>\n{premise[:3000]}\n</passage>\n\n<claim>\n{hypothesis}\n</claim>"
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse answer
        answer_match = re.search(r'<answer>\s*(entailed|contradicted|neutral)\s*</answer>', response, re.IGNORECASE)
        predicted = answer_match.group(1).lower() if answer_match else "unknown"

        # Parse reasoning
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""

        # Binary: entailed vs not-entailed
        gold_binary = "entailed" if gold_3class == "entailed" else "not_entailed"
        pred_binary = "entailed" if predicted == "entailed" else "not_entailed"

        is_correct_binary = gold_binary == pred_binary
        is_correct_3class = gold_3class == predicted

        if is_correct_binary:
            correct_binary += 1
        if is_correct_3class:
            correct_3class += 1
        total += 1

        results.append({
            "id": case.get("id", i),
            "category": case.get("category", "unknown"),
            "gold_label": gold_label,
            "gold_3class": gold_3class,
            "predicted": predicted,
            "correct_binary": is_correct_binary,
            "correct_3class": is_correct_3class,
            "reasoning": reasoning[:300],
            "raw_response": response[:500],
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(eval_data)}] "
                  f"Binary: {correct_binary}/{total} ({correct_binary/total:.1%}) "
                  f"3-class: {correct_3class}/{total} ({correct_3class/total:.1%})")

    # Final stats
    print(f"\n{'='*70}")
    print(f"RESULTS (n={total})")
    print(f"{'='*70}")
    print(f"  Binary accuracy:  {correct_binary}/{total} ({correct_binary/total:.1%})")
    print(f"  3-class accuracy: {correct_3class}/{total} ({correct_3class/total:.1%})")

    # Per-category breakdown
    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"correct_b": 0, "correct_3": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if r["correct_binary"]:
            by_cat[cat]["correct_b"] += 1
        if r["correct_3class"]:
            by_cat[cat]["correct_3"] += 1

    print(f"\nPer-category binary accuracy:")
    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]
        print(f"  {cat}: {s['correct_b']}/{s['total']} ({s['correct_b']/s['total']:.0%})")

    # Prediction distribution
    pred_dist = Counter(r["predicted"] for r in results)
    print(f"\nPrediction distribution: {dict(sorted(pred_dist.items()))}")

    # Show sample reasoning
    print(f"\nSample outputs:")
    for r in results[:5]:
        print(f"\n  [{r['gold_3class']}→{r['predicted']}] {'✓' if r['correct_3class'] else '✗'}")
        print(f"  Reasoning: {r['reasoning'][:150]}...")

    # Save results
    output = {
        "model": f"{MODEL_ID} + LoRA SFT",
        "eval_samples": total,
        "binary_accuracy": correct_binary / total,
        "three_class_accuracy": correct_3class / total,
        "per_category": by_cat,
        "prediction_distribution": dict(pred_dist),
        "details": results,
    }

    with open(f"{VOLUME_PATH}/eval/sft_eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    vol.commit()

    print(f"\nResults saved to volume: eval/sft_eval_results.json")
    return output


@app.function(image=modal.Image.debian_slim(), timeout=60)
def trigger():
    evaluate.spawn()
    return "Eval spawned successfully"


if __name__ == "__main__":
    import modal as m
    fn = m.Function.from_name("legalkit-eval-sft", "trigger")
    result = fn.remote()
    print(result)
