"""Standalone eval script for EC2 (no Modal dependency).

Usage:
    python eval_local.py \
        --model Qwen/Qwen2.5-7B \
        --adapter ~/legal-reasoning-grpo-7b \
        --eval-data ~/legal_nli_eval_expanded.json \
        --output ~/eval_results.json \
        --s3-upload s3://<bucket>/eval/grpo_7b_eval_results.json
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B", help="Base model ID")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--eval-data", required=True, help="Path to eval JSON file")
    parser.add_argument("--output", default="~/eval_results.json", help="Output path for results JSON")
    parser.add_argument("--s3-upload", default=None, help="S3 URI to upload results to (optional)")
    parser.add_argument("--no-quantize", action="store_true", help="Load in bf16 instead of 4-bit")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    output_path = Path(args.output).expanduser()
    adapter_path = str(Path(args.adapter).expanduser())
    eval_path = str(Path(args.eval_data).expanduser())

    print("=" * 70)
    print(f"Legal Reasoning — Local Evaluation")
    print("=" * 70)
    print(f"  Model:   {args.model}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Eval:    {eval_path}")
    print(f"  Output:  {output_path}")

    # Load eval data
    raw = json.loads(Path(eval_path).read_text())
    eval_data = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
    print(f"\nEval set: {len(eval_data)} cases")

    # Load model
    print(f"\nLoading {args.model} + LoRA adapter...")
    if args.no_quantize:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("  Model loaded.")

    # Must match SFT + GRPO training prompt
    SYSTEM_MSG = (
        "You are a legal reasoning expert. Given a passage from a legal contract "
        "and a claim about it, reason step-by-step about the relationship, then classify it. "
        "Output your reasoning in <think> tags and your answer in <answer> tags."
    )

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

        answer_match = re.search(r'<answer>\s*(entailed|contradicted|neutral)\s*</answer>', response, re.IGNORECASE)
        predicted = answer_match.group(1).lower() if answer_match else "unknown"

        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""

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
    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"correct_b": 0, "correct_3": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if r["correct_binary"]:
            by_cat[cat]["correct_b"] += 1
        if r["correct_3class"]:
            by_cat[cat]["correct_3"] += 1

    pred_dist = Counter(r["predicted"] for r in results)

    print(f"\n{'='*70}")
    print(f"RESULTS (n={total})")
    print(f"{'='*70}")
    print(f"  Binary accuracy:  {correct_binary}/{total} ({correct_binary/total:.1%})")
    print(f"  3-class accuracy: {correct_3class}/{total} ({correct_3class/total:.1%})")
    print(f"\nPer-category binary accuracy:")
    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]
        print(f"  {cat}: {s['correct_b']}/{s['total']} ({s['correct_b']/s['total']:.0%})")
    print(f"\nPrediction distribution: {dict(sorted(pred_dist.items()))}")

    output_data = {
        "model": f"{args.model} + LoRA adapter ({Path(adapter_path).name})",
        "adapter": adapter_path,
        "eval_samples": total,
        "binary_accuracy": correct_binary / total,
        "three_class_accuracy": correct_3class / total,
        "per_category": by_cat,
        "prediction_distribution": dict(pred_dist),
        "details": results,
    }

    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to {output_path}")

    if args.s3_upload:
        print(f"Uploading to {args.s3_upload}...")
        result = subprocess.run(
            ["aws", "s3", "cp", str(output_path), args.s3_upload],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Uploaded successfully.")
        else:
            print(f"  Upload failed: {result.stderr}")

    return output_data


if __name__ == "__main__":
    main()
