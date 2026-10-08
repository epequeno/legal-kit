#!/usr/bin/env python3
"""GRPO training — standalone script for EC2/local GPU.

No Modal dependency. Runs directly on any machine with a CUDA GPU.

Usage:
    python scripts/train_grpo_local.py
    python scripts/train_grpo_local.py --epochs 2 --group-size 8
    python scripts/train_grpo_local.py --resume-from models/legal-reasoning-grpo/checkpoint-200
"""

import json
import os
import re
import argparse
import inspect
from pathlib import Path
from collections import Counter

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

ROOT = Path(__file__).parent.parent

# --- Reward function ---

COMMON_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'must', 'ought',
    'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'and', 'but', 'or', 'nor', 'not', 'no', 'so',
    'if', 'than', 'that', 'this', 'these', 'those', 'it', 'its',
    'any', 'all', 'each', 'every', 'both', 'such', 'other', 'some',
    'which', 'who', 'whom', 'what', 'where', 'when', 'how', 'there',
    'here', 'also', 'only', 'very', 'just', 'more', 'most', 'own',
    'same', 'about', 'up', 'out', 'then', 'them', 'they', 'their',
    'we', 'our', 'he', 'she', 'his', 'her', 'you', 'your', 'my',
    'me', 'him', 'us', 'i', 'party', 'agreement', 'section', 'article',
    'pursuant', 'herein', 'thereof', 'hereto', 'provided', 'upon',
    'respect', 'including', 'without', 'limitation', 'whether', 'otherwise',
}


def compute_reward(passage, claim, completion, gold_label):
    """Multi-signal heuristic reward.
    
    Weights: correctness 0.4, grounding 0.2, format 0.15, length 0.1, 
             consistency 0.1, termination 0.05 = 1.0 max
    
    Key fix from v1: heavily penalize unterminated outputs. In the first GRPO run,
    85-100% of completions hit max_completion_length without producing </answer>.
    The model was learning to ramble instead of concluding.
    """
    score = 0.0

    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    answer_match = re.search(r'<answer>\s*(entailed|contradicted|neutral)\s*</answer>',
                             completion, re.IGNORECASE)

    reasoning = think_match.group(1).strip() if think_match else ""
    predicted = answer_match.group(1).lower() if answer_match else ""

    # --- Termination check (critical) ---
    # If no </answer> tag, the model hit max tokens without finishing.
    # Give a harsh penalty to teach it to be concise and terminate properly.
    has_answer_close = '</answer>' in completion
    if not has_answer_close:
        # No proper termination — return strong negative signal
        # Still give partial credit for any correct partial output
        if think_match:
            score += 0.05  # at least started reasoning
        return score - 0.2  # net negative reward

    # Format (0.15) — proper structure with both tags
    if think_match and answer_match:
        score += 0.15
    elif answer_match:
        score += 0.05  # answer without think tags — partial credit

    # Correctness (0.4) — the most important signal
    if predicted == gold_label:
        score += 0.4

    # Passage grounding (0.2) — reasoning cites specific contract language
    if reasoning:
        passage_words = set(passage.lower().split()) - COMMON_WORDS
        reasoning_words = set(reasoning.lower().split()) - COMMON_WORDS
        shared = {w for w in (passage_words & reasoning_words) if len(w) >= 4}
        score += 0.2 * min(len(shared) / 10.0, 1.0)

    # Length (0.1) — reward concise reasoning, penalize verbosity
    word_count = len(reasoning.split()) if reasoning else 0
    if 40 <= word_count <= 120:
        score += 0.1  # sweet spot
    elif 25 <= word_count <= 150:
        score += 0.05  # acceptable
    elif word_count > 150:
        score += 0.0  # too verbose, no reward (but no penalty beyond missing bonus)

    # Consistency (0.1) — reasoning should match the label
    if reasoning and predicted:
        r_lower = reasoning.lower()
        inconsistent = (
            (predicted == 'entailed' and any(w in r_lower for w in ['contradict', 'conflicts with', 'negates'])) or
            (predicted == 'contradicted' and any(w in r_lower for w in ['supports', 'confirms', 'establishes'])) or
            (predicted == 'neutral' and any(w in r_lower for w in ['directly supports', 'explicitly confirms', 'directly contradicts']))
        )
        score += -0.05 if inconsistent else 0.1

    # Termination bonus (0.05) — extra reward for clean, complete output
    # Check that the completion ends shortly after </answer> (no trailing garbage)
    post_answer = completion.split('</answer>')[-1].strip()
    if len(post_answer) < 10:
        score += 0.05

    return score


def reward_fn(completions, passage, claim, label, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            comp_text = ""
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    comp_text = msg.get("content", "")
            if not comp_text and completion:
                comp_text = str(completion[-1].get("content", "")) if isinstance(completion[-1], dict) else str(completion[-1])
        else:
            comp_text = str(completion)

        r = compute_reward(
            passage[i] if i < len(passage) else "",
            claim[i] if i < len(claim) else "",
            comp_text,
            label[i] if i < len(label) else "",
        )
        rewards.append(r)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B",
                        help="Base model ID (default: Qwen/Qwen2.5-3B)")
    parser.add_argument("--sft-adapter", default=None,
                        help="Path to SFT adapter. If omitted, trains from base model.")
    parser.add_argument("--prompts", default="data/grpo_prompts/grpo_prompts.jsonl")
    parser.add_argument("--output", default="models/legal-reasoning-grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.05, help="KL penalty")
    parser.add_argument("--max-completion-length", type=int, default=256,
                        help="Max tokens for completion. Bumped from 192 to give model room to terminate properly.")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--no-quantize", action="store_true", help="Load model in fp16 instead of 4-bit")
    parser.add_argument("--shutdown-on-complete", action="store_true",
                        help="Shutdown the machine after training completes (for EC2 spot instances)")
    parser.add_argument("--s3-upload", type=str, default=None,
                        help="S3 URI to upload results to before shutdown (e.g. s3://my-bucket/legalkit)")
    args = parser.parse_args()

    # Support both relative-to-ROOT paths (local dev) and absolute paths (EC2)
    sft_path = None
    if args.sft_adapter:
        sft_path = Path(args.sft_adapter) if os.path.isabs(args.sft_adapter) else ROOT / args.sft_adapter
    prompts_path = Path(args.prompts) if os.path.isabs(args.prompts) else ROOT / args.prompts
    output_dir = Path(args.output) if os.path.isabs(args.output) else ROOT / args.output

    print("=" * 70)
    print("Legal Reasoning GRPO Training (Local)")
    print("=" * 70)

    # Load prompts
    print(f"\nLoading prompts from {prompts_path}...")
    prompts_data = [json.loads(l) for l in open(prompts_path)]
    print(f"  {len(prompts_data)} prompts")
    print(f"  Labels: {dict(sorted(Counter(p['label'] for p in prompts_data).items()))}")

    # Must match SFT training prompt (modal_sft_qwen.py) — no "default to neutral"
    # which empirically causes the model to over-predict neutral during GRPO training.
    SYSTEM_MSG = (
        "You are a legal reasoning expert. Given a passage from a legal contract "
        "and a claim about it, reason step-by-step about the relationship, then classify it. "
        "Output your reasoning in <think> tags and your answer in <answer> tags."
    )

    def format_prompt(p):
        user_msg = f"<passage>\n{p['passage'][:2500]}\n</passage>\n\n<claim>\n{p['claim']}\n</claim>"
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg},
            ],
            "passage": p["passage"][:2500],
            "claim": p["claim"],
            "label": p["label"],
        }

    dataset = Dataset.from_list([format_prompt(p) for p in prompts_data])

    # Load model
    model_id = args.model
    print(f"\nLoading {model_id}...")

    if args.no_quantize:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load SFT adapter or create fresh LoRA
    if sft_path and sft_path.exists():
        print(f"  Loading SFT adapter from {sft_path}...")
        model = PeftModel.from_pretrained(base_model, str(sft_path), is_trainable=True)
    else:
        from peft import LoraConfig, get_peft_model
        if sft_path and not sft_path.exists():
            print(f"  WARNING: SFT adapter not found at {sft_path}, creating fresh LoRA")
        else:
            print(f"  No SFT adapter specified, creating fresh LoRA for GRPO from base model")
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # Config
    config_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        max_completion_length=args.max_completion_length,
        num_generations=args.group_size,
        report_to="none",
        seed=42,
        beta=args.beta,
        gradient_checkpointing=True,
    )

    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    if 'max_prompt_length' in grpo_params:
        config_kwargs['max_prompt_length'] = 768

    training_args = GRPOConfig(**config_kwargs)

    # Train
    print(f"\nStarting GRPO training...")
    print(f"  Epochs: {args.epochs}, Group size: {args.group_size}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  LR: {args.lr}, KL beta: {args.beta}")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    result = trainer.train(resume_from_checkpoint=args.resume_from)

    print(f"\nTraining complete!")
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  Runtime: {result.metrics['train_runtime']:.0f}s")

    for key, val in sorted(result.metrics.items()):
        if 'reward' in key:
            print(f"  {key}: {val}")

    # Save
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    results = {
        "model": f"{model_id}" + (" + SFT" if sft_path else "") + " + GRPO",
        "prompts": len(prompts_data),
        "train_loss": result.training_loss,
        "epochs": args.epochs,
        "group_size": args.group_size,
        "kl_beta": args.beta,
        "learning_rate": args.lr,
        "runtime_seconds": result.metrics["train_runtime"],
        "metrics": {k: v for k, v in result.metrics.items() if isinstance(v, (int, float))},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_dir}")

    # Upload results to S3 if bucket specified (for unattended EC2 runs)
    if args.s3_upload:
        import subprocess
        s3_dest = f"{args.s3_upload.rstrip('/')}/{output_dir.name}/"
        print(f"\nUploading results to {s3_dest}...")
        cp = subprocess.run(
            ["aws", "s3", "sync", str(output_dir), s3_dest],
            capture_output=True, text=True
        )
        if cp.returncode == 0:
            print(f"  Upload complete!")
        else:
            print(f"  Upload failed: {cp.stderr}")

    # Auto-shutdown for unattended EC2 runs
    if args.shutdown_on_complete:
        import subprocess
        print("\n*** --shutdown-on-complete: shutting down in 60 seconds ***")
        print("    Cancel with: sudo shutdown -c")
        subprocess.run(["sudo", "shutdown", "-h", "+1"], check=False)


if __name__ == "__main__":
    main()
