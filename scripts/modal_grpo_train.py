"""GRPO training for legal reasoning model via Modal.

Takes the SFT checkpoint and trains with Group Relative Policy Optimization
using a heuristic multi-signal reward (no LLM judge).

Deploy+spawn pattern:
    1. Upload prompts: cd python && uv run python ../scripts/modal_grpo_train.py --upload
    2. Deploy: cd python && uv run modal deploy ../scripts/modal_grpo_train.py
    3. Trigger: cd python && uv run python ../scripts/modal_grpo_train.py --trigger
    4. Check: cd python && uv run modal volume ls legalkit-data models/legal-reasoning-grpo/
"""

import modal
import json
import re
import argparse
import subprocess
from pathlib import Path

app = modal.App("legalkit-grpo")
vol = modal.Volume.from_name("legalkit-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6",
        "transformers>=4.51",
        "peft>=0.14",
        "trl>=0.16",
        "datasets>=3.2",
        "accelerate>=1.2",
        "bitsandbytes>=0.45",
    )
)

VOLUME_PATH = "/data"
MODEL_ID = "Qwen/Qwen2.5-3B"
SFT_ADAPTER_PATH = f"{VOLUME_PATH}/models/legal-reasoning-sft"
OUTPUT_DIR = f"{VOLUME_PATH}/models/legal-reasoning-grpo"
PROMPTS_PATH = f"{VOLUME_PATH}/grpo_prompts/grpo_prompts.jsonl"

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
    """Multi-signal heuristic reward for legal reasoning (v2).
    
    Weights: correctness 0.4, grounding 0.2, format 0.15, length 0.1, 
             consistency 0.1, termination 0.05 = 1.0 max
    
    Key fix from v1: heavily penalize unterminated outputs. In the first GRPO run,
    85-100% of completions hit max_completion_length without producing </answer>.
    The model was learning to ramble instead of concluding.
    """
    score = 0.0

    # Parse completion
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    answer_match = re.search(r'<answer>\s*(entailed|contradicted|neutral)\s*</answer>',
                             completion, re.IGNORECASE)

    reasoning = think_match.group(1).strip() if think_match else ""
    predicted = answer_match.group(1).lower() if answer_match else ""

    # --- Termination check (critical) ---
    has_answer_close = '</answer>' in completion
    if not has_answer_close:
        # No proper termination — return strong negative signal
        if think_match:
            score += 0.05  # at least started reasoning
        return score - 0.2  # net negative reward

    # 1. Format compliance (0.15) — proper structure with both tags
    if think_match and answer_match:
        score += 0.15
    elif answer_match:
        score += 0.05

    # 2. Correctness (0.4)
    if predicted == gold_label:
        score += 0.4

    # 3. Passage grounding (0.2)
    if reasoning:
        passage_words = set(passage.lower().split()) - COMMON_WORDS
        reasoning_words = set(reasoning.lower().split()) - COMMON_WORDS
        shared = {w for w in (passage_words & reasoning_words) if len(w) >= 4}
        score += 0.2 * min(len(shared) / 10.0, 1.0)

    # 4. Length (0.1) — reward concise, penalize verbosity
    word_count = len(reasoning.split()) if reasoning else 0
    if 40 <= word_count <= 120:
        score += 0.1
    elif 25 <= word_count <= 150:
        score += 0.05

    # 5. Consistency (0.1)
    if reasoning and predicted:
        r_lower = reasoning.lower()
        inconsistent = (
            (predicted == 'entailed' and any(w in r_lower for w in ['contradict', 'conflicts with', 'negates'])) or
            (predicted == 'contradicted' and any(w in r_lower for w in ['supports', 'confirms', 'establishes'])) or
            (predicted == 'neutral' and any(w in r_lower for w in ['directly supports', 'explicitly confirms', 'directly contradicts']))
        )
        score += -0.05 if inconsistent else 0.1

    # 6. Termination bonus (0.05)
    post_answer = completion.split('</answer>')[-1].strip()
    if len(post_answer) < 10:
        score += 0.05

    return score


def reward_fn(completions, passage, claim, label, **kwargs):
    """Score a batch of completions.
    
    TRL GRPOTrainer passes completions — could be strings or chat message lists.
    Extra dataset columns come as keyword args (lists, one per sample).
    """
    rewards = []
    for i, completion in enumerate(completions):
        # Handle both string and chat-format completions
        if isinstance(completion, list):
            # Chat format: extract the assistant message content
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


@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: vol},
    timeout=21600,  # 6 hours
)
def train_grpo():
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from trl import GRPOTrainer, GRPOConfig

    vol.reload()

    print("=" * 70)
    print("Legal Reasoning GRPO Training")
    print("=" * 70)

    # Load prompts
    print("\nLoading GRPO prompts...")
    prompts_data = [json.loads(l) for l in open(PROMPTS_PATH)]
    print(f"  {len(prompts_data)} prompts")

    from collections import Counter
    label_dist = Counter(p["label"] for p in prompts_data)
    source_dist = Counter(p["source"] for p in prompts_data)
    print(f"  Labels: {dict(sorted(label_dist.items()))}")
    print(f"  Sources: {dict(sorted(source_dist.items()))}")

    # Format dataset
    SYSTEM_MSG = (
        "You are a legal reasoning assistant specializing in contract interpretation. "
        "Given a passage from a legally binding agreement and a claim about its meaning or implications, "
        "carefully analyze the text with strict attention to contractual language, including modal verbs "
        '(e.g., "may" vs. "must"), defined terms, exceptions, exclusions, and temporal or conditional clauses. '
        "Begin by restating the key provisions in the passage, then assess whether the claim is fully supported, "
        "logically contradicted, or neither. Your reasoning must be precise, logically structured, and grounded "
        "exclusively in the text—do not infer intent or rely on external knowledge. "
        "Construct a step-by-step analysis (50–150 words) that traces the legal logic, highlights critical "
        "linguistic nuances, and resolves any hierarchical qualifications. "
        "If the claim overgeneralizes, adds obligations, or misreads negations or exceptions, treat it as contradicted. "
        "When in doubt, default to neutral. "
        "Output your reasoning in <think> tags and your classification in <answer> tags."
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
    print(f"  Dataset: {len(dataset)} examples")

    # Load model with SFT adapter
    print(f"\nLoading {MODEL_ID} with SFT adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    # Load SFT adapter — continue training it with GRPO
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # GRPO config
    print("\nConfiguring GRPO trainer...")
    # Inspect GRPOConfig to find valid parameter names
    import inspect
    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    print(f"  GRPOConfig params: {', '.join(list(grpo_params.keys())[:20])}...")
    
    config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,  # faster steps
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        max_completion_length=256,  # bumped from 192 — give model room to terminate
        num_generations=4,
        report_to="none",
        seed=42,
        beta=0.05,
        gradient_checkpointing=True,
    )
    
    # Add max_prompt_length only if supported
    if 'max_prompt_length' in grpo_params:
        config_kwargs['max_prompt_length'] = 768
    
    training_args = GRPOConfig(**config_kwargs)

    # Train
    print("\nStarting GRPO training...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    result = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Train loss: {result.training_loss:.4f}")
    print(f"  Runtime: {result.metrics['train_runtime']:.0f}s")

    # Log reward stats from training
    print("\n  Training metrics:")
    for key, val in sorted(result.metrics.items()):
        if 'reward' in key or 'loss' in key:
            print(f"    {key}: {val}")

    # Save
    print(f"\nSaving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    results = {
        "model": f"{MODEL_ID} + SFT + GRPO",
        "sft_adapter": SFT_ADAPTER_PATH,
        "prompts": len(prompts_data),
        "train_loss": result.training_loss,
        "epochs": 3,
        "group_size": 4,
        "kl_beta": 0.05,
        "learning_rate": 5e-6,
        "runtime_seconds": result.metrics["train_runtime"],
        "metrics": {k: v for k, v in result.metrics.items()
                    if isinstance(v, (int, float))},
    }
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    vol.commit()
    print("Done!")
    return results


@app.function(image=modal.Image.debian_slim(), timeout=60)
def trigger():
    train_grpo.spawn()
    return "GRPO training spawned successfully"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    if args.upload:
        print("Uploading GRPO prompts to Modal volume...")
        subprocess.run([
            "modal", "volume", "put", "legalkit-data",
            "data/grpo_prompts/grpo_prompts.jsonl",
            "grpo_prompts/grpo_prompts.jsonl",
        ], check=True)
        print("Done!")

    elif args.trigger:
        import modal as m
        fn = m.Function.from_name("legalkit-grpo", "trigger")
        result = fn.remote()
        print(result)

    else:
        print("Usage:")
        print("  cd python && uv run python ../scripts/modal_grpo_train.py --upload")
        print("  cd python && uv run modal deploy ../scripts/modal_grpo_train.py")
        print("  cd python && uv run python ../scripts/modal_grpo_train.py --trigger")
