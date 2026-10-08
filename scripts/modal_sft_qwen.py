"""SFT fine-tuning of Qwen3-8B on legal reasoning chains via Modal.

Follows the deploy+spawn pattern for laptop-safe training:
    1. modal deploy scripts/modal_sft_qwen.py
    2. python scripts/modal_sft_qwen.py --trigger
    3. Check: modal volume ls legalkit-data models/legal-reasoning-sft/
    4. Download: modal volume get legalkit-data models/legal-reasoning-sft/ models/legal-reasoning-sft/
"""

import modal
import argparse
import sys

app = modal.App("legalkit-sft-qwen")
vol = modal.Volume.from_name("legalkit-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
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
OUTPUT_DIR = f"{VOLUME_PATH}/models/legal-reasoning-sft"
TRAIN_PATH = f"{VOLUME_PATH}/sft_reasoning/train.jsonl"
VAL_PATH = f"{VOLUME_PATH}/sft_reasoning/val.jsonl"


@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: vol},
    timeout=7200,  # 2 hours
)
def train_sft():
    import json
    import os
    import torch
    from pathlib import Path
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    vol.reload()

    print("=" * 70)
    print("Legal Reasoning SFT — Qwen2.5-3B + LoRA")
    print("=" * 70)

    # Load data
    print("\nLoading training data...")
    train_data = [json.loads(l) for l in open(TRAIN_PATH)]
    val_data = [json.loads(l) for l in open(VAL_PATH)]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Format as chat messages for Qwen3
    SYSTEM_MSG = (
        "You are a legal reasoning expert. Given a passage from a legal contract "
        "and a claim about it, reason step-by-step about the relationship, then classify it. "
        "Output your reasoning in <think> tags and your answer in <answer> tags."
    )

    def format_chat(example):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]},
            ]
        }

    train_ds = Dataset.from_list([format_chat(d) for d in train_data])
    val_ds = Dataset.from_list([format_chat(d) for d in val_data])

    print(f"  Train examples: {len(train_ds)}")
    print(f"  Val examples: {len(val_ds)}")

    # Load model with 4-bit quantization
    print(f"\nLoading {MODEL_ID} with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # Training config
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch = 16
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        max_length=1024,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
    )

    # Train
    print("\nStarting SFT training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    result = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Train loss: {result.training_loss:.4f}")
    print(f"  Train runtime: {result.metrics['train_runtime']:.0f}s")

    # Eval
    eval_result = trainer.evaluate()
    print(f"  Eval loss: {eval_result['eval_loss']:.4f}")

    # Save LoRA adapter + tokenizer
    print(f"\nSaving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save results
    results = {
        "model": MODEL_ID,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_loss": result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "epochs": 3,
        "lora_rank": 16,
        "effective_batch_size": 16,
        "learning_rate": 2e-5,
        "trainable_params": trainable,
        "total_params": total,
        "runtime_seconds": result.metrics["train_runtime"],
    }
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    vol.commit()
    print("Done! Results saved to volume.")
    return results


# --- Trigger entrypoint (for deploy+spawn pattern) ---
@app.function(image=modal.Image.debian_slim(), timeout=60)
def trigger():
    train_sft.spawn()
    return "Training spawned successfully"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger", action="store_true",
                        help="Trigger training on deployed app")
    parser.add_argument("--upload", action="store_true",
                        help="Upload training data to Modal volume")
    args = parser.parse_args()

    if args.upload:
        import subprocess
        print("Uploading SFT training data to Modal volume...")
        subprocess.run([
            "modal", "volume", "put", "legalkit-data",
            "data/sft_reasoning/train.jsonl",
            "sft_reasoning/train.jsonl",
        ], check=True)
        subprocess.run([
            "modal", "volume", "put", "legalkit-data",
            "data/sft_reasoning/val.jsonl",
            "sft_reasoning/val.jsonl",
        ], check=True)
        print("Upload complete!")

    elif args.trigger:
        import modal
        fn = modal.Function.from_name("legalkit-sft-qwen", "trigger")
        result = fn.remote()
        print(result)

    else:
        print("Usage:")
        print("  python scripts/modal_sft_qwen.py --upload     # Upload data")
        print("  modal deploy scripts/modal_sft_qwen.py        # Deploy app")
        print("  python scripts/modal_sft_qwen.py --trigger    # Start training")
        print("  modal volume ls legalkit-data models/legal-reasoning-sft/  # Check")
