"""Train a legal-specific cross-encoder reranker on Modal.

Converts (query, positive, hard_negative) triplets from the embedding training
set into binary classification pairs and fine-tunes a cross-encoder using the
sentence-transformers v5 CrossEncoderTrainer API.

Usage:
    cd python && uv run modal deploy ../scripts/modal_train_reranker.py
    cd python && uv run python ../scripts/modal_train_reranker.py --trigger

Output: models/legal-reranker/ on the Modal volume legalkit-data
"""

import modal
import json

app = modal.App("legalkit-train-reranker")
vol = modal.Volume.from_name("legalkit-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "sentence-transformers>=5.0",
        "transformers>=4.40",
        "datasets>=2.14",
        "scikit-learn>=1.3",
        "numpy",
    )
)

VOLUME_PATH = "/data"
BASE_MODEL = "BAAI/bge-reranker-v2-m3"  # Multilingual, strong on legal text, ~570M params


@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: vol},
    timeout=7200,
)
def train():
    import json
    import random
    import numpy as np
    import torch
    from pathlib import Path
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder import losses as ce_losses
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

    vol.reload()

    print("=" * 70)
    print("Legal Reranker — Cross-Encoder Fine-Tuning (sentence-transformers v5)")
    print("=" * 70)
    print(f"  Base model: {BASE_MODEL}")

    random.seed(42)
    np.random.seed(42)

    # ── Load triplets ──────────────────────────────────────────────────────────
    train_path = f"{VOLUME_PATH}/embedding_training/train.jsonl"
    val_path   = f"{VOLUME_PATH}/embedding_training/val.jsonl"

    with open(train_path) as f:
        train_raw = [json.loads(l) for l in f]
    with open(val_path) as f:
        val_raw = [json.loads(l) for l in f]

    print(f"\n  Train triplets: {len(train_raw)}")
    print(f"  Val triplets:   {len(val_raw)}")

    # ── Build labeled-pair dataset ─────────────────────────────────────────────
    # v5 BinaryCrossEntropyLoss expects columns: query, response, label (1/0)
    queries, responses, labels = [], [], []
    for ex in train_raw:
        queries.append(ex["query"])
        responses.append(ex["positive"])
        labels.append(1)
        queries.append(ex["query"])
        responses.append(ex["hard_negative"])
        labels.append(0)

    # Shuffle pairs together
    combined = list(zip(queries, responses, labels))
    random.shuffle(combined)
    queries, responses, labels = zip(*combined)

    train_dataset = Dataset.from_dict({
        "query":    list(queries),
        "response": list(responses),
        "label":    list(labels),
    })

    print(f"\n  Total training pairs: {len(train_dataset)} ({len(train_raw)} pos + {len(train_raw)} neg)")

    # ── Build reranking evaluator ──────────────────────────────────────────────
    # v5 CrossEncoderRerankingEvaluator expects list of dicts:
    # [{"query": str, "positive": [str], "negative": [str]}, ...]
    val_samples = [
        {
            "query":    ex["query"],
            "positive": [ex["positive"]],
            "negative": [ex["hard_negative"]],
        }
        for ex in val_raw
    ]

    evaluator = CrossEncoderRerankingEvaluator(
        samples=val_samples,
        name="legal-val",
        at_k=10,
        show_progress_bar=True,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {BASE_MODEL}...")
    model = CrossEncoder(
        BASE_MODEL,
        num_labels=1,
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── Evaluate base model first ─────────────────────────────────────────────
    print("\nBase model performance (before fine-tuning):")
    base_score = evaluator(model)
    print(f"  MRR@10: {base_score:.4f}")

    # ── Training setup ────────────────────────────────────────────────────────
    output_dir = f"{VOLUME_PATH}/models/legal-reranker"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_dataset) // 16  # batch_size=16
    total_steps = steps_per_epoch * 3            # 3 epochs

    args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=100,
        fp16=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_legal-val_MRR@10",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    loss = ce_losses.BinaryCrossEntropyLoss(model)

    print(f"\nTraining config:")
    print(f"  Epochs: 3  |  Batch size: 16  |  LR: 2e-5")
    print(f"  Steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps}")
    print(f"  Eval every: 200 steps")
    print(f"  Output: {output_dir}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()

    # ── Final eval ────────────────────────────────────────────────────────────
    print("\nFinal evaluation (best checkpoint):")
    final_score = evaluator(model)
    print(f"  MRR@10: {final_score:.4f}  (base: {base_score:.4f}  Δ={final_score - base_score:+.4f})")

    # Save final model explicitly alongside HF checkpoints
    model.save(f"{output_dir}/final")

    meta = {
        "base_model": BASE_MODEL,
        "train_pairs": len(train_dataset),
        "val_triplets": len(val_raw),
        "epochs": 3,
        "batch_size": 16,
        "lr": 2e-5,
        "max_length": 512,
        "base_mrr10": base_score,
        "final_mrr10": final_score,
        "delta_mrr10": final_score - base_score,
    }
    with open(f"{output_dir}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    vol.commit()
    print(f"\nSaved to {output_dir}")
    print(f"  final/        ← use this for inference")
    print(f"  checkpoint-*  ← HF Trainer checkpoints")
    return meta


@app.function(image=modal.Image.debian_slim(), timeout=60)
def trigger():
    train.spawn()
    return "Reranker training spawned"


if __name__ == "__main__":
    import sys
    import modal as m

    if "--trigger" in sys.argv:
        fn = m.Function.from_name("legalkit-train-reranker", "trigger")
        print(fn.remote())
    else:
        print("Usage: python modal_train_reranker.py --trigger")
        print("       (deploy first with: modal deploy modal_train_reranker.py)")
