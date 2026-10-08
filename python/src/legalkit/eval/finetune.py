"""Fine-tune a DeBERTa model on legal entailment data.

Trains a 4-class NLI classifier:
  0: entailed
  1: partially_entailed
  2: contradicted
  3: neutral

Usage:
  legalkit finetune --epochs 3 --batch-size 16
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
import numpy as np


TRAINING_DIR = Path("data/training")
MODEL_OUTPUT_DIR = Path("models/legal-entailment")

# Default to NLI-pretrained DeBERTa large (best starting point for legal entailment)
BASE_MODEL = "cross-encoder/nli-deberta-v3-large"
NUM_LABELS = 4
LABEL_NAMES = ["entailed", "partially_entailed", "contradicted", "neutral"]


class EntailmentDataset(Dataset):
    """PyTorch dataset for legal entailment pairs."""

    def __init__(self, path: Path, tokenizer, max_length: int = 512):
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["premise"],
            sample["hypothesis"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }


def _load_model_for_finetuning(model_name: str, num_labels: int = 4):
    """Load model, handling NLI checkpoints with mismatched label count.
    
    For cross-encoder NLI models: load the transformer backbone and
    attach a fresh classification head for num_labels classes.
    Uses AutoModelForSequenceClassification with ignore_mismatched_sizes=True.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # handles 3-label → 4-label head replacement
    )
    return model


def compute_metrics(pred: EvalPrediction) -> dict:
    """Compute accuracy and per-class F1 for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = (preds == labels).mean()

    # Per-class metrics
    metrics = {"accuracy": float(accuracy)}
    for i, name in enumerate(LABEL_NAMES):
        tp = ((preds == i) & (labels == i)).sum()
        fp = ((preds == i) & (labels != i)).sum()
        fn = ((preds != i) & (labels == i)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"{name}_precision"] = float(precision)
        metrics[f"{name}_recall"] = float(recall)
        metrics[f"{name}_f1"] = float(f1)

    # Macro F1
    f1s = [metrics[f"{name}_f1"] for name in LABEL_NAMES]
    metrics["macro_f1"] = sum(f1s) / len(f1s)

    return metrics


def run_finetuning(
    base_model: str = BASE_MODEL,
    training_dir: Path | None = None,
    output_dir: Path | None = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> dict:
    """Fine-tune DeBERTa on legal entailment data.

    Args:
        base_model: HuggingFace model name.
        training_dir: Directory with train.jsonl, val.jsonl, test.jsonl.
        output_dir: Where to save the fine-tuned model.
        epochs: Training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        max_length: Max token length.
        warmup_ratio: Warmup fraction of training.
        weight_decay: Weight decay.

    Returns:
        Dict with training results and test metrics.
    """
    training_dir = training_dir or TRAINING_DIR
    output_dir = output_dir or MODEL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate data exists
    for split in ["train", "val", "test"]:
        path = training_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run 'legalkit prepare-training' first."
            )

    # Load tokenizer and model
    print(f"Loading base model: {base_model}")
    is_nli_checkpoint = "cross-encoder" in base_model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = _load_model_for_finetuning(base_model, num_labels=NUM_LABELS)

    # Create datasets
    print("Loading datasets...")
    train_dataset = EntailmentDataset(training_dir / "train.jsonl", tokenizer, max_length)
    val_dataset = EntailmentDataset(training_dir / "val.jsonl", tokenizer, max_length)
    test_dataset = EntailmentDataset(training_dir / "test.jsonl", tokenizer, max_length)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    print(f"Training complete. Loss: {train_result.training_loss:.4f} (took {training_time:.1f}s)")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test macro F1: {test_results['eval_macro_f1']:.4f}")

    for name in LABEL_NAMES:
        f1 = test_results.get(f"eval_{name}_f1", 0)
        print(f"  {name}: F1={f1:.4f}")

    # Save model
    model_path = output_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save results
    results = {
        "base_model": base_model,
        "is_nli_checkpoint": is_nli_checkpoint,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "training_time_seconds": training_time,
        "timestamp": datetime.now().isoformat(),
        "training_loss": train_result.training_loss,
        "test_metrics": {
            k.replace("eval_", ""): v
            for k, v in test_results.items()
            if k.startswith("eval_")
        },
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))

    return results
