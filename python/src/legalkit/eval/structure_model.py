"""Fine-tune a transformer for legal document structure detection.

Frames structure detection as sequence classification (one label per line).
Each line of the input document is classified into one of 15 BIO labels.
Uses legal-roberta-base as the backbone — pre-trained on legal domain text.

Usage:
    legalkit structure-train            # train with defaults
    legalkit structure-train --epochs 5 # more epochs
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
import numpy as np


TRAINING_DIR = Path("data/structure_model")
MODEL_OUTPUT_DIR = Path("models/structure-parser")

# Default to legal-domain pre-trained RoBERTa
BASE_MODEL = "lexlms/legal-roberta-base"

# BIO label schema for document structure
LABELS = [
    "O",  # outside/body text
    "B-ARTICLE_HEADER", "I-ARTICLE_HEADER",
    "B-SECTION_HEADER", "I-SECTION_HEADER",
    "B-SUBSECTION_HEADER", "I-SUBSECTION_HEADER",
    "B-PREAMBLE", "I-PREAMBLE",
    "B-RECITAL", "I-RECITAL",
    "B-EXHIBIT_HEADER", "I-EXHIBIT_HEADER",
    "B-DEFINITION", "I-DEFINITION",
]

NUM_LABELS = len(LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}


class StructureDataset(Dataset):
    """Line-level sequence classification dataset.
    
    Each example is a SINGLE LINE of text, tokenised and labelled with
    a single BIO label (the label of that line).
    
    We use line-level (not token-level) labels because the regex parser
    labels entire lines, not individual tokens within a line. The model
    classifies the entire line by predicting the label for the [CLS] token.
    
    Sequence classification (not token classification) is therefore appropriate
    here — the label is per-line, not per-token.
    """

    def __init__(self, path: Path, tokenizer, max_length: int = 128):
        """Load JSONL, flatten all lines from all documents.
        
        Each JSONL line: {doc_id, file_path, contract_type, lines: [{text, label, label_id, line_number}]}
        We flatten this to get one dataset example per line.
        """
        self.examples = []
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                for line_obj in doc.get("lines", []):
                    self.examples.append({
                        "text": line_obj["text"],
                        "label_id": line_obj["label_id"],
                        "doc_id": doc["doc_id"],
                    })
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        encoding = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(example["label_id"], dtype=torch.long),
        }


def compute_structure_metrics(pred: EvalPrediction) -> dict:
    """Per-label precision, recall, F1 + macro F1 + heading F1.
    
    Key metric: heading_f1 — aggregate F1 on the heading label types
    (ARTICLE_HEADER, SECTION_HEADER, SUBSECTION_HEADER) regardless of B/I.
    This is the primary eval metric.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = (preds == labels).mean()

    # Per-label metrics
    metrics = {"accuracy": float(accuracy)}
    
    for i, label_name in enumerate(LABELS):
        tp = ((preds == i) & (labels == i)).sum()
        fp = ((preds == i) & (labels != i)).sum()
        fn = ((preds != i) & (labels == i)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"{label_name}_precision"] = float(precision)
        metrics[f"{label_name}_recall"] = float(recall)
        metrics[f"{label_name}_f1"] = float(f1)

    # Macro F1 (average over all labels)
    f1s = [metrics[f"{label}_f1"] for label in LABELS]
    metrics["macro_f1"] = sum(f1s) / len(f1s)

    # Heading F1 (key metric: aggregate ARTICLE/SECTION/SUBSECTION headers)
    heading_labels = [
        "B-ARTICLE_HEADER", "I-ARTICLE_HEADER",
        "B-SECTION_HEADER", "I-SECTION_HEADER",
        "B-SUBSECTION_HEADER", "I-SUBSECTION_HEADER",
    ]
    
    # Compute aggregate TP/FP/FN for all heading types
    heading_label_ids = [LABEL2ID[label] for label in heading_labels]
    heading_tp = sum(((preds == i) & (labels == i)).sum() for i in heading_label_ids)
    heading_fp = sum(((preds == i) & (labels != i)).sum() for i in heading_label_ids)
    heading_fn = sum(((preds != i) & (labels == i)).sum() for i in heading_label_ids)
    
    heading_precision = heading_tp / (heading_tp + heading_fp) if (heading_tp + heading_fp) > 0 else 0
    heading_recall = heading_tp / (heading_tp + heading_fn) if (heading_tp + heading_fn) > 0 else 0
    heading_f1 = (
        2 * heading_precision * heading_recall / (heading_precision + heading_recall)
        if (heading_precision + heading_recall) > 0 else 0
    )
    
    metrics["heading_precision"] = float(heading_precision)
    metrics["heading_recall"] = float(heading_recall)
    metrics["heading_f1"] = float(heading_f1)

    return metrics


def run_structure_finetuning(
    base_model: str = BASE_MODEL,
    training_dir: Path | None = None,
    output_dir: Path | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> dict:
    """Fine-tune legal-roberta-base on structure detection.

    Args:
        base_model: HuggingFace model name (default: legal-roberta-base).
        training_dir: Directory with train.jsonl, val.jsonl, test.jsonl.
        output_dir: Where to save the fine-tuned model.
        epochs: Training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        max_length: Max token length (128 is usually enough for single lines).
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
                f"Missing {path}. Run 'legalkit structure-annotate' first."
            )

    # Load tokenizer and model
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
    )

    # Create datasets
    print("Loading datasets...")
    train_dataset = StructureDataset(training_dir / "train.jsonl", tokenizer, max_length)
    val_dataset = StructureDataset(training_dir / "val.jsonl", tokenizer, max_length)
    test_dataset = StructureDataset(training_dir / "test.jsonl", tokenizer, max_length)

    print(f"  Train: {len(train_dataset)} lines")
    print(f"  Val:   {len(val_dataset)} lines")
    print(f"  Test:  {len(test_dataset)} lines")

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
        metric_for_best_model="heading_f1",
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
        compute_metrics=compute_structure_metrics,
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
    print(f"Test heading F1: {test_results['eval_heading_f1']:.4f}")

    # Print key heading metrics
    for label in ["B-ARTICLE_HEADER", "B-SECTION_HEADER", "B-SUBSECTION_HEADER"]:
        f1 = test_results.get(f"eval_{label}_f1", 0)
        print(f"  {label}: F1={f1:.4f}")

    # Save model
    model_path = output_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save results
    results = {
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
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
