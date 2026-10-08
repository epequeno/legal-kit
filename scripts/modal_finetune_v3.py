"""Fine-tune legal entailment model v3: 3-class with class weights.

Two-step workflow — upload data locally, then trigger training that runs
independently on Modal (survives laptop sleep/disconnect):

  Step 1 — Upload data and deploy (fast, local):
    modal run scripts/modal_finetune_v3.py

  Step 2 — Trigger training (returns immediately, runs on Modal):
    python scripts/modal_finetune_v3.py --trigger

  Check results:
    modal volume ls legalkit-data models/legal-entailment-v3/
    modal volume get legalkit-data models/legal-entailment-v3/results.json eval/finetune-v3-results.json

  Download model:
    modal volume get legalkit-data models/legal-entailment-v3/model models/legal-entailment-v3/model
"""

import modal
from datetime import datetime

app = modal.App("legalkit-finetune-v3")

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-large"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.45",
        "accelerate>=1.1",
        "sentencepiece",
        "protobuf>=4.25,<5",
        "numpy>=1.26",
    )
)

vol = modal.Volume.from_name("legalkit-data", create_if_missing=True)


# ---------------------------------------------------------------------------
# Step 1: Upload (local entrypoint — runs on your machine, fast)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(data_dir: str = "data/training_v3") -> None:
    """Upload v3 training data to Modal volume, then deploy the app.

    After this completes, trigger training with:
        python scripts/modal_finetune_v3.py --trigger
    """
    from pathlib import Path

    data_path = Path(data_dir)
    needed = ["train.jsonl", "val.jsonl", "test.jsonl", "metadata.json"]

    for f in needed:
        if not (data_path / f).exists():
            print(f"ERROR: {data_path / f} not found.")
            print(f"Run: python scripts/prepare_entailment_v3.py --audit-corrections")
            return

    print(f"Uploading v3 training data from {data_dir}...")
    with vol.batch_upload(force=True) as batch:
        for f in needed:
            local_path = data_path / f
            batch.put_file(str(local_path), f"training_v3/{f}")
            print(f"  ✓ {f} ({local_path.stat().st_size / 1024:.1f} KB)")

    print(f"\n✓ Data uploaded.")
    print(f"\nNow deploy and trigger training:")
    print(f"  modal deploy scripts/modal_finetune_v3.py")
    print(f"  python scripts/modal_finetune_v3.py --trigger")


def _trigger():
    """Spawn training on a deployed app. Returns immediately — laptop can sleep."""
    import modal

    train_fn = modal.Function.from_name("legalkit-finetune-v3", "train")
    call = train_fn.spawn()
    print(f"✓ Training spawned (call ID: {call.object_id})")
    print(f"  Training runs independently on Modal — laptop can sleep/disconnect.")
    print(f"\nCheck progress:")
    print(f"  modal volume ls legalkit-data models/legal-entailment-v3/")
    print(f"\nWhen results.json appears, training is done. Download with:")
    print(f"  modal volume get legalkit-data models/legal-entailment-v3/results.json eval/finetune-v3-results.json")
    print(f"  modal volume get legalkit-data models/legal-entailment-v3/model models/legal-entailment-v3/model")


if __name__ == "__main__":
    import sys
    if "--trigger" in sys.argv:
        _trigger()
    else:
        print("Usage:")
        print("  modal run scripts/modal_finetune_v3.py   # Upload data")
        print("  modal deploy scripts/modal_finetune_v3.py  # Deploy app")
        print("  python scripts/modal_finetune_v3.py --trigger  # Start training (laptop-safe)")
        print()
        print("Or upload + deploy in one go:")
        print("  modal run scripts/modal_finetune_v3.py && modal deploy scripts/modal_finetune_v3.py && python scripts/modal_finetune_v3.py --trigger")


# ---------------------------------------------------------------------------
# Step 2: Train (remote function — fully detachable)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours — needed for 14K+ samples
    volumes={"/vol": vol},
)
def train(
    base_model: str = DEFAULT_MODEL,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    max_length: int = 512,
):
    """Fine-tune DeBERTa on 3-class legal entailment data with class weights.

    Reads training data and metadata from the Modal volume (uploaded via the
    `upload` entrypoint). Saves the trained model + results back to the volume.
    """
    import json
    from pathlib import Path
    import time
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EvalPrediction,
    )
    import numpy as np

    NUM_LABELS = 3
    LABEL_NAMES = ["entailed", "contradicted", "neutral"]

    class EntailmentDataset(Dataset):
        def __init__(self, path, tokenizer, max_length=512):
            self.samples = [json.loads(l) for l in open(path)]
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            enc = self.tokenizer(
                s["premise"], s["hypothesis"],
                truncation=True, max_length=self.max_length,
                padding="max_length", return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(s["label"], dtype=torch.long),
            }

    class WeightedTrainer(Trainer):
        """Trainer with per-class loss weighting."""
        def __init__(self, class_weights=None, **kwargs):
            super().__init__(**kwargs)
            if class_weights is not None:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            else:
                self.class_weights = None

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            if self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
                loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fn = torch.nn.CrossEntropyLoss()

            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(pred: EvalPrediction):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        metrics = {"accuracy": float((preds == labels).mean())}

        for i, name in enumerate(LABEL_NAMES):
            tp = ((preds == i) & (labels == i)).sum()
            fp = ((preds == i) & (labels != i)).sum()
            fn = ((preds != i) & (labels == i)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            metrics[f"{name}_f1"] = float(f1)
        metrics["macro_f1"] = sum(metrics[f"{n}_f1"] for n in LABEL_NAMES) / NUM_LABELS

        # Binary: entailed (supported) vs contradicted+neutral (not supported)
        binary_true = (labels == 0).astype(int)
        binary_pred = (preds == 0).astype(int)
        metrics["binary_accuracy"] = float((binary_true == binary_pred).mean())

        tp = ((binary_pred == 1) & (binary_true == 1)).sum()
        fp = ((binary_pred == 1) & (binary_true == 0)).sum()
        fn = ((binary_pred == 0) & (binary_true == 1)).sum()
        metrics["supported_precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        metrics["supported_recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return metrics

    # --- Setup ---
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    print(f"Training v3: 3-class entailment with class weights")

    data_dir = Path("/vol/training_v3")
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(
            "v3 training data not found on volume. "
            "Run first: modal run scripts/modal_finetune_v3.py::upload"
        )

    # Load class weights from metadata
    metadata_path = data_dir / "metadata.json"
    class_weights = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        weights_dict = metadata.get("class_weights", {})
        if weights_dict:
            class_weights = [
                weights_dict.get("entailed", 1.0),
                weights_dict.get("contradicted", 1.0),
                weights_dict.get("neutral", 1.0),
            ]
            print(f"Class weights: {dict(zip(LABEL_NAMES, class_weights))}")

    # --- Load model & data ---
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )

    train_ds = EntailmentDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds = EntailmentDataset(data_dir / "val.jsonl", tokenizer, max_length)
    test_ds = EntailmentDataset(data_dir / "test.jsonl", tokenizer, max_length)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    out_dir = Path("/vol/models/legal-entailment-v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    grad_accum = max(1, batch_size // 8)
    micro_batch = min(8, batch_size)

    # --- Train ---
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="binary_accuracy",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        save_total_limit=2,
        bf16=True,
        max_grad_norm=1.0,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining {epochs} epochs...")
    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time

    # --- Evaluate ---
    test_results = trainer.evaluate(test_ds)
    print(f"\nTest results:")
    print(f"  3-class accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"  Binary accuracy:   {test_results['eval_binary_accuracy']:.4f}")
    print(f"  Macro F1:          {test_results['eval_macro_f1']:.4f}")
    print(f"  Supported recall:  {test_results['eval_supported_recall']:.4f}")
    print(f"  Supported prec:    {test_results['eval_supported_precision']:.4f}")
    for name in LABEL_NAMES:
        print(f"  {name}: F1={test_results.get(f'eval_{name}_f1', 0):.4f}")

    # --- Save ---
    model_path = out_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    results_dict = {
        "version": "v3",
        "num_classes": NUM_LABELS,
        "class_names": LABEL_NAMES,
        "class_weights": class_weights,
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "training_time_seconds": training_time,
        "timestamp": datetime.now().isoformat(),
        "test_metrics": {k.replace("eval_", ""): v for k, v in test_results.items()
                         if k.startswith("eval_")},
    }
    (out_dir / "results.json").write_text(json.dumps(results_dict, indent=2))
    vol.commit()

    print(f"\n{'='*60}")
    print(f"✓ Training complete in {training_time:.0f}s")
    print(f"  Model saved to volume: models/legal-entailment-v3/model")
    print(f"  Results saved to volume: models/legal-entailment-v3/results.json")
    print(f"\nDownload:")
    print(f"  modal volume get legalkit-data models/legal-entailment-v3/model models/legal-entailment-v3/model")
    print(f"  modal volume get legalkit-data models/legal-entailment-v3/results.json eval/finetune-v3-results.json")

    return results_dict
