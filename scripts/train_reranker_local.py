"""Train a legal-specific cross-encoder reranker on EC2.

Fine-tunes BAAI/bge-reranker-v2-m3 on (query, positive, hard_negative) triplets
using the sentence-transformers v5 CrossEncoderTrainer API.

Usage (on EC2 after setup):
    source ~/legalkit-env/bin/activate
    python ~/train_reranker_local.py \
        --train  ~/embedding_training/train.jsonl \
        --val    ~/embedding_training/val.jsonl \
        --output ~/legal-reranker \
        --s3-upload s3://<bucket> \
        --shutdown-on-complete

Arguments:
    --train             Path to train.jsonl (triplets: query, positive, hard_negative)
    --val               Path to val.jsonl
    --output            Local output directory (default: ~/legal-reranker)
    --base-model        HF model ID (default: BAAI/bge-reranker-v2-m3)
    --epochs            Number of epochs (default: 3)
    --batch-size        Per-device train batch size (default: 16)
    --lr                Learning rate (default: 2e-5)
    --max-length        Max token length (default: 512)
    --eval-steps        Eval every N steps (default: 200)
    --s3-upload         S3 prefix to upload results (optional)
    --shutdown-on-complete  Shut down instance when done (for auto-terminate)
    --log-group         CloudWatch log group (default: /legalkit/training)
    --log-stream        CloudWatch log stream (default: reranker-{timestamp})
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ── CloudWatch log handler ──────────────────────────────────────────────────

class CloudWatchHandler(logging.Handler):
    """Streams log records to a CloudWatch Logs stream in real-time."""

    def __init__(self, log_group: str, log_stream: str, region: str = "us-east-1"):
        super().__init__()
        self.log_group = log_group
        self.log_stream = log_stream
        self.region = region
        self._sequence_token = None
        self._buffer = []
        self._last_flush = time.time()
        self._enabled = False

        try:
            import boto3
            self._client = boto3.client("logs", region_name=region)
            self._ensure_group_and_stream()
            self._enabled = True
        except Exception as e:
            print(f"[CloudWatch] Not available: {e}. Logging to stdout only.", flush=True)

    def _ensure_group_and_stream(self):
        try:
            self._client.create_log_group(logGroupName=self.log_group)
        except self._client.exceptions.ResourceAlreadyExistsException:
            pass
        try:
            self._client.create_log_stream(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
            )
        except self._client.exceptions.ResourceAlreadyExistsException:
            pass

    def emit(self, record):
        if not self._enabled:
            return
        msg = self.format(record)
        self._buffer.append({
            "timestamp": int(record.created * 1000),
            "message": msg,
        })
        # Flush every 5 seconds or every 20 messages
        now = time.time()
        if len(self._buffer) >= 20 or (now - self._last_flush) >= 5:
            self._flush()

    def _flush(self):
        if not self._buffer or not self._enabled:
            return
        events = sorted(self._buffer, key=lambda e: e["timestamp"])
        self._buffer = []
        self._last_flush = time.time()
        kwargs = dict(
            logGroupName=self.log_group,
            logStreamName=self.log_stream,
            logEvents=events,
        )
        if self._sequence_token:
            kwargs["sequenceToken"] = self._sequence_token
        try:
            resp = self._client.put_log_events(**kwargs)
            self._sequence_token = resp.get("nextSequenceToken")
        except Exception as e:
            print(f"[CloudWatch] flush error: {e}", flush=True)

    def close(self):
        self._flush()
        super().close()


def setup_logging(log_group: str, log_stream: str) -> logging.Logger:
    logger = logging.getLogger("reranker")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # CloudWatch
    cw = CloudWatchHandler(log_group, log_stream)
    cw.setFormatter(fmt)
    logger.addHandler(cw)

    return logger


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",    default="~/embedding_training/train.jsonl")
    p.add_argument("--val",      default="~/embedding_training/val.jsonl")
    p.add_argument("--output",   default="~/legal-reranker")
    p.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--epochs",   type=int,   default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr",       type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--s3-upload", default=None)
    p.add_argument("--shutdown-on-complete", action="store_true")
    p.add_argument("--log-group",  default="/legalkit/training")
    p.add_argument("--log-stream", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    train_path  = Path(os.path.expanduser(args.train))
    val_path    = Path(os.path.expanduser(args.val))
    output_dir  = Path(os.path.expanduser(args.output))
    timestamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_stream  = args.log_stream or f"reranker-{timestamp}"

    log = setup_logging(args.log_group, log_stream)

    log.info("=" * 70)
    log.info("Legal Reranker — Cross-Encoder Fine-Tuning")
    log.info("=" * 70)
    log.info(f"  Base model:  {args.base_model}")
    log.info(f"  Train data:  {train_path}")
    log.info(f"  Val data:    {val_path}")
    log.info(f"  Output:      {output_dir}")
    log.info(f"  Epochs:      {args.epochs}")
    log.info(f"  Batch size:  {args.batch_size}")
    log.info(f"  LR:          {args.lr}")
    log.info(f"  Max length:  {args.max_length}")
    log.info(f"  CloudWatch:  {args.log_group}/{log_stream}")

    random.seed(42)
    np.random.seed(42)

    # ── Imports (after logging is set up so any errors are captured) ──────────
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder import losses as ce_losses
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading training data...")
    with open(train_path) as f:
        train_raw = [json.loads(l) for l in f]
    with open(val_path) as f:
        val_raw = [json.loads(l) for l in f]

    log.info(f"  Train triplets: {len(train_raw)}")
    log.info(f"  Val triplets:   {len(val_raw)}")

    # ── Build labeled-pair dataset ────────────────────────────────────────────
    queries, responses, labels = [], [], []
    for ex in train_raw:
        queries.append(ex["query"]);        responses.append(ex["positive"]);      labels.append(1)
        queries.append(ex["query"]);        responses.append(ex["hard_negative"]);  labels.append(0)

    combined = list(zip(queries, responses, labels))
    random.shuffle(combined)
    queries, responses, labels = zip(*combined)

    train_dataset = Dataset.from_dict({
        "query":    list(queries),
        "response": list(responses),
        "label":    list(labels),
    })
    log.info(f"  Total training pairs: {len(train_dataset)} ({len(train_raw)} pos + {len(train_raw)} neg)")

    # ── Build evaluator ───────────────────────────────────────────────────────
    val_samples = [
        {"query": ex["query"], "positive": [ex["positive"]], "negative": [ex["hard_negative"]]}
        for ex in val_raw
    ]
    evaluator = CrossEncoderRerankingEvaluator(
        samples=val_samples,
        name="legal-val",
        at_k=10,
        show_progress_bar=True,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        log.warning("No GPU detected — training will be very slow!")

    log.info(f"Loading {args.base_model}...")
    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        max_length=args.max_length,
        device=device,
    )

    # ── Baseline eval ─────────────────────────────────────────────────────────
    # v5 evaluator returns a dict of prefixed metrics, e.g. {"legal-val_MRR@10": 0.72, ...}
    log.info("Evaluating base model (before fine-tuning)...")
    base_metrics = evaluator(model)
    mrr_key = f"{evaluator.name}_mrr@{evaluator.at_k}"
    base_score = base_metrics[mrr_key]
    log.info(f"  Base MRR@10: {base_score:.4f}")

    # ── Training args ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = len(train_dataset) // args.batch_size
    total_steps     = steps_per_epoch * args.epochs

    log.info(f"Training config:")
    log.info(f"  Steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps}")
    log.info(f"  Eval every {args.eval_steps} steps")

    train_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_legal-val_mrr@10",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    loss = ce_losses.BinaryCrossEntropyLoss(model)

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training...")
    t0 = time.time()

    trainer = CrossEncoderTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    elapsed = time.time() - t0
    log.info(f"Training complete in {elapsed/3600:.2f}h")

    # ── Final eval ────────────────────────────────────────────────────────────
    log.info("Final evaluation (best checkpoint)...")
    final_metrics = evaluator(model)
    final_score = final_metrics[mrr_key]
    log.info(f"  Final MRR@10:  {final_score:.4f}")
    log.info(f"  Base  MRR@10:  {base_score:.4f}")
    log.info(f"  Delta MRR@10:  {final_score - base_score:+.4f}")

    # Save final model
    final_dir = output_dir / "final"
    model.save(str(final_dir))
    log.info(f"Model saved to {final_dir}")

    # Save metadata
    meta = {
        "base_model":   args.base_model,
        "train_pairs":  len(train_dataset),
        "val_triplets": len(val_raw),
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "max_length":   args.max_length,
        "runtime_hours": elapsed / 3600,
        "base_mrr10":   base_score,
        "final_mrr10":  final_score,
        "delta_mrr10":  final_score - base_score,
        "cloudwatch_log_group":  args.log_group,
        "cloudwatch_log_stream": log_stream,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata: {json.dumps(meta, indent=2)}")

    # ── Upload to S3 ──────────────────────────────────────────────────────────
    if args.s3_upload:
        s3_dest = args.s3_upload.rstrip("/") + "/legal-reranker/"
        log.info(f"Uploading to {s3_dest} ...")
        result = subprocess.run(
            ["aws", "s3", "sync", str(output_dir), s3_dest, "--region", "us-east-1"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log.info("Upload complete.")
        else:
            log.error(f"Upload failed: {result.stderr}")

    log.info("Done.")

    # Flush CloudWatch before shutdown
    for h in log.handlers:
        h.close()

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if args.shutdown_on_complete:
        log.info("Shutting down instance...")
        time.sleep(5)
        subprocess.run(["sudo", "shutdown", "-h", "now"])


if __name__ == "__main__":
    main()
