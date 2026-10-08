# LegalKit — Agent Guidelines

## Training Infrastructure

### EC2 (preferred for GRPO)

Infrastructure is managed by CDK in `./infra/`. Use the justfile for all operations:

```bash
just infra-deploy        # one-time: deploy S3, IAM, SG, launch template, key pair
just infra-fetch-key     # one-time: download SSH key from SSM to ~/.ssh/legalkit-training.pem
just train-launch        # launch g5.xlarge spot instance
just train-upload <ip>   # upload SFT adapter + prompts + training script
just train-ssh <ip>      # SSH into instance
just train-status        # check running instances
just train-stop          # terminate all training instances
just train-download      # download results from S3
just train-price         # check current spot pricing
```

**On the EC2 instance:**
```bash
bash setup.sh            # install deps, download base model, verify GPU

# Fire-and-forget (uploads to S3, then auto-terminates):
nohup bash -c 'source ~/legalkit-env/bin/activate && python ~/train_grpo_local.py \
  --sft-adapter ~/legal-reasoning-sft/legal-reasoning-sft \
  --prompts ~/grpo_prompts.jsonl \
  --output ~/legal-reasoning-grpo \
  --s3-upload s3://<bucket> \
  --shutdown-on-complete \
  --grad-accum 4' > ~/train.log 2>&1 &
```

**Instance auto-terminates on shutdown** (`instance-initiated-shutdown-behavior=terminate`).

**Known gotchas:**
- The SFT adapter has a nested structure: the real adapter is at `models/legal-reasoning-sft/legal-reasoning-sft/` (the inner dir from Modal download)
- Ubuntu 24.04 DL AMI needs `sudo apt install python3.12-venv` before venv creation
- TRL 0.29: `generation_batch_size` (batch_size × grad_accum) must be divisible by `num_generations` (group_size). Use `--grad-accum 4` with default `--group-size 4`

### Modal (for eval and short jobs)

Modal is good for eval runs (<2 hours) but expensive for full GRPO training ($1.50/hr vs EC2 spot ~$0.40/hr) and has a 6-hour timeout.

**Never run Modal training synchronously in the agent.** Use deploy+spawn:

```bash
cd python
uv run modal deploy ../scripts/<script>.py
uv run python ../scripts/<script>.py --trigger
```

Check results:
```bash
cd python && uv run modal volume ls legalkit-data eval/
cd python && uv run modal volume ls legalkit-data models/
```

### Justfile

All project automation is in the root `justfile`. Run `just help` to list recipes.

Key recipes:
- `just eval-sft` / `just eval-grpo` — deploy and run evals on Modal
- `just modal-eval-status` — check eval output files
- `just train-*` — EC2 training lifecycle
- `just infra-*` — CDK infrastructure management

## Key File Paths

### Training Scripts
| File | Purpose |
|---|---|
| `scripts/train_grpo_local.py` | GRPO training (EC2/local GPU) |
| `scripts/modal_grpo_train.py` | GRPO training (Modal) |
| `scripts/modal_sft_qwen.py` | SFT training (Modal) |
| `scripts/modal_eval_sft.py` | SFT evaluation (Modal) |
| `scripts/modal_eval_grpo.py` | GRPO evaluation (Modal) |
| `scripts/dspy_ceiling_test.py` | DSPy ceiling test (OpenRouter API) |
| `scripts/generate_more_sft_data.py` | Generate SFT chains (OpenRouter) |
| `scripts/build_grpo_prompts.py` | Build GRPO prompt pool |

### Data
| File | Purpose |
|---|---|
| `data/grpo_prompts/grpo_prompts.jsonl` | 969 GRPO training prompts |
| `data/sft_reasoning/train.jsonl` | SFT v1 training data (3,012 chains) |
| `data/sft_reasoning/train_v2.jsonl` | SFT v2 training data (improved neutral) |
| `data/eval/legal_nli_eval_expanded.json` | 243-case eval set |
| `data/prompts/optimized_legal_reasoning.md` | DSPy MIPROv2 optimized instructions |

### Models
| Path | Description |
|---|---|
| `models/legal-reasoning-sft/` | SFT LoRA adapter (Qwen2.5-3B) |
| `models/legal-reasoning-grpo/` | GRPO LoRA adapter (output) |
| `models/legal-entailment/` | Cross-encoder v2 (DeBERTa-large) |
| `models/legal-embeddings/` | BGE-base embeddings (ship base, fine-tuned adds nothing) |

### Research Docs
| File | Topic |
|---|---|
| `docs/research/building-legal-reasoning-model.md` | Blog-style writeup of full journey |
| `docs/research/sft-reasoning-eval-findings.md` | SFT v1/v2 eval results |
| `docs/research/lodo-embedding-eval-findings.md` | Embedding fine-tuning dead-end |
| `docs/research/label-audit-findings.md` | Training data quality audit |
| `docs/reference/papers.md` | Reading list behind the RL work (links) |

### Infrastructure
| Path | Purpose |
|---|---|
| `infra/` | CDK project (S3, IAM, SG, launch template, key pair) |
| `justfile` | Project automation recipes |
| `scripts/ec2_setup.sh` | Instance setup script (baked into launch template user-data) |
