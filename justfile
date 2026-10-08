# LegalKit — project automation
# Usage: just <recipe>

set dotenv-load := false

aws_profile := "arch"
region := "us-east-1"
launch_template := "legalkit-grpo-training"
s3_bucket := env_var_or_default("LEGALKIT_BUCKET", "legalkit-training")
key_file := "~/.ssh/legalkit-training.pem"
tag := "legalkit-grpo"

# ─── App Dev & Build ──────────────────────────────────────────

# Start the app in development mode (builds Elm first, then launches Tauri)
dev:
    cd tauri-app && npm run dev

# Build Elm frontend only (fast check during UI work)
elm:
    cd tauri-app && npm run elm:debug

# Type-check Elm without producing output
elm-check:
    cd tauri-app && npm run check

# Build the full release app (.app + .dmg) — unsigned
build:
    cd tauri-app && npm run build

# Build release binary only, skip bundling (faster iteration)
build-fast:
    cd tauri-app && npm run build:no-bundle

# Open the built .dmg (after `just build`)
open-dmg:
    #!/usr/bin/env bash
    DMG=$(ls tauri-app/src-tauri/target/release/bundle/dmg/*.dmg 2>/dev/null | head -1)
    if [ -z "$DMG" ]; then echo "No .dmg found — run 'just build' first"; exit 1; fi
    echo "Opening: $DMG"
    open "$DMG"

# Show build output location
build-info:
    #!/usr/bin/env bash
    echo "=== Build artifacts ==="
    APP=$(ls -d tauri-app/src-tauri/target/release/bundle/macos/*.app 2>/dev/null | head -1)
    DMG=$(ls tauri-app/src-tauri/target/release/bundle/dmg/*.dmg 2>/dev/null | head -1)
    BIN=tauri-app/src-tauri/target/release/legalkit
    [ -n "$APP" ] && echo "  .app  $(du -sh "$APP" | cut -f1)   $APP" || echo "  .app  not built"
    [ -n "$DMG" ] && echo "  .dmg  $(du -sh "$DMG" | cut -f1)   $DMG" || echo "  .dmg  not built"
    [ -f "$BIN" ] && echo "  bin   $(du -sh "$BIN" | cut -f1)   $BIN" || echo "  bin   not built"

# ─── Infrastructure ──────────────────────────────────────────

# Deploy training infrastructure (S3, IAM, SG, launch template)
infra-deploy:
    cd infra && source .venv/bin/activate && AWS_PROFILE={{aws_profile}} npx cdk deploy --require-approval never

# Preview infrastructure changes
infra-diff:
    cd infra && source .venv/bin/activate && AWS_PROFILE={{aws_profile}} npx cdk diff

# Destroy training infrastructure
infra-destroy:
    cd infra && source .venv/bin/activate && AWS_PROFILE={{aws_profile}} npx cdk destroy

# Fetch the EC2 private key from SSM (run once after infra-deploy)
infra-fetch-key:
    #!/usr/bin/env bash
    set -euo pipefail
    export AWS_PROFILE={{aws_profile}}
    KEY=$(eval echo {{key_file}})
    KEY_ID=$(aws ec2 describe-key-pairs \
        --region {{region}} \
        --key-names legalkit-training \
        --query 'KeyPairs[0].KeyPairId' \
        --output text)
    aws ssm get-parameter \
        --region {{region}} \
        --name "/ec2/keypair/$KEY_ID" \
        --with-decryption \
        --query 'Parameter.Value' \
        --output text > "$KEY"
    chmod 400 "$KEY"
    echo "Saved private key to $KEY"

# ─── EC2 Training ────────────────────────────────────────────

# Launch a g5.xlarge on-demand instance for GRPO training
train-launch:
    #!/usr/bin/env bash
    set -euo pipefail
    export AWS_PROFILE={{aws_profile}}

    echo "Launching g5.xlarge on-demand instance..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --region {{region}} \
        --launch-template LaunchTemplateName={{launch_template}} \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value={{tag}}},{Key=Project,Value={{tag}}}]' \
        --query 'Instances[0].InstanceId' \
        --output text)

    echo "Instance: $INSTANCE_ID"
    echo "Waiting for public IP..."

    for i in $(seq 1 24); do
        IP=$(aws ec2 describe-instances \
            --region {{region}} \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text 2>/dev/null)
        [ "$IP" != "None" ] && [ -n "$IP" ] && break
        sleep 5
    done

    echo ""
    echo "========================================="
    echo "  Instance: $INSTANCE_ID"
    echo "  IP:       $IP"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  just train-upload $IP"
    echo "  just train-ssh $IP"
    echo "  # on the instance: bash setup.sh"
    echo "  # then: just train-run (fire-and-forget)"

# Launch a g5.xlarge instance for eval (same as training, cheaper than Modal)
eval-launch:
    #!/usr/bin/env bash
    set -euo pipefail
    export AWS_PROFILE={{aws_profile}}

    echo "Launching g5.xlarge for eval..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --region {{region}} \
        --launch-template LaunchTemplateName={{launch_template}} \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value={{tag}}},{Key=Project,Value={{tag}}}]' \
        --query 'Instances[0].InstanceId' \
        --output text)

    echo "Instance: $INSTANCE_ID"
    echo "Waiting for public IP..."

    for i in $(seq 1 24); do
        IP=$(aws ec2 describe-instances \
            --region {{region}} \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text 2>/dev/null)
        [ "$IP" != "None" ] && [ -n "$IP" ] && break
        sleep 5
    done

    echo ""
    echo "========================================="
    echo "  Instance: $INSTANCE_ID (g5.xlarge)"
    echo "  IP:       $IP"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  just eval-upload <ip> <adapter-dir>"
    echo "  just eval-run-7b $IP   (or eval-run-3b)"

# Upload eval dependencies to an instance
eval-upload ip adapter_dir:
    #!/usr/bin/env bash
    set -euo pipefail
    KEY=$(eval echo {{key_file}})
    SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no"
    echo "Uploading eval script + data to {{ip}}..."
    scp $SSH_OPTS scripts/eval_local.py ubuntu@{{ip}}:~/
    scp $SSH_OPTS data/eval/legal_nli_eval_expanded.json ubuntu@{{ip}}:~/
    echo "Uploading adapter from {{adapter_dir}}..."
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
        {{adapter_dir}}/ ubuntu@{{ip}}:~/adapter/
    echo "Done."

# Run 7B GRPO eval on EC2 (fire-and-forget, uploads to S3, shuts down)
eval-run-7b ip:
    #!/usr/bin/env bash
    KEY=$(eval echo {{key_file}})
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@{{ip}} "
        source ~/legalkit-env/bin/activate
        nohup python -u ~/eval_local.py \
            --model Qwen/Qwen2.5-7B \
            --adapter ~/adapter \
            --eval-data ~/legal_nli_eval_expanded.json \
            --output ~/eval_results.json \
            --s3-upload s3://{{s3_bucket}}/eval/grpo_7b_eval_results.json \
        > ~/eval.log 2>&1 &
        echo \"Eval started PID \$!\"
        echo \"Monitor: tail -f ~/eval.log\"
    "

# Run 3B GRPO eval on EC2 (fire-and-forget, uploads to S3, shuts down)
eval-run-3b ip:
    #!/usr/bin/env bash
    KEY=$(eval echo {{key_file}})
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@{{ip}} "
        source ~/legalkit-env/bin/activate
        nohup python -u ~/eval_local.py \
            --model Qwen/Qwen2.5-3B \
            --adapter ~/adapter \
            --eval-data ~/legal_nli_eval_expanded.json \
            --output ~/eval_results.json \
            --s3-upload s3://{{s3_bucket}}/eval/grpo_3b_eval_results.json \
        > ~/eval.log 2>&1 &
        echo \"Eval started PID \$!\"
        echo \"Monitor: tail -f ~/eval.log\"
    "

# SSH into eval instance
eval-ssh ip:
    ssh -i $(eval echo {{key_file}}) -o StrictHostKeyChecking=no ubuntu@{{ip}}

# ─── Reranker Training (EC2) ─────────────────────────────────────────────────

# Launch a g5.xlarge spot instance for reranker training
reranker-launch:
    #!/usr/bin/env bash
    set -euo pipefail
    export AWS_PROFILE={{aws_profile}}
    echo "Launching g5.xlarge on-demand instance for reranker training..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --region {{region}} \
        --launch-template LaunchTemplateName={{launch_template}} \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=legalkit-grpo},{Key=Job,Value=reranker}]' \
        --query 'Instances[0].InstanceId' \
        --output text)
    echo "Instance: $INSTANCE_ID"
    echo "Waiting for public IP..."
    sleep 15
    IP=$(aws ec2 describe-instances \
        --region {{region}} \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    echo "IP: $IP"
    echo ""
    echo "Next steps:"
    echo "  just reranker-upload $IP"
    echo "  just train-ssh $IP"
    echo "  bash ~/setup_reranker.sh"
    echo "  just reranker-run  (run on the instance)"

# Upload reranker training files to EC2
reranker-upload ip:
    #!/usr/bin/env bash
    set -euo pipefail
    KEY=$(eval echo {{key_file}})
    SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no"
    echo "Uploading to {{ip}}..."
    # Training script and setup
    scp $SSH_OPTS scripts/train_reranker_local.py ubuntu@{{ip}}:~/train_reranker_local.py
    scp $SSH_OPTS scripts/ec2_setup_reranker.sh   ubuntu@{{ip}}:~/setup_reranker.sh
    # Training data
    ssh $SSH_OPTS ubuntu@{{ip}} "mkdir -p ~/embedding_training"
    scp $SSH_OPTS data/embedding_training/train.jsonl ubuntu@{{ip}}:~/embedding_training/train.jsonl
    scp $SSH_OPTS data/embedding_training/val.jsonl   ubuntu@{{ip}}:~/embedding_training/val.jsonl
    echo "Upload complete."
    echo "  SSH in:  just train-ssh {{ip}}"
    echo "  Setup:   bash ~/setup_reranker.sh"

# Fire-and-forget: train reranker, upload to S3, auto-shutdown
# Run ON the EC2 instance after setup_reranker.sh
reranker-run:
    #!/usr/bin/env bash
    source ~/legalkit-env/bin/activate
    nohup python ~/train_reranker_local.py \
        --train  ~/embedding_training/train.jsonl \
        --val    ~/embedding_training/val.jsonl \
        --output ~/legal-reranker \
        --s3-upload s3://{{s3_bucket}} \
        --shutdown-on-complete \
        > ~/train.log 2>&1 &
    echo "Training PID: $!"
    echo "Logs: tail -f ~/train.log"
    echo "CloudWatch: just reranker-logs"

# Stream reranker training logs from CloudWatch (real-time)
reranker-logs:
    #!/usr/bin/env bash
    export AWS_PROFILE={{aws_profile}}
    LOG_GROUP="/legalkit/training"
    echo "Fetching latest reranker log stream..."
    STREAM=$(aws logs describe-log-streams \
        --region {{region}} \
        --log-group-name "$LOG_GROUP" \
        --order-by LastEventTime \
        --descending \
        --query 'logStreams[?starts_with(logStreamName, `reranker-`)].logStreamName | [0]' \
        --output text 2>/dev/null || echo "None")
    if [ "$STREAM" = "None" ] || [ -z "$STREAM" ]; then
        echo "No reranker log streams found yet in $LOG_GROUP"
        echo "Tip: training must start before logs appear (usually ~2-3 min after setup)"
        exit 0
    fi
    echo "Tailing: $LOG_GROUP/$STREAM"
    echo "---"
    aws logs tail "$LOG_GROUP" \
        --region {{region}} \
        --log-stream-names "$STREAM" \
        --follow

# Download trained reranker from S3
reranker-download:
    AWS_PROFILE={{aws_profile}} aws s3 sync \
        s3://{{s3_bucket}}/legal-reranker/ \
        models/legal-reranker/ \
        --region {{region}}
    @echo "Downloaded to models/legal-reranker/"
    @cat models/legal-reranker/training_meta.json 2>/dev/null || true

# Download eval results from S3
eval-download:
    AWS_PROFILE={{aws_profile}} aws s3 sync \
        s3://{{s3_bucket}}/eval/ \
        data/eval/results/

# Launch a g5.2xlarge instance for 7B GRPO training (same A10G GPU, more system RAM)
train-launch-7b:
    #!/usr/bin/env bash
    set -euo pipefail
    export AWS_PROFILE={{aws_profile}}

    echo "Launching g5.2xlarge on-demand instance for 7B training..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --region {{region}} \
        --launch-template LaunchTemplateName={{launch_template}} \
        --instance-type g5.2xlarge \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value={{tag}}},{Key=Project,Value={{tag}}}]' \
        --query 'Instances[0].InstanceId' \
        --output text)

    echo "Instance: $INSTANCE_ID"
    echo "Waiting for public IP..."

    for i in $(seq 1 24); do
        IP=$(aws ec2 describe-instances \
            --region {{region}} \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text 2>/dev/null)
        [ "$IP" != "None" ] && [ -n "$IP" ] && break
        sleep 5
    done

    echo ""
    echo "========================================="
    echo "  Instance: $INSTANCE_ID (g5.2xlarge)"
    echo "  IP:       $IP"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  just train-upload $IP"
    echo "  just train-ssh $IP"
    echo "  # on the instance: bash setup.sh"
    echo "  # then: just train-run-7b (fire-and-forget)"

# Upload training data to an instance
train-upload ip:
    #!/usr/bin/env bash
    set -euo pipefail
    KEY=$(eval echo {{key_file}})
    SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no"
    echo "Uploading to {{ip}}..."
    scp $SSH_OPTS -r models/legal-reasoning-sft/ ubuntu@{{ip}}:~/
    if [ -d models/legal-reasoning-sft-7b ]; then
        echo "Uploading 7B SFT adapter..."
        scp $SSH_OPTS -r models/legal-reasoning-sft-7b/ ubuntu@{{ip}}:~/
    fi
    scp $SSH_OPTS data/grpo_prompts/grpo_prompts.jsonl ubuntu@{{ip}}:~/
    scp $SSH_OPTS scripts/train_grpo_local.py ubuntu@{{ip}}:~/
    echo "Done."

# SSH into a training instance
train-ssh ip:
    ssh -i $(eval echo {{key_file}}) -o StrictHostKeyChecking=no ubuntu@{{ip}}

# Fire-and-forget: start training, upload to S3, shutdown on completion
# Run this ON the EC2 instance (after setup.sh)
train-run:
    #!/usr/bin/env bash
    source ~/legalkit-env/bin/activate
    nohup python ~/train_grpo_local.py \
        --sft-adapter ~/legal-reasoning-sft/legal-reasoning-sft \
        --prompts ~/grpo_prompts.jsonl \
        --output ~/legal-reasoning-grpo-v2 \
        --s3-upload s3://{{s3_bucket}} \
        --shutdown-on-complete \
        > ~/train.log 2>&1 &
    echo "Training started (PID $!). Instance will auto-terminate when done."
    echo "Monitor: tail -f ~/train.log"

# Fire-and-forget: GRPO training with Qwen2.5-7B
# Run this ON the EC2 instance (after setup.sh)
train-run-7b:
    #!/usr/bin/env bash
    source ~/legalkit-env/bin/activate
    SFT_FLAG=""
    if [ -d ~/legal-reasoning-sft-7b ]; then
        SFT_FLAG="--sft-adapter ~/legal-reasoning-sft-7b"
        echo "Using 7B SFT adapter"
    else
        echo "No 7B SFT adapter found, training from base model with fresh LoRA"
    fi
    nohup python ~/train_grpo_local.py \
        --model Qwen/Qwen2.5-7B \
        $SFT_FLAG \
        --prompts ~/grpo_prompts.jsonl \
        --output ~/legal-reasoning-grpo-7b \
        --s3-upload s3://{{s3_bucket}} \
        --shutdown-on-complete \
        --grad-accum 4 \
        > ~/train.log 2>&1 &
    echo "Training started — 7B GRPO (PID $!). Instance will auto-terminate when done."
    echo "Monitor: tail -f ~/train.log"

# Check status of running training instances
train-status:
    #!/usr/bin/env bash
    export AWS_PROFILE={{aws_profile}}
    aws ec2 describe-instances \
        --region {{region}} \
        --filters "Name=tag:Project,Values={{tag}}" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[].Instances[].{ID:InstanceId,State:State.Name,IP:PublicIpAddress,Launch:LaunchTime,Type:InstanceType}' \
        --output table

# Terminate all training instances
train-stop:
    #!/usr/bin/env bash
    set -euo pipefail
    export AWS_PROFILE={{aws_profile}}
    INSTANCES=$(aws ec2 describe-instances \
        --region {{region}} \
        --filters "Name=tag:Project,Values={{tag}}" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[].Instances[].InstanceId' \
        --output text)
    if [ -z "$INSTANCES" ]; then
        echo "No running instances."
    else
        echo "Terminating: $INSTANCES"
        aws ec2 terminate-instances --region {{region}} --instance-ids $INSTANCES --output text
    fi

# Download training results from S3
train-download:
    AWS_PROFILE={{aws_profile}} aws s3 sync s3://{{s3_bucket}}/legal-reasoning-grpo/ models/legal-reasoning-grpo/

# Download 7B training results from S3
train-download-7b:
    AWS_PROFILE={{aws_profile}} aws s3 sync s3://{{s3_bucket}}/legal-reasoning-grpo-7b/ models/legal-reasoning-grpo-7b/

# Download 7B SFT adapter from Modal volume
modal-download-sft-7b:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Downloading 7B SFT adapter from Modal..."
    cd python && uv run modal volume get legalkit-data models/legal-reasoning-sft-7b/ ../models/legal-reasoning-sft-7b/
    echo "Done. Adapter saved to models/legal-reasoning-sft-7b/"

# Check current spot price for g5.xlarge
train-price:
    #!/usr/bin/env bash
    export AWS_PROFILE={{aws_profile}}
    aws ec2 describe-spot-price-history \
        --region {{region}} \
        --instance-types g5.xlarge \
        --product-descriptions "Linux/UNIX" \
        --start-time $(date -u +%Y-%m-%dT%H:%M:%S) \
        --query 'SpotPriceHistory[:5].{AZ:AvailabilityZone,Price:SpotPrice}' \
        --output table

# ─── Modal ───────────────────────────────────────────────────

# Check GRPO eval results on Modal
modal-eval-status:
    cd python && uv run modal volume ls legalkit-data eval/

# Check GRPO model on Modal
modal-model-status:
    cd python && uv run modal volume ls legalkit-data models/legal-reasoning-grpo/

# Download eval results from Modal
modal-eval-download:
    cd python && uv run modal volume get legalkit-data eval/sft_eval_results.json ../eval/sft_eval_results_v3.json

# ─── Eval ────────────────────────────────────────────────────

# Deploy and run SFT eval on Modal (with current optimized prompt)
eval-sft:
    cd python && uv run modal deploy ../scripts/modal_eval_sft.py && uv run python ../scripts/modal_eval_sft.py --trigger

# Deploy and run GRPO eval on Modal
eval-grpo:
    cd python && uv run modal deploy ../scripts/modal_eval_grpo.py && uv run python ../scripts/modal_eval_grpo.py --trigger

# ─── Utilities ───────────────────────────────────────────────

# List all recipes
help:
    @just --list
