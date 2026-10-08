#!/bin/bash
# Launch a g5.xlarge spot instance for GRPO training.
#
# Usage:
#   bash scripts/ec2_launch.sh          # launch instance
#   bash scripts/ec2_launch.sh status   # check running instances
#   bash scripts/ec2_launch.sh stop     # terminate all legalkit instances
#
# After launch, connect with:
#   ssh -i ~/.ssh/legalkit-ec2.pem ubuntu@<public-ip>
#
# Then run the setup + training:
#   bash setup.sh  # (uploaded via user-data)

set -e

export AWS_PROFILE=arch
REGION="us-east-1"
INSTANCE_TYPE="g5.xlarge"
AMI="ami-010a2cfb0748c104b"  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04)
KEY_NAME="legalkit-ec2"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
SG_NAME="legalkit-gpu"
TAG="legalkit-grpo"

# --- Helper: status ---
if [ "${1}" = "status" ]; then
    echo "Running LegalKit instances:"
    aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=tag:Project,Values=$TAG" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[].Instances[].{ID:InstanceId,Type:InstanceType,State:State.Name,IP:PublicIpAddress,Launch:LaunchTime,Spot:SpotInstanceRequestId}' \
        --output table
    exit 0
fi

# --- Helper: stop ---
if [ "${1}" = "stop" ]; then
    INSTANCES=$(aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=tag:Project,Values=$TAG" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[].Instances[].InstanceId' \
        --output text)
    if [ -z "$INSTANCES" ]; then
        echo "No running LegalKit instances."
    else
        echo "Terminating: $INSTANCES"
        aws ec2 terminate-instances --region "$REGION" --instance-ids $INSTANCES --output text
    fi
    exit 0
fi

# --- 1. Key pair ---
if [ ! -f "$KEY_FILE" ]; then
    echo "Creating key pair: $KEY_NAME"
    aws ec2 create-key-pair \
        --region "$REGION" \
        --key-name "$KEY_NAME" \
        --key-type ed25519 \
        --query 'KeyMaterial' \
        --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    echo "  Saved to $KEY_FILE"
else
    echo "Key pair exists: $KEY_FILE"
    # Make sure it exists in AWS too
    aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1 || {
        echo "  Key file exists locally but not in AWS. Importing..."
        # Generate public key from private
        ssh-keygen -y -f "$KEY_FILE" > /tmp/legalkit-ec2.pub
        aws ec2 import-key-pair \
            --region "$REGION" \
            --key-name "$KEY_NAME" \
            --public-key-material fileb:///tmp/legalkit-ec2.pub
        rm /tmp/legalkit-ec2.pub
    }
fi

# --- 2. Security group with SSH ---
SG_ID=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null)

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group: $SG_NAME"
    SG_ID=$(aws ec2 create-security-group \
        --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "LegalKit GPU training - SSH only" \
        --query 'GroupId' \
        --output text)

    # Allow SSH from anywhere (you can restrict to your IP)
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0
    echo "  Created: $SG_ID (SSH open)"
else
    echo "Security group exists: $SG_ID"
fi

# --- 3. Launch spot instance ---
echo ""
echo "Launching $INSTANCE_TYPE spot instance..."
echo "  AMI: $AMI"
echo "  Region: $REGION"

# Best AZ by price (us-east-1d was cheapest)
RESULT=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=legalkit-grpo},{Key=Project,Value=$TAG}]" \
    --query 'Instances[0].{ID:InstanceId,State:State.Name}' \
    --output json)

INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['ID'])")
echo "  Instance: $INSTANCE_ID"

echo "  Waiting for public IP..."
sleep 10

for i in $(seq 1 12); do
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    if [ "$PUBLIC_IP" != "None" ] && [ -n "$PUBLIC_IP" ]; then
        break
    fi
    sleep 5
done

echo ""
echo "============================================"
echo "  Instance launched!"
echo "  ID:  $INSTANCE_ID"
echo "  IP:  $PUBLIC_IP"
echo "============================================"
echo ""
echo "Connect (wait ~2 min for boot):"
echo "  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo ""
echo "Upload data:"
echo "  scp -i $KEY_FILE -r models/legal-reasoning-sft/ ubuntu@$PUBLIC_IP:~/"
echo "  scp -i $KEY_FILE data/grpo_prompts/grpo_prompts.jsonl ubuntu@$PUBLIC_IP:~/"
echo "  scp -i $KEY_FILE scripts/train_grpo_local.py ubuntu@$PUBLIC_IP:~/"
echo "  scp -i $KEY_FILE scripts/ec2_setup.sh ubuntu@$PUBLIC_IP:~/"
echo ""
S3_BUCKET="${LEGALKIT_S3_BUCKET:-s3://legalkit-artifacts}"

echo "On the instance:"
echo "  bash ec2_setup.sh"
echo ""
echo "  # Option A: fire-and-forget (uploads to S3, then shuts down)"
echo "  nohup bash -c 'source ~/legalkit-env/bin/activate && python ~/train_grpo_local.py \\"
echo "    --sft-adapter ~/legal-reasoning-sft \\"
echo "    --prompts ~/grpo_prompts.jsonl \\"
echo "    --output ~/legal-reasoning-grpo \\"
echo "    --s3-upload $S3_BUCKET \\"
echo "    --shutdown-on-complete' > ~/train.log 2>&1 &"
echo "  # Then disconnect. Instance auto-terminates when done."
echo ""
echo "  # Option B: interactive (stay connected)"
echo "  tmux new -s train"
echo "  source ~/legalkit-env/bin/activate"
echo "  python ~/train_grpo_local.py --sft-adapter ~/legal-reasoning-sft --prompts ~/grpo_prompts.jsonl --output ~/legal-reasoning-grpo"
echo ""
echo "The instance auto-terminates on shutdown (--instance-initiated-shutdown-behavior terminate)."
echo ""
echo "Download results from S3:"
echo "  AWS_PROFILE=arch aws s3 sync $S3_BUCKET/legal-reasoning-grpo/ models/legal-reasoning-grpo/"
echo ""
echo "Manual stop if needed:"
echo "  bash scripts/ec2_launch.sh stop"
