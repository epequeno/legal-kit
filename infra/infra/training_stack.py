"""CDK stack for LegalKit GPU training infrastructure.

Creates:
- S3 bucket for training artifacts (models, logs)
- IAM role for EC2 with S3 write access
- Security group with SSH
- EC2 key pair
- Launch template for g5.xlarge spot instances (auto-terminate on shutdown)
"""

from aws_cdk import (
    Stack,
    RemovalPolicy,
    CfnOutput,
    aws_s3 as s3,
    aws_ec2 as ec2,
    aws_iam as iam,
)
from constructs import Construct


class TrainingStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- S3 bucket for training artifacts ---
        bucket = s3.Bucket(
            self, "ArtifactsBucket",
            bucket_name=f"legalkit-training-{self.account}",
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
        )

        # --- VPC (use default) ---
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # --- Security group: SSH only ---
        sg = ec2.SecurityGroup(
            self, "TrainingSG",
            vpc=vpc,
            description="LegalKit GPU training - SSH access",
            allow_all_outbound=True,
        )
        sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(22),
            "SSH access",
        )

        # --- IAM role for EC2 ---
        role = iam.Role(
            self, "TrainingRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="LegalKit GPU training instance role",
        )
        bucket.grant_read_write(role)

        # CloudWatch Logs — for real-time training log streaming
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams",
                "logs:DescribeLogGroups",
            ],
            resources=["arn:aws:logs:*:*:log-group:/legalkit/*"],
        ))

        # --- Key pair ---
        key_pair = ec2.KeyPair(
            self, "TrainingKeyPair",
            key_pair_name="legalkit-training",
            type=ec2.KeyPairType.ED25519,
        )

        # --- User data: write setup script to /home/ubuntu ---
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "cat > /home/ubuntu/setup.sh << 'SETUP_EOF'",
            "#!/bin/bash",
            "set -e",
            'echo "=== LegalKit GRPO Training Setup ==="',
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true",
            "",
            "python3 -m venv ~/legalkit-env 2>/dev/null || true",
            "source ~/legalkit-env/bin/activate",
            "pip install --upgrade pip",
            'pip install "torch>=2.6" "transformers>=4.51" "peft>=0.14" "trl>=0.16" \\',
            '  "datasets>=3.2" "accelerate>=1.2" "bitsandbytes>=0.45" "huggingface-hub"',
            "",
            'python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; '
            "AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True); "
            "AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True); "
            "print('Model cached!')\"",
            "",
            'python -c "import torch; print(f\'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\')"',
            "",
            'echo "=== Setup complete ==="',
            "SETUP_EOF",
            "chmod +x /home/ubuntu/setup.sh",
            "chown ubuntu:ubuntu /home/ubuntu/setup.sh",
        )

        # --- Launch template ---
        ami = ec2.MachineImage.generic_linux({
            "us-east-1": "ami-010a2cfb0748c104b",  # DL OSS Nvidia PyTorch 2.8 Ubuntu 24.04
        })

        launch_template = ec2.LaunchTemplate(
            self, "TrainingLaunchTemplate",
            launch_template_name="legalkit-grpo-training",
            instance_type=ec2.InstanceType("g5.xlarge"),
            machine_image=ami,
            key_pair=key_pair,
            security_group=sg,
            role=role,
            user_data=user_data,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/sda1",
                    volume=ec2.BlockDeviceVolume.ebs(
                        100, volume_type=ec2.EbsDeviceVolumeType.GP3,
                    ),
                )
            ],
            instance_initiated_shutdown_behavior=ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
        )

        # --- Outputs ---
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "SecurityGroupId", value=sg.security_group_id)
        CfnOutput(self, "LaunchTemplateId", value=launch_template.launch_template_id)
        CfnOutput(self, "KeyPairName", value=key_pair.key_pair_name)
