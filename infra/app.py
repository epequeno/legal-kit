#!/usr/bin/env python3
import aws_cdk as cdk

from infra.training_stack import TrainingStack

app = cdk.App()
# Account and region resolve from the CDK CLI environment (CDK_DEFAULT_ACCOUNT,
# CDK_DEFAULT_REGION). The original deployment's account id has been removed.
TrainingStack(app, "LegalKitTraining")

app.synth()
