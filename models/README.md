# Models

Model weights are **not included in this repository**.

When the project was active, the trained artifacts (`legal-embeddings`,
`legal-entailment`, `structure-parser`, and the reasoning adapters) were backed up to S3.
The bucket name and account id have been removed, and the weights are not redistributed here.

The app expects weights under `models/legal-embeddings/` and `models/legal-entailment/`;
without them it will not run. See the repository README, "What is not in this repository".
