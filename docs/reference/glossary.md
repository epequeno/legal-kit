# Glossary

## RAG & Retrieval

**RAG (Retrieval-Augmented Generation)**: Architecture pattern where an LLM's response is grounded in externally retrieved documents rather than relying solely on training data.

**BM25 (Best Match 25)**: A keyword-based ranking function used in information retrieval. Scores documents by term frequency, inverse document frequency, and document length normalization. Complements semantic search by catching exact-match queries.

**Reciprocal Rank Fusion (RRF)**: A method for combining ranked results from multiple retrieval systems. Each result's score is `1/(k + rank)`, summed across all lists. Simple, effective, and parameter-light.

**Contextual Retrieval**: Technique (introduced by Anthropic, 2024) where an LLM generates a short context description for each chunk before embedding. Reduces retrieval failure by up to 67% when combined with hybrid search and reranking.

**Late Chunking**: Processes an entire document through an embedding model before splitting into chunks, preserving global context in the embeddings. More efficient than contextual retrieval but with lower semantic preservation.

**Hybrid Search**: Combining multiple retrieval methods (typically semantic/dense + keyword/sparse) to improve recall and precision. Standard approach in production RAG systems.

## Evaluation & Metrics

**Precision@k**: Of the top-k retrieved results, what fraction are relevant? Measures retrieval accuracy.

**Recall@k**: Of all relevant results in the corpus, what fraction appear in the top-k? Measures retrieval completeness.

**MRR (Mean Reciprocal Rank)**: Average of `1/rank` for the first relevant result across queries. Measures how early the first good result appears.

**nDCG (Normalized Discounted Cumulative Gain)**: Evaluates both relevance and ranking position with graded relevance scores.

**Faithfulness**: Whether a generated response is factually consistent with the retrieved context. Distinct from correctness (which measures agreement with ground truth).

**Entailment**: In NLI (Natural Language Inference), whether a premise logically supports a hypothesis. We use this to verify that cited passages support the claims made about them.

**Hallucination**: Generated content that appears plausible but is not grounded in the provided context or source documents. In legal AI, this includes fabricated citations, invented quotes, and unsupported claims.

## Legal Document Structure

**Clause**: A distinct provision within a legal document, typically a numbered paragraph or sub-paragraph.

**Cross-Reference**: A reference within one section of a document to another section (e.g., "as defined in Section 1.1" or "subject to the limitations in Article VIII").

**Recital**: Introductory paragraphs in a contract (typically "WHEREAS" clauses) that provide background context.

**Exhibit/Schedule**: Supplementary documents attached to and incorporated by reference in the main agreement.

## Benchmarks

**LegalBench-RAG**: Open-source benchmark for evaluating retrieval in legal RAG systems. 6,858 queries over 79M characters of legal text, with character-level span annotations. Datasets: CUAD, MAUD, ContractNLI, PrivacyQA.

**LegalBench**: 162-task benchmark for legal reasoning in LLMs (not retrieval-focused). Covers issue-spotting, rule-recall, rule-conclusion, rule-application, interpretation, and rhetorical understanding.

**ALCE**: Princeton benchmark for evaluating LLM citation quality. Measures fluency, correctness, and citation precision/recall across ASQA, QAMPARI, and ELI5 datasets.

**CRAG**: Comprehensive RAG benchmark. State-of-the-art industry solutions achieve 63% hallucination-free responses.
