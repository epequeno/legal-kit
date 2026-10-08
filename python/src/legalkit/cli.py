"""LegalKit CLI — citation engine proof of concept."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from legalkit.pipeline.parse import parse_document, document_summary
from legalkit.pipeline.chunk import chunk_document
from legalkit.pipeline.index import DocumentIndex
from legalkit.pipeline.embed import embed_texts, embed_query
from legalkit.pipeline.retrieve import retrieve

console = Console()

# In-memory store for ingested documents
_store: dict[str, dict] = {}


@click.group()
@click.version_option()
def main():
    """LegalKit — legal AI with verifiable citation accuracy."""
    pass


@main.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--no-context", is_flag=True, help="Skip LLM contextual enrichment")
def ingest(file: Path, no_context: bool):
    """Ingest a legal document — parse, chunk, embed, and index.

    Supported formats: .txt, .pdf, .docx
    """
    from legalkit.pipeline.extract import extract_text, SUPPORTED_EXTENSIONS

    suffix = file.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        console.print(
            f"[red]Error:[/red] Unsupported file type {suffix!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
        raise SystemExit(1)

    console.print(f"\n📄 Ingesting [bold]{file.name}[/bold]...\n")

    # Stage 1: Extract text
    try:
        with console.status(f"Extracting text from {suffix} file..."):
            text = extract_text(file)
    except RuntimeError as e:
        console.print(f"[red]Extraction failed:[/red] {e}")
        raise SystemExit(1)

    console.print(f"  ✓ Extracted: {len(text):,} characters")

    doc_id = file.stem
    t0 = time.time()

    with console.status("Parsing document structure..."):
        root = parse_document(text, doc_id=doc_id)

    nodes = root.all_nodes()
    console.print(f"  ✓ Parsed: {len(nodes)} structural nodes")

    # Stage 2: Chunk
    with console.status("Chunking..."):
        chunks = chunk_document(root, doc_id=doc_id)

    console.print(f"  ✓ Chunked: {len(chunks)} chunks")

    # Stage 3: Contextual enrichment (optional)
    if not no_context:
        try:
            from legalkit.llm.client import LLMClient
            from legalkit.pipeline.contextualize import build_document_context, contextualize_chunks

            llm = LLMClient()
            doc_context = build_document_context(root, text)

            progress_state = {"last_reported": 0}

            def _progress(completed: int, total: int) -> None:
                # Report every 10 chunks or on completion
                if completed == total or completed - progress_state["last_reported"] >= 10:
                    console.print(f"    Contextualized {completed}/{total} chunks")
                    progress_state["last_reported"] = completed

            with console.status(f"Contextualizing {len(chunks)} chunks (concurrent)..."):
                contextualize_chunks(
                    chunks,
                    doc_context=doc_context,
                    llm=llm,
                    progress_callback=_progress,
                )

            enriched = sum(1 for c in chunks if c.context_text)
            console.print(f"  ✓ Contextualized: {enriched}/{len(chunks)} chunks")
        except Exception as e:
            console.print(f"  ⚠ Skipping contextual enrichment: {e}")
    else:
        console.print("  ⊘ Skipping contextual enrichment (--no-context)")

    # Stage 3b: Embed
    with console.status("Generating embeddings..."):
        embedding_texts = [chunk.embedding_text for chunk in chunks]
        embeddings = embed_texts(embedding_texts)

    console.print(f"  ✓ Embedded: {embeddings.shape}")

    # Build index
    with console.status("Building index..."):
        index = DocumentIndex()
        index.build(chunks, embeddings)

    elapsed = time.time() - t0
    console.print(f"  ✓ Indexed: {index.size} chunks in {elapsed:.1f}s")

    # Build definition index
    with console.status("Building definition index..."):
        from legalkit.pipeline.definitions import DefinitionIndex
        def_index = DefinitionIndex()
        def_index.build(root)
    console.print(f"  ✓ Definitions: {len(def_index)} terms indexed")

    # Store in memory
    _store[doc_id] = {
        "root": root,
        "chunks": chunks,
        "index": index,
        "def_index": def_index,
        "text": text,
        "file": str(file),
    }

    # Print document structure
    console.print(f"\n[dim]{document_summary(root)}[/dim]\n")
    console.print(f"[green]✓ Ready.[/green] Use [bold]legalkit ask[/bold] to query.\n")

    # Save index to disk for persistence across CLI calls
    _save_store(doc_id, _store[doc_id])


@main.command()
@click.argument("question")
@click.option("--doc", "-d", required=True, help="Document ID or filename")
@click.option("--top-k", default=20, help="Number of chunks to retrieve")
@click.option("--verbose", "-v", is_flag=True, help="Show retrieval details")
@click.option("--json-out", "-j", type=click.Path(path_type=Path), help="Save report as JSON")
@click.option("--retrieval-only", is_flag=True, help="Only show retrieval, skip generation/verification")
@click.option("--no-expansion", is_flag=True, help="Skip LLM query expansion")
@click.option("--expansions", default=3, show_default=True, help="Number of query expansions to generate")
@click.option("--local-nli", "local_nli_path", default=None,
              help="Path to fine-tuned NLI model dir to use instead of LLM entailment "
                   "(e.g. models/legal-entailment/model). Faster and free; "
                   "falls back to LLM if path doesn't exist.")
def ask(question: str, doc: str, top_k: int, verbose: bool, json_out: Path | None, retrieval_only: bool, no_expansion: bool, expansions: int, local_nli_path: str | None):
    """Ask a question about an ingested document."""
    doc_id = Path(doc).stem if "." in doc else doc

    # Load from disk if not in memory
    if doc_id not in _store:
        store_data = _load_store(doc_id)
        if store_data is None:
            console.print(f"[red]Error:[/red] Document '{doc_id}' not found. Run [bold]legalkit ingest[/bold] first.")
            raise SystemExit(1)
        _store[doc_id] = store_data

    data = _store[doc_id]
    index = data["index"]
    root = data["root"]

    console.print(f"\n🔍 [bold]{doc_id}[/bold]: [italic]{question}[/italic]\n")

    # Stage 4-6: Retrieval
    with console.status("Embedding query..."):
        query_emb = embed_query(question)

    def_index = data.get("def_index")

    if no_expansion:
        with console.status("Retrieving..."):
            result = retrieve(
                query=question,
                query_embedding=query_emb,
                index=index,
                doc_tree=root,
                top_k=top_k,
                definition_index=def_index,
            )
        console.print(
            f"  Retrieved [bold]{len(result.ranked_chunks)}[/bold] chunks "
            f"+ [bold]{len(result.cross_ref_chunks)}[/bold] cross-refs "
            f"+ [bold]{len(result.definition_chunks)}[/bold] definitions\n"
        )
    else:
        from legalkit.llm.client import LLMClient
        from legalkit.pipeline.expand import expand_query_with_retrieval
        from legalkit.pipeline.embed import embed_texts

        llm = LLMClient()
        with console.status(f"Expanding query + retrieving ({expansions} expansions)..."):
            result = expand_query_with_retrieval(
                query=question,
                query_embedding=query_emb,
                index=index,
                doc_tree=root,
                llm=llm,
                top_k=top_k,
                n_expansions=expansions,
                embed_fn=embed_texts,
                definition_index=def_index,
            )

        exp_str = f" via {len(result.expanded_queries)} expansions" if result.expanded_queries else ""
        console.print(
            f"  Retrieved [bold]{len(result.ranked_chunks)}[/bold] chunks "
            f"+ [bold]{len(result.cross_ref_chunks)}[/bold] cross-refs "
            f"+ [bold]{len(result.definition_chunks)}[/bold] definitions{exp_str}\n"
        )
        if verbose and result.expanded_queries:
            console.print("  [dim]Expansions used:[/dim]")
            for eq in result.expanded_queries:
                console.print(f"    [dim]• {eq}[/dim]")
            console.print()

    if retrieval_only:
        _print_retrieval_table(result, verbose)
        return

    # Stage 7-9: Generate, Verify, Report
    from legalkit.llm.client import LLMClient
    from legalkit.pipeline.report import build_evidence_report
    from legalkit.pipeline.nli_model import LocalNLIModel

    llm = LLMClient()

    # Load local NLI model if requested
    local_nli = LocalNLIModel.load(local_nli_path) if local_nli_path else None
    if local_nli_path and local_nli is None:
        console.print(f"[yellow]⚠ Local NLI model not found at {local_nli_path}, falling back to LLM[/yellow]")

    # Print entailment backend info
    if local_nli:
        console.print(f"  [dim]Entailment: local DeBERTa ({local_nli.model_path})[/dim]")
    else:
        console.print(f"  [dim]Entailment: LLM-as-judge[/dim]")

    with console.status("Generating answer..."):
        report = build_evidence_report(
            question=question,
            retrieval_result=result,
            doc_tree=root,
            llm=llm,
            doc_id=doc_id,
            local_nli_model=local_nli,
        )

    # Display the evidence report
    _print_evidence_report(report, verbose)

    # Save JSON if requested
    if json_out:
        json_out.write_text(report.to_json())
        console.print(f"\n[dim]Report saved to {json_out}[/dim]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_retrieval_table(result, verbose: bool) -> None:
    """Print retrieval results as a table."""
    from legalkit.models.retrieval import RetrievalResult

    table = Table(title="Retrieved Chunks", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Section", style="cyan", width=12)
    table.add_column("Score", width=8)
    table.add_column("Method", width=10)
    table.add_column("Text Preview", ratio=1)

    for i, rc in enumerate(result.all_chunks):
        method_style = {
            "semantic": "green", "bm25": "yellow",
            "hybrid": "blue", "cross_reference": "magenta",
        }.get(rc.method.value, "white")
        preview = rc.chunk.text[:150].replace("\n", " ")
        table.add_row(
            str(i + 1), rc.chunk.node_id, f"{rc.score:.4f}",
            Text(rc.method.value, style=method_style), preview + "...",
        )
    console.print(table)


def _print_evidence_report(report, verbose: bool) -> None:
    """Print a rich evidence report to the console."""
    from legalkit.models.report import EvidenceReport, ConfidenceLevel
    from legalkit.models.citation import VerificationStatus, EntailmentClass

    # ── Answer ───────────────────────────────────────────────────────
    conf = report.overall_confidence
    conf_style = {
        ConfidenceLevel.HIGH: "green",
        ConfidenceLevel.MEDIUM: "yellow",
        ConfidenceLevel.LOW: "red",
    }[conf.level]

    console.print(Panel(
        report.answer,
        title="[bold]Answer[/bold]",
        subtitle=f"confidence: [{conf_style}]{conf.level.value}[/{conf_style}] ({conf.score:.2f}) | {report.elapsed_seconds:.1f}s",
        border_style=conf_style,
        padding=(1, 2),
    ))

    # ── Verified Claims ──────────────────────────────────────────────
    console.print(f"\n[bold]Verified Claims[/bold] ({len(report.claims)})\n")

    for i, claim in enumerate(report.claims, 1):
        if claim.is_gap:
            console.print(f"  [dim]{i}. ℹ️  {claim.text}[/dim]")
            continue

        cl = claim.confidence.level
        icon = {"high": "✅", "medium": "⚠️ ", "low": "❌"}[cl.value]
        style = {"high": "green", "medium": "yellow", "low": "red"}[cl.value]

        console.print(f"  {i}. {icon} [{style}][{cl.value.upper()}][/{style}] {claim.text}")

        for cit in claim.citations:
            status_icon = {
                VerificationStatus.VERIFIED: "✓",
                VerificationStatus.PARTIALLY_VERIFIED: "~",
                VerificationStatus.UNVERIFIED: "✗",
                VerificationStatus.NOT_CHECKED: "?",
            }[cit.status]

            status_style = {
                VerificationStatus.VERIFIED: "green",
                VerificationStatus.PARTIALLY_VERIFIED: "yellow",
                VerificationStatus.UNVERIFIED: "red",
                VerificationStatus.NOT_CHECKED: "dim",
            }[cit.status]

            console.print(
                f"     [{status_style}]{status_icon}[/{status_style}] "
                f"[cyan]{cit.section_id}[/cyan] — "
                f"exists={'✓' if cit.exists else '✗'} "
                f"entailment={cit.entailment.value}({cit.entailment_score:.2f})"
            )

            if cit.entailment_explanation and verbose:
                console.print(f"       [dim]{cit.entailment_explanation}[/dim]")

            if cit.source_text and verbose:
                preview = cit.source_text[:200].replace("\n", " ")
                console.print(f"       [dim]source: \"{preview}...\"[/dim]")

        if claim.confidence.flags:
            flags_str = ", ".join(claim.confidence.flags)
            console.print(f"     [dim]flags: {flags_str}[/dim]")

        console.print()

    # ── Gaps ─────────────────────────────────────────────────────────
    if report.gaps:
        console.print("[bold]Identified Gaps[/bold]\n")
        for gap in report.gaps:
            console.print(f"  ⊘ {gap}")
        console.print()

    # ── Summary ──────────────────────────────────────────────────────
    rs = report.retrieval_summary
    verified_count = sum(
        1 for c in report.claims
        for cit in c.citations
        if cit.status == VerificationStatus.VERIFIED
    )
    total_citations = sum(len(c.citations) for c in report.claims)

    summary_table = Table(title="Report Summary", show_header=False, box=None, padding=(0, 2))
    summary_table.add_column(style="dim")
    summary_table.add_column()
    summary_table.add_row("Chunks retrieved", f"{rs.chunks_retrieved} + {rs.cross_ref_chunks_added} cross-ref")
    summary_table.add_row("Claims extracted", str(len(report.claims)))
    summary_table.add_row("Citations verified", f"{verified_count}/{total_citations}")
    summary_table.add_row("Overall confidence", f"[{conf_style}]{conf.level.value} ({conf.score:.2f})[/{conf_style}]")
    summary_table.add_row("Gaps identified", str(len(report.gaps)))
    summary_table.add_row("Model", report.model)
    summary_table.add_row("Time", f"{report.elapsed_seconds:.1f}s")
    console.print(summary_table)


# ---------------------------------------------------------------------------
# Persistence helpers (simple pickle-based for PoC)
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(".legalkit_cache")


def _save_store(doc_id: str, data: dict) -> None:
    """Save ingested document data to disk."""
    import pickle
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{doc_id}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)


def _load_store(doc_id: str) -> dict | None:
    """Load ingested document data from disk."""
    import pickle
    cache_file = _CACHE_DIR / f"{doc_id}.pkl"
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as f:
        return pickle.load(f)


@main.command()
@click.option("--download", is_flag=True, help="Download the LegalBench-RAG dataset")
@click.option("--mini/--full", default=True, help="Use mini (776 queries) or full (6858) dataset")
@click.option("--dataset", "-d", multiple=True, help="Specific dataset(s): privacy_qa, contractnli, maud, cuad")
@click.option("--max-per-dataset", "-n", type=int, help="Max queries per dataset")
@click.option("--top-k", default=64, help="Max chunks to retrieve per query")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Save results JSON")
@click.option("--ml-parser", is_flag=True, default=False, help="Use ML+regex hybrid parser instead of regex-only.")
@click.option("--embedding-model", default=None, help="Path to local sentence-transformers model (default: API).")
def bench(download: bool, mini: bool, dataset: tuple, max_per_dataset: int | None, top_k: int, output: Path | None, ml_parser: bool, embedding_model: str | None):
    """Run the LegalBench-RAG benchmark."""
    from legalkit.eval.benchmark import (
        download_dataset,
        load_benchmark,
        run_benchmark,
        DATASET_NAMES,
    )

    if download:
        download_dataset()
        return

    datasets = list(dataset) if dataset else None

    console.print("\n[bold]LegalBench-RAG Benchmark[/bold]\n")
    mode = "mini" if mini else "full"
    ds_str = ", ".join(datasets or DATASET_NAMES)
    console.print(f"  Mode: {mode}")
    console.print(f"  Datasets: {ds_str}")
    if max_per_dataset:
        console.print(f"  Max per dataset: {max_per_dataset}")
    console.print(f"  Top-k: {top_k}")
    console.print()

    # Load queries
    try:
        queries = load_benchmark(
            datasets=datasets,
            mini=mini,
            max_per_dataset=max_per_dataset,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run [bold]legalkit bench --download[/bold] to get the dataset.")
        raise SystemExit(1)

    corpus_files = set()
    for q in queries:
        for s in q.snippets:
            corpus_files.add(s.file_path)

    console.print(f"  Queries: {len(queries)}")
    console.print(f"  Corpus files: {len(corpus_files)}")
    
    # Build embedding function if local model is specified
    embedding_fn = None
    if embedding_model:
        from legalkit.pipeline.embed import embed_texts_local
        console.print(f"  Embedding model: {embedding_model} (local)")
        embedding_fn = lambda texts: embed_texts_local(texts, model_path=embedding_model)
    else:
        console.print(f"  Embedding model: text-embedding-3-large (API)")
    console.print()

    t0 = time.time()

    def progress_cb(stage: str, current: int, total: int):
        if current == 0:
            console.print(f"  [dim]{stage}...[/dim]")
        elif current == total or current % 20 == 0:
            console.print(f"    {stage}: {current}/{total}")

    # Run
    results = run_benchmark(
        queries=queries,
        max_chunks_per_query=top_k,
        use_ml_parser=ml_parser,
        embedding_fn=embedding_fn,
        progress_callback=progress_cb,
    )

    elapsed = time.time() - t0

    # Display results
    console.print(f"\n[bold]Results[/bold] ({elapsed:.1f}s)\n")
    console.print(f"[dim]{results.summary_table()}[/dim]\n")

    # Detailed table with Rich
    ks = [1, 2, 4, 8, 16, 32, 64]

    table = Table(title="Precision@k / Recall@k (%)", show_lines=True)
    table.add_column("Dataset", style="cyan")
    for k in ks:
        table.add_column(f"P@{k}", justify="right", width=7)
    for k in ks:
        table.add_column(f"R@{k}", justify="right", width=7)

    for tag in results.tags:
        row = [tag]
        row.extend(f"{results.precision_at_k(k, tag)*100:.2f}" for k in ks)
        row.extend(f"{results.recall_at_k(k, tag)*100:.2f}" for k in ks)
        table.add_row(*row)

    # Overall
    row = ["[bold]ALL[/bold]"]
    row.extend(f"[bold]{results.precision_at_k(k)*100:.2f}[/bold]" for k in ks)
    row.extend(f"[bold]{results.recall_at_k(k)*100:.2f}[/bold]" for k in ks)
    table.add_row(*row)

    console.print(table)

    # Save results
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        import datetime
        result_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": mode,
            "datasets": datasets or DATASET_NAMES,
            "num_queries": len(queries),
            "num_corpus_files": len(corpus_files),
            "top_k": top_k,
            "elapsed_seconds": round(elapsed, 2),
            "metrics": {
                "overall": {
                    "precision": {f"@{k}": round(results.precision_at_k(k), 4) for k in ks},
                    "recall": {f"@{k}": round(results.recall_at_k(k), 4) for k in ks},
                },
                "per_dataset": {
                    tag: {
                        "precision": {f"@{k}": round(results.precision_at_k(k, tag), 4) for k in ks},
                        "recall": {f"@{k}": round(results.recall_at_k(k, tag), 4) for k in ks},
                    }
                    for tag in results.tags
                },
            },
        }
        output.write_text(json.dumps(result_data, indent=2))
        console.print(f"\nResults saved to [bold]{output}[/bold]")


@main.command()
@click.option("--dataset", "-d", multiple=True, help="Specific dataset(s)")
@click.option("-n", "--num-queries", type=int, default=5, help="Queries per dataset")
@click.option("--no-baseline", is_flag=True, help="Skip full-context baseline comparison")
@click.option("--top-k", default=20, help="Chunks to retrieve")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Save results JSON")
@click.option("--local-nli", "local_nli_path", default=None,
              help="Path to fine-tuned NLI model for entailment checking")
def e2e(dataset: tuple, num_queries: int, no_baseline: bool, top_k: int, output: Path | None, local_nli_path: str | None):
    """Run end-to-end evaluation: RAG pipeline vs full-context baseline."""
    from legalkit.eval.benchmark import load_benchmark, DATASET_NAMES
    from legalkit.eval.e2e import run_e2e_evaluation
    from legalkit.llm.client import LLMClient

    datasets = list(dataset) if dataset else None

    console.print("\n[bold]End-to-End Evaluation[/bold]\n")
    console.print(f"  Datasets: {', '.join(datasets or DATASET_NAMES)}")
    console.print(f"  Queries per dataset: {num_queries}")
    console.print(f"  Baseline comparison: {'no' if no_baseline else 'yes'}")
    console.print(f"  Top-k: {top_k}")
    console.print()

    try:
        queries = load_benchmark(
            datasets=datasets,
            mini=True,
            max_per_dataset=num_queries,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    console.print(f"  Loaded {len(queries)} queries\n")

    llm = LLMClient()

    def progress(current, total, msg):
        console.print(f"  [{current}/{total}] {msg}")

    results = run_e2e_evaluation(
        queries=queries,
        llm=llm,
        run_baseline_comparison=not no_baseline,
        top_k=top_k,
        progress_callback=progress,
    )

    # Display results
    console.print(f"\n{'='*60}")
    console.print("[bold]Results[/bold]\n")

    # Per-query results
    for r in results.results:
        rag_j = r.rag_judgment
        base_j = r.baseline_judgment

        rag_icon = {"correct": "✅", "partially_correct": "⚠️ ", "incorrect": "❌"}.get(
            rag_j.correctness if rag_j else "", "?"
        )
        base_icon = {"correct": "✅", "partially_correct": "⚠️ ", "incorrect": "❌"}.get(
            base_j.correctness if base_j else "", "—"
        )

        cit_str = f"{r.rag_citations_verified}/{r.rag_citations_total} verified"
        console.print(
            f"  {rag_icon} RAG | {base_icon} Base | "
            f"cit: {cit_str} | "
            f"[dim]{r.dataset}: {r.query[:60]}...[/dim]"
        )
        if rag_j and rag_j.explanation:
            console.print(f"    [dim]RAG:  {rag_j.explanation}[/dim]")
        if base_j and base_j.explanation:
            console.print(f"    [dim]Base: {base_j.explanation}[/dim]")

    # Summary table
    console.print(f"\n{'='*60}")
    console.print("[bold]Summary[/bold]\n")

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column(style="dim", width=35)
    summary.add_column(width=15)
    summary.add_column(width=15)

    summary.add_row("", "[bold]RAG Pipeline[/bold]", "[bold]Full Context[/bold]")
    summary.add_row("", "─" * 13, "─" * 13)

    rag_cr = results.rag_correctness_rate
    base_cr = results.baseline_correctness_rate
    rag_style = "green" if rag_cr >= 0.8 else "yellow" if rag_cr >= 0.6 else "red"
    base_style = "green" if base_cr >= 0.8 else "yellow" if base_cr >= 0.6 else "red"

    summary.add_row(
        "Correctness (incl. partial)",
        f"[{rag_style}]{rag_cr:.0%}[/{rag_style}]",
        f"[{base_style}]{base_cr:.0%}[/{base_style}]" if not no_baseline else "—",
    )
    summary.add_row(
        "Strict correctness",
        f"{results.rag_strict_correctness_rate:.0%}",
        f"{results.baseline_strict_correctness_rate:.0%}" if not no_baseline else "—",
    )
    summary.add_row(
        "Citation verification rate",
        f"{results.rag_avg_verification_rate:.0%}",
        "n/a",
    )
    summary.add_row(
        "Strict verification rate",
        f"{results.rag_avg_strict_verification_rate:.0%}",
        "n/a",
    )
    summary.add_row(
        "Hallucination rate",
        f"{results.rag_hallucination_rate:.0%}",
        "n/a",
    )
    summary.add_row(
        "Queries evaluated",
        str(len(results.results)),
        str(len(results.results)) if not no_baseline else "—",
    )
    summary.add_row(
        "Total time",
        f"{results.total_elapsed:.0f}s",
        "",
    )

    console.print(summary)

    # Save
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results.to_dict(), indent=2))
        console.print(f"\nResults saved to [bold]{output}[/bold]")


@main.command()
@click.option("--export", "-e", type=click.Path(path_type=Path), help="Export all samples as JSON")
def training_data(export: Path | None):
    """Inspect entailment training data collected from pipeline runs."""
    from legalkit.eval.entailment_log import EntailmentLogger

    stats = EntailmentLogger.stats()

    if stats["total"] == 0:
        console.print("\n[dim]No entailment samples collected yet.[/dim]")
        console.print("Run [bold]legalkit ask[/bold] or [bold]legalkit e2e[/bold] to start collecting.\n")
        return

    console.print("\n[bold]Entailment Training Data[/bold]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=20)
    table.add_column()

    table.add_row("Total samples", str(stats["total"]))
    table.add_row("Date range", f"{stats['date_range'][0]} — {stats['date_range'][1]}")
    table.add_row("Unique documents", str(stats["unique_docs"]))
    table.add_row("Reviewed", f"{stats['reviewed']} / {stats['total']}")
    console.print(table)

    # Class distribution
    console.print("\n[bold]By entailment class:[/bold]")
    for cls, count in sorted(stats.get("by_class", {}).items(), key=lambda x: -x[1]):
        pct = count / stats["total"] * 100
        bar = "█" * int(pct / 2)
        console.print(f"  {cls:<25} {count:>5}  ({pct:>5.1f}%)  {bar}")

    # Source distribution
    console.print("\n[bold]By source:[/bold]")
    for src, count in sorted(stats.get("by_source", {}).items(), key=lambda x: -x[1]):
        console.print(f"  {src:<25} {count:>5}")

    if export:
        samples = EntailmentLogger.load_all()
        from dataclasses import asdict
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(json.dumps([asdict(s) for s in samples], indent=2))
        console.print(f"\nExported {len(samples)} samples to [bold]{export}[/bold]")

    console.print()


@main.command()
@click.option("--max-docs", default=10, help="Max corpus documents to process")
@click.option("--passages-per-doc", default=5, help="Passages per doc for LLM generation")
@click.option("--cross-passage", default=20, help="Cross-passage pairs per doc")
@click.option("--strategy", "-s", multiple=True, help="Strategies: llm_generated, perturbation, cross_passage")
@click.option("--no-llm", is_flag=True, help="Skip LLM-based generation (perturbation + cross-passage only)")
def generate_data(max_docs: int, passages_per_doc: int, cross_passage: int, strategy: tuple, no_llm: bool):
    """Generate synthetic entailment training data from the legal corpus."""
    from legalkit.eval.synthetic import generate_synthetic_dataset

    strategies = list(strategy) if strategy else None
    if no_llm:
        strategies = [s for s in (strategies or ["perturbation", "cross_passage"]) if s != "llm_generated"]

    console.print("\n[bold]Synthetic Data Generation[/bold]\n")
    console.print(f"  Max documents: {max_docs}")
    console.print(f"  Passages per doc: {passages_per_doc}")
    console.print(f"  Cross-passage per doc: {cross_passage}")
    console.print(f"  Strategies: {', '.join(strategies or ['all'])}")
    console.print()

    llm = None
    if not no_llm:
        from legalkit.llm.client import LLMClient
        try:
            llm = LLMClient()
        except Exception as e:
            console.print(f"  [yellow]LLM unavailable ({e}), skipping llm_generated strategy[/yellow]")
            strategies = [s for s in (strategies or ["perturbation", "cross_passage"]) if s != "llm_generated"]

    def progress(stage, current, total):
        if current == 0:
            console.print(f"  [dim]{stage}...[/dim]")
        elif current == total or current % 10 == 0:
            console.print(f"    {stage}: {current}/{total}")

    results = generate_synthetic_dataset(
        llm=llm,
        max_docs=max_docs,
        passages_per_doc=passages_per_doc,
        cross_passage_per_doc=cross_passage,
        strategies=strategies,
        progress_callback=progress,
    )

    console.print(f"\n[bold]Generated samples:[/bold]")
    total = 0
    for strategy_name, count in results.items():
        console.print(f"  {strategy_name}: {count}")
        total += count
    console.print(f"  [bold]Total: {total}[/bold]")

    # Show updated stats
    from legalkit.eval.entailment_log import EntailmentLogger
    stats = EntailmentLogger.stats()
    console.print(f"\n[dim]Total samples in log: {stats['total']}[/dim]\n")


@main.command("ablation")
@click.option("--dataset", "-d", multiple=True,
              help="Specific dataset(s): privacy_qa, contractnli, maud, cuad")
@click.option("--max-per-dataset", "-n", type=int, default=50, show_default=True,
              help="Max queries per dataset.")
@click.option("--top-k", default=64, show_default=True,
              help="Retrieval depth for each run.")
@click.option("--output", "-o", type=click.Path(path_type=Path),
              help="Save full results as JSON.")
def ablation(dataset: tuple, max_per_dataset: int, top_k: int, output: Path | None):
    """Ablation study — measure the contribution of each pipeline component.

    Runs retrieval benchmark with each component added incrementally:
    semantic → bm25 → hybrid → hybrid+xref

    Marginal recall deltas show how much each component actually contributes.
    """
    from legalkit.eval.benchmark import load_benchmark, DATASET_NAMES
    from legalkit.eval.ablation import run_ablation, DEFAULT_CONFIGS

    datasets = list(dataset) if dataset else None

    console.print("\n[bold]Ablation Study[/bold]\n")
    console.print(f"  Configs:          {', '.join(c.label for c in DEFAULT_CONFIGS)}")
    console.print(f"  Datasets:         {', '.join(datasets or DATASET_NAMES)}")
    console.print(f"  Queries per ds:   {max_per_dataset}")
    console.print(f"  Retrieval top-k:  {top_k}")
    console.print()

    try:
        queries = load_benchmark(
            datasets=datasets,
            mini=True,
            max_per_dataset=max_per_dataset,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run [bold]legalkit bench --download[/bold] to get the dataset.")
        raise SystemExit(1)

    console.print(f"  Loaded {len(queries)} queries across "
                  f"{len(set(s.file_path for q in queries for s in q.snippets))} corpus files\n")

    def progress(stage: str, current: int, total: int) -> None:
        if "ingesting" in stage and current == 0:
            console.print(f"  [dim]{stage}...[/dim]")
        elif current == total and total > 0 and "querying" in stage:
            console.print(f"    {stage}: {current}/{total} ✓")
        elif current % 20 == 0 and current > 0:
            console.print(f"    {stage}: {current}/{total}")

    import time as _time
    t0 = _time.time()

    abl = run_ablation(
        queries=queries,
        max_chunks_per_query=top_k,
        progress_callback=progress,
    )

    total_elapsed = _time.time() - t0

    # ── Recall@k table ────────────────────────────────────────────────────
    console.print(f"\n[bold]Recall@k Results[/bold]\n")

    ks = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= top_k]
    best_k = ks[-1]
    best_score = max(cr.recall_at_k(best_k) for cr in abl.config_results)

    from rich.table import Table as RichTable
    table = RichTable(show_lines=True)
    table.add_column("Config", style="cyan", width=16)
    table.add_column("Description", width=42)
    for k in ks:
        table.add_column(f"R@{k}", justify="right", width=7)
    table.add_column("Time", justify="right", width=7)

    for cr in abl.config_results:
        is_best = cr.recall_at_k(best_k) == best_score
        label = f"[bold]{cr.config.label}[/bold]" if is_best else cr.config.label
        vals = [f"{cr.recall_at_k(k)*100:.2f}" for k in ks]
        table.add_row(label, cr.config.description, *vals, f"{cr.elapsed_seconds:.0f}s")

    console.print(table)

    # ── Marginal contribution table ───────────────────────────────────────
    console.print(f"\n[bold]Marginal Contribution at R@{best_k}[/bold]\n")

    delta_table = RichTable(show_lines=False, box=None, padding=(0, 2))
    delta_table.add_column("Config", style="cyan", width=16)
    delta_table.add_column(f"R@{best_k}", justify="right", width=8)
    delta_table.add_column("Δ vs prev", justify="right", width=10)
    delta_table.add_column("Description", style="dim")

    for d in abl.component_deltas(k=best_k):
        sign = "+" if d["delta_pp"] >= 0 else ""
        delta_style = "green" if d["delta_pp"] > 0.005 else "yellow" if d["delta_pp"] >= 0 else "red"
        delta_str = f"[{delta_style}]{sign}{d['delta_pp']*100:.2f}pp[/{delta_style}]"
        delta_table.add_row(
            d["label"],
            f"{d['recall']*100:.2f}%",
            delta_str,
            d["description"],
        )

    console.print(delta_table)

    # ── Per-dataset breakdown ─────────────────────────────────────────────
    tags = abl.config_results[0].results.tags if abl.config_results else []
    if tags and len(tags) > 1:
        console.print(f"\n[bold]Per-dataset R@{best_k}[/bold]\n")

        ds_table = RichTable(show_lines=True)
        ds_table.add_column("Dataset", style="cyan", width=16)
        for cr in abl.config_results:
            ds_table.add_column(cr.config.label, justify="right", width=12)

        for tag in tags:
            row = [tag] + [f"{cr.recall_at_k(best_k, tag)*100:.2f}%" for cr in abl.config_results]
            ds_table.add_row(*row)

        console.print(ds_table)

    best = abl.best_config(k=best_k)
    console.print(
        f"\n[green]Best config at R@{best_k}:[/green] [bold]{best.config.label}[/bold]  "
        f"({best.recall_at_k(best_k)*100:.2f}%)"
    )
    console.print(f"Total time: {total_elapsed:.0f}s\n")

    # ── JSON output ───────────────────────────────────────────────────────
    if output:
        import json, datetime
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "top_k": top_k,
            "queries_evaluated": abl.queries_evaluated,
            "configs": [
                {
                    "label": cr.config.label,
                    "retrieval_method": cr.config.retrieval_method,
                    "use_cross_references": cr.config.use_cross_references,
                    "description": cr.config.description,
                    "elapsed_seconds": round(cr.elapsed_seconds, 2),
                    "overall": {
                        f"recall@{k}": round(cr.recall_at_k(k), 4) for k in ks
                    },
                    "per_dataset": {
                        tag: {f"recall@{k}": round(cr.recall_at_k(k, tag), 4) for k in ks}
                        for tag in tags
                    },
                }
                for cr in abl.config_results
            ],
            "marginal_deltas": {
                f"@{best_k}": abl.component_deltas(k=best_k),
            },
        }
        output.write_text(json.dumps(data, indent=2))
        console.print(f"[dim]Results saved to {output}[/dim]")


@main.command("chunk-sweep")
@click.option("--chunk-sizes", "-c", multiple=True, type=int,
              help="max_chars values to sweep, e.g. -c 500 -c 800 -c 1500 -c 3000. "
                   "Defaults to [500, 800, 1500, 3000].")
@click.option("--min-chars", default=100, show_default=True,
              help="Minimum chunk size in characters (applied to all configs).")
@click.option("--dataset", "-d", multiple=True,
              help="Specific dataset(s): privacy_qa, contractnli, maud, cuad")
@click.option("--max-per-dataset", "-n", type=int, default=50, show_default=True,
              help="Max queries per dataset (keep small for fast iteration).")
@click.option("--top-k", default=64, show_default=True,
              help="Retrieval depth for each run.")
@click.option("--metric", type=click.Choice(["recall", "precision"]), default="recall",
              show_default=True, help="Primary metric for comparison table.")
@click.option("--output", "-o", type=click.Path(path_type=Path),
              help="Save full results as JSON.")
def chunk_sweep(
    chunk_sizes: tuple,
    min_chars: int,
    dataset: tuple,
    max_per_dataset: int,
    top_k: int,
    metric: str,
    output: Path | None,
):
    """Sweep chunk sizes and compare retrieval metrics.

    Tests multiple max_chars configurations against LegalBench-RAG to find
    the optimal chunk size. Smaller chunks (500–800 chars) can improve recall
    on precise legal clauses; larger chunks (1500–3000) preserve more context.
    """
    from legalkit.eval.benchmark import load_benchmark, DATASET_NAMES
    from legalkit.eval.chunk_sweep import ChunkConfig, run_chunk_sweep

    sizes = list(chunk_sizes) if chunk_sizes else [500, 800, 1500, 3000]
    datasets = list(dataset) if dataset else None
    configs = [ChunkConfig(min_chars=min_chars, max_chars=mx) for mx in sizes]

    console.print("\n[bold]Chunk Size Sweep[/bold]\n")
    console.print(f"  Configs:          {', '.join(c.label for c in configs)}")
    console.print(f"  Datasets:         {', '.join(datasets or DATASET_NAMES)}")
    console.print(f"  Queries per ds:   {max_per_dataset}")
    console.print(f"  Retrieval top-k:  {top_k}")
    console.print(f"  Primary metric:   {metric}")
    console.print()

    try:
        queries = load_benchmark(
            datasets=datasets,
            mini=True,
            max_per_dataset=max_per_dataset,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run [bold]legalkit bench --download[/bold] to get the dataset.")
        raise SystemExit(1)

    console.print(f"  Loaded {len(queries)} queries across "
                  f"{len(set(s.file_path for q in queries for s in q.snippets))} corpus files\n")

    current_config = {"label": ""}

    def progress(stage: str, current: int, total: int) -> None:
        if stage.startswith("config ") and current == 0:
            return
        if stage != current_config["label"] and stage.startswith("["):
            current_config["label"] = stage
        if "ingesting" in stage and current == 0:
            console.print(f"  [dim]{stage}...[/dim]")
        elif current == total and total > 0:
            console.print(f"    {stage}: {current}/{total} ✓")
        elif current % 20 == 0 and current > 0:
            console.print(f"    {stage}: {current}/{total}")

    import time as _time
    t0 = _time.time()

    sweep = run_chunk_sweep(
        queries=queries,
        configs=configs,
        max_chunks_per_query=top_k,
        progress_callback=progress,
    )

    total_elapsed = _time.time() - t0

    # ── Results table ─────────────────────────────────────────────────────
    console.print(f"\n[bold]Results — {metric.capitalize()}@k (%)[/bold]\n")

    from rich.table import Table

    ks = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= top_k]
    fn = "recall_at_k" if metric == "recall" else "precision_at_k"
    best_k = ks[-1]
    best_score = max(getattr(cr, fn)(best_k) for cr in sweep.config_results)

    table = Table(show_lines=True)
    table.add_column("Config (chars)", style="cyan", width=18)
    for k in ks:
        table.add_column(f"{'R' if metric=='recall' else 'P'}@{k}", justify="right", width=8)
    table.add_column("Time", justify="right", width=8)

    for cr in sweep.config_results:
        score_at_best_k = getattr(cr, fn)(best_k)
        is_best = score_at_best_k == best_score
        values = [f"{getattr(cr, fn)(k)*100:.2f}" for k in ks]
        label = f"[bold]{cr.config.label}[/bold]" if is_best else cr.config.label
        table.add_row(label, *values, f"{cr.elapsed_seconds:.0f}s")

    console.print(table)

    # ── Delta vs baseline (3000) ──────────────────────────────────────────
    if any(cr.config.max_chars == 3000 for cr in sweep.config_results):
        console.print(f"\n[dim]{sweep.delta_table(baseline_max_chars=3000, k=best_k)}[/dim]\n")

    # ── Best config callout ───────────────────────────────────────────────
    best = sweep.best_config(k=best_k, metric=metric)
    console.print(
        f"[green]Best config at {metric}@{best_k}:[/green] "
        f"[bold]{best.config.label}[/bold]  "
        f"({getattr(best, fn)(best_k)*100:.2f}%)"
    )
    console.print(f"\nTotal time: {total_elapsed:.0f}s\n")

    # ── Per-dataset breakdown ─────────────────────────────────────────────
    tags = sweep.config_results[0].results.tags if sweep.config_results else []
    if tags and len(tags) > 1:
        console.print("[bold]Per-dataset breakdown[/bold]\n")
        for tag in tags:
            console.print(f"  [cyan]{tag}[/cyan]")
            for cr in sweep.config_results:
                score = getattr(cr, fn)(best_k, tag)
                console.print(f"    {cr.config.label:<18} {score*100:.2f}%")
        console.print()

    # ── JSON output ───────────────────────────────────────────────────────
    if output:
        import json, datetime
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metric": metric,
            "top_k": top_k,
            "queries_evaluated": sweep.queries_evaluated,
            "configs": [
                {
                    "label": cr.config.label,
                    "min_chars": cr.config.min_chars,
                    "max_chars": cr.config.max_chars,
                    "elapsed_seconds": round(cr.elapsed_seconds, 2),
                    "overall": {
                        f"{metric}@{k}": round(getattr(cr, fn)(k), 4) for k in ks
                    },
                    "per_dataset": {
                        tag: {
                            f"{metric}@{k}": round(getattr(cr, fn)(k, tag), 4) for k in ks
                        }
                        for tag in tags
                    },
                }
                for cr in sweep.config_results
            ],
        }
        output.write_text(json.dumps(data, indent=2))
        console.print(f"[dim]Results saved to {output}[/dim]")


@main.command()
@click.option("--balance", type=click.Choice(["downsample", "none"]), default="downsample")
@click.option("--max-per-class", type=int, help="Cap samples per class")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default="data/training")
@click.option("--hard-negatives", type=click.Path(path_type=Path),
              default="data/training/hard_negatives.jsonl", show_default=True,
              help="Path to hard_negatives.jsonl (generated by 'legalkit generate-hard-negatives').")
@click.option("--cross-passage-cap", type=float, default=0.15, show_default=True,
              help="Max fraction of neutral class that may be cross-passage samples (0.0–1.0).")
@click.option("--no-hard-negatives", is_flag=True, default=False,
              help="Skip hard negatives even if the file exists.")
def prepare_training(
    balance: str,
    max_per_class: int | None,
    output_dir: Path,
    hard_negatives: Path,
    cross_passage_cap: float,
    no_hard_negatives: bool,
):
    """Prepare balanced train/val/test splits from collected entailment data.

    If data/training/hard_negatives.jsonl exists (generated by
    'legalkit generate-hard-negatives'), it is merged in automatically
    and cross-passage neutrals are capped to reduce training noise.

    Recommended workflow:
      legalkit generate-hard-negatives
      legalkit prepare-training
    """
    from legalkit.eval.prepare_training import prepare_dataset

    hn_path: Path | None = None
    if not no_hard_negatives and hard_negatives.exists():
        hn_path = hard_negatives
        console.print(f"\n  [dim]Hard negatives: {hard_negatives} (cross-passage cap: {cross_passage_cap:.0%})[/dim]")

    console.print("\n[bold]Preparing Training Data[/bold]\n")

    stats = prepare_dataset(
        output_dir=output_dir,
        balance_strategy=balance,
        max_samples_per_class=max_per_class,
        hard_negatives_path=hn_path,
        cross_passage_cap=cross_passage_cap,
    )

    console.print(f"  Raw samples: {stats['total_raw']}")
    console.print(f"  Balanced: {stats['total_balanced']} ({stats['balance_strategy']}, {stats['target_per_class']}/class)")
    console.print(f"  Train: {stats['train']}")
    console.print(f"  Val: {stats['val']}")
    console.print(f"  Test: {stats['test']}")
    console.print(f"\n  Train distribution: {stats['train_distribution']}")
    if stats.get("source_distribution"):
        console.print(f"  Source distribution: {stats['source_distribution']}")
    console.print(f"  Output: {output_dir}")
    console.print()


@main.command("generate-hard-negatives")
@click.option("--output", "-o", type=click.Path(path_type=Path),
              default="data/training/hard_negatives.jsonl", show_default=True,
              help="Output path for hard negatives JSONL.")
@click.option("--max-from-log", type=int, default=None,
              help="Cap pairs loaded from entailment log (for testing).")
def generate_hard_negatives_cmd(output: Path, max_from_log: int | None):
    """Generate hard-negative training pairs from entailment log + curated catalog.

    Replaces the cross-passage random negatives that caused v0 fine-tuning
    failure. Applies pattern-based perturbations to existing entailed pairs
    and merges with the 96-example curated catalog.

    Run this before 'legalkit prepare-training' to improve training data quality:

      legalkit generate-hard-negatives

      legalkit prepare-training
    """
    from legalkit.eval.generate_hard_negatives import generate_hard_negatives

    console.print("\n[bold]Generating Hard-Negative Training Pairs[/bold]\n")

    # Progress tracking state
    progress_state = {"last_stage": "", "last_reported": 0}

    def _progress(stage: str, completed: int, total: int) -> None:
        if stage != progress_state["last_stage"]:
            progress_state["last_stage"] = stage
            progress_state["last_reported"] = 0
            if stage == "load":
                console.print(f"  Loading entailed pairs from log...")
            elif stage == "perturb":
                console.print(f"  Generating perturbation negatives...")
            elif stage == "catalog":
                console.print(f"  Loading curated catalog...")

        # Report every 50 items or on completion
        if completed == total or completed - progress_state["last_reported"] >= 50:
            console.print(f"    {stage}: {completed}/{total}")
            progress_state["last_reported"] = completed

    stats = generate_hard_negatives(
        output_path=output,
        max_from_log=max_from_log,
        progress_callback=_progress,
    )

    # Display stats with rich tables
    console.print(f"\n[green]✓ Generated {stats['total']} hard-negative training pairs[/green]")
    console.print(f"  Output: {stats['output_path']}\n")

    # By label table
    label_table = Table(title="By Label", show_header=True)
    label_table.add_column("Label", style="cyan")
    label_table.add_column("Count", style="magenta", justify="right")

    for label, count in sorted(stats["by_label"].items()):
        label_table.add_row(label, str(count))

    console.print(label_table)
    console.print()

    # By source table
    source_table = Table(title="By Source", show_header=True)
    source_table.add_column("Source", style="cyan")
    source_table.add_column("Count", style="magenta", justify="right")

    for source, count in sorted(stats["by_source"].items()):
        source_table.add_row(source, str(count))

    console.print(source_table)
    console.print()

    # Additional stats
    console.print(f"[dim]Input entailed pairs: {stats['input_entailed_pairs']}[/dim]")
    console.print(f"[dim]Perturbation-generated: {stats['perturbation_generated']}[/dim]")
    console.print(f"[dim]Catalog examples: {stats['catalog_examples']}[/dim]")


@main.command("structure-annotate")
@click.option("--corpus", "corpus_dirs", multiple=True,
              type=click.Path(exists=True, path_type=Path),
              help="Corpus directories to annotate. Defaults to all in data/corpus/.")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path),
              default="data/structure_model", show_default=True)
@click.option("--max-per-corpus", type=int, default=None)
def structure_annotate(corpus_dirs: tuple, output_dir: Path, max_per_corpus: int | None):
    """Generate BIO-labelled training data for the document structure model.
    
    Runs the regex parser over the contract corpus as a weak supervisor,
    producing line-level BIO labels for training a transformer-based
    structure detection model.
    
    Example:
      legalkit structure-annotate                          # all corpora
      legalkit structure-annotate --corpus data/corpus/cuad  # CUAD only
    """
    from legalkit.eval.structure_annotator import generate_training_data

    # Default to all corpora if none specified
    if not corpus_dirs:
        corpus_base = Path("data/corpus")
        if corpus_base.exists():
            corpus_dirs = tuple(
                p for p in corpus_base.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
        else:
            console.print("[red]Error:[/red] data/corpus/ not found and no --corpus specified")
            raise SystemExit(1)

    # Build corpus dict {contract_type: path}
    corpus_dict = {}
    for corpus_path in corpus_dirs:
        corpus_path = Path(corpus_path)
        contract_type = corpus_path.name
        corpus_dict[contract_type] = corpus_path

    console.print("\n[bold]Structure Annotation — BIO Labelling[/bold]\n")
    console.print(f"  Corpora: {', '.join(corpus_dict.keys())}")
    console.print(f"  Output: {output_dir}")
    if max_per_corpus:
        console.print(f"  Max per corpus: {max_per_corpus}")
    console.print()

    # Run annotation
    stats = generate_training_data(
        corpus_dirs=corpus_dict,
        output_path=output_dir,
        max_per_corpus=max_per_corpus,
    )

    # Display results with Rich
    console.print(f"\n[green]✓ Generated training data[/green]\n")

    # Per-corpus table
    corpus_table = Table(title="Per-Corpus Stats", show_header=True)
    corpus_table.add_column("Corpus", style="cyan")
    corpus_table.add_column("Documents", style="magenta", justify="right")

    for corpus_name, count in sorted(stats["corpus_stats"].items()):
        corpus_table.add_row(corpus_name, str(count))

    console.print(corpus_table)
    console.print()

    # Split sizes
    console.print(f"[bold]Split Sizes:[/bold]")
    console.print(f"  Train: {stats['split_sizes']['train']} documents")
    console.print(f"  Val:   {stats['split_sizes']['val']} documents")
    console.print(f"  Test:  {stats['split_sizes']['test']} documents")
    console.print(f"  Total: {stats['total_documents']} documents, {stats['total_lines']:,} lines")
    console.print()

    # Label distribution table
    label_dist = stats["train_stats"]["label_distribution"]
    total_labels = sum(label_dist.values())

    label_table = Table(title="Training Set Label Distribution", show_header=True)
    label_table.add_column("Label", style="cyan")
    label_table.add_column("Count", style="magenta", justify="right")
    label_table.add_column("Percentage", style="yellow", justify="right")

    for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
        pct = count / total_labels * 100 if total_labels > 0 else 0
        label_table.add_row(label, f"{count:,}", f"{pct:.2f}%")

    console.print(label_table)
    console.print()

    console.print(f"[dim]Files saved to {output_dir}/[/dim]")
    console.print(f"[dim]  - train.jsonl ({stats['train_stats']['total_lines']:,} lines)[/dim]")
    console.print(f"[dim]  - val.jsonl ({stats['val_stats']['total_lines']:,} lines)[/dim]")
    console.print(f"[dim]  - test.jsonl ({stats['test_stats']['total_lines']:,} lines)[/dim]")
    console.print(f"[dim]  - stats.json[/dim]")
    console.print()


@main.command('build-embedding-pairs')
@click.option('--max-per-dataset', '-n', type=int, default=None,
              help='Cap pairs per dataset. None = all available.')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              default='data/embedding_training', show_default=True)
def build_embedding_pairs(max_per_dataset, output_dir):
    """Build training pairs for embedding model fine-tuning.
    
    Extracts (query, positive, hard_negative) triples from the
    LegalBench-RAG benchmark ground-truth spans. Run this before
    kicking off the Modal embedding training job.
    
    Then train:
      modal run --detach scripts/modal_embedding_train.py
    """
    from legalkit.eval.embedding_pairs import build_training_data
    
    console.print("\n[bold]Building Embedding Training Pairs[/bold]\n")
    
    if max_per_dataset:
        console.print(f"  Max per dataset: {max_per_dataset}")
    console.print(f"  Output: {output_dir}\n")
    
    try:
        stats = build_training_data(
            datasets=None,  # all datasets
            output_dir=output_dir,
            max_per_dataset=max_per_dataset,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run [bold]legalkit bench --download[/bold] to get the benchmark data first.")
        raise SystemExit(1)
    
    # Display results with Rich
    console.print(f"[green]✓ Generated {stats['total']} training pairs[/green]\n")
    
    # Per-dataset table
    ds_table = Table(title="Pairs per Dataset", show_header=True)
    ds_table.add_column("Dataset", style="cyan")
    ds_table.add_column("Pairs", style="magenta", justify="right")
    
    for ds_name, count in sorted(stats['by_dataset'].items()):
        ds_table.add_row(ds_name, str(count))
    
    ds_table.add_row("[bold]Total[/bold]", f"[bold]{stats['total']}[/bold]")
    
    console.print(ds_table)
    console.print()
    
    # Split info
    console.print(f"[bold]Train/Val Split:[/bold]")
    console.print(f"  Train: {stats['train']} pairs")
    console.print(f"  Val:   {stats['val']} pairs")
    console.print()
    
    # Length stats
    console.print(f"[bold]Average Lengths:[/bold]")
    console.print(f"  Positive passages:   {stats['avg_positive_chars']:.0f} chars")
    console.print(f"  Hard negatives:      {stats['avg_hard_neg_chars']:.0f} chars")
    console.print()
    
    # Output files
    console.print(f"[dim]Files written:[/dim]")
    console.print(f"[dim]  {stats['train_path']}[/dim]")
    console.print(f"[dim]  {stats['val_path']}[/dim]")
    console.print()
    
    console.print("[bold]Next step:[/bold]")
    console.print("  uv run modal run --detach scripts/modal_embedding_train.py")
    console.print()


@main.command()
@click.option("--model", default="microsoft/deberta-v3-base", help="Base model to fine-tune")
@click.option("--epochs", default=3, help="Training epochs")
@click.option("--batch-size", default=16, help="Batch size")
@click.option("--lr", default=2e-5, type=float, help="Learning rate")
@click.option("--training-dir", type=click.Path(path_type=Path), default="data/training")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default="models/legal-entailment")
def finetune(model: str, epochs: int, batch_size: int, lr: float, training_dir: Path, output_dir: Path):
    """Fine-tune a DeBERTa model on legal entailment data."""
    from legalkit.eval.finetune import run_finetuning

    console.print(f"\n[bold]Fine-tuning Legal Entailment Model[/bold]\n")
    console.print(f"  Base model: {model}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Learning rate: {lr}")
    console.print(f"  Training data: {training_dir}")
    console.print(f"  Output: {output_dir}")
    console.print()

    results = run_finetuning(
        base_model=model,
        training_dir=training_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )

    console.print(f"\n[bold]Results[/bold]")
    test = results["test_metrics"]
    console.print(f"  Accuracy: {test['accuracy']:.4f}")
    console.print(f"  Macro F1: {test['macro_f1']:.4f}")
    console.print()


@main.command("structure-train")
@click.option("--model", default="lexlms/legal-roberta-base", show_default=True, help="Base model to fine-tune")
@click.option("--epochs", default=3, show_default=True, help="Training epochs")
@click.option("--batch-size", default=32, show_default=True, help="Batch size")
@click.option("--training-dir", type=click.Path(path_type=Path), default="data/structure_model", show_default=True)
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default="models/structure-parser", show_default=True)
@click.option("--modal", "use_modal", is_flag=True, help="Run on Modal GPU instead of local CPU.")
def structure_train(model: str, epochs: int, batch_size: int, training_dir: Path, output_dir: Path, use_modal: bool):
    """Fine-tune the document structure parser on BIO-labelled contract data.
    
    Trains legal-roberta-base to classify each line of a contract as a heading
    type (article, section, subsection, preamble, recital, exhibit, definition)
    or body text. The model generalises beyond the regex parser's rigid patterns.
    
    Generate training data first:
    
      legalkit structure-annotate
      legalkit structure-train
    
    Or train on Modal GPU (recommended for faster training):
    
      legalkit structure-train --modal
    """
    if use_modal:
        console.print("[yellow]⚠ Modal training not yet implemented. Use: modal run scripts/modal_structure_train.py[/yellow]")
        return
    
    from legalkit.eval.structure_model import run_structure_finetuning

    console.print(f"\n[bold]Fine-tuning Document Structure Model[/bold]\n")
    console.print(f"  Base model: {model}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Training data: {training_dir}")
    console.print(f"  Output: {output_dir}")
    console.print()

    try:
        results = run_structure_finetuning(
            base_model=model,
            training_dir=training_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
        )

        console.print(f"\n[bold]Results[/bold]")
        test = results["test_metrics"]
        console.print(f"  Accuracy: {test['accuracy']:.4f}")
        console.print(f"  Macro F1: {test['macro_f1']:.4f}")
        console.print(f"  Heading F1: {test['heading_f1']:.4f}")
        console.print()
        
        console.print("[dim]Key heading metrics:[/dim]")
        for label in ["B-ARTICLE_HEADER", "B-SECTION_HEADER", "B-SUBSECTION_HEADER"]:
            f1 = test.get(f"{label}_f1", 0)
            console.print(f"  {label}: F1={f1:.4f}")
    
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("\n[yellow]Run 'legalkit structure-annotate' first to generate training data.[/yellow]")
        raise SystemExit(1)


@main.command("nli-eval")
@click.option("--llm", "run_llm", is_flag=True, default=False,
              help="Include LLM-as-judge (requires API key, costs money).")
@click.option("--deberta-model", default="cross-encoder/nli-deberta-v3-base",
              show_default=True, help="DeBERTa model to evaluate.")
@click.option("--no-deberta", is_flag=True, help="Skip DeBERTa evaluation.")
@click.option("--finetuned-model", "finetuned_model_path", default=None,
              help="Path to fine-tuned model dir (e.g. models/legal-entailment/model).")
@click.option("--device", default="cpu", show_default=True,
              help="Device for DeBERTa: 'cpu' or 'cuda'.")
@click.option("--output", "-o", type=click.Path(path_type=Path),
              help="Save full results as JSON.")
@click.option("--cases-file", type=click.Path(path_type=Path),
              help="Override path to edge cases JSON.")
def nli_eval(
    run_llm: bool,
    deberta_model: str,
    no_deberta: bool,
    finetuned_model_path: str | None,
    device: str,
    output: Path | None,
    cases_file: Path | None,
):
    """Evaluate NLI models on curated legal edge cases.

    Compares DeBERTa-NLI (local, fast, free) against LLM-as-judge (accurate,
    slow, costs money) on 30 hand-labelled legal entailment pairs covering:
    notwithstanding, shall vs may, knowledge qualifiers, MAE carve-outs,
    quantitative precision, temporal qualifiers, conditions precedent, and
    defined terms.
    """
    from legalkit.eval.nli_eval import (
        load_edge_cases,
        run_deberta_nli,
        run_finetuned_nli,
        run_llm_judge,
        compare_results,
        DEBERTA_MODELS,
    )
    from rich.table import Table as RichTable

    # Load test cases
    try:
        cases = load_edge_cases(Path(cases_file) if cases_file else None)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    console.print(f"\n[bold]Legal NLI Edge Case Evaluation[/bold]\n")
    console.print(f"  Test cases:  {len(cases)}")
    console.print(f"  Categories:  {len({c.category for c in cases})}")
    console.print(f"  Labels:      {dict(sorted({c.label: sum(1 for x in cases if x.label == c.label) for c in cases}.items()))}")
    console.print()

    results = []

    # ── DeBERTa ──────────────────────────────────────────────────────
    if not no_deberta:
        resolved = DEBERTA_MODELS.get(deberta_model, deberta_model)
        console.print(f"  Running [bold]{resolved}[/bold]...")
        with console.status("Loading model and running inference..."):
            try:
                deberta_result = run_deberta_nli(
                    cases,
                    model_name=deberta_model,
                    device=device,
                    progress_callback=lambda c, t: None,
                )
                results.append(deberta_result)
                console.print(f"  ✓ Done ({deberta_result.avg_latency_ms:.0f}ms avg)")
            except Exception as e:
                console.print(f"  [yellow]⚠ DeBERTa failed: {e}[/yellow]")
        console.print()

    # ── Fine-tuned model ──────────────────────────────────────────────────
    if finetuned_model_path:
        fpath = Path(finetuned_model_path)
        if not fpath.exists():
            console.print(f"  [red]Fine-tuned model not found: {fpath}[/red]")
            console.print("  Run: modal volume get legalkit-data /models/legal-entailment/model models/legal-entailment/model")
        else:
            console.print(f"  Running [bold]fine-tuned legal DeBERTa-large[/bold] from {fpath}...")
            with console.status("Loading model and running inference..."):
                try:
                    ft_result = run_finetuned_nli(cases, model_path=str(fpath), device=device)
                    results.append(ft_result)
                    console.print(f"  ✓ Done ({ft_result.avg_latency_ms:.0f}ms avg)")
                except Exception as e:
                    console.print(f"  [yellow]⚠ Fine-tuned model failed: {e}[/yellow]")
            console.print()

    # ── LLM judge ────────────────────────────────────────────────────
    if run_llm:
        from legalkit.llm.client import LLMClient
        llm = LLMClient()
        console.print(f"  Running [bold]{llm.config.generation_model}[/bold] (LLM judge)...")

        completed_count = {"n": 0}
        def llm_progress(c, t):
            completed_count["n"] = c
            if c % 5 == 0 or c == t:
                console.print(f"    {c}/{t} cases judged")

        llm_result = run_llm_judge(cases, llm, progress_callback=llm_progress)
        results.append(llm_result)
        cost = llm_result.estimated_cost_usd
        console.print(f"  ✓ Done ({llm_result.avg_latency_ms:.0f}ms avg, est. ${cost:.4f})")
        console.print()

    if not results:
        console.print("[yellow]No models ran. Use --llm to include LLM judge, or remove --no-deberta.[/yellow]")
        raise SystemExit(0)

    # ── Results ───────────────────────────────────────────────────────
    console.print("[bold]Overall Results[/bold]\n")

    summary_table = RichTable(show_lines=True)
    summary_table.add_column("Model", style="cyan", min_width=40)
    summary_table.add_column("Binary Acc", justify="right", width=12)
    summary_table.add_column("4-class Acc", justify="right", width=12)
    summary_table.add_column("Avg Latency", justify="right", width=12)
    summary_table.add_column("Est. Cost", justify="right", width=12)

    for r in results:
        cost = r.estimated_cost_usd
        cost_str = f"${cost:.4f}" if cost is not None else "free (local)"
        lat_style = "green" if r.avg_latency_ms < 100 else "yellow" if r.avg_latency_ms < 1000 else "red"
        acc_style = "green" if r.accuracy_binary >= 0.80 else "yellow" if r.accuracy_binary >= 0.60 else "red"
        summary_table.add_row(
            r.model_name,
            f"[{acc_style}]{r.accuracy_binary:.1%}[/{acc_style}]",
            f"{r.accuracy_4class:.1%}",
            f"[{lat_style}]{r.avg_latency_ms:.0f}ms[/{lat_style}]",
            cost_str,
        )

    console.print(summary_table)

    # ── Per-category breakdown ────────────────────────────────────────
    console.print(f"\n[bold]Per-Category Binary Accuracy[/bold]\n")

    cat_table = RichTable(show_lines=True)
    cat_table.add_column("Category", style="cyan", width=24)
    cat_table.add_column("N", width=4)
    for r in results:
        short_name = r.model_name.split("/")[-1][:20]
        cat_table.add_column(short_name, justify="right", width=12)

    all_cats = sorted({c.category for c in cases})
    for cat in all_cats:
        row = [cat.replace("_", " ")]
        n = sum(1 for c in cases if c.category == cat)
        row.append(str(n))
        for r in results:
            cat_acc = r.per_category_accuracy.get(cat, {}).get("accuracy_binary", 0)
            style = "green" if cat_acc >= 0.8 else "yellow" if cat_acc >= 0.5 else "red"
            row.append(f"[{style}]{cat_acc:.0%}[/{style}]")
        cat_table.add_row(*row)

    console.print(cat_table)

    # ── Failure analysis with categories ─────────────────────────────
    if len(results) == 1:
        from legalkit.eval.false_positive import analyze_nli_failures

        result = results[0]
        fp_report = analyze_nli_failures(result)

        if fp_report.cases:
            console.print(f"\n[bold]Failure Analysis ({len(fp_report.cases)} failures)[/bold]\n")

            # Category breakdown
            if fp_report.top_categories:
                console.print("  [bold]By category:[/bold]")
                for cat, count in fp_report.top_categories:
                    pct = count / len(fp_report.false_positives) * 100 if fp_report.false_positives else 0
                    bar = "█" * count
                    console.print(
                        f"    {cat.value:<28}  {count:>2}  ({pct:.0f}%)  {bar}"
                    )
                console.print()

            # Individual failures
            fps = fp_report.false_positives
            fns = fp_report.false_negatives
            console.print(f"  False positives ({len(fps)}) — model said supported, ground truth: not supported:")
            for fc in fps[:10]:
                console.print(
                    f"    [red]{fc.case_id}[/red] [{fc.category.value}] "
                    f"predicted=[yellow]{fc.predicted_label}[/yellow] "
                    f"true=[green]{fc.true_label}[/green]"
                )
                if fc.explanation:
                    console.print(f"      [dim]{fc.explanation}[/dim]")
            if len(fps) > 10:
                console.print(f"    [dim]... and {len(fps) - 10} more[/dim]")

            if fns:
                console.print(f"\n  False negatives ({len(fns)}) — model said not supported, ground truth: supported:")
                for fc in fns[:5]:
                    console.print(
                        f"    [cyan]{fc.case_id}[/cyan] [{fc.category.value}] "
                        f"predicted=[yellow]{fc.predicted_label}[/yellow] "
                        f"true=[green]{fc.true_label}[/green]"
                    )

    # ── JSON output ───────────────────────────────────────────────────
    if output:
        import datetime
        comparison = compare_results(results)
        comparison["timestamp"] = datetime.datetime.now().isoformat()
        comparison["num_cases"] = len(cases)
        comparison["confusion_matrices"] = {
            r.model_name: r.confusion_matrix for r in results
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(comparison, indent=2))
        console.print(f"\n[dim]Results saved to {output}[/dim]")


@main.command("suite-stats")
@click.option("--path", type=click.Path(path_type=Path), help="Override suite file path.")
def suite_stats(path: Path | None):
    """Show statistics for the internal test suite."""
    from legalkit.eval.test_suite import load_suite, SuiteStats, DEFAULT_PATH

    suite_path = path or DEFAULT_PATH
    try:
        triples = load_suite(suite_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    stats = SuiteStats.from_triples(triples)
    console.print(f"\n[bold]Internal Test Suite[/bold]  ({suite_path})\n")
    console.print(stats.summary())

    # Show what needs review
    needs = [t for t in triples if t.needs_review]
    if needs:
        console.print(f"\n[yellow]Needs review ({len(needs)} triples):[/yellow]")
        for t in needs[:10]:
            console.print(f"  [{t.id}] [{t.metadata.difficulty}] {t.query[:70]}")
        if len(needs) > 10:
            console.print(f"  ... and {len(needs) - 10} more")
        console.print(f"\nRun [bold]legalkit annotate[/bold] to label them.")
    console.print()


@main.command("validate-suite")
@click.option("--path", type=click.Path(path_type=Path), help="Override suite file path.")
def validate_suite_cmd(path: Path | None):
    """Validate the internal test suite for schema compliance."""
    from legalkit.eval.test_suite import load_suite, validate_suite, DEFAULT_PATH

    suite_path = path or DEFAULT_PATH
    try:
        triples = load_suite(suite_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    result = validate_suite(triples)
    console.print(f"\n[bold]Suite Validation[/bold]  ({suite_path})\n")
    console.print(result.summary())

    if result.is_valid:
        console.print(f"\n[green]✓ Suite is valid ({len(triples)} triples)[/green]\n")
    else:
        console.print(f"\n[red]✗ Validation failed — fix errors before using for training[/red]\n")
        raise SystemExit(1)


@main.command("annotate")
@click.option("--path", type=click.Path(path_type=Path), help="Override suite file path.")
@click.option("--id", "triple_id", default=None, help="Annotate a specific triple by ID.")
@click.option("--status", type=click.Choice(["needs_review", "reviewed", "approved"]),
              default="needs_review", help="Filter by status to annotate.")
@click.option("--run-pipeline", is_flag=True, default=False,
              help="Run the pipeline on each query and show proposed answer.")
def annotate(path: Path | None, triple_id: str | None, status: str, run_pipeline: bool):
    """Interactively annotate test suite triples.

    Steps through triples with the given status, shows the query and
    proposed answer, and lets you confirm, edit, or skip each one.

    Saves progress after each triple so you can stop and resume.
    """
    from legalkit.eval.test_suite import (
        load_suite, save_suite, validate_suite, DEFAULT_PATH, TripleStatus,
    )
    import datetime

    suite_path = path or DEFAULT_PATH
    try:
        triples = load_suite(suite_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    # Filter triples to annotate
    if triple_id:
        to_annotate = [t for t in triples if t.id == triple_id]
        if not to_annotate:
            console.print(f"[red]Triple {triple_id!r} not found.[/red]")
            raise SystemExit(1)
    else:
        to_annotate = [t for t in triples if t.metadata.status == status]

    if not to_annotate:
        console.print(f"[dim]No triples with status={status!r} to annotate.[/dim]")
        return

    console.print(f"\n[bold]Annotation Tool[/bold]  ({len(to_annotate)} triples to review)\n")
    console.print("Commands: [y]es/confirm  [e]dit  [s]kip  [q]uit\n")

    reviewed_count = 0

    for i, triple in enumerate(to_annotate):
        console.print(f"[bold cyan]─── {triple.id} [{i+1}/{len(to_annotate)}] ───[/bold cyan]")
        console.print(f"  Difficulty:  {triple.metadata.difficulty}")
        console.print(f"  Query type:  {triple.metadata.query_type}")
        console.print(f"  Clause type: {triple.metadata.clause_type}")
        if triple.metadata.notes:
            console.print(f"  Notes:       [dim]{triple.metadata.notes}[/dim]")
        console.print()
        console.print(f"  [bold]Query:[/bold] {triple.query}")
        console.print()

        if triple.ground_truth.answer:
            console.print(f"  [bold]Proposed answer:[/bold]")
            console.print(f"  {triple.ground_truth.answer}")
            console.print()
            if triple.ground_truth.citations:
                console.print(f"  [bold]Citations:[/bold]")
                for c in triple.ground_truth.citations:
                    console.print(f"    [{c.section_id}] ({c.entailment}) {c.quote[:80]}...")
            console.print()
        else:
            console.print("  [dim](No proposed answer — needs to be written)[/dim]\n")

        action = click.prompt(
            "  Action",
            type=click.Choice(["y", "e", "s", "q"], case_sensitive=False),
            default="s",
            show_choices=True,
        )

        if action == "q":
            console.print("\n[dim]Saved and exiting.[/dim]")
            break
        elif action == "s":
            continue
        elif action == "y":
            # Mark as reviewed/approved
            new_status = click.prompt(
                "  Mark as",
                type=click.Choice(["reviewed", "approved"], case_sensitive=False),
                default="reviewed",
            )
            triple.metadata.status = new_status
            triple.metadata.reviewer = "manual"
            triple.metadata.reviewed_at = datetime.datetime.now().isoformat()[:10]
            save_suite(triples, suite_path)
            reviewed_count += 1
            console.print(f"  [green]✓ Marked as {new_status}[/green]\n")
        elif action == "e":
            # Edit answer
            console.print("  Enter new answer (press Enter twice when done):")
            lines = []
            while True:
                line = input("  > ")
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            new_answer = "\n".join(lines[:-1] if lines and lines[-1] == "" else lines).strip()
            if new_answer:
                triple.ground_truth.answer = new_answer

            # Edit citations
            if click.confirm("  Edit citations?", default=False):
                console.print("  Enter section IDs (comma-separated, e.g. §2.1,§8.3(b)):")
                section_ids_str = input("  > ").strip()
                if section_ids_str:
                    from legalkit.eval.test_suite import CitationRef
                    triple.ground_truth.citations = [
                        CitationRef(section_id=sid.strip())
                        for sid in section_ids_str.split(",")
                        if sid.strip()
                    ]

            new_status = click.prompt(
                "  Mark as",
                type=click.Choice(["needs_review", "reviewed", "approved"], case_sensitive=False),
                default="reviewed",
            )
            triple.metadata.status = new_status
            triple.metadata.reviewer = "manual"
            triple.metadata.reviewed_at = datetime.datetime.now().isoformat()[:10]
            save_suite(triples, suite_path)
            reviewed_count += 1
            console.print(f"  [green]✓ Updated and marked as {new_status}[/green]\n")

    console.print(f"\n[bold]Done.[/bold] Reviewed {reviewed_count} triple(s).")
    console.print(f"Suite saved to [dim]{suite_path}[/dim]\n")


@main.command("alce-eval")
@click.option("--dataset", "-d", multiple=True,
              help="Datasets: privacy_qa, contractnli, maud, cuad")
@click.option("--n", default=5, show_default=True,
              help="Queries per dataset.")
@click.option("--top-k", default=20, show_default=True,
              help="Retrieval depth.")
@click.option("--output", "-o", type=click.Path(path_type=Path),
              help="Save results as JSON.")
def alce_eval(dataset: tuple, n: int, top_k: int, output: Path | None):
    """ALCE-style citation quality evaluation.

    Runs the full pipeline (retrieve → generate → verify) on benchmark
    queries and reports citation recall, citation precision, citation F1,
    answer string F1, and hallucination rate.

    Citation Recall  = fraction of claims with ≥1 verified citation
    Citation Precision = fraction of citations that support their claim
    Hallucination Rate = fraction of claims with NO verified citation

    Quick run (5 per dataset, ~4 API calls per query):
      legalkit alce-eval --n 5

    Per-dataset breakdown:
      legalkit alce-eval --dataset contractnli --dataset cuad --n 10
    """
    import time as _time
    from legalkit.eval.alce import run_alce_eval
    from legalkit.llm.client import LLMClient
    from rich.table import Table as RichTable

    datasets = list(dataset) if dataset else None

    console.print(f"\n[bold]ALCE Citation Quality Evaluation[/bold]\n")
    console.print(f"  Datasets:  {', '.join(datasets or ['contractnli', 'cuad', 'maud', 'privacy_qa'])}")
    console.print(f"  Queries:   {n}/dataset")
    console.print(f"  Top-k:     {top_k}")
    console.print()

    completed = {"n": 0}

    def progress(current: int, total: int, msg: str) -> None:
        completed["n"] = current
        if current == total:
            console.print(f"  ✓ {total} queries evaluated")
        elif current % 5 == 0 and current > 0:
            console.print(f"    [{current}/{total}] {msg[:60]}")
        elif current == 0 and msg:
            console.print(f"  [dim]Starting: {msg[:60]}...[/dim]")

    try:
        llm = LLMClient()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    t0 = _time.time()
    try:
        report = run_alce_eval(
            datasets=datasets,
            n_per_dataset=n,
            top_k=top_k,
            llm=llm,
            progress_callback=progress,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run [bold]legalkit bench --download[/bold] to get the dataset.")
        raise SystemExit(1)

    elapsed = _time.time() - t0

    # ── Overall metrics ───────────────────────────────────────────────
    console.print(f"\n[bold]Results[/bold]\n")

    def _fmt(v: float | None, good: float = 0.80) -> str:
        if v is None:
            return "[dim]N/A[/dim]"
        style = "green" if v >= good else "yellow" if v >= 0.60 else "red"
        return f"[{style}]{v:.1%}[/{style}]"

    summary_table = RichTable(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column(style="dim", width=28)
    summary_table.add_column(width=10)
    summary_table.add_column(style="dim")

    summary_table.add_row(
        "Citation Recall", _fmt(report.avg_citation_recall),
        "claims with ≥1 verified citation",
    )
    summary_table.add_row(
        "Citation Precision", _fmt(report.avg_citation_precision),
        "citations that actually support claims",
    )
    summary_table.add_row(
        "Citation F1", _fmt(report.avg_citation_f1),
        "harmonic mean",
    )
    summary_table.add_row(
        "Hallucination Rate", _fmt(report.avg_hallucination_rate, good=0.15),
        "claims with no verified citation",
    )
    if report.avg_answer_string_f1 is not None:
        summary_table.add_row(
            "Answer String F1", _fmt(report.avg_answer_string_f1, good=0.40),
            "token overlap vs ground truth",
        )
    summary_table.add_row("", "", "")
    summary_table.add_row(
        "Queries evaluated", str(len(report.results)), "",
    )
    summary_table.add_row("Total claims", str(report.total_claims), "")
    summary_table.add_row("Total citations", str(report.total_citations), "")
    summary_table.add_row("Time", f"{elapsed:.0f}s", "")

    console.print(summary_table)

    # ── Per-dataset breakdown ─────────────────────────────────────────
    per_ds = report.per_dataset()
    if len(per_ds) > 1:
        console.print(f"\n[bold]Per-Dataset[/bold]\n")
        ds_table = RichTable(show_lines=True)
        ds_table.add_column("Dataset", style="cyan", width=16)
        ds_table.add_column("Cit-R", justify="right", width=8)
        ds_table.add_column("Cit-P", justify="right", width=8)
        ds_table.add_column("Cit-F1", justify="right", width=8)
        ds_table.add_column("Halluc", justify="right", width=8)
        ds_table.add_column("Str-F1", justify="right", width=8)
        ds_table.add_column("N", justify="right", width=4)

        for ds_name, ds_report in sorted(per_ds.items()):
            sf1 = ds_report.avg_answer_string_f1
            ds_table.add_row(
                ds_name,
                f"{ds_report.avg_citation_recall:.1%}",
                f"{ds_report.avg_citation_precision:.1%}",
                f"{ds_report.avg_citation_f1:.1%}",
                f"{ds_report.avg_hallucination_rate:.1%}",
                f"{sf1:.1%}" if sf1 is not None else "—",
                str(len(ds_report.results)),
            )

        console.print(ds_table)

    # ── JSON output ───────────────────────────────────────────────────
    if output:
        import datetime
        output.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()
        data["timestamp"] = datetime.datetime.now().isoformat()
        data["elapsed_seconds"] = round(elapsed, 2)
        output.write_text(json.dumps(data, indent=2))
        console.print(f"\n[dim]Results saved to {output}[/dim]")


@main.command("cuad-eval")
@click.option("--max-per-clause", "-n", type=int, default=None,
              help="Max queries per clause type (omit for full run).")
@click.option("--max-total", type=int, default=None,
              help="Cap total queries across all types.")
@click.option("--top-k", default=64, show_default=True,
              help="Retrieval depth.")
@click.option("--output", "-o", type=click.Path(path_type=Path),
              help="Save results as JSON.")
def cuad_eval(
    max_per_clause: int | None,
    max_total: int | None,
    top_k: int,
    output: Path | None,
):
    """Evaluate retrieval accuracy by CUAD clause type.

    Runs the CUAD clause QA benchmark and reports Recall@k broken down
    across 35 clause categories — governing law, expiration date,
    non-compete, IP assignment, liability cap, audit rights, etc.

    This answers: which clause types can the pipeline reliably find?

    Quick run (5 per clause, ~175 queries):
      legalkit cuad-eval --max-per-clause 5

    Full CUAD evaluation (4,042 queries):
      legalkit cuad-eval
    """
    import time as _time
    from legalkit.eval.cuad_qa import run_cuad_eval

    n_str = f"{max_per_clause}/clause" if max_per_clause else "all"
    console.print(f"\n[bold]CUAD Clause QA Evaluation[/bold]\n")
    console.print(f"  Queries:    {n_str}")
    console.print(f"  Top-k:      {top_k}")
    console.print()

    def progress(stage: str, current: int, total: int) -> None:
        if current == 0:
            console.print(f"  [dim]{stage}[/dim]")
        elif "querying" in stage and (current % 50 == 0 or current == total):
            console.print(f"    {stage}: {current}/{total}")
        elif current == total and total > 0:
            console.print(f"    {stage}: {current}/{total} ✓")

    t0 = _time.time()
    try:
        report = run_cuad_eval(
            max_per_clause=max_per_clause,
            max_total=max_total,
            top_k=top_k,
            progress_callback=progress,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run [bold]legalkit bench --download[/bold] to get the dataset.")
        raise SystemExit(1)

    elapsed = _time.time() - t0

    # ── Results table ─────────────────────────────────────────────────
    console.print(f"\n[bold]Per-Clause-Type Recall@k (%)[/bold]\n")

    from rich.table import Table as RichTable
    ks = [k for k in [1, 4, 16, 64] if k <= top_k]

    table = RichTable(show_lines=True)
    table.add_column("Clause Type", style="cyan", width=28)
    for k in ks:
        table.add_column(f"R@{k}", justify="right", width=7)
    table.add_column("N", justify="right", width=5)

    for cr in report.sorted_by_recall:
        vals = [f"{cr.recall_at_k(k)*100:.1f}" for k in ks]
        r_top = cr.recall_at_k(ks[-1])
        style = "green" if r_top >= 0.95 else "yellow" if r_top >= 0.80 else "red"
        table.add_row(
            f"[{style}]{cr.clause_type}[/{style}]",
            *vals, str(cr.n_queries),
        )

    # Overall row
    overall_vals = [f"[bold]{report.recall_at_k(k)*100:.1f}[/bold]" for k in ks]
    n_total = sum(cr.n_queries for cr in report.clause_results.values())
    table.add_row("[bold]OVERALL[/bold]", *overall_vals, f"[bold]{n_total}[/bold]")

    console.print(table)

    # ── Hardest clauses callout ───────────────────────────────────────
    hardest = report.sorted_by_difficulty[:5]
    if hardest:
        console.print(f"\n[bold]Hardest clauses (lowest R@{ks[-1]})[/bold]")
        for cr in hardest:
            console.print(
                f"  [red]{cr.clause_type:<28}[/red]  "
                f"R@{ks[-1]}={cr.recall_at_k(ks[-1])*100:.1f}%  "
                f"(n={cr.n_queries})"
            )

    # ── Overall ───────────────────────────────────────────────────────
    console.print(
        f"\n  Overall R@{ks[-1]}: [bold]{report.recall_at_k(ks[-1])*100:.2f}%[/bold]"
        f"  ({n_total} queries, {elapsed:.0f}s)"
    )
    console.print()

    # ── JSON output ───────────────────────────────────────────────────
    if output:
        import datetime
        output.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict(ks=ks)
        data["timestamp"] = datetime.datetime.now().isoformat()
        data["elapsed_seconds"] = round(elapsed, 2)
        output.write_text(json.dumps(data, indent=2))
        console.print(f"[dim]Results saved to {output}[/dim]")


@main.command("fp-analysis")
@click.option("--source", type=click.Choice(["nli-edge-cases", "entailment-log", "both"]),
              default="nli-edge-cases", show_default=True,
              help="Where to get failures from.")
@click.option("--deberta-model", default="cross-encoder/nli-deberta-v3-base",
              show_default=True, help="DeBERTa model for nli-edge-cases source.")
@click.option("--threshold", default=0.7, show_default=True,
              help="Confidence threshold for entailment-log scan.")
@click.option("--reviewed-only", is_flag=True,
              help="For entailment-log: only analyze human-reviewed samples.")
@click.option("--output", "-o", type=click.Path(path_type=Path),
              help="Save full report as JSON.")
def fp_analysis(
    source: str,
    deberta_model: str,
    threshold: float,
    reviewed_only: bool,
    output: Path | None,
):
    """Categorize verification false positives by failure mode.

    Identifies why the model incorrectly marks claims as supported, using
    a taxonomy of eight legal NLI failure patterns:

    \b
    surface_form_agreement  — topic match hides logical contradiction
    modal_confusion         — shall vs may, right vs obligation
    qualifier_omission      — knowledge/materiality qualifier dropped
    carveout_missed         — exception or notwithstanding ignored
    quantitative_imprecision — up-to vs exact, deductible inversion
    condition_stripped      — condition precedent omitted
    scope_overbroad         — claim generalizes beyond passage scope
    negation_missed         — shall not / excluded / does not constitute

    Outputs category distribution, per-category examples, and training
    data recommendations for reducing each failure mode.
    """
    from legalkit.eval.false_positive import (
        analyze_nli_failures,
        analyze_entailment_log,
        FalsePositiveReport,
    )
    from rich.table import Table as RichTable

    reports: list[FalsePositiveReport] = []

    # ── NLI edge cases ────────────────────────────────────────────────
    if source in ("nli-edge-cases", "both"):
        from legalkit.eval.nli_eval import load_edge_cases, run_deberta_nli
        console.print(f"\n  [bold]Source: NLI edge cases[/bold]")
        try:
            cases = load_edge_cases()
        except FileNotFoundError as e:
            console.print(f"  [red]Error:[/red] {e}")
            raise SystemExit(1)

        with console.status(f"Running {deberta_model}..."):
            try:
                nli_result = run_deberta_nli(cases, model_name=deberta_model)
                report = analyze_nli_failures(nli_result)
                reports.append(report)
                console.print(f"  ✓ {len(report.cases)} failures found in {len(cases)} cases")
            except Exception as e:
                console.print(f"  [yellow]⚠ Failed: {e}[/yellow]")

    # ── Entailment log ────────────────────────────────────────────────
    if source in ("entailment-log", "both"):
        console.print(f"\n  [bold]Source: Entailment log[/bold]")
        report = analyze_entailment_log(threshold=threshold, reviewed_only=reviewed_only)
        if not report.cases:
            console.print("  [dim]No flagged samples found.[/dim]")
        else:
            reports.append(report)
            console.print(f"  ✓ {len(report.cases)} suspicious samples flagged")

    if not reports:
        console.print("\n[yellow]No failure data available.[/yellow]")
        raise SystemExit(0)

    # ── Display ───────────────────────────────────────────────────────
    for report in reports:
        console.print(f"\n[bold]False Positive Report — {report.source}[/bold]\n")

        fps = report.false_positives
        fns = report.false_negatives

        console.print(
            f"  Total failures: {len(report.cases)}  |  "
            f"False positives: {len(fps)}  |  False negatives: {len(fns)}"
        )
        console.print()

        if not fps:
            console.print("  [green]No false positives found.[/green]")
            continue

        # Category distribution table
        cat_table = RichTable(show_lines=True, title="Failure Category Distribution")
        cat_table.add_column("Category", style="cyan", width=28)
        cat_table.add_column("N", justify="right", width=4)
        cat_table.add_column("%", justify="right", width=6)
        cat_table.add_column("Description", width=55)

        for cat, count in report.top_categories:
            pct = count / len(fps) * 100
            from legalkit.eval.false_positive import _CATEGORY_DESCRIPTIONS
            desc = _CATEGORY_DESCRIPTIONS[cat][:53]
            style = "red" if pct > 30 else "yellow" if pct > 15 else "white"
            cat_table.add_row(
                f"[{style}]{cat.value}[/{style}]",
                str(count), f"{pct:.0f}%", desc,
            )

        console.print(cat_table)

        # Per-category examples
        console.print(f"\n[bold]Examples by Category[/bold]\n")
        for cat, count in report.top_categories[:5]:
            examples = report.examples_for(cat, n=2)
            if not examples:
                continue
            from legalkit.eval.false_positive import _CATEGORY_RECOMMENDATIONS
            console.print(f"  [cyan]{cat.value}[/cyan] ({count} cases)")
            console.print(f"  [dim]→ Fix: {_CATEGORY_RECOMMENDATIONS[cat]}[/dim]")
            for ex in examples:
                console.print(f"    [{ex.case_id}] predicted={ex.predicted_label} true={ex.true_label}")
                if ex.explanation:
                    console.print(f"      [dim]{ex.explanation[:120]}[/dim]")
            console.print()

    # ── JSON output ───────────────────────────────────────────────────
    if output:
        import datetime
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "reports": [r.to_dict() for r in reports],
        }
        output.write_text(json.dumps(data, indent=2))
        console.print(f"[dim]Results saved to {output}[/dim]")


def _to_binary(label: str) -> str:
    from legalkit.eval.nli_eval import _BINARY_POSITIVE
    return "positive" if label in _BINARY_POSITIVE else "negative"


@main.command("calibrate")
@click.option("--source", type=click.Choice(["nli-edge-cases", "entailment-log", "both"]),
              default="nli-edge-cases", show_default=True,
              help="Where to get (score, label) pairs from.")
@click.option("--deberta-model", default="cross-encoder/nli-deberta-v3-base",
              show_default=True, help="DeBERTa model to use (for nli-edge-cases source).")
@click.option("--reviewed-only", is_flag=True,
              help="For entailment-log: only use human-reviewed samples.")
@click.option("--n-buckets", default=10, show_default=True,
              help="Number of confidence buckets for reliability diagram.")
@click.option("--output", "-o", type=click.Path(path_type=Path),
              help="Save calibration results as JSON.")
def calibrate(source: str, deberta_model: str, reviewed_only: bool,
              n_buckets: int, output: Path | None):
    """Measure confidence calibration of the entailment model.

    A well-calibrated model reports score=0.85 when it is actually correct
    85% of the time. This command computes Expected Calibration Error (ECE),
    Maximum Calibration Error (MCE), and Brier score, and displays a
    reliability diagram showing where confidence and accuracy diverge.

    Sources:
      nli-edge-cases  — 33 hand-labelled legal edge cases (ground truth known)
      entailment-log  — pipeline entailment log (LLM-generated labels)
      both            — combines both sources
    """
    from legalkit.eval.calibration import (
        compute_calibration, calibrate_entailment_log, CalibrationResult,
    )
    from legalkit.eval.nli_eval import (
        load_edge_cases, run_deberta_nli, _BINARY_POSITIVE,
    )

    results: dict[str, CalibrationResult] = {}

    # ── NLI edge cases ────────────────────────────────────────────────
    if source in ("nli-edge-cases", "both"):
        console.print(f"\n  [bold]Source: NLI edge cases[/bold]")
        try:
            cases = load_edge_cases()
        except FileNotFoundError as e:
            console.print(f"  [red]Error:[/red] {e}")
            raise SystemExit(1)

        console.print(f"  Loading {deberta_model}...")
        with console.status("Running DeBERTa inference..."):
            try:
                nli_result = run_deberta_nli(cases, model_name=deberta_model)
                cal = nli_result.calibration()
                results["nli_edge_cases"] = cal
                console.print(f"  ✓ {len(cases)} cases evaluated")
            except Exception as e:
                console.print(f"  [yellow]⚠ DeBERTa failed: {e}[/yellow]")

    # ── Entailment log ────────────────────────────────────────────────
    if source in ("entailment-log", "both"):
        console.print(f"\n  [bold]Source: Entailment log[/bold]")
        label_note = " (reviewed only)" if reviewed_only else " (all, incl. LLM-generated)"
        console.print(f"  Loading samples{label_note}...")

        cal = calibrate_entailment_log(reviewed_only=reviewed_only)
        if cal is None:
            label_msg = "reviewed " if reviewed_only else ""
            console.print(f"  [yellow]No {label_msg}entailment samples found.[/yellow]")
            console.print("  Run [bold]legalkit ask[/bold] or [bold]legalkit e2e[/bold] to collect samples.")
        else:
            results["entailment_log"] = cal
            console.print(f"  ✓ {cal.n_total} samples loaded")

    if not results:
        console.print("\n[yellow]No calibration data available.[/yellow]")
        raise SystemExit(0)

    # ── Display results ───────────────────────────────────────────────
    console.print(f"\n[bold]Calibration Results[/bold]\n")

    for source_name, cal in results.items():
        console.print(f"  [cyan]{source_name.replace('_', ' ').title()}[/cyan]")
        console.print(f"  {cal.summary()}")
        console.print()

        # Reliability diagram
        console.print(f"[dim]{cal.reliability_diagram_text()}[/dim]\n")

        # Interpretation
        if cal.ece < 0.05:
            console.print(f"  [green]✓ Well calibrated (ECE < 0.05)[/green]")
        elif cal.ece < 0.10:
            console.print(f"  [yellow]~ Moderate miscalibration (ECE 0.05–0.10)[/yellow]")
        else:
            console.print(f"  [red]✗ Significant miscalibration (ECE > 0.10)[/red]")

        if cal.is_overconfident:
            console.print(f"  [yellow]Model is overconfident — scores are higher than accuracy warrants[/yellow]")
        elif cal.is_underconfident:
            console.print(f"  [yellow]Model is underconfident — accuracy is better than scores suggest[/yellow]")
        console.print()

    # ── JSON output ───────────────────────────────────────────────────
    if output:
        import datetime
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": source,
            "n_buckets": n_buckets,
            "results": {
                name: {
                    "ece": cal.ece,
                    "mce": cal.mce,
                    "brier_score": cal.brier_score,
                    "n_total": cal.n_total,
                    "overconfident": cal.is_overconfident,
                    "underconfident": cal.is_underconfident,
                    "buckets": [
                        {
                            "low": b.confidence_low,
                            "high": b.confidence_high,
                            "n": b.n,
                            "avg_confidence": b.avg_confidence,
                            "accuracy": b.accuracy,
                            "gap": b.gap,
                        }
                        for b in cal.buckets if b.n > 0
                    ],
                }
                for name, cal in results.items()
            },
        }
        output.write_text(json.dumps(data, indent=2))
        console.print(f"[dim]Results saved to {output}[/dim]")


@main.command("legalbench")
@click.option(
    "--task-set", "-s",
    type=click.Choice(["contract", "cuad", "maud", "nda", "all"]),
    default="contract",
    show_default=True,
    help="Predefined task group to evaluate.",
)
@click.option(
    "--tasks", "-t", multiple=True,
    help="Explicit task names (overrides --task-set). Can repeat: -t cuad_governing_law -t contract_qa",
)
@click.option(
    "--n", default=20, show_default=True,
    help="Max examples per task (keep small for fast iteration).",
)
@click.option(
    "--split", default="test", show_default=True,
    help="Dataset split to evaluate: 'test' or 'train'.",
)
@click.option(
    "--compare-baselines", is_flag=True, default=True, show_default=True,
    help="Show GPT-4/Gemini baseline comparison table.",
)
@click.option(
    "--download", is_flag=True, default=False,
    help="Download and cache task data from HuggingFace, then exit.",
)
@click.option(
    "--list-tasks", "do_list_tasks", is_flag=True, default=False,
    help="List all registered tasks and exit.",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path),
    help="Save full results as JSON.",
)
def legalbench(
    task_set: str,
    tasks: tuple,
    n: int,
    split: str,
    compare_baselines: bool,
    download: bool,
    do_list_tasks: bool,
    output: Path | None,
):
    """Evaluate LLM reasoning on LegalBench (Stanford, 162 tasks).

    Tests the *generation/reasoning* step — given the right legal text,
    can the model answer the question correctly? This complements the
    LegalBench-RAG retrieval benchmark ('legalkit bench').

    Each task is a Yes/No or multi-choice classification problem drawn
    from real contract clauses (CUAD, MAUD, NDA, ContractNLI). Results
    are compared against published GPT-4 and Gemini Pro baselines from
    the paper (Guha et al., NeurIPS 2023).

    \b
    Quick run — contract tasks, 20 examples each:
      legalkit legalbench

    Full CUAD clause set, 50 examples each:
      legalkit legalbench --task-set cuad --n 50

    Specific tasks only:
      legalkit legalbench -t cuad_governing_law -t contract_qa --n 30

    Download data first (needed on first run without HF access):
      legalkit legalbench --download
    """
    import datetime
    import time as _time

    from legalkit.eval.legalbench import (
        TASK_SETS,
        _TASK_REGISTRY,
        download_tasks,
        list_tasks,
        run_legalbench_eval,
        task_count_by_category,
    )
    from rich.table import Table as RichTable

    # ── --list-tasks ──────────────────────────────────────────────────────
    if do_list_tasks:
        console.print(f"\n[bold]LegalBench Registered Tasks[/bold]\n")
        counts = task_count_by_category()
        for cat, cnt in sorted(counts.items()):
            console.print(f"  [cyan]{cat:<12}[/cyan]  {cnt} tasks")
        console.print(f"\n  [bold]Total:[/bold] {sum(counts.values())} tasks\n")

        all_meta = list_tasks("all")
        table = RichTable(show_lines=False, box=None, padding=(0, 2))
        table.add_column("Task", style="cyan", width=44)
        table.add_column("Category", width=10)
        table.add_column("Type", width=12)
        table.add_column("GPT-4", justify="right", width=7)
        for meta in all_meta:
            gpt4 = f"{meta.baseline_gpt4:.0%}" if meta.baseline_gpt4 else "—"
            table.add_row(meta.name, meta.category, meta.label_type, gpt4)
        console.print(table)
        console.print()
        return

    # ── --download ────────────────────────────────────────────────────────
    if download:
        task_list = list(tasks) if tasks else TASK_SETS.get(task_set, TASK_SETS["all"])
        console.print(f"\n[bold]Downloading LegalBench tasks ({len(task_list)} tasks)[/bold]\n")
        console.print(f"  Source: HuggingFace (nguyen-brat/legalbench)")
        console.print(f"  Cache:  data/legalbench/\n")

        results = {}

        def dl_progress(task_name: str, current: int, total: int) -> None:
            if task_name == "done":
                console.print(f"\n  [green]✓ Download complete.[/green]")
            else:
                status = "…" if current < total else "✓"
                console.print(f"  [{current+1:>3}/{total}] {status} {task_name}")

        results = download_tasks(task_list, split=split, progress_callback=dl_progress)

        ok = sum(1 for v in results.values() if v >= 0)
        failed = sum(1 for v in results.values() if v < 0)
        total_ex = sum(v for v in results.values() if v >= 0)
        console.print(f"\n  Cached: {ok} tasks  ({total_ex:,} examples total)")
        if failed:
            console.print(f"  [yellow]Failed: {failed} tasks[/yellow]")
            for name, v in results.items():
                if v < 0:
                    console.print(f"    [red]{name}[/red]")
        console.print()
        return

    # ── Run evaluation ─────────────────────────────────────────────────────
    explicit_tasks = list(tasks) if tasks else None
    effective_task_set = task_set if not explicit_tasks else None
    display_tasks = explicit_tasks or TASK_SETS.get(task_set, [])

    console.print(f"\n[bold]LegalBench Evaluation[/bold]  (Guha et al., NeurIPS 2023)\n")
    console.print(f"  Task set:       {task_set if not explicit_tasks else 'custom'}")
    console.print(f"  Tasks:          {len(display_tasks)}")
    console.print(f"  Examples/task:  {n}")
    console.print(f"  Split:          {split}")
    console.print()

    try:
        from legalkit.llm.client import LLMClient
        llm = LLMClient()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    console.print(f"  Model:          {llm.config.generation_model}\n")

    completed_tasks: dict[str, int] = {}

    def progress(task_name: str, current: int, total: int) -> None:
        if task_name == "done":
            console.print(f"  [green]✓ All {total} tasks complete.[/green]\n")
        elif current == 0 and task_name:
            pct = int(current / total * 100) if total > 0 else 0
            console.print(f"  [{current+1:>3}/{total}] {task_name} …")
        else:
            completed_tasks[task_name] = current

    t0 = _time.time()

    try:
        report = run_legalbench_eval(
            task_names=explicit_tasks,
            task_set=effective_task_set or task_set,
            max_per_task=n,
            split=split,
            llm=llm,
            progress_callback=progress,
        )
    except Exception as e:
        console.print(f"[red]Error during evaluation:[/red] {e}")
        raise SystemExit(1)

    elapsed = _time.time() - t0

    # ── Overall results ───────────────────────────────────────────────────
    def _acc_style(acc: float) -> str:
        if acc >= 0.85:
            return "green"
        if acc >= 0.70:
            return "yellow"
        return "red"

    console.print(f"[bold]Overall Results[/bold]\n")
    macro = report.overall_accuracy
    micro = report.micro_accuracy
    console.print(
        f"  Macro accuracy (avg over tasks):  "
        f"[{_acc_style(macro)}][bold]{macro:.1%}[/bold][/{_acc_style(macro)}]"
    )
    console.print(
        f"  Micro accuracy (weighted by size): "
        f"[{_acc_style(micro)}]{micro:.1%}[/{_acc_style(micro)}]"
    )
    console.print(f"  Tasks evaluated:  {len(report.successful_tasks)}")
    console.print(f"  Examples total:   {report.total_examples:,}")
    console.print(f"  Time:             {elapsed:.0f}s\n")

    # ── Per-category breakdown ─────────────────────────────────────────────
    cats = report.per_category()
    if len(cats) > 1:
        console.print(f"[bold]By Category[/bold]\n")
        cat_table = RichTable(show_lines=True)
        cat_table.add_column("Category", style="cyan", width=12)
        cat_table.add_column("Tasks", justify="right", width=6)
        cat_table.add_column("Examples", justify="right", width=10)
        cat_table.add_column("Macro Acc", justify="right", width=10)
        cat_table.add_column("Micro Acc", justify="right", width=10)

        for cat, stats in sorted(cats.items()):
            m_acc = stats["macro_accuracy"]
            u_acc = stats["micro_accuracy"]
            cat_table.add_row(
                cat,
                str(stats["n_tasks"]),
                str(stats["n_examples"]),
                f"[{_acc_style(m_acc)}]{m_acc:.1%}[/{_acc_style(m_acc)}]",
                f"{u_acc:.1%}",
            )
        console.print(cat_table)
        console.print()

    # ── Per-task results table ─────────────────────────────────────────────
    console.print(f"[bold]Per-Task Results[/bold]\n")
    task_table = RichTable(show_lines=True)
    task_table.add_column("Task", style="cyan", width=44)
    task_table.add_column("Cat", width=8)
    task_table.add_column("N", justify="right", width=5)
    task_table.add_column("Acc", justify="right", width=7)
    if compare_baselines:
        task_table.add_column("GPT-4", justify="right", width=7)
        task_table.add_column("Δ GPT-4", justify="right", width=8)

    for r in sorted(report.successful_tasks, key=lambda r: r.category):
        meta = _TASK_REGISTRY.get(r.task_name)
        acc_str = f"[{_acc_style(r.accuracy)}]{r.accuracy:.1%}[/{_acc_style(r.accuracy)}]"

        if compare_baselines and meta and meta.baseline_gpt4 is not None:
            delta = r.accuracy - meta.baseline_gpt4
            delta_str = (
                f"[green]+{delta:.1%}[/green]" if delta > 0
                else f"[red]{delta:.1%}[/red]" if delta < -0.05
                else f"[yellow]{delta:+.1%}[/yellow]"
            )
            task_table.add_row(
                r.task_name, r.category[:8], str(r.n_examples),
                acc_str, f"{meta.baseline_gpt4:.1%}", delta_str,
            )
        elif compare_baselines:
            task_table.add_row(
                r.task_name, r.category[:8], str(r.n_examples),
                acc_str, "—", "—",
            )
        else:
            task_table.add_row(r.task_name, r.category[:8], str(r.n_examples), acc_str)

    console.print(task_table)

    # ── Failed tasks ──────────────────────────────────────────────────────
    failed = [r for r in report.task_results if r.error]
    if failed:
        console.print(f"\n[yellow]⚠ {len(failed)} task(s) failed to load:[/yellow]")
        for r in failed[:10]:
            console.print(f"  [red]{r.task_name}[/red] — {r.error[:80]}")
        console.print(
            "\n  Run [bold]legalkit legalbench --download[/bold] "
            "to fetch missing task data."
        )

    # ── Best / hardest callouts ───────────────────────────────────────────
    successful = report.successful_tasks
    if len(successful) >= 3:
        best = report.best_tasks(k=3)
        hardest = report.hardest_tasks(k=3)
        console.print(f"\n[bold]Best tasks:[/bold]")
        for r in best:
            console.print(f"  [green]{r.task_name:<44}[/green] {r.accuracy:.1%}")
        console.print(f"\n[bold]Hardest tasks:[/bold]")
        for r in hardest:
            console.print(f"  [red]{r.task_name:<44}[/red] {r.accuracy:.1%}")

    console.print()

    # ── JSON output ───────────────────────────────────────────────────────
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()
        data["timestamp"] = datetime.datetime.now().isoformat()
        data["elapsed_seconds"] = round(elapsed, 2)
        data["n_per_task"] = n
        data["split"] = split
        output.write_text(json.dumps(data, indent=2))
        console.print(f"[dim]Results saved to {output}[/dim]\n")


@main.command("contractnli-eval")
@click.option("--split", default="test", show_default=True, help="Dataset split: test or train.")
@click.option("--n", default=50, show_default=True, help="Max examples to evaluate.")
@click.option("--top-k", default=10, show_default=True, help="Retrieval depth per hypothesis.")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Save results as JSON.")
@click.option("--download", is_flag=True, help="Download dataset from HuggingFace and exit.")
def contractnli_eval(split: str, n: int, top_k: int, output: Path | None, download: bool):
    """Evaluate retrieval + entailment on ContractNLI document-level NLI.

    For each NDA document + hypothesis pair, the pipeline retrieves
    relevant passages and classifies the hypothesis as Entailed,
    NotMentioned, or Contradicted. Results are compared against gold labels
    from Koreeda & Manning (EMNLP 2021).

    This tests end-to-end pipeline performance (retrieval accuracy AND
    entailment accuracy together), unlike bench (retrieval only) or
    nli-eval (entailment only).

    \b
    Quick run (50 examples):
      legalkit contractnli-eval --n 50

    Download dataset first:
      legalkit contractnli-eval --download
    """
    import datetime
    import time as _time
    from legalkit.eval.contractnli_doc import (
        load_contractnli,
        run_contractnli_eval,
        download_contractnli,
    )
    from rich.table import Table as RichTable

    # ── --download ────────────────────────────────────────────────────────
    if download:
        console.print(f"\n[bold]Downloading ContractNLI dataset[/bold]\n")
        console.print(f"  Source: HuggingFace (koreeda/ContractNLI)")
        console.print(f"  Cache:  data/contractnli/\n")

        try:
            count = download_contractnli(split=split)
            console.print(f"  [green]✓ Downloaded {count} examples for split '{split}'[/green]\n")
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}\n")
            raise SystemExit(1)
        return

    # ── Load examples ─────────────────────────────────────────────────────
    console.print(f"\n[bold]ContractNLI Document-Level NLI Evaluation[/bold]\n")
    console.print(f"  Split:   {split}")
    console.print(f"  Max n:   {n}")
    console.print(f"  Top-k:   {top_k}")
    console.print()

    try:
        examples = load_contractnli(split=split, max_n=n)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    console.print(f"  Loaded:  {len(examples)} examples")

    # Count unique documents
    unique_docs = len({ex.doc_id for ex in examples})
    console.print(f"  Docs:    {unique_docs} unique NDAs")
    console.print()

    # ── Run evaluation ────────────────────────────────────────────────────
    try:
        from legalkit.llm.client import LLMClient

        llm = LLMClient()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    console.print(f"  Model:   {llm.config.generation_model}\n")

    completed = {"n": 0}

    def progress(current: int, total: int) -> None:
        completed["n"] = current
        if current % 10 == 0 or current == total:
            console.print(f"    [{current}/{total}] hypotheses evaluated")

    t0 = _time.time()

    with console.status("Running pipeline..."):
        report = run_contractnli_eval(
            examples=examples,
            llm=llm,
            top_k=top_k,
            progress_callback=progress,
        )

    elapsed = _time.time() - t0

    # ── Results ───────────────────────────────────────────────────────────
    console.print(f"\n[bold]Results[/bold]\n")

    def _acc_style(acc: float) -> str:
        if acc >= 0.80:
            return "green"
        if acc >= 0.60:
            return "yellow"
        return "red"

    overall_acc = report.accuracy()
    console.print(
        f"  Overall accuracy:  [{_acc_style(overall_acc)}][bold]{overall_acc:.1%}[/bold][/{_acc_style(overall_acc)}]"
    )

    # Per-label accuracy
    per_label = report.per_label_accuracy()
    console.print(f"\n[bold]Per-Label Accuracy[/bold]\n")
    for label, acc in sorted(per_label.items()):
        console.print(f"  {label:<16} [{_acc_style(acc)}]{acc:.1%}[/{_acc_style(acc)}]")

    # Retrieval recall
    ret_recall = report.retrieval_recall()
    nm_precision = report.not_mentioned_precision()
    console.print(f"\n[bold]Retrieval Metrics[/bold]\n")
    console.print(
        f"  Retrieval recall (E/C):      [{_acc_style(ret_recall)}]{ret_recall:.1%}[/{_acc_style(ret_recall)}]"
    )
    console.print(
        f"  NotMentioned precision:      [{_acc_style(nm_precision)}]{nm_precision:.1%}[/{_acc_style(nm_precision)}]"
    )

    # Confusion matrix
    console.print(f"\n[bold]Confusion Matrix[/bold]\n")
    conf_matrix = report.confusion_matrix()

    conf_table = RichTable(show_lines=True)
    conf_table.add_column("True \\ Pred", style="cyan", width=14)
    for label in ["Entailed", "NotMentioned", "Contradicted"]:
        conf_table.add_column(label[:10], justify="right", width=10)

    for true_label in ["Entailed", "NotMentioned", "Contradicted"]:
        row = [true_label]
        for pred_label in ["Entailed", "NotMentioned", "Contradicted"]:
            count = conf_matrix[true_label][pred_label]
            if true_label == pred_label:
                row.append(f"[bold]{count}[/bold]")
            else:
                row.append(str(count))
        conf_table.add_row(*row)

    console.print(conf_table)

    console.print(f"\n  Time: {elapsed:.0f}s\n")

    # ── JSON output ───────────────────────────────────────────────────────
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()
        data["timestamp"] = datetime.datetime.now().isoformat()
        data["elapsed_seconds"] = round(elapsed, 2)
        data["n_examples"] = n
        data["split"] = split
        data["top_k"] = top_k
        output.write_text(json.dumps(data, indent=2))
        console.print(f"[dim]Results saved to {output}[/dim]\n")


if __name__ == "__main__":
    main()
