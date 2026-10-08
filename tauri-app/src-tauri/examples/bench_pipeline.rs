/// Pipeline timing benchmark — measures each local model stage independently.
///
/// Run with:
///   cargo run --release --example bench_pipeline
///
/// Simulates realistic workloads:
///   - Ingest: embed 80 chunks of ~300 chars each (typical mid-size contract)
///   - Ask:    embed 1 query, rerank 20 passages, verify 8 citations
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  LegalKit Pipeline Benchmark");
    println!("═══════════════════════════════════════════════════════\n");

    let embed_path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/legal-embeddings/model"
    ));
    let entail_path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/legal-entailment/model"
    ));
    let rerank_path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/legal-reranker/final"
    ));

    // ── 1. Model load times ──────────────────────────────────────────────
    println!("── Model Loading ───────────────────────────────────────");

    let t = Instant::now();
    if embed_path.join("model.safetensors").exists() {
        legalkit_lib::pipeline::embed_local::ensure_loaded(embed_path)
            .expect("embed load failed");
        print_timing("BGE-base embeddings", t.elapsed());
    } else {
        println!("  [skip] BGE embeddings — not found at {}", embed_path.display());
    }

    let t = Instant::now();
    if entail_path.join("model.safetensors").exists() {
        legalkit_lib::pipeline::entailment_local::ensure_loaded(entail_path)
            .expect("entailment load failed");
        print_timing("DeBERTa entailment ", t.elapsed());
    } else {
        println!("  [skip] Entailment — not found at {}", entail_path.display());
    }

    let t = Instant::now();
    if rerank_path.join("model.safetensors").exists() {
        legalkit_lib::pipeline::rerank_local::ensure_loaded(rerank_path)
            .expect("reranker load failed");
        print_timing("XLM-R reranker     ", t.elapsed());
    } else {
        println!("  [skip] Reranker — not found at {}", rerank_path.display());
    }

    // ── 2. Embedding — ingest workload ───────────────────────────────────
    println!("\n── Embedding (Ingest Simulation) ───────────────────────");

    if legalkit_lib::pipeline::embed_local::is_loaded() {
        let chunk_text = "This Agreement shall be governed by and construed in accordance \
            with the laws of the State of Delaware, without regard to its conflict of law \
            provisions. The parties agree to submit to the personal jurisdiction of the \
            courts located within Delaware for the resolution of any disputes.";

        for n_chunks in [10, 40, 80] {
            let texts: Vec<String> = (0..n_chunks).map(|_| chunk_text.to_string()).collect();
            let t = Instant::now();
            legalkit_lib::pipeline::embed_local::embed_texts(&texts)
                .expect("embed_texts failed");
            let elapsed = t.elapsed();
            println!(
                "  {:>3} chunks × ~{} chars  →  {:>6}  ({:.0} ms/chunk)",
                n_chunks,
                chunk_text.len(),
                fmt_duration(elapsed),
                elapsed.as_millis() as f64 / n_chunks as f64
            );
        }

        // Single query embed
        let t = Instant::now();
        legalkit_lib::pipeline::embed_local::embed_query("What are the termination conditions?")
            .expect("embed_query failed");
        print_timing("1 query embed      ", t.elapsed());
    } else {
        println!("  [skip] embed_local not loaded");
    }

    // ── 3. Reranking ─────────────────────────────────────────────────────
    println!("\n── Reranking (Ask Simulation) ──────────────────────────");

    if legalkit_lib::pipeline::rerank_local::is_loaded() {
        let query = "What are the termination conditions?";
        let passage = "Either party may terminate this Agreement upon thirty (30) days \
            written notice to the other party. Termination for cause may occur immediately \
            upon written notice if the other party materially breaches this Agreement.";

        for n_passages in [5, 10, 20] {
            let passages: Vec<(String, usize)> = (0..n_passages)
                .map(|i| (passage.to_string(), i))
                .collect();
            let t = Instant::now();
            legalkit_lib::pipeline::rerank_local::rerank(query, &passages)
                .expect("rerank failed");
            let elapsed = t.elapsed();
            println!(
                "  {:>2} passages            →  {:>6}  ({:.0} ms/passage)",
                n_passages,
                fmt_duration(elapsed),
                elapsed.as_millis() as f64 / n_passages as f64
            );
        }
    } else {
        println!("  [skip] rerank_local not loaded");
    }

    // ── 4. Entailment / citation verification ────────────────────────────
    println!("\n── Entailment: Sequential (old) vs Batched (new) ───────");

    if legalkit_lib::pipeline::entailment_local::is_loaded() {
        let passage = "Either party may terminate this Agreement upon thirty (30) days \
            prior written notice. In the event of a material breach, the non-breaching \
            party may terminate immediately upon written notice.";
        let claim = "The agreement can be terminated with 30 days notice.";

        for n_pairs in [1, 3, 8] {
            // Old: sequential
            let t = Instant::now();
            for _ in 0..n_pairs {
                legalkit_lib::pipeline::entailment_local::verify(passage, claim)
                    .expect("verify failed");
            }
            let seq_elapsed = t.elapsed();

            // New: single batched forward pass
            let pairs: Vec<(&str, &str)> = (0..n_pairs).map(|_| (passage, claim)).collect();
            let t = Instant::now();
            legalkit_lib::pipeline::entailment_local::verify_batch(&pairs)
                .expect("verify_batch failed");
            let batch_elapsed = t.elapsed();

            println!(
                "  {:>2} pairs — sequential: {:>6}  batched: {:>6}  ({:.1}× faster)",
                n_pairs,
                fmt_duration(seq_elapsed),
                fmt_duration(batch_elapsed),
                seq_elapsed.as_secs_f64() / batch_elapsed.as_secs_f64()
            );
        }
    } else {
        println!("  [skip] entailment_local not loaded");
    }

    // ── 5. Summary ───────────────────────────────────────────────────────
    println!("\n── Estimated End-to-End Times ──────────────────────────");
    println!("  Ingest (80 chunks):  embed only");
    println!("  Ask:                 1 query embed + rerank 20 + verify 5 citations");
    println!("  (LLM API call not included — typically 3–15s depending on model)");
    println!();
}

fn print_timing(label: &str, d: Duration) {
    println!("  {}  →  {}", label, fmt_duration(d));
}

fn fmt_duration(d: Duration) -> String {
    let ms = d.as_millis();
    if ms >= 1000 {
        format!("{:.1}s", d.as_secs_f64())
    } else {
        format!("{}ms", ms)
    }
}
