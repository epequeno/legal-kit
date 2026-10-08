//! Stage 4: Hybrid retrieval with cross-reference + definition expansion.

use crate::models::{RankedChunk, RetrievalMethod, RetrievalResult};
use crate::pipeline::index::DocumentIndex;

pub fn retrieve(
    query: &str,
    query_embedding: &[f32],
    index: &DocumentIndex,
    top_k: usize,
) -> RetrievalResult {
    let hybrid = index.search_hybrid(query_embedding, query, top_k);

    let ranked_chunks: Vec<RankedChunk> = hybrid
        .into_iter()
        .map(|(idx, score, sem_rank, bm25_rank)| {
            let method = match (sem_rank, bm25_rank) {
                (Some(_), Some(_)) => RetrievalMethod::Hybrid,
                (Some(_), None) => RetrievalMethod::Semantic,
                _ => RetrievalMethod::Bm25,
            };
            RankedChunk {
                chunk: index.chunks[idx].clone(),
                score,
                method,
                semantic_rank: sem_rank,
                bm25_rank,
            }
        })
        .collect();

    // Cross-reference expansion
    let retrieved_ids: std::collections::HashSet<String> =
        ranked_chunks.iter().map(|rc| rc.chunk.node_id.clone()).collect();

    let mut cross_ref_chunks: Vec<RankedChunk> = Vec::new();
    let mut seen_ids = retrieved_ids.clone();

    for rc in &ranked_chunks {
        for xref in &rc.chunk.cross_references {
            let normalized = normalize_section_id(xref);
            if seen_ids.contains(&normalized) {
                continue;
            }
            if let Some(chunk) = index.chunks.iter().find(|c| c.node_id == normalized) {
                cross_ref_chunks.push(RankedChunk {
                    chunk: chunk.clone(),
                    score: 0.0,
                    method: RetrievalMethod::CrossReference,
                    semantic_rank: None,
                    bm25_rank: None,
                });
                seen_ids.insert(normalized);
                if cross_ref_chunks.len() >= 10 {
                    break;
                }
            }
        }
        if cross_ref_chunks.len() >= 10 {
            break;
        }
    }

    RetrievalResult {
        query: query.to_string(),
        ranked_chunks,
        cross_ref_chunks,
        definition_chunks: vec![],
    }
}

fn normalize_section_id(raw: &str) -> String {
    // "4.2(a)" → "§4.2(a)", "Article IV" → "Article-IV", etc.
    let s = raw.trim();
    if s.starts_with('§') || s.starts_with("Article-") || s.starts_with("Exhibit-") {
        return s.to_string();
    }
    if s.chars().next().map_or(false, |c| c.is_ascii_digit()) {
        return format!("§{s}");
    }
    s.to_string()
}
