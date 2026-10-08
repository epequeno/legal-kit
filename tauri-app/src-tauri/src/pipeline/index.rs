//! In-memory vector + BM25 index.

use crate::models::Chunk;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentIndex {
    pub chunks: Vec<Chunk>,
    pub embeddings: Vec<Vec<f32>>, // shape: [n_chunks][dim]
    bm25_corpus: Vec<Vec<String>>, // tokenized docs
    avg_doc_len: f32,
}

impl DocumentIndex {
    pub fn build(chunks: Vec<Chunk>, embeddings: Vec<Vec<f32>>) -> Self {
        let bm25_corpus: Vec<Vec<String>> = chunks.iter().map(|c| tokenize(&c.embedding_text)).collect();
        let avg_len = compute_avg_len(&bm25_corpus);
        Self { chunks, embeddings, bm25_corpus, avg_doc_len: avg_len }
    }

    // ── Semantic search ──────────────────────────────────────────────────

    pub fn search_semantic(&self, query_emb: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        if self.embeddings.is_empty() {
            return vec![];
        }
        let q_norm = normalize(query_emb);
        let mut scores: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, dot(&normalize(emb), &q_norm)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);
        scores
    }

    // ── BM25 search ──────────────────────────────────────────────────────

    pub fn search_bm25(&self, query: &str, top_k: usize) -> Vec<(usize, f32)> {
        if self.bm25_corpus.is_empty() {
            return vec![];
        }
        let q_tokens = tokenize(query);
        let k1 = 1.5f32;
        let b = 0.75f32;
        let avg_len = self.avg_doc_len;
        let n = self.bm25_corpus.len() as f32;

        // Compute IDF per query token against the corpus
        let idfs: Vec<f32> = q_tokens
            .iter()
            .map(|token| {
                let df = self.bm25_corpus.iter().filter(|doc| doc.contains(token)).count() as f32;
                ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
            })
            .collect();

        let mut scores: Vec<(usize, f32)> = self
            .bm25_corpus
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let doc_len = doc.len() as f32;
                let score: f32 = q_tokens
                    .iter()
                    .zip(idfs.iter())
                    .map(|(token, &idf)| {
                        let tf = doc.iter().filter(|t| *t == token).count() as f32;
                        let num = tf * (k1 + 1.0);
                        let den = tf + k1 * (1.0 - b + b * doc_len / avg_len.max(1.0));
                        idf * num / den.max(1e-9)
                    })
                    .sum();
                (i, score)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);
        scores
    }

    // ── Hybrid RRF ───────────────────────────────────────────────────────

    pub fn search_hybrid(
        &self,
        query_emb: &[f32],
        query_text: &str,
        top_k: usize,
    ) -> Vec<(usize, f32, Option<usize>, Option<usize>)> {
        let sem = self.search_semantic(query_emb, top_k * 2);
        let bm25 = self.search_bm25(query_text, top_k * 2);

        let sem_ranks: std::collections::HashMap<usize, usize> =
            sem.iter().enumerate().map(|(r, (i, _))| (*i, r)).collect();
        let bm25_ranks: std::collections::HashMap<usize, usize> =
            bm25.iter().enumerate().map(|(r, (i, _))| (*i, r)).collect();

        let rrf_k = 60usize;
        let all_idx: std::collections::HashSet<usize> =
            sem_ranks.keys().chain(bm25_ranks.keys()).copied().collect();

        let mut rrf: Vec<(usize, f32, Option<usize>, Option<usize>)> = all_idx
            .into_iter()
            .map(|idx| {
                let sr = sem_ranks.get(&idx).copied();
                let br = bm25_ranks.get(&idx).copied();
                let score = sr.map_or(0.0, |r| 1.0 / (rrf_k + r) as f32)
                    + br.map_or(0.0, |r| 1.0 / (rrf_k + r) as f32);
                (idx, score, sr, br)
            })
            .collect();
        rrf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rrf.truncate(top_k);
        rrf
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tokenize(text: &str) -> Vec<String> {
    let text = text.to_lowercase();
    // Keep alphanumeric and § symbol
    let re = regex::Regex::new(r"[a-z0-9§]+(?:[.\-'][a-z0-9]+)*").unwrap();
    re.find_iter(&text).map(|m| m.as_str().to_string()).collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-10;
    v.iter().map(|x| x / norm).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn compute_avg_len(corpus: &[Vec<String>]) -> f32 {
    let n = corpus.len() as f32;
    if n == 0.0 {
        1.0
    } else {
        corpus.iter().map(|d| d.len() as f32).sum::<f32>() / n
    }
}
