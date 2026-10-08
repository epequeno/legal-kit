//! Tauri commands — called from the Elm frontend via JS ports.

use legalkit_lib::models::{AnswerReport, DocumentMeta};
use legalkit_lib::pipeline::{chunk, embed, embed_local, entailment_local, extract, generate, rerank_local, retrieve};
use legalkit_lib::store;
use std::path::Path;
use tauri::State;
use tokio::sync::Mutex;

/// Shared config held in Tauri's managed state.
///
/// Priority: if `openrouter_key` is set it is used for both embeddings and
/// generation (via OpenRouter's OpenAI-compatible API).  Otherwise
/// `openai_key` is used for embeddings and `anthropic_key` for generation.
pub struct AppConfig {
    pub openrouter_key: String,
    pub openrouter_model: String,  // e.g. "minimax/minimax-m2.7"
    pub openai_key: String,
    pub anthropic_key: String,
    pub anthropic_model: String,   // e.g. "claude-sonnet-4-20250514"
}

impl AppConfig {
    /// Returns (embed_key, embed_via_openrouter, gen_key, gen_via_openrouter, gen_model)
    pub fn resolve(&self) -> (String, bool, String, bool, String) {
        if !self.openrouter_key.is_empty() {
            let model = if self.openrouter_model.is_empty() {
                "anthropic/claude-sonnet-4".to_string()
            } else {
                self.openrouter_model.clone()
            };
            (
                self.openrouter_key.clone(), true,
                self.openrouter_key.clone(), true,
                model,
            )
        } else {
            let model = if self.anthropic_model.is_empty() {
                "claude-sonnet-4-20250514".to_string()
            } else {
                self.anthropic_model.clone()
            };
            (
                self.openai_key.clone(), false,
                self.anthropic_key.clone(), false,
                model,
            )
        }
    }

    pub fn ready(&self) -> bool {
        !self.openrouter_key.is_empty()
            || (!self.openai_key.is_empty() && !self.anthropic_key.is_empty())
    }

    pub fn active_model(&self) -> String {
        if !self.openrouter_key.is_empty() {
            if self.openrouter_model.is_empty() {
                "anthropic/claude-sonnet-4".to_string()
            } else {
                self.openrouter_model.clone()
            }
        } else if !self.anthropic_model.is_empty() {
            self.anthropic_model.clone()
        } else {
            "claude-sonnet-4-20250514".to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

/// Open a file at `path`, run the full pipeline, persist to ~/.legalkit.
/// Returns DocumentMeta on success.
#[tauri::command]
pub async fn ingest_document(
    path: String,
    config: State<'_, Mutex<AppConfig>>,
) -> Result<DocumentMeta, String> {
    let (embed_key, embed_openrouter, _, _, _) = {
        let cfg = config.lock().await;
        if !cfg.ready() {
            return Err(
                "No API keys configured. Set OPENROUTER_API_KEY (or both OPENAI_API_KEY and ANTHROPIC_API_KEY).".to_string()
            );
        }
        cfg.resolve()
    };

    let file_path = Path::new(&path);
    let filename = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // doc_id from stem — strip extension, sanitise
    let doc_id = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("doc")
        .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "_");

    let size_bytes = std::fs::metadata(file_path)
        .map(|m| m.len())
        .unwrap_or(0);

    // Extract text
    let text = extract::extract_text(file_path).map_err(|e| e.to_string())?;

    // Parse → chunk
    let root = legalkit_lib::pipeline::parse::parse_document(&text, &doc_id);
    let chunks = chunk::chunk_document(&root, &doc_id);
    let n_chunks = chunks.len();

    // Embed — use local model if loaded, otherwise API
    let embedding_texts: Vec<String> = chunks.iter().map(|c| c.embedding_text.clone()).collect();
    let embeddings = if embed_local::is_loaded() {
        embed_local::embed_texts(&embedding_texts).map_err(|e| e.to_string())?
    } else {
        embed::embed_texts(&embedding_texts, &embed_key, embed_openrouter)
            .await
            .map_err(|e| e.to_string())?
    };

    // Build index
    let index = legalkit_lib::pipeline::index::DocumentIndex::build(chunks, embeddings);

    // Persist
    let meta = DocumentMeta {
        id: doc_id,
        filename,
        original_path: path,
        size_bytes,
        n_chunks,
        ingested_at: chrono::Utc::now().to_rfc3339(),
    };

    store::save_document(&meta, &index).map_err(|e| e.to_string())?;

    Ok(meta)
}

// ---------------------------------------------------------------------------
// Ask
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn ask_question(
    doc_id: String,
    question: String,
    config: State<'_, Mutex<AppConfig>>,
) -> Result<AnswerReport, String> {
    let (embed_key, embed_openrouter, gen_key, gen_openrouter, gen_model) = {
        let cfg = config.lock().await;
        if !cfg.ready() {
            return Err(
                "No API keys configured. Set OPENROUTER_API_KEY (or both OPENAI_API_KEY and ANTHROPIC_API_KEY).".to_string()
            );
        }
        cfg.resolve()
    };

    // Load index
    let index = store::load_index(&doc_id).map_err(|e| e.to_string())?;

    // Embed query — use local model if loaded, otherwise API
    let query_emb = if embed_local::is_loaded() {
        embed_local::embed_query(&question).map_err(|e| e.to_string())?
    } else {
        embed::embed_query(&question, &embed_key, embed_openrouter)
            .await
            .map_err(|e| e.to_string())?
    };

    // Retrieve — fetch a wide candidate set (top 20)
    let mut retrieval = retrieve::retrieve(&question, &query_emb, &index, 20);

    // Rerank — if model is loaded, rescore & reorder ranked_chunks by legal relevance
    if rerank_local::is_loaded() {
        let passages: Vec<(String, usize)> = retrieval
            .ranked_chunks
            .iter()
            .enumerate()
            .map(|(i, rc)| (rc.chunk.text.clone(), i))
            .collect();

        if let Ok(reranked) = rerank_local::rerank(&question, &passages) {
            let original = retrieval.ranked_chunks.clone();
            retrieval.ranked_chunks = reranked
                .into_iter()
                .map(|(orig_idx, score)| {
                    let mut rc = original[orig_idx].clone();
                    rc.score = score;
                    rc
                })
                .collect();
        }
    }

    // Generate
    let mut report = generate::generate_answer(&question, &retrieval, &doc_id, &gen_key, gen_openrouter, &gen_model)
        .await
        .map_err(|e| e.to_string())?;

    // NOTE: DeBERTa entailment verification is disabled.
    // At 3.1s per citation pair on CPU, it adds 15-25s to every query.
    // The reranker (XLM-R, 1.9s for 20 passages) already provides strong
    // relevance filtering. Entailment will be re-enabled once replaced with
    // a faster model (e.g. a fine-tuned MiniLM cross-encoder at ~50ms/pair).
    // if entailment_local::is_loaded() && !report.citations.is_empty() {
    //     entailment_local::verify_citations(&mut report.citations, &index.chunks);
    // }

    // Persist to chat history (best-effort — don't fail the response if save fails)
    let _ = store::append_chat_entry(&doc_id, &report);

    Ok(report)
}

// ---------------------------------------------------------------------------
// Document outline
// ---------------------------------------------------------------------------

/// Return a structural summary of a document — top sections, stats.
/// Used by the contract overview panel in the UI.
#[tauri::command]
pub fn get_document_outline(doc_id: String) -> Result<serde_json::Value, String> {
    let index = store::load_index(&doc_id).map_err(|e| e.to_string())?;

    // Collect top-level sections: chunks whose path has ≤ 2 elements
    // path = [doc_root, section], so len==2 means top-level
    let mut seen = std::collections::HashSet::new();
    let mut top_sections: Vec<serde_json::Value> = Vec::new();

    for chunk in &index.chunks {
        if chunk.node_path.len() <= 2 && chunk.node_path.len() >= 1 {
            let section_id = chunk.node_id.clone();
            if seen.contains(&section_id) {
                continue;
            }
            seen.insert(section_id.clone());

            // Derive display title: last element of node_path
            let title = chunk.node_path.last()
                .cloned()
                .unwrap_or_else(|| section_id.clone());

            // Rough node type from node_id prefix
            let node_type = if section_id.starts_with('§') {
                "section"
            } else if section_id.to_lowercase().contains("article") {
                "article"
            } else if section_id.to_lowercase().contains("exhibit") {
                "exhibit"
            } else if section_id.to_lowercase().contains("schedule") {
                "schedule"
            } else if section_id.to_lowercase().contains("preamble") || section_id.to_lowercase().contains("recital") {
                "preamble"
            } else {
                "section"
            };

            top_sections.push(serde_json::json!({
                "id": section_id,
                "title": title,
                "node_type": node_type,
            }));

            if top_sections.len() >= 25 {
                break;
            }
        }
    }

    let total_cross_refs: usize = index.chunks.iter()
        .map(|c| c.cross_references.len())
        .sum();

    // Count chunks at deeper levels as a proxy for complexity
    let subsection_count = index.chunks.iter()
        .filter(|c| c.node_path.len() > 2)
        .count();

    Ok(serde_json::json!({
        "doc_id": doc_id,
        "total_sections": index.chunks.len(),
        "top_sections": top_sections,
        "cross_refs_count": total_cross_refs,
        "subsection_count": subsection_count,
    }))
}

// ---------------------------------------------------------------------------
// Document management
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn list_documents() -> Result<Vec<DocumentMeta>, String> {
    store::list_documents().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_document(doc_id: String) -> Result<(), String> {
    store::delete_document(&doc_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn load_chat_history(doc_id: String) -> Result<Vec<legalkit_lib::models::AnswerReport>, String> {
    store::load_chat_history(&doc_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn clear_chat_history(doc_id: String) -> Result<(), String> {
    store::clear_chat_history(&doc_id).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn save_settings(
    openrouter_key: String,
    openrouter_model: String,
    openai_key: String,
    anthropic_key: String,
    anthropic_model: String,
    config: State<'_, Mutex<AppConfig>>,
) -> Result<(), String> {
    use legalkit_lib::settings;

    let s = settings::Settings {
        openrouter_key: openrouter_key.clone(),
        openrouter_model: openrouter_model.clone(),
        openai_key: openai_key.clone(),
        anthropic_key: anthropic_key.clone(),
        anthropic_model: anthropic_model.clone(),
    };
    settings::save(&s).map_err(|e| e.to_string())?;

    // Also update the in-memory config so changes take effect immediately
    let merged = settings::merge_with_env(&s);
    let mut cfg = config.lock().await;
    cfg.openrouter_key = merged.openrouter_key;
    cfg.openrouter_model = merged.openrouter_model;
    cfg.openai_key = merged.openai_key;
    cfg.anthropic_key = merged.anthropic_key;
    cfg.anthropic_model = merged.anthropic_model;

    Ok(())
}

#[tauri::command]
pub async fn get_settings(
    config: State<'_, Mutex<AppConfig>>,
) -> Result<serde_json::Value, String> {
    let cfg = config.lock().await;
    Ok(serde_json::json!({
        "openrouter_key": if cfg.openrouter_key.is_empty() { "" } else { "set" },
        "openrouter_model": cfg.openrouter_model,
        "openai_key": if cfg.openai_key.is_empty() { "" } else { "set" },
        "anthropic_key": if cfg.anthropic_key.is_empty() { "" } else { "set" },
        "anthropic_model": cfg.anthropic_model,
    }))
}

#[tauri::command]
pub async fn clear_all_data(
    config: State<'_, Mutex<AppConfig>>,
) -> Result<(), String> {
    legalkit_lib::settings::clear_all_data().map_err(|e| e.to_string())?;

    // Reset in-memory config
    let mut cfg = config.lock().await;
    cfg.openrouter_key = String::new();
    cfg.openrouter_model = String::new();
    cfg.openai_key = String::new();
    cfg.anthropic_key = String::new();
    cfg.anthropic_model = String::new();

    Ok(())
}

// ---------------------------------------------------------------------------
// Settings — update API keys at runtime
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn set_api_keys(
    openrouter_key: String,
    openrouter_model: String,
    openai_key: String,
    anthropic_key: String,
    anthropic_model: String,
    config: State<'_, Mutex<AppConfig>>,
) -> Result<(), String> {
    let mut cfg = config.lock().await;
    cfg.openrouter_key = openrouter_key;
    cfg.openrouter_model = openrouter_model;
    cfg.openai_key = openai_key;
    cfg.anthropic_key = anthropic_key;
    cfg.anthropic_model = anthropic_model;
    Ok(())
}

#[tauri::command]
pub async fn get_api_key_status(
    config: State<'_, Mutex<AppConfig>>,
) -> Result<serde_json::Value, String> {
    let cfg = config.lock().await;
    Ok(serde_json::json!({
        "openrouter": !cfg.openrouter_key.is_empty(),
        "openai": !cfg.openai_key.is_empty(),
        "anthropic": !cfg.anthropic_key.is_empty(),
        "ready": cfg.ready(),
        "active_model": cfg.active_model(),
        "local_embeddings": embed_local::is_loaded(),
        "local_entailment": entailment_local::is_loaded(),
        "local_reranker": rerank_local::is_loaded(),
    }))
}
