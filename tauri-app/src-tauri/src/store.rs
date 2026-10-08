//! Flat-file persistence under ~/.legalkit/docs/<doc-id>/
//!
//! Layout per document:
//!   meta.json        — DocumentMeta
//!   chunks.json      — Vec<Chunk>
//!   embeddings.bin   — little-endian f32 array, row-major [n_chunks × dim]

use crate::models::{Chunk, DocumentMeta};
use crate::pipeline::index::DocumentIndex;
use anyhow::{bail, Result};
use std::path::PathBuf;

pub fn docs_dir() -> Result<PathBuf> {
    let base = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot locate home directory"))?;
    let dir = base.join(".legalkit").join("docs");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn doc_dir(doc_id: &str) -> Result<PathBuf> {
    let dir = docs_dir()?.join(doc_id);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

pub fn save_document(meta: &DocumentMeta, index: &DocumentIndex) -> Result<()> {
    let dir = doc_dir(&meta.id)?;

    // meta.json
    let meta_json = serde_json::to_string_pretty(meta)?;
    std::fs::write(dir.join("meta.json"), meta_json)?;

    // chunks.json
    let chunks_json = serde_json::to_string(&index.chunks)?;
    std::fs::write(dir.join("chunks.json"), chunks_json)?;

    // embeddings.bin — raw little-endian f32 values
    let flat: Vec<f32> = index.embeddings.iter().flatten().copied().collect();
    let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(dir.join("embeddings.bin"), bytes)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

pub fn load_index(doc_id: &str) -> Result<DocumentIndex> {
    let dir = docs_dir()?.join(doc_id);
    if !dir.exists() {
        bail!("Document '{doc_id}' not found");
    }

    let chunks_json = std::fs::read_to_string(dir.join("chunks.json"))?;
    let chunks: Vec<Chunk> = serde_json::from_str(&chunks_json)?;

    let emb_bytes = std::fs::read(dir.join("embeddings.bin"))?;
    if emb_bytes.len() % 4 != 0 {
        bail!("Corrupted embeddings.bin for '{doc_id}'");
    }
    let flat: Vec<f32> = emb_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let dim = if chunks.is_empty() {
        0
    } else {
        flat.len() / chunks.len()
    };
    let embeddings: Vec<Vec<f32>> = flat.chunks(dim.max(1)).map(|s| s.to_vec()).collect();

    Ok(DocumentIndex::build(chunks, embeddings))
}

pub fn load_meta(doc_id: &str) -> Result<DocumentMeta> {
    let dir = docs_dir()?.join(doc_id);
    let json = std::fs::read_to_string(dir.join("meta.json"))?;
    Ok(serde_json::from_str(&json)?)
}

// ---------------------------------------------------------------------------
// List / delete
// ---------------------------------------------------------------------------

pub fn list_documents() -> Result<Vec<DocumentMeta>> {
    let dir = docs_dir()?;
    let mut metas: Vec<DocumentMeta> = Vec::new();

    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let meta_path = entry.path().join("meta.json");
        if meta_path.exists() {
            if let Ok(json) = std::fs::read_to_string(&meta_path) {
                if let Ok(meta) = serde_json::from_str::<DocumentMeta>(&json) {
                    metas.push(meta);
                }
            }
        }
    }

    // Sort newest first
    metas.sort_by(|a, b| b.ingested_at.cmp(&a.ingested_at));
    Ok(metas)
}

pub fn delete_document(doc_id: &str) -> Result<()> {
    let dir = docs_dir()?.join(doc_id);
    if !dir.exists() {
        bail!("Document '{doc_id}' not found");
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

pub fn document_exists(doc_id: &str) -> bool {
    docs_dir()
        .map(|d| d.join(doc_id).join("meta.json").exists())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Chat history
// ---------------------------------------------------------------------------

pub fn append_chat_entry(doc_id: &str, report: &crate::models::AnswerReport) -> Result<()> {
    let path = docs_dir()?.join(doc_id).join("history.json");
    let mut history: Vec<crate::models::AnswerReport> = if path.exists() {
        let json = std::fs::read_to_string(&path)?;
        serde_json::from_str(&json).unwrap_or_default()
    } else {
        vec![]
    };
    history.push(report.clone());
    std::fs::write(&path, serde_json::to_string(&history)?)?;
    Ok(())
}

pub fn load_chat_history(doc_id: &str) -> Result<Vec<crate::models::AnswerReport>> {
    let path = docs_dir()?.join(doc_id).join("history.json");
    if !path.exists() {
        return Ok(vec![]);
    }
    let json = std::fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&json).unwrap_or_default())
}

pub fn clear_chat_history(doc_id: &str) -> Result<()> {
    let path = docs_dir()?.join(doc_id).join("history.json");
    if path.exists() {
        std::fs::remove_file(&path)?;
    }
    Ok(())
}
