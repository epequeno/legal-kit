//! Stage 2: Structural chunking.

use crate::models::{Chunk, DocumentNode, NodeType};
use crate::pipeline::parse::extract_cross_references;

const MIN_CHUNK_CHARS: usize = 100;
const MAX_CHUNK_CHARS: usize = 1500;

pub fn chunk_document(root: &DocumentNode, doc_id: &str) -> Vec<Chunk> {
    let mut raw: Vec<Chunk> = Vec::new();
    walk_and_chunk(root, doc_id, &mut raw, MAX_CHUNK_CHARS);

    if raw.is_empty() {
        let all_text = root.all_text();
        if !all_text.is_empty() {
            raw = fallback_chunk(&all_text, doc_id, MAX_CHUNK_CHARS);
        }
    }

    let merged = merge_small_chunks(raw, MIN_CHUNK_CHARS, MAX_CHUNK_CHARS);

    // Assign sequential IDs
    merged
        .into_iter()
        .enumerate()
        .map(|(i, mut c)| {
            c.id = format!("{doc_id}-chunk-{i:04}");
            c.embedding_text = c.text.clone();
            c
        })
        .collect()
}

fn walk_and_chunk(
    node: &DocumentNode,
    doc_id: &str,
    chunks: &mut Vec<Chunk>,
    max_chars: usize,
) {
    if node.node_type == NodeType::Document {
        for child in &node.children {
            walk_and_chunk(child, doc_id, chunks, max_chars);
        }
        return;
    }

    let text = node.text.trim();
    if text.is_empty() && node.children.is_empty() {
        return;
    }

    if !text.is_empty() {
        if text.len() <= max_chars {
            chunks.push(make_chunk(node, text, doc_id));
        } else {
            for (sub_text, offset) in split_text(text, max_chars) {
                let mut chunk = make_chunk(node, &sub_text, doc_id);
                chunk.char_start = node.char_start + offset;
                chunk.char_end = node.char_start + offset + sub_text.len();
                chunks.push(chunk);
            }
        }
    }

    for child in &node.children {
        walk_and_chunk(child, doc_id, chunks, max_chars);
    }
}

fn make_chunk(node: &DocumentNode, text: &str, doc_id: &str) -> Chunk {
    Chunk::new(
        text.to_string(),
        doc_id,
        &node.id,
        node.path.clone(),
        node.char_start,
        node.char_end,
        node.cross_references.clone(),
    )
}

fn split_text(text: &str, max_chars: usize) -> Vec<(String, usize)> {
    // Try double-newline splits first, then single, then sentences
    let paragraphs: Vec<&str> = if text.contains("\n\n") {
        text.split("\n\n").collect()
    } else if text.contains('\n') {
        text.split('\n').collect()
    } else {
        split_sentences(text)
    };

    let mut pieces: Vec<(String, usize)> = Vec::new();
    let mut current = String::new();
    let mut current_offset = 0usize;

    for para in paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if !current.is_empty() && current.len() + para.len() + 2 > max_chars {
            pieces.push((current.trim().to_string(), current_offset));
            current_offset = text.find(para).unwrap_or(0);
            current = para.to_string();
        } else {
            if current.is_empty() {
                current_offset = text.find(para).unwrap_or(0);
            }
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
        }
    }
    if !current.trim().is_empty() {
        pieces.push((current.trim().to_string(), current_offset));
    }
    pieces
}

fn split_sentences(text: &str) -> Vec<&str> {
    // Very simple sentence splitter
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = text.chars().collect();
    for i in 0..chars.len() {
        if matches!(chars[i], '.' | '!' | '?')
            && i + 2 < chars.len()
            && chars[i + 1] == ' '
            && chars[i + 2].is_uppercase()
        {
            parts.push(&text[start..i + 1]);
            start = i + 2;
        }
    }
    if start < text.len() {
        parts.push(&text[start..]);
    }
    if parts.is_empty() {
        parts.push(text);
    }
    parts
}

fn merge_small_chunks(
    chunks: Vec<Chunk>,
    min_chars: usize,
    max_chars: usize,
) -> Vec<Chunk> {
    if chunks.is_empty() {
        return vec![];
    }
    let mut merged: Vec<Chunk> = Vec::new();
    for chunk in chunks {
        if chunk.table_repr.is_some() {
            merged.push(chunk);
            continue;
        }
        let can_merge_with_prev = merged.last().map_or(false, |prev: &Chunk| {
            prev.table_repr.is_none()
                && prev.node_id == chunk.node_id
                && prev.text.len() + chunk.text.len() + 2 <= max_chars
        });
        if can_merge_with_prev
            && (merged.last().map_or(0, |p: &Chunk| p.text.len()) < min_chars
                || chunk.text.len() < min_chars)
        {
            let prev = merged.last_mut().unwrap();
            prev.text.push_str("\n\n");
            prev.text.push_str(&chunk.text);
            prev.char_end = chunk.char_end;
            prev.cross_references.extend(chunk.cross_references);
            prev.cross_references.dedup();
        } else {
            merged.push(chunk);
        }
    }
    merged
}

fn fallback_chunk(text: &str, doc_id: &str, max_chars: usize) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut current = String::new();
    let mut current_start = 0usize;
    let mut pos = 0usize;

    for part in text.split("\n\n") {
        let stripped = part.trim();
        if stripped.len() <= 20 {
            pos += part.len() + 2;
            continue;
        }
        let idx = text[pos..].find(stripped).map(|i| pos + i).unwrap_or(pos);

        if !current.is_empty() && current.len() + stripped.len() + 2 > max_chars {
            let xrefs = extract_cross_references(&current);
            let end = current_start + current.len();
            chunks.push(Chunk::new(
                current.trim().to_string(),
                doc_id,
                &format!("{doc_id}-para"),
                vec![doc_id.to_string()],
                current_start,
                end,
                xrefs,
            ));
            current = stripped.to_string();
            current_start = idx;
        } else {
            if current.is_empty() {
                current_start = idx;
            } else {
                current.push_str("\n\n");
            }
            current.push_str(stripped);
        }
        pos = idx + stripped.len();
    }
    if !current.trim().is_empty() {
        let xrefs = extract_cross_references(&current);
        let end = current_start + current.len();
        chunks.push(Chunk::new(
            current.trim().to_string(),
            doc_id,
            &format!("{doc_id}-para"),
            vec![doc_id.to_string()],
            current_start,
            end,
            xrefs,
        ));
    }
    chunks
}
