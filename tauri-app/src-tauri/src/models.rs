use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Document tree
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    Document,
    Preamble,
    Recital,
    Article,
    Section,
    Subsection,
    Subclause,
    Definition,
    Exhibit,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentNode {
    pub id: String,
    pub node_type: NodeType,
    pub title: String,
    pub text: String,
    pub char_start: usize,
    pub char_end: usize,
    pub level: u32,
    pub parent_id: Option<String>,
    pub path: Vec<String>,
    pub children: Vec<DocumentNode>,
    pub cross_references: Vec<String>,
    pub defined_terms: Vec<String>,
}

impl DocumentNode {
    pub fn new_root(id: &str, total_len: usize) -> Self {
        Self {
            id: id.to_string(),
            node_type: NodeType::Document,
            title: id.to_string(),
            text: String::new(),
            char_start: 0,
            char_end: total_len,
            level: 0,
            parent_id: None,
            path: vec![id.to_string()],
            children: Vec::new(),
            cross_references: Vec::new(),
            defined_terms: Vec::new(),
        }
    }

    /// Iterate all nodes in depth-first order (including self).
    pub fn all_nodes(&self) -> Vec<&DocumentNode> {
        let mut result = vec![self];
        for child in &self.children {
            result.extend(child.all_nodes());
        }
        result
    }

    /// All text in this node and all descendants.
    pub fn all_text(&self) -> String {
        let mut parts = vec![self.text.clone()];
        for child in &self.children {
            parts.push(child.all_text());
        }
        parts.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Chunks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub doc_id: String,
    pub node_id: String,
    pub node_path: Vec<String>,
    pub char_start: usize,
    pub char_end: usize,
    pub cross_references: Vec<String>,
    /// Serialized table representation (if this chunk is a table).
    pub table_repr: Option<String>,
    /// Text used for embedding (may include context prefix).
    pub embedding_text: String,
}

impl Chunk {
    pub fn new(
        text: String,
        doc_id: &str,
        node_id: &str,
        node_path: Vec<String>,
        char_start: usize,
        char_end: usize,
        cross_references: Vec<String>,
    ) -> Self {
        let embedding_text = text.clone();
        Self {
            id: String::new(),
            text,
            doc_id: doc_id.to_string(),
            node_id: node_id.to_string(),
            node_path,
            char_start,
            char_end,
            cross_references,
            table_repr: None,
            embedding_text,
        }
    }
}

// ---------------------------------------------------------------------------
// Definition index entry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefinitionEntry {
    pub term: String,
    pub node_id: String,
    pub definition_text: String,
}

// ---------------------------------------------------------------------------
// Document metadata (persisted to meta.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMeta {
    pub id: String,
    pub filename: String,
    pub original_path: String,
    pub size_bytes: u64,
    pub n_chunks: usize,
    pub ingested_at: String,
}

// ---------------------------------------------------------------------------
// Retrieval
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalMethod {
    Semantic,
    Bm25,
    Hybrid,
    CrossReference,
    Definition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedChunk {
    pub chunk: Chunk,
    pub score: f32,
    pub method: RetrievalMethod,
    pub semantic_rank: Option<usize>,
    pub bm25_rank: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub query: String,
    pub ranked_chunks: Vec<RankedChunk>,
    pub cross_ref_chunks: Vec<RankedChunk>,
    pub definition_chunks: Vec<RankedChunk>,
}

// ---------------------------------------------------------------------------
// Answer / report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub node_id: String,
    pub node_path: Vec<String>,
    pub quote: String,
    pub char_start: usize,
    pub char_end: usize,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerReport {
    pub question: String,
    pub answer: String,
    pub citations: Vec<Citation>,
    pub doc_id: String,
    pub elapsed_seconds: f64,
}
