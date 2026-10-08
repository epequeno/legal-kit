//! Stage 1: Parse legal document text into a DocumentNode tree.

use crate::models::{DocumentNode, NodeType};
use once_cell::sync::Lazy;
use regex::Regex;

// ---------------------------------------------------------------------------
// Compiled regexes (compiled once, reused across calls)
// ---------------------------------------------------------------------------

static ARTICLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?m)^(?:ARTICLE|Article)\s+([IVXLCDM]+|\d+)(?:\s*[-:.]\s*(.+))?$").unwrap()
});

static SECTION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?m)^(?:(?:SECTION|Section)\s+)((\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\(([a-zA-Z0-9]+)\))?)(?:[\s.:-]+([A-Z].+))?$",
    )
    .unwrap()
});

static SUBCLAUSE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?m)^\(([a-zA-Z0-9]+)\)\s+(.+)").unwrap()
});

static BARE_SECTION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?m)^((\d+)\.(\d+)?(?:\.(\d+))?(?:\(([a-zA-Z0-9]+)\))?)\.?\s{1,4}([A-Z][^\n,;]{1,70})$").unwrap()
});

static ROMAN_HEADER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?m)^([IVXLCDM]+)\.?\s{1,4}([A-Z][^\n]{3,70})$").unwrap()
});

static ROMAN_NUMERAL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$").unwrap()
});

static EXHIBIT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?mi)^(EXHIBIT|SCHEDULE|ANNEX|APPENDIX)\s+([A-Z0-9]+)(?:\s*[-:.]\s*(.+))?$").unwrap()
});

static RECITAL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?m)^WHEREAS[,:]?\s+(.+)").unwrap()
});

static DEFINITION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)["\u{201c}]([^"\u{201d}]+)["\u{201d}]\s+(?:shall\s+)?(?:means?|refers?\s+to|has\s+the\s+meaning)([^"\u{201c}]*?)(?:\n\s*\n|["\u{201c}]|$)"#).unwrap()
});

static XREF_SECTION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[Ss]ections?\s+([\d]+(?:\.[\d]+)*(?:\([a-zA-Z0-9]+\))?)").unwrap()
});
static XREF_ARTICLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[Aa]rticles?\s+([IVXLCDM]+)").unwrap()
});
static XREF_EXHIBIT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:Exhibit|Schedule|Annex|Appendix)\s+([A-Z0-9]+)").unwrap()
});

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn parse_document(text: &str, doc_id: &str) -> DocumentNode {
    let mut root = DocumentNode::new_root(doc_id, text.len());

    let mut markers = find_all_markers(text);
    if markers.is_empty() {
        root.text = text.to_string();
        root.cross_references = extract_cross_references(text);
        return root;
    }

    markers.sort_by_key(|m| m.start);

    // Assign body spans
    for i in 0..markers.len() {
        let end = if i + 1 < markers.len() {
            markers[i + 1].start
        } else {
            text.len()
        };
        let heading_end = text[markers[i].start..]
            .find('\n')
            .map(|p| markers[i].start + p)
            .unwrap_or(end)
            .min(end);
        markers[i].body_start = heading_end;
        markers[i].body_end = end;
        markers[i].body = text[heading_end..end].trim().to_string();
    }

    // Preamble
    if markers[0].start > 0 {
        let preamble_text = text[..markers[0].start].trim().to_string();
        if !preamble_text.is_empty() {
            let xrefs = extract_cross_references(&preamble_text);
            let preamble_id = format!("{doc_id}-preamble");
            let mut node = DocumentNode {
                id: preamble_id.clone(),
                node_type: NodeType::Preamble,
                title: "Preamble".to_string(),
                text: preamble_text,
                char_start: 0,
                char_end: markers[0].start,
                level: 1,
                parent_id: Some(doc_id.to_string()),
                path: vec![doc_id.to_string(), preamble_id],
                children: vec![],
                cross_references: xrefs,
                defined_terms: vec![],
            };
            annotate_definitions_node(&mut node);
            root.children.push(node);
        }
    }

    build_tree(&mut root, &markers, doc_id);
    root
}

// ---------------------------------------------------------------------------
// Marker detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Marker {
    node_type: NodeType,
    start: usize,
    id_raw: String,
    title: String,
    level: u32,
    body_start: usize,
    body_end: usize,
    body: String,
}

fn find_all_markers(text: &str) -> Vec<Marker> {
    let mut markers: Vec<Marker> = Vec::new();
    let mut seen: std::collections::HashSet<usize> = Default::default();

    macro_rules! add {
        ($m:expr) => {{
            let m: Marker = $m;
            if seen.insert(m.start) {
                markers.push(m);
            }
        }};
    }

    for cap in ARTICLE_RE.captures_iter(text) {
        let m = cap.get(0).unwrap();
        add!(Marker {
            node_type: NodeType::Article,
            start: m.start(),
            id_raw: cap[1].to_string(),
            title: cap.get(2).map(|x| x.as_str().trim().to_string()).unwrap_or_default(),
            level: 1,
            body_start: 0, body_end: 0, body: String::new(),
        });
    }

    for cap in SECTION_RE.captures_iter(text) {
        let m = cap.get(0).unwrap();
        let full_id = cap[1].to_string();
        let title = cap.get(6).map(|x| x.as_str().trim().to_string()).unwrap_or_default();
        let depth = [cap.get(2), cap.get(3), cap.get(4), cap.get(5)]
            .iter()
            .filter(|x| x.is_some())
            .count() as u32;
        let (node_type, level) = classify_section_depth(depth);
        add!(Marker {
            node_type,
            start: m.start(),
            id_raw: full_id,
            title,
            level,
            body_start: 0, body_end: 0, body: String::new(),
        });
    }

    for cap in EXHIBIT_RE.captures_iter(text) {
        let m = cap.get(0).unwrap();
        add!(Marker {
            node_type: NodeType::Exhibit,
            start: m.start(),
            id_raw: cap[2].to_string(),
            title: cap.get(3).map(|x| x.as_str().trim().to_string()).unwrap_or_default(),
            level: 1,
            body_start: 0, body_end: 0, body: String::new(),
        });
    }

    let mut recital_count = 0usize;
    for cap in RECITAL_RE.captures_iter(text) {
        recital_count += 1;
        let m = cap.get(0).unwrap();
        add!(Marker {
            node_type: NodeType::Recital,
            start: m.start(),
            id_raw: format!("recital-{recital_count}"),
            title: String::new(),
            level: 1,
            body_start: 0, body_end: 0, body: String::new(),
        });
    }

    for cap in SUBCLAUSE_RE.captures_iter(text) {
        let m = cap.get(0).unwrap();
        let letter = cap[1].to_string();
        let inline_text = cap[2].trim().to_string();
        let level = if letter.len() == 1 && letter.chars().all(|c| c.is_ascii_lowercase()) {
            4
        } else {
            5
        };
        add!(Marker {
            node_type: NodeType::Subclause,
            start: m.start(),
            id_raw: letter,
            title: inline_text,
            level,
            body_start: 0, body_end: 0, body: String::new(),
        });
    }

    for cap in BARE_SECTION_RE.captures_iter(text) {
        let m = cap.get(0).unwrap();
        let full_id = cap[1].trim_end_matches('.').to_string();
        let title = cap[6].trim().to_string();
        let depth = [cap.get(2), cap.get(3), cap.get(4), cap.get(5)]
            .iter()
            .filter(|x| x.is_some())
            .count() as u32;
        let (node_type, level) = classify_section_depth(depth);
        add!(Marker {
            node_type,
            start: m.start(),
            id_raw: full_id,
            title,
            level,
            body_start: 0, body_end: 0, body: String::new(),
        });
    }

    for cap in ROMAN_HEADER_RE.captures_iter(text) {
        let numeral = &cap[1];
        if !is_roman_numeral(numeral) {
            continue;
        }
        let m = cap.get(0).unwrap();
        add!(Marker {
            node_type: NodeType::Article,
            start: m.start(),
            id_raw: numeral.to_string(),
            title: cap[2].trim().to_string(),
            level: 1,
            body_start: 0, body_end: 0, body: String::new(),
        });
    }

    markers
}

fn classify_section_depth(depth: u32) -> (NodeType, u32) {
    match depth {
        d if d >= 3 => (NodeType::Subclause, d + 1),
        2 => (NodeType::Subsection, 3),
        _ => (NodeType::Section, 2),
    }
}

fn is_roman_numeral(s: &str) -> bool {
    !s.is_empty() && ROMAN_NUMERAL_RE.is_match(s) && s != "I" // "I" is too ambiguous
        || matches!(s, "II" | "III" | "IV" | "V" | "VI" | "VII" | "VIII" | "IX" | "X"
            | "XI" | "XII" | "XIII" | "XIV" | "XV" | "XVI" | "XVII" | "XVIII" | "XIX" | "XX")
}

// ---------------------------------------------------------------------------
// Tree building
// ---------------------------------------------------------------------------

fn make_section_id(marker: &Marker, doc_id: &str) -> String {
    match marker.node_type {
        NodeType::Article => format!("Article-{}", marker.id_raw),
        NodeType::Exhibit => format!("Exhibit-{}", marker.id_raw),
        NodeType::Recital => marker.id_raw.clone(),
        NodeType::Section | NodeType::Subsection | NodeType::Subclause => {
            format!("§{}", marker.id_raw)
        }
        _ => format!("{doc_id}-{}", marker.id_raw),
    }
}

fn build_tree(root: &mut DocumentNode, markers: &[Marker], doc_id: &str) {
    // We need to track a stack of (level, node_id, path) for parenting.
    // Because we're building into a tree owned by root, we'll collect
    // node specs and insert them afterwards using indices.

    struct NodeSpec {
        node: DocumentNode,
        parent_path: Vec<String>, // path of parent node
    }

    let mut stack: Vec<(u32, Vec<String>)> = vec![(0, vec![doc_id.to_string()])];
    let mut specs: Vec<NodeSpec> = Vec::new();

    for marker in markers {
        // Pop stack until we find a parent at a lower level
        while stack.len() > 1 && stack.last().map(|(l, _)| *l).unwrap_or(0) >= marker.level {
            stack.pop();
        }
        let parent_path = stack.last().map(|(_, p)| p.clone()).unwrap_or_default();

        // Build node ID (compound for subclauses)
        let node_id = if marker.node_type == NodeType::Subclause {
            if let Some(parent_id) = parent_path.last() {
                if parent_id.starts_with('§') {
                    let base = parent_id.trim_start_matches('§');
                    format!("§{base}({})", marker.id_raw)
                } else {
                    make_section_id(marker, doc_id)
                }
            } else {
                make_section_id(marker, doc_id)
            }
        } else {
            make_section_id(marker, doc_id)
        };

        // Resolve text
        let (title, body_text) = if marker.node_type == NodeType::Subclause {
            let parts: Vec<&str> = [marker.title.as_str(), marker.body.as_str()]
                .iter()
                .filter(|s| !s.is_empty())
                .copied()
                .collect();
            (String::new(), parts.join("\n"))
        } else if marker.body.is_empty() && marker.title.len() > 80 {
            let dot_pos = marker.title.find(". ");
            if let Some(pos) = dot_pos.filter(|&p| p < 60) {
                (marker.title[..pos + 1].to_string(), marker.title[pos + 2..].to_string())
            } else {
                (String::new(), marker.title.clone())
            }
        } else {
            (marker.title.clone(), marker.body.clone())
        };

        let xrefs = extract_cross_references(&body_text);
        let mut node_path = parent_path.clone();
        node_path.push(node_id.clone());

        let mut node = DocumentNode {
            id: node_id.clone(),
            node_type: marker.node_type.clone(),
            title,
            text: body_text,
            char_start: marker.start,
            char_end: marker.body_end,
            level: marker.level,
            parent_id: parent_path.last().cloned(),
            path: node_path.clone(),
            children: vec![],
            cross_references: xrefs,
            defined_terms: vec![],
        };
        annotate_definitions_node(&mut node);

        stack.push((marker.level, node_path.clone()));
        specs.push(NodeSpec { node, parent_path });
    }

    // Insert nodes into tree by parent path
    for spec in specs {
        insert_node(root, spec.node, &spec.parent_path);
    }
}

fn insert_node(root: &mut DocumentNode, node: DocumentNode, parent_path: &[String]) {
    // parent_path is the full path of the parent node.
    // root.path is vec![doc_id].
    // We navigate down the path (skipping root itself) to find the parent.
    if parent_path.len() <= 1 {
        root.children.push(node);
        return;
    }
    // Navigate into root's children following parent_path[1..].
    let mut current = root;
    for segment in &parent_path[1..] {
        if let Some(pos) = current.children.iter().position(|c| &c.id == segment) {
            current = &mut current.children[pos];
        } else {
            // Parent not found — attach to root as fallback
            current.children.push(node);
            return;
        }
    }
    current.children.push(node);
}

// ---------------------------------------------------------------------------
// Cross-reference extraction
// ---------------------------------------------------------------------------

pub fn extract_cross_references(text: &str) -> Vec<String> {
    let mut refs: Vec<String> = Vec::new();
    for cap in XREF_SECTION_RE.captures_iter(text) {
        refs.push(cap[1].to_string());
    }
    for cap in XREF_ARTICLE_RE.captures_iter(text) {
        refs.push(cap[1].to_string());
    }
    for cap in XREF_EXHIBIT_RE.captures_iter(text) {
        refs.push(cap[1].to_string());
    }
    // Deduplicate preserving order
    let mut seen = std::collections::HashSet::new();
    refs.retain(|r| seen.insert(r.clone()));
    refs
}

// ---------------------------------------------------------------------------
// Definition annotation
// ---------------------------------------------------------------------------

pub fn find_definitions(text: &str) -> Vec<(String, usize, String)> {
    let mut results = Vec::new();
    for cap in DEFINITION_RE.captures_iter(text) {
        let term = cap[1].trim().to_string();
        let body = cap.get(2).map(|x| x.as_str().trim().to_string()).unwrap_or_default();
        let offset = cap.get(0).unwrap().start();
        results.push((term, offset, body));
    }
    results
}

fn annotate_definitions_node(node: &mut DocumentNode) {
    if matches!(node.node_type, NodeType::Document) {
        return;
    }
    let defs = find_definitions(&node.text);
    if defs.is_empty() {
        return;
    }
    let terms: Vec<String> = defs.iter().map(|(t, _, _)| t.clone()).collect();
    node.defined_terms = terms.clone();
    // Promote to Definition node type if it looks like a definitions block
    if node.children.is_empty() {
        let def_text_len: usize = defs.iter().map(|(_, _, b)| b.len()).sum();
        if def_text_len > 0 && def_text_len as f64 / node.text.len().max(1) as f64 > 0.4 {
            node.node_type = NodeType::Definition;
        }
    }
}
