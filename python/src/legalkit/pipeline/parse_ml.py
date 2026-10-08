"""ML-based document structure parser for legal contracts.

Uses a fine-tuned legal-roberta-base sequence classifier to detect
structural headings (articles, sections, subsections, preambles,
recitals, exhibits, definitions) in plain-text contracts.

Designed as a drop-in enhancement to parse.py: the hybrid function
parse_document_hybrid() runs the regex parser first and only invokes
the ML model when the regex parse is sparse (few nodes detected).

Model: lexlms/legal-roberta-base fine-tuned on 402 contracts
       (weak supervision from regex parser outputs)
       Heading F1: 88.7% | Test accuracy: 97.9%
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from legalkit.models.document import DocumentNode, NodeType

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Label taxonomy and mapping
# ---------------------------------------------------------------------------

LABELS = [
    "O",                    # outside (body text, blank lines)
    "B-ARTICLE_HEADER",     # "ARTICLE IV - INDEMNIFICATION"
    "I-ARTICLE_HEADER",
    "B-SECTION_HEADER",     # "Section 4.2 Cap on Liability"
    "I-SECTION_HEADER",
    "B-SUBSECTION_HEADER",  # "4.2.1 General Rule" or "(a)" at start of line
    "I-SUBSECTION_HEADER",
    "B-PREAMBLE",           # Agreement preamble / parties block
    "I-PREAMBLE",
    "B-RECITAL",            # "WHEREAS, ..."
    "I-RECITAL",
    "B-EXHIBIT_HEADER",     # "EXHIBIT A — BILL OF SALE"
    "I-EXHIBIT_HEADER",
    "B-DEFINITION",         # "'Affiliate' means..."
    "I-DEFINITION",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Map ML label -> NodeType
LABEL_TO_NODE_TYPE = {
    "B-ARTICLE_HEADER": NodeType.ARTICLE,
    "B-SECTION_HEADER": NodeType.SECTION,
    "B-SUBSECTION_HEADER": NodeType.SUBSECTION,
    "B-PREAMBLE": NodeType.PREAMBLE,
    "B-RECITAL": NodeType.RECITAL,
    "B-EXHIBIT_HEADER": NodeType.EXHIBIT,
    "B-DEFINITION": NodeType.DEFINITION,
}

# Node type hierarchy levels (for tree building)
NODE_TYPE_LEVELS = {
    NodeType.ARTICLE: 1,
    NodeType.SECTION: 2,
    NodeType.SUBSECTION: 3,
    NodeType.CLAUSE: 4,
    NodeType.SUBCLAUSE: 5,
    NodeType.PREAMBLE: 1,
    NodeType.RECITAL: 1,
    NodeType.EXHIBIT: 1,
    NodeType.DEFINITION: 2,
}


# ---------------------------------------------------------------------------
# StructureClassifier
# ---------------------------------------------------------------------------


class StructureClassifier:
    """Cached wrapper for the fine-tuned structure classification model.
    
    Loads on first use and caches by model path (same pattern as LocalNLIModel).
    Falls back to None if the model path doesn't exist.
    """
    
    # Class-level cache
    _cache: dict[str, 'StructureClassifier | None'] = {}
    
    def __init__(self, model, tokenizer, device: str):
        """Initialize with loaded model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_path: str | None = None
    
    @classmethod
    def load(cls, model_path: str | None, device: str = "cpu") -> 'StructureClassifier | None':
        """Load model from path. Returns None if path doesn't exist.
        
        Args:
            model_path: Path to model directory containing config.json, model.safetensors, etc.
            device: Device to load model on ('cpu' or 'cuda')
            
        Returns:
            StructureClassifier instance or None if path doesn't exist
        """
        if model_path is None:
            return None
        
        # Check cache
        cache_key = f"{model_path}:{device}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        # Check if path exists
        path = Path(model_path)
        if not path.exists():
            cls._cache[cache_key] = None
            return None
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            model.eval()
            
            instance = cls(model, tokenizer, device)
            instance.model_path = model_path
            cls._cache[cache_key] = instance
            return instance
            
        except Exception:
            # Failed to load — cache None to avoid repeated attempts
            cls._cache[cache_key] = None
            return None
    
    def classify_lines(self, lines: list[str]) -> list[str]:
        """Classify a list of text lines. Returns list of label strings.
        
        Args:
            lines: List of text lines to classify
            
        Returns:
            List of label strings (one per input line), e.g. ["O", "B-SECTION_HEADER", ...]
        """
        if not lines:
            return []
        
        import torch
        
        # Batch tokenize all lines
        encodings = self.tokenizer(
            lines,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to device
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Convert to label strings
        pred_labels = [ID2LABEL[pred.item()] for pred in predictions]
        return pred_labels
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the model cache."""
        cls._cache.clear()


# ---------------------------------------------------------------------------
# ML Parser
# ---------------------------------------------------------------------------


def parse_document_ml(
    text: str,
    doc_id: str = "doc",
    classifier: StructureClassifier | None = None,
    model_path: str | None = None,
) -> DocumentNode:
    """Parse a legal document using the ML structure classifier.
    
    Algorithm:
    1. Split text into lines, record char offsets
    2. Run classifier on all lines to get BIO labels
    3. For each B-* label: create a DocumentNode of the appropriate type
    4. Assign body text lines as children of the preceding heading
    5. Build the hierarchy: ARTICLE > SECTION > SUBSECTION > CLAUSE
    6. Return root DocumentNode (type=DOCUMENT)
    
    The hierarchy is assembled using a simple stack: maintain a stack of
    open nodes, pop when a heading of equal or higher level is encountered.
    
    Args:
        text: Document text to parse
        doc_id: Document identifier
        classifier: Pre-loaded StructureClassifier instance
        model_path: Path to model directory (auto-load if classifier not provided)
        
    Returns:
        Root DocumentNode with children representing document structure
    """
    # Load classifier if not provided
    if classifier is None:
        classifier = StructureClassifier.load(model_path)
        if classifier is None:
            # Fallback: return empty document
            return DocumentNode(
                id=doc_id,
                type=NodeType.DOCUMENT,
                title=doc_id,
                text=text,
                char_start=0,
                char_end=len(text),
                level=0,
            )
    
    # Split text into lines with char offsets
    lines_with_offsets: list[tuple[str, int, int]] = []
    char_offset = 0
    for line_text in text.splitlines():
        start = char_offset
        end = char_offset + len(line_text)
        lines_with_offsets.append((line_text, start, end))
        char_offset = end + 1  # +1 for newline
    
    if not lines_with_offsets:
        # Empty document
        return DocumentNode(
            id=doc_id,
            type=NodeType.DOCUMENT,
            title=doc_id,
            text="",
            char_start=0,
            char_end=0,
            level=0,
        )
    
    # Run classifier on all lines
    line_texts = [line for line, _, _ in lines_with_offsets]
    labels = classifier.classify_lines(line_texts)
    
    # Combine lines with their labels
    lines_with_labels = [
        (line, label, start, end)
        for (line, start, end), label in zip(lines_with_offsets, labels)
    ]
    
    # Build tree from labeled lines
    root = _build_tree_from_labels(lines_with_labels, doc_id)
    return root


def _build_tree_from_labels(
    lines_with_labels: list[tuple[str, str, int, int]],
    doc_id: str,
) -> DocumentNode:
    """Build a DocumentNode tree from BIO-labeled lines.
    
    Args:
        lines_with_labels: List of (line_text, label, char_start, char_end) tuples
        doc_id: Document identifier
        
    Returns:
        Root DocumentNode (type=DOCUMENT)
    """
    # Create root
    if not lines_with_labels:
        return DocumentNode(
            id=doc_id,
            type=NodeType.DOCUMENT,
            title=doc_id,
            text="",
            char_start=0,
            char_end=0,
            level=0,
        )
    
    root = DocumentNode(
        id=doc_id,
        type=NodeType.DOCUMENT,
        title=doc_id,
        text="",
        char_start=0,
        char_end=lines_with_labels[-1][3],  # End of last line
        level=0,
    )
    root.path = [doc_id]
    
    # Stack of (level, node) to track open nodes
    stack: list[tuple[int, DocumentNode]] = [(0, root)]
    
    # Counters for generating IDs
    counters = {
        NodeType.ARTICLE: 0,
        NodeType.SECTION: 0,
        NodeType.SUBSECTION: 0,
        NodeType.PREAMBLE: 0,
        NodeType.RECITAL: 0,
        NodeType.EXHIBIT: 0,
        NodeType.DEFINITION: 0,
    }
    
    # Current heading being built (for I-* continuation labels)
    current_heading: DocumentNode | None = None
    current_heading_lines: list[str] = []
    
    # Body text accumulator (for O labels)
    body_lines: list[str] = []
    body_start: int | None = None
    
    def _flush_body():
        """Flush accumulated body text to the current node."""
        nonlocal body_lines, body_start
        if body_lines and stack:
            parent = stack[-1][1]
            body_text = "\n".join(body_lines)
            if parent.text:
                parent.text += "\n" + body_text
            else:
                parent.text = body_text
            # Update char_end
            if body_start is not None:
                parent.char_end = lines_with_labels[i][3]
        body_lines = []
        body_start = None
    
    def _flush_heading():
        """Flush the current heading node."""
        nonlocal current_heading, current_heading_lines
        if current_heading is not None:
            # Set heading title from accumulated lines
            current_heading.title = " ".join(current_heading_lines).strip()
        current_heading = None
        current_heading_lines = []
    
    for i, (line_text, label, char_start, char_end) in enumerate(lines_with_labels):
        # Check if this is a B-* (new heading)
        if label.startswith("B-"):
            # Flush any accumulated body text and previous heading
            _flush_body()
            _flush_heading()
            
            # Get node type from label
            node_type = LABEL_TO_NODE_TYPE.get(label)
            if node_type is None:
                continue
            
            # Get level for this node type
            level = NODE_TYPE_LEVELS.get(node_type, 2)
            
            # Pop stack until we find a parent with lower level
            while len(stack) > 1 and stack[-1][0] >= level:
                stack.pop()
            
            parent = stack[-1][1]
            
            # Generate node ID
            counters[node_type] += 1
            if node_type == NodeType.ARTICLE:
                node_id = f"Article-{counters[node_type]}"
            elif node_type == NodeType.SECTION:
                node_id = f"§{counters[node_type]}"
            elif node_type == NodeType.SUBSECTION:
                node_id = f"§{counters[NodeType.SECTION]}.{counters[node_type]}"
            elif node_type == NodeType.PREAMBLE:
                node_id = f"{doc_id}-preamble"
            elif node_type == NodeType.RECITAL:
                node_id = f"{doc_id}-recital-{counters[node_type]}"
            elif node_type == NodeType.EXHIBIT:
                node_id = f"Exhibit-{counters[node_type]}"
            elif node_type == NodeType.DEFINITION:
                # Try to extract term from line
                term = line_text.strip().split()[0] if line_text.strip() else "Term"
                node_id = f"Def-{term}"
            else:
                node_id = f"{doc_id}-{node_type.value}-{counters[node_type]}"
            
            # Create node
            node = DocumentNode(
                id=node_id,
                type=node_type,
                title="",  # Will be set when we flush
                text="",
                char_start=char_start,
                char_end=char_end,
                level=level,
                parent_id=parent.id,
            )
            node.path = parent.path + [node.id]
            
            # Add to parent
            parent.children.append(node)
            
            # Push to stack
            stack.append((level, node))
            
            # Start accumulating heading lines
            current_heading = node
            current_heading_lines = [line_text.strip()]
            
        elif label.startswith("I-"):
            # Continuation of current heading
            if current_heading is not None:
                current_heading_lines.append(line_text.strip())
                # Update char_end
                current_heading.char_end = char_end
        
        else:  # label == "O"
            # Flush heading if any
            _flush_heading()
            
            # Accumulate body text
            if line_text.strip():  # Skip blank lines
                if body_start is None:
                    body_start = char_start
                body_lines.append(line_text.strip())
    
    # Final flush
    _flush_body()
    _flush_heading()
    
    return root


# ---------------------------------------------------------------------------
# Hybrid Parser
# ---------------------------------------------------------------------------


def parse_document_hybrid(
    text: str,
    doc_id: str = "doc",
    model_path: str | None = None,
    min_nodes_threshold: int = 5,
) -> tuple[DocumentNode, str]:
    """Hybrid parser: regex first, ML fallback for sparse parses.
    
    Returns (root, method) where method is 'regex' or 'ml'.
    
    Heuristic: if the regex parser returns fewer than min_nodes_threshold
    non-PARAGRAPH/non-DOCUMENT nodes, the document is probably non-standard
    formatting. In that case, run the ML parser and return its result.
    
    Falls back to regex if ML model is not available (model_path=None or
    path doesn't exist).
    
    Args:
        text: Document text to parse
        doc_id: Document identifier
        model_path: Path to ML model directory (None = skip ML fallback)
        min_nodes_threshold: Minimum structural nodes before considering regex successful
        
    Returns:
        Tuple of (DocumentNode root, method string "regex" or "ml")
    """
    from legalkit.pipeline.parse import parse_document
    
    # Try regex parser first
    root = parse_document(text, doc_id=doc_id)
    
    # Count structural nodes (exclude DOCUMENT and PARAGRAPH)
    structural_nodes = [
        n for n in root.all_nodes()
        if n.type not in (NodeType.PARAGRAPH, NodeType.DOCUMENT)
    ]
    
    # If regex found enough structure, use it
    if len(structural_nodes) >= min_nodes_threshold:
        return root, "regex"
    
    # Check if ML fallback is available
    if model_path is None:
        return root, "regex"
    
    if not Path(model_path).exists():
        return root, "regex"
    
    # Try ML parser
    classifier = StructureClassifier.load(model_path)
    if classifier is None:
        return root, "regex"
    
    ml_root = parse_document_ml(text, doc_id=doc_id, classifier=classifier)
    return ml_root, "ml"
