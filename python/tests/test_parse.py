"""Tests for document parsing."""

from pathlib import Path

from legalkit.models.document import NodeType
from legalkit.pipeline.parse import (
    parse_document,
    extract_cross_references,
    find_definitions,
    document_summary,
    _is_roman_numeral,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_sample() -> str:
    return (FIXTURE_DIR / "sample_contract.txt").read_text()


def _load_nda() -> str:
    return (FIXTURE_DIR / "nda_bare_sections.txt").read_text()


def _load_roman() -> str:
    return (FIXTURE_DIR / "roman_articles.txt").read_text()


class TestCrossReferences:
    def test_section_refs(self):
        text = "as defined in Section 3.1 and subject to Section 8.3(a)"
        refs = extract_cross_references(text)
        assert "3.1" in refs
        assert "8.3(a)" in refs

    def test_article_refs(self):
        text = "the representations set forth in Article IV and Article V"
        refs = extract_cross_references(text)
        assert "IV" in refs
        assert "V" in refs

    def test_exhibit_refs(self):
        text = "in the form attached as Exhibit A"
        refs = extract_cross_references(text)
        assert "A" in refs

    def test_no_duplicates(self):
        text = "Section 3.1 and Section 3.1"
        refs = extract_cross_references(text)
        assert refs.count("3.1") == 1


class TestDefinitions:
    def test_finds_definitions(self):
        text = '"Affiliate" means any Person that controls another Person.'
        defs = find_definitions(text)
        assert len(defs) == 1
        assert defs[0][0] == "Affiliate"

    def test_shall_mean(self):
        text = '"Business Day" shall mean any day other than Saturday.'
        defs = find_definitions(text)
        assert len(defs) == 1
        assert defs[0][0] == "Business Day"


class TestParseDocument:
    def test_parses_sample_contract(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")

        assert root.id == "test-apa"
        assert root.type == NodeType.DOCUMENT
        assert len(root.children) > 0

    def test_finds_articles(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")

        articles = [n for n in root.all_nodes() if n.type == NodeType.ARTICLE]
        assert len(articles) >= 8  # Articles I through X

    def test_finds_sections(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")

        sections = [
            n
            for n in root.all_nodes()
            if n.type in (NodeType.SECTION, NodeType.SUBSECTION)
        ]
        assert len(sections) >= 20

    def test_section_ids_are_stable(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")

        # Should be able to find specific sections by ID
        node_8_3 = root.find_by_id("§8.3")
        assert node_8_3 is not None
        assert "indemnification" in node_8_3.text.lower() or "limitation" in node_8_3.title.lower() or "cap" in node_8_3.text.lower()

    def test_cross_references_detected(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")

        # The Closing definition references Section 3.1
        all_xrefs = []
        for node in root.all_nodes():
            all_xrefs.extend(node.cross_references)

        assert len(all_xrefs) > 0
        # At least some section cross-references should exist
        assert any("3.1" in ref or "8.1" in ref for ref in all_xrefs)

    def test_char_offsets_are_valid(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")

        for node in root.all_nodes():
            if node.type == NodeType.DOCUMENT:
                continue
            assert node.char_start >= 0
            assert node.char_end >= node.char_start
            assert node.char_end <= len(text)

    def test_document_summary_runs(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")

        summary = document_summary(root)
        assert "DOCUMENT" in summary
        assert "ARTICLE" in summary
        assert "SECTION" in summary


# ---------------------------------------------------------------------------
# Subclause parsing
# ---------------------------------------------------------------------------


class TestSubclauseParsing:
    def test_sample_contract_has_subclauses(self):
        """The sample APA has (a), (b), (c) items under Section 2.1."""
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        subclauses = [n for n in root.all_nodes() if n.type == NodeType.SUBCLAUSE]
        assert len(subclauses) > 0

    def test_subclause_compound_id(self):
        """Subclauses under a section get compound IDs like §2.1(a)."""
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        # Section 2.1 has (a) through (f) subclauses
        node = root.find_by_id("§2.1(a)")
        assert node is not None, "Expected §2.1(a) to exist"

    def test_subclause_text_captured(self):
        """Inline text on the (a) line is preserved in node.text."""
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        node = root.find_by_id("§2.1(a)")
        assert node is not None
        assert "tangible personal property" in node.text

    def test_subclause_nested_under_section(self):
        """Subclauses are children of their parent section, not the root."""
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        section_2_1 = root.find_by_id("§2.1")
        assert section_2_1 is not None
        child_ids = [c.id for c in section_2_1.children]
        assert "§2.1(a)" in child_ids

    def test_subclause_path_includes_parent(self):
        """Subclause path traces back through parent section to root."""
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        node = root.find_by_id("§2.1(a)")
        assert node is not None
        assert "§2.1" in node.path

    def test_subclause_has_no_title(self):
        """Subclause inline text should be node.text, not node.title."""
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        node = root.find_by_id("§2.1(a)")
        assert node is not None
        assert node.title == ""
        assert node.text != ""

    def test_multiple_subclauses_parsed(self):
        """Section 2.1 should have multiple lettered subclauses."""
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        section_2_1 = root.find_by_id("§2.1")
        assert section_2_1 is not None
        subclause_children = [c for c in section_2_1.children if c.type == NodeType.SUBCLAUSE]
        assert len(subclause_children) >= 3  # (a) through (f) in the fixture

    def test_subclause_no_duplicate_ids(self):
        """Each subclause under different sections gets a unique ID."""
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        all_ids = [n.id for n in root.all_nodes()]
        assert len(all_ids) == len(set(all_ids)), f"Duplicate node IDs found"


# ---------------------------------------------------------------------------
# Bare numeric section parsing (NDA format)
# ---------------------------------------------------------------------------


class TestBareNumericSections:
    def test_parses_nda_fixture(self):
        text = _load_nda()
        root = parse_document(text, doc_id="test-nda")
        assert len(root.all_nodes()) > 1

    def test_finds_top_level_sections(self):
        """Bare '1.', '2.', etc. sections should be parsed."""
        text = _load_nda()
        root = parse_document(text, doc_id="test-nda")
        all_ids = [n.id for n in root.all_nodes()]
        # Should find §1 or §1.1 style sections
        assert any("§1" in nid for nid in all_ids), f"No §1.x found in {all_ids}"

    def test_finds_subsections(self):
        """'1.1', '2.3' etc. parsed as sections."""
        text = _load_nda()
        root = parse_document(text, doc_id="test-nda")
        node = root.find_by_id("§1.1")
        assert node is not None, "Expected §1.1 to be found"

    def test_subsection_title_captured(self):
        text = _load_nda()
        root = parse_document(text, doc_id="test-nda")
        node = root.find_by_id("§1.1")
        assert node is not None
        # Title should be "Confidential Information" or text should reference it
        assert (
            "Confidential Information" in node.title
            or "Confidential Information" in node.text
        )

    def test_nda_subclauses_get_compound_ids(self):
        """NDA (a) subclauses under bare sections also get compound IDs."""
        text = _load_nda()
        root = parse_document(text, doc_id="test-nda")
        # Section 2.1 has (a), (b), (c) subclauses
        node = root.find_by_id("§2.1(a)")
        assert node is not None, "Expected §2.1(a) in NDA"

    def test_nda_all_sections_present(self):
        text = _load_nda()
        root = parse_document(text, doc_id="test-nda")
        all_ids = [n.id for n in root.all_nodes()]
        for expected in ["§1.1", "§1.2", "§2.1", "§2.2", "§3.1"]:
            assert expected in all_ids, f"Missing {expected}, got {all_ids}"

    def test_no_false_positive_body_text(self):
        """Body text with numbers like '$50,000,000' must not become sections."""
        text = "This is a contract.\n\nThe price is $50,000,000 as agreed.\n\n2.1 Price\n\nThe price is fifty million dollars."
        root = parse_document(text, doc_id="test")
        all_ids = [n.id for n in root.all_nodes()]
        # Should have §2.1 but not parse the money amount as a section
        assert "§2.1" in all_ids
        # No node should have "50" as an id component
        assert not any("§50" in nid for nid in all_ids)


# ---------------------------------------------------------------------------
# Roman numeral article parsing (without "ARTICLE" keyword)
# ---------------------------------------------------------------------------


class TestRomanNumeralHeaders:
    def test_parses_roman_fixture(self):
        text = _load_roman()
        root = parse_document(text, doc_id="test-roman")
        assert len(root.all_nodes()) > 1

    def test_finds_roman_articles(self):
        text = _load_roman()
        root = parse_document(text, doc_id="test-roman")
        articles = [n for n in root.all_nodes() if n.type == NodeType.ARTICLE]
        assert len(articles) >= 3  # I through V

    def test_article_i_found(self):
        text = _load_roman()
        root = parse_document(text, doc_id="test-roman")
        node = root.find_by_id("Article-I")
        assert node is not None

    def test_article_iv_found(self):
        text = _load_roman()
        root = parse_document(text, doc_id="test-roman")
        node = root.find_by_id("Article-IV")
        assert node is not None

    def test_roman_article_title_captured(self):
        text = _load_roman()
        root = parse_document(text, doc_id="test-roman")
        node = root.find_by_id("Article-I")
        assert node is not None
        assert "DEFINITION" in node.title.upper()

    def test_roman_subclauses_nested(self):
        """(a), (b), (c) under Roman numeral articles should nest correctly."""
        text = _load_roman()
        root = parse_document(text, doc_id="test-roman")
        subclauses = [n for n in root.all_nodes() if n.type == NodeType.SUBCLAUSE]
        assert len(subclauses) > 0

    def test_no_false_positives_from_inline_i(self):
        """'I' appearing mid-sentence should not create a spurious Article-I."""
        text = (
            "This Agreement (the \"Agreement\") is entered into by Party A.\n\n"
            "I. DEFINITIONS\n\n"
            "\"Term\" means the period of this Agreement.\n\n"
            "The parties agree that I. is the first article."
        )
        root = parse_document(text, doc_id="test")
        articles = [n for n in root.all_nodes() if n.type == NodeType.ARTICLE]
        # Only the real article header should be detected, not the inline "I."
        assert len(articles) == 1


# ---------------------------------------------------------------------------
# _is_roman_numeral helper
# ---------------------------------------------------------------------------


class TestIsRomanNumeral:
    def test_valid_numerals(self):
        for s in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                  "XI", "XIV", "XIX", "XX", "XL", "L", "XC", "C", "CD", "D", "CM", "M"]:
            assert _is_roman_numeral(s), f"{s!r} should be valid"

    def test_invalid_strings(self):
        for s in ["", "IIII", "VV", "LL", "DD", "hello", "123", "IVXLCDM"]:
            assert not _is_roman_numeral(s), f"{s!r} should be invalid"

    def test_case_insensitive(self):
        assert _is_roman_numeral("iv")
        assert _is_roman_numeral("xiv")


# ---------------------------------------------------------------------------
# Visual / integration
# ---------------------------------------------------------------------------


class TestParsePrintsTree:
    """Quick visual test — run with pytest -s to see output."""

    def test_print_tree(self, capsys):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        print("\n" + document_summary(root))
