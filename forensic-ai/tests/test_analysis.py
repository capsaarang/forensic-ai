"""
Tests for analysis pipeline: scorer and finding structures.

Note: LLM auditor tests are integration tests (require ANTHROPIC_API_KEY).
Run them with: pytest tests/test_analysis.py -m integration
"""

import pytest
from src.analysis.scorer import compute_heuristic_score, severity_breakdown, top_findings
from src.analysis.focus_areas import FOCUS_AREAS, get_all_queries, get_focus_area
from src.analysis.auditor import Finding, _parse_findings, _extract_followup_request


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_finding(severity: str, focus_area: str = "risk_factors") -> Finding:
    return Finding(
        id="F1",
        severity=severity,
        section="Item 1A",
        title="Test finding",
        detail="Detail text.",
        flagged_text="some flagged text",
        recommendation="Monitor closely.",
        focus_area=focus_area,
    )


# ── Focus Areas ───────────────────────────────────────────────────────────────

class TestFocusAreas:
    def test_all_focus_areas_present(self):
        expected = {"risk_factors", "revenue", "debt", "litigation", "related_party", "forward_guidance"}
        assert set(FOCUS_AREAS.keys()) == expected

    def test_each_area_has_queries(self):
        for key, area in FOCUS_AREAS.items():
            assert len(area["queries"]) >= 3, f"{key} has too few queries"

    def test_each_area_has_instructions(self):
        for key, area in FOCUS_AREAS.items():
            assert len(area["audit_instructions"]) > 50, f"{key} missing audit instructions"

    def test_get_focus_area_valid(self):
        area = get_focus_area("revenue")
        assert area["label"] == "Revenue Anomalies"

    def test_get_focus_area_invalid(self):
        with pytest.raises(ValueError):
            get_focus_area("nonexistent_area")

    def test_get_all_queries_returns_pairs(self):
        pairs = get_all_queries(["risk_factors", "revenue"])
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
        focus_keys = {p[0] for p in pairs}
        assert "risk_factors" in focus_keys
        assert "revenue" in focus_keys


# ── Scorer ────────────────────────────────────────────────────────────────────

class TestScorer:
    def test_empty_findings_score_zero(self):
        assert compute_heuristic_score([]) == 0

    def test_high_findings_raise_score(self):
        findings = [make_finding("HIGH") for _ in range(3)]
        score = compute_heuristic_score(findings)
        assert score > 40

    def test_info_findings_no_score(self):
        findings = [make_finding("INFO") for _ in range(5)]
        assert compute_heuristic_score(findings) == 0

    def test_score_bounded_0_to_100(self):
        findings = [make_finding("HIGH", "related_party") for _ in range(20)]
        score = compute_heuristic_score(findings)
        assert 0 <= score <= 100

    def test_related_party_higher_weight(self):
        rp_findings = [make_finding("HIGH", "related_party") for _ in range(2)]
        rf_findings = [make_finding("HIGH", "risk_factors") for _ in range(2)]
        assert compute_heuristic_score(rp_findings) >= compute_heuristic_score(rf_findings)

    def test_severity_breakdown_counts(self):
        findings = [
            make_finding("HIGH"),
            make_finding("HIGH"),
            make_finding("MEDIUM"),
            make_finding("LOW"),
            make_finding("INFO"),
        ]
        breakdown = severity_breakdown(findings)
        assert breakdown["HIGH"] == 2
        assert breakdown["MEDIUM"] == 1
        assert breakdown["LOW"] == 1
        assert breakdown["INFO"] == 1

    def test_top_findings_ordering(self):
        findings = [
            make_finding("LOW"),
            make_finding("HIGH"),
            make_finding("MEDIUM"),
            make_finding("INFO"),
        ]
        top = top_findings(findings, n=2)
        assert top[0].severity == "HIGH"
        assert top[1].severity == "MEDIUM"

    def test_top_findings_respects_n(self):
        findings = [make_finding("HIGH") for _ in range(5)]
        top = top_findings(findings, n=3)
        assert len(top) == 3


# ── Finding Parser ────────────────────────────────────────────────────────────

class TestFindingParser:
    VALID_JSON = """
[
  {
    "id": "F1",
    "severity": "HIGH",
    "section": "Item 1A — Risk Factors",
    "title": "Revenue concentration in iPhone",
    "detail": "iPhone accounts for 52% of revenue, creating concentration risk.",
    "flagged_text": "iPhone net sales were $200.6 billion",
    "recommendation": "Assess product diversification strategy."
  },
  {
    "id": "F2",
    "severity": "MEDIUM",
    "section": "Item 7 — MD&A",
    "title": "Gross margin pressure noted",
    "detail": "Component costs rising, compressing margins.",
    "flagged_text": "gross margins may be under pressure",
    "recommendation": "Monitor supply chain cost trends."
  }
]
"""

    def test_parses_valid_json(self):
        findings = _parse_findings(self.VALID_JSON, "risk_factors")
        assert len(findings) == 2

    def test_finding_fields_populated(self):
        findings = _parse_findings(self.VALID_JSON, "risk_factors")
        f = findings[0]
        assert f.severity == "HIGH"
        assert f.title == "Revenue concentration in iPhone"
        assert f.focus_area == "risk_factors"
        assert f.flagged_text != ""

    def test_handles_json_with_markdown_fences(self):
        wrapped = f"```json\n{self.VALID_JSON}\n```"
        findings = _parse_findings(wrapped, "revenue")
        assert len(findings) == 2

    def test_returns_empty_on_invalid_json(self):
        findings = _parse_findings("This is not JSON at all.", "debt")
        assert findings == []

    def test_focus_area_set_on_all_findings(self):
        findings = _parse_findings(self.VALID_JSON, "litigation")
        for f in findings:
            assert f.focus_area == "litigation"


# ── Followup Extractor ────────────────────────────────────────────────────────

class TestFollowupExtractor:
    def test_extracts_followup_request(self):
        text = "Some analysis here.\nFOLLOWUP_REQUEST: debt covenants and credit facility terms"
        result = _extract_followup_request(text)
        assert result == "debt covenants and credit facility terms"

    def test_returns_none_when_absent(self):
        text = "Here are the findings in JSON format: [...]"
        result = _extract_followup_request(text)
        assert result is None

    def test_handles_extra_whitespace(self):
        text = "FOLLOWUP_REQUEST:   litigation settlement amounts in notes"
        result = _extract_followup_request(text)
        assert "litigation" in result


# ── Integration test (requires API key) ──────────────────────────────────────

@pytest.mark.integration
class TestAuditorIntegration:
    """
    Integration tests that call the real Anthropic API.
    Run with: pytest tests/test_analysis.py -m integration
    Requires: ANTHROPIC_API_KEY environment variable
    """

    def test_auditor_produces_findings(self):
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        from src.analysis.auditor import Auditor
        from src.ingestion.section_detector import detect_sections
        from src.ingestion.chunker import chunk_sections

        sample = """
Item 1A. Risk Factors
The Company faces material risks including revenue concentration in a single product line.
iPhone represents 52% of net sales. Increased competition from Android manufacturers
may reduce market share. Regulatory investigations in the EU could result in significant fines.

Item 7. Management's Discussion and Analysis
Net revenues increased 8.2% to $394.3 billion. Services segment grew 9% to $85.2B.
Non-GAAP gross margin excludes $2.1B in stock-based compensation charges.
Deferred revenue increased $1.8B year-over-year.
"""
        from src.ingestion.section_detector import detect_sections
        from src.ingestion.chunker import chunk_sections
        from src.retrieval.embedder import Embedder
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.retriever import Retriever

        sections = detect_sections(sample)
        chunks = chunk_sections(sections)
        embedder = Embedder()
        embeddings = embedder.embed_chunks([c.text for c in chunks], show_progress=False)
        store = VectorStore(dim=embedder.dim)
        store.add(chunks, embeddings)
        retriever = Retriever(embedder, store)
        retrieved = retriever.retrieve(["risk_factors", "revenue"], k_per_query=2, max_chunks_per_focus=4)

        auditor = Auditor()
        result = auditor.run(
            ticker="AAPL",
            fiscal_year="2023",
            retrieved_context=retrieved,
            retriever=None,
            total_chunks=len(chunks),
        )

        assert len(result.findings) >= 1
        assert 0 <= result.risk_score <= 100
        assert result.summary
        for f in result.findings:
            assert f.severity in ("HIGH", "MEDIUM", "LOW", "INFO")
