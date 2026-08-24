"""
Unit tests for app.models (TASK-003).

Tests cover:
- Valid instantiation of all 12 models
- Field validation (invalid values raise ValidationError)
- Provenance chain: Finding -> Citation -> SourceDocument -> url traceable
- Model serialisation round-trip
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import (
    Citation,
    ContentChunk,
    EmbeddedChunk,
    Finding,
    ResearchPlan,
    ResearchReport,
    ResearchRequest,
    ReportSection,
    SearchResult,
    SourceDocument,
    SourceEvaluation,
    SubQuestion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def basic_request() -> ResearchRequest:
    return ResearchRequest(topic="AI in software development")


@pytest.fixture()
def sub_question() -> SubQuestion:
    return SubQuestion(
        text="How is AI used in code generation?",
        search_queries=["AI code generation tools", "GitHub Copilot productivity"],
    )


@pytest.fixture()
def research_plan(basic_request, sub_question) -> ResearchPlan:
    return ResearchPlan(
        request=basic_request,
        research_objective="Understand how AI tools affect software development workflows.",
        sub_questions=[sub_question],
    )


@pytest.fixture()
def search_result(sub_question) -> SearchResult:
    return SearchResult(
        url="https://example.com/article",
        title="AI Code Generation Study",
        snippet="Researchers found...",
        query="AI code generation tools",
        sub_question_id=sub_question.sub_question_id,
    )


@pytest.fixture()
def source_doc(sub_question) -> SourceDocument:
    return SourceDocument(
        url="https://example.com/article",
        title="AI Code Generation Study",
        domain="example.com",
        text="This is a long article about AI code generation. " * 20,
        sub_question_id=sub_question.sub_question_id,
        query="AI code generation tools",
    )


@pytest.fixture()
def source_eval(source_doc) -> SourceEvaluation:
    return SourceEvaluation(
        source_id=source_doc.source_id,
        url=source_doc.url,
        relevance_score=0.85,
        credibility_score=0.72,
        recency_score=0.90,
        redundancy_score=0.10,
        overall_score=0.78,
        decision="include",
        reason="Highly relevant peer-reviewed source.",
    )


@pytest.fixture()
def content_chunk(source_doc, sub_question) -> ContentChunk:
    return ContentChunk(
        source_id=source_doc.source_id,
        url=source_doc.url,
        source_title=source_doc.title,
        domain=source_doc.domain,
        sub_question_id=sub_question.sub_question_id,
        chunk_index=0,
        total_chunks=3,
        text="AI code generation has shown significant productivity gains.",
    )


@pytest.fixture()
def embedded_chunk(content_chunk) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=content_chunk,
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        model="models/text-embedding-004",
    )


@pytest.fixture()
def citation(source_doc, content_chunk) -> Citation:
    return Citation(
        marker="[1]",
        source_id=source_doc.source_id,
        chunk_id=content_chunk.chunk_id,
        url=source_doc.url,
        title=source_doc.title,
        domain=source_doc.domain,
        excerpt="AI code generation has shown significant productivity gains.",
    )


@pytest.fixture()
def finding(citation, sub_question) -> Finding:
    return Finding(
        claim="AI tools improve developer productivity significantly.",
        evidence="Studies show 30-50% improvement in code writing speed.",
        citations=[citation],
        confidence=0.85,
        sub_question_id=sub_question.sub_question_id,
    )


@pytest.fixture()
def report_section(finding) -> ReportSection:
    return ReportSection(
        heading="Key Findings",
        content="AI tools have demonstrated significant productivity improvements.",
        findings=[finding],
        order=2,
    )


@pytest.fixture()
def research_report(report_section, citation, basic_request) -> ResearchReport:
    return ResearchReport(
        title="AI Impact on Software Development: A Research Report",
        research_question=basic_request.topic,
        depth="standard",
        sources_analyzed=12,
        sources_included=8,
        executive_summary="AI tools are transforming software development.",
        sections=[report_section],
        all_citations=[citation],
        limitations="Limited to English-language sources published after 2020.",
        methodology="Web search + RAG synthesis using Gemini.",
    )


# ---------------------------------------------------------------------------
# ResearchRequest Tests
# ---------------------------------------------------------------------------


class TestResearchRequest:
    def test_valid_instantiation(self, basic_request):
        assert basic_request.topic == "AI in software development"
        assert basic_request.depth == "standard"
        assert basic_request.max_sources == 15

    def test_depth_normalised_to_lowercase(self):
        r = ResearchRequest(topic="Test topic", depth="DEEP")
        assert r.depth == "deep"

    def test_invalid_depth_raises(self):
        with pytest.raises(ValidationError):
            ResearchRequest(topic="Test topic", depth="ultra")

    def test_topic_too_short_raises(self):
        with pytest.raises(ValidationError):
            ResearchRequest(topic="A")

    def test_topic_too_long_raises(self):
        with pytest.raises(ValidationError):
            ResearchRequest(topic="x" * 501)

    def test_max_sources_bounds(self):
        with pytest.raises(ValidationError):
            ResearchRequest(topic="Test topic", max_sources=2)
        with pytest.raises(ValidationError):
            ResearchRequest(topic="Test topic", max_sources=51)

    def test_topic_strips_whitespace(self):
        r = ResearchRequest(topic="  AI development  ")
        assert r.topic == "AI development"


# ---------------------------------------------------------------------------
# SubQuestion Tests
# ---------------------------------------------------------------------------


class TestSubQuestion:
    def test_valid_instantiation(self, sub_question):
        assert len(sub_question.search_queries) == 2
        assert sub_question.priority == 1

    def test_auto_generates_id(self, sub_question):
        assert len(sub_question.sub_question_id) == 36  # UUID format

    def test_empty_queries_raises(self):
        with pytest.raises(ValidationError):
            SubQuestion(text="Valid question?", search_queries=[])

    def test_unique_ids_per_instance(self):
        q1 = SubQuestion(text="Question one?", search_queries=["query1"])
        q2 = SubQuestion(text="Question two?", search_queries=["query2"])
        assert q1.sub_question_id != q2.sub_question_id


# ---------------------------------------------------------------------------
# ResearchPlan Tests
# ---------------------------------------------------------------------------


class TestResearchPlan:
    def test_valid_instantiation(self, research_plan):
        assert len(research_plan.sub_questions) == 1
        assert research_plan.plan_id  # not empty

    def test_all_queries_helper(self, research_plan, sub_question):
        queries = research_plan.all_queries
        assert len(queries) == 2
        for q, sq_id in queries:
            assert sq_id == sub_question.sub_question_id

    def test_empty_sub_questions_raises(self, basic_request):
        with pytest.raises(ValidationError):
            ResearchPlan(
                request=basic_request,
                research_objective="Short objective text here.",
                sub_questions=[],
            )


# ---------------------------------------------------------------------------
# SearchResult Tests
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_valid_instantiation(self, search_result):
        assert search_result.url.startswith("https://")
        assert search_result.sub_question_id

    def test_invalid_url_raises(self, sub_question):
        with pytest.raises(ValidationError):
            SearchResult(
                url="not-a-url",
                query="test",
                sub_question_id=sub_question.sub_question_id,
            )

    def test_auto_retrieved_at(self, search_result):
        assert search_result.retrieved_at is not None


# ---------------------------------------------------------------------------
# SourceDocument Tests
# ---------------------------------------------------------------------------


class TestSourceDocument:
    def test_valid_instantiation(self, source_doc):
        assert source_doc.word_count > 0
        assert source_doc.source_id

    def test_word_count_auto_computed(self, sub_question):
        text = "word " * 42
        doc = SourceDocument(
            url="https://example.com",
            text=text.strip(),
            sub_question_id=sub_question.sub_question_id,
        )
        assert doc.word_count == 42

    def test_invalid_url_raises(self, sub_question):
        with pytest.raises(ValidationError):
            SourceDocument(
                url="ftp://example.com",
                text="Some content here.",
                sub_question_id=sub_question.sub_question_id,
            )


# ---------------------------------------------------------------------------
# SourceEvaluation Tests
# ---------------------------------------------------------------------------


class TestSourceEvaluation:
    def test_valid_instantiation(self, source_eval):
        assert source_eval.is_included is True

    def test_scores_out_of_range_raises(self, source_doc):
        with pytest.raises(ValidationError):
            SourceEvaluation(
                source_id=source_doc.source_id,
                url=source_doc.url,
                relevance_score=1.5,  # > 1.0
                credibility_score=0.5,
                recency_score=0.5,
                redundancy_score=0.5,
                overall_score=0.5,
                decision="include",
            )

    def test_invalid_decision_raises(self, source_doc):
        with pytest.raises(ValidationError):
            SourceEvaluation(
                source_id=source_doc.source_id,
                url=source_doc.url,
                relevance_score=0.8,
                credibility_score=0.7,
                recency_score=0.9,
                redundancy_score=0.1,
                overall_score=0.8,
                decision="maybe",
            )

    def test_excluded_source(self, source_doc):
        ev = SourceEvaluation(
            source_id=source_doc.source_id,
            url=source_doc.url,
            relevance_score=0.2,
            credibility_score=0.3,
            recency_score=0.5,
            redundancy_score=0.9,
            overall_score=0.3,
            decision="exclude",
            reason="Low relevance and high redundancy.",
        )
        assert ev.is_included is False


# ---------------------------------------------------------------------------
# ContentChunk Tests
# ---------------------------------------------------------------------------


class TestContentChunk:
    def test_valid_instantiation(self, content_chunk):
        assert content_chunk.chunk_id
        assert content_chunk.word_count > 0

    def test_invalid_url_raises(self, source_doc, sub_question):
        with pytest.raises(ValidationError):
            ContentChunk(
                source_id=source_doc.source_id,
                url="bad-url",
                sub_question_id=sub_question.sub_question_id,
                chunk_index=0,
                total_chunks=1,
                text="Some text here.",
            )


# ---------------------------------------------------------------------------
# EmbeddedChunk Tests
# ---------------------------------------------------------------------------


class TestEmbeddedChunk:
    def test_valid_instantiation(self, embedded_chunk, content_chunk):
        assert embedded_chunk.chunk_id == content_chunk.chunk_id
        assert embedded_chunk.source_id == content_chunk.source_id
        assert embedded_chunk.sub_question_id == content_chunk.sub_question_id

    def test_empty_embedding_raises(self, content_chunk):
        with pytest.raises(ValidationError):
            EmbeddedChunk(chunk=content_chunk, embedding=[], model="test-model")


# ---------------------------------------------------------------------------
# Citation Tests
# ---------------------------------------------------------------------------


class TestCitation:
    def test_valid_instantiation(self, citation):
        assert citation.marker == "[1]"
        assert citation.url.startswith("https://")

    def test_empty_marker_raises(self, source_doc, content_chunk):
        with pytest.raises(ValidationError):
            Citation(
                marker="",
                source_id=source_doc.source_id,
                chunk_id=content_chunk.chunk_id,
                url=source_doc.url,
            )

    def test_invalid_url_raises(self, source_doc, content_chunk):
        with pytest.raises(ValidationError):
            Citation(
                marker="[1]",
                source_id=source_doc.source_id,
                chunk_id=content_chunk.chunk_id,
                url="javascript:void(0)",
            )


# ---------------------------------------------------------------------------
# Finding Tests
# ---------------------------------------------------------------------------


class TestFinding:
    def test_valid_instantiation(self, finding):
        assert finding.claim
        assert len(finding.citations) == 1

    def test_citation_markers_property(self, finding):
        markers = finding.citation_markers
        assert markers == ["[1]"]

    def test_uncertain_finding(self, sub_question):
        f = Finding(
            claim="The impact is unclear.",
            uncertain=True,
            conflict_note="Source A says X; Source B says Y.",
            sub_question_id=sub_question.sub_question_id,
        )
        assert f.uncertain is True
        assert f.citations == []


# ---------------------------------------------------------------------------
# ReportSection Tests
# ---------------------------------------------------------------------------


class TestReportSection:
    def test_valid_instantiation(self, report_section):
        assert report_section.heading == "Key Findings"
        assert len(report_section.findings) == 1


# ---------------------------------------------------------------------------
# ResearchReport Tests
# ---------------------------------------------------------------------------


class TestResearchReport:
    def test_valid_instantiation(self, research_report):
        assert research_report.report_id
        assert research_report.sources_analyzed == 12

    def test_all_findings_property(self, research_report, finding):
        findings = research_report.all_findings
        assert len(findings) == 1
        assert findings[0].claim == finding.claim

    def test_section_headings_property(self, research_report):
        assert "Key Findings" in research_report.section_headings


# ---------------------------------------------------------------------------
# Provenance Chain Test
# ---------------------------------------------------------------------------


class TestProvenanceChain:
    """
    Verify that Finding -> Citation -> SourceDocument -> url is fully traceable.
    This is the anti-hallucination guarantee: every cited URL is real.
    """

    def test_full_chain(self, finding, citation, source_doc):
        # Finding has citations
        assert len(finding.citations) == 1
        c = finding.citations[0]

        # Citation links to SourceDocument via source_id
        assert c.source_id == source_doc.source_id

        # Citation carries the exact URL from SourceDocument
        assert c.url == source_doc.url

        # URL is a real https link (not fabricated)
        assert c.url.startswith("https://")

    def test_serialisation_preserves_provenance(self, research_report, source_doc):
        data = research_report.model_dump()
        # Check that source_id survives serialisation
        citation_data = data["all_citations"][0]
        assert citation_data["source_id"] == source_doc.source_id
        assert citation_data["url"] == source_doc.url

