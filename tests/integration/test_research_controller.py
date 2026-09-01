"""
Integration tests for ResearchController (TASK-018).
All external dependencies are fully mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from app.agent.research_controller import ResearchController, ResearchControllerError
from app.models.research import ResearchRequest, ResearchPlan, SubQuestion
from app.models.sources import SearchResult, SourceDocument, SourceEvaluation
from app.models.report import Citation, Finding, ReportSection, ResearchReport


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _make_sub_question(n: int = 1) -> SubQuestion:
    return SubQuestion(
        sub_question_id=f"sq-{n}",
        text=f"Sub-question {n}?",
        search_queries=[f"query {n}a", f"query {n}b"],
        priority=n,
    )


def _make_plan() -> ResearchPlan:
    req = ResearchRequest(topic="AI", depth="standard")
    return ResearchPlan(
        request=req,
        research_objective="Understand AI trends.",
        sub_questions=[_make_sub_question(1), _make_sub_question(2)],
    )


def _make_source_doc(url: str = "http://example.com") -> SourceDocument:
    return SourceDocument(
        source_id="src-1",
        url=url,
        title="Example",
        domain="example.com",
        sub_question_id="sq-1",
        text="Some very informative content about AI.",
    )


def _make_report() -> ResearchReport:
    return ResearchReport(
        title="AI Report",
        research_question="AI",
        depth="standard",
        executive_summary="AI is growing.",
        sections=[
            ReportSection(heading="Introduction", content="Intro", order=0),
            ReportSection(heading="References", content="[1] http://x.com", order=1),
        ],
    )


@pytest.fixture
def mocks():
    """Return a dict of all mocked pipeline components."""
    planner = MagicMock(); planner.generate_plan = AsyncMock(return_value=_make_plan())
    searcher = MagicMock(); searcher.search = AsyncMock(return_value=[
        SearchResult(url="http://a.com", title="A", snippet="AI stuff", sub_question_id="sq-1", query="query 1a")
    ])
    extractor = MagicMock(); extractor.extract = AsyncMock(return_value=_make_source_doc("http://a.com"))
    evaluator = MagicMock(); evaluator.evaluate = AsyncMock(return_value=SourceEvaluation(
        source_id="src-1", url="http://a.com",
        relevance_score=0.8, credibility_score=0.8,
        recency_score=0.8, redundancy_score=0.1,
        overall_score=0.8, decision="include"
    ))
    chunker = MagicMock()
    chunker.chunk_documents = MagicMock(return_value=[])
    embed_provider = MagicMock()
    embed_provider.embed_chunks = AsyncMock(return_value=[])
    embed_provider.model_name = "models/text-embedding-004"
    vector_store = MagicMock()
    vector_store.clear = AsyncMock()
    vector_store.add = AsyncMock()
    retriever = MagicMock()
    
    finding = Finding(claim="AI is growing quickly.", evidence="Ev.", citations=[], sub_question_id="sq-1")
    summarizer = MagicMock(); summarizer.summarize = AsyncMock(return_value=finding)
    
    report_builder = MagicMock(); report_builder.build = AsyncMock(return_value=_make_report())
    docx_exporter = MagicMock(); docx_exporter.export = AsyncMock(return_value=Path("/tmp/report.docx"))
    pdf_exporter = MagicMock(); pdf_exporter.export = AsyncMock(return_value=Path("/tmp/report.pdf"))
    
    return {
        "planner": planner, "searcher": searcher, "extractor": extractor,
        "evaluator": evaluator, "chunker": chunker, "embedding_provider": embed_provider,
        "vector_store": vector_store, "retriever": retriever, "summarizer": summarizer,
        "report_builder": report_builder, "docx_exporter": docx_exporter, "pdf_exporter": pdf_exporter,
    }


@pytest.fixture
def controller(mocks):
    return ResearchController(**mocks)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_returns_result(controller, mocks):
    """Full pipeline should return a ResearchResult with a report and file paths."""
    request = ResearchRequest(topic="AI", depth="standard")
    result = await controller.run(request)

    assert result.report.title == "AI Report"
    assert result.docx_path == Path("/tmp/report.docx")
    assert result.pdf_path == Path("/tmp/report.pdf")

    # Verify key stages were called
    mocks["planner"].generate_plan.assert_called_once()
    mocks["searcher"].search.assert_called()
    mocks["extractor"].extract.assert_called()
    mocks["evaluator"].evaluate.assert_called()
    mocks["summarizer"].summarize.assert_called()
    mocks["report_builder"].build.assert_called_once()
    mocks["docx_exporter"].export.assert_called_once()
    mocks["pdf_exporter"].export.assert_called_once()


@pytest.mark.asyncio
async def test_failed_source_extraction_is_skipped(controller, mocks):
    """If extraction fails for a source, pipeline continues with remaining sources."""
    mocks["extractor"].extract.side_effect = Exception("Connection timeout")

    request = ResearchRequest(topic="AI", depth="standard")
    # Should not raise — failed extractions are logged and skipped
    result = await controller.run(request)

    assert result.report is not None
    mocks["report_builder"].build.assert_called_once()


@pytest.mark.asyncio
async def test_failed_summarization_is_skipped(controller, mocks):
    """If summarization fails for one sub-question, pipeline continues."""
    mocks["summarizer"].summarize.side_effect = Exception("LLM timeout")

    request = ResearchRequest(topic="AI", depth="standard")
    result = await controller.run(request)

    # Report should still be built, just with no findings
    assert result.report is not None
    mocks["report_builder"].build.assert_called_once()


@pytest.mark.asyncio
async def test_progress_callbacks_fired(controller):
    """Progress callback should be called for each pipeline stage."""
    stages_seen = []

    def track_progress(stage: str, message: str):
        stages_seen.append(stage)

    request = ResearchRequest(topic="AI", depth="standard")
    await controller.run(request, progress=track_progress)

    expected_stages = {"Start", "Planning", "Search", "Extraction", "Evaluation",
                       "Chunking", "Embedding", "Summarization", "Citations", "Report", "Export", "Done"}
    assert expected_stages.issubset(set(stages_seen)), \
        f"Missing stages: {expected_stages - set(stages_seen)}"


@pytest.mark.asyncio
async def test_planning_failure_raises_controller_error(controller, mocks):
    """A fatal planning failure should surface as ResearchControllerError."""
    mocks["planner"].generate_plan.side_effect = Exception("Planning API down")

    request = ResearchRequest(topic="AI", depth="standard")
    with pytest.raises(ResearchControllerError, match="Planning failed"):
        await controller.run(request)



