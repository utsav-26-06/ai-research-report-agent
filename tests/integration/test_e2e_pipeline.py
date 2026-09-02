"""
End-to-End pipeline test (TASK-019).
Verifies the complete pipeline from topic input to PDF/DOCX output
using fully mocked LLM, search, and extraction.
"""

from __future__ import annotations

import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from app.agent.research_controller import ResearchController
from app.models.rag import ContentChunk
from tests.fixtures.mock_responses import (
    make_research_request,
    make_plan,
    make_search_result,
    make_source_doc,
    make_evaluation,
    make_content_chunk,
    make_finding,
    make_report,
)


@pytest.fixture
def e2e_controller(tmp_path):
    """Build a fully mocked ResearchController backed by in-memory mock fixtures."""
    request = make_research_request()
    plan = make_plan(request)

    # --- Planner ---
    planner = MagicMock()
    planner.generate_plan = AsyncMock(return_value=plan)

    # --- Searcher: one result per sub-question ---
    search_results = [
        make_search_result(sq_id="sq-1", idx=0),
        make_search_result(sq_id="sq-2", idx=1),
    ]
    searcher = MagicMock()
    searcher.search = AsyncMock(side_effect=lambda query, sub_question_id: [
        make_search_result(sub_question_id, int(sub_question_id.split("-")[1]))
    ])

    # --- Extractor ---
    extractor = MagicMock()
    extractor.extract = AsyncMock(side_effect=lambda url, sub_question_id, query="": 
        make_source_doc(sub_question_id, int(sub_question_id.split("-")[1]))
    )

    # --- Evaluator ---
    evaluator = MagicMock()
    evaluator.evaluate = AsyncMock(side_effect=lambda doc, query, already_included=None:
        make_evaluation(doc.source_id, doc.url)
    )

    # --- Chunker: returns pre-built chunks ---
    chunker = MagicMock()
    chunks = [make_content_chunk("sq-1", 0), make_content_chunk("sq-2", 1)]
    chunker.chunk_documents = MagicMock(return_value=chunks)

    # --- Embedding provider ---
    embed_provider = MagicMock()
    embedded = [MagicMock(chunk=c, embedding=[0.1] * 10, chunk_id=c.chunk_id, sub_question_id=c.sub_question_id) for c in chunks]
    embed_provider.embed_chunks = AsyncMock(return_value=embedded)
    embed_provider.model_name = "models/text-embedding-004"

    # --- Vector store ---
    vector_store = MagicMock()
    vector_store.clear = AsyncMock()
    vector_store.add = AsyncMock()

    # --- Retriever ---
    retriever = MagicMock()

    # --- Summarizer: one finding per sub-question ---
    findings = [make_finding("sq-1", chunks[0]), make_finding("sq-2", chunks[1])]
    summarizer = MagicMock()
    call_count = [0]
    async def _summarize(sq):
        idx = call_count[0] % len(findings)
        call_count[0] += 1
        return findings[idx]
    summarizer.summarize = _summarize

    # --- Report builder ---
    mock_report = make_report(findings)
    report_builder = MagicMock()
    report_builder.build = AsyncMock(return_value=mock_report)

    # --- Exporters ---
    docx_exporter = MagicMock()
    docx_exporter.export = AsyncMock(return_value=tmp_path / "report.docx")
    pdf_exporter = MagicMock()
    pdf_exporter.export = AsyncMock(return_value=tmp_path / "report.pdf")

    return ResearchController(
        planner=planner,
        searcher=searcher,
        extractor=extractor,
        evaluator=evaluator,
        chunker=chunker,
        embedding_provider=embed_provider,
        vector_store=vector_store,
        retriever=retriever,
        summarizer=summarizer,
        report_builder=report_builder,
        docx_exporter=docx_exporter,
        pdf_exporter=pdf_exporter,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_report_has_all_sections(e2e_controller):
    """Full pipeline should produce a ResearchReport with all 5 sections."""
    request = make_research_request()
    result = await e2e_controller.run(request)

    report = result.report
    assert report.title == "The Future of AI: Research Report"
    assert report.research_question == "The Future of AI"

    headings = [s.heading for s in report.sections]
    assert "Introduction" in headings
    assert "Key Findings" in headings
    assert "Analysis" in headings
    assert "Conclusion" in headings
    assert "References" in headings


@pytest.mark.asyncio
async def test_e2e_output_files_created(e2e_controller):
    """Pipeline should return DOCX and PDF file paths."""
    request = make_research_request()
    result = await e2e_controller.run(request)

    assert result.docx_path is not None, "DOCX path should not be None"
    assert result.pdf_path is not None, "PDF path should not be None"
    assert str(result.docx_path).endswith(".docx")
    assert str(result.pdf_path).endswith(".pdf")


@pytest.mark.asyncio
async def test_e2e_citation_provenance_chain(e2e_controller):
    """Every citation must have a valid URL and source_id linking back to a source."""
    request = make_research_request()
    result = await e2e_controller.run(request)

    report = result.report
    kf_section = next(s for s in report.sections if s.heading == "Key Findings")

    for finding in kf_section.findings:
        for citation in finding.citations:
            # Marker must be non-empty
            assert citation.marker, f"Citation missing marker: {citation}"
            # URL must be valid
            assert citation.url.startswith("http"), f"Invalid URL in citation: {citation.url}"
            # source_id must be set
            assert citation.source_id, f"Citation missing source_id"
            # chunk_id must be set
            assert citation.chunk_id, f"Citation missing chunk_id"


@pytest.mark.asyncio
async def test_e2e_with_progress_tracking(e2e_controller):
    """Progress callback should be called for all expected stages."""
    stages = []
    def on_progress(stage: str, message: str):
        stages.append(stage)

    request = make_research_request()
    await e2e_controller.run(request, progress=on_progress)

    required_stages = {"Planning", "Search", "Extraction", "Evaluation",
                       "Chunking", "Embedding", "Summarization", "Citations",
                       "Report", "Export", "Done"}
    missing = required_stages - set(stages)
    assert not missing, f"Missing progress stages: {missing}"
