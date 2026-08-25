"""
Unit tests for ReportBuilder (TASK-015).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.generation.report_builder import ReportBuilder, ReportBuilderError, _ReportContent
from app.generation.citation_manager import CitationManager
from app.models.research import ResearchRequest
from app.models.report import Finding, Citation


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.structured_complete = AsyncMock()
    return llm


@pytest.fixture
def builder(mock_llm):
    return ReportBuilder(llm=mock_llm)


@pytest.fixture
def research_req():
    return ResearchRequest(topic="Quantum Computing", depth="standard")


@pytest.fixture
def findings():
    return [
        Finding(claim="Claim A [1]", evidence="Ev A", citations=[
            Citation(marker="[1]", source_id="s1", chunk_id="c1", url="http://a.com")
        ]),
        Finding(claim="Claim B [2]", evidence="Ev B", citations=[
            Citation(marker="[2]", source_id="s2", chunk_id="c2", url="http://b.com")
        ])
    ]


@pytest.fixture
def citation_manager(findings):
    manager = CitationManager()
    manager.process_findings(findings)
    return manager


@pytest.mark.asyncio
async def test_build_creates_all_sections(builder, mock_llm, research_req, findings, citation_manager):
    # Mock LLM response
    mock_llm.structured_complete.return_value = {
        "title": "Quantum Report",
        "executive_summary": "Exec summary.",
        "introduction": "Intro content.",
        "analysis": "Analysis content.",
        "conclusion": "Conclusion content."
    }

    report = await builder.build(research_req, findings, citation_manager)

    assert report.title == "Quantum Report"
    assert report.executive_summary == "Exec summary."
    assert report.research_question == "Quantum Computing"
    
    # 5 sections (Intro, Key Findings, Analysis, Conclusion, References)
    assert len(report.sections) == 5
    
    headings = [s.heading for s in report.sections]
    assert headings == ["Introduction", "Key Findings", "Analysis", "Conclusion", "References"]

    # Check Key Findings content
    kf_section = report.sections[1]
    assert kf_section.findings == findings
    assert "Claim A" in kf_section.content
    assert "Claim B" in kf_section.content

    # Check References content
    ref_section = report.sections[4]
    assert "http://a.com" in ref_section.content
    assert "http://b.com" in ref_section.content
    
    # Check citations mapped
    assert len(report.all_citations) == 2


@pytest.mark.asyncio
async def test_llm_error_raises_builder_error(builder, mock_llm, research_req, findings, citation_manager):
    mock_llm.structured_complete.side_effect = Exception("API failure")

    with pytest.raises(ReportBuilderError, match="Failed to generate report narrative"):
        await builder.build(research_req, findings, citation_manager)


