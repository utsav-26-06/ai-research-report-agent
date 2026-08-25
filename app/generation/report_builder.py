"""
Report Builder module (TASK-015).
Assembles findings into a complete ResearchReport.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.generation.base import LLMProvider, LLMProviderError
from app.generation.citation_manager import CitationManager
from app.models.report import Finding, ReportSection, ResearchReport
from app.models.research import ResearchRequest

logger = logging.getLogger(__name__)


class ReportBuilderError(Exception):
    """Raised when report generation fails."""


class _ReportContent(BaseModel):
    """Structured response expected from the LLM for report synthesis."""
    title: str = Field(..., description="Professional title for the report.")
    executive_summary: str = Field(..., description="Short high-level summary of all findings.")
    introduction: str = Field(..., description="Markdown introduction to the topic based on findings.")
    analysis: str = Field(..., description="Markdown analysis synthesizing the findings.")
    conclusion: str = Field(..., description="Markdown conclusion summarizing the outcomes.")


class ReportBuilder:
    """
    Synthesizes the final research report from a set of findings.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def build(
        self,
        request: ResearchRequest,
        findings: list[Finding],
        citation_manager: CitationManager,
    ) -> ResearchReport:
        """
        Assemble the final report using the provided LLM and findings.
        
        Args:
            request: Original user request.
            findings: Deduplicated findings with in-text citation markers.
            citation_manager: The citation manager holding unique citations.
            
        Returns:
            A fully populated ResearchReport.
        """
        logger.info(f"Building report for '{request.topic}' using {len(findings)} findings.")

        # 1. Build context string for the LLM
        findings_context = []
        for i, f in enumerate(findings):
            findings_context.append(f"Finding {i + 1}: {f.claim}\nEvidence: {f.evidence}")
        context_str = "\n\n".join(findings_context)

        prompt = (
            f"You are an expert research analyst writing a formal report.\n"
            f"Write the narrative sections of the report based ONLY on the findings below.\n"
            f"Do not introduce outside knowledge or fabricate facts.\n\n"
            f"Topic: {request.topic}\n"
            f"Research Objective: {request.topic}\n\n"
            f"Findings:\n{context_str}\n\n"
            f"Generate a professional title, an executive summary, an introduction, "
            f"an analysis section synthesizing these findings, and a conclusion."
        )

        # 2. Call LLM
        try:
            schema = _ReportContent.model_json_schema()
            raw_output = await self.llm.structured_complete(
                prompt=prompt,
                schema=schema,
                temperature=0.3
            )
            report_content = _ReportContent.model_validate(raw_output)
        except Exception as e:
            logger.error(f"Failed to generate report narrative: {e}")
            raise ReportBuilderError(f"Failed to generate report narrative: {e}") from e

        # 3. Assemble Key Findings Section locally
        # We present the claims as a bulleted markdown list
        kf_lines = ["The following key findings were identified:\n"]
        for f in findings:
            kf_lines.append(f"* {f.claim}")
        key_findings_content = "\n".join(kf_lines)

        # 4. Construct Sections
        sections = [
            ReportSection(
                heading="Introduction",
                content=report_content.introduction,
                findings=[],
                order=0
            ),
            ReportSection(
                heading="Key Findings",
                content=key_findings_content,
                findings=findings,  # Attach actual finding models here
                order=1
            ),
            ReportSection(
                heading="Analysis",
                content=report_content.analysis,
                findings=[],
                order=2
            ),
            ReportSection(
                heading="Conclusion",
                content=report_content.conclusion,
                findings=[],
                order=3
            ),
            ReportSection(
                heading="References",
                content=citation_manager.get_reference_list(),
                findings=[],
                order=4
            )
        ]

        # 5. Return Final Report Model
        return ResearchReport(
            title=report_content.title,
            research_question=request.topic,
            depth=request.depth,
            executive_summary=report_content.executive_summary,
            sections=sections,
            all_citations=citation_manager.all_citations,
            methodology="Retrieval-Augmented Generation (RAG) using automated web search and LLM synthesis."
        )
