"""
Report output models - citations, findings, sections, and the final report.

Traceability chain:
    Citation.source_id  -> SourceDocument.source_id -> url
    Citation.chunk_id   -> ContentChunk.chunk_id
    Finding.citations   -> list[Citation]
    ReportSection.findings -> list[Finding]
    ResearchReport.sections -> list[ReportSection]
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Citation(BaseModel):
    """
    A reference to a specific source used to support a claim.

    Attributes:
        citation_id:  Stable UUID for this citation instance.
        marker:       In-text marker displayed to the reader, e.g. "[1]".
        source_id:    Links to SourceDocument.source_id (provenance).
        chunk_id:     Links to ContentChunk.chunk_id (granular provenance).
        url:          Exact URL - never fabricated, always from a retrieved source.
        title:        Title of the source page.
        domain:       Apex domain.
        excerpt:      Short quote from the source that supports the claim.
    """

    citation_id: str = Field(default_factory=_new_id)
    marker: str = Field(..., description="In-text marker, e.g. [1] or [S1].")
    source_id: str = Field(
        ..., description="Links to SourceDocument.source_id for provenance."
    )
    chunk_id: str = Field(
        ..., description="Links to ContentChunk.chunk_id for granular provenance."
    )
    url: str = Field(
        ..., description="Exact URL - must come from a retrieved SourceDocument."
    )
    title: str = Field(default="")
    domain: str = Field(default="")
    excerpt: str = Field(
        default="",
        description="Short supporting quote from the source.",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("Citation url must start with http:// or https://")
        return v

    @field_validator("marker")
    @classmethod
    def validate_marker(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Citation marker must not be empty")
        return v


class Finding(BaseModel):
    """
    A single research finding supported by retrieved evidence.

    IMPORTANT: Every factual claim must have at least one Citation.
    If evidence is insufficient, use the 'uncertain' flag and state so explicitly.

    Attributes:
        finding_id:      Stable UUID.
        claim:           The factual statement being made.
        evidence:        Direct quote or close paraphrase from source(s).
        citations:       List of Citations supporting this finding.
        confidence:      0.0-1.0 - model confidence given the available evidence.
        uncertain:       True if evidence was insufficient or conflicting.
        conflict_note:   Description of conflicting evidence (if any).
        sub_question_id: The sub-question this finding addresses.
    """

    finding_id: str = Field(default_factory=_new_id)
    claim: str = Field(..., min_length=5, description="The factual statement.")
    evidence: str = Field(
        default="",
        description="Supporting evidence from retrieved sources.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations backing this finding. Must be non-empty for factual claims.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence given available evidence.",
    )
    uncertain: bool = Field(
        default=False,
        description="True if evidence is insufficient or conflicting.",
    )
    conflict_note: str = Field(
        default="",
        description="Describes conflicting evidence when uncertain=True.",
    )
    sub_question_id: str = Field(
        default="",
        description="The SubQuestion.sub_question_id this finding addresses.",
    )

    @property
    def citation_markers(self) -> list[str]:
        """Return list of in-text markers for all citations."""
        return [c.marker for c in self.citations]


class ReportSection(BaseModel):
    """
    A single named section of the research report.

    Attributes:
        heading:  Section heading (e.g. "Key Findings").
        content:  Section body text in Markdown.
        findings: Findings contained in or referenced by this section.
        order:    Display order within the report (0 = first).
    """

    heading: str = Field(..., min_length=1)
    content: str = Field(default="")
    findings: list[Finding] = Field(default_factory=list)
    order: int = Field(default=0, ge=0)


class ResearchReport(BaseModel):
    """
    The final structured research report.

    Attributes:
        report_id:          Stable UUID.
        title:              Report title.
        research_question:  The original research topic/question.
        research_date:      UTC timestamp when the report was generated.
        depth:              Research depth mode used.
        sources_analyzed:   Total number of sources evaluated.
        sources_included:   Number of sources that passed evaluation.
        executive_summary:  Short high-level summary of findings.
        sections:           Ordered list of report sections.
        all_citations:      Deduplicated list of all citations used.
        limitations:        Honest statement of research limitations.
        methodology:        Brief description of how the research was conducted.
    """

    report_id: str = Field(default_factory=_new_id)
    title: str = Field(..., min_length=3)
    research_question: str = Field(...)
    research_date: datetime = Field(default_factory=_now)
    depth: str = Field(default="standard")
    sources_analyzed: int = Field(default=0, ge=0)
    sources_included: int = Field(default=0, ge=0)
    executive_summary: str = Field(default="")
    sections: list[ReportSection] = Field(default_factory=list)
    all_citations: list[Citation] = Field(
        default_factory=list,
        description="Deduplicated master citation list ordered by marker.",
    )
    limitations: str = Field(default="")
    methodology: str = Field(default="")

    @property
    def all_findings(self) -> list[Finding]:
        """Flatten findings from all sections."""
        return [f for s in self.sections for f in s.findings]

    @property
    def section_headings(self) -> list[str]:
        """Return headings of all sections in display order."""
        return [s.heading for s in sorted(self.sections, key=lambda s: s.order)]
