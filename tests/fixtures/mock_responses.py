"""
Shared mock responses and fixtures for E2E tests (TASK-019).
"""

from app.models.research import ResearchRequest, ResearchPlan, SubQuestion
from app.models.sources import SearchResult, SourceDocument, SourceEvaluation
from app.models.rag import ContentChunk, EmbeddedChunk
from app.models.report import Citation, Finding, ReportSection, ResearchReport


def make_research_request() -> ResearchRequest:
    return ResearchRequest(topic="The Future of AI", depth="standard")


def make_sub_question(n: int = 1) -> SubQuestion:
    return SubQuestion(
        sub_question_id=f"sq-{n}",
        text=f"What are the key trends in AI area {n}?",
        search_queries=[f"AI trends {n}", f"AI future {n}"],
        priority=n,
    )


def make_plan(request: ResearchRequest) -> ResearchPlan:
    return ResearchPlan(
        request=request,
        research_objective="Understand how AI is evolving.",
        sub_questions=[make_sub_question(1), make_sub_question(2)],
    )


def make_search_result(sq_id: str = "sq-1", idx: int = 0) -> SearchResult:
    return SearchResult(
        url=f"http://ai-news-{idx}.com/article",
        title=f"AI Article {idx}",
        snippet=f"AI is transforming everything in field {idx}.",
        sub_question_id=sq_id,
        query=f"AI trends {idx}",
    )


def make_source_doc(sq_id: str = "sq-1", idx: int = 0) -> SourceDocument:
    return SourceDocument(
        source_id=f"src-{idx}",
        url=f"http://ai-news-{idx}.com/article",
        title=f"AI Article {idx}",
        domain=f"ai-news-{idx}.com",
        sub_question_id=sq_id,
        text=f"AI is fundamentally changing industry sector {idx}. "
             f"New models demonstrate improved reasoning capabilities. "
             f"Researchers from leading institutions have verified these findings.",
    )


def make_evaluation(source_id: str = "src-0", url: str = "http://ai-news-0.com/article") -> SourceEvaluation:
    return SourceEvaluation(
        source_id=source_id,
        url=url,
        relevance_score=0.85,
        credibility_score=0.80,
        recency_score=0.75,
        redundancy_score=0.05,
        overall_score=0.81,
        decision="include",
        reason="High quality, relevant source.",
    )


def make_content_chunk(sq_id: str = "sq-1", idx: int = 0) -> ContentChunk:
    return ContentChunk(
        source_id=f"src-{idx}",
        url=f"http://ai-news-{idx}.com/article",
        source_title=f"AI Article {idx}",
        domain=f"ai-news-{idx}.com",
        sub_question_id=sq_id,
        chunk_index=0,
        total_chunks=1,
        text=f"AI is fundamentally changing industry sector {idx}.",
    )


def make_citation(chunk: ContentChunk, marker: str = "[1]") -> Citation:
    return Citation(
        marker=marker,
        source_id=chunk.source_id,
        chunk_id=chunk.chunk_id,
        url=chunk.url,
        title=chunk.source_title,
        domain=chunk.domain,
        excerpt=chunk.text[:80] + "...",
    )


def make_finding(sq_id: str = "sq-1", chunk: ContentChunk = None) -> Finding:
    if chunk is None:
        chunk = make_content_chunk(sq_id)
    cit = make_citation(chunk, marker="[1]")
    return Finding(
        claim=f"AI is advancing significantly in domain {sq_id}. [1]",
        evidence="AI is fundamentally changing industry sector.",
        citations=[cit],
        confidence=0.88,
        uncertain=False,
        sub_question_id=sq_id,
    )


def make_report(findings: list[Finding]) -> ResearchReport:
    all_citations = [c for f in findings for c in f.citations]
    return ResearchReport(
        title="The Future of AI: Research Report",
        research_question="The Future of AI",
        depth="standard",
        sources_analyzed=4,
        sources_included=2,
        executive_summary="AI is evolving rapidly across multiple domains.",
        sections=[
            ReportSection(heading="Introduction", content="AI is transforming the world.", order=0),
            ReportSection(heading="Key Findings", content="* AI advances\n* Model improvements", findings=findings, order=1),
            ReportSection(heading="Analysis", content="The evidence shows consistent improvement.", order=2),
            ReportSection(heading="Conclusion", content="AI will continue to reshape industries.", order=3),
            ReportSection(heading="References", content="\n".join(f"{c.marker} {c.url}" for c in all_citations), order=4),
        ],
        all_citations=all_citations,
    )
