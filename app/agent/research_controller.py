"""
Research Controller — main pipeline orchestrator (TASK-018).

Coordinates all stages from topic input to final exported reports:
  Plan -> Search -> Extract -> Evaluate -> Chunk -> Embed
       -> Store -> Retrieve -> Summarize -> Cite -> Build -> Export
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable

from app.export.docx_exporter import DocxExporter
from app.export.pdf_exporter import PDFExporter
from app.generation.citation_manager import CitationManager
from app.generation.rag_summarizer import RAGSummarizer
from app.generation.report_builder import ReportBuilder
from app.models.report import Finding, ResearchReport
from app.models.research import ResearchRequest
from app.models.sources import SourceDocument
from app.rag.chunker import DocumentChunker
from app.rag.embedding_provider import GeminiEmbeddingProvider
from app.rag.retriever import SemanticRetriever
from app.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# A progress callback: (stage_name, message) -> None (sync or async)
ProgressCallback = Callable[[str, str], None] | Callable[[str, str], Awaitable[None]]


@dataclass
class ResearchResult:
    """Container returned by ResearchController.run()."""
    report: ResearchReport
    docx_path: Path | None = None
    pdf_path: Path | None = None


class ResearchControllerError(Exception):
    """Raised when a fatal, unrecoverable pipeline error occurs."""


class ResearchController:
    """
    Orchestrates the full research pipeline from topic to exported report.

    All components are injected for testability. In production, use
    ResearchController.build_default() for the wired-up instance.
    """

    def __init__(
        self,
        planner,
        searcher,
        extractor,
        evaluator,
        chunker: DocumentChunker,
        embedding_provider: GeminiEmbeddingProvider,
        vector_store: ChromaVectorStore,
        retriever: SemanticRetriever,
        summarizer: RAGSummarizer,
        report_builder: ReportBuilder,
        docx_exporter: DocxExporter,
        pdf_exporter: PDFExporter,
        max_concurrent_searches: int = 3,
        max_concurrent_extractions: int = 5,
        min_accepted_sources: int = 1,
    ):
        self.planner = planner
        self.searcher = searcher
        self.extractor = extractor
        self.evaluator = evaluator
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.retriever = retriever
        self.summarizer = summarizer
        self.report_builder = report_builder
        self.docx_exporter = docx_exporter
        self.pdf_exporter = pdf_exporter
        self.max_concurrent_searches = max_concurrent_searches
        self.max_concurrent_extractions = max_concurrent_extractions
        self.min_accepted_sources = min_accepted_sources

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        request: ResearchRequest,
        *,
        progress: ProgressCallback | None = None,
        export_docx: bool = True,
        export_pdf: bool = True,
    ) -> ResearchResult:
        """
        Execute the full research pipeline.

        Args:
            request:     The user's research request.
            progress:    Optional callback(stage, message) for progress updates.
            export_docx: Whether to produce a .docx file.
            export_pdf:  Whether to produce a .pdf file.

        Returns:
            ResearchResult containing the report and optional file paths.
        """
        await self._emit(progress, "Start", f"Starting research: '{request.topic}'")

        # ----------------------------------------------------------
        # Stage 1: Planning
        # ----------------------------------------------------------
        await self._emit(progress, "Planning", "Generating research plan...")
        try:
            plan = await self.planner.generate_plan(request)
        except Exception as e:
            raise ResearchControllerError(f"Planning failed: {e}") from e
        await self._emit(progress, "Planning", f"Plan generated: {len(plan.sub_questions)} sub-questions.")

        # ----------------------------------------------------------
        # Stage 2: Search (concurrent, per sub-question)
        # ----------------------------------------------------------
        await self._emit(progress, "Search", "Running web searches...")
        search_sem = asyncio.Semaphore(self.max_concurrent_searches)
        search_results = await self._search_all(plan, search_sem, progress)
        await self._emit(progress, "Search", f"Collected {len(search_results)} search results.")

        # ----------------------------------------------------------
        # Stage 3: Extract (concurrent)
        # ----------------------------------------------------------
        await self._emit(progress, "Extraction", "Extracting page content...")
        ext_sem = asyncio.Semaphore(self.max_concurrent_extractions)
        raw_docs = await self._extract_all(search_results, ext_sem, progress)
        await self._emit(progress, "Extraction", f"Extracted {len(raw_docs)} documents.")

        # ----------------------------------------------------------
        # Stage 4: Evaluate & Filter
        # ----------------------------------------------------------
        await self._emit(progress, "Evaluation", "Evaluating source quality...")
        accepted_docs = await self._evaluate_all(raw_docs, request.topic, progress)
        sources_analyzed = len(raw_docs)
        sources_included = len(accepted_docs)
        await self._emit(
            progress, "Evaluation",
            f"Accepted {sources_included}/{sources_analyzed} sources."
        )

        if sources_included < self.min_accepted_sources:
            logger.warning("Too few sources accepted; proceeding with what is available.")

        # ----------------------------------------------------------
        # Stage 5: Chunk
        # ----------------------------------------------------------
        await self._emit(progress, "Chunking", "Splitting documents into chunks...")
        all_chunks = self.chunker.chunk_documents(accepted_docs)
        await self._emit(progress, "Chunking", f"{len(all_chunks)} chunks created.")

        # ----------------------------------------------------------
        # Stage 6: Embed + Store
        # ----------------------------------------------------------
        await self._emit(progress, "Embedding", "Embedding and storing chunks...")
        await self.vector_store.clear()
        if all_chunks:
            embedded = await self.embedding_provider.embed_chunks(all_chunks)
            await self.vector_store.add(embedded)
        await self._emit(progress, "Embedding", f"{len(all_chunks)} chunks embedded and stored.")

        # ----------------------------------------------------------
        # Stage 7: RAG Summarize (one Finding per SubQuestion)
        # ----------------------------------------------------------
        await self._emit(progress, "Summarization", "Synthesizing findings...")
        findings: list[Finding] = []
        for sq in plan.sub_questions:
            try:
                finding = await self.summarizer.summarize(sq)
                findings.append(finding)
                await self._emit(progress, "Summarization", f"Finding generated: {sq.text[:60]}...")
            except Exception as e:
                logger.warning(f"Summarization failed for '{sq.text}': {e}. Skipping.")
                await self._emit(progress, "Summarization", f"⚠ Skipped (error): {sq.text[:60]}...")

        # ----------------------------------------------------------
        # Stage 8: Citation Management
        # ----------------------------------------------------------
        await self._emit(progress, "Citations", "Deduplicating citations...")
        citation_manager = CitationManager()
        citation_manager.process_findings(findings)
        await self._emit(
            progress, "Citations",
            f"{len(citation_manager.all_citations)} unique citations."
        )

        # ----------------------------------------------------------
        # Stage 9: Build Report
        # ----------------------------------------------------------
        await self._emit(progress, "Report", "Assembling research report...")
        try:
            report = await self.report_builder.build(request, findings, citation_manager)
        except Exception as e:
            raise ResearchControllerError(f"Report building failed: {e}") from e
        report.sources_analyzed = sources_analyzed
        report.sources_included = sources_included
        await self._emit(progress, "Report", f"Report built: '{report.title}'")

        # ----------------------------------------------------------
        # Stage 10: Export
        # ----------------------------------------------------------
        docx_path: Path | None = None
        pdf_path: Path | None = None

        if export_docx:
            await self._emit(progress, "Export", "Exporting DOCX...")
            try:
                docx_path = await self.docx_exporter.export(report)
                await self._emit(progress, "Export", f"DOCX saved: {docx_path}")
            except Exception as e:
                logger.error(f"DOCX export failed: {e}")

        if export_pdf:
            await self._emit(progress, "Export", "Exporting PDF...")
            try:
                pdf_path = await self.pdf_exporter.export(report)
                await self._emit(progress, "Export", f"PDF saved: {pdf_path}")
            except Exception as e:
                logger.error(f"PDF export failed: {e}")

        await self._emit(progress, "Done", "Research complete!")
        return ResearchResult(report=report, docx_path=docx_path, pdf_path=pdf_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _search_all(self, plan, sem, progress):
        """Run all search queries concurrently."""
        from app.models.sources import SearchResult

        async def _search_one(query: str, sq_id: str):
            async with sem:
                try:
                    return await self.searcher.search(query=query, sub_question_id=sq_id)
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    return []

        tasks = [
            _search_one(query, sq.sub_question_id)
            for sq in plan.sub_questions
            for query in sq.search_queries
        ]
        nested = await asyncio.gather(*tasks)
        # Flatten and deduplicate by URL
        seen_urls: set[str] = set()
        results = []
        for batch in nested:
            for r in batch:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    results.append(r)
        return results

    async def _extract_all(self, search_results, sem, progress):
        """Extract page content for all search results concurrently."""
        async def _extract_one(r):
            async with sem:
                try:
                    doc = await self.extractor.extract(
                        url=r.url,
                        sub_question_id=r.sub_question_id,
                        query=r.query if hasattr(r, "query") else "",
                    )
                    return doc
                except Exception as e:
                    logger.warning(f"Extraction failed for {r.url}: {e}")
                    return None

        tasks = [_extract_one(r) for r in search_results]
        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc is not None]

    async def _evaluate_all(self, docs: list[SourceDocument], topic: str, progress):
        """Evaluate sources sequentially (avoids LLM rate-limit bursts)."""
        accepted = []
        for doc in docs:
            try:
                evaluation = await self.evaluator.evaluate(doc=doc, query=topic, already_included=accepted)
                if evaluation.is_included:
                    accepted.append(doc)
            except Exception as e:
                logger.warning(f"Evaluation failed for {doc.url}: {e}. Skipping source.")
        return accepted

    @staticmethod
    async def _emit(callback: ProgressCallback | None, stage: str, message: str):
        """Fire the progress callback (sync or async)."""
        if callback is None:
            return
        try:
            result = callback(stage, message)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.debug(f"Progress callback error: {e}")

