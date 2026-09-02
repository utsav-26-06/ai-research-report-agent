# Architecture

## Overview

The AI Research & Report Agent is a fully autonomous, multi-stage RAG pipeline that transforms a plain-text research topic into a professionally formatted, citation-backed research report.

```
User Topic
    │
    ▼
┌─────────────────────┐
│   QueryPlanner       │  LLM decomposes topic into 4-6 SubQuestions + search queries
└─────────┬───────────┘
          │
    ┌─────▼─────┐   ┌─────────────┐
    │  Searcher  │──►│ TavilyAPI   │  Concurrent web search per sub-question
    └─────┬─────┘   └─────────────┘
          │
    ┌─────▼─────┐
    │ Extractor  │  Trafilatura + BeautifulSoup (async, thread-safe)
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ Evaluator  │  LLM-based scoring: relevance, credibility, recency, redundancy
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │  Chunker   │  LangChain RecursiveCharacterTextSplitter
    └─────┬─────┘
          │
    ┌─────▼───────────┐
    │ EmbeddingProvider│  Gemini text-embedding-004 (768-dim, batched)
    └─────┬───────────┘
          │
    ┌─────▼───────┐
    │ ChromaDB     │  Persistent local vector store (cosine similarity)
    └─────┬───────┘
          │
    ┌─────▼──────────┐
    │ SemanticRetriever│  Top-N chunks per sub-question
    └─────┬──────────┘
          │
    ┌─────▼──────────┐
    │  RAGSummarizer  │  Grounded LLM synthesis → Finding + Citations
    └─────┬──────────┘
          │
    ┌─────▼──────────┐
    │CitationManager  │  Global deduplication + sequential markers [1], [2]...
    └─────┬──────────┘
          │
    ┌─────▼──────────┐
    │  ReportBuilder  │  LLM writes narrative sections (Introduction, Analysis, Conclusion)
    └─────┬──────────┘
          │
    ┌─────▼────────────────┐
    │  DocxExporter         │  python-docx → .docx
    │  PDFExporter          │  ReportLab Platypus → .pdf
    └─────┬────────────────┘
          │
    ┌─────▼───────┐
    │ Streamlit UI │  Web interface with download buttons
    └─────────────┘
```

## Module Layout

| Package | Purpose |
|---------|---------|
| `app/agent/` | Pipeline orchestration (QueryPlanner, ResearchController) |
| `app/config/` | Settings (pydantic-settings), logging config |
| `app/models/` | Pydantic domain models (research, sources, rag, report) |
| `app/tools/search/` | Web search providers (Tavily) |
| `app/tools/extraction/` | Content extraction (Trafilatura) |
| `app/tools/evaluation/` | LLM-based source scoring |
| `app/rag/` | Chunker, Embedder, ChromaVectorStore, SemanticRetriever |
| `app/generation/` | RAGSummarizer, CitationManager, ReportBuilder, LLMProvider |
| `app/export/` | DocxExporter, PDFExporter |
| `app/ui/` | Streamlit web interface |

## Data Flow & Provenance

Every piece of data carries its provenance chain:

```
SearchResult.sub_question_id
    → SourceDocument.sub_question_id + source_id
        → ContentChunk.sub_question_id + source_id + chunk_id
            → EmbeddedChunk.chunk (ContentChunk preserved)
                → Citation.source_id + chunk_id + url
                    → Finding.citations
                        → ReportSection.findings
                            → ResearchReport.all_citations
```

This guarantees every claim in the report can be traced back to the exact sentence in the exact URL that was retrieved from the web.

## Concurrency Model

- Search: `asyncio.gather` behind a `Semaphore(3)` — up to 3 parallel queries
- Extraction: `asyncio.gather` behind a `Semaphore(5)` — up to 5 concurrent fetches
- ChromaDB / Embedding: `asyncio.to_thread` — sync ops run in a thread pool
- LLM calls: Sequential (avoids rate limit bursts)

## Error Handling

Each layer has its own exception class defined in `app/exceptions.py`. The `ResearchController` catches layer-specific failures at the source level (individual sources) and only raises a `ResearchControllerError` if the entire pipeline is unrecoverable (e.g., planning fails).
