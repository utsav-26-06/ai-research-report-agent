# Research & Report Agent

An autonomous AI-powered research and report-generation system built for academic demonstration (MCA project).

## What It Does

Given a natural-language research topic, the agent automatically:

1. **Plans** the research by decomposing the topic into focused sub-questions
2. **Searches** the web using Tavily API
3. **Extracts** clean, readable content from discovered URLs
4. **Evaluates** sources on relevance, credibility, recency, and redundancy
5. **Builds a RAG knowledge base** using OpenAI Embeddings + ChromaDB
6. **Retrieves** relevant evidence for each finding
7. **Generates grounded findings** (no hallucinations — evidence-backed only)
8. **Manages citations** with full provenance tracking
9. **Produces a structured report** (Title / Introduction / Key Findings / Analysis / Conclusion / References)
10. **Exports** the report as **PDF** and **DOCX**

## Quick Start

### 1. Clone & set up environment

```bash
git clone <repo-url>
cd research-report-agent
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your real keys
```

You need:
- `OPENAI_API_KEY` — [platform.openai.com](https://platform.openai.com)
- `TAVILY_API_KEY` — [tavily.com](https://tavily.com)

### 3. Run the Streamlit UI

```bash
streamlit run app/ui/streamlit_app.py
```

### 4. Run tests

```bash
pytest
```

## Architecture

```
Streamlit UI
      ↓
Research Controller (Orchestrator)
      ↓
┌────────────────────────────────────┐
│           TOOL LAYER               │
│  QueryPlanner │ TavilySearch       │
│  TrafilaturaExtractor │ Evaluator  │
└────────────────────────────────────┘
      ↓
┌────────────────────────────────────┐
│           RAG LAYER                │
│  Chunker → Embeddings → ChromaDB   │
│               → Retrieval          │
└────────────────────────────────────┘
      ↓
┌────────────────────────────────────┐
│          OUTPUT LAYER              │
│  Summarizer │ CitationMgr          │
│  ReportBuilder │ PDF │ DOCX        │
└────────────────────────────────────┘
```

## Project Structure

```
research-report-agent/
├── app/
│   ├── config/       # Settings & environment loading
│   ├── models/       # Pydantic data models
│   ├── agent/        # Research controller / orchestrator
│   ├── tools/
│   │   ├── search/       # Tavily search provider
│   │   ├── extraction/   # trafilatura content extractor
│   │   └── evaluation/   # Source evaluator
│   ├── rag/          # Chunking, embeddings, ChromaDB, retrieval
│   ├── generation/   # Summarizer, citation manager, report builder
│   ├── export/       # PDF (ReportLab) and DOCX (python-docx)
│   └── ui/           # Streamlit UI
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/             # ChromaDB and intermediate data (gitignored)
├── outputs/          # Generated reports (gitignored)
├── .env.example
├── pyproject.toml
└── TASKS.md
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangChain |
| Search | Tavily API |
| Extraction | trafilatura + BeautifulSoup4 |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | ChromaDB |
| UI | Streamlit |
| DOCX | python-docx |
| PDF | ReportLab |
| Testing | pytest |

## License

MIT
