# Research & Report Agent

An autonomous AI-powered research and report-generation system built for academic demonstration (MCA project).

## What It Does

Given a natural-language research topic, the agent automatically:

1. **Plans** the research by decomposing the topic into focused sub-questions.
2. **Searches** the web using the Tavily API.
3. **Extracts** clean, readable content from discovered URLs.
4. **Evaluates** sources based on relevance, credibility, recency, and redundancy.
5. **Builds a RAG knowledge base** using OpenAI Embeddings and ChromaDB.
6. **Retrieves** relevant evidence for each finding.
7. **Generates grounded findings** using evidence-backed information to minimize hallucinations.
8. **Manages citations** with full provenance tracking.
9. **Produces a structured report** containing:

   * Title
   * Introduction
   * Key Findings
   * Analysis
   * Conclusion
   * References
10. **Exports** the report as **PDF** and **DOCX**.

---

## Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/utsav-26-06/ai-research-report-agent
cd research-report-agent

python -m venv .venv
```

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

### 2. Configure API Keys

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys.

You need:

* `OPENAI_API_KEY` — [OpenAI Platform](https://platform.openai.com/)
* `TAVILY_API_KEY` — [Tavily](https://tavily.com/)

### 3. Run the Streamlit UI

```bash
streamlit run app/ui/streamlit_app.py
```

### 4. Run Tests

```bash
pytest
```

---

## Architecture

```text
                    ┌─────────────────┐
                    │   Streamlit UI  │
                    └────────┬────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Research Controller         │
              │     (Orchestrator)          │
              └─────────────┬───────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │               TOOL LAYER               │
        │                                        │
        │  QueryPlanner     │     TavilySearch   │
        │  TrafilaturaExtractor │ SourceEvaluator│
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │                RAG LAYER                │
        │                                        │
        │  Chunker → Embeddings → ChromaDB       │
        │                     ↓                  │
        │                 Retrieval              │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │              OUTPUT LAYER               │
        │                                        │
        │  Summarizer    │    CitationManager    │
        │  ReportBuilder │    PDF / DOCX Export  │
        └────────────────────────────────────────┘
```

---

## Project Structure

```text
research-report-agent/
│
├── app/
│   ├── config/                    # Settings & environment loading
│   ├── models/                    # Pydantic data models
│   ├── agent/                     # Research controller / orchestrator
│   ├── tools/
│   │   ├── search/                # Tavily search provider
│   │   ├── extraction/            # Trafilatura content extractor
│   │   └── evaluation/            # Source evaluator
│   ├── rag/                       # Chunking, embeddings, ChromaDB, retrieval
│   ├── generation/                # Summarizer, citation manager, report builder
│   ├── export/                    # PDF (ReportLab) and DOCX (python-docx)
│   └── ui/                        # Streamlit UI
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── data/                          # ChromaDB and intermediate data (gitignored)
├── outputs/                       # Generated reports (gitignored)
├── .env.example
├── pyproject.toml
└── TASKS.md
```

---

## Technology Stack

| Layer               | Technology                      |
| ------------------- | ------------------------------- |
| **LLM**             | OpenAI GPT-4o-mini              |
| **Orchestration**   | LangChain                       |
| **Search**          | Tavily API                      |
| **Extraction**      | trafilatura + BeautifulSoup4    |
| **Embeddings**      | OpenAI `text-embedding-3-small` |
| **Vector Database** | ChromaDB                        |
| **UI**              | Streamlit                       |
| **DOCX Export**     | python-docx                     |
| **PDF Export**      | ReportLab                       |
| **Testing**         | pytest                          |

---

## License

This project is licensed under the **MIT License**.

---

# AI Research Report Agent

An autonomous AI-powered research and report-generation system for academic research and demonstration.
