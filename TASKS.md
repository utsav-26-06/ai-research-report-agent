# Research & Report Agent â€” Task Roadmap

## Legend
```
[ ] Not Started
[~] In Progress
[x] Completed
[!] Blocked
```

## Dependency Graph

```
TASK-001 (Foundation)
    â””â”€â”€ TASK-002 (Config)
            â””â”€â”€ TASK-003 (Models)
                    â””â”€â”€ TASK-004 (Interfaces)
                            â”œâ”€â”€ TASK-005 (Query Planner)
                            â”œâ”€â”€ TASK-006 (Web Search)      â”€â”€> TASK-007 (Extraction)
                            â”‚                                        â””â”€â”€> TASK-008 (Evaluation)
                            â”‚                                                  â””â”€â”€> TASK-009 (Chunking)
                            â”‚                                                             â””â”€â”€> TASK-010 (Embeddings)
                            â”‚                                                                        â””â”€â”€> TASK-011 (ChromaDB)
                            â”‚                                                                                   â””â”€â”€> TASK-012 (Retrieval)
                            â””â”€â”€ TASK-013 (RAG Summarization)  [needs 005 + 012]
                                    â””â”€â”€ TASK-014 (Citation Mgmt)
                                                â””â”€â”€ TASK-015 (Report Model & Builder)
                                                            â”œâ”€â”€ TASK-016 (DOCX Export)
                                                            â””â”€â”€ TASK-017 (PDF Export)
                                                                        â””â”€â”€ TASK-018 (Research Controller)
                                                                                    â””â”€â”€ TASK-019 (E2E Pipeline Test)
                                                                                                â””â”€â”€ TASK-020 (Streamlit UI)
                                                                                                            â””â”€â”€ TASK-021 (Error Handling & Logging)
                                                                                                                        â””â”€â”€ TASK-022 (Full Test Suite)
                                                                                                                                    â””â”€â”€ TASK-023 (Real API Smoke Test)
                                                                                                                                                â””â”€â”€ TASK-024 (Docs & Audit)
```

---

## TASK-001 â€” Project Foundation

**Status:** [x]
**Purpose:** Establish the complete project skeleton â€” directories, packaging, git, configuration files.
**Dependencies:** None

### Implementation Requirements
- Create all package directories with `__init__.py`
- Create `pyproject.toml` with all dependencies declared
- Create `.gitignore`
- Create `.env.example` with all required env vars documented
- Create `README.md`
- Create `app/main.py` entry point stub
- Initialize git repository
- Create initial commit

### Files / Components
- `pyproject.toml`
- `.gitignore`
- `.env.example`
- `README.md`
- `app/__init__.py` and all sub-package `__init__.py`
- `app/main.py`
- `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/chroma/.gitkeep`, `outputs/.gitkeep`

### Testing Requirements
- Verify directory structure exists
- Verify `pyproject.toml` is valid TOML
- Verify `app` is importable

### Smoke Test
```bash
python -c "import app; print('app importable')"
```

### Acceptance Criteria
- [x] All directories created
- [x] All `__init__.py` files present
- [x] `pyproject.toml` parses without error
- [x] `.gitignore` present and covers `.env`, `data/chroma/`
- [x] `.env.example` documents all env vars
- [x] `README.md` present
- [x] `git init` complete, initial commit made

---

## TASK-002 â€” Configuration Module

**Status:** [x]
**Purpose:** Central settings object â€” loads `.env`, validates, and exposes typed config to the rest of the app.
**Dependencies:** TASK-001

### Implementation Requirements
- Use `pydantic-settings` or manual Pydantic `BaseSettings` with `python-dotenv`
- Load from `.env` automatically
- Validate all required fields (API keys, paths, numeric params)
- Raise descriptive errors for missing required keys
- Provide a singleton `get_settings()` function
- Support environment-variable overrides

### Files / Components
- `app/config/settings.py` â€” `Settings` model
- `app/config/__init__.py` â€” re-exports `get_settings`
- `tests/unit/test_config.py`

### Testing Requirements
- Test that all fields load correctly from env vars
- Test that missing required fields raise `ValidationError`
- Test that optional fields have correct defaults
- Test singleton behavior

### Smoke Test
```bash
python -c "from app.config import get_settings; s = get_settings(); print(s.openai_model)"
```

### Acceptance Criteria
- [ ] `Settings` model covers all env vars in `.env.example`
- [ ] Missing `OPENAI_API_KEY` raises clear error
- [ ] All defaults match `.env.example` defaults
- [ ] Unit tests pass

---

## TASK-003 â€” Core Data Models

**Status:** [x]
**Purpose:** Define typed Pydantic models for all entities flowing through the pipeline, ensuring provenance is never lost.
**Dependencies:** TASK-002

### Implementation Requirements
Define at minimum:
- `ResearchRequest` â€” user input
- `ResearchPlan` â€” planner output with sub-questions
- `SubQuestion` â€” id, text, search_queries
- `SearchResult` â€” url, title, snippet, query, sub_question_id
- `SourceDocument` â€” full content, metadata, provenance
- `SourceEvaluation` â€” scores, decision, reasons
- `ContentChunk` â€” text, metadata, provenance
- `EmbeddedChunk` â€” chunk + embedding vector
- `Citation` â€” source reference, in-text marker
- `Finding` â€” text, supporting chunks, citations
- `ReportSection` â€” title, content, citations
- `ResearchReport` â€” full structured report

All models must:
- Use Pydantic v2
- Include field validation where appropriate
- Preserve `source_id`, `url`, `sub_question_id` traceability

### Files / Components
- `app/models/research.py` â€” `ResearchRequest`, `ResearchPlan`, `SubQuestion`
- `app/models/sources.py` â€” `SearchResult`, `SourceDocument`, `SourceEvaluation`
- `app/models/rag.py` â€” `ContentChunk`, `EmbeddedChunk`
- `app/models/report.py` â€” `Citation`, `Finding`, `ReportSection`, `ResearchReport`
- `app/models/__init__.py` â€” re-exports all models
- `tests/unit/test_models.py`

### Testing Requirements
- Test instantiation of each model with valid data
- Test field validation (e.g., invalid URL, negative scores)
- Test that provenance fields survive model serialization

### Smoke Test
```bash
python -c "from app.models import ResearchRequest; r = ResearchRequest(topic='AI'); print(r)"
```

### Acceptance Criteria
- [ ] All 12 models defined and importable
- [ ] Provenance chain: `Finding â†’ Citation â†’ SourceDocument â†’ url` traceable
- [ ] Unit tests pass

---

## TASK-004 â€” Provider Interfaces (Abstract Base Classes)

**Status:** [x]
**Purpose:** Define abstract interfaces for all external providers so the business logic is decoupled from specific APIs.
**Dependencies:** TASK-003

### Implementation Requirements
Define abstract base classes for:
- `SearchProvider` â€” `search(query: str) -> list[SearchResult]`
- `ContentExtractor` â€” `extract(url: str) -> SourceDocument | None`
- `SourceEvaluator` â€” `evaluate(doc: SourceDocument, query: str) -> SourceEvaluation`
- `EmbeddingProvider` â€” `embed(texts: list[str]) -> list[list[float]]`
- `VectorStore` â€” `add(chunks)`, `query(text, n)`, `clear()`
- `LLMProvider` â€” `complete(prompt: str) -> str`, `structured_complete(prompt, schema) -> dict`

### Files / Components
- `app/tools/search/base.py`
- `app/tools/extraction/base.py`
- `app/tools/evaluation/base.py`
- `app/rag/base.py`
- `app/generation/base.py`
- `tests/unit/test_interfaces.py`

### Testing Requirements
- Confirm ABCs cannot be instantiated directly
- Confirm subclassing and overriding works as expected

### Smoke Test
```bash
python -c "from app.tools.search.base import SearchProvider; print('interfaces ok')"
```

### Acceptance Criteria
- [ ] All 6 interfaces defined
- [ ] Concrete class missing `abstractmethod` raises `TypeError`
- [ ] Unit tests pass

---

## TASK-005 â€” Query Planning Module

**Status:** [x]
**Purpose:** Break a broad research topic into 4â€“6 focused, non-overlapping sub-questions using an LLM.
**Dependencies:** TASK-004

### Implementation Requirements
- Use OpenAI GPT via LangChain with structured output
- Generate 4â€“6 sub-questions per topic
- Deduplicate overlapping questions
- Return a validated `ResearchPlan` model
- Handle LLM errors with retry logic (tenacity)

### Files / Components
- `app/agent/query_planner.py` â€” `QueryPlanner` class
- `app/agent/__init__.py`
- `tests/unit/test_query_planner.py` (mocked LLM)
- `tests/integration/test_planner_integration.py` (real API, optional)

### Testing Requirements
- Mock LLM response â€” verify `ResearchPlan` is correctly parsed
- Test deduplication of near-identical questions
- Test error path when LLM returns malformed output

### Smoke Test
```bash
python -c "
from unittest.mock import MagicMock
from app.agent.query_planner import QueryPlanner
# verified via unit test
print('planner importable')
"
```

### Acceptance Criteria
- [ ] Returns `ResearchPlan` with 4â€“6 `SubQuestion` objects
- [ ] Duplicate questions are removed
- [ ] LLM errors are caught and re-raised as domain exceptions
- [ ] Unit tests pass

---

## TASK-006 â€” Web Search Module (Tavily)

**Status:** [x]
**Purpose:** For each sub-question, search the web and collect candidate `SearchResult` objects.
**Dependencies:** TASK-004, TASK-005

### Implementation Requirements
- Implement `TavilySearchProvider(SearchProvider)`
- Accept per-query parameters (max_results, search_depth)
- Normalize and deduplicate URLs across sub-questions
- Preserve `sub_question_id` and `search_query` provenance
- Handle Tavily API errors, timeouts, rate limits
- Do NOT let Tavily response structures leak into the rest of the app

### Files / Components
- `app/tools/search/tavily_provider.py`
- `tests/unit/test_tavily_provider.py` (mocked HTTP)

### Testing Requirements
- Mock Tavily HTTP response â€” verify correct `SearchResult` mapping
- Test URL normalization (trailing slashes, fragments)
- Test deduplication
- Test error propagation

### Smoke Test
```bash
python -c "from app.tools.search.tavily_provider import TavilySearchProvider; print('search ok')"
```

### Acceptance Criteria
- [ ] `TavilySearchProvider` implements `SearchProvider`
- [ ] Returns `list[SearchResult]` with provenance fields
- [ ] Duplicate URLs removed
- [ ] Unit tests pass

---

## TASK-007 â€” Content Extraction Module

**Status:** [x]
**Purpose:** Fetch web pages and extract clean, readable text using trafilatura.
**Dependencies:** TASK-004, TASK-006

### Implementation Requirements
- Implement `TrafilaturaExtractor(ContentExtractor)`
- Use `trafilatura.fetch_url()` + `trafilatura.extract()`
- Fall back to BeautifulSoup if trafilatura returns empty
- Extract: title, text, publication_date, url
- Reject pages with < 100 characters of content
- One failed URL must NOT crash the extraction loop
- Return `None` for failed extractions

### Files / Components
- `app/tools/extraction/trafilatura_extractor.py`
- `tests/unit/test_extractor.py` (mocked HTTP)
- `tests/fixtures/sample_page.html`

### Testing Requirements
- Test successful extraction returns `SourceDocument`
- Test short content rejection
- Test failed fetch returns `None`
- Test BeautifulSoup fallback triggers correctly

### Smoke Test
```bash
python -c "from app.tools.extraction.trafilatura_extractor import TrafilaturaExtractor; print('extractor ok')"
```

### Acceptance Criteria
- [ ] `TrafilaturaExtractor` implements `ContentExtractor`
- [ ] Returns `SourceDocument` with populated fields
- [ ] Failed URLs return `None`, not exceptions
- [ ] Unit tests pass

---

## TASK-008 â€” Source Evaluation Module

**Status:** [x]
**Purpose:** Score and filter extracted sources on relevance, credibility, recency, and redundancy.
**Dependencies:** TASK-004, TASK-007

### Implementation Requirements
- Implement `LLMSourceEvaluator(SourceEvaluator)`
- Score each dimension 0.0â€“1.0
- Produce a structured `SourceEvaluation` with reasons
- Apply configurable thresholds (from Settings)
- Detect and flag redundant sources (high content overlap)
- Use cosine similarity or simple heuristic for redundancy

### Files / Components
- `app/tools/evaluation/llm_evaluator.py`
- `tests/unit/test_evaluator.py` (mocked LLM)

### Testing Requirements
- Test high-relevance source passes threshold
- Test low-relevance source is rejected
- Test redundant source is flagged
- Test evaluation scores are within [0, 1]

### Smoke Test
```bash
python -c "from app.tools.evaluation.llm_evaluator import LLMSourceEvaluator; print('evaluator ok')"
```

### Acceptance Criteria
- [ ] Returns `SourceEvaluation` with scores + reasons
- [ ] Filtering removes sources below thresholds
- [ ] Unit tests pass

---

## TASK-009 â€” Document Chunking

**Status:** [x]
**Purpose:** Split accepted source documents into overlapping text chunks suitable for embedding.
**Dependencies:** TASK-003, TASK-008

### Implementation Requirements
- Use LangChain `RecursiveCharacterTextSplitter`
- Configurable `chunk_size` and `chunk_overlap` from Settings
- Each `ContentChunk` must retain: `chunk_id`, `source_id`, `url`, `title`, `sub_question_id`, `text`
- Reject empty chunks

### Files / Components
- `app/rag/chunker.py` â€” `DocumentChunker`
- `tests/unit/test_chunker.py`

### Testing Requirements
- Test chunk count for known-length text
- Test overlap is correct
- Test provenance fields are retained

### Smoke Test
```bash
python -c "from app.rag.chunker import DocumentChunker; print('chunker ok')"
```

### Acceptance Criteria
- [ ] `DocumentChunker` produces `list[ContentChunk]`
- [ ] Chunk provenance matches source
- [ ] Unit tests pass

---

## TASK-010 â€” Embeddings (OpenAI)

**Status:** [x]
**Purpose:** Generate vector embeddings for all content chunks using OpenAI Embeddings API.
**Dependencies:** TASK-004, TASK-009

### Implementation Requirements
- Implement `OpenAIEmbeddingProvider(EmbeddingProvider)`
- Use `text-embedding-3-small` model (configurable)
- Batch requests to avoid rate limits
- Handle API errors with retry
- Return `list[EmbeddedChunk]`

### Files / Components
- `app/rag/embedding_provider.py`
- `tests/unit/test_embedding_provider.py` (mocked API)

### Testing Requirements
- Test embedding dimension matches expected (1536 for text-embedding-3-small)
- Test batch processing
- Test retry on API failure

### Smoke Test
```bash
python -c "from app.rag.embedding_provider import OpenAIEmbeddingProvider; print('embeddings ok')"
```

### Acceptance Criteria
- [ ] Returns vectors of correct dimension
- [ ] Batching works for > 100 chunks
- [ ] Unit tests pass

---

## TASK-011 â€” ChromaDB Vector Store

**Status:** [ ]
**Purpose:** Persist embedded chunks in ChromaDB for semantic retrieval.
**Dependencies:** TASK-004, TASK-010

### Implementation Requirements
- Implement `ChromaVectorStore(VectorStore)`
- Use persistent ChromaDB (not in-memory) for production
- Support: `add(chunks)`, `query(text, n_results)`, `clear()`
- Store chunk metadata alongside vectors
- Handle collection creation if not exists

### Files / Components
- `app/rag/vector_store.py`
- `tests/unit/test_vector_store.py` (in-memory ChromaDB for tests)

### Testing Requirements
- Test add + query returns expected chunks
- Test metadata is preserved
- Test `clear()` empties collection
- Test query with no results

### Smoke Test
```bash
python -c "from app.rag.vector_store import ChromaVectorStore; print('vectorstore ok')"
```

### Acceptance Criteria
- [ ] `ChromaVectorStore` implements `VectorStore`
- [ ] Chunks are retrievable by semantic similarity
- [ ] Unit tests pass

---

## TASK-012 â€” Semantic Retrieval

**Status:** [ ]
**Purpose:** Given a query string, retrieve the top-N most relevant chunks from ChromaDB.
**Dependencies:** TASK-010, TASK-011

### Implementation Requirements
- Implement `SemanticRetriever`
- Accept query text + optional filter (by sub_question_id)
- Return ranked `list[ContentChunk]` with similarity scores
- Configure `n_results` from Settings

### Files / Components
- `app/rag/retriever.py`
- `tests/unit/test_retriever.py`

### Testing Requirements
- Test retrieval returns correct chunks
- Test filtering by metadata field
- Test empty-store returns empty list

### Smoke Test
```bash
python -c "from app.rag.retriever import SemanticRetriever; print('retriever ok')"
```

### Acceptance Criteria
- [ ] Returns `list[ContentChunk]` sorted by relevance
- [ ] Unit tests pass

---

## TASK-013 â€” RAG Summarization Module

**Status:** [ ]
**Purpose:** For each sub-question, retrieve relevant chunks and generate a grounded finding using GPT.
**Dependencies:** TASK-005, TASK-012

### Implementation Requirements
- Implement `RAGSummarizer`
- For each `SubQuestion`: retrieve evidence chunks â†’ build prompt â†’ call GPT â†’ parse `Finding`
- Prompt must include evidence text and instruct GPT not to invent facts
- If evidence is insufficient, indicate clearly (do not fabricate)
- Preserve chunk/source provenance in each `Finding`

### Files / Components
- `app/generation/rag_summarizer.py`
- `tests/unit/test_rag_summarizer.py` (mocked LLM + retriever)

### Testing Requirements
- Test finding generated with evidence references
- Test insufficient-evidence path
- Test that fabricated content is not present when evidence is empty

### Smoke Test
```bash
python -c "from app.generation.rag_summarizer import RAGSummarizer; print('summarizer ok')"
```

### Acceptance Criteria
- [ ] Each `Finding` includes `supporting_chunks` references
- [ ] Insufficient evidence case is handled
- [ ] Unit tests pass

---

## TASK-014 â€” Citation Management

**Status:** [ ]
**Purpose:** Build a deduplicated reference list and attach in-text citation markers to findings.
**Dependencies:** TASK-013

### Implementation Requirements
- Implement `CitationManager`
- Generate sequential citation numbers [1], [2], ...
- Deduplicate: same URL â†’ same citation number
- Attach `Citation` objects to each `Finding`
- Generate formatted reference list (APA-ish style)

### Files / Components
- `app/generation/citation_manager.py`
- `tests/unit/test_citation_manager.py`

### Testing Requirements
- Test same URL â†’ same citation number
- Test different URLs â†’ different numbers
- Test reference list format
- Test `Finding` citations are populated

### Smoke Test
```bash
python -c "from app.generation.citation_manager import CitationManager; print('citations ok')"
```

### Acceptance Criteria
- [ ] No duplicate citations for same URL
- [ ] In-text markers appear in finding text
- [ ] Reference list is complete
- [ ] Unit tests pass

---

## TASK-015 â€” Report Generation

**Status:** [ ]
**Purpose:** Assemble findings into a structured `ResearchReport` with all sections.
**Dependencies:** TASK-014

### Implementation Requirements
- Implement `ReportBuilder`
- Sections: Title, Introduction, Key Findings, Analysis, Conclusion, References
- Use LLM to write Introduction, Analysis, Conclusion from findings (grounded, not invented)
- Key Findings section = structured list of findings with citations
- References section = formatted citation list

### Files / Components
- `app/generation/report_builder.py`
- `tests/unit/test_report_builder.py` (mocked LLM)

### Testing Requirements
- Test all sections are present in output
- Test citations appear in references
- Test no section is empty

### Smoke Test
```bash
python -c "from app.generation.report_builder import ReportBuilder; print('report builder ok')"
```

### Acceptance Criteria
- [ ] `ResearchReport` contains all 6 sections
- [ ] Citations are present throughout
- [ ] Unit tests pass

---

## TASK-016 â€” DOCX Export

**Status:** [ ]
**Purpose:** Export the `ResearchReport` as a professionally formatted `.docx` file.
**Dependencies:** TASK-015

### Implementation Requirements
- Implement `DocxExporter`
- Use `python-docx`
- Apply heading styles, body styles, bold for citations
- Save to `outputs/` directory
- Return file path

### Files / Components
- `app/export/docx_exporter.py`
- `tests/unit/test_docx_exporter.py`

### Testing Requirements
- Test file is created and non-empty
- Test all sections appear in document
- Test heading hierarchy is correct

### Smoke Test
```bash
python -c "from app.export.docx_exporter import DocxExporter; print('docx exporter ok')"
```

### Acceptance Criteria
- [ ] Valid `.docx` produced
- [ ] All sections and citations present
- [ ] Unit tests pass

---

## TASK-017 â€” PDF Export

**Status:** [ ]
**Purpose:** Export the `ResearchReport` as a professionally formatted `.pdf` file.
**Dependencies:** TASK-015

### Implementation Requirements
- Implement `PDFExporter`
- Use `ReportLab` (Platypus / RLPDF)
- Professional styling: fonts, headings, body, footers with page numbers
- Save to `outputs/` directory
- Return file path

### Files / Components
- `app/export/pdf_exporter.py`
- `tests/unit/test_pdf_exporter.py`

### Testing Requirements
- Test file is created and non-empty
- Test PDF is valid (parseable)
- Test all sections appear

### Smoke Test
```bash
python -c "from app.export.pdf_exporter import PDFExporter; print('pdf exporter ok')"
```

### Acceptance Criteria
- [ ] Valid `.pdf` produced
- [ ] All sections and citations present
- [ ] Unit tests pass

---

## TASK-018 â€” Research Controller (Orchestrator)

**Status:** [ ]
**Purpose:** Coordinate the entire pipeline from topic input to final report output.
**Dependencies:** TASK-005 through TASK-017

### Implementation Requirements
- Implement `ResearchController`
- Accept `ResearchRequest`, yield/report progress at each stage
- Orchestrate: Plan â†’ Search â†’ Extract â†’ Evaluate â†’ Filter â†’ Chunk â†’ Embed â†’ Store â†’ Retrieve â†’ Summarize â†’ Cite â†’ Build Report â†’ Export
- Log each stage completion
- Handle partial failures gracefully (skip failed sources, continue)
- Return `ResearchReport` + file paths

### Files / Components
- `app/agent/research_controller.py`
- `tests/integration/test_research_controller.py` (fully mocked)

### Testing Requirements
- Test complete pipeline with all mocked dependencies
- Test failed source is skipped, pipeline continues
- Test progress callbacks are called at each stage

### Smoke Test
```bash
python -c "from app.agent.research_controller import ResearchController; print('controller ok')"
```

### Acceptance Criteria
- [ ] Full pipeline orchestrated correctly
- [ ] Progress reporting at each stage
- [ ] Failed sources do not crash pipeline
- [ ] Integration tests pass

---

## TASK-019 â€” End-to-End Pipeline Test

**Status:** [ ]
**Purpose:** Verify the complete pipeline works with fully mocked external dependencies.
**Dependencies:** TASK-018

### Implementation Requirements
- Complete E2E test using mocked LLM, search, and extraction
- Verify output: `ResearchReport` with all sections, PDF file, DOCX file
- Verify citation chain: Finding â†’ Citation â†’ Source â†’ URL

### Files / Components
- `tests/integration/test_e2e_pipeline.py`
- `tests/fixtures/mock_responses.py`

### Testing Requirements
- Run complete pipeline from topic input to file output
- Verify citation provenance chain

### Acceptance Criteria
- [ ] E2E test passes with mocked dependencies
- [ ] All output files created
- [ ] Citation chain complete

---

## TASK-020 â€” Streamlit UI

**Status:** [ ]
**Purpose:** Professional web interface for the Research & Report Agent.
**Dependencies:** TASK-018, TASK-019

### Implementation Requirements
- Build `app/ui/streamlit_app.py`
- Input: research topic text area
- Show progress bar + stage status during research
- Display full report (sections, citations)
- Download buttons: PDF, DOCX
- Professional, clean design

### Files / Components
- `app/ui/streamlit_app.py`
- `app/ui/components.py` (reusable UI helpers)

### Testing Requirements
- Manual UI verification
- Verify downloads work

### Acceptance Criteria
- [ ] UI launches without errors
- [ ] Research completes and report is displayed
- [ ] PDF and DOCX download buttons work

---

## TASK-021 â€” Error Handling & Logging Polish

**Status:** [ ]
**Purpose:** Add structured logging and robust error handling throughout.
**Dependencies:** TASK-018

### Implementation Requirements
- Configure Python `logging` with structured output
- Add log statements at all key pipeline stages
- Define custom exception classes: `SearchError`, `ExtractionError`, `EvaluationError`, `EmbeddingError`, `RAGError`, `ReportError`
- Ensure no secrets are logged
- Handle all API timeouts, rate limits, auth errors

### Files / Components
- `app/config/logging_config.py`
- `app/exceptions.py`

### Acceptance Criteria
- [ ] Log messages appear at all pipeline stages
- [ ] No API keys in logs
- [ ] Custom exceptions used throughout

---

## TASK-022 â€” Complete Test Suite

**Status:** [ ]
**Purpose:** Ensure all unit and integration tests are green.
**Dependencies:** TASK-021

### Implementation Requirements
- All unit tests pass: `pytest tests/unit/`
- All integration tests pass: `pytest tests/integration/`
- Coverage â‰¥ 80%

### Acceptance Criteria
- [ ] `pytest tests/unit/` â€” all pass
- [ ] `pytest tests/integration/` â€” all pass
- [ ] Coverage report generated

---

## TASK-023 â€” Real API Smoke Test

**Status:** [ ]
**Purpose:** Run the full pipeline with real API keys using the canonical test topic.
**Dependencies:** TASK-022

### Topic
```
Research the impact of artificial intelligence on software development.
```

### Acceptance Criteria
- [ ] Query planning generates 4â€“6 sub-questions
- [ ] Web search finds results for each sub-question
- [ ] Content extraction succeeds for â‰¥ 3 sources
- [ ] Source evaluation filters and scores sources
- [ ] ChromaDB populated with chunks
- [ ] Retrieval returns relevant evidence
- [ ] Findings generated with citations
- [ ] Report has all 6 sections
- [ ] PDF exported successfully
- [ ] DOCX exported successfully
- [ ] No fabricated citations

---

## TASK-024 â€” Documentation & Final Audit

**Status:** [ ]
**Purpose:** Finalize README, add docstrings, audit code quality.
**Dependencies:** TASK-023

### Implementation Requirements
- Update README with final setup and usage instructions
- Add docstrings to all public classes and methods
- Run `ruff` and `black`; fix all warnings
- Verify `.env` is not committed
- Create `docs/architecture.md`
- Create `docs/api_keys.md`

### Acceptance Criteria
- [ ] README complete and accurate
- [ ] All public APIs documented
- [ ] `ruff` passes with no warnings
- [ ] No secrets in git history
- [ ] `docs/` folder complete








