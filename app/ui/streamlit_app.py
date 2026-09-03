from __future__ import annotations
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so 'import app' works from any cwd.
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Streamlit web UI for the AI Research & Report Agent (TASK-020).

Run with:  streamlit run app/ui/streamlit_app.py
"""


import asyncio
import logging
from pathlib import Path

import streamlit as st

# --- Page config must be first ---
st.set_page_config(
    page_title="AI Research & Report Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from app.ui.components import (
    render_header,
    render_full_report,
    render_progress_tracker,
    download_buttons,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Session state helpers
# ------------------------------------------------------------------

def _init_state():
    defaults = {
        "result": None,
        "stages_done": [],
        "current_stage": "",
        "error": None,
        "running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _build_controller(settings, output_dir: Path):
    """Wire up the full ResearchController from settings."""
    from app.agent.query_planner import QueryPlanner
    from app.agent.research_controller import ResearchController
    from app.export.docx_exporter import DocxExporter
    from app.export.pdf_exporter import PDFExporter
    from app.generation.rag_summarizer import RAGSummarizer
    from app.generation.report_builder import ReportBuilder
    from app.rag.chunker import DocumentChunker
    from app.rag.embedding_provider import GeminiEmbeddingProvider
    from app.rag.retriever import SemanticRetriever
    from app.rag.vector_store import ChromaVectorStore
    from app.tools.evaluation.llm_evaluator import LLMSourceEvaluator
    from app.tools.extraction.trafilatura_extractor import TrafilaturaExtractor
    from app.tools.search.tavily_provider import TavilySearchProvider
    from app.generation.base import LLMProvider

    # LLM provider — use a simple Gemini wrapper
    try:
        from app.generation.gemini_provider import GeminiLLMProvider
        llm: LLMProvider = GeminiLLMProvider(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    except ImportError:
        raise RuntimeError(
            "GeminiLLMProvider not found. Ensure TASK-012 is complete or "
            "create app/generation/gemini_provider.py."
        )

    embedding_provider = GeminiEmbeddingProvider(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_embedding_model,
    )
    vector_store = ChromaVectorStore(
        embedding_provider=embedding_provider,
        persist_directory=str(settings.ensure_chroma_dir()),
        collection_name=settings.chroma_collection_name,
    )
    retriever = SemanticRetriever(vector_store=vector_store, n_results=5)

    return ResearchController(
        planner=QueryPlanner(llm=llm),
        searcher=TavilySearchProvider(
            api_key=settings.tavily_api_key,
            max_results=settings.tavily_max_results,
            search_depth=settings.tavily_search_depth,
        ),
        extractor=TrafilaturaExtractor(),
        evaluator=LLMSourceEvaluator(llm=llm),
        chunker=DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        retriever=retriever,
        summarizer=RAGSummarizer(llm=llm, retriever=retriever),
        report_builder=ReportBuilder(llm=llm),
        docx_exporter=DocxExporter(output_dir=output_dir),
        pdf_exporter=PDFExporter(output_dir=output_dir),
    )


async def _run_research(topic: str, depth: str, output_dir: Path):
    """Async entry point: wires up the controller and runs the pipeline."""
    from app.config.settings import get_settings
    from app.models.research import ResearchRequest

    settings = get_settings()
    controller = _build_controller(settings, output_dir)
    request = ResearchRequest(topic=topic, depth=depth)

    def on_progress(stage: str, message: str):
        st.session_state.current_stage = stage
        if stage not in st.session_state.stages_done:
            st.session_state.stages_done.append(stage)
        logger.info(f"[{stage}] {message}")

    result = await controller.run(request, progress=on_progress)
    return result


# ------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------

def main():
    _init_state()
    render_header()

    # --- Sidebar: settings ---
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        depth = st.selectbox(
            "Research Depth",
            options=["quick", "standard", "deep"],
            index=1,
            help="quick = 3 sub-questions, standard = 5, deep = 8+"
        )
        export_docx = st.checkbox("Export DOCX", value=True)
        export_pdf = st.checkbox("Export PDF", value=True)
        st.markdown("---")
        st.caption("API keys are loaded from `.env`")

    # --- Main pane ---
    with st.container():
        topic = st.text_area(
            "Research Topic",
            placeholder="e.g. 'The impact of large language models on software development'",
            height=100,
            label_visibility="collapsed",
            key="topic_input",
        )

        col_run, col_clear = st.columns([3, 1])
        run_clicked = col_run.button(
            "🚀 Start Research",
            type="primary",
            disabled=st.session_state.running,
            use_container_width=True,
        )
        if col_clear.button("🗑️ Clear", use_container_width=True):
            for k in ["result", "stages_done", "current_stage", "error"]:
                st.session_state[k] = [] if k == "stages_done" else None
            st.rerun()

    # --- Run ---
    if run_clicked and topic.strip():
        st.session_state.running = True
        st.session_state.result = None
        st.session_state.stages_done = []
        st.session_state.error = None

        output_dir = Path("./outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_container = st.empty()
        status_text = st.empty()

        try:
            with st.spinner("Research in progress…"):
                result = asyncio.run(
                    _run_research(topic.strip(), depth, output_dir)
                )
            st.session_state.result = result
            st.session_state.running = False
            st.success("✅ Research complete!")
        except Exception as e:
            st.session_state.error = str(e)
            st.session_state.running = False
            logger.exception("Pipeline error")

    elif run_clicked and not topic.strip():
        st.warning("Please enter a research topic.")

    # --- Error display ---
    if st.session_state.error:
        with st.expander("❌ Error Details", expanded=True):
            st.error(st.session_state.error)
            st.info(
                "Check that your `.env` file contains valid `GEMINI_API_KEY` "
                "and `TAVILY_API_KEY`."
            )

    # --- Results ---
    if st.session_state.result is not None:
        result = st.session_state.result

        st.markdown("---")
        download_buttons(result.docx_path, result.pdf_path)
        st.markdown("")
        render_full_report(result.report)


if __name__ == "__main__":
    main()


