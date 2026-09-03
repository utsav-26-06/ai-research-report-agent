from __future__ import annotations
import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import asyncio, logging
from pathlib import Path
import streamlit as st
st.set_page_config(
    page_title='AI Research & Report Agent',
    page_icon='🔬',
    layout='wide',
    initial_sidebar_state='expanded',
)
from app.ui.components import (
    inject_css, render_hero, render_section_header,
    render_progress_tracker, render_full_report, download_buttons,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _init_state():
    for k, v in {'result': None, 'stages_done': [], 'current_stage': '', 'error': None, 'running': False}.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _build_controller(settings, output_dir):
    from app.agent.query_planner import QueryPlanner
    from app.agent.research_controller import ResearchController
    from app.export.docx_exporter import DocxExporter
    from app.export.pdf_exporter import PDFExporter
    from app.generation.gemini_provider import GeminiLLMProvider
    from app.generation.rag_summarizer import RAGSummarizer
    from app.generation.report_builder import ReportBuilder
    from app.rag.chunker import DocumentChunker
    from app.rag.embedding_provider import GeminiEmbeddingProvider
    from app.rag.retriever import SemanticRetriever
    from app.rag.vector_store import ChromaVectorStore
    from app.tools.evaluation.llm_evaluator import LLMSourceEvaluator
    from app.tools.extraction.trafilatura_extractor import TrafilaturaExtractor
    from app.tools.search.tavily_provider import TavilySearchProvider
    llm = GeminiLLMProvider(api_key=settings.gemini_api_key, model_name=settings.gemini_model,
        max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature)
    emb = GeminiEmbeddingProvider(api_key=settings.gemini_api_key, model_name=settings.gemini_embedding_model)
    vs = ChromaVectorStore(embedding_provider=emb, persist_directory=str(settings.ensure_chroma_dir()),
        collection_name=settings.chroma_collection_name)
    ret = SemanticRetriever(vector_store=vs, n_results=5)
    return ResearchController(
        planner=QueryPlanner(llm=llm),
        searcher=TavilySearchProvider(api_key=settings.tavily_api_key,
            max_results=settings.tavily_max_results, search_depth=settings.tavily_search_depth),
        extractor=TrafilaturaExtractor(),
        evaluator=LLMSourceEvaluator(llm=llm),
        chunker=DocumentChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
        embedding_provider=emb, vector_store=vs, retriever=ret,
        summarizer=RAGSummarizer(llm=llm, retriever=ret),
        report_builder=ReportBuilder(llm=llm),
        docx_exporter=DocxExporter(output_dir=output_dir),
        pdf_exporter=PDFExporter(output_dir=output_dir),
    )

async def _run_research(topic, depth, output_dir):
    from app.config.settings import get_settings
    from app.models.research import ResearchRequest
    settings = get_settings()
    ctrl = _build_controller(settings, output_dir)
    req = ResearchRequest(topic=topic, depth=depth)
    def on_progress(stage, message):
        st.session_state.current_stage = stage
        if stage not in st.session_state.stages_done:
            st.session_state.stages_done.append(stage)
    return await ctrl.run(req, progress=on_progress)

def main():
    _init_state()
    inject_css()
    render_hero()
    with st.sidebar:
        st.markdown('### ⚙️ Settings')
        st.markdown('---')
        depth = st.selectbox('Research Depth', ['quick', 'standard', 'deep'], index=1,
            help='quick = 3 sub-questions | standard = 5 | deep = 8+')
        st.markdown('')
        st.checkbox('📄 Export DOCX', value=True)
        st.checkbox('📕 Export PDF', value=True)
        st.markdown('---')
        st.caption('API keys loaded from .env')
        with st.expander('💡 Example topics'):
            st.markdown('- Impact of AI on software development\n- Quantum computing and cybersecurity\n- Rise of autonomous AI agents in 2025\n- How LLMs are changing healthcare')
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    render_section_header('Research Topic', '🔍')
    topic = st.text_area('topic',
        placeholder='e.g. The impact of large language models on software development',
        height=110, label_visibility='collapsed', key='topic_input')
    col_run, col_clear = st.columns([4, 1])
    run_clicked = col_run.button('🚀  Start Research', type='primary',
        disabled=st.session_state.running, use_container_width=True)
    if col_clear.button('🗑️  Clear', use_container_width=True):
        for k in ['result', 'stages_done', 'current_stage', 'error']:
            st.session_state[k] = [] if k == 'stages_done' else None
        st.rerun()
    if run_clicked and topic.strip():
        st.session_state.running = True
        st.session_state.result = None
        st.session_state.stages_done = []
        st.session_state.error = None
        out = Path('./outputs')
        out.mkdir(parents=True, exist_ok=True)
        try:
            with st.spinner('Research pipeline running…'):
                result = asyncio.run(_run_research(topic.strip(), depth, out))
            st.session_state.result = result
            st.session_state.running = False
            st.success('✅ Research complete!')
        except Exception as e:
            st.session_state.error = str(e)
            st.session_state.running = False
            logger.exception('Pipeline error')
    elif run_clicked:
        st.warning('Please enter a research topic.')
    if st.session_state.running or st.session_state.stages_done:
        render_progress_tracker(st.session_state.stages_done, st.session_state.current_stage)
    if st.session_state.error:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        with st.expander('❌ Error details', expanded=True):
            st.error(st.session_state.error)
            st.info('Check that your .env contains valid GEMINI_API_KEY and TAVILY_API_KEY.')
    if st.session_state.result is not None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        download_buttons(st.session_state.result.docx_path, st.session_state.result.pdf_path)
        render_full_report(st.session_state.result.report)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
