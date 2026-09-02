"""
Reusable UI components for the Streamlit app (TASK-020).
"""

import streamlit as st
from app.models.report import ResearchReport, Finding, Citation


def render_header():
    """Renders the app header with logo and tagline."""
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0 1rem 0;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.2rem;">
                🔬 AI Research & Report Agent
            </h1>
            <p style="color: #6c757d; font-size: 1.05rem; margin: 0;">
                Autonomous web research · RAG synthesis · Professional report export
            </p>
        </div>
        <hr style="margin-bottom: 1.5rem; border-color: #dee2e6;">
    """, unsafe_allow_html=True)


def render_citation(citation: Citation, idx: int):
    """Renders a single citation entry in the reference list."""
    with st.container():
        st.markdown(
            f"**{citation.marker}** {citation.title or 'Untitled'} — "
            f"[{citation.url}]({citation.url})",
            unsafe_allow_html=False,
        )
        if citation.excerpt:
            st.caption(f"> {citation.excerpt[:120]}...")


def render_finding(finding: Finding):
    """Renders a single research finding with its claim and citations."""
    if finding.uncertain:
        st.warning(f"⚠️ **Uncertain:** {finding.claim}")
    else:
        confidence_pct = int(finding.confidence * 100)
        st.markdown(f"• {finding.claim}")
        if finding.citations:
            cite_links = " ".join(
                f"[{c.marker}]({c.url})" for c in finding.citations
            )
            st.caption(f"Sources: {cite_links}")


def render_section(section, expanded: bool = True):
    """Renders a report section inside an expander."""
    with st.expander(f"📄 {section.heading}", expanded=expanded):
        if section.content:
            st.markdown(section.content)
        if section.findings:
            st.markdown("---")
            for finding in section.findings:
                render_finding(finding)


def render_full_report(report: ResearchReport):
    """Renders the complete research report."""
    # Title & metadata
    st.markdown(f"## 📑 {report.title}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sources Analyzed", report.sources_analyzed)
    col2.metric("Sources Used", report.sources_included)
    col3.metric("Sections", len(report.sections))

    st.markdown("---")

    # Executive Summary
    if report.executive_summary:
        st.info(f"**Executive Summary:** {report.executive_summary}")

    st.markdown("")

    # Sections
    for i, section in enumerate(sorted(report.sections, key=lambda s: s.order)):
        render_section(section, expanded=(i < 2))


def render_progress_tracker(stages_done: list[str], current_stage: str):
    """Renders a compact progress status tracker."""
    pipeline = [
        "Planning", "Search", "Extraction", "Evaluation",
        "Chunking", "Embedding", "Summarization",
        "Citations", "Report", "Export",
    ]
    cols = st.columns(len(pipeline))
    for col, stage in zip(cols, pipeline):
        if stage in stages_done:
            col.markdown(f"✅ **{stage}**")
        elif stage == current_stage:
            col.markdown(f"⏳ **{stage}**")
        else:
            col.markdown(f"⬜ {stage}")


def download_buttons(docx_path, pdf_path):
    """Renders download buttons for DOCX and PDF."""
    col1, col2 = st.columns(2)

    if docx_path and docx_path.exists():
        with open(docx_path, "rb") as f:
            col1.download_button(
                label="⬇️ Download DOCX",
                data=f.read(),
                file_name=docx_path.name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
    else:
        col1.button("⬇️ DOCX not available", disabled=True, use_container_width=True)

    if pdf_path and pdf_path.exists():
        with open(pdf_path, "rb") as f:
            col2.download_button(
                label="⬇️ Download PDF",
                data=f.read(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        col2.button("⬇️ PDF not available", disabled=True, use_container_width=True)
