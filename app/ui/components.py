from __future__ import annotations
import streamlit as st
from app.models.report import ResearchReport

_CSS_PATH = __file__.replace('components.py', '.css_cache')
with open(_CSS_PATH, encoding='utf-8') as _f:
    CSS = '<style>' + _f.read() + '</style>'

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)

def render_hero():
    h = [
        '<div class="hero-section">',
        '<div class="hero-badge">AI-Powered Research</div>',
        '<h1 class="hero-title">Research &amp; Report Agent</h1>',
        '<p class="hero-subtitle">Autonomous web research &middot; RAG synthesis &middot; Citation-backed findings &middot; Professional export</p>',
        '<div class="hero-stats">',
        '<div class="hero-stat"><div class="hero-stat-value">RAG</div><div class="hero-stat-label">Grounded AI</div></div>',
        '<div class="hero-stat"><div class="hero-stat-value">PDF &amp; DOCX</div><div class="hero-stat-label">Export formats</div></div>',
        '<div class="hero-stat"><div class="hero-stat-value">100%</div><div class="hero-stat-label">Free APIs</div></div>',
        '</div></div>',
    ]
    st.markdown(''.join(h), unsafe_allow_html=True)

def render_section_header(label, icon=''):
    pre = f'{icon} ' if icon else ''
    st.markdown(
        f'<div class="section-header"><div class="section-header-text">{pre}{label}</div>'
        f'<div class="section-header-line"></div></div>',
        unsafe_allow_html=True)

def render_progress_tracker(stages_done, current_stage):
    pipeline = [('Planning','🧠'),('Search','🔍'),('Extraction','📄'),
        ('Evaluation','⚖️'),('Chunking','✂️'),('Embedding','🔢'),
        ('Summarization','✍️'),('Citations','📎'),('Report','📑'),('Export','💾')]
    render_section_header('Pipeline Progress')
    parts = []
    for name, ico in pipeline:
        if name in stages_done:
            parts.append(f'<span class="progress-stage done">&#10003; {ico} {name}</span>')
        elif name == current_stage:
            parts.append(f'<span class="progress-stage active">&#9203; {ico} {name}</span>')
        else:
            parts.append(f'<span class="progress-stage">{ico} {name}</span>')
    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:4px;">'+''.join(parts)+'</div>',
        unsafe_allow_html=True)

def render_finding_card(finding):
    cls = 'finding-card uncertain' if finding.uncertain else 'finding-card'
    icon = '&#9888;' if finding.uncertain else '&#9670;'
    cite_html = ''
    if finding.citations:
        badges = ''.join(f'<a class="cite-badge" href="{c.url}" target="_blank">{c.marker}</a>' for c in finding.citations)
        cite_html = f'<div class="finding-citations">{badges}</div>'
    st.markdown(f'<div class="{cls}"><p class="finding-claim">{icon} {finding.claim}</p>{cite_html}</div>',
        unsafe_allow_html=True)

def render_full_report(report):
    st.markdown(
        f'<div class="report-title-block"><h2>{report.title}</h2>'
        f'<p>Depth: {report.depth.capitalize()} &middot; {report.sources_included} sources used</p></div>',
        unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Analyzed', report.sources_analyzed)
    col2.metric('Used', report.sources_included)
    col3.metric('Sections', len(report.sections))
    col4.metric('Citations', len(report.all_citations))
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if report.executive_summary:
        render_section_header('Executive Summary', '💡')
        st.markdown(f'<div class="exec-summary"><p>{report.executive_summary}</p></div>',
            unsafe_allow_html=True)
    render_section_header('Report Sections', '📋')
    for i, section in enumerate(sorted(report.sections, key=lambda s: s.order)):
        if section.heading == 'References': continue
        with st.expander(f'  {section.heading}', expanded=(i < 2)):
            if section.content:
                for line in section.content.split('\n'):
                    line = line.strip()
                    if not line: continue
                    if line.startswith('* '):
                        st.markdown(f'&bull; {line[2:]}', unsafe_allow_html=True)
                    else:
                        st.markdown(line)
            if section.findings:
                for f in section.findings: render_finding_card(f)
    if report.all_citations:
        render_section_header('References', '📎')
        for cit in report.all_citations:
            st.markdown(
                f'<div class="ref-item"><span class="ref-num">{cit.marker}</span>'
                f'<div class="ref-content"><p class="ref-title">{cit.title or "Untitled"}</p>'
                f'<a class="ref-url" href="{cit.url}" target="_blank">{cit.url}</a></div></div>',
                unsafe_allow_html=True)

def download_buttons(docx_path, pdf_path):
    render_section_header('Download Report', '💾')
    col1, col2 = st.columns(2)
    if docx_path and docx_path.exists():
        with open(docx_path, 'rb') as f:
            col1.download_button('📄  Download DOCX', data=f.read(),
                file_name=docx_path.name,
                mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                use_container_width=True)
    else:
        col1.button('📄  DOCX unavailable', disabled=True, use_container_width=True)
    if pdf_path and pdf_path.exists():
        with open(pdf_path, 'rb') as f:
            col2.download_button('📕  Download PDF', data=f.read(),
                file_name=pdf_path.name, mime='application/pdf', use_container_width=True)
    else:
        col2.button('📕  PDF unavailable', disabled=True, use_container_width=True)
