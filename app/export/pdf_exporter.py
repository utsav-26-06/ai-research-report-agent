"""
PDF Exporter (TASK-017).
Exports a ResearchReport as a professionally formatted .pdf file using ReportLab.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Spacer,
)

from app.models.report import ReportSection, ResearchReport

logger = logging.getLogger(__name__)

MAX_SAFE_TITLE_LEN = 100


class PDFExporterError(Exception):
    """Raised when PDF export fails."""


def _sanitize_filename(title: str, max_len: int = MAX_SAFE_TITLE_LEN) -> str:
    """Sanitize title for use as a filename without cutting words mid-token."""
    safe = "".join(c for c in title if c.isalnum() or c in " _-").strip()
    if len(safe) > max_len:
        safe = safe[:max_len].rsplit(" ", 1)[0].strip()
    return safe or "Research_Report"


def _markdown_to_reportlab(text: str) -> str:
    """Convert basic markdown bold and italic syntax to ReportLab XML tags."""
    # Convert **bold** to <b>bold</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Convert *italic* to <i>italic</i>
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    return text


def _get_pdf_styles() -> dict[str, ParagraphStyle]:
    """Create and return the collection of ParagraphStyles used across the PDF."""
    sample = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=sample["Title"],
        fontSize=24,
        textColor=colors.HexColor("#1A1A2E"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=sample["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=sample["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1A1A2E"),
        spaceBefore=14,
        spaceAfter=6,
        borderPad=2,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=sample["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#222222"),
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=18,
        spaceAfter=3,
        alignment=TA_LEFT,
    )
    citation_style = ParagraphStyle(
        "Citation",
        parent=sample["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#0070BB"),
        leftIndent=12,
        spaceAfter=2,
    )
    reference_style = ParagraphStyle(
        "ReferenceItem",
        parent=sample["Normal"],
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor("#222222"),
        alignment=TA_LEFT,
        leftIndent=24,
        firstLineIndent=-24,
        spaceAfter=7,
    )

    return {
        "title": title_style,
        "meta": meta_style,
        "h1": h1_style,
        "body": body_style,
        "bullet": bullet_style,
        "citation": citation_style,
        "reference": reference_style,
    }


def _make_page_callback(report: ResearchReport) -> Callable[[Any, Any], None]:
    """Build the onPage callback to draw the header/footer with page numbers."""

    def on_page(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#777777"))

        # Divider line above footer
        canvas.setStrokeColor(colors.HexColor("#D8D8D8"))
        canvas.setLineWidth(0.5)
        line_y = 1.6 * cm
        text_y = 1.1 * cm
        left_x = doc.leftMargin
        right_x = A4[0] - doc.rightMargin
        canvas.line(left_x, line_y, right_x, line_y)

        # Page number on far right
        page_str = f"Page {canvas.getPageNumber()}"
        canvas.drawRightString(right_x, text_y, page_str)

        # Topic / title on far left (clean word-boundary truncation if too long)
        page_str_width = canvas.stringWidth(page_str, "Helvetica", 8)
        max_title_width = (right_x - left_x) - page_str_width - 1.5 * cm

        title_text = (report.title or report.research_question or "").strip()
        if canvas.stringWidth(title_text, "Helvetica", 8) > max_title_width:
            words = title_text.split()
            truncated = ""
            for w in words:
                cand = f"{truncated} {w}".strip() if truncated else w
                if canvas.stringWidth(cand + "...", "Helvetica", 8) <= max_title_width:
                    truncated = cand
                else:
                    break
            title_text = f"{truncated}..." if truncated else title_text[:30] + "..."

        canvas.drawString(left_x, text_y, title_text)
        canvas.restoreState()

    return on_page


def _render_section_content(
    section: ReportSection,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    """Convert a section's markdown lines and findings into ReportLab flowables."""
    flowables: list[Flowable] = []
    is_references = section.heading.strip().lower() == "references"

    if section.content:
        for raw_line in section.content.split("\n"):
            line = raw_line.strip()
            if not line:
                flowables.append(Spacer(1, 0.15 * cm))
                continue

            if is_references:
                flowables.append(Paragraph(line, styles["reference"]))
            elif line.startswith(("* ", "- ")):
                bullet_content = _markdown_to_reportlab(line[2:])
                flowables.append(Paragraph(f"• {bullet_content}", styles["bullet"]))
            elif line.startswith("## "):
                heading_content = _markdown_to_reportlab(line[3:])
                flowables.append(Paragraph(f"<b>{heading_content}</b>", styles["body"]))
            elif line.startswith("### "):
                subheading_content = _markdown_to_reportlab(line[4:])
                flowables.append(Paragraph(f"<i><b>{subheading_content}</b></i>", styles["body"]))
            else:
                formatted_body = _markdown_to_reportlab(line)
                flowables.append(Paragraph(formatted_body, styles["body"]))

    if section.findings:
        for finding in section.findings:
            icon = "⚠ " if finding.uncertain else "• "
            formatted_claim = _markdown_to_reportlab(finding.claim)
            flowables.append(Paragraph(f"{icon}{formatted_claim}", styles["bullet"]))

            # Render citation marker line only if markers are not already in the claim text
            if finding.citations:
                markers = " ".join(c.marker for c in finding.citations)
                if markers and markers not in finding.claim:
                    flowables.append(
                        Paragraph(
                            f'<font color="#0070BB">{markers}</font>',
                            styles["citation"],
                        )
                    )

    return flowables


class PDFExporter:
    """
    Exports a ResearchReport to a .pdf file using ReportLab.

    Args:
        output_dir: Directory where the file will be saved.
    """

    def __init__(self, output_dir: str | Path = "./outputs"):
        self.output_dir = Path(output_dir)

    async def export(self, report: ResearchReport) -> Path:
        """
        Export the report to a .pdf file.

        Returns:
            Path to the saved .pdf file.

        Raises:
            PDFExporterError: On any file-creation failure.
        """
        try:
            return await asyncio.to_thread(self._build_pdf, report)
        except Exception as e:
            raise PDFExporterError(f"Failed to export PDF: {e}") from e

    def _build_pdf(self, report: ResearchReport) -> Path:
        """Synchronous PDF construction (runs in a thread)."""
        styles = _get_pdf_styles()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = _sanitize_filename(report.title)
        filename = f"{safe_title}.pdf"
        filepath = self.output_dir / filename

        on_page_callback = _make_page_callback(report)

        doc = BaseDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=2.5 * cm,
            leftMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
        )
        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
        template = PageTemplate(id="main", frames=frame, onPage=on_page_callback)
        doc.addPageTemplates([template])

        # Story (content flow)
        story: list[Flowable] = [
            Paragraph(_markdown_to_reportlab(report.title), styles["title"]),
            Paragraph(
                f"Research Date: {report.research_date.strftime('%Y-%m-%d')} &nbsp;|&nbsp; "
                f"Depth: {report.depth.capitalize()} &nbsp;|&nbsp; "
                f"Sources: {report.sources_included}",
                styles["meta"],
            ),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1A1A2E")),
            Spacer(1, 0.5 * cm),
        ]

        # Executive Summary
        if report.executive_summary:
            story.append(Paragraph("Executive Summary", styles["h1"]))
            story.append(
                Paragraph(
                    _markdown_to_reportlab(report.executive_summary),
                    styles["body"],
                )
            )
            story.append(Spacer(1, 0.3 * cm))

        # Sections
        sorted_sections = sorted(report.sections, key=lambda s: s.order)
        for section in sorted_sections:
            story.append(Paragraph(section.heading, styles["h1"]))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
            story.append(Spacer(1, 0.2 * cm))

            story.extend(_render_section_content(section, styles))
            story.append(Spacer(1, 0.4 * cm))

        # Build PDF
        doc.build(story)
        logger.info(f"PDF saved to: {filepath}")
        return filepath
