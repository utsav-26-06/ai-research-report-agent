"""
PDF Exporter (TASK-017).
Exports a ResearchReport as a professionally formatted .pdf file using ReportLab.
"""

import asyncio
import logging
from pathlib import Path

from app.models.report import ResearchReport

logger = logging.getLogger(__name__)


class PDFExporterError(Exception):
    """Raised when PDF export fails."""


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
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            BaseDocTemplate,
            Frame,
            PageTemplate,
            Paragraph,
            Spacer,
            HRFlowable,
            ListFlowable,
            ListItem,
        )

        # -----------------------------------------------------------------
        # Styles
        # -----------------------------------------------------------------
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=24,
            textColor=colors.HexColor("#1A1A2E"),
            spaceAfter=6,
            alignment=TA_CENTER,
        )
        meta_style = ParagraphStyle(
            "Meta",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=12,
        )
        h1_style = ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#1A1A2E"),
            spaceBefore=14,
            spaceAfter=6,
            borderPad=2,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
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
        )
        citation_style = ParagraphStyle(
            "Citation",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#0070BB"),
            leftIndent=12,
            spaceAfter=2,
        )

        # -----------------------------------------------------------------
        # Document setup with page numbers
        # -----------------------------------------------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c for c in report.title if c.isalnum() or c in " _-")[:60].strip()
        filename = f"{safe_title}.pdf"
        filepath = self.output_dir / filename

        def add_page_number(canvas, doc):
            """Footer with page number."""
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#888888"))
            page_num = canvas.getPageNumber()
            canvas.drawCentredString(A4[0] / 2, 1.5 * cm, f"Page {page_num}")
            canvas.drawString(2 * cm, 1.5 * cm, report.title[:60])
            canvas.restoreState()

        doc = BaseDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=2.5 * cm,
            leftMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
        )
        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
        template = PageTemplate(id="main", frames=frame, onPage=add_page_number)
        doc.addPageTemplates([template])

        # -----------------------------------------------------------------
        # Story (content flow)
        # -----------------------------------------------------------------
        story = []

        # Title
        story.append(Paragraph(report.title, title_style))
        story.append(Paragraph(
            f"Research Date: {report.research_date.strftime('%Y-%m-%d')} &nbsp;|&nbsp; "
            f"Depth: {report.depth.capitalize()} &nbsp;|&nbsp; "
            f"Sources: {report.sources_included}",
            meta_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1A1A2E")))
        story.append(Spacer(1, 0.5 * cm))

        # Executive Summary
        if report.executive_summary:
            story.append(Paragraph("Executive Summary", h1_style))
            story.append(Paragraph(report.executive_summary, body_style))
            story.append(Spacer(1, 0.3 * cm))

        # Sections
        sorted_sections = sorted(report.sections, key=lambda s: s.order)
        for section in sorted_sections:
            story.append(Paragraph(section.heading, h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
            story.append(Spacer(1, 0.2 * cm))

            if section.content:
                for line in section.content.split("\n"):
                    line = line.strip()
                    if not line:
                        story.append(Spacer(1, 0.15 * cm))
                        continue
                    if line.startswith("* "):
                        story.append(Paragraph(f"• {line[2:]}", bullet_style))
                    elif line.startswith("## "):
                        story.append(Paragraph(f"<b>{line[3:]}</b>", body_style))
                    elif line.startswith("### "):
                        story.append(Paragraph(f"<i><b>{line[4:]}</b></i>", body_style))
                    else:
                        story.append(Paragraph(line, body_style))

            if section.findings:
                for finding in section.findings:
                    icon = "⚠ " if finding.uncertain else "• "
                    story.append(Paragraph(f"{icon}{finding.claim}", bullet_style))
                    if finding.citations:
                        markers = " ".join(c.marker for c in finding.citations)
                        story.append(Paragraph(
                            f'<font color="#0070BB">{markers}</font>', citation_style
                        ))

            story.append(Spacer(1, 0.4 * cm))

        # -----------------------------------------------------------------
        # Build
        # -----------------------------------------------------------------
        doc.build(story)
        logger.info(f"PDF saved to: {filepath}")
        return filepath
