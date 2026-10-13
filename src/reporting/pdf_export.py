"""PDF export for DrainMind using fpdf2.

Gracefully degrades if fpdf2 is not installed.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.utils.logging_config import get_logger

logger = get_logger("reporting.pdf_export")

_FPDF_AVAILABLE = False
try:
    from fpdf import FPDF, XPos, YPos
    _FPDF_AVAILABLE = True
except ImportError:
    logger.warning("fpdf2 not installed — PDF export unavailable.")


def is_pdf_available() -> bool:
    return _FPDF_AVAILABLE


def _sanitize_pdf_text(text: Any) -> str:
    if text is None:
        return ""
    text_str = str(text)
    replacements = {
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "*",
        "…": "...",
        "°": " deg",
    }
    for orig, rep in replacements.items():
        text_str = text_str.replace(orig, rep)
    return text_str.encode("latin-1", "replace").decode("latin-1")


class _DrainMindPDF(FPDF):
    def __init__(self, title: str = "DrainMind Inspection Report"):
        super().__init__()
        self.report_title = _sanitize_pdf_text(title)
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(True, 15)

    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(36, 91, 69)  # #245B45
        self.cell(0, 8, "DrainMind", align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(115, 122, 115)
        self.cell(0, 5, self.report_title, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_draw_color(221, 225, 218)
        self.line(15, self.get_y(), self.w - 15, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(115, 122, 115)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 6, f"Generated {ts} - DrainMind v2.0 - Page {self.page_no()}", align="C")


def generate_pdf_report(
    inspections: List[Dict[str, Any]],
    title: str = "Inspection Report",
    filters_applied: Optional[str] = None,
    include_methodology: bool = True,
) -> Optional[bytes]:
    """
    Generate a PDF report from inspection records.
    Returns bytes on success, None if fpdf2 is unavailable or an error occurs.
    """
    if not _FPDF_AVAILABLE:
        logger.error("fpdf2 not available — cannot generate PDF.")
        return None

    try:
        pdf = _DrainMindPDF(title=title)
        pdf.add_page()

        # ── Title Section ──────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(32, 37, 34)
        pdf.cell(0, 8, _sanitize_pdf_text(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(115, 122, 115)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 5, f"Generated: {ts}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if filters_applied:
            pdf.cell(0, 5, f"Filters: {_sanitize_pdf_text(filters_applied)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

        # ── Summary Statistics ─────────────────────────────────────────
        if inspections:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(36, 91, 69)
            pdf.cell(0, 6, "Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(32, 37, 34)
            scores = [r.get("priority_score") for r in inspections if r.get("priority_score") is not None]
            pdf.cell(0, 5, f"Total inspections: {len(inspections)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if scores:
                pdf.cell(0, 5, f"Average priority score: {sum(scores)/len(scores):.1f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.cell(0, 5, f"Maximum priority score: {max(scores):.1f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(4)

        # ── Records Table ─────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(36, 91, 69)
        pdf.cell(0, 6, "Inspection Records", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

        # Table header
        headers = ["Location", "Timestamp", "Status", "Debris%", "Water%", "Score", "Category"]
        col_w = [40, 35, 20, 16, 16, 16, 37]

        pdf.set_fill_color(245, 246, 243)
        pdf.set_draw_color(221, 225, 218)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(32, 37, 34)

        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 6, h, border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 7)
        fill = False
        for row in inspections:
            pdf.set_fill_color(251, 252, 250) if fill else pdf.set_fill_color(255, 255, 255)

            debris = f"{row['debris_coverage']:.1f}" if row.get("debris_coverage") is not None else "-"
            water = f"{row['water_coverage']:.1f}" if row.get("water_coverage") is not None else "-"
            score = f"{row['priority_score']:.1f}" if row.get("priority_score") is not None else "-"

            loc = str(row.get("location_name", row.get("location_id", "")))[:22]
            ts_raw = str(row.get("timestamp", ""))[:16].replace("T", " ")
            status = str(row.get("processing_status", ""))[:10]
            cat = str(row.get("priority_category", ""))[:20]

            cells = [loc, ts_raw, status, debris, water, score, cat]
            for i, cell in enumerate(cells):
                # Ensure clean ASCII string for Helvetica
                clean_cell = cell.encode("ascii", "replace").decode("ascii")
                pdf.cell(col_w[i], 5, clean_cell, border=1, fill=True)
            pdf.ln()
            fill = not fill

        # ── Methodology ───────────────────────────────────────────────
        if include_methodology:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(36, 91, 69)
            pdf.cell(0, 6, "Methodology and Limitations", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(32, 37, 34)
            methodology_text = (
                "DrainMind uses computer vision to analyse images and videos of drainage systems. "
                "Detection is performed using a YOLOv8 object detection model (Ultralytics). "
                "Where a trained drainage-specific model is not available, a generic COCO-pretrained model "
                "is used; results in this mode reflect visible objects in general and may not accurately "
                "represent drainage blockage.\n\n"
                "COVERAGE MEASUREMENT: Visible obstruction coverage is computed as the fraction of a "
                "central region of interest (ROI) occupied by detected or segmented objects. "
                "This is a pixel-level measurement and does NOT represent hydraulic blockage or "
                "physical drain capacity.\n\n"
                "PRIORITY SCORE: The inspection priority score R = 0.50*D + 0.30*W + 0.20*T is a "
                "configurable composite of visible obstruction (D), water indicator (W), and worsening "
                "trend (T). It is an INSPECTION PRIORITY INDICATOR, not a validated flood-risk coefficient. "
                "Scores should be interpreted alongside actual field observations.\n\n"
                "LIMITATIONS: Accuracy depends on image quality, lighting, camera angle, and model "
                "training. Results from different angles or lighting conditions may not be directly "
                "comparable. The system is intended for academic and demonstration use. It is a "
                "single-user local deployment and has not been validated for safety-critical decisions."
            )
            pdf.multi_cell(0, 4.5, methodology_text)

        buf = io.BytesIO()
        pdf.output(buf)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return None
