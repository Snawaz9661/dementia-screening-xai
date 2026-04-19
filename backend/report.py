from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .ml.features import CLASS_LABELS, FEATURE_LABELS, FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text).replace("&", "&amp;"), style)


def build_pdf_report(result: dict[str, Any], respondent: str = "Not specified") -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    filename = f"dementia_screening_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}.pdf"
    path = REPORT_DIR / filename

    styles = getSampleStyleSheet()
    title = styles["Title"]
    heading = styles["Heading2"]
    body = styles["BodyText"]
    body.leading = 14

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    story = [
        Paragraph("Explainable AI Dementia Screening Report", title),
        Spacer(1, 10),
        _paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", body),
        _paragraph(f"Completed by: {respondent}", body),
        Spacer(1, 12),
        Paragraph("Screening Result", heading),
    ]

    risk_data = [
        ["Risk Category", result["riskLabel"]],
        ["High-Risk Probability", f"{result['riskScore']}%"],
        ["Low / Moderate / High", " / ".join(f"{label}: {prob:.1%}" for label, prob in result["probabilities"].items())],
    ]
    risk_table = Table(risk_data, colWidths=[2.0 * inch, 4.8 * inch])
    risk_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([risk_table, Spacer(1, 12), Paragraph("Plain-Language Explanation", heading)])
    story.append(_paragraph(result["summary"], body))
    story.append(Spacer(1, 10))

    factor_rows = [["Factor", "Value", "Impact", "Explanation"]]
    for factor in result["topFactors"]:
        factor_rows.append(
            [
                factor["label"],
                str(factor["value"]),
                f"{factor['contribution']:+.3f}",
                factor["text"],
            ]
        )
    factor_table = Table(factor_rows, colWidths=[1.55 * inch, 0.75 * inch, 0.75 * inch, 3.75 * inch])
    factor_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([Paragraph("Top Contributing Factors", heading), factor_table, Spacer(1, 12)])

    input_rows = [["Indicator", "Value"]]
    values = result["input"]
    for name in FEATURE_NAMES:
        input_rows.append([FEATURE_LABELS[name], str(values[name])])
    input_table = Table(input_rows, colWidths=[3.2 * inch, 3.4 * inch])
    input_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([Paragraph("Input Summary", heading), input_table, Spacer(1, 12)])

    next_steps = {
        CLASS_LABELS[0]: "Continue routine cognitive wellness habits and repeat screening if new symptoms appear.",
        CLASS_LABELS[1]: "Schedule a clinical follow-up for a formal cognitive assessment and medication/history review.",
        CLASS_LABELS[2]: "Arrange timely clinical evaluation. Bring this report and any prior MMSE, MoCA, lab, or imaging records.",
    }
    story.extend(
        [
            Paragraph("Recommended Next Step", heading),
            _paragraph(next_steps.get(result["riskLabel"], next_steps[CLASS_LABELS[1]]), body),
            Spacer(1, 8),
            _paragraph(result["disclaimer"], body),
        ]
    )

    doc.build(story)
    return path
