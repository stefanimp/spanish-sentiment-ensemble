"""Build the technical PDF from docs/report.md.

Install the optional report dependencies: python -m pip install -e ".[report]".
The source supports headings, paragraphs, lists, pipe tables and fenced code.
A horizontal rule starts a new page. Generated reports use embedded TrueType fonts.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#15263d")
MUTED = colors.HexColor("#53647a")
ACCENT = colors.HexColor("#137f7b")


def rich(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)


def main() -> None:
    font_dir = Path(reportlab.__file__).parent / "fonts"
    windows = Path("C:/Windows/Fonts")
    regular = windows / "segoeui.ttf" if windows.exists() else font_dir / "Vera.ttf"
    bold = windows / "segoeuib.ttf" if windows.exists() else font_dir / "VeraBd.ttf"
    pdfmetrics.registerFont(TTFont("Report", str(regular)))
    pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold")
    base = getSampleStyleSheet()
    styles = {
        "body": ParagraphStyle(
            "Body",
            fontName="Report",
            fontSize=10,
            leading=14.3,
            textColor=NAVY,
            spaceAfter=9,
            alignment=TA_LEFT,
        ),
        "title": ParagraphStyle(
            "Title", fontName="ReportBold", fontSize=29, leading=34, textColor=NAVY, spaceAfter=15
        ),
        "section": ParagraphStyle(
            "Section",
            fontName="ReportBold",
            fontSize=15,
            leading=19,
            textColor=ACCENT,
            spaceBefore=9,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "meta": ParagraphStyle(
            "Meta", fontName="Report", fontSize=9, leading=13, textColor=MUTED, spaceAfter=16
        ),
        "cell": ParagraphStyle(
            "Cell", fontName="Report", fontSize=8.6, leading=12, textColor=NAVY, splitLongWords=True
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=11,
            textColor=NAVY,
            backColor=colors.HexColor("#f0f4f7"),
            borderPadding=8,
            spaceBefore=4,
            spaceAfter=12,
        ),
    }
    source = (ROOT / "docs/report.md").read_text(encoding="utf-8")
    lines = source.splitlines()
    title = lines[0].lstrip("# ")
    story = [
        Paragraph("APPLIED MACHINE LEARNING / TECHNICAL REPORT", styles["meta"]),
        HRFlowable(width="100%", thickness=2, color=ACCENT),
        Spacer(1, 6 * mm),
    ]
    width = 174 * mm
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        if line == "---":
            story.append(PageBreak())
        elif line.startswith("### "):
            story.append(Paragraph(rich(line[4:]), styles["meta"]))
        elif line.startswith("## "):
            story.append(Paragraph(rich(line[3:]), styles["section"]))
        elif line.startswith("# "):
            story.append(Paragraph(rich(line[2:]), styles["title"]))
        elif line.startswith("```"):
            code = []
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(html.escape(lines[index]).replace(" ", "&nbsp;"))
                index += 1
            index += 1
            story.append(Paragraph("<br/>".join(code), styles["code"]))
        elif line.startswith("|"):
            rows = [line]
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index].strip())
                index += 1
            cells = [[value.strip() for value in row.strip("|").split("|")] for row in rows]
            cells = [row for row in cells if not all(re.fullmatch(r":?-+:?", item) for item in row)]
            count = len(cells[0])
            fractions = [0.31, 0.69] if count == 2 else [0.27, 0.37, 0.36]
            table = Table(
                [
                    [
                        Paragraph(
                            rich(value).replace("/", "/<br/>", 1)
                            if value.startswith(("pysentimiento/", "cardiffnlp/", "dccuchile/"))
                            else rich(value),
                            styles["cell"],
                        )
                        for value in row
                    ]
                    for row in cells
                ],
                colWidths=[width * value for value in fractions],
                repeatRows=1,
                hAlign="LEFT",
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfece9")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f4f6f8")],
                        ),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACCENT),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            story.extend([table, Spacer(1, 4 * mm)])
        elif line.startswith("- "):
            story.append(Paragraph("• " + rich(line[2:]), styles["body"]))
        else:
            paragraph = [line]
            while (
                index < len(lines)
                and lines[index].strip()
                and not lines[index].startswith(("#", "|", "```", "- "))
                and lines[index].strip() != "---"
            ):
                paragraph.append(lines[index].strip())
                index += 1
            story.append(Paragraph(rich(" ".join(paragraph)), styles["body"]))

    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7e0e7"))
        canvas.line(18 * mm, 17 * mm, 192 * mm, 17 * mm)
        canvas.setFont("Report", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 12 * mm, f"stefanimp · {title}")
        canvas.drawRightString(192 * mm, 12 * mm, str(doc.page))
        if doc.page > 1:
            canvas.setFont("Report", 8)
            canvas.drawString(18 * mm, 284 * mm, "TECHNICAL REPORT / STEFANIMP")
        canvas.restoreState()

    output = ROOT / "docs/report.pdf"
    doc = SimpleDocTemplate(
        str(output),
        pagesize=(210 * mm, 297 * mm),
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=24 * mm,
        title=title,
        author="stefanimp",
        subject="Project design, results and reproducibility",
    )
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(output)


if __name__ == "__main__":
    main()
