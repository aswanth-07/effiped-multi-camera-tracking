"""Generate the EffiPed technical report from the canonical result fixture."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "research" / "results" / "summary.json"
ARCHITECTURE = ROOT / "docs" / "architecture" / "effiped-architecture.png"
OUTPUT = ROOT / "research" / "report" / "effiped-technical-report.pdf"
PUBLIC_OUTPUT = ROOT / "docs" / "report" / "effiped-technical-report.pdf"

PAGE = landscape(A4)
NAVY = colors.HexColor("#06100F")
PANEL = colors.HexColor("#0A211D")
TEAL = colors.HexColor("#24E6BD")
TEAL_SOFT = colors.HexColor("#8CF8DF")
BRONZE = colors.HexColor("#E8A94D")
INK = colors.HexColor("#ECF9F6")
MUTED = colors.HexColor("#92AAA5")
LINE = colors.HexColor("#25554C")
RED = colors.HexColor("#FF6A76")


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def background(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE[0], PAGE[1], fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#12332D"))
    canvas.setLineWidth(0.3)
    step = 18 * mm
    x = 0
    while x < PAGE[0]:
        canvas.line(x, 0, x, PAGE[1])
        x += step
    y = 0
    while y < PAGE[1]:
        canvas.line(0, y, PAGE[0], y)
        y += step
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(PAGE[0] - 15 * mm, 9 * mm, f"EffiPed technical report · {doc.page}")
    canvas.restoreState()


def metric_card(value: str, label: str, note: str, styles) -> Table:
    content = [
        [para(value, styles["metric"])],
        [para(label, styles["card_title"])],
        [para(note, styles["tiny"])],
    ]
    card = Table(content, colWidths=[57 * mm], rowHeights=[14 * mm, 8 * mm, 9 * mm])
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("LINEBELOW", (0, 0), (-1, 0), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return card


def evidence_table(rows, widths, styles) -> Table:
    formatted = []
    for row_index, row in enumerate(rows):
        style = styles["table_head"] if row_index == 0 else styles["table"]
        formatted.append([para(str(cell), style) for cell in row])
    table = Table(formatted, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14372F")),
                ("BACKGROUND", (0, 1), (-1, -1), PANEL),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("GRID", (0, 0), (-1, -1), 0.45, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    contest = data["verified_contest_system"]
    poster = data["contest_submission_snapshot"]
    partjde = data["post_contest_evolution"]["partjde"]
    boxjde = data["post_contest_evolution"]["boxjde"]

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Eyebrow", fontName="Helvetica-Bold", fontSize=7.5, leading=10, textColor=TEAL, spaceAfter=3, uppercase=True))
    styles.add(ParagraphStyle(name="TitleWhite", fontName="Helvetica-Bold", fontSize=28, leading=30, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle(name="H1White", fontName="Helvetica-Bold", fontSize=22, leading=24, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2White", fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=INK, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyWhite", fontName="Helvetica", fontSize=9.2, leading=13, textColor=colors.HexColor("#BDD0CC"), spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallWhite", fontName="Helvetica", fontSize=7.8, leading=10.5, textColor=MUTED))
    styles.add(ParagraphStyle(name="TinyWhite", fontName="Helvetica", fontSize=6.6, leading=8.5, textColor=MUTED))
    styles.add(ParagraphStyle(name="Metric", fontName="Courier-Bold", fontSize=21, leading=22, textColor=TEAL))
    styles.add(ParagraphStyle(name="CardTitle", fontName="Helvetica-Bold", fontSize=8.5, leading=10, textColor=INK))
    styles.add(ParagraphStyle(name="TableHead", fontName="Helvetica-Bold", fontSize=7.4, leading=9, textColor=TEAL_SOFT))
    styles.add(ParagraphStyle(name="TableText", fontName="Helvetica", fontSize=7.4, leading=9.5, textColor=INK))
    styles.add(ParagraphStyle(name="Equation", fontName="Times-Roman", fontSize=10, leading=12, textColor=TEAL_SOFT, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Award", fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=colors.HexColor("#221608"), alignment=TA_CENTER))
    alias = {
        "eyebrow": styles["Eyebrow"], "title": styles["TitleWhite"], "h1": styles["H1White"],
        "h2": styles["H2White"], "body": styles["BodyWhite"], "small": styles["SmallWhite"],
        "tiny": styles["TinyWhite"], "metric": styles["Metric"], "card_title": styles["CardTitle"],
        "table_head": styles["TableHead"], "table": styles["TableText"], "equation": styles["Equation"],
    }

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=PAGE,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=15 * mm,
        title=data["project"]["title"],
        author="Aswanth Raj",
        subject="Contest system architecture, evidence, and research evolution",
    )
    story = []

    award = Table([[para("3RD PRIZE · STUDENT INNOVATION PROJECT CONTEST 2026", styles["Award"])]], colWidths=[90 * mm], rowHeights=[11 * mm])
    award.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BRONZE), ("BOX", (0, 0), (-1, -1), 0, BRONZE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [
        award,
        Spacer(1, 16 * mm),
        para("EFFIPED / TECHNICAL REPORT", alias["eyebrow"]),
        para(data["project"]["title"], alias["title"]),
        para(
            "A compact joint system for pedestrian detection, local tracking, part-aware "
            "appearance description, cross-camera candidate retrieval, and analyst review.",
            alias["body"],
        ),
        Spacer(1, 10 * mm),
        Table(
            [[
                metric_card(f"{contest['pdestre']['validation']['rank1_cross']}%", "P-DESTRE validation Rank-1", "Cross-camera · Protocol D", alias),
                metric_card(f"{contest['mot17']['mota']:.2f}", "MOT17 val-half MOTA", "Protocol A · BoT-SORT", alias),
                metric_card(f"{contest['footprint']['parameters_m']}M", "Canonical Tier-1", f"approx. {contest['footprint']['pipeline_fps_approx']} full-pipeline FPS", alias),
            ]],
            colWidths=[62 * mm] * 3,
            hAlign="LEFT",
        ),
        Spacer(1, 12 * mm),
        para(
            "<b>Aswanth Raj</b> · Guide: Sri Preethaa KR · Vertical 1: AI &amp; Intelligent Systems · "
            "VIT Vellore School of Computer Science and Engineering (SCOPE)",
            alias["small"],
        ),
        PageBreak(),
    ]

    story += [
        para("01 / END-TO-END SYSTEM", alias["eyebrow"]),
        para("One visual backbone carries a person from pixels to reviewable evidence.", alias["h1"]),
        Image(str(ARCHITECTURE), width=235 * mm, height=132.2 * mm),
        Spacer(1, 3 * mm),
        para(
            "The PowerPoint source is fully editable. Detection, four-strip RoI identity readout, "
            "CoordinateAttention fusion, BoT-SORT association, and the final analyst evidence view "
            "are shown as one connected pipeline.",
            alias["small"],
        ),
        PageBreak(),
    ]

    contest_rows = [
        ["Evaluation boundary", "Rank-1", "Detection", "Tracking"],
        ["P-DESTRE fold-0 validation · Protocol D", f"{contest['pdestre']['validation']['rank1_cross']}%", f"{contest['pdestre']['validation']['detection_map50']}% mAP@0.5", "—"],
        ["P-DESTRE fold-0 test · Protocol E", f"{contest['pdestre']['test']['rank1_cross']}%", f"{contest['pdestre']['test']['detection_map50']}% mAP@0.5", "—"],
        ["MOT17 val-half · Protocol A", "—", "—", f"{contest['mot17']['mota']:.2f} MOTA · {contest['mot17']['idf1']:.2f} IDF1 · {contest['mot17']['hota']:.2f} HOTA"],
    ]
    story += [
        para("02 / EVIDENCE LEDGER", alias["eyebrow"]),
        para("Contest evidence is separated from the submitted poster and later research.", alias["h1"]),
        evidence_table(contest_rows, [82 * mm, 35 * mm, 54 * mm, 82 * mm], alias),
        Spacer(1, 8 * mm),
        KeepTogether([
            para("Submitted poster snapshot", alias["h2"]),
            para(
                f"The contest poster displayed <b>{poster['reported_parameters_m']}M / "
                f"{poster['reported_fps']} FPS / {poster['reported_rank1_cross']}% Rank-1</b>. "
                "The later canonical registry associates Tier-1 with <b>7.78M parameters and "
                "approximately 18 FPS</b> for the full pipeline. The +16.2 pp row combined multiple "
                "changes and is not a pure part-only ablation.",
                alias["body"],
            ),
        ]),
        Spacer(1, 4 * mm),
        para("Canonical footprint", alias["h2"]),
        evidence_table(
            [
                ["Parameters", "Input", "Descriptor", "Benchmark device", "Throughput boundary"],
                [
                    f"{contest['footprint']['parameters_m']}M",
                    contest["footprint"]["input_resolution"],
                    f"{contest['footprint']['descriptor_dim']}-D",
                    contest["footprint"]["device"],
                    f"approx. {contest['footprint']['pipeline_fps_approx']} FPS · full pipeline",
                ],
            ],
            [35 * mm, 35 * mm, 37 * mm, 80 * mm, 66 * mm],
            alias,
        ),
        Spacer(1, 8 * mm),
        para(
            "Every value above is loaded from research/results/summary.json. Unsupported headline "
            "claims such as “state of the art,” unconditional “real-time,” or privacy guarantees are excluded.",
            alias["small"],
        ),
        PageBreak(),
    ]

    evolution_rows = [
        ["Stage", "Research question", "Evidence"],
        ["EffiPed · contest", "Can one compact model detect, track, and describe people across four cameras?", "62.8% validation · 61.3% test Rank-1"],
        ["PartJDE · matched refinement", "What is the matched gain from part-aware RoI-strip readout?", f"+{partjde['matched_part_readout_gain_pp']} pp validation Rank-1 · 7.92M · 27.0 FPS"],
        ["BoxJDE · five-fold", "How much does full-box descriptor support change retrieval under a matched JDE model?", f"+{boxjde['natural_predicted_rank1_gain_pp']} pp predicted-box Rank-1 · +{boxjde['natural_e2e_rank1_gain_pp']} pp E2E"],
    ]
    story += [
        para("03 / CONTEST TO RESEARCH", alias["eyebrow"]),
        para("The contest prototype became a controlled descriptor-readout investigation.", alias["h1"]),
        evidence_table(evolution_rows, [44 * mm, 116 * mm, 93 * mm], alias),
        Spacer(1, 9 * mm),
        para("Part-aware descriptor", alias["h2"]),
        evidence_table(
            [
                ["Component", "Role", "Editable equation"],
                ["Four body strips", "Preserve localized appearance under partial occlusion", "z = normalize(sum_k alpha_k z_k)"],
                ["Mutual visibility", "Compare only evidence visible in both observations", "s(q,g) = sum_k v_k(q)v_k(g) cos(zq,k,zg,k) / sum_k v_k(q)v_k(g)"],
                ["Metric learning", "Separate hard identities with XBM-expanded negatives", "Ltri = max(0, m + d(a,p) - d(a,n))"],
                ["ArcFace", "Angular identity classification during training", "cos(theta_y + m)"],
            ],
            [55 * mm, 94 * mm, 104 * mm],
            alias,
        ),
        Spacer(1, 7 * mm),
        para(
            "<b>Protocol boundary:</b> BoxJDE’s primary P-DESTRE study is a constructed per-date "
            "readout ablation, not official Task 4. The public BoxJDE repository carries its full "
            "five-fold report rather than duplicating that code here.",
            alias["body"],
        ),
        PageBreak(),
    ]

    story += [
        para("04 / PRODUCT, LIMITATIONS, AND RELEASE", alias["eyebrow"]),
        para("A research model is only useful when its uncertainty and data lifecycle remain visible.", alias["h1"]),
        evidence_table(
            [
                ["Surface", "What ships", "Safety boundary"],
                ["Vercel demo", "Precomputed four-camera replay with clickable tracks and candidate bands", "No upload or remote inference"],
                ["Local FastAPI mode", "Uploads, indexing progress, WebSockets, gallery review, search-by-example", "Loopback by default; explicit job deletion"],
                ["Model artifacts", "Versioned manifest and unavailable-weight behavior", "Weights withheld pending source-by-source rights review"],
                ["Media", "Optimized still and seven-second VP9 excerpt", "P-DESTRE attribution · CC BY-NC-SA 4.0 · non-commercial"],
            ],
            [48 * mm, 116 * mm, 89 * mm],
            alias,
        ),
        Spacer(1, 8 * mm),
        Table(
            [[
                [
                    para("Known limitations", alias["h2"]),
                    para("Occlusion, pose, lighting, clothing ambiguity, localization error, crowd density, timing, and domain shift can change rankings. A high score is not identity proof.", alias["body"]),
                ],
                [
                    para("Responsible operation", alias["h2"]),
                    para("Use authorized video, minimize retention, restrict access, preserve human review, validate performance locally, and perform a purpose-specific legal and privacy assessment.", alias["body"]),
                ],
            ]],
            colWidths=[124 * mm, 124 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]),
        ),
        Spacer(1, 9 * mm),
        para("Attribution and links", alias["h2"]),
        para(
            "Software © 2026 Aswanth Raj · Apache-2.0. P-DESTRE-derived media is separately "
            "licensed CC BY-NC-SA 4.0 for this non-commercial showcase. "
            "P-DESTRE paper: https://arxiv.org/abs/2004.02782 · "
            "Repository: https://github.com/aswanth-07/effiped-multi-camera-tracking · "
            "BoxJDE: https://github.com/aswanth-07/boxjde-person-search",
            alias["body"],
        ),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=background, onLaterPages=background)
    PUBLIC_OUTPUT.write_bytes(OUTPUT.read_bytes())
    print(OUTPUT)


if __name__ == "__main__":
    build()
