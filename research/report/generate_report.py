"""Generate the standalone EffiPed technical report from the evidence fixture."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "research" / "results" / "summary.json"
REPORT_OUTPUT = ROOT / "research" / "report" / "effiped-technical-report.pdf"
DOCS_OUTPUT = ROOT / "docs" / "report" / "effiped-technical-report.pdf"

PAGE = landscape(A4)
NAVY = colors.HexColor("#070A0F")
SURFACE = colors.HexColor("#0D151D")
SURFACE_2 = colors.HexColor("#111D26")
LINE = colors.HexColor("#294049")
TEXT = colors.HexColor("#ECF5F4")
MUTED = colors.HexColor("#9CB0B2")
FAINT = colors.HexColor("#647A7E")
TEAL = colors.HexColor("#22C7B8")
TEAL_LIGHT = colors.HexColor("#75F0E2")
BLUE = colors.HexColor("#70B8FF")


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle(
            "Eyebrow",
            parent=sample["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=TEAL,
            spaceAfter=5,
        ),
        "title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=31,
            textColor=TEXT,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=23,
            textColor=TEXT,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=TEXT,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=13,
            textColor=MUTED,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=10,
            textColor=FAINT,
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=sample["Normal"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=24,
            textColor=TEAL_LIGHT,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=sample["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=TEXT,
            alignment=TA_CENTER,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        ),
        "cell_bold": ParagraphStyle(
            "CellBold",
            parent=sample["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=TEXT,
        ),
    }


def metric_card(value: str, label: str, note: str, styles: dict[str, ParagraphStyle]) -> Table:
    table = Table(
        [
            [paragraph(value, styles["metric"])],
            [paragraph(label, styles["metric_label"])],
            [paragraph(note, styles["small"])],
        ],
        colWidths=[63 * mm],
        rowHeights=[14 * mm, 8 * mm, 10 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def flow_stage(number: str, title: str, detail: str, styles: dict[str, ParagraphStyle]) -> Table:
    table = Table(
        [
            [paragraph(number, styles["eyebrow"])],
            [paragraph(title, styles["h2"])],
            [paragraph(detail, styles["small"])],
        ],
        colWidths=[43 * mm],
        rowHeights=[8 * mm, 11 * mm, 22 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def evidence_table(rows: list[list[str]], widths: list[float], styles: dict[str, ParagraphStyle]) -> Table:
    rendered = []
    for index, row in enumerate(rows):
        style = styles["cell_bold"] if index == 0 else styles["cell"]
        rendered.append([paragraph(value, style) for value in row])
    table = Table(rendered, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SURFACE_2),
                ("BACKGROUND", (0, 1), (-1, -1), SURFACE),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def page_canvas(canvas, doc) -> None:
    canvas.saveState()
    width, height = PAGE
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setStrokeColor(LINE)
    canvas.line(15 * mm, 12 * mm, width - 15 * mm, 12 * mm)
    canvas.setFillColor(FAINT)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(15 * mm, 7 * mm, "EffiPed Identity Review - technical report")
    canvas.drawRightString(width - 15 * mm, 7 * mm, f"{doc.page}")
    canvas.restoreState()


def build_report(output: Path) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    project = data["project"]
    bench = data["system_benchmarks"]
    demo = data["demo_case"]
    extensions = data["research_extensions"]
    styles = build_styles()

    output.parent.mkdir(parents=True, exist_ok=True)
    frame = Frame(15 * mm, 16 * mm, PAGE[0] - 30 * mm, PAGE[1] - 29 * mm, id="main")
    document = BaseDocTemplate(
        str(output),
        pagesize=PAGE,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=16 * mm,
        title=project["title"],
        author=project["author"],
        subject="Multi-camera pedestrian detection, tracking, re-identification, and identity review",
    )
    document.addPageTemplates([PageTemplate(id="dark", frames=[frame], onPage=page_canvas)])

    story = [
        Spacer(1, 11 * mm),
        paragraph("EFFIPED / MULTI-CAMERA VIDEO INTELLIGENCE", styles["eyebrow"]),
        paragraph(project["title"], styles["title"]),
        paragraph(
            "A compact system that carries a person from synchronized camera pixels to "
            "reviewable local tracks and ranked cross-camera candidate evidence.",
            styles["body"],
        ),
        Spacer(1, 7 * mm),
        Table(
            [[
                metric_card(
                    f"{bench['pdestre']['validation']['rank1_cross']}%",
                    "Cross-camera Rank-1",
                    "P-DESTRE validation - Protocol D",
                    styles,
                ),
                metric_card(
                    f"{bench['mot17']['mota']:.2f}",
                    "MOT17 MOTA",
                    "Val-half - Protocol A",
                    styles,
                ),
                metric_card(
                    f"{bench['footprint']['parameters_m']}M",
                    "Model parameters",
                    f"Approx. {bench['footprint']['pipeline_fps_approx']} FPS full pipeline",
                    styles,
                ),
                metric_card(
                    f"{bench['footprint']['descriptor_dim']}-D",
                    "Identity descriptor",
                    "RoIAlign and four body strips",
                    styles,
                ),
            ]],
            colWidths=[66 * mm] * 4,
        ),
        Spacer(1, 10 * mm),
        paragraph(
            f"<b>{project['author']}</b><br/>"
            "Software: Apache-2.0<br/>"
            "P-DESTRE-derived demonstration media: CC BY-NC-SA 4.0",
            styles["body"],
        ),
        PageBreak(),
        paragraph("01 / SYSTEM DESIGN", styles["eyebrow"]),
        paragraph("A shared feature hierarchy connects perception to analyst review.", styles["h1"]),
        paragraph(
            "ConvNeXt V2 produces detail-rich P2 and context-rich P3 features. A CenterNet "
            "branch decodes pedestrian boxes while a part-aware branch extracts a normalized "
            "identity descriptor from each detected region.",
            styles["body"],
        ),
        Spacer(1, 4 * mm),
        Table(
            [[
                flow_stage("01", "Camera input", "Four synchronized views, letterboxed to 1088 x 608.", styles),
                flow_stage("02", "ConvNeXt V2", "Shared P2/P3 features preserve edges and semantic context.", styles),
                flow_stage("03", "CenterNet", "Heatmap, box, offset, and IoU quality outputs.", styles),
                flow_stage("04", "Part descriptor", "RoIAlign, four strips, Coordinate Attention, 256-D fusion.", styles),
                flow_stage("05", "BoT-SORT", "Motion, overlap, and appearance maintain camera-local tracks.", styles),
                flow_stage("06", "Identity review", "Cross-camera gallery ranking exposes evidence for human review.", styles),
            ]],
            colWidths=[44 * mm] * 6,
            hAlign="LEFT",
        ),
        Spacer(1, 10 * mm),
        Table(
            [[
                [
                    paragraph("Detection objective", styles["eyebrow"]),
                    paragraph("Ldet = Lfocal + lambda_box L1 + lambda_iou LGIoU", styles["h2"]),
                    paragraph("Center heatmaps and decoded boxes share the same fused feature map.", styles["small"]),
                ],
                [
                    paragraph("Part fusion", styles["eyebrow"]),
                    paragraph("z = normalize(sum alpha_k z_k)", styles["h2"]),
                    paragraph("Coordinate Attention weights the evidence retained by each body strip.", styles["small"]),
                ],
                [
                    paragraph("Gallery similarity", styles["eyebrow"]),
                    paragraph("s(q,g) = weighted cosine(z_q, z_g)", styles["h2"]),
                    paragraph("Similarity ranks candidates; it is not a statement of identity.", styles["small"]),
                ],
            ]],
            colWidths=[88 * mm] * 3,
            style=[
                ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ],
        ),
        PageBreak(),
        paragraph("02 / MEASURED SYSTEM", styles["eyebrow"]),
        paragraph("Detection, tracking, retrieval, and efficiency use separate protocol labels.", styles["h1"]),
        evidence_table(
            [
                ["Evaluation", "Retrieval", "Detection", "Tracking"],
                [
                    "P-DESTRE fold-0 validation - Protocol D",
                    f"{bench['pdestre']['validation']['rank1_cross']}% Rank-1",
                    f"{bench['pdestre']['validation']['detection_map50']}% mAP@0.5",
                    "-",
                ],
                [
                    "P-DESTRE fold-0 test - Protocol E",
                    f"{bench['pdestre']['test']['rank1_cross']}% Rank-1",
                    f"{bench['pdestre']['test']['detection_map50']}% mAP@0.5",
                    "-",
                ],
                [
                    "MOT17 val-half - Protocol A",
                    "-",
                    "-",
                    f"{bench['mot17']['mota']:.2f} MOTA / {bench['mot17']['idf1']:.2f} IDF1 / {bench['mot17']['hota']:.2f} HOTA",
                ],
            ],
            [83 * mm, 50 * mm, 58 * mm, 75 * mm],
            styles,
        ),
        Spacer(1, 9 * mm),
        evidence_table(
            [
                ["System profile", "Parameters", "Input", "Descriptor", "Throughput condition"],
                [
                    bench["model"],
                    f"{bench['footprint']['parameters_m']}M",
                    bench["footprint"]["input_resolution"],
                    f"{bench['footprint']['descriptor_dim']}-D",
                    f"Approx. {bench['footprint']['pipeline_fps_approx']} FPS - full pipeline - {bench['footprint']['device']}",
                ],
            ],
            [52 * mm, 32 * mm, 36 * mm, 39 * mm, 107 * mm],
            styles,
        ),
        Spacer(1, 9 * mm),
        paragraph("Interpretation", styles["h2"]),
        paragraph(
            "These values come from the checked-in aggregate evidence fixture. The hosted "
            "replay is an application demonstration and is not used to compute the benchmark "
            "cards. Performance is sensitive to occlusion, localization quality, pose, "
            "illumination, camera topology, time gaps, and domain shift.",
            styles["body"],
        ),
        PageBreak(),
        paragraph("03 / IDENTITY REVIEW APPLICATION", styles["eyebrow"]),
        paragraph("The browser replay mirrors the original PedestrianTracker workflow.", styles["h1"]),
        evidence_table(
            [
                ["Demo evidence", "Value", "Role in the interface"],
                ["Synchronized views", str(demo["session_diagnostic"]["cameras"]), "Switchable four-camera playback"],
                ["Processed frames", str(demo["session_diagnostic"]["frames"]), "Frame stepping and seek timeline"],
                ["Camera-local tracks", str(demo["session_diagnostic"]["local_tracks"]), "Detector and tracker annotations"],
                ["Cross-camera IDs", str(demo["session_diagnostic"]["cross_camera_ids"]), "Ranked gallery evidence"],
                ["Indexed query examples", str(len(demo["subjects"])), "Query selection and candidate inspection"],
            ],
            [75 * mm, 35 * mm, 156 * mm],
            styles,
        ),
        Spacer(1, 9 * mm),
        Table(
            [[
                [
                    paragraph("Hosted demo", styles["eyebrow"]),
                    paragraph("Precomputed and Vercel-safe", styles["h2"]),
                    paragraph(
                        "The site serves two optimized archived replays, query crops, and gallery "
                        "candidates. The video already contains the tracker-rendered boxes; the "
                        "browser does not fabricate or reposition detections.",
                        styles["body"],
                    ),
                ],
                [
                    paragraph("Local mode", styles["eyebrow"]),
                    paragraph("FastAPI and CUDA inference", styles["h2"]),
                    paragraph(
                        "Uploads, job progress, search-by-example, asset retrieval, and explicit "
                        "cleanup are available when an authorized checkpoint is mounted locally.",
                        styles["body"],
                    ),
                ],
                [
                    paragraph("Related research", styles["eyebrow"]),
                    paragraph("PartJDE and BoxJDE", styles["h2"]),
                    paragraph(
                        f"Matched PartJDE readout gain: +{extensions['partjde']['matched_part_readout_gain_pp']} pp Rank-1. "
                        f"BoxJDE source-level gain: +{extensions['boxjde']['source_detected_rank1_gain_pp']}/"
                        f"+{extensions['boxjde']['source_detected_map_gain_pp']} pp Rank-1/mAP.",
                        styles["body"],
                    ),
                ],
            ]],
            colWidths=[88 * mm] * 3,
            style=[
                ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ],
        ),
        Spacer(1, 8 * mm),
        paragraph("Responsible-use boundary", styles["h2"]),
        paragraph(
            "The system is designed for user-authorized, human-in-the-loop review. Candidate "
            "similarity is not identity proof. Public checkpoints remain withheld until all "
            "training-data redistribution terms are confirmed.",
            styles["body"],
        ),
        paragraph(
            "References: P-DESTRE - arxiv.org/abs/2004.02782 | "
            "Media license - creativecommons.org/licenses/by-nc-sa/4.0/ | "
            "Source - github.com/aswanth-07/effiped-multi-camera-tracking",
            styles["small"],
        ),
    ]

    document.build(story)


def main() -> None:
    build_report(REPORT_OUTPUT)
    DOCS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT.write_bytes(REPORT_OUTPUT.read_bytes())
    print(f"Wrote {REPORT_OUTPUT}")
    print(f"Wrote {DOCS_OUTPUT}")


if __name__ == "__main__":
    main()
