"""Generate the standalone EffiPed technical report.

Numbers come from research/results/summary.json, loss weights and training settings
from configs/system/effiped.yaml, and the figure from docs/architecture. Nothing in
the report is typed in by hand that one of those files already records.

    pip install -e ".[report]"
    python research/report/generate_report.py

writes research/report/effiped-technical-report.pdf and a copy under docs/report/.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "research" / "results" / "summary.json"
CONFIG = ROOT / "configs" / "system" / "effiped.yaml"
DIAGRAM = ROOT / "docs" / "architecture" / "effiped-architecture.png"
REPORT_OUTPUT = ROOT / "research" / "report" / "effiped-technical-report.pdf"
DOCS_OUTPUT = ROOT / "docs" / "report" / "effiped-technical-report.pdf"

PAGE = landscape(A4)

# The console's palette, so the report, the site and the diagram read as one thing.
GROUND = colors.HexColor("#000000")
SURFACE = colors.HexColor("#0A0D12")
SURFACE_2 = colors.HexColor("#11161D")
LINE = colors.HexColor("#30363D")
TEXT = colors.HexColor("#E6EDF3")
MUTED = colors.HexColor("#8B949E")
FAINT = colors.HexColor("#7D8590")
ACCENT = colors.HexColor("#59A5FC")
ACCENT_LIGHT = colors.HexColor("#79C0FF")


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()

    def style(name: str, parent: str, **overrides) -> ParagraphStyle:
        return ParagraphStyle(name, parent=sample[parent], **overrides)

    return {
        "eyebrow": style("Eyebrow", "Normal", fontName="Courier-Bold", fontSize=7.5, leading=10,
                         textColor=ACCENT, spaceAfter=5),
        "title": style("Title", "Title", fontName="Helvetica-Bold", fontSize=27, leading=31,
                       textColor=TEXT, alignment=TA_LEFT, spaceAfter=10),
        "h1": style("H1", "Heading1", fontName="Helvetica-Bold", fontSize=18, leading=22,
                    textColor=TEXT, spaceAfter=10),
        "h2": style("H2", "Heading2", fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                    textColor=TEXT, spaceAfter=5),
        "body": style("Body", "BodyText", fontName="Helvetica", fontSize=8.5, leading=12.5,
                      textColor=MUTED, spaceAfter=6),
        "small": style("Small", "BodyText", fontName="Helvetica", fontSize=7, leading=9.5,
                       textColor=FAINT),
        "formula": style("Formula", "Normal", fontName="Courier", fontSize=9, leading=12,
                         textColor=TEXT, spaceAfter=4),
        "metric": style("Metric", "Normal", fontName="Helvetica-Bold", fontSize=21, leading=24,
                        textColor=ACCENT_LIGHT, alignment=TA_CENTER),
        "metric_label": style("MetricLabel", "Normal", fontName="Helvetica-Bold", fontSize=7.5,
                              leading=10, textColor=TEXT, alignment=TA_CENTER),
        "cell": style("Cell", "Normal", fontName="Helvetica", fontSize=7.4, leading=9.8,
                      textColor=MUTED),
        "cell_bold": style("CellBold", "Normal", fontName="Helvetica-Bold", fontSize=7.4,
                           leading=9.8, textColor=TEXT),
        "cell_num": style("CellNum", "Normal", fontName="Courier-Bold", fontSize=7.6, leading=9.8,
                          textColor=ACCENT_LIGHT),
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


def grid(rows: list[list[str]], widths: list[float], styles: dict[str, ParagraphStyle],
         numeric_last: bool = False) -> Table:
    """A table with a header row; the first column is set in bold."""
    rendered = []
    for index, row in enumerate(rows):
        cells = []
        for column, value in enumerate(row):
            if index == 0:
                cell_style = styles["cell_bold"]
            elif numeric_last and column == len(row) - 1:
                cell_style = styles["cell_num"]
            elif column == 0:
                cell_style = styles["cell_bold"]
            else:
                cell_style = styles["cell"]
            cells.append(paragraph(value, cell_style))
        rendered.append(cells)
    table = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SURFACE_2),
                ("BACKGROUND", (0, 1), (-1, -1), SURFACE),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def panels(columns: list[list], widths: list[float]) -> Table:
    return Table(
        [columns],
        colWidths=widths,
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
    )


def page_canvas(canvas, doc) -> None:
    canvas.saveState()
    width, height = PAGE
    canvas.setFillColor(GROUND)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setStrokeColor(LINE)
    canvas.line(15 * mm, 12 * mm, width - 15 * mm, 12 * mm)
    canvas.setFillColor(FAINT)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(15 * mm, 7 * mm, "EffiPed - technical report")
    canvas.drawRightString(width - 15 * mm, 7 * mm, f"{doc.page}")
    canvas.restoreState()


def loss_rows(loss: dict, model: dict) -> tuple[list[list[str]], list[list[str]]]:
    parts = model["num_parts_v"]
    detection = [
        ["Term", "Loss", "Weight"],
        ["Heatmap", "Gaussian focal loss (alpha 2, beta 4); low-visibility positives weighted up to 1.5x",
         f"{loss['cls_weight']}"],
        ["Box size", "GIoU on left, top, right, bottom distances at every pixel inside the centre Gaussian",
         f"{loss['reg_weight']}"],
        ["Centre offset", "L1 on the sub-pixel offset", f"{loss['ctr_weight']}"],
        ["Repulsion", "Smooth L1 on overlap with a neighbour beyond their true overlap", f"{loss['rep_weight']}"],
        ["Box quality", "Binary cross-entropy between predicted and actual IoU", f"{loss['iou_weight']}"],
    ]
    identity = [
        ["Term", "Loss", "Weight"],
        ["Identity", f"ArcFace (s {loss['arcface_s']:g}, m {loss['arcface_m']}) after BNNeck; label smoothing "
         f"{loss['label_smoothing']}, dropout {loss['reid_dropout']}, visibility-weighted", f"{loss['id_weight']}"],
        ["Strip identity", f"ArcFace on each of the {parts} body strips", f"{loss['part_loss_weight']}"],
        ["Metric", f"Batch-hard triplet (margin {loss['triplet_margin']}) before BNNeck, mined from a "
         f"{loss['xbm_memory_size']:,}-entry cross-batch memory", f"{loss['triplet_weight']}"],
        ["Strip metric", "Batch-hard triplet on each strip, within the batch", f"{loss['part_triplet_weight']}"],
        ["Strip diversity", "Penalises similarity between one person's strip descriptors",
         f"{loss['diversity_loss_weight']}"],
    ]
    return detection, identity


def build_report(output: Path) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    project = data["project"]
    bench = data["system_benchmarks"]
    session = data["demo_session"]
    footprint = bench["footprint"]
    model, loss, training, data_cfg, tracker = (
        config["model"], config["loss"], config["training"], config["data"], config["tracker"]
    )
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

    detection_losses, identity_losses = loss_rows(loss, model)
    diagram_height = 172 * mm
    diagram = Image(str(DIAGRAM), width=diagram_height * 780 / 1300, height=diagram_height)

    story = [
        # ---- cover ----------------------------------------------------------
        Spacer(1, 11 * mm),
        paragraph("EFFIPED / MULTI-CAMERA PEDESTRIAN DETECTION, TRACKING AND RE-IDENTIFICATION", styles["eyebrow"]),
        paragraph(project["title"], styles["title"]),
        paragraph(
            "One compact network finds people in each camera and describes how they look; a "
            "tracker follows them within each camera, and search ranks people in the other "
            "cameras for a person to review.",
            styles["body"],
        ),
        Spacer(1, 7 * mm),
        Table(
            [[
                metric_card(f"{bench['pdestre']['validation']['rank1_cross']}%", "Cross-camera Rank-1",
                            "P-DESTRE validation split", styles),
                metric_card(f"{bench['mot17']['mota']:.2f}", "MOT17 MOTA", "MOT17 validation half", styles),
                metric_card(f"{footprint['parameters_m']}M", "Model parameters",
                            f"Approx. {footprint['pipeline_fps_approx']} FPS full pipeline", styles),
                metric_card(f"{footprint['descriptor_dim']}-D", "Identity descriptor",
                            "RoIAlign and four body strips", styles),
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
        # ---- architecture ---------------------------------------------------
        Table(
            [[
                diagram,
                [
                    paragraph("01 / ARCHITECTURE", styles["eyebrow"]),
                    paragraph("One network for detection and identity.", styles["h1"]),
                    paragraph(
                        "A frame letterboxed to 1088 x 608 passes once through the stem and first "
                        "two stages of ConvNeXt V2 Tiny. A two-scale neck fuses stride-4 and stride-8 "
                        "features into one 256-channel map at stride 4, read by a detection head "
                        "and an identity head. The figure reads from the bottom up.",
                        styles["body"],
                    ),
                    Spacer(1, 2 * mm),
                    grid(
                        [
                            ["Part", "What it does", "Output"],
                            ["Backbone", "ConvNeXt V2 Tiny stem and stages 1-2. Block: 7x7 depthwise conv, "
                             "LayerNorm, 1x1 expand x4, GELU, GRN, 1x1.", "P2 96 x 152 x 272, P3 192 x 76 x 136"],
                            ["Neck", f"P3: 1x1 conv, GroupNorm, SiLU, {model['p3_refiner_depth']} depthwise-separable "
                             "convs, bilinear 2x upsample. P2: 3x3 conv. Learned weighted sum, residual refine.",
                             f"{model['fpn_out_channels']} x 152 x 272"],
                            ["Detection head", "3x3 conv, then 3x3 DCNv2; per output a 3x3 and a 1x1 conv: centre "
                             "heatmap, box distances, offset, IoU.", "4 maps at stride 4"],
                            ["Identity head", f"Own conv branch to a 256-channel map; RoIAlign "
                             f"{model['roi_output_size'][0]} x {model['roi_output_size'][1]}, Coordinate Attention, "
                             f"{model['num_parts_v']} strips with softmax attention, BNNeck, L2.",
                             f"{model['embedding_dim']}-D descriptor"],
                        ],
                        [27 * mm, 92 * mm, 36 * mm],
                        styles,
                    ),
                    Spacer(1, 4 * mm),
                    paragraph(
                        f"{footprint['parameters_m']}M parameters in total. GroupNorm is used throughout "
                        "the neck and heads, so training works at two images per batch.",
                        styles["small"],
                    ),
                ],
            ]],
            colWidths=[108 * mm, 159 * mm],
            style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (1, 0), (1, 0), 8 * mm)],
        ),
        PageBreak(),
        # ---- training -------------------------------------------------------
        paragraph("02 / TRAINING", styles["eyebrow"]),
        paragraph("Detection and identity balanced by learned task uncertainty.", styles["h1"]),
        paragraph("L = 1/2 [ exp(-s_det) L_det + s_det + exp(-s_id) L_id + s_id ]", styles["formula"]),
        paragraph(
            "s_det and s_id are learned with the network, starting at -1.85 and -1.05. Box targets "
            "cover the centre Gaussian; identity targets sit on the centre pixel.",
            styles["small"],
        ),
        Spacer(1, 4 * mm),
        Table(
            [[
                [paragraph("Detection, L_det", styles["h2"]), grid(detection_losses, [24 * mm, 87 * mm, 15 * mm],
                                                                    styles, numeric_last=True)],
                [paragraph("Identity, L_id", styles["h2"]), grid(identity_losses, [24 * mm, 93 * mm, 15 * mm],
                                                                  styles, numeric_last=True)],
            ]],
            colWidths=[131 * mm, 136 * mm],
            style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)],
        ),
        Spacer(1, 6 * mm),
        grid(
            [
                ["Setting", "Value"],
                ["Data", "P-DESTRE, MOT17, MOT20 and SOMPT22 sequences; CrowdHuman mixed in at "
                 f"{round(data_cfg['additional_sources'][0]['ratio'] * 100)}%"],
                ["Sampling", f"Identity-aware batches, +{round(data_cfg['cross_camera_ratio'] * 100)}% cross-camera "
                 f"batches, 1 frame in {data_cfg['frame_skip']}, boxes at least "
                 f"{round(data_cfg['min_visibility'] * 100)}% visible"],
                ["Augmentation", "Colour jitter, random erasing, mosaic and mixup, blur, noise, grayscale, "
                 "JPEG and resolution degradation, gamma"],
                ["Optimiser", f"AdamW, learning rate {training['lr']:g}, weight decay 0.01, backbone at "
                 f"{training['backbone_lr_mult']}x"],
                ["Schedule", f"{training['epochs']} epochs, linear warm-up over {training['warmup_batches']:,} "
                 "batches, cosine warm restarts (T0 5, x2)"],
                ["Batch", f"{data_cfg['batch_size']} images x {training['gradient_accumulation_steps']} "
                 f"accumulation steps, gradient clip {training['grad_clip']:g}"],
            ],
            [30 * mm, 237 * mm],
            styles,
        ),
        PageBreak(),
        # ---- inference and results -------------------------------------------
        paragraph("03 / INFERENCE AND MEASURED RESULTS", styles["eyebrow"]),
        paragraph("Peaks become boxes, boxes become descriptors, descriptors become candidates.", styles["h1"]),
        grid(
            [
                ["Step", "What happens"],
                ["Decode", "3x3 max-pool peaks, top 300, score = sqrt(heatmap x predicted IoU), "
                 "Gaussian Soft-NMS (sigma 0.35)"],
                ["Track", f"Kalman prediction; detections scoring at least {tracker['det_thresh']} matched first on "
                 f"{tracker['embedding_weight']} appearance + {1 - tracker['embedding_weight']:.1f} motion, then "
                 f"down to {tracker['low_thresh']} on box overlap; lost tracks kept {tracker['track_buffer']} frames. "
                 f"Appearance: {1 - tracker['part_aware_alpha']:.1f} fused + {tracker['part_aware_alpha']} "
                 "strip cosine, strips weighted by mutual visibility"],
                ["Search", "Mean descriptor per person; people in the other clips ranked by cosine similarity "
                 "for human review"],
            ],
            [30 * mm, 237 * mm],
            styles,
        ),
        Spacer(1, 7 * mm),
        grid(
            [
                ["Evaluation", "Retrieval", "Detection", "Tracking"],
                [bench["pdestre"]["validation"]["protocol"],
                 f"{bench['pdestre']['validation']['rank1_cross']}% Rank-1",
                 f"{bench['pdestre']['validation']['detection_map50']}% mAP@0.5", "-"],
                [bench["pdestre"]["test"]["protocol"],
                 f"{bench['pdestre']['test']['rank1_cross']}% Rank-1",
                 f"{bench['pdestre']['test']['detection_map50']}% mAP@0.5", "-"],
                [bench["mot17"]["protocol"], "-", "-",
                 f"{bench['mot17']['mota']:.2f} MOTA / {bench['mot17']['idf1']:.2f} IDF1 / "
                 f"{bench['mot17']['hota']:.2f} HOTA"],
            ],
            [101 * mm, 40 * mm, 46 * mm, 80 * mm],
            styles,
        ),
        Spacer(1, 4 * mm),
        paragraph(
            f"{bench['model']}: {footprint['parameters_m']}M parameters, {footprint['input_resolution']} input, "
            f"{footprint['descriptor_dim']}-D descriptor, approximately {footprint['pipeline_fps_approx']} FPS for "
            f"the full pipeline on an {footprint['device']}. Single archived measurements: no repeated runs, "
            "intervals or significance tests. Performance depends on occlusion, box quality, pose, "
            "lighting, camera layout, time gaps and domain shift.",
            styles["body"],
        ),
        PageBreak(),
        # ---- application ----------------------------------------------------
        paragraph("04 / IDENTITY REVIEW CONSOLE", styles["eyebrow"]),
        paragraph("The hosted console replays saved results; a person makes every decision.", styles["h1"]),
        grid(
            [
                ["Archived session diagnostic", "Value"],
                ["Synchronised cameras", str(session["cameras"])],
                ["Frames per clip", str(session["frames"])],
                ["Camera-local tracks", str(session["local_tracks"])],
                ["Cross-camera groups", str(session["cross_camera_ids"])],
                ["Pairwise association precision", f"{session['pairwise_association_precision']:.2f}"],
            ],
            [90 * mm, 40 * mm],
            styles,
        ),
        Spacer(1, 2 * mm),
        paragraph(session["description"], styles["small"]),
        Spacer(1, 7 * mm),
        panels(
            [
                [
                    paragraph("Hosted demo", styles["eyebrow"]),
                    paragraph("Precomputed, no model in the browser", styles["h2"]),
                    paragraph(
                        "Four clips, every saved detection and track, and the ranked candidates. "
                        "Thresholds filter those records; they cannot create a detection the model "
                        "never made.",
                        styles["body"],
                    ),
                ],
                [
                    paragraph("Local service", styles["eyebrow"]),
                    paragraph("FastAPI and CUDA inference", styles["h2"]),
                    paragraph(
                        "Uploads, job progress, search by example, asset retrieval and explicit "
                        "cleanup, when an authorised checkpoint is available locally.",
                        styles["body"],
                    ),
                ],
                [
                    paragraph("Responsible use", styles["eyebrow"]),
                    paragraph("Candidates, not identifications", styles["h2"]),
                    paragraph(
                        "Similarity ranks candidates for review and is not proof of identity. Public "
                        "weights are withheld until every training-data source permits redistribution.",
                        styles["body"],
                    ),
                ],
            ],
            [89 * mm] * 3,
        ),
        Spacer(1, 7 * mm),
        paragraph(
            "References: P-DESTRE - arxiv.org/abs/2004.02782 | "
            "Media licence - creativecommons.org/licenses/by-nc-sa/4.0/ | "
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
