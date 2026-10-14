"""Draw the EffiPed architecture diagram as a standalone SVG.

The figure follows the convention of a paper block diagram: data flows from
the bottom to the top, related layers sit inside a grey group, every block is
coloured by what kind of operation it is, and arrows are orthogonal. It is
drawn on the console's black ground so the Architecture tab can show it as is.

Every layer and number in it comes from configs/system/effiped.yaml and
src/effiped; the console's architecture test pins the same values.

    python tools/architecture/build_diagram.py

writes docs/architecture/effiped-architecture.svg. Export the PNG with
tools/architecture/export_png.mjs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "architecture" / "effiped-architecture.svg"

WIDTH, HEIGHT = 780, 1300

SANS = "'Segoe UI', 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif"
MONO = "Consolas, 'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace"

# The console's palette: a black ground, near-white text, and each block kind
# given a dark tint of one hue with a brighter edge of the same hue.
GROUND = "#000000"
GROUP_FILL, GROUP_EDGE = "#0b0f14", "#262c34"
INK, INK_SOFT, INK_MUTED = "#e6edf3", "#8b949e", "#7d8590"
WIRE = "#9aa4ae"

KINDS = {
    "input": ("#2a1421", "#d0659c", "Input"),
    "conv": ("#0f1f33", "#4a8fdc", "Convolution"),
    "project": ("#211a33", "#a27ddf", "Projection"),
    "attend": ("#2a1f0c", "#c58b22", "Attention, DCN"),
    "norm": ("#25230f", "#b9aa45", "Normalisation"),
    "output": ("#0f2618", "#47b562", "Output"),
    "fixed": ("#161b22", "#4b535d", "Fixed operation"),
}


@dataclass
class Svg:
    parts: list[str]

    def add(self, markup: str) -> None:
        self.parts.append(markup)


def text(
    svg: Svg,
    x: float,
    y: float,
    content: str,
    *,
    size: float = 12.5,
    fill: str = INK,
    weight: int = 500,
    anchor: str = "middle",
    family: str = SANS,
    spacing: float = 0,
    sub: str | None = None,
) -> None:
    """One line of text; ``sub`` is set as a subscript after ``content``."""
    letter = f' letter-spacing="{spacing}"' if spacing else ""
    tail = (
        f'<tspan baseline-shift="sub" font-size="{size * 0.72:.1f}">{escape(sub)}</tspan>' if sub else ""
    )
    svg.add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{letter}>{escape(content)}{tail}</text>'
    )


def group(svg: Svg, x: float, y: float, w: float, h: float, caption: str) -> None:
    svg.add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{GROUP_FILL}" '
        f'stroke="{GROUP_EDGE}" stroke-width="1.2"/>'
    )
    text(svg, x + 14, y + 19, caption.upper(), size=9.5, fill=INK_MUTED, weight=400, anchor="start",
         family=MONO, spacing=1.1)


def block(svg: Svg, cx: float, top: float, w: float, label: str, kind: str, h: float = 30, size: float = 12.5) -> None:
    fill, edge, _ = KINDS[kind]
    svg.add(
        f'<rect x="{cx - w / 2:.1f}" y="{top}" width="{w}" height="{h}" rx="7" fill="{fill}" '
        f'stroke="{edge}" stroke-width="1.3"/>'
    )
    text(svg, cx, top + h / 2 + size * 0.36, label, size=size)


def wire(svg: Svg, *points: tuple[float, float], head: bool = True, dashed: bool = False, radius: float = 7) -> None:
    """An orthogonal connector through ``points`` with softened corners."""
    d = [f"M{points[0][0]:.1f},{points[0][1]:.1f}"]
    for index in range(1, len(points) - 1):
        (x0, y0), (x1, y1), (x2, y2) = points[index - 1], points[index], points[index + 1]
        r = min(radius, abs(x1 - x0 + y1 - y0) / 2, abs(x2 - x1 + y2 - y1) / 2)
        ux, uy = _unit(x0, y0, x1, y1)
        vx, vy = _unit(x1, y1, x2, y2)
        d.append(f"L{x1 - ux * r:.1f},{y1 - uy * r:.1f}")
        d.append(f"Q{x1:.1f},{y1:.1f} {x1 + vx * r:.1f},{y1 + vy * r:.1f}")
    d.append(f"L{points[-1][0]:.1f},{points[-1][1]:.1f}")
    marker = ' marker-end="url(#head)"' if head else ""
    dash = ' stroke-dasharray="3 3"' if dashed else ""
    svg.add(
        f'<path d="{" ".join(d)}" fill="none" stroke="{WIRE}" stroke-width="1.5" '
        f'stroke-linejoin="round"{dash}{marker}/>'
    )


def _unit(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
    dx, dy = x1 - x0, y1 - y0
    length = max(abs(dx), abs(dy)) or 1
    return dx / length, dy / length


def add_node(svg: Svg, cx: float, cy: float, r: float = 10) -> None:
    svg.add(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{GROUND}" stroke="{WIRE}" stroke-width="1.5"/>'
        f'<path d="M{cx - r * 0.55},{cy} H{cx + r * 0.55} M{cx},{cy - r * 0.55} V{cy + r * 0.55}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )


def junction(svg: Svg, cx: float, cy: float) -> None:
    svg.add(f'<circle cx="{cx}" cy="{cy}" r="3" fill="{WIRE}"/>')


def loss_tag(svg: Svg, cx: float, cy: float, name: str) -> None:
    """A dashed pill naming the training loss that supervises what it touches."""
    svg.add(
        f'<rect x="{cx - 22}" y="{cy - 12}" width="44" height="24" rx="12" fill="{GROUND}" '
        f'stroke="{INK_MUTED}" stroke-width="1.1" stroke-dasharray="3 2.5"/>'
        f'<text x="{cx}" y="{cy + 3.5}" font-family="{MONO}" font-size="12" font-weight="600" '
        f'fill="{INK_SOFT}" text-anchor="middle">L<tspan dy="3" font-size="9">{escape(name)}</tspan></text>'
    )


def band(svg: Svg, top: float, bottom: float, label: str) -> None:
    """A section label down the left edge, read bottom to top."""
    svg.add(f'<line x1="30" y1="{top}" x2="30" y2="{bottom}" stroke="{GROUP_EDGE}" stroke-width="1.2"/>')
    middle = (top + bottom) / 2
    svg.add(
        f'<text transform="translate(19 {middle}) rotate(-90)" font-family="{MONO}" font-size="9.5" '
        f'fill="{INK_MUTED}" text-anchor="middle" letter-spacing="1.4">{escape(label.upper())}</text>'
    )


def draw() -> str:
    svg = Svg([])
    xl, xr, xc = 184, 584, 384  # detection column, identity column, shared trunk
    bw = 210  # block width in the head and association groups

    svg.add(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{GROUND}"/>')

    # ---- association (inference) ------------------------------------------
    band(svg, 66, 232, "Association")
    group(svg, 44, 66, 280, 166, "Per-camera tracker")
    block(svg, xl, 96, bw, "Match ≥ 0.4, then ≥ 0.1", "conv")
    block(svg, xl, 142, bw, "0.8 appearance + 0.2 motion", "attend")
    block(svg, xl, 188, bw, "Kalman predict · IoU", "fixed")
    wire(svg, (xl, 188), (xl, 172))
    wire(svg, (xl, 142), (xl, 126))

    group(svg, 444, 66, 280, 166, "Cross-camera search")
    block(svg, xr, 96, bw, "Top-k ranking", "output")
    block(svg, xr, 142, bw, "Cosine vs other cameras", "project")
    block(svg, xr, 188, bw, "Mean descriptor per track", "norm")
    wire(svg, (xr, 188), (xr, 172))
    wire(svg, (xr, 142), (xr, 126))

    # outputs
    wire(svg, (xl, 96), (xl, 56))
    wire(svg, (xr, 96), (xr, 56))
    text(svg, xl, 30, "Tracks", size=13.5, weight=600)
    text(svg, xl, 46, "per camera", size=11, fill=INK_SOFT, weight=400)
    text(svg, xr, 30, "Ranked candidates", size=13.5, weight=600)
    text(svg, xr, 46, "for human review", size=11, fill=INK_SOFT, weight=400)

    # tracks feed the per-track mean descriptor
    wire(svg, (xl + bw / 2, 111), (xc, 111), (xc, 203), (xr - bw / 2, 203))
    text(svg, xc + 6, 160, "tracks", size=10.5, fill=INK_SOFT, weight=400, anchor="start", family=MONO)

    # ---- heads --------------------------------------------------------------
    band(svg, 296, 602, "Heads")

    group(svg, 444, 296, 280, 306, "Identity head")
    rows = [
        (326, "BNNeck · L2 norm", "norm"),
        (372, "Strip attention, weighted sum", "attend"),
        (418, "4 horizontal strips · pool", "fixed"),
        (464, "Coordinate Attention", "attend"),
        (510, "RoIAlign 32 × 8", "project"),
        (556, "Conv 3×3 ×4 · 1×1 → 256", "conv"),
    ]
    for top, label, kind in rows:
        block(svg, xr, top, bw, label, kind)
    for (upper, _, _), (lower, _, _) in zip(rows, rows[1:]):
        wire(svg, (xr, lower), (xr, upper + 30))
    loss_tag(svg, 752, 341, "id")
    wire(svg, (xr + bw / 2, 341), (730, 341), head=False, dashed=True)

    group(svg, 44, 296, 280, 306, "Detection head")
    block(svg, xl, 556, bw, "Conv 3×3 · GN", "conv")
    block(svg, xl, 510, bw, "DCNv2 3×3 · GN", "attend")
    outputs = ["Heatmap", "Box", "Offset", "IoU"]
    centres = [xl - 90, xl - 30, xl + 30, xl + 90]
    for centre, label in zip(centres, outputs):
        block(svg, centre, 450, 56, "3×3 · 1×1", "conv", size=10.5)
        block(svg, centre, 392, 56, label, "output", size=11)
        wire(svg, (centre, 496), (centre, 480))
        wire(svg, (centre, 450), (centre, 422))
        wire(svg, (centre, 392), (centre, 374), head=False)
    wire(svg, (xl, 556), (xl, 540))
    wire(svg, (xl, 510), (xl, 496), head=False)
    for y in (496, 374):
        svg.add(f'<path d="M{centres[0]},{y} H{centres[-1]}" stroke="{WIRE}" stroke-width="1.5" fill="none"/>')
    wire(svg, (xl, 374), (xl, 356))
    block(svg, xl, 326, bw, "Peaks · top-300 · Soft-NMS", "fixed")
    loss_tag(svg, 362, 407, "det")
    wire(svg, (centres[-1] + 28, 407), (340, 407), head=False, dashed=True)

    # boxes select the regions the identity head pools
    wire(svg, (xl + bw / 2, 341), (410, 341), (410, 525), (xr - bw / 2, 525))
    text(svg, 416, 474, "boxes", size=10.5, fill=INK_SOFT, weight=400, anchor="start", family=MONO)

    # detections go up to the tracker
    wire(svg, (xl, 326), (xl, 218))
    text(svg, xl - 8, 262, "detections", size=10.5, fill=INK_SOFT, weight=400, anchor="end", family=MONO)

    # one descriptor per detection: to the tracker's appearance cost and to search
    wire(svg, (xr, 326), (xr, 218))
    junction(svg, xr, 262)
    wire(svg, (xr, 262), (306, 262), (306, 157), (xl + bw / 2, 157))
    text(svg, xr + 8, 286, "256-D descriptor", size=10.5, fill=INK_SOFT, weight=400, anchor="start", family=MONO)

    # ---- shared trunk -------------------------------------------------------
    band(svg, 668, 1198, "Shared trunk")

    # split of the fused map into the two heads
    junction(svg, xc, 628)
    wire(svg, (xc, 628), (xl, 628), (xl, 586))
    wire(svg, (xc, 628), (xr, 628), (xr, 586))
    wire(svg, (xc, 694), (xc, 628), head=False)
    text(svg, xc + 8, 652, "256 × 152 × 272", size=10.5, fill=INK_SOFT, weight=400, anchor="start", family=MONO)

    group(svg, 134, 668, 500, 290, "Neck · two-scale fusion")
    add_node(svg, xc, 704, r=9)
    block(svg, xc, 722, 180, "DW-sep conv ×2", "conv")
    wire(svg, (xc, 722), (xc, 713))
    wire(svg, (xc, 779), (xc, 752))
    # residual around the refine block
    junction(svg, xc, 768)
    wire(svg, (xc, 768), (494, 768), (494, 704), (xc + 9, 704))
    add_node(svg, xc, 790, r=11)
    text(svg, xc + 18, 794, "learned weighted sum", size=10.5, fill=INK_SOFT, weight=400, anchor="start", family=MONO)
    block(svg, xc, 820, 180, "Bilinear upsample ×2 · GN", "fixed")
    block(svg, xc, 866, 180, "DW-sep conv ×3", "conv")
    block(svg, xc, 912, 180, "1×1 conv · GN · SiLU", "project")
    wire(svg, (xc, 912), (xc, 896))
    wire(svg, (xc, 866), (xc, 850))
    wire(svg, (xc, 820), (xc, 801))
    block(svg, 214, 820, 110, "3×3 conv · GN", "project")
    wire(svg, (214, 820), (214, 790), (xc - 11, 790))

    group(svg, 234, 984, 300, 214, "Backbone")
    text(svg, 548, 1086, "ConvNeXt V2 Tiny", size=13, weight=600, anchor="start")
    text(svg, 548, 1103, "stem + first two stages", size=10.5, fill=INK_SOFT, weight=400, anchor="start", family=MONO)
    text(svg, 548, 1127, "block: 7×7 depthwise conv,", size=10.5, fill=INK_MUTED, weight=400, anchor="start", family=MONO)
    text(svg, 548, 1142, "LN, 1×1 ×4, GELU, GRN, 1×1", size=10.5, fill=INK_MUTED, weight=400, anchor="start", family=MONO)
    block(svg, xc, 1014, 200, "ConvNeXt V2 block ×3", "conv")
    block(svg, xc, 1060, 170, "Downsample 2×2, /2", "fixed")
    block(svg, xc, 1106, 200, "ConvNeXt V2 block ×3", "conv")
    block(svg, xc, 1152, 200, "Stem 4×4, /4 · LN", "input")
    wire(svg, (xc, 1152), (xc, 1136))
    wire(svg, (xc, 1106), (xc, 1090))
    wire(svg, (xc, 1060), (xc, 1044))
    # P3 from stage two, P2 tapped from stage one
    wire(svg, (xc, 1014), (xc, 942))
    text(svg, xc + 8, 975, "P3 · 192 ch · /8", size=10.5, fill=INK_SOFT, weight=400, anchor="start", family=MONO)
    wire(svg, (xc - 100, 1121), (214, 1121), (214, 850))
    text(svg, 206, 1004, "P2 · 96 ch · /4", size=10.5, fill=INK_SOFT, weight=400, anchor="end", family=MONO)

    # input
    wire(svg, (xc, 1236), (xc, 1182))
    text(svg, xc, 1258, "Camera frame", size=13.5, weight=600)
    text(svg, xc, 1276, "3 × 608 × 1088, letterboxed", size=10.5, fill=INK_SOFT, weight=400, family=MONO)

    # ---- legend -------------------------------------------------------------
    for index, kind in enumerate(KINDS):
        fill, edge, name = KINDS[kind]
        x = 556 + (index % 2) * 104
        y = 1206 + (index // 2) * 21
        svg.add(f'<rect x="{x}" y="{y}" width="14" height="11" rx="3" fill="{fill}" stroke="{edge}" stroke-width="1.2"/>')
        text(svg, x + 20, y + 9.5, name, size=10.5, fill=INK_SOFT, weight=400, anchor="start")

    body = "\n  ".join(svg.parts)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">EffiPed system architecture</title>
  <desc id="desc">A camera frame passes through the stem and first two stages of ConvNeXt V2 Tiny. A two-scale neck fuses the stride-4 and stride-8 features into a 256-channel map at stride 4. A detection head predicts a centre heatmap, box, offset and IoU quality, decoded with Soft-NMS. An identity head pools each detected box with RoIAlign, applies Coordinate Attention, splits four body strips, weights them with strip attention and normalises a 256-D descriptor. A per-camera tracker matches detections with motion, IoU and appearance; cross-camera search ranks other cameras' tracks by cosine similarity for human review.</desc>
  <defs>
    <marker id="head" viewBox="0 0 10 10" refX="9.2" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0.8 L9.6,5 L0,9.2 Z" fill="{WIRE}"/>
    </marker>
  </defs>
  {body}
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(draw(), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
