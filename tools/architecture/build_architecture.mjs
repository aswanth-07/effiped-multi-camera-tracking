import pptxgen from "pptxgenjs";
import path from "node:path";

const OUT = process.argv[2] || "effiped-architecture.pptx";

/* Palette: one dominant dark slate, teal carries the signal path, amber is
   spent once on the human-review endpoint so it actually means something. */
const C = {
  bg: "0B1014",
  card: "141E25",
  cardAlt: "18242C",
  line: "273942",
  lineSoft: "1E2C34",
  ink: "EDF4F3",
  body: "B4C6C9",
  muted: "8397A0",
  teal: "22C7B8",
  tealDim: "17786F",
  amber: "E0A458",
  amberDim: "7A5A2E"
};

const F = { head: "Calibri", body: "Calibri" };

const pres = new pptxgen();
pres.defineLayout({ name: "DIAGRAM", width: 13.333, height: 7.5 });
pres.layout = "DIAGRAM";

const s = pres.addSlide();
s.background = { color: C.bg };

/* ------------------------------------------------------------------ helpers */

const txt = (text, o) =>
  s.addText(text, {
    fontFace: F.body,
    color: C.body,
    margin: 0,
    isTextBox: true,
    ...o
  });

function card(x, y, w, h, { accent = C.line, fill = C.card } = {}) {
  s.addShape(pres.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.05,
    fill: { color: fill },
    line: { color: accent, width: 1 }
  });
}

/** Column heading sitting above a card — small, muted, no rule beneath it. */
function stage(n, label, x, y, w) {
  txt(`${n}   ${label}`, {
    x,
    y,
    w,
    h: 0.2,
    fontSize: 8,
    bold: true,
    charSpacing: 1.1,
    color: C.muted
  });
}

function block(x, y, w, h, title, lines, opts = {}) {
  const accent = opts.accent || C.line;
  card(x, y, w, h, { accent, fill: opts.fill });
  txt(title, {
    x: x + 0.14,
    y: y + 0.13,
    w: w - 0.28,
    h: 0.26,
    fontSize: 11.5,
    bold: true,
    color: opts.titleColor || C.ink,
    fontFace: F.head
  });
  txt(lines.join("\n"), {
    x: x + 0.14,
    y: y + 0.45,
    w: w - 0.28,
    h: h - 0.58,
    fontSize: 8.6,
    color: C.body,
    lineSpacingMultiple: 1.24,
    valign: "top"
  });
}

function arrow(x1, y1, x2, y2, color = C.tealDim) {
  s.addShape(pres.ShapeType.line, {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1),
    line: {
      color,
      width: 1.25,
      endArrowType: "triangle",
      beginArrowType: "none"
    },
    flipH: x2 < x1,
    flipV: y2 < y1
  });
}

/** Plain connector with no head, for the fork spine. */
function rule(x1, y1, x2, y2, color = C.tealDim) {
  s.addShape(pres.ShapeType.line, {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1),
    line: { color, width: 1.25 }
  });
}

/* -------------------------------------------------------------------- title */

txt("EffiPed", {
  x: 0.55,
  y: 0.38,
  w: 4,
  h: 0.24,
  fontSize: 10,
  bold: true,
  charSpacing: 1.6,
  color: C.teal
});
txt("System architecture", {
  x: 0.55,
  y: 0.62,
  w: 8.6,
  h: 0.45,
  fontSize: 26,
  bold: true,
  color: C.ink,
  fontFace: F.head
});
txt(
  "One ConvNeXt V2 trunk feeds both detection and identity. Association runs per camera first, then across cameras.",
  { x: 0.55, y: 1.09, w: 9.6, h: 0.26, fontSize: 10.5, color: C.muted }
);

/* ------------------------------------------------------------------- layout */

const TOP = 1.9;           // top of the tall columns
const TALL_H = 2.95;       // spans both fork rows exactly
const ROW_A = { y: 1.9, h: 1.3 };    // detection path  -> ends 3.20
const ROW_B = { y: 3.4, h: 1.45 };   // identity path   -> ends 4.85

const col = {
  input: { x: 0.55, w: 1.45 },
  trunk: { x: 2.18, w: 1.75 },
  fusion: { x: 4.11, w: 1.75 },
  head: { x: 6.31, w: 2.1 },
  assoc: { x: 8.63, w: 2.1 },
  review: { x: 10.95, w: 1.83 }
};

/* ------------------------------------------------------------------- column 1 */

stage("01", "INPUT", col.input.x, TOP - 0.28, col.input.w);
block(
  col.input.x,
  TOP,
  col.input.w,
  TALL_H,
  "Camera feed",
  [
    "Four synchronised",
    "views",
    "",
    "RGB · 10 FPS",
    "Letterbox to",
    "1088 × 608",
    "",
    "Camera and time",
    "metadata retained"
  ]
);

/* ------------------------------------------------------------------- column 2 */

stage("02", "SHARED TRUNK", col.trunk.x, TOP - 0.28, col.trunk.w);
block(
  col.trunk.x,
  TOP,
  col.trunk.w,
  TALL_H,
  "ConvNeXt V2",
  [
    "C1  stride 4    edges",
    "C2  stride 8    texture",
    "C3  stride 16  semantics",
    "C4  stride 32  context",
    "",
    "7.78 M parameters",
    "",
    "A single trunk serves",
    "detection and identity —",
    "no second backbone."
  ]
);

/* ------------------------------------------------------------------- column 3 */

stage("03", "FUSION", col.fusion.x, TOP - 0.28, col.fusion.w);
block(
  col.fusion.x,
  TOP,
  col.fusion.w,
  TALL_H,
  "Adaptive P2 / P3",
  [
    "P3 depthwise refine",
    "2 × upsample",
    "Weighted P2 merge",
    "",
    "256 channels at",
    "stride 4",
    "",
    "Keeps fine edges for",
    "small, distant people",
    "and semantic context",
    "for appearance."
  ]
);

/* ---------------------------------------------------------------- the fork */

stage("04", "TASK HEADS", col.head.x, TOP - 0.28, col.head.w);
block(
  col.head.x,
  ROW_A.y,
  col.head.w,
  ROW_A.h,
  "CenterNet detection",
  ["Heatmap · LTRB box", "Centre offset · IoU quality", "Max-pool decode, no NMS"]
);
block(
  col.head.x,
  ROW_B.y,
  col.head.w,
  ROW_B.h,
  "Part-aware identity",
  [
    "RoIAlign 32 × 8",
    "Four horizontal body strips",
    "Coordinate Attention fusion",
    "BNNeck → 256-D, L2 normalised"
  ],
  { accent: C.tealDim }
);

/* --------------------------------------------------------------- column 5 */

stage("05", "ASSOCIATION", col.assoc.x, TOP - 0.28, col.assoc.w);
block(
  col.assoc.x,
  ROW_A.y,
  col.assoc.w,
  ROW_A.h,
  "BoT-SORT, per camera",
  ["Kalman motion prediction", "IoU cascade + cosine appearance", "→ camera-local tracks"]
);
block(
  col.assoc.x,
  ROW_B.y,
  col.assoc.w,
  ROW_B.h,
  "Cross-camera gallery",
  [
    "Per-strip cosine similarity",
    "Mutual-visibility weighting",
    "Top-k candidate ranking",
    "→ global identity candidates"
  ],
  { accent: C.tealDim }
);

/* --------------------------------------------------------------- column 6 */

stage("06", "REVIEW", col.review.x, TOP - 0.28, col.review.w);
block(
  col.review.x,
  TOP,
  col.review.w,
  TALL_H,
  "Analyst review",
  [
    "Boxes and scores",
    "Camera-local tracks",
    "256-D identity",
    "Ranked candidates",
    "",
    "A human confirms every",
    "match. Candidates are",
    "appearance evidence,",
    "not proof of identity."
  ],
  { accent: C.amberDim, fill: C.cardAlt, titleColor: C.amber }
);

/* ------------------------------------------------------------------- arrows */

const midTall = TOP + TALL_H / 2;
const midA = ROW_A.y + ROW_A.h / 2;
const midB = ROW_B.y + ROW_B.h / 2;

arrow(col.input.x + col.input.w, midTall, col.trunk.x, midTall);
arrow(col.trunk.x + col.trunk.w, midTall, col.fusion.x, midTall);

// fork: fusion -> both heads, drawn as an orthogonal spine
const forkX = col.fusion.x + col.fusion.w + 0.22;
rule(col.fusion.x + col.fusion.w, midTall, forkX, midTall);
rule(forkX, midA, forkX, midB);
arrow(forkX, midA, col.head.x, midA);
arrow(forkX, midB, col.head.x, midB);

// heads -> association
arrow(col.head.x + col.head.w, midA, col.assoc.x, midA);
arrow(col.head.x + col.head.w, midB, col.assoc.x, midB);

// identity descriptors also feed the per-camera tracker's appearance term
const linkX = col.head.x + col.head.w + 0.11;
rule(linkX, midB, linkX, midA, C.tealDim);

// join: both association paths -> review
const joinX = col.assoc.x + col.assoc.w + 0.22;
rule(col.assoc.x + col.assoc.w, midA, joinX, midA, C.amberDim);
rule(col.assoc.x + col.assoc.w, midB, joinX, midB, C.amberDim);
rule(joinX, midA, joinX, midB, C.amberDim);
arrow(joinX, midTall, col.review.x, midTall, C.amberDim);

// The box titles already name the two paths; an extra pair of italic labels
// here only collided with the stage headings.

/* ---------------------------------------------------------------- equations */

const EQ_Y = 5.42;
s.addShape(pres.ShapeType.line, {
  x: 0.55,
  y: EQ_Y - 0.18,
  w: 12.23,
  h: 0,
  line: { color: C.lineSoft, width: 1 }
});

// Consistent underscore notation throughout — mixing unicode subscripts with
// underscores rendered unevenly across font substitutions.
const equations = [
  ["Detection objective", "L_det  =  L_focal(Ĉ, C)  +  λ_box · L1(b̂, b)  +  λ_iou · L_GIoU(b̂, b)"],
  ["Part fusion", "z  =  normalise( Σ_k  α_k · z_k ),      α  =  softmax( g(F_roi) )"],
  ["Cross-camera similarity", "s(q, g)  =  Σ_k  v_k(q) v_k(g) · cos( z_q,k , z_g,k )   /   Σ_k  v_k(q) v_k(g)"]
];

equations.forEach(([label, formula], i) => {
  const y = EQ_Y + i * 0.44;
  txt(label, {
    x: 0.55,
    y,
    w: 2.1,
    h: 0.24,
    fontSize: 8.5,
    color: C.muted,
    align: "left"
  });
  txt(formula, {
    x: 2.75,
    y,
    w: 10.03,
    h: 0.24,
    fontSize: 10,
    color: C.ink,
    fontFace: "Cambria"
  });
});

/* ------------------------------------------------------------------- footer */

txt("EffiPed · multi-camera pedestrian detection, tracking and re-identification", {
  x: 0.55,
  y: 7.02,
  w: 8,
  h: 0.22,
  fontSize: 8.5,
  color: C.muted
});
txt("Aswanth Raj · 2026", {
  x: 9.5,
  y: 7.02,
  w: 3.28,
  h: 0.22,
  fontSize: 8.5,
  color: C.muted,
  align: "right"
});

s.addNotes(
  "EffiPed system architecture. A single ConvNeXt V2 trunk feeds an adaptive P2/P3 fusion " +
    "neck at stride 4. The fused map drives two heads: CenterNet detection and a part-aware " +
    "256-D identity descriptor. BoT-SORT associates detections into camera-local tracks using " +
    "motion, overlap and the identity descriptor; a gallery stage then ranks candidates across " +
    "cameras for human review."
);

await pres.writeFile({ fileName: OUT });
console.log("wrote", path.resolve(OUT));
