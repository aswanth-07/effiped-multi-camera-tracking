import summaryJson from "../../../../research/results/summary.json";

/**
 * What the Architecture tab prints, in one place.
 *
 * Every hyperparameter shown is kept here under the key it has in
 * configs/system/effiped.yaml, and architecture.test.ts reads that file and
 * fails if any of them drift. The reported results come straight from
 * research/results/summary.json, the same file the release validator pins.
 */
export const CONFIG = {
  "model.embedding_dim": 256,
  "model.fpn_out_channels": 256,
  "model.p3_refiner_depth": 3,
  "model.num_parts_v": 4,
  "model.reid_head_depth": 3,
  "loss.cls_weight": 1.0,
  "loss.reg_weight": 0.5,
  "loss.ctr_weight": 1.0,
  "loss.rep_weight": 0.4,
  "loss.iou_weight": 1.0,
  "loss.id_weight": 1.0,
  "loss.part_loss_weight": 0.3,
  "loss.triplet_weight": 0.3,
  "loss.part_triplet_weight": 0.09,
  "loss.diversity_loss_weight": 0.1,
  "loss.arcface_s": 16.0,
  "loss.arcface_m": 0.25,
  "loss.label_smoothing": 0.1,
  "loss.reid_dropout": 0.1,
  "loss.triplet_margin": 0.5,
  "loss.xbm_memory_size": 8192,
  "training.epochs": 15,
  "training.lr": 0.0001,
  "training.backbone_lr_mult": 0.1,
  "training.warmup_batches": 1000,
  "training.gradient_accumulation_steps": 8,
  "training.grad_clip": 10.0,
  "data.batch_size": 2,
  "data.frame_skip": 3,
  "data.min_visibility": 0.3,
  "data.cross_camera_ratio": 0.5,
  "tracker.det_thresh": 0.4,
  "tracker.low_thresh": 0.1,
  "tracker.match_thresh": 0.4,
  "tracker.track_buffer": 30,
  "tracker.embedding_weight": 0.8,
  "tracker.part_aware_alpha": 0.4,
  "tracker.memory_bank_capacity": 500
} as const;

export type ConfigKey = keyof typeof CONFIG;

const c = (key: ConfigKey) => CONFIG[key];
const fixed = (key: ConfigKey, digits = 1) => CONFIG[key].toFixed(digits);
// 0.3 * 100 is 30.000000000000004 in floating point.
const percent = (key: ConfigKey) => Math.round(CONFIG[key] * 100);

/** The diagram, drawn by tools/architecture/build_diagram.py and served from docs/. */
export const DIAGRAM = {
  src: "/architecture/effiped-architecture.svg",
  width: 780,
  height: 1300,
  alt:
    "EffiPed architecture. A camera frame passes through the stem and first two stages of ConvNeXt V2 Tiny; " +
    "a two-scale neck fuses the stride-4 and stride-8 features into a 256-channel map. A detection head " +
    "predicts heatmap, box, offset and IoU, decoded with Soft-NMS. An identity head pools each box with " +
    "RoIAlign, applies Coordinate Attention, weights four body strips and normalises a 256-D descriptor. " +
    "A per-camera tracker matches detections by motion, IoU and appearance; cross-camera search ranks " +
    "other cameras' tracks by cosine similarity for human review."
} as const;

/* ---- the training objective ---- */

export type LossTerm = {
  name: string;
  loss: string;
  weight: number;
  /** The configuration key the weight is read from, when it has one. */
  key: ConfigKey;
};

export const DETECTION_LOSSES: LossTerm[] = [
  {
    name: "Heatmap",
    loss: "Gaussian focal loss (α 2, β 4); occluded centres weighted up to 1.5×",
    weight: c("loss.cls_weight"),
    key: "loss.cls_weight"
  },
  {
    name: "Box size",
    loss: "GIoU on left, top, right, bottom distances at every pixel inside the centre Gaussian",
    weight: c("loss.reg_weight"),
    key: "loss.reg_weight"
  },
  {
    name: "Centre offset",
    loss: "L1 on the sub-pixel offset to the true centre",
    weight: c("loss.ctr_weight"),
    key: "loss.ctr_weight"
  },
  {
    name: "Repulsion",
    loss: "Smooth L1 on overlap with a neighbour beyond their true overlap",
    weight: c("loss.rep_weight"),
    key: "loss.rep_weight"
  },
  {
    name: "Box quality",
    loss: "Binary cross-entropy between predicted and actual IoU",
    weight: c("loss.iou_weight"),
    key: "loss.iou_weight"
  }
];

export const IDENTITY_LOSSES: LossTerm[] = [
  {
    name: "Identity",
    loss: `ArcFace (s ${c("loss.arcface_s")}, m ${c("loss.arcface_m")}) after BNNeck; label smoothing ${c("loss.label_smoothing")}, dropout ${c("loss.reid_dropout")}, visibility-weighted`,
    weight: c("loss.id_weight"),
    key: "loss.id_weight"
  },
  {
    name: "Strip identity",
    loss: `ArcFace on each of the ${c("model.num_parts_v")} body strips`,
    weight: c("loss.part_loss_weight"),
    key: "loss.part_loss_weight"
  },
  {
    name: "Metric",
    loss: `Batch-hard triplet (margin ${c("loss.triplet_margin")}) before BNNeck, mined from a ${c("loss.xbm_memory_size").toLocaleString("en-US")}-entry cross-batch memory`,
    weight: c("loss.triplet_weight"),
    key: "loss.triplet_weight"
  },
  {
    name: "Strip metric",
    loss: "Batch-hard triplet on each strip, within the batch",
    weight: c("loss.part_triplet_weight"),
    key: "loss.part_triplet_weight"
  },
  {
    name: "Strip diversity",
    loss: "Penalises similarity between the strip descriptors of one person",
    weight: c("loss.diversity_loss_weight"),
    key: "loss.diversity_loss_weight"
  }
];

/* ---- training and inference ---- */

export type Fact = [label: string, value: string];

export const TRAINING: Fact[] = [
  ["Data", "P-DESTRE, MOT17, MOT20, SOMPT22, plus 12% CrowdHuman"],
  [
    "Sampling",
    `Identity-aware batches, +${percent("data.cross_camera_ratio")}% cross-camera batches, 1 frame in ${c("data.frame_skip")}, boxes ≥ ${percent("data.min_visibility")}% visible`
  ],
  ["Input", "1088 × 608"],
  [
    "Augmentation",
    "Colour jitter, random erasing, mosaic and mixup, blur, noise, grayscale, JPEG and resolution degradation"
  ],
  [
    "Optimiser",
    `AdamW, lr ${c("training.lr").toExponential()}, weight decay 0.01, backbone at ${c("training.backbone_lr_mult")}× lr`
  ],
  [
    "Schedule",
    `${c("training.epochs")} epochs, ${c("training.warmup_batches").toLocaleString("en-US")}-batch linear warm-up, cosine warm restarts (T₀ 5, ×2)`
  ],
  [
    "Batch",
    `${c("data.batch_size")} images × ${c("training.gradient_accumulation_steps")} accumulation steps, gradient clip ${fixed("training.grad_clip", 0)}`
  ],
  ["Normalisation", "GroupNorm in the neck and heads"]
];

export const INFERENCE: Fact[] = [
  ["Decode", "3×3 max-pool peaks, top 300, score = √(heatmap × IoU)"],
  ["Suppression", "Gaussian Soft-NMS, σ 0.35"],
  [
    "Matching",
    `Scores ≥ ${c("tracker.det_thresh")} first, then ≥ ${c("tracker.low_thresh")}; cost ≤ ${c("tracker.match_thresh")}; lost tracks kept ${c("tracker.track_buffer")} frames`
  ],
  [
    "Appearance cost",
    `${fixed("tracker.embedding_weight")} appearance + ${(1 - c("tracker.embedding_weight")).toFixed(1)} motion; appearance = ${(1 - c("tracker.part_aware_alpha")).toFixed(1)} fused + ${fixed("tracker.part_aware_alpha")} strip cosine, strips weighted by mutual visibility`
  ],
  ["Memory", `${c("tracker.memory_bank_capacity")}-entry appearance memory for re-acquiring lost tracks`],
  ["Cross-camera", "Each person's mean descriptor ranked by cosine against people in the other clips"]
];

/* ---- the reported results ---- */

type Summary = {
  system_benchmarks: {
    pdestre: {
      validation: { rank1_cross: number; detection_map50: number };
      test: { rank1_cross: number; detection_map50: number };
    };
    mot17: { mota: number; idf1: number; hota: number };
    footprint: {
      parameters_m: number;
      pipeline_fps_approx: number;
      device: string;
      input_resolution: string;
      descriptor_dim: number;
    };
  };
};

const benchmarks = (summaryJson as unknown as Summary).system_benchmarks;

export const FOOTPRINT = benchmarks.footprint;

export const RESULTS: Fact[] = [
  [
    "P-DESTRE cross-camera Rank-1",
    `${benchmarks.pdestre.validation.rank1_cross}% val · ${benchmarks.pdestre.test.rank1_cross}% test`
  ],
  [
    "P-DESTRE detection mAP@0.5",
    `${benchmarks.pdestre.validation.detection_map50}% val · ${benchmarks.pdestre.test.detection_map50}% test`
  ],
  [
    "MOT17 validation half",
    `MOTA ${benchmarks.mot17.mota} · IDF1 ${benchmarks.mot17.idf1} · HOTA ${benchmarks.mot17.hota}`
  ],
  [
    "Size and speed",
    `${FOOTPRINT.parameters_m}M parameters · ≈${FOOTPRINT.pipeline_fps_approx} FPS end to end on an ${FOOTPRINT.device.replace("NVIDIA ", "")}`
  ]
];
