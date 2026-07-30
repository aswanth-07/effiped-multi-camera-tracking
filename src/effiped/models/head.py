#!/usr/bin/env python
"""
effiped CenterNet Head — Detection + ReID Head Architecture
══════════════════════════════════════════════════════════════════

Table of Contents (approximate line numbers):
─────────────────────────────────────────────
  L40   soft_nms() — Gaussian Soft-NMS (score decay instead of hard removal)
  L95   DeformableConv2d — DCNv2 wrapper (3×3 kernel, stores last_offset)
  L177  ConvBlock — Standard or DCN conv + Norm + ReLU (supports GroupNorm)
  L205  CoordinateAttention — H+W axis spatial gates for per-person RoI features
  L272  PartBasedExtractor — RoI-Align + CoordAttn + 4×1 strip split + fusion
  L448  CenterNetHead — Main head class
  L466    __init__() — Det trunk, ReID trunk/head, IoU branch, P2 heatmap setup
  L690    _init_weights() — Kaiming + focal-loss bias init (hm weight zeroed)
  L740    forward() — Full head: features → det outputs + ReID embeddings + hm_s4
  L827    decode_detections() — CenterNet decode: peaks → boxes + embeddings

Key Architecture:
  Fused Features [B, 256, H, W]
    ├── Det Trunk (shared_convs: 2× Conv, last can be DCN)
    │   ├── hm, wh, offset branches (1×1 heads)
    │   └── iou branch
    └── ReID Trunk (optional decoupled: 2× StdConv)
        └── ReID Head (2× Conv + 1×1 proj → embedding map)
            └── RoI-Align → CoordinateAttention → PartBasedExtractor
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, deform_conv2d, roi_align

from .common import sample_feature_map_bilinear

# ── Soft-NMS ──────────────────────────────────────────────────────────────────

def soft_nms(boxes, scores, sigma=0.5, score_thresh=0.001, method='gaussian'):
    """
    Soft-NMS: Decays scores of overlapping boxes instead of removing them.
    Better than hard NMS for crowded pedestrian scenes.
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device), scores.clone()

    # Work on CPU to avoid CUDA async issues with box_iou
    device = boxes.device
    boxes = boxes.cpu()
    scores = scores.cpu().clone()
    order = torch.arange(len(scores))
    keep = []
    kept_scores = []

    while len(scores) > 0:
        max_idx = scores.argmax().item()
        keep.append(order[max_idx].item())
        kept_scores.append(scores[max_idx].item())

        if len(scores) == 1:
            break

        current_box = boxes[max_idx:max_idx+1]
        other_boxes = torch.cat([boxes[:max_idx], boxes[max_idx+1:]])
        other_scores = torch.cat([scores[:max_idx], scores[max_idx+1:]])
        other_order = torch.cat([order[:max_idx], order[max_idx+1:]])

        if len(other_boxes) == 0:
            break

        ious = box_iou(current_box, other_boxes)[0]
        # Guard against NaN from degenerate zero-area boxes
        ious = torch.nan_to_num(ious, nan=0.0)

        if method == 'gaussian':
            decay = torch.exp(-(ious ** 2) / sigma)
        else:
            decay = torch.where(ious > 0.3, 1 - ious, torch.ones_like(ious))

        other_scores = other_scores * decay

        mask = other_scores > score_thresh
        boxes = other_boxes[mask]
        scores = other_scores[mask]
        order = other_order[mask]

    keep = torch.tensor(keep, dtype=torch.long, device=device)
    kept_scores = torch.tensor(kept_scores, device=device)
    return keep, kept_scores


# ── DeformableConv2d ──────────────────────────────────────────────────────────

class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 (DCNv2).

    Learns spatial offsets and modulation masks to adaptively sample input locations.
    Enables better feature extraction for pedestrians with varying poses, scales,
    and partial occlusions. Uses torchvision.ops.deform_conv2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=True):
        super(DeformableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Main convolution weight
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Offset and mask prediction
        num_offset_groups = 1
        offset_channels = 2 * self.kernel_size[0] * self.kernel_size[1] * num_offset_groups
        mask_channels = self.kernel_size[0] * self.kernel_size[1] * num_offset_groups

        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            offset_channels + mask_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            padding_mode='replicate',
            dilation=dilation,
            bias=True
        )

        self._reset_parameters()
        self.last_offset = None  # Set in forward(), used for DCN bbox penalty

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize offsets to zero (start with regular conv behavior)
        nn.init.zeros_(self.offset_mask_conv.weight)
        nn.init.zeros_(self.offset_mask_conv.bias)

    def forward(self, x):
        offset_mask = self.offset_mask_conv(x)
        offset_channels = 2 * self.kernel_size[0] * self.kernel_size[1]

        offset = offset_mask[:, :offset_channels, :, :]
        mask = torch.sigmoid(offset_mask[:, offset_channels:, :, :])

        # Save offset for bounding-box constraint penalty
        self.last_offset = offset

        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask
        )


# ── ConvBlock ─────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Standard Conv-Norm-ReLU block with optional DCNv2 and GroupNorm.
    
    When use_group_norm=True, replaces BatchNorm2d with GroupNorm(32).
    GroupNorm is batch-size-independent — critical for BS=2 training where
    BN statistics are extremely noisy (2 samples per mean/var estimation).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 use_dcn=False, use_group_norm=False):
        super(ConvBlock, self).__init__()
        if use_dcn:
            self.conv = DeformableConv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_group_norm)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False, padding_mode='replicate')
        if use_group_norm:
            # 32 groups is standard (Wu & He, 2018). If channels < 32, fall back to fewer groups.
            num_groups = min(32, out_channels)
            self.bn = nn.GroupNorm(num_groups, out_channels)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ── CoordinateAttention ───────────────────────────────────────────────────────

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (Hou et al., CVPR 2021).

    Decomposes channel attention into H-axis and W-axis spatial gates.
    Unlike SE (spatially uniform), produces direction-aware attention
    that can distinguish "attend to head region" from "attend to leg region".

    Used inside PartBasedExtractor after RoI-Align to provide per-person
    spatial context before horizontal strip partitioning. The H-axis gate
    directly aligns with 4×1 body-part decomposition (head/torso/legs).

    Architecture:
      x [N, C, H, W]
        ├─ H-pool: mean(W) → [N, C, H, 1]  (row-level summary)
        └─ W-pool: mean(H) → [N, C, 1, W]  (column-level summary)
      concat → shared 1×1 bottleneck → split → separate 1×1 expand → sigmoid gates
      out = x * attn_h * attn_w

    Params: ~12.5K for C=256, r=16 (0.05% of total model).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(8, channels // reduction)  # 256/16 = 16, min 8

        # Shared 1×1 bottleneck for joint H+W encoding
        self.conv_reduce = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn = nn.GroupNorm(min(16, mid), mid)  # GN for batch-size independence
        self.act = nn.SiLU(inplace=True)

        # Separate 1×1 expansions for H and W gates
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C, H, W] per-person RoI features (e.g. [N, 256, 32, 8])
        Returns:
            [N, C, H, W] spatially-gated features (same shape)
        """
        _, _, H, W = x.shape

        # Pool along each axis independently
        x_h = x.mean(dim=3, keepdim=True)     # [N, C, H, 1]  row summaries
        x_w = x.mean(dim=2, keepdim=True)     # [N, C, 1, W]  col summaries

        # Concatenate along spatial dim for joint encoding
        x_w_t = x_w.permute(0, 1, 3, 2)       # [N, C, W, 1]
        y = torch.cat([x_h, x_w_t], dim=2)    # [N, C, H+W, 1]

        # Shared bottleneck: reduce channels, normalize, activate
        y = self.act(self.bn(self.conv_reduce(y)))  # [N, mid, H+W, 1]

        # Split back into H and W components
        y_h, y_w = y.split([H, W], dim=2)

        # Expand to full channels with separate projections → sigmoid gates
        attn_h = self.conv_h(y_h).sigmoid()                    # [N, C, H, 1]
        attn_w = self.conv_w(y_w.permute(0, 1, 3, 2)).sigmoid()  # [N, C, 1, W]

        return x * attn_h * attn_w


# ── PartBasedExtractor ────────────────────────────────────────────────────────

class PartBasedExtractor(nn.Module):
    """
    4x1 Horizontal Strip Part-Based ReID Feature Extractor.

    Splits RoI-Aligned features into 4 vertical strips (horizontal slices):
      4 Vertical (Head, Upper Torso, Lower Torso, Legs) x 1 Horizontal = 4 parts.

    Optimized for stride-4 high-resolution features with doubled RoI output
    size (32x8) to capture fine-grained detail at P2 resolution.

    Pipeline per person:
      1. RoI-Align → [N, C, 32, 8]
      2. CoordinateAttention → spatial gating (H-axis=body parts, W-axis=edges)
      3. Split into 4×1 horizontal strips → [N, 4, C, 8, 8]
      4. GAP each strip → [N, 4, C]
      5. L2-normalize each part independently
      6. Attention MLP → per-part visibility weights (softmax over 4 parts)
      7. Weighted fusion → [N, C] final embedding

    The Coordinate Attention at step 2 provides cross-spatial context BEFORE
    the rigid strip split, enabling the network to emphasize discriminative
    body regions (e.g., distinctive jacket) and suppress uninformative ones
    (e.g., occluded legs). This directly addresses the limitation of per-part
    attention being context-blind.

    Usage:
      fused, parts, attn = extractor(embedding_map, roi_boxes)
      # fused: [N, C]   — final embedding for tracker/ArcFace
      # parts: [N, 4, C] — per-part embeddings for auxiliary losses
      # attn:  [N, 4]    — attention weights (sum=1)
    """

    def __init__(self, embed_dim: int = 256, num_parts_v: int = 4, num_parts_h: int = 1,
                 roi_output_size: tuple = (32, 8), part_fusion_type: str = 'attention_sum',
                 use_coord_attention: bool = True,
                 part_attention_dropout_p: float = 0.0,
                 part_dropout_p: float = None):
        super(PartBasedExtractor, self).__init__()
        self.embed_dim = embed_dim
        self.num_parts_v = num_parts_v
        self.num_parts_h = num_parts_h
        self.num_parts = num_parts_v * num_parts_h  # 4
        self.roi_output_size = roi_output_size  # (32, 8)
        self.part_fusion_type = part_fusion_type
        self.use_coord_attention = use_coord_attention
        if part_dropout_p is not None:
            part_attention_dropout_p = part_dropout_p
        self.part_attention_dropout_p = float(part_attention_dropout_p)
        self.part_dropout_p = self.part_attention_dropout_p  # backward-compatible alias

        # Patch size after partition: (32/4, 8/1) = (8, 8)
        self.patch_h = roi_output_size[0] // num_parts_v
        self.patch_w = roi_output_size[1] // num_parts_h

        # Coordinate Attention: per-person spatial gating before strip split.
        # H-axis gate → row-level importance (head/torso/legs differentiation)
        # W-axis gate → column-level importance (center vs. edge reliability)
        self.coord_attn = CoordinateAttention(embed_dim, reduction=16) if use_coord_attention else nn.Identity()

        # Attention MLP: learns per-part visibility/importance weights
        # Input: part feature [C], Output: scalar logit
        self.attention_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # Concat+MLP fusion: adds residual discriminative features from
        # all-parts concatenation on top of attention-weighted base.
        # [N, num_parts*C] → [N, 2*C] → [N, C]
        self.fusion_mlp = None
        if part_fusion_type == 'concat_mlp':
            print("  [deprecated] part_fusion_type='concat_mlp' is retained for old checkpoints only; "
                  "new part configs use attention_sum. Part-attention dropout does not mask "
                  "the concat MLP residual path.")
            concat_dim = embed_dim * self.num_parts  # 4*256 = 1024
            self.fusion_mlp = nn.Sequential(
                nn.Linear(concat_dim, embed_dim * 2),   # 1024 → 512
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),    # 512 → 256
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.attention_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize last layer bias so initial attention is uniform (all zeros -> softmax = 1/4)
        nn.init.zeros_(self.attention_mlp[-1].weight)
        nn.init.zeros_(self.attention_mlp[-1].bias)

        # Fusion MLP: Xavier uniform for GELU, zero-init final layer
        # so fused output starts as zeros → first few batches rely on
        # the L2-normalized parts while MLP warms up.
        if self.fusion_mlp is not None:
            for m in self.fusion_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Zero-init final projection for stable warmup
            nn.init.zeros_(self.fusion_mlp[-1].weight)
            nn.init.zeros_(self.fusion_mlp[-1].bias)

    def forward(self, embedding_map: torch.Tensor, roi_boxes: torch.Tensor,
                spatial_scale: float = 1.0,
                return_attention_details: bool = False) -> tuple:
        """
        Args:
            embedding_map: [B, C, H, W] embedding feature map
            roi_boxes: [N, 5] with [batch_idx, x1, y1, x2, y2] in feature-map coords
            spatial_scale: scale factor for roi_align (1.0 if boxes already in feature coords)

        Returns:
            fused_embedding: [N, C] attention-weighted fusion of parts
            part_embeddings: [N, 4, C] individual part vectors
            attention_weights: [N, 4] softmax weights
        """
        N = roi_boxes.size(0)
        C = self.embed_dim
        device = embedding_map.device

        if N == 0:
            empty = (
                torch.zeros(0, C, device=device),
                torch.zeros(0, self.num_parts, C, device=device),
                torch.zeros(0, self.num_parts, device=device)
            )
            if return_attention_details:
                return empty + (torch.zeros(0, self.num_parts, device=device),)
            return empty

        # RoI-Align to fixed grid: [N, C, 32, 8]
        roi_features = roi_align(
            embedding_map, roi_boxes,
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            aligned=True
        )

        # Coordinate Attention: per-person spatial gating before strip split.
        # The H-axis gate learns row-level importance (e.g., "head region is
        # discriminative for this person") while the W-axis gate learns
        # column-level importance (e.g., "center is more reliable than edges").
        # This provides cross-spatial context that individual strip attention
        # cannot compute — addressing the context-blind limitation.
        roi_features = self.coord_attn(roi_features)

        # Partition into 4x1 grid of horizontal strips
        # roi_features: [N, C, 32, 8] -> reshape to [N, C, 4, 8, 1, 8]
        roi_features = roi_features.view(
            N, C,
            self.num_parts_v, self.patch_h,
            self.num_parts_h, self.patch_w
        )
        # Permute to [N, 4, 1, C, 8, 8] then merge grid dims -> [N, 4, C, 8, 8]
        roi_features = roi_features.permute(0, 2, 4, 1, 3, 5).contiguous()
        roi_features = roi_features.view(N, self.num_parts, C, self.patch_h, self.patch_w)

        # Global Average Pool each patch -> [N, 4, C]
        part_embeddings = roi_features.mean(dim=[3, 4])  # [N, 4, C]

        # L2-normalize each part vector independently
        part_embeddings = F.normalize(part_embeddings, p=2, dim=2, eps=1e-6)

        # Attention: compute per-part importance weights
        # [N, 4, C] -> [N, 4, 1] -> softmax over parts -> [N, 4]
        raw_attn_logits = self.attention_mlp(part_embeddings)  # [N, 4, 1]
        raw_attention_weights = F.softmax(raw_attn_logits.squeeze(-1), dim=1)
        attn_logits = raw_attn_logits
        if self.training and self.part_attention_dropout_p > 0:
            drop_mask = torch.rand(N, self.num_parts, device=device) < self.part_attention_dropout_p
            all_dropped = drop_mask.all(dim=1)
            if all_dropped.any():
                drop_mask[all_dropped, 0] = False
            attn_logits = attn_logits.masked_fill(drop_mask.unsqueeze(-1), -1e4)
        attention_weights = F.softmax(attn_logits.squeeze(-1), dim=1)  # [N, 4]

        # Attention-weighted sum: occlusion-aware base embedding
        base = (attention_weights.unsqueeze(-1) * part_embeddings).sum(dim=1)  # [N, C]

        if self.fusion_mlp is not None:
            # Residual concat+MLP: base stays trained via attention gradients,
            # MLP adds discriminative delta from all 4 part concatenation.
            # fusion_mlp final layer is zero-init → delta starts at 0 →
            # fused == base initially, then MLP learns corrective features.
            delta = self.fusion_mlp(part_embeddings.view(N, -1))  # [N, 4*C] → [N, C]
            fused = base + delta
        else:
            fused = base

        # L2-normalize fused embedding
        fused = F.normalize(fused, p=2, dim=1, eps=1e-6)

        if return_attention_details:
            return fused, part_embeddings, attention_weights, raw_attention_weights
        return fused, part_embeddings, attention_weights


# ── CenterNetHead ─────────────────────────────────────────────────────────────

class CenterNetHead(nn.Module):
    """
    CenterNet-style Detection Head with ReID Branch.

    Architecture:
      - Detection Trunk: Conv0 (Standard) + Conv1 (DCNv2)
      - Detection: heatmap + WH + offset (direct from trunk)
      - ReID Trunk (Decoupled): Conv0 (Standard) + Conv1 (Standard) — NO DCN
      - ReID Head: Conv layers + 1x1 Proj (DCN controlled by reid_head_use_dcn)
      - Decoding: 3x3 MaxPool peak extraction + Gaussian Soft-NMS

    Stride modes:
      - det_stride=4: Default. Detection heads at stride-4 (152×272 for 1088×608).
      - det_stride=8: AvgPool2d(2) before det trunk → 4× fewer positions → ~75% det FLOP savings.
      - reid_stride=4: Default. ReID at stride-4.
      - reid_stride=8: AvgPool2d(2) before ReID trunk → ~75% ReID FLOP savings.

    ReID extraction modes (reid_extraction config):
      - 'center': Bilinear interpolation at sub-pixel center (FairMOT-style).
          DCN recommended: helps single-point sampling reach wider context.
      - 'part_based': 4x1 horizontal strip RoI-Align with attention fusion.
          Standard conv recommended: preserves spatial equivariance for strips.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,
        embedding_dim: int = 128,
        head_channels: int = 256,
        num_convs: int = 2,
        use_decoupled_reid: bool = False,
        prior_prob: float = 0.01,

        reid_head_depth: int = 3,
        use_dcn: bool = False,
        reid_head_use_dcn: bool = None,
        use_iou_branch: bool = False,
        reid_extraction: str = 'center',
        num_parts_v: int = 4,
        num_parts_h: int = 1,
        roi_output_size: tuple = (32, 8),
        use_group_norm: bool = False,
        shared_det_head: bool = False,
        reid_stride: int = 4,
        det_stride: int = 4,
        input_stride: int = 4,
        use_p2_heatmap: bool = False,
        p2_channels: int = 96,
        part_fusion_type: str = 'attention_sum',
        use_coord_attention: bool = True,
        use_reid_bnneck: bool = True,
        part_attention_dropout_p: float = 0.0,
        part_dropout_p: float = 0.0,
    ):
        super(CenterNetHead, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_decoupled_reid = use_decoupled_reid

        self.reid_head_depth = reid_head_depth
        self.use_dcn = use_dcn
        # ReID head DCN: if not explicitly set, inherit from global use_dcn.
        # For part_based extraction, standard convs maintain spatial equivariance
        # needed for horizontal strip decomposition. DCN disrupts the spatial
        # correspondence that part-based ReID depends on.
        self.reid_head_use_dcn = use_dcn if reid_head_use_dcn is None else reid_head_use_dcn
        self.use_iou_branch = use_iou_branch
        self.reid_extraction = reid_extraction
        self.use_group_norm = use_group_norm
        self.shared_det_head = shared_det_head
        self.reid_stride = reid_stride
        self.det_stride = det_stride
        self.input_stride = input_stride
        # Stride ratio between fusion feature map and det/reid heads.
        # 1 = same resolution (default), 2 = heads at half resolution.
        self._reid_stride_ratio = reid_stride // input_stride
        self._det_stride_ratio = det_stride // input_stride
        self.use_p2_heatmap = use_p2_heatmap
        self.p2_channels = p2_channels
        if part_dropout_p is not None and part_attention_dropout_p == 0.0:
            part_attention_dropout_p = part_dropout_p
        self.part_attention_dropout_p = float(part_attention_dropout_p)
        self.part_dropout_p = self.part_attention_dropout_p  # backward-compatible alias
        self.use_reid_bnneck = use_reid_bnneck

        # === Detection Downsample (det_stride > input_stride) ===
        # AvgPool2d(ratio) before det trunk → ratio² fewer positions per conv layer.
        self.det_downsample = None
        if self._det_stride_ratio > 1:
            self.det_downsample = nn.AvgPool2d(kernel_size=self._det_stride_ratio,
                                                stride=self._det_stride_ratio)

        # === Detection Trunk (shared_convs) ===
        # Optimal DCN layout: Conv0 = Standard, Conv1 = DCNv2
        # Standard first layer handles channel projection cheaply;
        # DCN on last layer enables geometric adaptation at stride 4.
        self.shared_convs = nn.Sequential()
        for i in range(num_convs):
            ch_in = in_channels if i == 0 else head_channels
            use_dcn_here = use_dcn and (i == num_convs - 1)  # DCN on last layer only
            self.shared_convs.add_module(
                f'conv{i}',
                ConvBlock(ch_in, head_channels, use_dcn=use_dcn_here,
                          use_group_norm=use_group_norm)
            )

        # === Detection Heads ===

        def _make_norm(channels):
            """Create norm layer — GroupNorm(32) if use_group_norm, else BatchNorm2d."""
            if use_group_norm:
                return nn.GroupNorm(min(32, channels), channels)
            return nn.BatchNorm2d(channels)

        if shared_det_head:
            # EFFICIENT: 1 shared 3×3 intermediate + 4 cheap 1×1 projections.
            # Saves 3 × 3×3 Conv(256→256) at stride-4 (~73 GMACs, ~27% of head).
            self.det_head_shared = nn.Sequential(
                nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=False, padding_mode='replicate'),
                _make_norm(head_channels),
                nn.ReLU(inplace=True),
            )
            self.heatmap_head = nn.Conv2d(head_channels, num_classes, 1)
            self.wh_head = nn.Conv2d(head_channels, 4, 1)
            self.offset_head = nn.Conv2d(head_channels, 2, 1)
            self.iou_head = nn.Conv2d(head_channels, 1, 1) if use_iou_branch else None
        else:
            # STANDARD: per-branch 3×3 + 1×1 (backward-compatible with existing checkpoints)
            self.det_head_shared = None

            # Heatmap head (center detection)
            self.heatmap_head = nn.Sequential(
                nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=False, padding_mode='replicate'),
                _make_norm(head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, num_classes, 1)
            )

            # Width-Height regression head
            self.wh_head = nn.Sequential(
                nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=False, padding_mode='replicate'),
                _make_norm(head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, 4, 1)  # LTRB: Left, Top, Right, Bottom distances from center
            )

            # Offset head (sub-pixel center refinement)
            self.offset_head = nn.Sequential(
                nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=False, padding_mode='replicate'),
                _make_norm(head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, 2, 1)
            )

            # IoU prediction head (quality-aware scoring)
            self.iou_head = None
            if use_iou_branch:
                self.iou_head = nn.Sequential(
                    nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=False, padding_mode='replicate'),
                    _make_norm(head_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_channels, 1, 1)
                )

        # === Re-Identification Branch (Decoupled, NO DCN) ===
        # Standard convolutions for VRAM efficiency.
        # DCN is reserved for the reid_head where it matters most.
        # reid_stride > input_stride: inserts AvgPool before the trunk to reduce
        # spatial resolution. Saves ~75% of reid_branch + reid_head FLOPs
        # since only ~50 RoI boxes are sampled from the map anyway.
        self.reid_downsample = None
        if use_decoupled_reid:
            self.reid_branch = nn.Sequential()
            if self._reid_stride_ratio > 1:
                self.reid_downsample = nn.AvgPool2d(kernel_size=self._reid_stride_ratio,
                                                     stride=self._reid_stride_ratio)
            for i in range(num_convs):
                ch_in = in_channels if i == 0 else head_channels
                self.reid_branch.add_module(
                    f'reid_trunk_conv{i}',
                    ConvBlock(ch_in, head_channels, use_dcn=False,
                              use_group_norm=use_group_norm)
                )
        else:
            self.reid_branch = None

        # ReID head: Conv layers + 1x1 Projection
        # DCN usage controlled by reid_head_use_dcn (may differ from det trunk).
        # For part_based extraction: standard conv preserves spatial alignment.
        # For center extraction: DCN helps single-point sampling reach wider.
        reid_layers = []
        for i in range(reid_head_depth - 1):
            reid_layers.append(ConvBlock(head_channels, head_channels,
                                         use_dcn=self.reid_head_use_dcn,
                                         use_group_norm=use_group_norm))
        reid_layers.append(nn.Conv2d(head_channels, embedding_dim, 1))
        self.reid_head = nn.Sequential(*reid_layers)

        # Model-owned BNNeck modules. Training passes these into CenterNetLoss,
        # and inference applies them directly without rebuilding the criterion.
        self.reid_bnneck = None
        self.part_bnneck = None
        if use_reid_bnneck:
            self.reid_bnneck = nn.BatchNorm1d(embedding_dim, affine=True)
            nn.init.ones_(self.reid_bnneck.weight)
            nn.init.zeros_(self.reid_bnneck.bias)

        # Part-based ReID extractor (4x1 horizontal strips for stride-4)
        self.part_extractor = None
        if reid_extraction == 'part_based':
            self.part_extractor = PartBasedExtractor(
                embed_dim=embedding_dim,
                num_parts_v=num_parts_v, num_parts_h=num_parts_h,
                roi_output_size=roi_output_size,
                part_fusion_type=part_fusion_type,
                use_coord_attention=use_coord_attention,
                part_attention_dropout_p=self.part_attention_dropout_p,
            )
            if use_reid_bnneck:
                self.part_bnneck = nn.ModuleList([
                    nn.BatchNorm1d(embedding_dim, affine=True)
                    for _ in range(self.part_extractor.num_parts)
                ])
                for bn in self.part_bnneck:
                    nn.init.ones_(bn.weight)
                    nn.init.zeros_(bn.bias)

        # P2-guided stride-4 heatmap branch (split-stride detection)
        # Produces a high-resolution heatmap from raw P2 features + upsampled
        # fused context. Only the heatmap is at stride-4; WH/offset/IoU/ReID
        # stay at the det_stride (typically stride-8). This doubles peak separation
        # resolution at <1% FLOP overhead.
        self.p2_hm_compress = None
        self.fused_hm_compress = None
        self.hm_s4_refine = None
        self.hm_s4_out = None
        p2_hm_ch = 16
        if use_p2_heatmap:
            self.p2_hm_compress = nn.Conv2d(p2_channels, p2_hm_ch, 1, bias=False)
            self.fused_hm_compress = nn.Conv2d(head_channels, p2_hm_ch, 1, bias=False)
            self.hm_s4_refine = nn.Sequential(
                nn.Conv2d(p2_hm_ch, p2_hm_ch, 3, padding=1, bias=False),
                nn.GroupNorm(min(4, p2_hm_ch), p2_hm_ch),
                nn.ReLU(inplace=True),
            )
            self.hm_s4_out = nn.Conv2d(p2_hm_ch, num_classes, 1)

        # Initialize weights
        self._init_weights(prior_prob)
        
        print(f"  CenterNetHead initialized:{' [shared_det_head]' if shared_det_head else ''}{' [det@stride-' + str(det_stride) + ']' if det_stride != 4 else ''}{' [reid@stride-' + str(reid_stride) + ']' if reid_stride != 4 else ''}")
        print(f"   - Detection: {'shared 3x3 + 4x 1x1 proj' if shared_det_head else 'direct from trunk (no per-branch ECA)'}")
        print("   - ReID: direct (no ECA — proven near-uniform across models)")

        print(f"   - Decoupled ReID: {use_decoupled_reid} (Independent Trunk, NO DCN{', stride-' + str(reid_stride) if reid_stride != 4 else ''})")
        print(f"   - DCNv2 Detection Trunk: {use_dcn} (Conv1 only)")
        dcn_str = f"{reid_head_depth - 1} DCN + 1x1 proj" if self.reid_head_use_dcn else f"{reid_head_depth - 1} std + 1x1 proj"
        print(f"   - DCNv2 ReID Head: {self.reid_head_use_dcn} ({dcn_str})")
        print(f"   - IoU branch: {use_iou_branch}")
        if use_p2_heatmap:
            print(f"   - P2 heatmap: stride-4 ({p2_channels}ch P2 + fused → {p2_hm_ch}ch → 1ch)")
        if reid_extraction == 'part_based':
            layout = f"{num_parts_v}x{num_parts_h}" if num_parts_h > 1 else f"{num_parts_v} strips"
            coord_str = "CoordAttn" if use_coord_attention else "no CoordAttn"
            print(f"   - ReID extraction: {reid_extraction} ({layout}, roi={roi_output_size}, {coord_str}, part_attention_dropout={self.part_attention_dropout_p})")
        else:
            print(f"   - ReID extraction: {reid_extraction} (bilinear center)")
        print(f"   - Model-owned BNNeck: {use_reid_bnneck}")

    def _init_weights(self, prior_prob):
        # Collect DCNv2 modules with specialized initialization.
        # These must NOT be overwritten by the blanket kaiming init below.
        # DeformableConv2d._reset_parameters() zeros offset_mask_conv so DCN
        # starts as a regular conv and gradually learns offsets.
        skip_modules = set()
        for m in self.modules():
            if isinstance(m, DeformableConv2d):
                for child in m.modules():
                    skip_modules.add(child)

        for m in self.modules():
            if m in skip_modules:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Focal Loss initialization for heatmap head
        # Zeros init: weight=0, bias=-4.60 (prior_prob=0.01).
        # logit = 0·x + bias = bias for ALL pixels → sigmoid(bias) = 0.01.
        # Architecture-independent start. Kaiming init caused RNG-dependent
        # logit collapse (cls=13.82 stuck for 15K+ batches) in compact dual-scale models.
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # _get_final_layer handles both Sequential (per-branch) and plain Conv2d (shared)
        def _final(module):
            return module[-1] if isinstance(module, nn.Sequential) else module
        nn.init.zeros_(_final(self.heatmap_head).weight)
        nn.init.constant_(_final(self.heatmap_head).bias, bias_value)

        # Initialize WH head
        nn.init.zeros_(_final(self.wh_head).weight)
        nn.init.constant_(_final(self.wh_head).bias, 0.1)

        nn.init.zeros_(_final(self.offset_head).weight)
        nn.init.zeros_(_final(self.offset_head).bias)

        # IoU head init
        if self.iou_head is not None:
            nn.init.zeros_(_final(self.iou_head).weight)
            nn.init.zeros_(_final(self.iou_head).bias)

        # P2 stride-4 heatmap branch: same focal init (zeros weight + bias=-4.60)
        if self.hm_s4_out is not None:
            nn.init.zeros_(self.hm_s4_out.weight)
            nn.init.constant_(self.hm_s4_out.bias, bias_value)

    def migrate_reid_necks_from_criterion_state(self, criterion_state: dict) -> int:
        """Load old criterion-owned BNNeck weights into model-owned necks."""
        if not criterion_state:
            return 0

        migrated = 0
        if self.reid_bnneck is not None:
            own_state = self.reid_bnneck.state_dict()
            remap = {
                "weight": "bnneck.weight",
                "bias": "bnneck.bias",
                "running_mean": "bnneck.running_mean",
                "running_var": "bnneck.running_var",
                "num_batches_tracked": "bnneck.num_batches_tracked",
            }
            load_state = {}
            for dst, src in remap.items():
                v = criterion_state.get(src)
                if v is not None and dst in own_state and v.shape == own_state[dst].shape:
                    load_state[dst] = v
            if load_state:
                own_state.update(load_state)
                self.reid_bnneck.load_state_dict(own_state)
                migrated += len(load_state)

        if self.part_bnneck is not None:
            for idx, bn in enumerate(self.part_bnneck):
                own_state = bn.state_dict()
                load_state = {}
                for dst in own_state:
                    src = f"part_bnneck.{idx}.{dst}"
                    v = criterion_state.get(src)
                    if v is not None and v.shape == own_state[dst].shape:
                        load_state[dst] = v
                if load_state:
                    own_state.update(load_state)
                    bn.load_state_dict(own_state)
                    migrated += len(load_state)

        return migrated

    def apply_reid_necks(self, dets: dict) -> dict:
        """Apply model-owned BNNecks to decoded fused and part embeddings."""
        emb = dets.get('embeddings')
        if emb is not None and emb.numel() > 0 and self.reid_bnneck is not None:
            emb = F.normalize(emb, p=2, dim=1, eps=1e-6)
            emb = self.reid_bnneck(emb)
            dets['embeddings'] = F.normalize(emb, p=2, dim=1, eps=1e-6)

        part_emb = dets.get('part_embeddings')
        if (
            part_emb is not None
            and part_emb.numel() > 0
            and self.part_bnneck is not None
        ):
            part_emb = F.normalize(part_emb, p=2, dim=2, eps=1e-6)
            parts = []
            n_parts = min(part_emb.shape[1], len(self.part_bnneck))
            for pi in range(n_parts):
                p = self.part_bnneck[pi](part_emb[:, pi, :])
                p = F.normalize(p, p=2, dim=1, eps=1e-6)
                parts.append(p.unsqueeze(1))
            if parts:
                dets['part_embeddings'] = torch.cat(parts, dim=1)
        return dets

    def forward(self, features, raw_p2=None):
        """
        Forward pass.
        Args:
            features: Feature map from backbone/neck, shape [B, C, H, W]
            raw_p2:   Optional raw P2 features at stride-4 [B, P2_C, H*2, W*2]
                      for split-stride heatmap (use_p2_heatmap=True)
        Returns:
            dict with keys: 'hm', 'wh', 'offset', 'embedding', 'hm_s4' (optional)
        """
        # Detection Trunk (always runs)
        det_input = features
        if self.det_downsample is not None:
            det_input = self.det_downsample(det_input)
        det_feat = self.shared_convs(det_input)

        # Detection heads
        if self.det_head_shared is not None:
            # EFFICIENT: shared 3×3 intermediate, per-branch 1×1 projections
            det_head_feat = self.det_head_shared(det_feat)
            heatmap = self.heatmap_head(det_head_feat)
            wh = F.softplus(self.wh_head(det_head_feat))
            offset = self.offset_head(det_head_feat)
            iou_pred = self.iou_head(det_head_feat) if self.iou_head is not None else None
        else:
            # STANDARD: per-branch 3×3 + 1×1
            heatmap = self.heatmap_head(det_feat)
            wh = F.softplus(self.wh_head(det_feat))
            offset = self.offset_head(det_feat)
            iou_pred = None
            if self.iou_head is not None:
                iou_pred = self.iou_head(det_feat)

        # === ReID Stream ===
        if self.use_decoupled_reid:
            reid_input = features
            if self.reid_downsample is not None:
                reid_input = self.reid_downsample(reid_input)
            reid_feat_base = self.reid_branch(reid_input)
        else:
            reid_feat_base = det_feat

        # ReID features go directly to head (no ECA — analysis showed uniform
        # weights 0.45-0.53 across both backbone types tested)

        # Final embedding projection
        embedding = self.reid_head(reid_feat_base)

        # Collect DCN offsets from ReID path only for bbox-constraint penalty.
        # Detection trunk DCN is EXCLUDED: it needs to sample context outside
        # GT boxes (edges, ground plane) for accurate WH regression.
        # When reid_head_use_dcn=False (e.g., part_based extraction), this
        # list stays empty and the DCN penalty gracefully returns 0.
        all_dcn_offsets = []
                
        # ReID Trunk (Decoupled path — currently has NO DCN, but future-proof)
        if self.use_decoupled_reid and self.reid_branch is not None:
            for m in self.reid_branch.modules():
                if isinstance(m, DeformableConv2d):
                    all_dcn_offsets.append(m.last_offset)
                    
        # ReID Head (DCN layers if reid_head_use_dcn=True)
        for m in self.reid_head.modules():
            if isinstance(m, DeformableConv2d):
                all_dcn_offsets.append(m.last_offset)

        # P2-guided stride-4 heatmap (split-stride detection)
        hm_s4 = None
        if self.use_p2_heatmap and raw_p2 is not None:
            # Compress fused features and upsample to stride-4
            ctx = self.fused_hm_compress(det_input)  # [B, 16, H, W] at det_stride
            ctx = F.interpolate(ctx, size=raw_p2.shape[2:], mode='bilinear', align_corners=False)
            # Compress raw P2 spatial detail
            detail = self.p2_hm_compress(raw_p2)     # [B, 16, H*2, W*2] at stride-4
            # Combine context + detail → refine → heatmap
            hm_s4 = self.hm_s4_out(self.hm_s4_refine(ctx + detail))

        return {
            'hm': heatmap,
            'wh': wh,
            'offset': offset,
            'embedding': embedding,
            'iou': iou_pred,  # None if IoU branch disabled
            'reid_offsets': all_dcn_offsets,  # Loss.py will heavily penalize these if they escape the BBox
            'hm_s4': hm_s4,  # None if P2 heatmap disabled
        }

    def decode_detections(
        self,
        outputs: dict,
        K: int = 100,
        conf_thresh: float = 0.3,
        original_size: tuple = None,
        input_size: tuple = None,
        nms_sigma: float = 0.35,
        use_max_pool_nms: bool = True,
        maxpool_kernel: int = 3,
        min_box_area: float = 100.0,
        min_box_height: float = 20.0,
        return_embeddings: bool = True,
        return_part_features: bool = False,
        hm_pre_sigmoided: bool = False,
        iou_power: float = 0.5,
    ):
        """
        Decode CenterNet outputs into detections.

        Pipeline:
          1. Sigmoid on heatmap (skip if hm_pre_sigmoided=True, e.g. from flip TTA)
          2. MaxPool peak extraction (local maxima only)
          3. Top-K candidate selection
          4. Confidence filtering
          5. Box decoding (center + WH + offset -> xyxy)
          6. Coordinate scaling (feature -> input -> original image)
          7. Minimum box size filtering
          8. Gaussian Soft-NMS at box level

        Args:
            outputs: dict with 'hm', 'wh', 'offset', optional 'embedding'
            K: Max detections per image (Top-K)
            conf_thresh: Minimum heatmap confidence
            original_size: (h, w) of original image (for naive rescaling)
            input_size: (h, w) of network input (for stride computation)
            nms_sigma: Gaussian Soft-NMS sigma (lower = more suppression, 0.35 is good for pedestrians)
            use_max_pool_nms: Apply MaxPool peak extraction before Top-K
            min_box_area: Minimum box area in pixels (after scaling)
            min_box_height: Minimum box height in pixels (after scaling)
            return_embeddings: If False, skip all ReID/part extraction and return boxes/scores only
            hm_pre_sigmoided: If True, skip sigmoid (heatmap already in [0,1] range from TTA merging)

        Returns:
            List[dict] per batch element with 'boxes', 'scores', and optional embeddings
        """
        return_embeddings = bool(return_embeddings or return_part_features)
        heatmap = outputs['hm'] if hm_pre_sigmoided else torch.sigmoid(outputs['hm'])
        wh = outputs['wh']
        offset = outputs['offset']
        embedding = outputs.get('embedding', None) if return_embeddings else None
        iou_raw = outputs.get('iou', None)  # Raw IoU logits (optional)

        # Split-stride: use stride-4 heatmap for peak extraction if available
        hm_s4 = outputs.get('hm_s4')
        use_s4 = hm_s4 is not None
        if use_s4:
            heatmap_for_peaks = torch.sigmoid(hm_s4)
            # Ratio to convert stride-4 coordinates → det_stride coordinates
            s4_to_det = self.det_stride / 4
        else:
            heatmap_for_peaks = heatmap
            s4_to_det = 1.0

        B, C, H, W = heatmap.shape
        _, _, H_pk, W_pk = heatmap_for_peaks.shape

        batch_results = []

        # MaxPool peak extraction: suppress non-local-maxima
        # 3x3 at stride 4 = 12px radius, 5x5 = 20px (standard CenterNet/FairMOT)
        if use_max_pool_nms:
            pad = maxpool_kernel // 2
            hmax = F.max_pool2d(heatmap_for_peaks, maxpool_kernel, stride=1, padding=pad)
            keep = (hmax == heatmap_for_peaks).float()
            heatmap_for_peaks = heatmap_for_peaks * keep

        for b in range(B):
            hm = heatmap_for_peaks[b]  # [C, H_pk, W_pk]

            # Top-K peak extraction
            hm_flat = hm.view(-1)
            topk_scores, topk_inds = torch.topk(hm_flat, min(K, hm_flat.size(0)))

            # Filter by confidence
            mask = topk_scores > conf_thresh
            topk_scores = topk_scores[mask]
            topk_inds = topk_inds[mask]

            if len(topk_scores) == 0:
                empty_result = {
                    'boxes': torch.zeros((0, 4), device=heatmap.device),
                    'scores': torch.zeros(0, device=heatmap.device),
                }
                if return_embeddings:
                    empty_result['embeddings'] = torch.zeros((0, self.embedding_dim), device=heatmap.device)
                if return_part_features and self.reid_extraction == 'part_based' and self.part_extractor is not None:
                    empty_result['part_embeddings'] = torch.zeros(
                        (0, self.part_extractor.num_parts, self.embedding_dim),
                        device=heatmap.device
                    )
                    empty_result['attention_weights'] = torch.zeros(
                        (0, self.part_extractor.num_parts),
                        device=heatmap.device
                    )
                batch_results.append(empty_result)
                continue

            topk_ys = (topk_inds % (H_pk * W_pk)) // W_pk   # Class-safe unravel
            topk_xs = topk_inds % W_pk

            # Map stride-4 peak coords → det_stride coords for WH/offset/emb
            if use_s4:
                det_ys = (topk_ys.float() / s4_to_det).long().clamp(0, H - 1)
                det_xs = (topk_xs.float() / s4_to_det).long().clamp(0, W - 1)
            else:
                det_ys = topk_ys
                det_xs = topk_xs

            # IoU-aware scoring: merge heatmap score with predicted IoU quality
            if iou_raw is not None:
                iou_scores = torch.sigmoid(iou_raw[b, 0, det_ys, det_xs])
                # Weighted geometric mean: score = hm^(1-iou_power) * iou^iou_power
                hm_power = 1.0 - iou_power
                topk_scores = (topk_scores ** hm_power) * (iou_scores ** iou_power)

            # Gather geometry from det-stride maps
            topk_wh = wh[b, :, det_ys, det_xs].T
            topk_offset = offset[b, :, det_ys, det_xs].T

            # Compute boxes with offset correction (sub-pixel refinement)
            # Use det-stride coordinates for box decoding (WH/offset are at det_stride)
            cx = det_xs.float() + topk_offset[:, 0]
            cy = det_ys.float() + topk_offset[:, 1]
            left_dist = topk_wh[:, 0].clamp(min=1e-4)
            t = topk_wh[:, 1].clamp(min=1e-4)
            r = topk_wh[:, 2].clamp(min=1e-4)
            b_dist = topk_wh[:, 3].clamp(min=1e-4)
            x1 = cx - left_dist
            y1 = cy - t
            x2 = cx + r
            y2 = cy + b_dist

            # Embedding extraction — configurable via self.reid_extraction
            # 'part_based': 4x1 horizontal strip RoI-Align with attention fusion
            # 'center': Bilinear interpolation at sub-pixel center (FairMOT-style)
            topk_emb = None
            topk_part_embeddings = None
            topk_attention_weights = None
            if return_embeddings:
                if embedding is None:
                    raise KeyError("decode_detections(return_embeddings=True) requires outputs['embedding']")
            # Map detection coordinates to embedding map coordinates.
            # det coords are in heatmap grid (det_stride), emb map is at reid_stride.
            # emb_coord = det_coord * (det_stride / reid_stride)
                det_to_emb = self._det_stride_ratio / self._reid_stride_ratio
                if self.reid_extraction == 'part_based' and self.part_extractor is not None and len(x1) > 0:
                # Build RoI boxes in embedding-map coordinates
                    roi_batch_idx = torch.zeros(len(x1), 1, device=x1.device)
                    roi_boxes = torch.stack([
                        roi_batch_idx.squeeze(1),
                        x1 * det_to_emb, y1 * det_to_emb,
                        x2 * det_to_emb, y2 * det_to_emb,
                    ], dim=1)
                    emb_map = embedding[b:b+1]  # [1, C, H_emb, W_emb]
                    topk_emb, topk_part_embeddings, topk_attention_weights = self.part_extractor(
                        emb_map,
                        roi_boxes,
                        spatial_scale=1.0,
                    )
                else:
                # Center-pixel bilinear interpolation (FairMOT-style)
                    emb_map = embedding[b:b+1]  # [1, C, H_emb, W_emb]
                    topk_emb = sample_feature_map_bilinear(
                        emb_map, cx.detach() * det_to_emb, cy.detach() * det_to_emb
                    )
                    topk_emb = F.normalize(topk_emb, p=2, dim=1, eps=1e-6)

            # Scale to input image coordinates
            if input_size is not None:
                in_h, in_w = input_size
                stride_x = in_w / W
                stride_y = in_h / H
                x1, y1, x2, y2 = x1 * stride_x, y1 * stride_y, x2 * stride_x, y2 * stride_y

            # Scale to original image coordinates (naive rescaling)
            if original_size is not None and input_size is not None:
                orig_h, orig_w = original_size
                in_h, in_w = input_size
                scale_x = orig_w / in_w
                scale_y = orig_h / in_h
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y

            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # Minimum box size filtering
            box_w = (x2 - x1).clamp(min=0)
            box_h = (y2 - y1).clamp(min=0)
            box_area = box_w * box_h
            size_mask = (box_area >= min_box_area) & (box_h >= min_box_height)
            if size_mask.sum() < len(boxes):
                boxes = boxes[size_mask]
                topk_scores = topk_scores[size_mask]
                if topk_emb is not None:
                    topk_emb = topk_emb[size_mask]
                if topk_part_embeddings is not None:
                    topk_part_embeddings = topk_part_embeddings[size_mask]
                if topk_attention_weights is not None:
                    topk_attention_weights = topk_attention_weights[size_mask]

            # Box-level Gaussian Soft-NMS
            if len(boxes) > 1:
                keep, kept_scores = soft_nms(boxes, topk_scores, sigma=nms_sigma)
                boxes = boxes[keep]
                topk_scores = kept_scores
                if topk_emb is not None:
                    topk_emb = topk_emb[keep]
                if topk_part_embeddings is not None:
                    topk_part_embeddings = topk_part_embeddings[keep]
                if topk_attention_weights is not None:
                    topk_attention_weights = topk_attention_weights[keep]

            result = {
                'boxes': boxes,
                'scores': topk_scores,
            }
            if return_embeddings:
                result['embeddings'] = topk_emb
            if return_part_features and topk_part_embeddings is not None and topk_attention_weights is not None:
                result['part_embeddings'] = topk_part_embeddings
                result['attention_weights'] = topk_attention_weights

            batch_results.append(result)

        return batch_results
