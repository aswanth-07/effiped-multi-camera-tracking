"""
EffiPed joint detection and embedding network.

ConvNeXt V2 backbone + configurable neck + P2 Fusion + CenterNet head
with part-based ReID at stride 4.

Two neck modes:
  use_bifpn=True:  BiFPN on P3+P4 (default, current training)
  use_bifpn=False: P3-only refiner, no P4 extracted (lighter, cleaner features)
"""

import torch.nn as nn

from .models.backbone import ConvNeXtBackbone
from .models.head import CenterNetHead


def resolve_part_attention_dropout_p(model_cfg):
    """Canonical fusion-attention dropout with backward-compatible alias."""
    if model_cfg is None:
        return 0.0
    return float(model_cfg.get('part_attention_dropout_p',
                               model_cfg.get('part_dropout_p', 0.0)))


def resolve_eval_use_bnneck(config, override='auto'):
    """Resolve eval-time BNNeck policy from CLI override plus config."""
    if override in (True, 'on', 'true', 'True', '1', 1):
        return True
    if override in (False, 'off', 'false', 'False', '0', 0):
        return False
    return bool(config.get('model', {}).get('eval_use_bnneck', True))


def build_jdenet_from_config(config, *, pretrained=None):
    """Build JDENet from a YAML config using the same surface everywhere."""
    model_cfg = config['model']
    loss_cfg = config.get('loss', {})
    data_cfg = config.get('data', {})
    img_size = data_cfg.get('img_size')
    img_hw = None
    if img_size is not None:
        w, h = img_size
        img_hw = (h, w)
    if pretrained is None:
        pretrained = model_cfg.get('pretrained', True)

    return JDENet(
        backbone=model_cfg['backbone'],
        embedding_dim=model_cfg['embedding_dim'],
        img_size=img_hw,
        pretrained=pretrained,
        use_bifpn=model_cfg.get('use_bifpn', True),
        fpn_out_channels=model_cfg.get('fpn_out_channels', 256),
        use_decoupled_reid=model_cfg.get('use_decoupled_reid', False),
        reid_head_depth=model_cfg.get('reid_head_depth', 3),
        use_dcn=model_cfg.get('use_dcn', False),
        reid_head_use_dcn=model_cfg.get('reid_head_use_dcn', None),
        use_iou_branch=model_cfg.get('use_iou_branch', False),
        num_bifpn_layers=model_cfg.get('num_bifpn_layers', 2),
        reid_extraction=model_cfg.get('reid_extraction', 'center'),
        num_parts_v=model_cfg.get('num_parts_v', 4),
        num_parts_h=model_cfg.get('num_parts_h', 1),
        roi_output_size=tuple(model_cfg.get('roi_output_size', [32, 8])),
        use_group_norm=model_cfg.get('use_group_norm', False),
        use_learned_upsample=model_cfg.get('use_learned_upsample', False),
        num_sharpening_dcn=model_cfg.get('num_sharpening_dcn', 0),
        fusion_mode=model_cfg.get('fusion_mode', 'weighted'),
        p3_refiner_depth=model_cfg.get('p3_refiner_depth', 1),
        shared_det_head=model_cfg.get('shared_det_head', False),
        reid_stride=model_cfg.get('reid_stride', 4),
        det_stride=model_cfg.get('det_stride', 4),
        fusion_stride=model_cfg.get('fusion_stride', 4),
        use_p2_heatmap=model_cfg.get('use_p2_heatmap', False),
        part_fusion_type=model_cfg.get('part_fusion_type', 'attention_sum'),
        drop_path_rate=model_cfg.get('drop_path_rate', 0.0),
        mix_style_p=model_cfg.get('mix_style_p', 0.0),
        mix_style_alpha=model_cfg.get('mix_style_alpha', 0.1),
        use_coord_attention=model_cfg.get('use_coord_attention', True),
        use_reid_bnneck=loss_cfg.get('use_bnneck', True),
        part_attention_dropout_p=resolve_part_attention_dropout_p(model_cfg),
    )


class JDENet(nn.Module):
    """
    EffiPed model: ConvNeXt V2 + BiFPN/P3 refiner + P2 fusion + CenterNet + ReID.

    Architecture:
      - Backbone: ConvNeXt V2 Tiny with out_indices=(0,1,2) for P2-P4
      - Neck: BiFPN on P3-P4 only (P2 bypasses BiFPN, P5 excluded)
      - Fusion: Adaptive P2 + BiFPN fusion at stride 4 via ECA/SE + 1x1 conv
      - Head: CenterNet (heatmap + WH + offset + IoU + ReID embedding) at stride 4
      - Optional: decoupled ReID, DCNv2, IoU quality branch
    """
    def __init__(self, backbone='convnextv2_tiny', embedding_dim=256, img_size=None,
                 pretrained=True,
                 fpn_out_channels=256,
                 use_bifpn=True,
                 use_decoupled_reid=False,

                 reid_head_depth=3,
                 use_dcn=False,
                 reid_head_use_dcn=None,
                 use_iou_branch=False,
                 num_bifpn_layers=2,
                 reid_extraction='center',
                 num_parts_v=4,
                 num_parts_h=1,
                 roi_output_size=(32, 8),
                 use_group_norm=False,
                 use_learned_upsample=False,
                 num_sharpening_dcn=0,
                 fusion_mode='weighted',
                 p3_refiner_depth=1,
                 shared_det_head=False,
                 reid_stride=4,
                 det_stride=4,
                 fusion_stride=4,
                 use_p2_heatmap=False,
                 part_fusion_type='attention_sum',
                 drop_path_rate=0.0,
                 mix_style_p=0.0,
                 mix_style_alpha=0.1,
                 use_coord_attention=True,
                 use_reid_bnneck=True,
                 part_attention_dropout_p=None,
                 part_dropout_p=0.0,
                 **kwargs):
        super(JDENet, self).__init__()
        self.use_p2_heatmap = use_p2_heatmap
        part_attention_dropout_p = (
            part_dropout_p if part_attention_dropout_p is None
            else part_attention_dropout_p
        )

        # Backbone with BiFPN (P3-P4) + P2 bypass fusion
        self.backbone = ConvNeXtBackbone(
            backbone=backbone,
            img_size=img_size,
            pretrained=pretrained,
            fpn_out_channels=fpn_out_channels,
            use_bifpn=use_bifpn,
            num_bifpn_layers=num_bifpn_layers,
            use_learned_upsample=use_learned_upsample,
            num_sharpening_dcn=num_sharpening_dcn,
            fusion_mode=fusion_mode,
            p3_refiner_depth=p3_refiner_depth,
            fusion_stride=fusion_stride,
            drop_path_rate=drop_path_rate,
            mix_style_p=mix_style_p,
            mix_style_alpha=mix_style_alpha,
        )

        # CenterNet Head (operates at fusion_stride)
        self.head = CenterNetHead(
            in_channels=self.backbone.out_channels,
            num_classes=1,
            embedding_dim=embedding_dim,
            head_channels=256,
            num_convs=2,
            use_decoupled_reid=use_decoupled_reid,

            reid_head_depth=reid_head_depth,
            use_dcn=use_dcn,
            reid_head_use_dcn=reid_head_use_dcn,
            use_iou_branch=use_iou_branch,
            reid_extraction=reid_extraction,
            num_parts_v=num_parts_v,
            num_parts_h=num_parts_h,
            roi_output_size=roi_output_size,
            use_group_norm=use_group_norm,
            shared_det_head=shared_det_head,
            reid_stride=reid_stride,
            det_stride=det_stride,
            input_stride=fusion_stride,
            use_p2_heatmap=use_p2_heatmap,
            p2_channels=self.backbone.p2_channels,
            part_fusion_type=part_fusion_type,
            use_coord_attention=use_coord_attention,
            use_reid_bnneck=use_reid_bnneck,
            part_attention_dropout_p=part_attention_dropout_p,
        )

        bifpn_str = f"BiFPN(P3-P4, {num_bifpn_layers}L)" if use_bifpn else "P3 Refiner (no P4)"
        det_str = f'Stride {det_stride}' if det_stride != 4 else 'Stride 4'
        print(f"  EffiPed JDENet: {backbone} -> {bifpn_str} -> P2 Fusion -> CenterNetHead @ {det_str}")
        norm_str = 'GN32' if use_group_norm else 'BN'
        effective_dcn = use_dcn if reid_head_use_dcn is None else reid_head_use_dcn
        print(f"    ReID: DCN={effective_dcn} (det_trunk={use_dcn}), depth={reid_head_depth}, extraction={reid_extraction}, norm={norm_str}")

    def forward(self, x):
        fused, raw_p2 = self.backbone(x)
        return self.head(fused, raw_p2=raw_p2 if self.use_p2_heatmap else None)
