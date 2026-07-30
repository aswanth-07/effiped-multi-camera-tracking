"""
ConvNeXt V2 backbone with configurable neck and stride-4 P2 fusion.

Extracts multi-scale features from a ConvNeXt V2 backbone.
Two modes:

  use_bifpn=True  (default):  P3+P4 BiFPN neck + P2 fusion
  use_bifpn=False (compact dual-scale):  P3-only refiner + P2 fusion

Compact dual-scale mode observation:
  At stride 16, each feature pixel covers a 16x16 area. Distant pedestrians
  (15x30 original pixels) collapse to <1x2 feature pixels -- they cease to exist
  as distinct objects. P4 activations reach ~180 max magnitude vs P3's ~51,
  overwhelming fine-grained P3 features during BiFPN top-down fusion.
  Removing P4 eliminates this magnitude intoxication AND saves ~30% VRAM
  (no stage-2 backprop, no BiFPN layers, no 384-ch activation maps).

ConvNeXt V2 Tiny feature stages:
  P2: 96 channels,  stride 4  (high-resolution, bypasses BiFPN)
  P3: 192 channels, stride 8
  P4: 384 channels, stride 16 (excluded when use_bifpn=False)
  (P5: 768ch, stride 32 -- always excluded)
"""

import random

import timm
import torch
import torch.nn as nn

from .fusion import AdaptiveFeatureFusion
from .neck import BiFPN, DepthwiseSeparableConv


class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR 2021) — feature statistics mixing for domain augmentation.

    Mixes instance normalization statistics (mean/std) between random sample pairs
    within a batch. Since feature statistics encode camera-specific "style" (lighting,
    color tone, contrast), mixing synthesizes virtual cross-camera domains for free.

    Applied after early backbone stages; training-only (identity at eval).
    """
    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + 1e-6).sqrt()
        x_norm = (x - mu) / sig
        perm = torch.randperm(B, device=x.device)
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((B, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix


class ConvNeXtBackbone(nn.Module):
    """
    Feature Extraction Backbone (ConvNeXt V2).

    Two modes:
      use_bifpn=True:  Extract P2-P4, fuse P3-P4 via BiFPN, then P2+P3 adaptive fusion
      use_bifpn=False: Extract P2-P3 only, refine P3 with lightweight DWSConv,
                       then P2+P3 adaptive fusion. No P4 extracted -- saves
                       ~30% backbone FLOPS and eliminates magnitude intoxication.
    """
    def __init__(self, backbone='convnextv2_tiny', pretrained=True, img_size=None,
                 fpn_out_channels=256,
                 use_bifpn=True,
                 num_bifpn_layers=2,
                 use_learned_upsample=False,  # ConvTranspose2d instead of bilinear in fusion
                 num_sharpening_dcn=0,        # DCN layers after fusion for spatial sharpening
                 fusion_mode='weighted',       # 'weighted' (new) or 'concat' (legacy V1)
                 p3_refiner_depth=1,           # Number of DWSConv layers in P3 refiner (1=default)
                 fusion_stride=4,              # 4=fuse at stride-4 (default), 8=fuse at stride-8
                 mix_style_p=0.0,              # MixStyle probability (0=disabled)
                 mix_style_alpha=0.1,          # MixStyle Beta distribution alpha
                 **kwargs):
        super(ConvNeXtBackbone, self).__init__()

        extra_kwargs = kwargs.copy()
        
        # Pop img_size if it was hiding inside **kwargs from the config dict
        cfg_img_size = extra_kwargs.pop('img_size', None)
        if img_size is None:
            img_size = cfg_img_size

        # Pop legacy kwargs that may arrive from old configs
        extra_kwargs.pop('use_fpn', None)
        extra_kwargs.pop('use_attention_fpn', None)
        extra_kwargs.pop('se_reduction', None)
        extra_kwargs.pop('backbone_pretrained_path', None)

        # ── MixStyle: virtual domain augmentation (training-only) ──
        self.mix_style = MixStyle(p=mix_style_p, alpha=mix_style_alpha) if mix_style_p > 0 else None
        if self.mix_style is not None:
            print(f"  MixStyle: p={mix_style_p}, alpha={mix_style_alpha} (applied to P2+P3)")

        # ── Feature extraction: select which stages to extract ──
        self.use_bifpn = use_bifpn
        if use_bifpn:
            # P2 (stride 4) + P3 (stride 8) + P4 (stride 16)
            out_indices = (0, 1, 2)
        else:
            # Compact dual-scale mode: P2 (stride 4) + P3 (stride 8) only
            # Saves ~25% backbone FLOPS — no stage-2 forward/backward
            out_indices = (0, 1)

        try:
            self.model = timm.create_model(backbone, features_only=True, pretrained=pretrained,
                                           out_indices=out_indices, **extra_kwargs)
            print(f"  Loaded backbone: {backbone} (pretrained={pretrained})")
        except Exception as e:
            print(f"  Warning: Error loading pretrained weights: {e}. Loading random weights.")
            self.model = timm.create_model(backbone, features_only=True, pretrained=False,
                                           out_indices=out_indices, **extra_kwargs)

        self.feature_channels = self.model.feature_info.channels()
        self.p2_channels = self.feature_channels[0]
        self.out_channels = fpn_out_channels

        if use_bifpn:
            # ── BiFPN mode: P3+P4 multi-scale fusion ──
            bifpn_in_channels = list(self.feature_channels[1:])  # [192, 384]
            bifpn_num_levels = len(bifpn_in_channels)  # 2
            bifpn_num_layers = num_bifpn_layers
            self.fpn = BiFPN(
                in_channels_list=bifpn_in_channels,
                out_channels=fpn_out_channels,
                num_layers=bifpn_num_layers,
                num_levels=bifpn_num_levels
            )
            level_names = ['P3', 'P4'][:bifpn_num_levels]
            print(f"  BiFPN: {bifpn_num_layers} layers, {bifpn_num_levels} levels ({'-'.join(level_names)}), P2 bypass")
        else:
            # ── P3 Refiner mode: lightweight DWSConv on P3 only ──
            # 1×1 projection (192→256) + DWSConv refinement
            # Equivalent to BiFPN's input_conv + one conv layer, minus all the
            # P4 contamination and bidirectional weight complexity.
            p3_in_channels = self.feature_channels[1]  # 192
            self.fpn = None  # No BiFPN
            refiner_layers = [
                nn.Conv2d(p3_in_channels, fpn_out_channels, 1, bias=False),
                nn.GroupNorm(min(32, fpn_out_channels), fpn_out_channels),
                nn.SiLU(inplace=True),
            ]
            for _ in range(p3_refiner_depth):
                refiner_layers.append(
                    DepthwiseSeparableConv(fpn_out_channels, fpn_out_channels,
                                          kernel_size=3, padding=1)
                )
            self.p3_refiner = nn.Sequential(*refiner_layers)
            depth_str = f"{p3_refiner_depth}xDWSConv" if p3_refiner_depth > 1 else "DWSConv"
            print(f"  P3 Refiner: {p3_in_channels}->{fpn_out_channels}ch (1x1+GN+SiLU+{depth_str}), no P4/BiFPN")

        # ── Stride-4 Adaptive Fusion: P2 + refined P3 ──
        # BiFPN-style learned weighted addition with dual GroupNorm
        # Optional: learned upsample (ConvTranspose2d) + DCN sharpening
        fusion_levels = 1  # Only P3 (always)
        self.adaptive_fusion = AdaptiveFeatureFusion(
            p2_channels=self.p2_channels,
            bifpn_channels=fpn_out_channels,
            head_channels=fpn_out_channels,
            num_bifpn_levels=fusion_levels,
            use_learned_upsample=use_learned_upsample,
            num_sharpening_dcn=num_sharpening_dcn,
            fusion_mode=fusion_mode,
            fusion_stride=fusion_stride,
        )
        upsample_str = "ConvTranspose2d" if use_learned_upsample else "bilinear"
        if fusion_stride == 8:
            upsample_str = "none(P2↓2)"
        dcn_str = f", +{num_sharpening_dcn}xDCN" if num_sharpening_dcn > 0 else ""
        mode_str = "" if use_bifpn else " (compact dual-scale)"
        print(f"  P2 Fusion: {self.p2_channels}ch P2 + P3({fpn_out_channels}ch) -> {fpn_out_channels}ch (upsample={upsample_str}{dcn_str}) @ stride {fusion_stride}{mode_str}")

    def forward(self, x):
        features = self.model(x)

        # P2 bypass: always extract P2 for stride-4 fusion
        p2 = features[0]

        # MixStyle: mix feature statistics to synthesize virtual camera domains
        if self.mix_style is not None:
            p2 = self.mix_style(p2)
            features = list(features)
            features[1] = self.mix_style(features[1])

        if self.use_bifpn:
            # BiFPN mode: P3+P4 → BiFPN → take finest level (P3)
            bifpn_inputs = features[1:]  # [P3, P4]
            fpn_features = self.fpn(bifpn_inputs)
            refined_p3 = fpn_features[:1]  # [P3_fused] — P4 context via top-down
        else:
            # P3-only refiner: no P4, no BiFPN
            p3 = features[1]
            refined_p3 = [self.p3_refiner(p3)]  # [P3_refined]

        # Adaptive fusion: raw P2 + refined P3 → stride-4 output
        fused = self.adaptive_fusion(p2, refined_p3)
        return fused, p2


