"""
BiFPN: Bidirectional Feature Pyramid Network.

Weighted bidirectional feature fusion with fast normalized attention,
based on EfficientDet (Tan et al., CVPR 2020).
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution - more efficient than standard conv.
    
    Uses GroupNorm(32) instead of BatchNorm for BS=2 stability, consistent
    with the head's GroupNorm convention (noisy BN stats at 2 samples).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, bias: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=False,
                                   padding_mode='replicate')
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # GroupNorm(32) for BS=2 stability — BN stats are noisy with 2 samples
        self.bn = nn.GroupNorm(min(32, out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNLayer(nn.Module):
    """
    Single BiFPN layer with bidirectional feature fusion.
    """
    def __init__(self, num_channels: int, num_levels: int = 5, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.num_levels = num_levels
        
        # Learnable weights for weighted fusion (fast normalized fusion)
        # Top-down weights: asymmetric init favoring lateral (finer) features.
        # softplus([2.0, 1.0]) → [2.13, 1.31] → normalize → [0.62, 0.38]
        # This preserves ~62% of P3's fine-grained detail in the initial
        # top-down pass (vs 50/50 which over-contaminates with coarse P4).
        # Weights are learnable — the network adjusts during training.
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([2.0, 1.0], dtype=torch.float32))
            for _ in range(num_levels - 1)
        ])
        
        # Bottom-up weights (3 inputs each - lateral + from below + skip from original)
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3, dtype=torch.float32))
            for _ in range(num_levels - 1)
        ])
        
        # Depthwise separable convs for feature processing
        self.td_convs = nn.ModuleList([
            DepthwiseSeparableConv(num_channels, num_channels)
            for _ in range(num_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            DepthwiseSeparableConv(num_channels, num_channels)
            for _ in range(num_levels - 1)
        ])
        
        # Learned downsample for bottom-up path (replaces MaxPool2d)
        # Strided DWSConv learns what to preserve, not just peaks
        self.downsample_convs = nn.ModuleList([
            DepthwiseSeparableConv(num_channels, num_channels, kernel_size=3,
                                  stride=2, padding=1)
            for _ in range(num_levels - 1)
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(features) == self.num_levels
        
        # Store original features for skip connections
        original = features
        
        # ============ Top-Down Path ============
        td_features = [None] * self.num_levels
        td_features[-1] = features[-1]  # Coarsest level unchanged
        
        for i in range(self.num_levels - 2, -1, -1):  # Coarser-to-finer
            # Upsample from coarser level (bilinear avoids block artifacts from nearest)
            feat_h, feat_w = features[i].shape[2:]
            upsampled = F.interpolate(td_features[i + 1], size=(feat_h, feat_w),
                                       mode='bilinear', align_corners=False)
            
            # Weighted fusion (softplus avoids dead-weight problem of relu)
            w = F.softplus(self.td_weights[i])
            w = w / (w.sum() + self.epsilon)
            
            fused = w[0] * features[i] + w[1] * upsampled
            td_features[i] = self.td_convs[i](fused)
        
        # ============ Bottom-Up Path ============
        bu_features = [None] * self.num_levels
        bu_features[0] = td_features[0]  # P3 from top-down
        
        for i in range(1, self.num_levels):  # Finer-to-coarser
            # Learned downsample from finer level (strided DWSConv)
            # Replaces MaxPool which had peak-bias unsuitable for detection features
            feat_h, feat_w = td_features[i].shape[2:]
            downsampled = self.downsample_convs[i - 1](bu_features[i - 1])
            # Handle size mismatch (may be ±1 pixel from exact target)
            if downsampled.shape[2:] != (feat_h, feat_w):
                downsampled = F.adaptive_avg_pool2d(downsampled, output_size=(feat_h, feat_w))
            
            # Weighted fusion (3 inputs: td_feature, downsampled, original)
            w = F.softplus(self.bu_weights[i - 1])
            w = w / (w.sum() + self.epsilon)
            
            fused = w[0] * td_features[i] + w[1] * downsampled + w[2] * original[i]
            bu_features[i] = self.bu_convs[i - 1](fused)
        
        return bu_features


class BiFPN(nn.Module):
    """
    BiFPN: Bi-directional Feature Pyramid Network
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_layers: int = 2,
        num_levels: int = 5
    ):
        super().__init__()
        self.num_levels = num_levels
        self.out_channels = out_channels
        
        # Input projection convs (align channels from backbone)
        # GroupNorm(32) for BS=2 stability
        self.input_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.input_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 1, bias=False),
                    nn.GroupNorm(min(32, out_channels), out_channels),
                    nn.SiLU(inplace=True)
                )
            )
        
        # Extra levels (P6, P7) from P5
        num_extra = num_levels - len(in_channels_list)
        self.extra_convs = nn.ModuleList()
        for i in range(num_extra):
            in_ch = in_channels_list[-1] if i == 0 else out_channels
            self.extra_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 3, stride=2, padding=1,
                              padding_mode='replicate', bias=False),
                    nn.GroupNorm(min(32, out_channels), out_channels),
                    nn.SiLU(inplace=True)
                )
            )
        
        # BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, num_levels)
            for _ in range(num_layers)
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Project backbone features to unified channels
        projected = [conv(f) for conv, f in zip(self.input_convs, features)]
        
        # Generate extra levels (P6, P7)
        # Generate extra levels if needed (empty when num_levels == num_inputs)
        extra = None
        for i, conv in enumerate(self.extra_convs):
            if i == 0:
                extra = conv(features[-1])  # First extra from coarsest backbone level
            else:
                extra = conv(extra)  # Chain subsequent extra levels
            projected.append(extra)
        
        # Apply BiFPN layers
        for bifpn_layer in self.bifpn_layers:
            projected = bifpn_layer(projected)
        
        return projected
