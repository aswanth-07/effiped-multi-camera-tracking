"""
Shared building blocks for EffiPed model components.

Contains reusable modules used across backbone, neck, fusion, and head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_grid_axis(coords: torch.Tensor, size: int) -> torch.Tensor:
    """Map feature-map pixel coordinates to [-1, 1] grid-sample coordinates."""
    if size <= 1:
        return torch.zeros_like(coords, dtype=torch.float32)
    return (2.0 * coords / float(size - 1)) - 1.0


def sample_feature_map_bilinear(
    feature_map: torch.Tensor,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    batch_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Bilinearly sample per-instance embeddings from a [B, C, H, W] feature map.

    Args:
        feature_map: Feature tensor shaped [B, C, H, W].
        x_coords: X coordinates in feature-map pixels, shape [N].
        y_coords: Y coordinates in feature-map pixels, shape [N].
        batch_indices: Optional batch index per point, shape [N]. Defaults to 0.

    Returns:
        Sampled features shaped [N, C].
    """
    if feature_map.ndim != 4:
        raise ValueError(f"Expected feature_map to be [B, C, H, W], got {tuple(feature_map.shape)}")

    batch_size, channels, height, width = feature_map.shape
    if x_coords.numel() == 0:
        return feature_map.new_empty((0, channels))

    device = feature_map.device
    x_coords = x_coords.to(device=device, dtype=torch.float32)
    y_coords = y_coords.to(device=device, dtype=torch.float32)

    if batch_indices is None:
        batch_indices = torch.zeros(x_coords.shape[0], device=device, dtype=torch.long)
    else:
        batch_indices = batch_indices.to(device=device, dtype=torch.long)

    if batch_indices.numel() != x_coords.numel():
        raise ValueError("batch_indices must have the same length as x_coords/y_coords")
    if batch_size <= 0:
        raise ValueError("feature_map batch dimension must be positive")

    sampled_features = feature_map.new_empty((x_coords.shape[0], channels))
    x_coords = x_coords.clamp(0.0, max(width - 1, 0))
    y_coords = y_coords.clamp(0.0, max(height - 1, 0))

    unique_batches = torch.unique(batch_indices, sorted=True).tolist()
    for batch_idx in unique_batches:
        if batch_idx < 0 or batch_idx >= batch_size:
            raise IndexError(f"batch_indices contains {batch_idx}, but batch size is {batch_size}")
        mask = batch_indices == batch_idx
        grid_x = _normalize_grid_axis(x_coords[mask], width)
        grid_y = _normalize_grid_axis(y_coords[mask], height)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)
        sampled = F.grid_sample(
            feature_map[batch_idx:batch_idx + 1],
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        sampled_features[mask] = sampled.squeeze(0).squeeze(1).transpose(0, 1)

    return sampled_features


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention with FC bottleneck.

    Unlike ECA's local 1D conv (k=9 sees only 9 neighbors), SE's FC layers
    model global cross-channel dependencies — critical when the channel
    dimension concatenates features from heterogeneous sources (P2+P3+P4).

    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )
        # Xavier init: sigmoid output ≈ 0.5 on average at start (near-identity scaling)
        nn.init.xavier_uniform_(self.fc[0].weight)
        nn.init.xavier_uniform_(self.fc[2].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


