"""
Checkpoint metadata helpers for runtime configuration compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

MODEL_META_KEYS = (
    "reid_extraction", "embedding_dim", "num_parts_v", "num_parts_h",
    "use_dcn", "use_iou_branch", "reid_head_depth",
    "use_decoupled_reid", "attention_type",
    "use_group_norm", "reid_head_use_dcn",
    "use_learned_upsample", "num_sharpening_dcn",
    "use_bifpn", "fpn_out_channels", "fusion_mode", "p3_refiner_depth",
    "shared_det_head", "reid_stride", "det_stride", "fusion_stride",
    "use_coord_attention", "part_attention_dropout_p", "part_fusion_type",
    "use_p2_heatmap", "drop_path_rate", "mix_style_p", "mix_style_alpha",
)


def build_model_meta(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build normalized model metadata from a model config dictionary."""
    return {
        "reid_extraction": model_cfg.get("reid_extraction", "center"),
        "embedding_dim": int(model_cfg.get("embedding_dim", 256)),
        "num_parts_v": int(model_cfg.get("num_parts_v", 4)),
        "num_parts_h": int(model_cfg.get("num_parts_h", 1)),
        "use_dcn": bool(model_cfg.get("use_dcn", False)),
        "use_iou_branch": bool(model_cfg.get("use_iou_branch", False)),
        "reid_head_depth": int(model_cfg.get("reid_head_depth", 3)),
        "use_decoupled_reid": bool(model_cfg.get("use_decoupled_reid", False)),
        "attention_type": str(model_cfg.get("attention_type", "se")),  # SE hardcoded in AdaptiveFeatureFusion
        "use_group_norm": bool(model_cfg.get("use_group_norm", False)),
        "reid_head_use_dcn": model_cfg.get("reid_head_use_dcn"),  # None means fall back to use_dcn
        "use_learned_upsample": bool(model_cfg.get("use_learned_upsample", False)),
        "num_sharpening_dcn": int(model_cfg.get("num_sharpening_dcn", 0)),
        "use_bifpn": bool(model_cfg.get("use_bifpn", True)),
        "fpn_out_channels": int(model_cfg.get("fpn_out_channels", 256)),
        "fusion_mode": str(model_cfg.get("fusion_mode", "weighted")),
        "p3_refiner_depth": int(model_cfg.get("p3_refiner_depth", 1)),
        "shared_det_head": bool(model_cfg.get("shared_det_head", False)),
        "reid_stride": int(model_cfg.get("reid_stride", 4)),
        "det_stride": int(model_cfg.get("det_stride", 4)),
        "fusion_stride": int(model_cfg.get("fusion_stride", 4)),
        "use_coord_attention": bool(model_cfg.get("use_coord_attention", True)),
        "part_attention_dropout_p": float(model_cfg.get(
            "part_attention_dropout_p", model_cfg.get("part_dropout_p", 0.0)
        )),
        "part_fusion_type": str(model_cfg.get("part_fusion_type", "attention_sum")),
        "use_p2_heatmap": bool(model_cfg.get("use_p2_heatmap", False)),
        "drop_path_rate": float(model_cfg.get("drop_path_rate", 0.0)),
        "mix_style_p": float(model_cfg.get("mix_style_p", 0.0)),
        "mix_style_alpha": float(model_cfg.get("mix_style_alpha", 0.1)),
    }


def get_checkpoint_model_meta(checkpoint: Any) -> Optional[Dict[str, Any]]:
    """Extract checkpoint metadata if present."""
    if not isinstance(checkpoint, dict):
        return None
    model_meta = checkpoint.get("model_meta")
    if not isinstance(model_meta, dict):
        return None
    return {k: model_meta.get(k) for k in MODEL_META_KEYS}


def compare_model_meta(
    runtime_meta: Dict[str, Any],
    checkpoint_meta: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return mismatched metadata fields."""
    mismatches = {}
    for key in MODEL_META_KEYS:
        ckpt_val = checkpoint_meta.get(key, None)
        if ckpt_val is None:
            continue
        run_val = runtime_meta.get(key, None)
        if run_val != ckpt_val:
            mismatches[key] = {"runtime": run_val, "checkpoint": ckpt_val}
    return mismatches


def ensure_checkpoint_compatibility(
    runtime_meta: Dict[str, Any],
    checkpoint: Any,
    checkpoint_path: str,
    allow_mismatch: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Validate runtime model config against checkpoint metadata.

    Raises ValueError on mismatch unless allow_mismatch=True.
    Returns checkpoint metadata when available.
    """
    checkpoint_meta = get_checkpoint_model_meta(checkpoint)
    if checkpoint_meta is None:
        print(
            f"[ModelMeta] WARNING: no model_meta found in checkpoint: {checkpoint_path}. "
            "Compatibility checks are skipped."
        )
        return None

    mismatches = compare_model_meta(runtime_meta, checkpoint_meta)
    if not mismatches:
        return checkpoint_meta

    details = ", ".join(
        f"{k}(cfg={v['runtime']}, ckpt={v['checkpoint']})"
        for k, v in mismatches.items()
    )
    msg = (
        f"[ModelMeta] Runtime config is incompatible with checkpoint '{checkpoint_path}': "
        f"{details}"
    )
    if allow_mismatch:
        print(f"{msg}. Proceeding due to override flag.")
        return checkpoint_meta

    raise ValueError(f"{msg}. Use --allow-checkpoint-mismatch to override.")
