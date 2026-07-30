"""Inference runtime backed by the installed :mod:`effiped` package."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import torch
import torch.nn.functional as F
import yaml

from .checkpoint_meta import build_model_meta, ensure_checkpoint_compatibility
from .dataset import letterbox
from .manifest import load_manifest
from .model import build_jdenet_from_config
from .settings import RuntimeSettings


@dataclass(frozen=True)
class RuntimePreset:
    key: str
    label: str
    config_path: Path
    checkpoint_path: Path
    description: str
    fold: int
    readout: str
    descriptor_dim: int
    artifact_version: str
    benchmark: dict
    retrieval_mode: str = "fused"
    return_part_features: bool = False

    @property
    def available(self) -> bool:
        return self.config_path.is_file() and self.checkpoint_path.is_file()


def default_presets(settings: RuntimeSettings | None = None) -> Dict[str, RuntimePreset]:
    """Resolve the two release artifacts without exposing local paths."""
    settings = settings or RuntimeSettings.from_env()
    _manifest, artifacts = load_manifest()
    return {
        key: RuntimePreset(
            key=artifact.key,
            label=artifact.label,
            config_path=artifact.config_path(),
            checkpoint_path=artifact.checkpoint_path(settings),
            description=artifact.description,
            fold=artifact.fold,
            readout=artifact.readout,
            descriptor_dim=artifact.descriptor_dim,
            artifact_version=artifact.artifact_version,
            benchmark=artifact.benchmark,
        )
        for key, artifact in artifacts.items()
    }


def available_model_choices(presets: Dict[str, RuntimePreset]) -> List[str]:
    """Return dropdown labels, marking missing optional checkpoints clearly."""
    labels = []
    for preset in presets.values():
        suffix = "" if preset.available else " (checkpoint missing)"
        labels.append(f"{preset.label}{suffix}")
    return labels


def preset_from_label(presets: Dict[str, RuntimePreset], label: str) -> RuntimePreset:
    clean_label = label.replace(" (checkpoint missing)", "")
    for preset in presets.values():
        if preset.label == clean_label:
            return preset
    raise KeyError(f"Unknown model preset: {label}")


def preset_from_key(presets: Dict[str, RuntimePreset], key: str) -> RuntimePreset:
    if key in presets:
        return presets[key]
    raise KeyError(f"Unknown model preset key: {key}")


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_contest_model(
    preset: RuntimePreset,
    device: torch.device,
    allow_checkpoint_mismatch: bool = False,
):
    """Build the contest JDENet model and load an authorized checkpoint."""
    if not preset.config_path.is_file():
        raise FileNotFoundError(f"Missing config: {preset.config_path}")
    if not preset.checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {preset.checkpoint_path}")

    config = load_yaml(preset.config_path)
    model = build_jdenet_from_config(config, pretrained=False)

    checkpoint = _torch_load(preset.checkpoint_path)
    ensure_checkpoint_compatibility(
        runtime_meta=build_model_meta(config.get("model", {})),
        checkpoint=checkpoint,
        checkpoint_path=str(preset.checkpoint_path),
        allow_mismatch=allow_checkpoint_mismatch,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_state and getattr(v, "shape", None) == model_state[k].shape
    }
    model.load_state_dict(filtered, strict=False)

    criterion_state = checkpoint.get("criterion_state_dict") if isinstance(checkpoint, dict) else None
    if criterion_state and hasattr(model.head, "migrate_reid_necks_from_criterion_state"):
        model.head.migrate_reid_necks_from_criterion_state(criterion_state)

    model.to(device)
    model.eval()
    return model, config, checkpoint


def scale_coords(
    input_shape: Tuple[int, int],
    coords: torch.Tensor,
    original_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[float, Tuple[float, float]]] = None,
) -> torch.Tensor:
    """Rescale xyxy coordinates from letterboxed input size to original frame."""
    if coords.numel() == 0:
        return coords

    in_h, in_w = input_shape
    orig_h, orig_w = original_shape
    if ratio_pad is None:
        gain = min(in_h / orig_h, in_w / orig_w)
        pad = ((in_w - orig_w * gain) / 2.0, (in_h - orig_h * gain) / 2.0)
    else:
        gain, pad = ratio_pad

    coords = coords.clone()
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= max(gain, 1e-12)
    coords[:, 0].clamp_(0, orig_w)
    coords[:, 1].clamp_(0, orig_h)
    coords[:, 2].clamp_(0, orig_w)
    coords[:, 3].clamp_(0, orig_h)
    return coords


class EffiPedRuntime:
    """Lazy EffiPed runtime for frame and batch inference."""

    def __init__(self, settings: RuntimeSettings | None = None):
        settings = settings or RuntimeSettings.from_env()
        self.model = None
        self.config = None
        self.checkpoint = None
        self.preset_key = None
        if settings.device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = settings.device
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("EFFIPED_DEVICE requests CUDA, but CUDA is unavailable.")
        self.device = torch.device(device_name)
        self.img_size = (1088, 608)
        self.mean = None
        self.std = None

    def ensure_loaded(self, preset: RuntimePreset):
        if self.model is not None and self.preset_key == preset.key:
            return
        self.model, self.config, self.checkpoint = load_contest_model(preset, self.device)
        self.preset_key = preset.key
        self.img_size = tuple(self.config.get("data", {}).get("img_size", [1088, 608]))
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def tracker_config(self) -> dict:
        cfg = dict((self.config or {}).get("tracker", {}))
        cfg.update({
            "low_thresh": min(float(cfg.get("low_thresh", 0.1)), 0.05),
            "det_thresh": float(cfg.get("det_thresh", 0.30)),
            "new_track_thresh": float(cfg.get("new_track_thresh", 0.40)),
            "track_buffer": int(cfg.get("track_buffer", 30)),
            "use_botsort": True,
            "use_gallery": True,
            "appearance_thresh": cfg.get("appearance_thresh", 0.60),
            "embedding_update_thresh": cfg.get("embedding_update_thresh", 0.35),
        })
        return cfg

    def _preprocess(self, frame_bgr):
        target_w, target_h = self.img_size
        orig_h, orig_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img, ratio, (pad_w, pad_h) = letterbox(frame_rgb, new_shape=(target_w, target_h))
        blob = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        blob = (blob - self.mean) / self.std
        return blob, (orig_h, orig_w), ratio, (pad_w, pad_h), (target_h, target_w)

    def process_frame(
        self,
        frame_bgr,
        *,
        conf_thresh: float = 0.05,
        topk: int = 300,
        min_box_area: float = 40.0,
        min_box_height: float = 10.0,
        nms_sigma: float = 0.35,
        return_part_features: bool = False,
    ):
        blob, orig_shape, ratio, pad, input_shape = self._preprocess(frame_bgr)
        start = time.time()
        with torch.no_grad():
            outputs = self.model(blob)
            dets = self.model.head.decode_detections(
                outputs,
                K=int(topk),
                conf_thresh=float(conf_thresh),
                input_size=input_shape,
                original_size=None,
                nms_sigma=float(nms_sigma),
                min_box_area=float(min_box_area),
                min_box_height=float(min_box_height),
                return_part_features=bool(return_part_features),
            )[0]
            if hasattr(self.model.head, "apply_reid_necks"):
                dets = self.model.head.apply_reid_necks(dets)
            if len(dets.get("boxes", [])) > 0:
                dets["boxes"] = scale_coords(input_shape, dets["boxes"], orig_shape, (ratio, pad))
        return dets, time.time() - start

    def process_batch(
        self,
        frames: Dict[str, object],
        *,
        conf_thresh: float = 0.05,
        topk: int = 300,
        min_box_area: float = 40.0,
        min_box_height: float = 10.0,
        nms_sigma: float = 0.35,
        return_part_features: bool = False,
    ) -> Dict[str, Tuple[dict, float]]:
        if not frames:
            return {}

        target_w, target_h = self.img_size
        cam_ids = list(frames.keys())
        blobs = []
        meta = {}
        for cam_id in cam_ids:
            frame_bgr = frames[cam_id]
            orig_h, orig_w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img, ratio, (pad_w, pad_h) = letterbox(frame_rgb, new_shape=(target_w, target_h))
            blob = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            blobs.append(blob)
            meta[cam_id] = ((orig_h, orig_w), ratio, (pad_w, pad_h))

        batch = torch.stack(blobs, dim=0).to(self.device)
        batch = (batch - self.mean) / self.std
        input_shape = (target_h, target_w)
        start = time.time()
        with torch.no_grad():
            outputs = self.model(batch)
            batch_dets = self.model.head.decode_detections(
                outputs,
                K=int(topk),
                conf_thresh=float(conf_thresh),
                input_size=input_shape,
                original_size=None,
                nms_sigma=float(nms_sigma),
                min_box_area=float(min_box_area),
                min_box_height=float(min_box_height),
                return_part_features=bool(return_part_features),
            )
            if hasattr(self.model.head, "apply_reid_necks"):
                batch_dets = [self.model.head.apply_reid_necks(d) for d in batch_dets]

        elapsed = (time.time() - start) / max(1, len(cam_ids))
        result = {}
        for cam_id, dets in zip(cam_ids, batch_dets):
            orig_shape, ratio, pad = meta[cam_id]
            if len(dets.get("boxes", [])) > 0:
                dets["boxes"] = scale_coords(input_shape, dets["boxes"], orig_shape, (ratio, pad))
            result[cam_id] = (dets, elapsed)
        return result


def detection_count(dets: dict) -> int:
    scores = dets.get("scores")
    return int(len(scores)) if scores is not None else 0


def iter_detection_rows(dets: dict) -> Iterable[Tuple[float, float, float, float, float]]:
    boxes = dets.get("boxes")
    scores = dets.get("scores")
    if boxes is None or scores is None or len(scores) == 0:
        return []
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    return [
        (float(x1), float(y1), float(x2), float(y2), float(score))
        for (x1, y1, x2, y2), score in zip(boxes, scores)
    ]


def normalize_embedding_array(x):
    if x is None:
        return None
    return F.normalize(x, p=2, dim=-1, eps=1e-6)
