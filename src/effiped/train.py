"""
effiped Training Script — Joint Detection + ReID Training
═══════════════════════════════════════════════════════════════════

Table of Contents (approximate line numbers):
─────────────────────────────────────────────
  L35    Imports & Constants
  L74    Helpers (set_seed, _concat_embedding, _compute_same_cam_r1)
  L229   Validation / Evaluation loop (evaluate)
  L410     BNNeck applied at eval time (aligns eval↔train spaces)
  L632     ReID Rank-1: per-camera gallery matching
  L1058  Main training function (train)
  L1076    Experiment config snapshot + CSV init
  L1103    Dataset initialization
  L1183    Model creation (JDENet)
  L1247    Loss + criterion setup
  L1285    Optimizer + scheduler
  L1350    Checkpoint resume logic
  L1497    Epoch training loop
  L1663      Epoch-end: scheduler step + metrics
  L1681      Validation + CSV logging
  L1693      Combined score + best model save
  L1744    Regular checkpoint save
  L1774  Entry point (__main__)

Key Invariants:
  • Loss returns 8-tuple: (total, hm, reg, id, acc, rep, iou, dcn)
  • EMA model is used for validation (if enabled)
  • Mid-epoch checkpoints save at configurable batch intervals
  • Config YAML is saved as snapshot to experiment folder on each run
  • Combined score = 0.3 * mAP@[.5:.95] + 0.2 * mAR@100 + 0.5 * Rank-1
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import glob
import json
import os
import shutil
import csv
import yaml
import torch
import random
import numpy as np
import itertools
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn.functional as F

from effiped.model import build_jdenet_from_config, resolve_eval_use_bnneck
from effiped.loss import CenterNetLoss
from effiped.ema import ModelEMA
from effiped.dataset import MOT17Dataset, collate_fn, SequenceGroupedSampler, IdentityAwareSampler
from effiped.checkpoint_meta import build_model_meta
from effiped.models.common import sample_feature_map_bilinear


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    """Seed all RNGs (Python, NumPy, PyTorch) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _csv_header():
    return [
        'Epoch', 'LR', 'Train_Loss', 'Train_Cls', 'Train_Reg', 'Train_ID',
        'Train_IoU', 'Train_Rep', 'Train_DCNPen', 'Train_Distill',
        'Train_FeatDistill', 'Train_Acc', 'Part_Attn_Entropy_Raw',
        'Part_Collapse_Rate_Raw', 'Part_Attn_Entropy_Fused',
        'Part_Collapse_Rate_Fused', 'Tiny_RoI_Rate', 'Val_mAP',
        'Val_mAP_50', 'Val_HM_Reg', 'Val_ID_Acc_UNUSED', 'ReID_R1',
        'Val_Rec', 'R1_Same_Cam', 'R1_Cross_Cam', 'Val_mAR_50'
    ]


def _ensure_csv_header(path):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(_csv_header())


def _copy_if_exists(src, dst):
    if not src or not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _save_checkpoint(payload, primary_path, alias_paths=None):
    os.makedirs(os.path.dirname(primary_path), exist_ok=True)
    torch.save(payload, primary_path)
    for alias_path in alias_paths or []:
        os.makedirs(os.path.dirname(alias_path), exist_ok=True)
        shutil.copy2(primary_path, alias_path)


def _latest_existing_file(paths):
    existing = [p for p in paths if p and os.path.isfile(p)]
    if not existing:
        return None
    return max(existing, key=lambda p: os.path.getmtime(p))


def _find_auto_resume_checkpoint(save_dir):
    """Find the latest recoverable checkpoint for this experiment folder."""
    search_dirs = [
        save_dir,
        os.path.join(save_dir, 'checkpoints'),
    ]
    search_dirs.extend(glob.glob(os.path.join(save_dir, 'runs', '*', 'checkpoints')))

    rolling_candidates = []
    for d in search_dirs:
        rolling_candidates.extend(glob.glob(os.path.join(d, 'checkpoint_mid_epoch.pth')))
        rolling_candidates.extend(glob.glob(os.path.join(d, 'checkpoint_epoch_*.pth')))
    latest = _latest_existing_file(rolling_candidates)
    if latest:
        return latest

    best_candidates = []
    for d in search_dirs:
        best_candidates.append(os.path.join(d, 'best_model.pth'))
    return _latest_existing_file(best_candidates)


def _filter_model_state(ckpt_sd, model):
    """Load-compatible checkpoint state with lightweight key migration."""
    migrated_sd = {}
    for k, v in ckpt_sd.items():
        new_k = k
        if 'post_refine.' in k and 'post_refine.0.' not in k and 'post_refine.1.' not in k:
            new_k = k.replace('post_refine.', 'post_refine.0.')
        migrated_sd[new_k] = v

    model_state = model.state_dict()
    loaded = {
        k: v for k, v in migrated_sd.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped = [k for k in migrated_sd if k not in loaded]
    return loaded, skipped, migrated_sd


def _restore_criterion_state(criterion, model, criterion_state, prefix="Criterion"):
    """Restore compatible loss/head-adjacent state without requiring exact config parity."""
    crit_state = criterion.state_dict()
    crit_loaded = {
        k: v for k, v in criterion_state.items()
        if k in crit_state and v.shape == crit_state[k].shape
    }
    if len(crit_loaded) < len(criterion_state):
        skipped = len(criterion_state) - len(crit_loaded)
        skipped_keys = [k for k in criterion_state if k not in crit_loaded]
        arcface_dropped = [k for k in skipped_keys if 'arcface' in k or 'part_arcface' in k]
        if arcface_dropped:
            print(f"  WARNING: {prefix} ArcFace keys skipped due to shape mismatch")
            print(f"     Dropped: {arcface_dropped[:5]}...")
        else:
            print(f"  WARNING: {prefix}: {skipped} keys skipped (shape mismatch or removed)")
    criterion.load_state_dict(crit_loaded, strict=False)

    if hasattr(model.head, 'migrate_reid_necks_from_criterion_state'):
        migrated_bn = model.head.migrate_reid_necks_from_criterion_state(criterion_state)
        if migrated_bn:
            print(f"  Loaded {migrated_bn} BNNeck tensors into model.head from {prefix}")
    if hasattr(criterion, 'migrate_shared_part_classifier_from_state'):
        migrated_part_cls = criterion.migrate_shared_part_classifier_from_state(criterion_state)
        if migrated_part_cls:
            print(f"  Initialized shared part classifier from {migrated_part_cls} old per-part heads")

    if hasattr(model.head, 'part_extractor') and model.head.part_extractor is not None:
        pe_keys = {k: v for k, v in criterion_state.items() if k.startswith('part_extractor.')}
        if pe_keys:
            head_pe_state = model.head.part_extractor.state_dict()
            migrated = 0
            for k, v in pe_keys.items():
                short_k = k.replace('part_extractor.', '', 1)
                if short_k in head_pe_state and v.shape == head_pe_state[short_k].shape:
                    head_pe_state[short_k] = v
                    migrated += 1
            if migrated > 0:
                model.head.part_extractor.load_state_dict(head_pe_state)
                print(f"  Migrated {migrated} part_extractor tensors into model.head from {prefix}")

    print(f"  {prefix} state restored ({len(crit_loaded)} keys)")
    return len(crit_loaded)


def _sample_box_center_embeddings(embed_map, rois, batch_index, scale_x, scale_y, device):
    """Extract GT-box center embeddings with the shared bilinear sampler."""
    if len(rois) == 0:
        return embed_map.new_empty((0, embed_map.shape[1]))

    center_x = torch.tensor(
        [((r[1] + r[3]) * 0.5) * scale_x for r in rois],
        device=device,
        dtype=torch.float32,
    )
    center_y = torch.tensor(
        [((r[2] + r[4]) * 0.5) * scale_y for r in rois],
        device=device,
        dtype=torch.float32,
    )
    batch_indices = torch.full((len(rois),), batch_index, device=device, dtype=torch.long)
    return sample_feature_map_bilinear(embed_map, center_x, center_y, batch_indices=batch_indices)


def _concat_embedding(fused, parts, part_scale=0.25):
    """
    Concatenate fused + scaled part embeddings into a single vector.

    This is the standard inference approach in PCB, MGN, TransReID, AAformer:
    all top part-based ReID models concatenate parts instead of weighted-summing.
    TransReID scales parts by 0.25 to prevent part features from dominating.

    Args:
        fused: [C] L2-normalized fused embedding
        parts: [num_parts, C] L2-normalized part embeddings, or None
        part_scale: scaling factor for parts (0.25 = TransReID default)
    Returns:
        [C*(1+num_parts)] L2-normalized concatenated embedding
    """
    if parts is None:
        return fused / (np.linalg.norm(fused) + 1e-8)
    scaled_parts = parts * part_scale
    concat = np.concatenate([fused, scaled_parts.flatten()])
    norm = np.linalg.norm(concat)
    return concat / (norm + 1e-8) if norm > 1e-8 else concat


def _compute_same_cam_r1(reid_embeddings, part_aware_alpha=0.4, min_samples_per_cam=4, concat_inference=False):
    """
    Same-camera Rank-1 accuracy with temporal-split data-leak prevention.

    Standard cross-camera ReID excludes same-camera gallery entries entirely.
    This separate metric measures within-camera re-identification ability:
    "Can the model re-identify a person from later frames of the same camera?"

    Anti-leak protocol:
    1. Group embeddings by (person_id, camera_id)
    2. For groups with >= min_samples_per_cam samples, split chronologically:
       first half -> gallery, second half -> query
       (temporal split prevents near-duplicate frame leak)
    3. Small groups (< min_samples_per_cam) go entirely to gallery as distractors
    4. Each query matches against ALL gallery entries (same + cross camera, all PIDs)
    5. R1 = is top-1 match the correct PID?

    Returns:
        same_cam_r1: float (0-1)
        n_queries: int (number of valid queries)
    """
    # Step 1: Group by (PID, camera)
    pid_cam_groups = {}  # (pid, cam) -> [(fused, parts, attn), ...]
    for pid, emb_list in reid_embeddings.items():
        for fused, cam, parts, attn in emb_list:
            key = (pid, cam)
            if key not in pid_cam_groups:
                pid_cam_groups[key] = []
            pid_cam_groups[key].append((fused, parts, attn))

    # Step 2: Split eligible groups into gallery/query
    gallery_entries = []  # (pid, cam, fused_mean, parts_mean, attn_mean)
    query_entries = []    # (pid, cam, fused, parts, attn)

    for (pid, cam), samples in pid_cam_groups.items():
        if len(samples) < min_samples_per_cam:
            # Too few for reliable split — contribute to gallery only (as distractors)
            fused_mean = np.mean([s[0] for s in samples], axis=0)
            fused_mean = fused_mean / (np.linalg.norm(fused_mean) + 1e-8)
            parts_list = [s[1] for s in samples if s[1] is not None]
            attn_list = [s[2] for s in samples if s[2] is not None]
            if parts_list:
                parts_mean = np.mean(parts_list, axis=0)
                parts_mean = parts_mean / (np.linalg.norm(parts_mean, axis=1, keepdims=True) + 1e-8)
                attn_mean = np.mean(attn_list, axis=0)
            else:
                parts_mean, attn_mean = None, None
            gallery_entries.append((pid, cam, fused_mean, parts_mean, attn_mean))
            continue

        # Temporal split: first half -> gallery, second half -> query
        half = len(samples) // 2
        gallery_half = samples[:half]
        query_half = samples[half:]

        # Gallery entry: mean-pooled from first half
        fused_mean = np.mean([s[0] for s in gallery_half], axis=0)
        fused_mean = fused_mean / (np.linalg.norm(fused_mean) + 1e-8)
        parts_list = [s[1] for s in gallery_half if s[1] is not None]
        attn_list = [s[2] for s in gallery_half if s[2] is not None]
        if parts_list:
            parts_mean = np.mean(parts_list, axis=0)
            parts_mean = parts_mean / (np.linalg.norm(parts_mean, axis=1, keepdims=True) + 1e-8)
            attn_mean = np.mean(attn_list, axis=0)
        else:
            parts_mean, attn_mean = None, None
        gallery_entries.append((pid, cam, fused_mean, parts_mean, attn_mean))

        # Query entries: individual samples from second half
        for fused, parts, attn in query_half:
            query_entries.append((pid, cam, fused, parts, attn))

    if len(gallery_entries) < 2 or not query_entries:
        return 0.0, 0

    # Step 3: Match each query against all gallery entries
    correct = 0
    total = 0

    for pid, qcam, q_fused, q_parts, q_attn in query_entries:
        best_sim = -1.0
        best_pid = None

        if concat_inference:
            # Concatenation matching (PCB/MGN/TransReID style)
            q_concat = _concat_embedding(q_fused, q_parts)
            for g_pid, g_cam, g_fused, g_parts, g_attn in gallery_entries:
                g_concat = _concat_embedding(g_fused, g_parts)
                sim = float(np.dot(q_concat, g_concat))
                if sim > best_sim:
                    best_sim = sim
                    best_pid = g_pid
        else:
            # Blended matching (legacy)
            q_norm = q_fused / (np.linalg.norm(q_fused) + 1e-8)
            if q_parts is not None:
                q_parts_norm = q_parts / (np.linalg.norm(q_parts, axis=1, keepdims=True) + 1e-8)
            else:
                q_parts_norm = None

            for g_pid, g_cam, g_fused, g_parts, g_attn in gallery_entries:
                # Fused cosine similarity
                fused_sim = float(np.dot(g_fused, q_norm))

                # Part-aware similarity
                if (q_parts_norm is not None and q_attn is not None and
                        g_parts is not None and g_attn is not None and part_aware_alpha > 0):
                    per_part = np.sum(g_parts * q_parts_norm, axis=1)
                    mutual_vis = np.minimum(g_attn, q_attn)
                    vis_sum = mutual_vis.sum()
                    if vis_sum > 1e-8:
                        part_sim = float(np.sum(mutual_vis * per_part) / vis_sum)
                    else:
                        part_sim = float(per_part.mean())
                    sim = (1 - part_aware_alpha) * fused_sim + part_aware_alpha * part_sim
                else:
                    sim = fused_sim

                if sim > best_sim:
                    best_sim = sim
                    best_pid = g_pid

        if best_pid is not None:
            correct += int(best_pid == pid)
            total += 1

    r1 = correct / total if total > 0 else 0.0
    return r1, total


def _get_eval_bnneck_modules(model, criterion):
    """Resolve BNNeck modules from the model currently being evaluated."""
    head_module = getattr(model, 'head', None)
    if head_module is None and hasattr(model, 'module'):
        head_module = getattr(model.module, 'head', None)

    eval_bnneck = getattr(head_module, 'reid_bnneck', None) if head_module is not None else None
    eval_part_bnneck = getattr(head_module, 'part_bnneck', None) if head_module is not None else None

    if eval_bnneck is None and criterion is not None and getattr(criterion, 'use_bnneck', False):
        eval_bnneck = getattr(criterion, 'bnneck', None)
    if eval_part_bnneck is None and criterion is not None and getattr(criterion, 'use_bnneck', False):
        eval_part_bnneck = getattr(criterion, 'part_bnneck', None)

    return eval_bnneck, eval_part_bnneck


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION / EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, dataloader, device, criterion=None, iou_threshold=0.5, use_ema=False, reid_extraction='center', dataset=None, return_pred_reid=False, eval_use_bnneck=True, measure_fps=False, part_aware_alpha=0.0, concat_inference=False):
    """
    Run validation: detection mAP/mAR + ReID Rank-1 (cross-cam & same-cam).

    Decodes CenterNet heatmaps, extracts ReID embeddings at GT centres,
    and computes per-camera gallery matching with part-aware similarity.

    When return_pred_reid=True, also runs a second ReID eval pass where
    embeddings are extracted from *predicted* box centres (IoU>=0.5 match
    to GT required to assign a person ID).  This is the stricter metric that
    accounts for false/missed detections and is more suitable for publication.

    When measure_fps=True, times actual GPU inference per batch and appends
    a fps_data dict as the last element of the return tuple.

    Returns:
        (mAP, mAP_50, recall, val_loss, val_id_acc,
         reid_rank1, r1_same_cam, r1_cross_cam, mar_50)
        +  r1_cross_pred  (only when return_pred_reid=True)
        +  fps_data dict  (only when measure_fps=True, always last)
    """
    import time
    model.eval()
    if criterion is not None:
        criterion.eval()
    eval_bnneck, eval_part_bnneck = _get_eval_bnneck_modules(model, criterion)
    val_loss = 0
    val_id_acc = 0  # NOTE: not meaningful (train/val ID spaces differ)
    num_batches = 0
    metric = MeanAveragePrecision(iou_type="bbox").to(device)
    metric_50 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5]).to(device)
    reid_embeddings = {}  # pid -> [(embedding, cam_id)]
    reid_embeddings_pred = {}  # pid -> [(embedding, cam_id)]  — predicted-box path
    pre_bn_reid_embeddings = {}  # pid -> [(pre_bn_fused_np, cam_id)] for diagnostics
    global_img_counter = 0  # Track global image index for camera lookup

    # FPS measurement state (only when measure_fps=True)
    _fps_infer_time = 0.0
    _fps_infer_imgs = 0

    # Detection recall by visibility bucket
    vis_buckets = {'high': {'total': 0}, 'mid': {'total': 0}, 'low': {'total': 0}}
    
    print("Evaluating...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Validation")):
            # Handle both 2-tuple and 3-tuple (with camera_ids) from collate_fn
            if len(batch_data) == 3:
                imgs, targets, cam_ids = batch_data
            else:
                imgs, targets = batch_data
                cam_ids = None
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Time GPU inference for real-world FPS measurement
            if measure_fps:
                torch.cuda.synchronize()
                _fps_t0 = time.perf_counter()
            
            outputs = model(imgs)
            
            if measure_fps:
                torch.cuda.synchronize()
                _fps_infer_time += time.perf_counter() - _fps_t0
                _fps_infer_imgs += imgs.shape[0]
            
            # CenterNet Outputs decoding
            cls_pred = outputs['hm']
            embed_pred = outputs['embedding']
            wh_pred = outputs['wh']
            offset_pred = outputs.get('offset', torch.zeros_like(wh_pred))

            _, _, img_H, img_W = imgs.shape
            _, _, feat_H, feat_W = cls_pred.shape

            if criterion is not None:
                # Validation loss only consumes detection terms. Disable the
                # ReID branch here so validation cannot mutate XBM state and
                # does not depend on training-model ReID modules during EMA eval.
                result = criterion(outputs, targets, id_weight_scale=0.0)
                # (total_loss, loss_hm, loss_reg, loss_id, id_acc, loss_rep, loss_iou, loss_dcn_penalty)
                loss, l_cls, l_reg = result[0], result[1], result[2]
                id_acc_val = result[4] if len(result) > 4 else 0.0

                l_cls_val = l_cls.item() if isinstance(l_cls, torch.Tensor) else l_cls
                l_reg_val = l_reg.item() if isinstance(l_reg, torch.Tensor) else l_reg
                val_loss += (l_cls_val + l_reg_val)
                # Skip val ArcFace accuracy — train/val ID spaces are independently
                # compacted so val ID indices don't map to the same ArcFace columns.
                num_batches += 1

            # === Per-Branch Metrics at GT Centers ===
            B = imgs.shape[0]
            for b_idx in range(B):
                t_mask = (targets[:, 0] == b_idx)
                if t_mask.sum() == 0:
                    continue
                t_boxes = targets[t_mask]

                for t in t_boxes:
                    # Get GT center in feature map coords
                    cx_norm, cy_norm = t[3].item(), t[4].item()
                    cx_feat = int(cx_norm * feat_W)
                    cy_feat = int(cy_norm * feat_H)
                    cx_feat = min(max(0, cx_feat), feat_W - 1)
                    cy_feat = min(max(0, cy_feat), feat_H - 1)

                    # GT visibility: index depends on whether conf column exists
                    # 9 cols [batch, 0, id, x, y, w, h, conf, vis] -> vis at 8
                    # 8 cols [batch, 0, id, x, y, w, h, vis]       -> vis at 7
                    if t.shape[0] > 8:
                        gt_vis = t[8].item()  # conf+vis format
                    elif t.shape[0] > 7:
                        gt_vis = t[7].item()  # vis-only format
                    else:
                        gt_vis = 1.0

                    # Bucket by visibility
                    if gt_vis >= 0.7:
                        bucket = 'high'
                    elif gt_vis >= 0.4:
                        bucket = 'mid'
                    else:
                        bucket = 'low'
                    vis_buckets[bucket]['total'] += 1

            # === ReID Logic ===
            B = imgs.shape[0]
            _, C_embed, H_feat, W_feat = embed_pred.shape
            for b_idx in range(B):
                t_boxes = targets[targets[:, 0] == b_idx]
                if len(t_boxes) == 0: continue
                rois, roi_ids = [], []
                for t in t_boxes:
                    pid = int(t[2].item())
                    if pid < 0: continue
                    cx, cy = t[3].item() * img_W, t[4].item() * img_H
                    w, h = t[5].item() * img_W, t[6].item() * img_H
                    x1, y1 = max(0, cx - w/2), max(0, cy - h/2)
                    x2, y2 = min(img_W, cx + w/2), min(img_H, cy + h/2)
                    if x2 > x1 and y2 > y1:
                        rois.append([b_idx, x1, y1, x2, y2])
                        roi_ids.append(pid)
                
                if len(rois) > 0:
                    scale_x = W_feat / img_W
                    scale_y = H_feat / img_H
                    
                    part_embs_batch = None
                    attn_weights_batch = None
                    if reid_extraction == 'part_based':
                        # Part-based extraction: use PartBasedExtractor from head module
                        head_mod = getattr(model, 'head', None)
                        if head_mod is None and hasattr(model, 'module'):
                            head_mod = getattr(model.module, 'head', None)
                        if head_mod is not None and hasattr(head_mod, 'part_extractor') and head_mod.part_extractor is not None:
                            roi_boxes_feat = []
                            for r in rois:
                                roi_boxes_feat.append([
                                    float(b_idx),
                                    r[1] * scale_x, r[2] * scale_y,
                                    r[3] * scale_x, r[4] * scale_y
                                ])
                            roi_boxes_t = torch.tensor(roi_boxes_feat, device=device)
                            fused_emb, part_embs_batch, attn_weights_batch = head_mod.part_extractor(
                                embed_pred, roi_boxes_t, spatial_scale=1.0
                            )
                            batch_embeds = fused_emb
                        else:
                            # Fallback to center-pixel if part_extractor not available
                            batch_embeds = _sample_box_center_embeddings(
                                embed_pred, rois, b_idx, scale_x, scale_y, device
                            )
                    else:
                        # Center-pixel extraction (FairMOT-style)
                        batch_embeds = _sample_box_center_embeddings(
                            embed_pred, rois, b_idx, scale_x, scale_y, device
                        )
                    
                    embeddings = F.normalize(batch_embeds, p=2, dim=1, eps=1e-6)
                    
                    # Save pre-BN embeddings for diagnostics comparison
                    pre_bn_embeddings = embeddings.clone()
                    
                    # Apply BNNeck at eval time to align eval space with ArcFace training space
                    # (Luo et al., "Bag of Tricks"): ArcFace trains post-BN embeddings, so
                    # evaluation should also use post-BN for better R1/mAP correlation.
                    # Set eval_use_bnneck=False to test pre-BN embeddings instead.
                    if eval_use_bnneck and eval_bnneck is not None:
                        embeddings = eval_bnneck(embeddings)
                        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-6)
                    
                    # Determine camera ID from image path (for camera-aware ReID eval)
                    cam_id = 'unknown'
                    if dataset is not None and hasattr(dataset, 'img_files'):
                        global_idx = global_img_counter + b_idx
                        if global_idx < len(dataset.img_files):
                            img_path = dataset.img_files[global_idx]
                            # Extract camera from path using centralized method
                            cam_id = MOT17Dataset.extract_camera_string(img_path)
                    
                    for i, pid in enumerate(roi_ids):
                        if pid not in reid_embeddings: reid_embeddings[pid] = []
                        # Apply part BNNeck at eval to align with training space (like fused BNNeck)
                        parts_np = None
                        if part_embs_batch is not None:
                            p_emb_i = part_embs_batch[i]  # [num_parts, C]
                            if eval_use_bnneck and eval_part_bnneck is not None:
                                bn_parts = []
                                for p_idx in range(p_emb_i.shape[0]):
                                    p_single = p_emb_i[p_idx:p_idx+1]  # [1, C]
                                    p_bn = eval_part_bnneck[p_idx](p_single)
                                    p_bn = F.normalize(p_bn, p=2, dim=1, eps=1e-6)
                                    bn_parts.append(p_bn)
                                p_emb_i = torch.cat(bn_parts, dim=0)  # [num_parts, C]
                            parts_np = p_emb_i.cpu().numpy()
                        attn_np = attn_weights_batch[i].cpu().numpy() if attn_weights_batch is not None else None
                        reid_embeddings[pid].append((embeddings[i].cpu().numpy(), cam_id, parts_np, attn_np))
                        # Store pre-BN version for pre/post BNNeck diagnostics comparison
                        if pid not in pre_bn_reid_embeddings: pre_bn_reid_embeddings[pid] = []
                        pre_bn_reid_embeddings[pid].append((pre_bn_embeddings[i].cpu().numpy(), cam_id))

            # === Decode Predictions for mAP ===
            head_module = getattr(model, 'head', None)
            if head_module is None and hasattr(model, 'module'):
                head_module = getattr(model.module, 'head', None)
            
            batch_preds = []
            if hasattr(head_module, 'decode_detections'):
                 iou_pred = outputs.get('iou', None)
                 outputs_dict = {'hm': cls_pred, 'wh': wh_pred, 'offset': offset_pred, 'embedding': embed_pred, 'iou': iou_pred}
                 decoded = head_module.decode_detections(
                     outputs_dict, K=200, conf_thresh=0.05, input_size=(img_H, img_W),
                     nms_sigma=0.5,
                     use_max_pool_nms=True,
                     return_embeddings=False
                 )
                 for d in decoded:
                     d['labels'] = torch.zeros(len(d['scores']), dtype=torch.long, device=device)
                     batch_preds.append(d)
            else:
                 # Fallback for checkpoints without the expected descriptor branch.
                 for _ in range(B):
                     batch_preds.append({'boxes': torch.zeros((0,4), device=device), 'scores': torch.zeros(0,device=device), 'labels': torch.zeros(0,dtype=torch.long, device=device)})

            batch_targets = []
            for b_idx in range(B):
                t_mask = (targets[:, 0] == b_idx)
                if t_mask.sum() > 0:
                    t_boxes = targets[t_mask]
                    tcx, tcy = t_boxes[:, 3] * img_W, t_boxes[:, 4] * img_H
                    tw, th = t_boxes[:, 5] * img_W, t_boxes[:, 6] * img_H
                    tx1, ty1 = (tcx - tw/2).clamp(0, img_W), (tcy - th/2).clamp(0, img_H)
                    tx2, ty2 = (tcx + tw/2).clamp(0, img_W), (tcy + th/2).clamp(0, img_H)
                    batch_targets.append({'boxes': torch.stack([tx1, ty1, tx2, ty2], dim=1), 'labels': torch.zeros(len(t_boxes), dtype=torch.long, device=device)})
                else:
                    batch_targets.append({'boxes': torch.zeros((0,4), device=device), 'labels': torch.zeros(0,dtype=torch.long,device=device)})

            metric.update(batch_preds, batch_targets)
            metric_50.update(batch_preds, batch_targets)

            # === Predicted-Box ReID Extraction (opt-in, return_pred_reid=True) ===
            # For each predicted box with IoU >= 0.5 against a GT box that has a
            # valid person ID, sample the embedding at the *predicted* box centre.
            # This tests whether the model can retrieve people when using its own
            # detections rather than GT boxes — a stricter, more realistic metric.
            if return_pred_reid:
                for b_idx in range(B):
                    pred_d = batch_preds[b_idx] if b_idx < len(batch_preds) else None
                    if pred_d is None or len(pred_d['boxes']) == 0:
                        continue

                    # Build GT boxes + person IDs for this image
                    t_mask = (targets[:, 0] == b_idx)
                    if t_mask.sum() == 0:
                        continue
                    t_boxes_b = targets[t_mask]
                    gt_pid_list = []
                    gt_box_list = []
                    for t in t_boxes_b:
                        pid_t = int(t[2].item())
                        if pid_t < 0:
                            continue
                        tcx, tcy = t[3].item() * img_W, t[4].item() * img_H
                        tw, th   = t[5].item() * img_W, t[6].item() * img_H
                        gx1 = max(0.0, tcx - tw / 2)
                        gy1 = max(0.0, tcy - th / 2)
                        gx2 = min(float(img_W), tcx + tw / 2)
                        gy2 = min(float(img_H), tcy + th / 2)
                        gt_pid_list.append(pid_t)
                        gt_box_list.append([gx1, gy1, gx2, gy2])

                    if not gt_pid_list:
                        continue

                    gt_boxes_arr = torch.tensor(gt_box_list, device=device)  # [N_gt, 4]
                    pred_boxes_b = pred_d['boxes']                             # [N_pred, 4]

                    # IoU between all pred and all GT boxes
                    # boxes_iou: [N_pred, N_gt]
                    def _box_iou(a, b):
                        x1 = torch.max(a[:, None, 0], b[None, :, 0])
                        y1 = torch.max(a[:, None, 1], b[None, :, 1])
                        x2 = torch.min(a[:, None, 2], b[None, :, 2])
                        y2 = torch.min(a[:, None, 3], b[None, :, 3])
                        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
                        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
                        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                        union = area_a[:, None] + area_b[None, :] - inter
                        return inter / (union + 1e-7)

                    iou_mat = _box_iou(pred_boxes_b, gt_boxes_arr)  # [N_pred, N_gt]

                    # Camera ID for this image
                    pred_cam_id = 'unknown'
                    if dataset is not None and hasattr(dataset, 'img_files'):
                        g_idx = global_img_counter + b_idx
                        if g_idx < len(dataset.img_files):
                            pred_cam_id = MOT17Dataset.extract_camera_string(dataset.img_files[g_idx])

                    for p_idx in range(len(pred_boxes_b)):
                        best_gt_iou, best_gt_idx = iou_mat[p_idx].max(0)
                        if best_gt_iou.item() < 0.5:
                            continue
                        matched_pid = gt_pid_list[best_gt_idx.item()]

                        # Extract embedding at predicted box using the same method as GT path
                        pb = pred_boxes_b[p_idx]

                        if reid_extraction == 'part_based':
                            head_mod = getattr(model, 'head', None)
                            if head_mod is None and hasattr(model, 'module'):
                                head_mod = getattr(model.module, 'head', None)
                            if head_mod is not None and hasattr(head_mod, 'part_extractor') and head_mod.part_extractor is not None:
                                scale_x_p = W_feat / img_W
                                scale_y_p = H_feat / img_H
                                roi_t = torch.tensor([[
                                    float(b_idx),
                                    pb[0].item() * scale_x_p, pb[1].item() * scale_y_p,
                                    pb[2].item() * scale_x_p, pb[3].item() * scale_y_p,
                                ]], device=device)
                                fused_e, parts_e, attn_e = head_mod.part_extractor(
                                    embed_pred, roi_t, spatial_scale=1.0
                                )
                                emb_vec = F.normalize(fused_e, p=2, dim=1, eps=1e-6)
                                if eval_use_bnneck and eval_bnneck is not None:
                                    emb_vec = eval_bnneck(emb_vec)
                                    emb_vec = F.normalize(emb_vec, p=2, dim=1, eps=1e-6)
                                parts_np_p = None
                                attn_np_p = None
                                if parts_e is not None and eval_use_bnneck and eval_part_bnneck is not None:
                                    bn_parts = []
                                    for p_idx2 in range(parts_e.shape[1]):
                                        p_s = parts_e[0, p_idx2:p_idx2+1]
                                        p_bn = eval_part_bnneck[p_idx2](p_s)
                                        p_bn = F.normalize(p_bn, p=2, dim=1, eps=1e-6)
                                        bn_parts.append(p_bn)
                                    parts_e_bn = torch.cat(bn_parts, dim=0)
                                    parts_np_p = parts_e_bn.cpu().numpy()
                                elif parts_e is not None:
                                    parts_np_p = parts_e[0].cpu().numpy()
                                if attn_e is not None:
                                    attn_np_p = attn_e[0].cpu().numpy()
                                emb_np = emb_vec[0].cpu().numpy()
                                if matched_pid not in reid_embeddings_pred:
                                    reid_embeddings_pred[matched_pid] = []
                                reid_embeddings_pred[matched_pid].append((emb_np, pred_cam_id, parts_np_p, attn_np_p))
                                continue  # handled — skip fallback below

                        # Fallback: center-pixel extraction (used for non-part_based modes)
                        pcx = ((pb[0] + pb[2]) / 2).item()
                        pcy = ((pb[1] + pb[3]) / 2).item()
                        px_feat = int(pcx * W_feat / img_W)
                        py_feat = int(pcy * H_feat / img_H)
                        px_feat = min(max(0, px_feat), W_feat - 1)
                        py_feat = min(max(0, py_feat), H_feat - 1)

                        emb_vec = embed_pred[b_idx, :, py_feat, px_feat]  # [C]
                        emb_vec = F.normalize(emb_vec.unsqueeze(0), p=2, dim=1, eps=1e-6)

                        if eval_use_bnneck and eval_bnneck is not None:
                            emb_vec = eval_bnneck(emb_vec)
                            emb_vec = F.normalize(emb_vec, p=2, dim=1, eps=1e-6)

                        emb_np = emb_vec[0].cpu().numpy()
                        if matched_pid not in reid_embeddings_pred:
                            reid_embeddings_pred[matched_pid] = []
                        reid_embeddings_pred[matched_pid].append((emb_np, pred_cam_id, None, None))

            global_img_counter += B  # Track global image index for camera lookup

    avg_val_loss = val_loss / num_batches if num_batches > 0 else 0
    avg_val_acc = 0  # Disabled: train/val ID spaces don't match
    metrics_dict = metric.compute()
    mAP, mAP_50, val_rec = metrics_dict['map'], metrics_dict['map_50'], metrics_dict['mar_100']
    
    # mAR@0.5: recall at IoU=0.5 specifically (not averaged over 0.5:0.95)
    metrics_50 = metric_50.compute()
    mar_50 = metrics_50.get('mar_100', val_rec)  # mar_100 at IoU=0.5 only
    
    # Diagnostic: per-IoU and per-size recall breakdown
    for k in ['map_75', 'mar_small', 'mar_medium', 'mar_large']:
        v = metrics_dict.get(k)
        if v is not None and not (hasattr(v, 'isnan') and v.isnan()):
            print(f"    {k}: {float(v):.4f}")
    
    recall = metrics_dict.get('mar_100', val_rec)    # Recall @ 100 detections (IoU 0.5:0.95)
    
    # ====================================================================
    # ReID Rank-1: Per-Camera Gallery with Max-Similarity Matching
    # ====================================================================
    # Protocol (mirrors tracker's gallery + Market-1501 standard):
    #   1. For each person, build SEPARATE gallery entries per camera
    #      (mean-pool within same camera only — preserves viewpoint info)
    #   2. Query from a DIFFERENT camera if possible (cross-cam test)
    #   3. Match using MAX similarity across gallery entries
    #      (not mean — same as tracker's gallery min-distance)
    #   4. Part-aware matching with SHARP per-camera attention weights
    #      (not blurred by cross-camera averaging)
    # ====================================================================
    reid_rank1_acc = 0.0
    r1_same_cam = 0.0
    r1_cross_cam = 0.0
    num_cameras = 0
    # part_aware_alpha passed as parameter (from config tracker.part_aware_alpha)
    
    if len(reid_embeddings) >= 2:
        # Step 1: Group embeddings by (person_id, camera) for per-camera gallery
        # pid -> cam -> [(fused, parts, attn), ...]
        pid_cam_groups = {}
        for pid, emb_cam_list in reid_embeddings.items():
            if len(emb_cam_list) < 2:
                continue
            cam_groups = {}
            for fused, cam, parts, attn in emb_cam_list:
                if cam not in cam_groups:
                    cam_groups[cam] = []
                cam_groups[cam].append((fused, parts, attn))
            if cam_groups:
                pid_cam_groups[pid] = cam_groups
        
        # Count unique cameras
        all_cams = set()
        for pid, cam_groups in pid_cam_groups.items():
            all_cams.update(cam_groups.keys())
        num_cameras = len(all_cams)
        
        # Step 2: Build per-camera gallery entries
        # Each entry: (pid, cam, fused_mean, parts_mean, attn_mean)
        gallery_entries = []  # list of (pid, cam, fused, parts, attn)
        query_data = []     # (pid, fused, cam, parts, attn)
        
        for pid, cam_groups in pid_cam_groups.items():
            if len(cam_groups) >= 2:
                # Multi-camera person:
                # Query from the camera with FEWER samples (maximizes cross-camera testing)
                query_cam = min(cam_groups.keys(), key=lambda c: len(cam_groups[c]))
                for fused, parts, attn in cam_groups[query_cam]:
                    query_data.append((pid, fused, query_cam, parts, attn))
                
                # Build gallery from ALL OTHER cameras (and exclude query_cam)
                for cam, samples in cam_groups.items():
                    if cam == query_cam: continue
                    fused_mean = np.mean([s[0] for s in samples], axis=0)
                    fused_mean = fused_mean / (np.linalg.norm(fused_mean) + 1e-8)
                    
                    parts_list = [s[1] for s in samples if s[1] is not None]
                    attn_list = [s[2] for s in samples if s[2] is not None]
                    if parts_list:
                        parts_mean = np.mean(parts_list, axis=0)
                        parts_mean = parts_mean / (np.linalg.norm(parts_mean, axis=1, keepdims=True) + 1e-8)
                        attn_mean = np.mean(attn_list, axis=0)
                    else:
                        parts_mean = None
                        attn_mean = None
                    gallery_entries.append((pid, cam, fused_mean, parts_mean, attn_mean))
            else:
                # Single-camera person:
                # MUST split samples first to prevent data leak!
                # If we mean-pool all 10 samples for the gallery, and use the last 5 as queries,
                # the query is finding ITSELF in the gallery mean.
                single_cam = list(cam_groups.keys())[0]
                samples = cam_groups[single_cam]
                half = max(1, len(samples) // 2)
                
                gallery_samples = samples[:half]
                query_samples = samples[half:]
                
                # Queries
                for fused, parts, attn in query_samples:
                    query_data.append((pid, fused, single_cam, parts, attn))
                
                # Gallery (only from first half)
                if gallery_samples:
                    fused_mean = np.mean([s[0] for s in gallery_samples], axis=0)
                    fused_mean = fused_mean / (np.linalg.norm(fused_mean) + 1e-8)
                    
                    parts_list = [s[1] for s in gallery_samples if s[1] is not None]
                    attn_list = [s[2] for s in gallery_samples if s[2] is not None]
                    if parts_list:
                        parts_mean = np.mean(parts_list, axis=0)
                        parts_mean = parts_mean / (np.linalg.norm(parts_mean, axis=1, keepdims=True) + 1e-8)
                        attn_mean = np.mean(attn_list, axis=0)
                    else:
                        parts_mean = None
                        attn_mean = None
                    gallery_entries.append((pid, single_cam, fused_mean, parts_mean, attn_mean))
        
        # Initialize before conditional block (avoids fragile dir() scoping)
        correct_all, total_all = 0, 0
        correct_cross, total_cross = 0, 0
        reid_rank1_acc = 0.0
        r1_cross_cam = 0.0

        if len(gallery_entries) >= 2 and len(query_data) > 0:
            
            # Pre-compute camera sets per PID for efficient same-cam exclusion
            pid_gallery_cams = {}
            for g_pid, g_cam, _, _, _ in gallery_entries:
                pid_gallery_cams.setdefault(g_pid, set()).add(g_cam)

            # Similarity diagnostics: track correct vs incorrect match similarities
            sim_correct_list = []  # similarities for correct top-1 matches
            sim_incorrect_list = []  # similarities for incorrect top-1 matches
            sim_true_pos_list = []  # similarity to the correct gallery entry (even if not top-1)

            for pid, q_fused, qcam, q_parts, q_attn in query_data:
                # Skip queries where the true PID has no cross-camera gallery entries
                # (standard ReID protocol: no valid positive = skip this query)
                pid_cams = pid_gallery_cams.get(pid, set())
                cross_cam_gallery = pid_cams - {qcam}
                if not cross_cam_gallery:
                    continue

                q_fused_norm = q_fused / (np.linalg.norm(q_fused) + 1e-8)
                if concat_inference:
                    q_concat = _concat_embedding(q_fused, q_parts)
                else:
                    if q_parts is not None:
                        q_parts_norm = q_parts / (np.linalg.norm(q_parts, axis=1, keepdims=True) + 1e-8)
                    else:
                        q_parts_norm = None
                
                # Max-similarity matching: for each gallery PERSON, take max sim across their camera entries
                # This mirrors the tracker's gallery min-distance approach
                person_best_sim = {}  # pid -> best similarity
                person_best_cam = {}  # pid -> camera of best gallery entry
                
                for g_pid, g_cam, g_fused, g_parts, g_attn in gallery_entries:
                    # Always exclude same-PID same-camera entries (standard ReID protocol)
                    if g_pid == pid and g_cam == qcam:
                        continue
                    
                    if concat_inference:
                        g_concat = _concat_embedding(g_fused, g_parts)
                        sim = float(np.dot(q_concat, g_concat))
                    else:
                        # Fused cosine similarity
                        fused_sim = float(np.dot(g_fused, q_fused_norm))
                        
                        # Part-aware similarity
                        if q_parts_norm is not None and q_attn is not None and g_parts is not None and g_attn is not None and part_aware_alpha > 0:
                            per_part = np.sum(g_parts * q_parts_norm, axis=1)  # [6]
                            mutual_vis = np.minimum(g_attn, q_attn)  # [6]
                            vis_sum = mutual_vis.sum()
                            if vis_sum > 1e-8:
                                part_sim = float(np.sum(mutual_vis * per_part) / vis_sum)
                            else:
                                part_sim = float(per_part.mean())
                            sim = (1 - part_aware_alpha) * fused_sim + part_aware_alpha * part_sim
                        else:
                            sim = fused_sim
                    
                    # Max-similarity per person (best camera match)
                    if g_pid not in person_best_sim or sim > person_best_sim[g_pid]:
                        person_best_sim[g_pid] = sim
                        person_best_cam[g_pid] = g_cam
                
                if not person_best_sim:
                    continue
                
                # Rank by similarity — R1 = is top-1 correct?
                pred_pid = max(person_best_sim, key=person_best_sim.get)
                is_correct = (pred_pid == pid)
                
                # Similarity diagnostics
                top1_sim = person_best_sim[pred_pid]
                true_sim = person_best_sim.get(pid, None)
                if is_correct:
                    sim_correct_list.append(top1_sim)
                else:
                    sim_incorrect_list.append(top1_sim)
                if true_sim is not None:
                    sim_true_pos_list.append(true_sim)
                
                correct_all += int(is_correct)
                total_all += 1
                
                # Cross-camera breakdown
                true_g_cam = person_best_cam.get(pid, 'unknown')
                if qcam != true_g_cam and qcam != 'unknown' and true_g_cam != 'unknown':
                    correct_cross += int(is_correct)
                    total_cross += 1
            
            reid_rank1_acc = correct_all / total_all if total_all > 0 else 0.0
            r1_cross_cam = correct_cross / total_cross if total_cross > 0 else 0.0

            # Print similarity diagnostics
            if sim_correct_list or sim_incorrect_list:
                import statistics
                def _sim_stats(lst):
                    if not lst: return "N/A"
                    return f"mean={statistics.mean(lst):.4f}, std={statistics.stdev(lst) if len(lst) > 1 else 0:.4f}, min={min(lst):.4f}, max={max(lst):.4f}"
                print(f"  Similarity diagnostics (BNNeck={'ON' if eval_use_bnneck else 'OFF'}):")
                print(f"    Correct top-1  ({len(sim_correct_list):5d}): {_sim_stats(sim_correct_list)}")
                print(f"    Incorrect top-1 ({len(sim_incorrect_list):5d}): {_sim_stats(sim_incorrect_list)}")
                print(f"    True positive   ({len(sim_true_pos_list):5d}): {_sim_stats(sim_true_pos_list)}")
                if sim_correct_list and sim_incorrect_list:
                    gap = statistics.mean(sim_correct_list) - statistics.mean(sim_incorrect_list)
                    print(f"    Gap (correct - incorrect): {gap:.4f}")

            # === Pre-BN Diagnostics (for comparison with post-BN above) ===
            # Uses the same query/gallery person assignments, but with pre-BN
            # embeddings (before BatchNorm). Fused-only similarity (no parts).
            if eval_use_bnneck and len(pre_bn_reid_embeddings) >= 2:
                # Build pre-BN per-camera galleries using same pid_cam_groups structure
                pre_bn_cam_groups = {}
                for pid, emb_list in pre_bn_reid_embeddings.items():
                    if len(emb_list) < 2:
                        continue
                    cg = {}
                    for fused, cam in emb_list:
                        if cam not in cg:
                            cg[cam] = []
                        cg[cam].append(fused)
                    if cg:
                        pre_bn_cam_groups[pid] = cg

                # Build pre-BN gallery + queries with same camera assignment
                pre_gallery = []  # (pid, cam, fused_mean)
                pre_queries = []  # (pid, fused, cam)
                for pid, cam_groups in pre_bn_cam_groups.items():
                    if len(cam_groups) >= 2:
                        query_cam = min(cam_groups.keys(), key=lambda c: len(cam_groups[c]))
                        for fused in cam_groups[query_cam]:
                            pre_queries.append((pid, fused, query_cam))
                        for cam, samples in cam_groups.items():
                            if cam == query_cam:
                                continue
                            fm = np.mean(samples, axis=0)
                            fm = fm / (np.linalg.norm(fm) + 1e-8)
                            pre_gallery.append((pid, cam, fm))
                    else:
                        single_cam = list(cam_groups.keys())[0]
                        samples = cam_groups[single_cam]
                        half = max(1, len(samples) // 2)
                        fm = np.mean(samples[:half], axis=0)
                        fm = fm / (np.linalg.norm(fm) + 1e-8)
                        pre_gallery.append((pid, single_cam, fm))
                        for fused in samples[half:]:
                            pre_queries.append((pid, fused, single_cam))

                # Compute pre-BN similarity diagnostics
                pre_gallery_pid_cams = {}
                for g_pid, g_cam, _ in pre_gallery:
                    pre_gallery_pid_cams.setdefault(g_pid, set()).add(g_cam)

                pre_sim_correct = []
                pre_sim_incorrect = []
                pre_sim_true_pos = []
                for pid, q_fused, qcam in pre_queries:
                    pid_cams = pre_gallery_pid_cams.get(pid, set())
                    if not (pid_cams - {qcam}):
                        continue
                    q_norm = q_fused / (np.linalg.norm(q_fused) + 1e-8)
                    person_best = {}
                    for g_pid, g_cam, g_fused in pre_gallery:
                        if g_pid == pid and g_cam == qcam:
                            continue
                        sim = float(np.dot(g_fused, q_norm))
                        if g_pid not in person_best or sim > person_best[g_pid]:
                            person_best[g_pid] = sim
                    if not person_best:
                        continue
                    pred = max(person_best, key=person_best.get)
                    top1 = person_best[pred]
                    true = person_best.get(pid, None)
                    if pred == pid:
                        pre_sim_correct.append(top1)
                    else:
                        pre_sim_incorrect.append(top1)
                    if true is not None:
                        pre_sim_true_pos.append(true)

                if pre_sim_correct or pre_sim_incorrect:
                    print(f"  Similarity diagnostics (BNNeck=OFF, pre-BN):")
                    print(f"    Correct top-1  ({len(pre_sim_correct):5d}): {_sim_stats(pre_sim_correct)}")
                    print(f"    Incorrect top-1 ({len(pre_sim_incorrect):5d}): {_sim_stats(pre_sim_incorrect)}")
                    print(f"    True positive   ({len(pre_sim_true_pos):5d}): {_sim_stats(pre_sim_true_pos)}")
                    if pre_sim_correct and pre_sim_incorrect:
                        pre_gap = statistics.mean(pre_sim_correct) - statistics.mean(pre_sim_incorrect)
                        print(f"    Gap (correct - incorrect): {pre_gap:.4f}")
                    # Side-by-side comparison
                    if sim_correct_list and pre_sim_correct:
                        post_gap = statistics.mean(sim_correct_list) - statistics.mean(sim_incorrect_list) if sim_incorrect_list else 0
                        print(f"  BNNeck effect: post-BN gap={post_gap:.4f}, pre-BN gap={pre_gap:.4f}, delta={post_gap - pre_gap:+.4f}")

    # === Same-Camera R1 (separate evaluation with temporal-split anti-leak) ===
    r1_same_cam = 0.0
    n_same_queries = 0
    if len(reid_embeddings) >= 2:
        r1_same_cam, n_same_queries = _compute_same_cam_r1(
            reid_embeddings, part_aware_alpha=part_aware_alpha, min_samples_per_cam=4,
            concat_inference=concat_inference
        )

    # === Predicted-Box Cross-Camera R1 (opt-in) ===
    # Re-runs the same per-camera gallery + cross-cam query matching logic, but
    # using reid_embeddings_pred (extracted at predicted box centres, IoU>=0.5).
    # People whose detections were missed entirely do not contribute a query,
    # so this penalises recall failures.
    r1_cross_pred = 0.0
    n_pred_queries = 0
    if return_pred_reid and len(reid_embeddings_pred) >= 2:
        pred_pid_cam_groups = {}
        for pid, emb_list in reid_embeddings_pred.items():
            if len(emb_list) < 2:
                continue
            cg = {}
            for fused, cam, parts, attn in emb_list:
                if cam not in cg:
                    cg[cam] = []
                cg[cam].append((fused, parts, attn))
            if cg:
                pred_pid_cam_groups[pid] = cg

        pred_gallery = []   # (pid, cam, fused_mean, parts_mean, attn_mean)
        pred_queries  = []  # (pid, fused, cam, parts, attn)
        for pid, cam_groups in pred_pid_cam_groups.items():
            if len(cam_groups) >= 2:
                query_cam = min(cam_groups.keys(), key=lambda c: len(cam_groups[c]))
                for f, p, a in cam_groups[query_cam]:
                    pred_queries.append((pid, f, query_cam, p, a))
                for cam, samples in cam_groups.items():
                    if cam == query_cam:
                        continue
                    fm = np.mean([s[0] for s in samples], axis=0)
                    fm = fm / (np.linalg.norm(fm) + 1e-8)
                    parts_list = [s[1] for s in samples if s[1] is not None]
                    attn_list  = [s[2] for s in samples if s[2] is not None]
                    pm = np.mean(parts_list, axis=0) / (np.linalg.norm(np.mean(parts_list, axis=0), axis=1, keepdims=True) + 1e-8) if parts_list else None
                    am = np.mean(attn_list, axis=0) if attn_list else None
                    pred_gallery.append((pid, cam, fm, pm, am))
            else:
                single_cam = list(cam_groups.keys())[0]
                samples = cam_groups[single_cam]
                half = max(1, len(samples) // 2)
                gf = np.mean([s[0] for s in samples[:half]], axis=0)
                gf = gf / (np.linalg.norm(gf) + 1e-8)
                parts_list = [s[1] for s in samples[:half] if s[1] is not None]
                attn_list  = [s[2] for s in samples[:half] if s[2] is not None]
                pm = np.mean(parts_list, axis=0) / (np.linalg.norm(np.mean(parts_list, axis=0), axis=1, keepdims=True) + 1e-8) if parts_list else None
                am = np.mean(attn_list, axis=0) if attn_list else None
                pred_gallery.append((pid, single_cam, gf, pm, am))
                for f, p, a in samples[half:]:
                    pred_queries.append((pid, f, single_cam, p, a))

        pred_gallery_pid_cams = {}
        for g_pid, g_cam, _, _, _ in pred_gallery:
            pred_gallery_pid_cams.setdefault(g_pid, set()).add(g_cam)

        correct_pred, total_pred = 0, 0
        for pid, q_fused, qcam, q_parts, q_attn in pred_queries:
            pid_cams = pred_gallery_pid_cams.get(pid, set())
            if not (pid_cams - {qcam}):
                continue
            best_pid = None
            best_sim = -1.0

            if concat_inference:
                q_concat = _concat_embedding(q_fused, q_parts)
                for g_pid, g_cam, g_fused, g_parts, g_attn in pred_gallery:
                    if g_pid == pid and g_cam == qcam:
                        continue
                    g_concat = _concat_embedding(g_fused, g_parts)
                    sim = float(np.dot(q_concat, g_concat))
                    if sim > best_sim:
                        best_sim = sim
                        best_pid = g_pid
            else:
                q_norm = q_fused / (np.linalg.norm(q_fused) + 1e-8)
                q_parts_norm = q_parts / (np.linalg.norm(q_parts, axis=1, keepdims=True) + 1e-8) if q_parts is not None else None
                for g_pid, g_cam, g_fused, g_parts, g_attn in pred_gallery:
                    if g_pid == pid and g_cam == qcam:
                        continue
                    fused_sim = float(np.dot(g_fused, q_norm))
                    if q_parts_norm is not None and q_attn is not None and g_parts is not None and g_attn is not None and part_aware_alpha > 0:
                        per_part = np.sum(g_parts * q_parts_norm, axis=1)
                        mutual_vis = np.minimum(g_attn, q_attn)
                        vis_sum = mutual_vis.sum()
                        part_sim = float(np.sum(mutual_vis * per_part) / vis_sum) if vis_sum > 1e-8 else float(per_part.mean())
                        sim = (1 - part_aware_alpha) * fused_sim + part_aware_alpha * part_sim
                    else:
                        sim = fused_sim
                    if sim > best_sim:
                        best_sim = sim
                        best_pid = g_pid

            if best_pid is not None:
                correct_pred += int(best_pid == pid)
                total_pred += 1

        r1_cross_pred = correct_pred / total_pred if total_pred > 0 else 0.0
        n_pred_queries = total_pred

    # === Print Performance Metrics ===
    print(f"  ReID Rank-1 (cross-cam): {reid_rank1_acc*100:.1f}% ({len(reid_embeddings)} identities, {num_cameras} cameras)")
    if n_same_queries > 0:
        print(f"  ReID Rank-1 (same-cam):  {r1_same_cam*100:.1f}% ({n_same_queries} queries, temporal-split protocol)")
    if num_cameras > 1:
        skipped = len(query_data) - total_all
        if skipped > 0:
            print(f"    Cross-camera only (standard ReID protocol) | {skipped} single-camera queries skipped")
    if return_pred_reid:
        print(f"  ReID Rank-1 (pred-box, cross-cam): {r1_cross_pred*100:.1f}% ({n_pred_queries} queries from {len(reid_embeddings_pred)} IDs with detection)")
    print(f"  Detection mAP@0.5: {mAP_50.item():.4f} | mAR@100: {recall.item():.4f} | mAR@0.5: {mar_50.item():.4f}")
    print(f"  GT Distribution - High(>=0.7): {vis_buckets['high']['total']}, Mid(0.4-0.7): {vis_buckets['mid']['total']}, Low(<0.4): {vis_buckets['low']['total']}")

    # === Real-world FPS (measured from actual inference batches) ===
    fps_data = None
    if measure_fps and _fps_infer_imgs > 0:
        fps_ms = (_fps_infer_time / _fps_infer_imgs) * 1000
        fps_val = 1000.0 / fps_ms
        total_params = sum(p.numel() for p in model.parameters())
        fps_data = {
            'fps': round(fps_val, 1),
            'ms_per_frame': round(fps_ms, 2),
            'total_frames': _fps_infer_imgs,
            'total_time_s': round(_fps_infer_time, 2),
            'params_M': round(total_params / 1e6, 2),
        }
        print(f"  Inference FPS (real): {fps_val:.1f} FPS ({fps_ms:.2f} ms/frame, {_fps_infer_imgs} frames)")
        print(f"  Params: {total_params/1e6:.2f}M")

    base_results = (mAP.item(), mAP_50.item(), recall.item(), avg_val_loss, avg_val_acc, reid_rank1_acc, r1_same_cam, r1_cross_cam, mar_50.item())
    if return_pred_reid:
        base_results = base_results + (r1_cross_pred,)
    if measure_fps:
        return base_results + (fps_data,)
    return base_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train(config):
    """
    End-to-end training pipeline: dataset → model → loss → optimizer → epoch loop.

    Handles checkpoint resume (including mid-epoch), EMA, AMP, gradient
    accumulation, ArcFace ID warm-up, and per-epoch CSV metric logging.
    Saves best_model.pth based on combined score (mAP + recall + R1).
    """
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = config.get('experiment', {}).get('save_dir', 'effiped/experiments')
    exp_name = config.get('experiment', {}).get('name', '')
    if exp_name:
        save_dir = os.path.join(save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    runconfigs_dir = os.path.join(save_dir, 'runconfigs')
    runs_dir = os.path.join(save_dir, 'runs')
    had_run_archives = os.path.isdir(runs_dir) and bool(glob.glob(os.path.join(runs_dir, '*')))
    run_dir = os.path.join(runs_dir, run_id)
    run_ckpt_dir = os.path.join(run_dir, 'checkpoints')
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    for d in (runconfigs_dir, run_dir, run_ckpt_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(save_dir, 'latest_run.txt'), 'w') as f:
        f.write(run_id + '\n')
    log_file = os.path.join(run_dir, 'v14_log.csv')
    legacy_log_file = os.path.join(save_dir, 'v14_log.csv')
    if not had_run_archives and os.path.exists(legacy_log_file):
        legacy_archive_dir = os.path.join(runs_dir, f'legacy_import_{run_id}')
        os.makedirs(legacy_archive_dir, exist_ok=True)
        for legacy_name in ('v14_log.csv', 'config.yaml', 'run_config.json'):
            _copy_if_exists(os.path.join(save_dir, legacy_name), os.path.join(legacy_archive_dir, legacy_name))

    # ── Experiment Config Snapshot ──────────────────────────────────────
    # Fixed filenames — overwrite on each run to avoid timestamp clutter.
    # config.yaml  = raw YAML copy (know which config was used)
    # run_config.json = fully-resolved runtime config (all defaults filled in)
    _src_cfg = config.get('_config_path')
    if _src_cfg and os.path.isfile(_src_cfg):
        _dst = os.path.join(run_dir, 'config.yaml')
        shutil.copy2(_src_cfg, os.path.join(runconfigs_dir, f'{run_id}.yaml'))
        shutil.copy2(_src_cfg, _dst)
        shutil.copy2(_src_cfg, os.path.join(save_dir, 'config.yaml'))
        print(f"  📋 Config snapshot saved: {_dst}")
    # Runtime-resolved config (JSON for easy diff / programmatic loading)
    _runtime_cfg = {k: v for k, v in config.items() if not k.startswith('_')}
    _runtime_path = os.path.join(run_dir, 'run_config.json')
    with open(_runtime_path, 'w') as _f:
        json.dump(_runtime_cfg, _f, indent=2, default=str)
    with open(os.path.join(runconfigs_dir, f'{run_id}.json'), 'w') as _f:
        json.dump(_runtime_cfg, _f, indent=2, default=str)
    with open(os.path.join(save_dir, 'run_config.json'), 'w') as _f:
        json.dump(_runtime_cfg, _f, indent=2, default=str)
    print(f"  📋 Runtime config saved:  {_runtime_path}")
    
    _ensure_csv_header(log_file)
    with open(legacy_log_file, 'w', newline='') as f:
        csv.writer(f).writerow(_csv_header())

    model_cfg = config.setdefault('model', {})
    training_cfg = config.setdefault('training', {})
    selected_auto_resume_path = None
    if not model_cfg.get('resume') and training_cfg.get('auto_resume', True):
        auto_resume_path = _find_auto_resume_checkpoint(save_dir)
        if auto_resume_path:
            selected_auto_resume_path = auto_resume_path
            model_cfg['resume'] = auto_resume_path
            model_cfg['reset_optimizer'] = False
            print(f"  Auto-resume checkpoint selected: {auto_resume_path}")
    elif not training_cfg.get('auto_resume', True):
        print("  Auto-resume disabled by config/CLI; starting from resume/pretrained/scratch policy.")
    config.setdefault('runtime', {}).update({
        'run_id': run_id,
        'run_dir': run_dir,
        'checkpoints_dir': run_ckpt_dir,
        'auto_resume_path': selected_auto_resume_path,
        'effective_resume': model_cfg.get('resume'),
    })
    _runtime_cfg = {k: v for k, v in config.items() if not k.startswith('_')}
    for _runtime_path in (
        os.path.join(runconfigs_dir, f'{run_id}.json'),
        os.path.join(run_dir, 'run_config.json'),
        os.path.join(save_dir, 'run_config.json'),
    ):
        with open(_runtime_path, 'w') as _f:
            json.dump(_runtime_cfg, _f, indent=2, default=str)

    # benchmark=True selects fastest cuDNN algorithm per input shape.
    # Disabled when deterministic mode was set by set_seed() — re-enabling
    # would break reproducibility. Users can override via config if needed.
    if device.type == 'cuda' and not torch.backends.cudnn.deterministic:
        torch.backends.cudnn.benchmark = True
    
    # ─── Dataset Initialization ─────────────────────────────────────────
    print("Initializing Datasets...")
    aug_config = config['data'].get('augmentation', {})
    frame_skip = config['data'].get('frame_skip', 1)
    
    train_dataset = MOT17Dataset(
        root=config['data']['root'],
        seqs=config['data']['train_seqs'],
        img_size=tuple(config['data']['img_size']),
        min_visibility=config['data'].get('min_visibility', 0.3),
        augment=True, aug_config=aug_config, frame_skip=frame_skip,
        crowdhuman_ratio=config['data'].get('crowdhuman_ratio', 1.0),
        additional_sources=config['data'].get('additional_sources', None)
    )
    val_dataset = MOT17Dataset(
        root=config['data']['root'], seqs=config['data']['val_seqs'],
        img_size=tuple(config['data']['img_size']), 
        min_visibility=config['data'].get('min_visibility', 0.3),
        augment=False, frame_skip=frame_skip
    )
    
    # Use identity-aware sampling for effective triplet loss
    # Options: 'identity_aware' (best, temporal gap + cross-camera), 'grouped' (consecutive frames), false/None (random)
    sampler_type = config['data'].get('sampler_type', 'grouped')  # default: grouped for backward compat
    use_grouped_sampling = config['data'].get('use_grouped_sampling', True)
    
    if sampler_type == 'identity_aware':
        cross_camera_ratio = config['data'].get('cross_camera_ratio', 0.2)
        sampler = IdentityAwareSampler(
            train_dataset, batch_size=config['data']['batch_size'],
            cross_camera_ratio=cross_camera_ratio
        )
        train_dl = DataLoader(
            train_dataset, batch_sampler=sampler,
            num_workers=config['data']['num_workers'], pin_memory=True,
            persistent_workers=True, prefetch_factor=4, collate_fn=collate_fn
        )
        print(f"  Using IdentityAwareSampler ({len(sampler)} batches per epoch)")
    elif use_grouped_sampling:
        sampler = SequenceGroupedSampler(train_dataset, batch_size=config['data']['batch_size'])
        train_dl = DataLoader(
            train_dataset, batch_sampler=sampler,
            num_workers=config['data']['num_workers'], pin_memory=True,
            persistent_workers=True, prefetch_factor=4, collate_fn=collate_fn
        )
        print(f"  Using SequenceGroupedSampler ({len(sampler)} batches per epoch)")
    else:
        train_dl = DataLoader(
            train_dataset, batch_size=config['data']['batch_size'], shuffle=True,
            num_workers=config['data']['num_workers'], pin_memory=True,
            persistent_workers=True, prefetch_factor=4, collate_fn=collate_fn
        )
    

    val_dl = DataLoader(
        val_dataset, batch_size=config['data']['batch_size'], shuffle=False,
        num_workers=config['data']['num_workers'], pin_memory=True,
        persistent_workers=True, collate_fn=collate_fn
    )

    # Secondary validation (MOT-type scenarios) — informational only, not used for best_model
    secondary_val_dl = None
    secondary_val_dataset = None
    secondary_val_seqs = config['data'].get('secondary_val_seqs', [])
    if secondary_val_seqs:
        secondary_val_dataset = MOT17Dataset(
            root=config['data']['root'], seqs=secondary_val_seqs,
            img_size=tuple(config['data']['img_size']),
            min_visibility=config['data'].get('min_visibility', 0.3),
            augment=False, frame_skip=frame_skip
        )
        secondary_val_dl = DataLoader(
            secondary_val_dataset, batch_size=config['data']['batch_size'], shuffle=False,
            num_workers=config['data']['num_workers'], pin_memory=True,
            persistent_workers=True, collate_fn=collate_fn
        )
        print(f"Secondary Validation: {len(secondary_val_dataset)} imgs ({len(secondary_val_seqs)} seqs)")

    # Global-ID validation (cross-date ReID) — informational only, not used for best_model
    global_id_val_dl = None
    global_id_val_dataset = None
    global_id_val_seqs = config['data'].get('global_id_val_seqs', [])
    if global_id_val_seqs:
        global_id_val_dataset = MOT17Dataset(
            root=config['data']['root'], seqs=global_id_val_seqs,
            img_size=tuple(config['data']['img_size']),
            min_visibility=config['data'].get('min_visibility', 0.3),
            augment=False, frame_skip=frame_skip
        )
        global_id_val_dl = DataLoader(
            global_id_val_dataset, batch_size=config['data']['batch_size'], shuffle=False,
            num_workers=config['data']['num_workers'], pin_memory=True,
            persistent_workers=True, collate_fn=collate_fn
        )
        print(f"Global-ID Validation (cross-date): {len(global_id_val_dataset)} imgs ({len(global_id_val_seqs)} seqs)")

    print(f"Training: {len(train_dataset)} imgs, Validation: {len(val_dataset)} imgs")
    
    # ─── Model Creation (ConvNeXt V2 + BiFPN + CenterNet) ──────────────
    w, h = config['data']['img_size']
    reid_extraction = config['model'].get('reid_extraction', 'center')
    eval_use_bnneck = resolve_eval_use_bnneck(config)
    loss_cfg = config.get('loss', {})
    model_meta = build_model_meta(config.get('model', {}))
    model = build_jdenet_from_config(config)
    pretrained_criterion_state = None
    
    # Load Weights
    if config['model'].get('pretrained_path'):
        p = config['model']['pretrained_path']
        if os.path.exists(p):
            print(f"Loading weights from {p}")
            ckpt = torch.load(p, map_location='cpu')
            sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            pretrained_criterion_state = ckpt.get('criterion_state_dict') if isinstance(ckpt, dict) else None
            
            # Filter out mismatched layers (e.g. reid_head depth changes)
            model_sd = model.state_dict()
            filtered_sd = {}
            skipped_keys = []
            for k, v in sd.items():
                if k in model_sd:
                    if model_sd[k].shape == v.shape:
                        filtered_sd[k] = v
                    else:
                        skipped_keys.append(k)
                else:
                    skipped_keys.append(k)
            
            if skipped_keys:
                print(f"  Skipped {len(skipped_keys)} keys with shape mismatch: {skipped_keys[:5]}...")
            
            model.load_state_dict(filtered_sd, strict=False)
            print(f"  Loaded {len(filtered_sd)}/{len(sd)} weights from checkpoint")
    
    model.to(device)
    
    # ─── Loss + Criterion ───────────────────────────────────────────────
    # Pass model's part_extractor to loss so both training and inference use the same instance
    head_part_extractor = getattr(model.head, 'part_extractor', None)
    criterion = CenterNetLoss(
        num_classes=1,
        embedding_dim=config['model']['embedding_dim'],
        num_identities=train_dataset.num_identities,
        hm_weight=loss_cfg.get('cls_weight', 1.0),
        wh_weight=loss_cfg.get('reg_weight', 0.1),
        off_weight=loss_cfg.get('ctr_weight', 1.0),
        id_weight=loss_cfg.get('id_weight', 0.0),  # 0 for pretrain, 1 for finetune
        rep_weight=loss_cfg.get('rep_weight', 0.2),
        iou_weight=loss_cfg.get('iou_weight', 1.0),
        use_visibility_weighted_loss=loss_cfg.get('use_visibility_weighted_loss', True),
        reid_extraction=reid_extraction,
        part_extractor=head_part_extractor,
        bnneck=getattr(model.head, 'reid_bnneck', None),
        part_bnneck=getattr(model.head, 'part_bnneck', None),
        num_parts_v=config['model'].get('num_parts_v', 4),
        num_parts_h=config['model'].get('num_parts_h', 1),
        roi_output_size=tuple(config['model'].get('roi_output_size', [32, 8])),
        part_loss_weight=loss_cfg.get('part_loss_weight', 0.5),
        use_uncertainty_weighting=loss_cfg.get('uncertainty_loss', False),
        arcface_s=loss_cfg.get('arcface_s', 16.0),
        arcface_m=loss_cfg.get('arcface_m', 0.25),
        arcface_subcenter_k=loss_cfg.get('arcface_subcenter_k', 1),
        triplet_weight=loss_cfg.get('triplet_weight', 0.3),
        triplet_margin=loss_cfg.get('triplet_margin', 0.5),
        reid_dropout=loss_cfg.get('reid_dropout', 0.1),
        diversity_loss_weight=loss_cfg.get('diversity_loss_weight', 0.0),
        label_smoothing=loss_cfg.get('label_smoothing', 0.0),
        reid_stride_ratio=config['model'].get('reid_stride', 4) // 4,
        det_stride_ratio=config['model'].get('det_stride', 4) // 4,
        loss_cfg=loss_cfg,
    ).to(device)

    if pretrained_criterion_state is not None and not config['model'].get('resume'):
        print("Loading compatible criterion/head-adjacent state from pretrained_path")
        _restore_criterion_state(criterion, model, pretrained_criterion_state, prefix="Pretrained criterion")

    # ─── Knowledge Distillation Teacher (OSNet) ─────────────────────────
    teacher_model = None
    distill_weight = loss_cfg.get('distillation_weight', 0.0)
    if distill_weight > 0:
        osnet_path = loss_cfg.get('teacher_weights', 'osnet_x0_25_msmt17.pt')
        from .osnet_teacher import build_osnet_teacher
        teacher_model = build_osnet_teacher(osnet_path, device)

    # ArcFace warm-up config: ramp id_weight from 0 to id_weight over N epochs
    id_warmup_epochs = int(config['training'].get('id_warmup_epochs', 0))

    # ─── Optimizer + Scheduler ──────────────────────────────────────────
    base_lr = float(config['training']['lr'])
    backbone_lr_mult = float(config['training'].get('backbone_lr_mult', 1.0))

    # Collect only trainable parameters
    if backbone_lr_mult < 1.0:
        # Split backbone into pretrained (slow LR) vs randomly-init modules (full LR).
        # The P3 refiner, BiFPN, and fusion are randomly initialized — they need full
        # LR to learn. Only the pretrained ConvNeXt stages need protection at 0.1×.
        # Without this split, the refiner+fusion get 1e-5 LR and can't learn to
        # transform raw backbone features → the heatmap collapses (cls=13.82 stuck).
        pretrained_params = list(model.backbone.model.parameters())  # ConvNeXt stages
        neck_params = []  # refiner + BiFPN + fusion (randomly init, need full LR)
        pretrained_ids = {id(p) for p in pretrained_params}
        for p in model.backbone.parameters():
            if id(p) not in pretrained_ids:
                neck_params.append(p)
        head_params = list(model.head.parameters())
        criterion_params = list(criterion.parameters())
        param_groups = [
            {'params': pretrained_params, 'lr': base_lr * backbone_lr_mult},
            {'params': neck_params + head_params, 'lr': base_lr},
            {'params': criterion_params, 'lr': base_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=0.01)
        print(f"  Differential LR: backbone={base_lr * backbone_lr_mult:.6f}, "
              f"neck+head={base_lr:.6f} ({len(neck_params)} neck params)")
    else:
        params = list(model.parameters()) + list(criterion.parameters())
        optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=0.01)
    
    # Scheduler
    total_epochs = int(config['training']['epochs'])
    warmup_epochs = int(config['training']['warmup_epochs'])
    warmup_batches = int(config['training'].get('warmup_batches', 1000))
    scheduler_type = config['training'].get('scheduler', 'cosine_warm_restarts')
    min_lr = float(config['training'].get('min_lr', 1e-6))
    if scheduler_type == 'cosine':
        # Simple cosine decay — no warm restarts (smoother for ReID convergence)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=min_lr
        )
        print(f"  📐 Scheduler: CosineAnnealingLR (T_max={total_epochs}, eta_min={min_lr})")
    else:
        # Cosine Warm Restarts (legacy default)
        # T_0=5, T_mult=2: first cycle E0-E4, second cycle E5-E14
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=min_lr
        )
        print(f"  📐 Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2)")
    
    # Store initial LR per param group for per-batch linear warmup
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']
    global_step = 0  # Tracks total optimizer steps across all epochs
    
    # AMP
    scaler = torch.amp.GradScaler('cuda', enabled=config['training'].get('use_amp', False))
    
    # EMA
    ema = None
    if config['training'].get('use_ema', False):
        ema = ModelEMA(model, decay=config['training'].get('ema_decay', 0.9999))
        print("✅ EMA Enabled")

    # ─── Checkpoint Resume + State Restoration ──────────────────────────
    best_score = 0.0
    start_epoch = 0
    start_batch = 0              # Mid-epoch resume: 0 = start from beginning
    if config['model'].get('resume') and os.path.exists(config['model']['resume']):
         print(f"Resuming from {config['model']['resume']}")
         ckpt = torch.load(config['model']['resume'], map_location='cpu')
         
         # Smart loading: filter out shape mismatches (e.g. ReID head dim change)
         # Also migrate renamed keys (e.g. post_refine.X → post_refine.0.X after Sequential wrap)
         ckpt_sd = ckpt['model_state_dict']
         
         # Key migration: post_refine was single DWSConv, now Sequential([DWSConv, DWSConv])
         # Old: backbone.adaptive_fusion.post_refine.{depthwise,pointwise,bn}.*
         # New: backbone.adaptive_fusion.post_refine.0.{depthwise,pointwise,bn}.*
         migrated_sd = {}
         for k, v in ckpt_sd.items():
             new_k = k
             if 'post_refine.' in k and 'post_refine.0.' not in k and 'post_refine.1.' not in k:
                 new_k = k.replace('post_refine.', 'post_refine.0.')
             migrated_sd[new_k] = v
         if len(migrated_sd) != len(ckpt_sd):
             print(f"  Key migration: {len(ckpt_sd)} → {len(migrated_sd)} keys")
         n_migrated = sum(1 for k, nk in zip(ckpt_sd.keys(), migrated_sd.keys()) if k != nk)
         if n_migrated:
             print(f"  ✅ Migrated {n_migrated} post_refine keys (single→Sequential DWSConv)")
         
         model_state = model.state_dict()
         pretrained_state = {k: v for k, v in migrated_sd.items() 
                             if k in model_state and v.shape == model_state[k].shape}
         
         if len(pretrained_state) < len(migrated_sd):
             skipped = len(migrated_sd) - len(pretrained_state)
             skipped_keys = [k for k in migrated_sd if k not in pretrained_state]
             print(f"⚠️ Warning: {skipped} keys skipped (shape mismatch/missing): {skipped_keys[:10]}...")
             
         model.load_state_dict(pretrained_state, strict=False)
         
         # Restore criterion state (ArcFace weights, uncertainty params)
         if 'criterion_state_dict' in ckpt:
             crit_state = criterion.state_dict()
             crit_loaded = {k: v for k, v in ckpt['criterion_state_dict'].items()
                           if k in crit_state and v.shape == crit_state[k].shape}
             if len(crit_loaded) < len(ckpt['criterion_state_dict']):
                 skipped = len(ckpt['criterion_state_dict']) - len(crit_loaded)
                 skipped_keys = [k for k in ckpt['criterion_state_dict'] if k not in crit_loaded]
                 # Check if ArcFace weights were dropped (ID count change)
                 arcface_dropped = [k for k in skipped_keys if 'arcface' in k or 'part_arcface' in k]
                 if arcface_dropped:
                     print(f"  ⚠️ IMPORTANT: ArcFace weights dropped due to shape mismatch (num_identities changed?)")
                     print(f"     Dropped: {arcface_dropped[:5]}...")
                     print(f"     ArcFace will reinitialize from random — consider using same dataset for resume")
                 else:
                     print(f"  ⚠️ Criterion: {skipped} keys skipped (shape mismatch or removed)")
             criterion.load_state_dict(crit_loaded, strict=False)
             if hasattr(model.head, 'migrate_reid_necks_from_criterion_state'):
                 migrated_bn = model.head.migrate_reid_necks_from_criterion_state(ckpt['criterion_state_dict'])
                 if migrated_bn:
                     print(f"  ✅ Migrated {migrated_bn} BNNeck tensors from criterion checkpoint to model.head")
             if hasattr(criterion, 'migrate_shared_part_classifier_from_state'):
                 migrated_part_cls = criterion.migrate_shared_part_classifier_from_state(ckpt['criterion_state_dict'])
                 if migrated_part_cls:
                     print(f"  ✅ Initialized shared part classifier from {migrated_part_cls} old per-part heads")
             print(f"  ✅ Criterion state restored ({len(crit_loaded)} keys)")
             
             # Backward compat: old checkpoints stored part_extractor in criterion_state_dict.
             # Now part_extractor lives in model.head — migrate trained weights if present.
             if hasattr(model.head, 'part_extractor') and model.head.part_extractor is not None:
                 pe_keys = {k: v for k, v in ckpt['criterion_state_dict'].items()
                           if k.startswith('part_extractor.')}
                 if pe_keys:
                     head_pe_state = model.head.part_extractor.state_dict()
                     migrated = 0
                     for k, v in pe_keys.items():
                         short_k = k.replace('part_extractor.', '', 1)
                         if short_k in head_pe_state and v.shape == head_pe_state[short_k].shape:
                             head_pe_state[short_k] = v
                             migrated += 1
                     if migrated > 0:
                         model.head.part_extractor.load_state_dict(head_pe_state)
                         print(f"  ✅ Migrated {migrated} part_extractor weights from old criterion checkpoint to model.head")
         else:
             print(f"  ⚠️ No criterion_state_dict in checkpoint — ArcFace/uncertainty will reinitialize")
         
         if not config['model'].get('reset_optimizer', False):
             optimizer_loaded = False
             try:
                 optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                 optimizer_loaded = True
             except (ValueError, KeyError) as e:
                 print(f"⚠️ Warning: Optimizer load failed (likely shape mismatch): {e}")
                 print("   Starting with fresh optimizer — resetting epoch to 0 with warmup.")
             
             if optimizer_loaded:
                 start_epoch = ckpt['epoch']
                 start_batch = ckpt.get('batch_idx', 0)   # Mid-epoch resume: skip already-processed batches
                 best_score = ckpt.get('best_combined_score', ckpt.get('best_mAP', 0.0))
                 if start_batch > 0:
                     print(f"   \u2705 Resuming from epoch {start_epoch}, batch {start_batch}, best_score={best_score:.4f}")
                 else:
                     print(f"   \u2705 Resuming from epoch {start_epoch}, best_score={best_score:.4f}")
                 # Skip per-batch warmup for resumed training (already warmed up)
                 global_step = warmup_batches + 1
             else:
                 # Optimizer failed — treat as fine-tune: fresh optimizer, epoch 0, warmup active
                 print("   Fine-tuning mode: model weights loaded, optimizer/epoch/scheduler reset.")
             
             # Load scheduler/EMA state only if full resume succeeded
             if optimizer_loaded:
                 if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
                     try:
                         scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                         print(f"   ✅ Scheduler state restored")
                     except Exception as e:
                         print(f"   ⚠️ Scheduler load failed: {e}")
                 if ema and 'ema_state_dict' in ckpt and ckpt['ema_state_dict'] is not None:
                     ema.load_state_dict(ckpt['ema_state_dict'])
                     print(f"   ✅ EMA state restored")
         else:
             print("   Resetting optimizer and epoch count as requested.")
             start_epoch = 0
             start_batch = 0
             best_score = 0.0
             # Sync EMA with resumed model weights so validation starts correct
             if ema:
                 from effiped.ema import copy_params_to_ema
                 copy_params_to_ema(ema.ema, model)
                 print("   ✅ EMA synced to resumed model weights")

    # ─── Validate-Only Mode ────────────────────────────────────────────
    def _scheduled_eval_flags(epoch_idx: int):
        """Delay part-composed validation until part branches have useful signal."""
        training_cfg = config.get('training', {})
        tracker_cfg = config.get('tracker', {})
        concat_base = bool(training_cfg.get('concat_inference', False))
        concat_start = int(training_cfg.get('concat_inference_start_epoch', 0))
        part_start = int(training_cfg.get('part_aware_eval_start_epoch', concat_start))

        concat_eval = concat_base and epoch_idx >= concat_start
        part_alpha = float(tracker_cfg.get('part_aware_alpha', 0.0))
        if epoch_idx < part_start:
            part_alpha = 0.0
        return concat_eval, part_alpha

    if config.get('_validate_only', False):
        print("\n" + "=" * 60)
        print("VALIDATE-ONLY MODE")
        print("=" * 60)
        eval_model = ema.ema if ema else model
        if hasattr(criterion, 'current_epoch'):
            criterion.current_epoch = 10**9
        mAP, mAP_50, rec, v_loss, v_acc, r1, r1_same, r1_cross, mar50 = evaluate(
            eval_model, val_dl, device, criterion,
            reid_extraction=reid_extraction, dataset=val_dataset,
            eval_use_bnneck=eval_use_bnneck,
            part_aware_alpha=config.get('tracker', {}).get('part_aware_alpha', 0.0),
            concat_inference=config.get('training', {}).get('concat_inference', False)
        )
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"  mAP@0.5:      {mAP_50:.4f}")
        print(f"  mAP@0.5:0.95: {mAP:.4f}")
        print(f"  mAR@50:       {mar50:.4f}")
        print(f"  Recall@100:   {rec:.4f}")
        print(f"  ReID R1:      {r1:.4f}")
        print(f"  R1 same-cam:  {r1_same:.4f}")
        print(f"  R1 cross-cam: {r1_cross:.4f}")
        print(f"  Val loss:     {v_loss:.4f}")
        print("=" * 60)
        return

    # ─── Epoch Training Loop ────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):
        # Per-batch warmup is applied inside the training loop (see below)
        # This replaces the old epoch-level warmup which was a no-op at warmup_epochs=1

        model.train()
        criterion.train()
        if hasattr(criterion, 'current_epoch'):
            criterion.current_epoch = epoch
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{total_epochs}")

        # Gradual ArcFace warm-up: ramp id_weight from 0→1.0 over id_warmup_epochs
        # epoch 0 → 0.0 (detection-only), epoch N → 1.0 (full ReID)
        # This prevents detection feature corruption when ReID loss activates
        if id_warmup_epochs > 0:
            id_weight_scale = min(1.0, epoch / id_warmup_epochs)
        else:
            id_weight_scale = 1.0

        # Accumulators
        acc_loss = torch.zeros(1, device=device)
        acc_cls = torch.zeros(1, device=device)
        acc_reg = torch.zeros(1, device=device)
        acc_id = torch.zeros(1, device=device)
        acc_iou = torch.zeros(1, device=device)
        acc_rep = torch.zeros(1, device=device)
        acc_id_acc = torch.zeros(1, device=device)
        acc_dcn = torch.zeros(1, device=device)
        acc_distill = 0.0  # KD relational distill loss (scalar, read from criterion attr)
        acc_feat_distill = 0.0  # KD feature-level distill loss
        acc_part_entropy_raw = 0.0
        acc_part_collapse_raw = 0.0
        acc_part_entropy_fused = 0.0
        acc_part_collapse_fused = 0.0
        acc_tiny_roi = 0.0
        part_diag_batches = 0
        valid_batches = 0
        nan_count = 0
        
        # Gradient accumulation: simulate larger batch size without extra VRAM
        accumulation_steps = config['training'].get('gradient_accumulation_steps', 4)
        optimizer.zero_grad()
        
        # Mid-epoch resume: skip already-processed batches at the SAMPLER level.
        # This avoids wasting ~30-40 min loading 20K images just to discard them.
        # The sampler skips batch indices < start_batch, so DataLoader workers
        # never load those images at all.
        batch_offset = 0
        if start_batch > 0 and epoch == start_epoch:
            print(f"\n   ⏩ Fast-skip: batches 0–{start_batch-1} excluded at sampler level (no I/O)")
            if hasattr(train_dl, 'batch_sampler') and hasattr(train_dl.batch_sampler, 'set_skip_batches'):
                train_dl.batch_sampler.set_skip_batches(start_batch)
                batch_offset = start_batch
                pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{total_epochs}",
                            total=len(train_dl.batch_sampler) - start_batch)
            else:
                # Fallback for non-custom samplers: islice (still loads skipped data)
                data_iter = itertools.islice(train_dl, start_batch, None)
                batch_offset = start_batch
                pbar = tqdm(data_iter, desc=f"Epoch {epoch+1}/{total_epochs}",
                            total=len(train_dl) - start_batch, initial=0)
        
        raw_idx = -1  # Track last processed batch (-1 = no batches processed)
        for raw_idx, batch_data in enumerate(pbar):
            batch_idx = raw_idx + batch_offset
            
            # Unpack batch: (imgs, targets) or (imgs, targets, cam_ids)
            if len(batch_data) == 3:
                imgs, targets, cam_ids = batch_data
                cam_ids = cam_ids.to(device, non_blocking=True)
            else:
                imgs, targets = batch_data
                cam_ids = None
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=config['training'].get('use_amp', False)):
                outputs = model(imgs)
                # Extract teacher embeddings for knowledge distillation
                teacher_embs_tuple = None
                if teacher_model is not None:
                    from .osnet_teacher import extract_teacher_embeddings
                    teacher_embs_tuple = extract_teacher_embeddings(teacher_model, imgs, targets)
                result = criterion(outputs, targets, id_weight_scale=id_weight_scale,
                                   cam_ids=cam_ids, teacher_embeddings=teacher_embs_tuple)
                # (total_loss, loss_hm, loss_reg, loss_id, id_acc, loss_rep, loss_iou, loss_dcn_penalty)
                loss, l_cls, l_reg, l_id, id_acc, l_rep, l_iou, l_dcn = result
                # Scale loss by accumulation steps to keep gradient magnitude consistent
                loss = loss / accumulation_steps
            
            if torch.isnan(loss):
                nan_count += 1
                print(f"Warning: NaN loss at batch {batch_idx} (total NaN count: {nan_count})")
                # Flush stale accumulated gradients to prevent corruption
                optimizer.zero_grad()
                continue
                
            scaler.scale(loss).backward()
            
            # Only step optimizer every accumulation_steps batches
            # Use raw_idx (not batch_idx) for alignment — batch_offset from mid-epoch
            # resume may not be aligned to accumulation_steps, causing undersized first window
            if (raw_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                all_params = list(model.parameters()) + list(criterion.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, config['training'].get('grad_clip', 10.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                
                # Per-batch linear warmup: ramp LR from 1% to 100% over warmup_batches
                # Replaces epoch-level warmup which was a no-op at warmup_epochs=1
                # This lets AdamW's v_t (EMA of squared gradients) stabilize before
                # applying full-scale updates to freshly initialized parameters
                if global_step <= warmup_batches:
                    warmup_factor = max(0.01, global_step / warmup_batches)
                    for pg in optimizer.param_groups:
                        pg['lr'] = pg['initial_lr'] * warmup_factor
                
                if ema: ema.update(model)
            
            # Undo the scaling for display (multiply back by accumulation_steps)
            acc_loss += (loss.detach() * accumulation_steps)
            acc_cls += l_cls.detach()
            acc_reg += l_reg.detach()
            acc_id += l_id.detach()
            acc_rep += l_rep.detach()
            acc_iou += l_iou.detach()
            acc_dcn += l_dcn.detach()
            acc_distill += getattr(criterion, '_last_loss_distill', 0.0)
            acc_feat_distill += getattr(criterion, '_last_loss_feat_distill', 0.0)
            part_entropy_raw = float(getattr(criterion, '_last_part_attention_entropy_raw', 0.0) or 0.0)
            part_collapse_raw = float(getattr(criterion, '_last_part_collapse_rate_raw', 0.0) or 0.0)
            part_entropy_fused = float(getattr(criterion, '_last_part_attention_entropy_fused', 0.0) or 0.0)
            part_collapse_fused = float(getattr(criterion, '_last_part_collapse_rate_fused', 0.0) or 0.0)
            tiny_roi = float(getattr(criterion, '_last_tiny_roi_rate', 0.0) or 0.0)
            if reid_extraction == 'part_based':
                acc_part_entropy_raw += part_entropy_raw
                acc_part_collapse_raw += part_collapse_raw
                acc_part_entropy_fused += part_entropy_fused
                acc_part_collapse_fused += part_collapse_fused
                acc_tiny_roi += tiny_roi
                part_diag_batches += 1
            
            id_acc_is_valid = bool(torch.isfinite(id_acc).item())
            if id_acc_is_valid:
                acc_id_acc += id_acc.detach()
                valid_batches += 1
                
            postfix = {
                'loss': f'{(loss.item() * accumulation_steps):.2f}',
                'cls': f'{l_cls.item():.2f}',
                'reg': f'{l_reg.item():.2f}',
                'id': f'{l_id.item():.2f}',
                'rep': f'{l_rep.item():.2f}',
                'iou': f'{l_iou.item():.2f}',
                'acc': f'{id_acc.item():.2f}' if id_acc_is_valid else 'n/a'
            }
            if l_dcn.item() > 0:
                postfix['dcn'] = f'{l_dcn.item():.3f}'
            _distill_val = getattr(criterion, '_last_loss_distill', 0.0)
            _feat_distill_val = getattr(criterion, '_last_loss_feat_distill', 0.0)
            if _distill_val > 0:
                postfix['kd'] = f'{_distill_val:.3f}'
            if _feat_distill_val > 0:
                postfix['fkd'] = f'{_feat_distill_val:.3f}'
            if reid_extraction == 'part_based':
                postfix['pHraw'] = f'{part_entropy_raw:.2f}'
                postfix['pHfuse'] = f'{part_entropy_fused:.2f}'
                postfix['pCol'] = f'{part_collapse_raw:.2f}'
                postfix['tiny'] = f'{tiny_roi:.2f}'
            pbar.set_postfix(postfix)

            # Mid-epoch checkpoint every 1000 batches (saves ~15min of work on crash)
            # Single rolling file — overwrites each time to avoid clutter.
            if (batch_idx + 1) % 1000 == 0:
                mid_ckpt_path = os.path.join(run_ckpt_dir, 'checkpoint_mid_epoch.pth')
                mid_payload = {
                    'epoch': epoch,
                    'batch_idx': batch_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'ema_state_dict': ema.state_dict() if ema else None,
                    'criterion_state_dict': criterion.state_dict(),
                    'best_combined_score': best_score,
                    'model_meta': model_meta,
                    'run_id': run_id,
                }
                _save_checkpoint(
                    mid_payload,
                    mid_ckpt_path,
                    [
                        os.path.join(ckpt_dir, 'checkpoint_mid_epoch.pth'),
                        os.path.join(save_dir, 'checkpoint_mid_epoch.pth'),
                    ],
                )
                print(f"\n   💾 Mid-epoch checkpoint saved: {mid_ckpt_path} (epoch {epoch}, batch {batch_idx+1})")

        # ─── End-of-epoch gradient flush ───
        # If the last batch didn't align with accumulation_steps, flush residual gradients
        # Guard: raw_idx == -1 if all batches were skipped (e.g., mid-epoch resume at last batch)
        if raw_idx >= 0 and (raw_idx + 1) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            all_params = list(model.parameters()) + list(criterion.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, config['training'].get('grad_clip', 10.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if global_step <= warmup_batches:
                warmup_factor = max(0.01, global_step / warmup_batches)
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['initial_lr'] * warmup_factor
            if ema: ema.update(model)

        # ─── Epoch-End: Scheduler Step + Metrics + Validation ───────────
        scheduler.step()
        
        # Compute actual batch count (accounts for mid-epoch resume skipping)
        actual_batches = len(train_dl) - (start_batch if epoch == start_epoch else 0)
        actual_batches = max(actual_batches, 1)  # safety
        
        avg_loss = acc_loss.item() / actual_batches
        avg_cls = acc_cls.item() / actual_batches
        avg_reg = acc_reg.item() / actual_batches
        avg_id = acc_id.item() / actual_batches
        avg_rep = acc_rep.item() / actual_batches
        avg_iou = acc_iou.item() / actual_batches
        avg_dcn = acc_dcn.item() / actual_batches
        avg_distill = acc_distill / actual_batches
        avg_feat_distill = acc_feat_distill / actual_batches
        avg_acc = acc_id_acc.item() / valid_batches if valid_batches > 0 else 0
        avg_part_entropy_raw = acc_part_entropy_raw / part_diag_batches if part_diag_batches > 0 else 0.0
        avg_part_collapse_raw = acc_part_collapse_raw / part_diag_batches if part_diag_batches > 0 else 0.0
        avg_part_entropy_fused = acc_part_entropy_fused / part_diag_batches if part_diag_batches > 0 else 0.0
        avg_part_collapse_fused = acc_part_collapse_fused / part_diag_batches if part_diag_batches > 0 else 0.0
        avg_tiny_roi = acc_tiny_roi / part_diag_batches if part_diag_batches > 0 else 0.0
        
        # Validation
        mAP = mAP_50 = rec = v_loss = v_acc = r1 = mar50 = None
        val_freq = config['training'].get('val_freq', 1)
        if (epoch + 1) % val_freq == 0:
            eval_model = ema.ema if ema else model
            concat_eval, part_alpha_eval = _scheduled_eval_flags(epoch)
            mAP, mAP_50, rec, v_loss, v_acc, r1, r1_same, r1_cross, mar50 = evaluate(eval_model, val_dl, device, criterion, reid_extraction=reid_extraction, dataset=val_dataset, eval_use_bnneck=eval_use_bnneck, part_aware_alpha=part_alpha_eval, concat_inference=concat_eval)
            
            current_lr = optimizer.param_groups[-1]['lr']  # head/criterion LR
            log_row = [epoch + 1, f'{current_lr:.6f}', avg_loss, avg_cls, avg_reg, avg_id, avg_iou, avg_rep, avg_dcn, avg_distill, avg_feat_distill, avg_acc, avg_part_entropy_raw, avg_part_collapse_raw, avg_part_entropy_fused, avg_part_collapse_fused, avg_tiny_roi, mAP, mAP_50, v_loss, v_acc, r1, rec, r1_same, r1_cross, mar50]
            for _log_path in (log_file, legacy_log_file):
                with open(_log_path, 'a', newline='') as f:
                    csv.writer(f).writerow(log_row)

            # Combined score: balance detection (mAP, recall) with ReID (Rank-1)
            # Default: ReID-dominant (0.3/0.2/0.5) — detection saturates early
            # Override via config training.best_model_weights: [mAP_w, mAR_w, R1_w]
            bm_weights = config.get('training', {}).get('best_model_weights', [0.3, 0.2, 0.5])
            combined_score = bm_weights[0] * mAP + bm_weights[1] * rec + bm_weights[2] * r1
            
            is_best = False
            if combined_score > best_score:
                best_score = combined_score
                is_best = True
                best_payload = {
                    'epoch': epoch + 1,
                    'model_state_dict': eval_model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema_state_dict': ema.state_dict() if ema else None,
                    'model_meta': model_meta,
                    'best_mAP': mAP,  # mAP@[.5:.95] (COCO-style)
                    'best_recall': rec,  # mAR@100
                    'best_combined_score': combined_score,
                    'run_id': run_id,
                }
                _save_checkpoint(
                    best_payload,
                    os.path.join(run_ckpt_dir, 'best_model.pth'),
                    [
                        os.path.join(ckpt_dir, 'best_model.pth'),
                        os.path.join(save_dir, 'best_model.pth'),
                    ],
                )
            
            # Print score every epoch (user request)
            mark = "✅ New Best Model" if is_best else "   Current Model"
            print(f"{mark}: score={combined_score:.4f} (mAP_coco={mAP:.4f}, mAR@100={rec:.4f}, R1={r1:.4f})")

            # === Secondary Validation (MOT-type scenarios) — informational only ===
            if secondary_val_dl is not None:
                print("\n--- Secondary Validation (MOT scenarios) ---")
                s_mAP, s_mAP50, s_rec, s_vloss, s_vacc, s_r1, _, _, _ = evaluate(
                    eval_model, secondary_val_dl, device, criterion,
                    reid_extraction=reid_extraction, dataset=secondary_val_dataset,
                    eval_use_bnneck=eval_use_bnneck,
                    part_aware_alpha=part_alpha_eval,
                    concat_inference=concat_eval
                )
                print(f"  MOT Val: mAP={s_mAP:.4f}, mAP@50={s_mAP50:.4f}, mAR={s_rec:.4f}, R1={s_r1:.4f}")
                print("--- End Secondary Validation ---\n")

            # === Global-ID Validation (cross-date ReID) — informational only ===
            if global_id_val_dl is not None:
                print("\n--- Cross-Date Validation (global person IDs) ---")
                g_mAP, g_mAP50, g_rec, g_vloss, g_vacc, g_r1, g_r1_same, g_r1_cross, g_mar50 = evaluate(
                    eval_model, global_id_val_dl, device, criterion,
                    reid_extraction=reid_extraction, dataset=global_id_val_dataset,
                    eval_use_bnneck=eval_use_bnneck,
                    part_aware_alpha=part_alpha_eval,
                    concat_inference=concat_eval
                )
                print(f"  Cross-Date R1_cross: {g_r1_cross*100:.1f}% | R1_same: {g_r1_same*100:.1f}%")
                print("--- End Cross-Date Validation ---\n")

        val_str = f" | mAP: {mAP:.4f} R1: {r1:.4f}" if mAP is not None else " | mAP: n/a R1: n/a"
        dcn_str = f" DCNPen: {avg_dcn:.4f}" if avg_dcn > 0 else ""
        part_diag_str = ""
        if part_diag_batches > 0:
            part_diag_str = (f" PartHraw: {avg_part_entropy_raw:.3f}"
                             f" PartHfused: {avg_part_entropy_fused:.3f}"
                             f" PartCollapseRaw: {avg_part_collapse_raw:.3f}"
                             f" TinyRoI: {avg_tiny_roi:.3f}")
        print(f"Epoch {epoch+1} Done. Loss: {avg_loss:.4f} Acc: {avg_acc:.4f} ID_scale: {id_weight_scale:.2f}{dcn_str}{part_diag_str}{val_str}")
        
        # Epoch-end uncertainty diagnostics
        if hasattr(criterion, 's_hm'):
            try:
                # Show CLAMPED effective weights (matching loss.py forward)
                s_hm_c = criterion.s_hm.clamp(-4, 4)
                s_wh_c = criterion.s_wh.clamp(-4, 4)
                s_off_c = criterion.s_off.clamp(-4, 4)
                s_id_c = criterion.s_id.clamp(-4, 0.0)
                s_rep_c = criterion.s_rep.clamp(-1.1, 4)
                s_iou_c = criterion.s_iou.clamp(-4, 4)
                w_hm = torch.exp(-s_hm_c).item()
                w_wh = torch.exp(-s_wh_c).item()
                w_off = torch.exp(-s_off_c).item()
                w_id = torch.exp(-s_id_c).item()
                w_rep = torch.exp(-s_rep_c).item()
                w_iou = torch.exp(-s_iou_c).item()
                print(f"  [Uncertainty Log] Effective Weights (clamped): hm={w_hm:.2f}, wh={w_wh:.2f}, off={w_off:.2f}, id={w_id:.2f}, rep={w_rep:.2f}, iou={w_iou:.2f}")
                print(f"                    Raw Params (s): hm={criterion.s_hm.item():.2f}, wh={criterion.s_wh.item():.2f}, off={criterion.s_off.item():.2f}, id={criterion.s_id.item():.2f}, rep={criterion.s_rep.item():.2f}, iou={criterion.s_iou.item():.2f}")
            except Exception:
                pass  # Non-critical diagnostics — don't interrupt training


        
        # Save Regular Checkpoint — per-epoch file for history/recovery
        if (epoch + 1) % config['training'].get('save_freq', 1) == 0:
            epoch_payload = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.state_dict() if ema else None,
                'model_meta': model_meta,
                'best_combined_score': best_score,
                'run_id': run_id,
            }
            epoch_name = f'checkpoint_epoch_{epoch+1}.pth'
            _save_checkpoint(
                epoch_payload,
                os.path.join(run_ckpt_dir, epoch_name),
                [
                    os.path.join(ckpt_dir, epoch_name),
                    os.path.join(save_dir, epoch_name),
                ],
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='effiped/config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Override model.resume from the config')
    parser.add_argument('--no-auto-resume', action='store_true', help='Disable default recovery from this experiment folder')
    parser.add_argument('--validate-only', action='store_true', help='Run validation only (no training)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.resume:
        config.setdefault('model', {})['resume'] = args.resume
    if args.no_auto_resume:
        config.setdefault('training', {})['auto_resume'] = False
    config['_config_path'] = os.path.abspath(args.config)  # Stash for experiment snapshot
    config['_validate_only'] = args.validate_only
    
    try:
        train(config)
    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("🛑 User stopped session (KeyboardInterrupt)")
        print("   Training was interrupted by user.")
        print("   Checkpoints saved up to the last completed epoch.")
        print("="*50)
