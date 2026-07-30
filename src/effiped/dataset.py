"""
effiped Dataset — JDE-format Multi-Camera Pedestrian Dataset
═══════════════════════════════════════════════════════════════════

Table of Contents (approximate line numbers):
─────────────────────────────────────────────
  L43    Constants & Defaults
  L55    letterbox() — Aspect-ratio-preserving resize with padding
  L85    MOT17Dataset class
  L86      __init__() — Parse sequences, label resolution, ID compaction
  L214     _compact_id_space() — Sparse → dense ID remapping for ArcFace
  L266     _get_camera_int() — Camera string → integer mapping
  L285     extract_camera_string() — Path-based camera ID heuristics
  L337     _load_image_list() — Text file list loading + label path discovery
  L460     _load_sequence() — MOT-format gt.txt + seqinfo.ini loading
  L527     Augmentation helpers — scale, jitter, erasing, blur, surveillance,
           channel_augment, clahe, color_transfer
  L678     _load_image_and_labels() — Single image + label loader (with retry)
  L698     _mosaic4() / _mixup() — Mosaic-4 and MixUp composition
  L765     __getitem__() — Main entry: load, augment, letterbox, normalize
  L878   SequenceGroupedSampler — Consecutive-frame batch sampling
  L983   IdentityAwareSampler — Temporal-gap + cross-camera sampling
  L1165  collate_fn() — Batch collation (2-tuple and 3-tuple support)

Key Invariants:
  • Label format: class_id  person_id  cx  cy  w  h  (normalized to [0,1])
  • __getitem__ returns 3-tuple: (img_tensor, targets_dict, cam_id)
  • ID compaction: sparse IDs → contiguous [0, N-1] at init time
  • Camera ID extracted from directory path for camera-aware losses
  • All labels padded to 8 columns: [cls, id, cx, cy, w, h, conf, vis]
"""

import os
import cv2
import torch
import numpy as np
import configparser
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
from collections import defaultdict


# ── Dataset Constants ──
_MIN_MOSAIC_BOX_PX = 10       # Minimum box dimension in pixels for mosaic labels
_MIN_AREA_RETENTION = 0.3     # Minimum area ratio for clipped boxes to survive
_DOUBLE_NORM_THRESHOLD = 2.0  # If coords exceed this after letterbox, re-normalize
_LABEL_FALLBACK_CLIMB = 4     # Max directory levels to search for labels_with_ids
_MAX_CORRUPT_RETRIES = 10     # Max retries on corrupt images before returning blank


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def letterbox(img, new_shape=(1088, 608), color=(114, 114, 114)):
    """
    Resize image with aspect ratio preservation (letterboxing).
    """
    shape = img.shape[:2]  # current [height, width]
    new_w, new_h = new_shape
    
    # Scale ratio (new / old)
    r = min(new_w / shape[1], new_h / shape[0])
    
    # Compute new unpadded size
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_w - new_unpad[0]) / 2  # width padding
    dh = (new_h - new_unpad[1]) / 2  # height padding
    
    if shape[::-1] != new_unpad:  # resize if needed
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add border (padding)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


# ═══════════════════════════════════════════════════════════════════════════════
# MOT17Dataset — Primary JDE Dataset (detection + tracking + ReID)
# ═══════════════════════════════════════════════════════════════════════════════

class MOT17Dataset(Dataset):
    def __init__(self, root, seqs, img_size=(1088, 608), min_visibility=0.3, augment=False, aug_config=None, frame_skip=1, crowdhuman_ratio=1.0, additional_sources=None):
        self.root = root
        self.img_size = img_size
        self.min_visibility = min_visibility
        self.augment = augment
        self.aug_config = aug_config or {}
        self.frame_skip = frame_skip
        self.crowdhuman_ratio = crowdhuman_ratio
        self.img_files = []
        self.labels = []
        self.labels_is_normalized = []  # per-image flag: True = 0-1 coords, False = pixel coords
        self.camera_ids = []            # per-image integer camera ID
        self.camera_map = {}            # camera_string -> integer mapping
        self.next_camera_id = 0
        self.id_map = {} 
        self.next_global_id = 0
        
        if isinstance(seqs, str) and seqs.endswith(('.train', '.val', '.txt')):
             list_path = seqs if os.path.isabs(seqs) else os.path.join(root, seqs)
             print(f"Loading from file list: {list_path}")
             self._load_image_list(list_path)
        else:
             print(f"Loading MOT dataset from {root}...")
             for seq in seqs:
                 if isinstance(seq, str) and seq.endswith(('.train', '.val', '.txt')):
                     list_path = seq if os.path.isabs(seq) else os.path.join(root, seq)
                     print(f"Loading from file list: {list_path}")
                     if 'crowdhuman' in list_path.lower() and self.crowdhuman_ratio < 1.0:
                         print(f"   Downsampling CrowdHuman to {self.crowdhuman_ratio*100:.1f}%...")
                         self._load_image_list(list_path, sample_ratio=self.crowdhuman_ratio)
                     else:
                         self._load_image_list(list_path)
                 else:
                     self._load_sequence(seq)

        # Load additional datasets (CrowdHuman, MOT20, etc.)
        if additional_sources:
            for source in additional_sources:
                # Support both formats:
                # 1. {root: "...", seqs: ["..."]} (sequence-based)
                # 2. {source: "...", ratio: 0.3} (list-based, like CrowdHuman)
                if 'source' in source:
                    src_path = source['source']
                    ratio = source.get('ratio', 1.0)
                    # Convert to absolute path if needed
                    if not os.path.isabs(src_path):
                        src_path = os.path.join(self.root, src_path)
                    
                    if os.path.exists(src_path):
                        print(f"Loading additional source list: {src_path} (ratio={ratio})")
                        self._load_image_list(src_path, sample_ratio=ratio)
                    else:
                        print(f"⚠️ Warning: Additional source path not found: {src_path}")
                elif 'root' in source and 'seqs' in source:
                    src_root = source['root']
                    print(f"Loading additional sequences from {src_root}...")
                    for s in source['seqs']:
                        self._load_sequence(s, root=src_root)

        # Pad all labels to a uniform 8-column format: [cls, id, cx, cy, w, h, conf, vis]
        # This prevents shape mismatches when mixing .train files (6-col) with MOT sequences (8-col)
        # in mosaic augmentation and collate_fn
        for i in range(len(self.labels)):
            if len(self.labels[i]) > 0 and self.labels[i].shape[1] < 8:
                cols = self.labels[i].shape[1]
                if cols == 7:
                    # 7-col: [cls, id, cx, cy, w, h, vis] -> insert conf=1.0 at index 6
                    conf_col = np.ones((self.labels[i].shape[0], 1), dtype=np.float32)
                    self.labels[i] = np.hstack([self.labels[i][:, :6], conf_col, self.labels[i][:, 6:7]])
                elif cols == 6:
                    # 6-col: [cls, id, cx, cy, w, h] -> append conf=1.0, vis=1.0
                    pad = np.ones((self.labels[i].shape[0], 2), dtype=np.float32)
                    self.labels[i] = np.hstack([self.labels[i], pad])
                else:
                    n_missing = 8 - cols
                    pad = np.ones((self.labels[i].shape[0], n_missing), dtype=np.float32)
                    self.labels[i] = np.hstack([self.labels[i], pad])
            elif len(self.labels[i]) == 0:
                self.labels[i] = np.zeros((0, 8), dtype=np.float32)

        # Compact the ID space: remap sparse IDs (e.g., 0-500 for MOT, 10000-10592 for P-DESTRE)
        # to a dense 0..N-1 range. This prevents a massive ArcFace weight matrix with dead rows.
        self._compact_id_space()

        print(f"Loaded {len(self.img_files)} images and {len(self.labels)} annotations.")
        print(f"Total unique identities: {self.next_global_id}")

        if self.augment:
            self.cj_dict = self.aug_config.get('color_jitter', {})
            self.cj_prob = self.cj_dict.get('probability', 0.7)
            
            re = self.aug_config.get('random_erasing', {})
            self.re_prob = re.get('probability', 0.5)
            self.re_scale = tuple(re.get('scale', [0.02, 0.1]))
            self.re_ratio = tuple(re.get('ratio', [0.3, 3.3]))
            
            self.random_scale = self.aug_config.get('random_scale', [0.8, 1.2])
            self.random_translate = self.aug_config.get('random_translate', 0.1)
            self.horizontal_flip_prob = self.aug_config.get('horizontal_flip', 0.5)
            self.cutout_prob = self.aug_config.get('cutout', 0.1)
            mosaic_cfg = self.aug_config.get('mosaic', {})
            self.mosaic_prob = mosaic_cfg.get('probability', 0.3) 
            self.mixup_prob = mosaic_cfg.get('mixup_probability', 0.1)
            self.mosaic_scale = mosaic_cfg.get('scale', [0.5, 1.5])
            self.aug_apply_prob = self.aug_config.get('apply_prob', 0.9)
            blur_cfg = self.aug_config.get('gaussian_blur', {})
            self.blur_prob = blur_cfg.get('probability', 0.15)
            self.blur_kernel_sizes = blur_cfg.get('kernel_sizes', [3, 5, 7])
            noise_cfg = self.aug_config.get('gaussian_noise', {})
            self.noise_prob = noise_cfg.get('probability', 0.1)
            self.noise_std = noise_cfg.get('std', 15)
            grayscale_cfg = self.aug_config.get('random_grayscale', {})
            self.grayscale_prob = grayscale_cfg.get('probability', 0.0)
            chan_aug_cfg = self.aug_config.get('channel_augment', {})
            self.channel_augment_prob = chan_aug_cfg.get('probability', 0.0)
            clahe_cfg = self.aug_config.get('clahe', {})
            self.clahe_prob = clahe_cfg.get('probability', 0.0)
            color_xfer_cfg = self.aug_config.get('color_transfer', {})
            self.color_transfer_prob = color_xfer_cfg.get('probability', 0.0)
            surv_cfg = self.aug_config.get('surveillance', {})
            self.surv_enabled = surv_cfg.get('enabled', False)
            if self.surv_enabled:
                jpeg_cfg = surv_cfg.get('jpeg_compression', {})
                self.jpeg_prob = jpeg_cfg.get('probability', 0.3)
                self.jpeg_quality_range = jpeg_cfg.get('quality_range', [30, 70])
                res_cfg = surv_cfg.get('resolution_degradation', {})
                self.res_degrade_prob = res_cfg.get('probability', 0.2)
                self.res_scale_range = res_cfg.get('scale_range', [0.3, 0.6])
                gamma_cfg = surv_cfg.get('gamma_shift', {})
                self.gamma_prob = gamma_cfg.get('probability', 0.3)
                self.gamma_range = gamma_cfg.get('gamma_range', [0.7, 1.3])
        
    def _compact_id_space(self):
        """Remap all identity IDs to a dense 0..N-1 range.
        
        This handles the case where P-DESTRE IDs have a large offset (e.g., 10000+)
        creating a sparse ID space that wastes ArcFace classifier capacity.
        IDs < 0 (distractors) are preserved as-is.
        """
        # Collect all unique positive IDs across the dataset
        all_ids = set()
        for labels in self.labels:
            if len(labels) > 0:
                ids = labels[:, 1]
                valid_ids = ids[ids >= 0]
                if len(valid_ids) > 0:
                    all_ids.update(valid_ids.astype(int).tolist())
        
        if not all_ids:
            self.next_global_id = 0
            return
        
        # Check if compaction is needed (sparse if max_id >> count)
        max_id = max(all_ids)
        n_ids = len(all_ids)
        if max_id < n_ids * 1.5:  # Already roughly dense, skip
            return
        
        # Build compact mapping: old_id -> new_id (0-based)
        sorted_ids = sorted(all_ids)
        compact_map = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
        
        print(f"  ID compaction: {n_ids} IDs in range [0, {max_id}] -> [0, {n_ids - 1}]")
        
        # Apply to all labels
        for i in range(len(self.labels)):
            if len(self.labels[i]) > 0:
                ids = self.labels[i][:, 1]
                for j in range(len(ids)):
                    old_id = int(ids[j])
                    if old_id >= 0 and old_id in compact_map:
                        self.labels[i][j, 1] = compact_map[old_id]
        
        self.next_global_id = n_ids

    @property
    def num_identities(self):
        return self.next_global_id

    @property
    def num_cameras(self):
        """Number of distinct cameras detected across the dataset."""
        return self.next_camera_id

    def _get_camera_int(self, camera_str: str) -> int:
        """Map a camera string to a unique integer ID.

        Camera strings are derived from image paths and represent distinct
        viewpoints. Examples: 'MOT17-02-SDP', '2019-01-22_08-00/cam1',
        'rgb_000001_camera1' → 'camera1'.

        Args:
            camera_str: Human-readable camera identifier.

        Returns:
            Integer camera ID (0-indexed, globally unique).
        """
        if camera_str not in self.camera_map:
            self.camera_map[camera_str] = self.next_camera_id
            self.next_camera_id += 1
        return self.camera_map[camera_str]

    @staticmethod
    def extract_camera_string(img_path: str) -> str:
        """Extract a camera identifier string from an image path.

        Heuristics (applied in order):
          1. MMPTrack: filename contains camera name after last '_'
             e.g. 'rgb_000123_camera1.jpg' → 'camera1'
          2. P-DESTRE: path has 'images/{date}/{camera}/frame.jpg'
             → '{date}/{camera}'
          3. MOT/SOMPT22: path has '{seq_name}/img1/frame.jpg'
             → seq_name
          4. CrowdHuman: path has 'CrowdHuman_train/Images/...'
             → 'crowdhuman' (single virtual camera)
          5. Fallback: parent directory name

        Args:
            img_path: Absolute or relative path to the image file.

        Returns:
            Camera identifier string.
        """
        path = img_path.replace('\\', '/')
        parts = path.split('/')
        basename = parts[-1] if parts else ''

        # MMPTrack: rgb_{frame}_{camera}.jpg
        if basename.startswith('rgb_') and basename.count('_') >= 2:
            camera_name = basename.rsplit('.', 1)[0].rsplit('_', 1)[-1]
            # Include session/environment for uniqueness
            for pi, part in enumerate(parts):
                if part.lower() in ('images', 'jde_splits'):
                    if pi + 1 < len(parts) - 1:
                        return '/'.join(parts[pi+1:-1]) + '/' + camera_name
            return camera_name

        # CrowdHuman: detection-only, single virtual camera
        if 'crowdhuman' in path.lower():
            return 'crowdhuman'

        # Search for 'images' or 'img1' anchor in path
        for pi, part in enumerate(parts):
            if part.lower() == 'images' and pi + 2 < len(parts):
                # P-DESTRE: images/date/camera/frame → "date/camera"
                return '/'.join(parts[pi+1:-1])
            if part.lower() == 'img1' and pi > 0:
                # MOT/SOMPT22: .../MOT17-02-SDP/img1/... → seq name
                return parts[pi-1]

        # Fallback: parent directory
        if len(parts) >= 2:
            return parts[-2]
        return 'unknown'
        
    def _load_image_list(self, list_path, sample_ratio=1.0):
        with open(list_path, 'r') as f:
            lines = f.readlines()
        img_paths = [x.strip() for x in lines if x.strip()]
        if sample_ratio < 1.0:
            original_len = len(img_paths)
            rng = random.Random(42)
            rng.shuffle(img_paths)
            keep_len = int(original_len * sample_ratio)
            img_paths = img_paths[:keep_len]
        
        for img_path in img_paths:
            if not os.path.exists(img_path):
                rel_path = os.path.join(self.root, img_path)
                if os.path.exists(rel_path):
                    img_path = rel_path
                else:
                    base_path = os.path.join(self.root, os.path.basename(img_path))
                    if os.path.exists(base_path):
                        img_path = base_path
                    else:
                        continue 
            
            # Normalize separators so os.sep-based replace works on all platforms
            img_path = os.path.normpath(img_path)
            
            # ── Label path resolution (3-tier fallback) ──
            # Tier 1: Replace 'images'/'Images' directory with 'labels_with_ids'
            #   e.g. .../P-DESTRE/images/date/cam/frame.jpg
            #      → .../P-DESTRE/labels_with_ids/date/cam/frame.txt
            if 'images' in img_path:
                label_path = img_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels_with_ids' + os.sep).replace('.jpg', '.txt').replace('.png', '.txt')
            elif 'Images' in img_path:
                label_path = img_path.replace(os.sep + 'Images' + os.sep, os.sep + 'labels_with_ids' + os.sep).replace('.jpg', '.txt').replace('.png', '.txt')
            else:
                # Tier 1 fallback: co-located .txt with same stem as image
                label_path = os.path.splitext(img_path)[0] + '.txt'
                
            if not os.path.exists(label_path):
                # Tier 2: co-located .txt with same stem as image
                label_path_same_dir = os.path.splitext(img_path)[0] + '.txt'
                if os.path.exists(label_path_same_dir):
                    label_path = label_path_same_dir
                else:
                    # Tier 3: walk up directories looking for labels_with_ids/basename.txt
                    # Handles datasets where labels_with_ids is at a different nesting
                    # level than images (e.g., crowdhuman_og: images in
                    # CrowdHuman_train/Images/, labels in labels_with_ids/ at project root)
                    basename_txt = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
                    parent = os.path.dirname(img_path)
                    found = False
                    for _ in range(_LABEL_FALLBACK_CLIMB):  # Try up to 4 levels
                        parent = os.path.dirname(parent)
                        candidate = os.path.join(parent, 'labels_with_ids', basename_txt)
                        if os.path.exists(candidate):
                            label_path = candidate
                            found = True
                            break
                    if not found:
                        continue

            if os.path.getsize(label_path) == 0:
                 gt_data = np.zeros((0, 6))
            else:
                try:
                    # Dynamically determine columns
                    gt_raw = np.loadtxt(label_path, dtype=np.float32)
                    if gt_raw.ndim == 1:
                        gt_raw = gt_raw.reshape(1, -1)
                    
                    # ── Defensive validation ──
                    # Whitespace-only files produce shape (1, 0) → treat as empty
                    if gt_raw.ndim == 2 and gt_raw.shape[1] == 0:
                        gt_raw = np.zeros((0, 6), dtype=np.float32)
                    # Reject labels with < 6 columns (wrong format)
                    if gt_raw.shape[0] > 0 and gt_raw.shape[1] < 6:
                        print(f"⚠️ Skipping malformed label ({gt_raw.shape[1]} cols < 6): {label_path}")
                        continue
                    # Reject labels containing NaN or Inf values
                    if gt_raw.shape[0] > 0 and (np.isnan(gt_raw).any() or np.isinf(gt_raw).any()):
                        print(f"⚠️ Skipping label with NaN/Inf values: {label_path}")
                        continue

                    # Check columns: 6 (class, id, x, y, w, h) or 7 (+vis)
                    cols = gt_raw.shape[1]
                    gt_data = gt_raw
                    
                    if 'CrowdHuman' in img_path or 'crowdhuman' in img_path:
                        gt_data[:, 1] = -1
                except Exception as e:
                    print(f"Error reading {label_path}: {e}")
                    continue

            if len(gt_data) > 0:
                # Namespace IDs at the list-file level, not camera level.
                # This preserves cross-camera identity links within datasets
                # like P-DESTRE (same person across cameras shares one ID),
                # while still isolating IDs between different list files
                # (e.g., P-DESTRE vs CrowdHuman loaded from separate .txt files).
                list_namespace = list_path
                ids = gt_data[:, 1]
                for j in range(len(ids)):
                    orig_id = int(ids[j])
                    if orig_id >= 0:
                        person_key = (list_namespace, orig_id)
                        if person_key not in self.id_map:
                            self.id_map[person_key] = self.next_global_id
                            self.next_global_id += 1
                        ids[j] = self.id_map[person_key]
                gt_data[:, 1] = ids
            
            # Determine normalization state per-image (not globally)
            is_normalized = True
            if len(gt_data) > 0 and gt_data[:, 2:6].max() > 1.1:
                is_normalized = False
            
            self.img_files.append(img_path)
            self.labels.append(gt_data)
            self.labels_is_normalized.append(is_normalized)
            # Extract camera ID
            cam_str = self.extract_camera_string(img_path)
            self.camera_ids.append(self._get_camera_int(cam_str))

    def _load_sequence(self, seq, root=None):
        seq_path = os.path.join(root or self.root, seq)
        img_dir = os.path.join(seq_path, 'img1')
        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        ini_path = os.path.join(seq_path, 'seqinfo.ini')
        seq_config = configparser.ConfigParser()
        seq_config.read(ini_path)
        if not os.path.exists(gt_path): return
        gt_data = np.loadtxt(gt_path, delimiter=',')
        if gt_data.ndim == 1:
            gt_data = gt_data.reshape(1, -1)  # Single-line gt → 2D
        # Filter ground-truth by confidence and visibility:
        #   conf > 0  → keeps class 1 (pedestrian) only.  In MOT17/MOT20 the
        #               conf column is 0 for non-pedestrian classes (vehicle=2,
        #               static=7, distractor=8).  SOMPT22 always has conf=1.
        #   vis ≥ min → drops heavily occluded detections (default threshold 0.3).
        mask = (gt_data[:, 6] > 0) & (gt_data[:, 8] >= self.min_visibility)
        gt_data = gt_data[mask]
        frames = np.unique(gt_data[:, 0]).astype(int)
        if self.frame_skip > 1:
            frames = frames[::self.frame_skip]
        try:
            imWidth = int(seq_config['Sequence']['imWidth'])
            imHeight = int(seq_config['Sequence']['imHeight'])
        except Exception as e:
            print(f"  ⚠️ Skipping sequence (seqinfo.ini error): {seq} — {e}")
            return

        for frame in frames:
            img_name = f"{frame:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path): continue
            frame_mask = (gt_data[:, 0] == frame)
            frame_boxes = gt_data[frame_mask]
            targets = []
            for box in frame_boxes:
                original_id = int(box[1])
                if original_id < 0:
                    global_id = -1
                else:
                    person_key = (seq, original_id)
                    if person_key not in self.id_map:
                        self.id_map[person_key] = self.next_global_id
                        self.next_global_id += 1
                    global_id = self.id_map[person_key]
                x1, y1, w, h = box[2], box[3], box[4], box[5]
                xc, yc = x1 + w / 2, y1 + h / 2
                xc /= imWidth; yc /= imHeight; w /= imWidth; h /= imHeight
                
                # Only class 1 (pedestrian) survives conf > 0 filter
                conf = 1.0
                vis = box[8] if len(box) > 8 else 1.0
                targets.append([0, global_id, xc, yc, w, h, conf, vis])
            self.img_files.append(img_path)
            self.labels.append(np.array(targets))
            self.labels_is_normalized.append(True)  # _load_sequence always normalizes
            # Extract camera ID — for MOT sequences, use seq name
            cam_str = seq if isinstance(seq, str) else self.extract_camera_string(img_path)
            self.camera_ids.append(self._get_camera_int(cam_str))

    def __len__(self):
        return len(self.img_files)

    # ── Augmentation Helpers ─────────────────────────────────────────────────
    # All augmentations operate on raw numpy images (H, W, 3) with pixel-space
    # target coords.  Normalization back to [0,1] happens later in __getitem__.

    def _random_scale_and_translate(self, img, targets, scale_range, translate_ratio):
        h, w = img.shape[:2]
        scale = random.uniform(scale_range[0], scale_range[1])
        new_h, new_w = int(h * scale), int(w * scale)
        img_scaled = cv2.resize(img, (new_w, new_h))
        max_dx = int(translate_ratio * w)
        max_dy = int(translate_ratio * h)
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        out_img = np.zeros((h, w, 3), dtype=img.dtype)
        src_x1, src_y1 = max(0, -dx), max(0, -dy)
        src_x2, src_y2 = min(new_w, w - dx), min(new_h, h - dy)
        dst_x1, dst_y1 = max(0, dx), max(0, dy)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        if src_x2 > src_x1 and src_y2 > src_y1:
            out_img[dst_y1:dst_y2, dst_x1:dst_x2] = img_scaled[src_y1:src_y2, src_x1:src_x2]
        if len(targets) > 0:
            targets[:, 2] *= scale; targets[:, 3] *= scale; targets[:, 4] *= scale; targets[:, 5] *= scale
            targets[:, 2] += dx; targets[:, 3] += dy
            valid_mask = []
            for t in targets:
                xc, yc, tw, th = t[2], t[3], t[4], t[5]
                x1, y1 = xc - tw/2, yc - th/2
                x2, y2 = xc + tw/2, yc + th/2
                x1_clip = max(0, min(w, x1)); y1_clip = max(0, min(h, y1))
                x2_clip = max(0, min(w, x2)); y2_clip = max(0, min(h, y2))
                orig_area = tw * th
                clip_area = (x2_clip - x1_clip) * (y2_clip - y1_clip)
                if orig_area > 0 and clip_area / orig_area > _MIN_AREA_RETENTION:
                    t[2] = (x1_clip + x2_clip) / 2
                    t[3] = (y1_clip + y2_clip) / 2
                    t[4] = x2_clip - x1_clip
                    t[5] = y2_clip - y1_clip
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)
            targets = targets[valid_mask]
        return out_img, targets
    
    def _random_cutout(self, img, num_cuts=3, max_size_ratio=0.2):
        h, w = img.shape[:2]
        for _ in range(num_cuts):
            cut_h = random.randint(int(h * 0.05), int(h * max_size_ratio))
            cut_w = random.randint(int(w * 0.05), int(w * max_size_ratio))
            y = random.randint(0, h - cut_h)
            x = random.randint(0, w - cut_w)
            img[y:y+cut_h, x:x+cut_w] = np.random.randint(0, 255, (cut_h, cut_w, 3), dtype=np.uint8)
        return img
        
    def _np_random_erasing(self, img):
        h, w = img.shape[:2]
        area = h * w
        target_area = random.uniform(self.re_scale[0], self.re_scale[1]) * area
        aspect_ratio = random.uniform(self.re_ratio[0], self.re_ratio[1])
        cut_h = int(round(np.sqrt(target_area * aspect_ratio)))
        cut_w = int(round(np.sqrt(target_area / aspect_ratio)))
        if cut_w < w and cut_h < h:
            x = random.randint(0, w - cut_w)
            y = random.randint(0, h - cut_h)
            img[y:y+cut_h, x:x+cut_w] = np.random.randint(0, 255, (cut_h, cut_w, 3), dtype=np.uint8)
        return img

    def _np_color_jitter(self, img):
        img_float = img.astype(np.float32)
        
        # Random brightness (beta) / contrast (alpha)
        beta = random.uniform(-self.cj_dict.get('brightness', 0.2), self.cj_dict.get('brightness', 0.2)) * 255
        alpha = random.uniform(1 - self.cj_dict.get('contrast', 0.2), 1 + self.cj_dict.get('contrast', 0.2))
        img = np.clip(img_float * alpha + beta, 0, 255).astype(np.uint8)
        
        # Hue/Saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat_mult = random.uniform(1 - self.cj_dict.get('saturation', 0.2), 1 + self.cj_dict.get('saturation', 0.2))
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
        
        hue_shift = random.uniform(-self.cj_dict.get('hue', 0.05), self.cj_dict.get('hue', 0.05)) * 180
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + int(hue_shift)) % 180
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def _gaussian_blur(self, img, kernel_sizes=None):
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        kernel_size = random.choice(kernel_sizes)
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def _gaussian_noise(self, img, std=15):
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy_img = img.astype(np.float32) + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    def _jpeg_compression(self, img, quality_range=None):
        if quality_range is None:
            quality_range = [30, 70]
        quality = random.randint(quality_range[0], quality_range[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    
    def _resolution_degradation(self, img, scale_range=None):
        if scale_range is None:
            scale_range = [0.3, 0.6]
        h, w = img.shape[:2]
        scale = random.uniform(scale_range[0], scale_range[1])
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def _gamma_shift(self, img, gamma_range=None):
        if gamma_range is None:
            gamma_range = [0.7, 1.3]
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img, table)

    def _channel_augment(self, img):
        """Random channel permutation + single-channel dropout (camera sensor variation)."""
        if random.random() < 0.5:
            # Channel permutation
            perm = list(range(3))
            random.shuffle(perm)
            img = img[:, :, perm]
        else:
            # Single channel dropout: replace one channel with its mean
            ch = random.randint(0, 2)
            img = img.copy()
            img[:, :, ch] = int(img[:, :, ch].mean())
        return img

    def _clahe(self, img):
        """CLAHE histogram equalization on L-channel of LAB colorspace."""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _color_transfer(self, img):
        """Random per-channel gain + bias (simulate camera white-balance shift)."""
        img_float = img.astype(np.float32)
        for ch in range(3):
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-15, 15)
            img_float[:, :, ch] = img_float[:, :, ch] * gain + bias
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def _load_image_and_labels(self, idx, _retries=0):
        img_path = self.img_files[idx]
        targets = self.labels[idx].copy()
        img = cv2.imread(img_path)
        if img is None:
            if _retries >= _MAX_CORRUPT_RETRIES:
                # Return blank image with empty targets after exhausting retries
                h, w = self.img_size[1], self.img_size[0]
                n_cols = self.labels[0].shape[1] if len(self.labels) > 0 else 6
                return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((0, n_cols))
            return self._load_image_and_labels((idx + 1) % len(self), _retries=_retries + 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if self.labels_is_normalized[idx] and len(targets) > 0:
            targets[:, 2] = targets[:, 2] * w   # xc
            targets[:, 3] = targets[:, 3] * h   # yc
            targets[:, 4] = targets[:, 4] * w   # width
            targets[:, 5] = targets[:, 5] * h   # height
        return img, targets
    
    def _mosaic4(self, idx):
        out_h, out_w = self.img_size[1], self.img_size[0]
        yc = int(random.uniform(out_h * 0.25, out_h * 0.75))
        xc = int(random.uniform(out_w * 0.25, out_w * 0.75))
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        random.shuffle(indices)
        mosaic_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        mosaic_labels = []
        quadrants = [(0, 0, xc, yc), (xc, 0, out_w, yc), (0, yc, xc, out_h), (xc, yc, out_w, out_h)]
        for i, index in enumerate(indices):
            img, labels = self._load_image_and_labels(index)
            h, w = img.shape[:2]
            qx1, qy1, qx2, qy2 = quadrants[i]
            quad_w = qx2 - qx1; quad_h = qy2 - qy1
            if quad_w <= 0 or quad_h <= 0: continue
            scale = random.uniform(self.mosaic_scale[0], self.mosaic_scale[1])
            new_w = max(1, int(quad_w * scale)); new_h = max(1, int(quad_h * scale))
            img_resized = cv2.resize(img, (new_w, new_h))
            paste_w = min(new_w, quad_w); paste_h = min(new_h, quad_h)
            if i == 0: src_x1, src_y1 = new_w - paste_w, new_h - paste_h; dst_x1, dst_y1 = qx2 - paste_w, qy2 - paste_h
            elif i == 1: src_x1, src_y1 = 0, new_h - paste_h; dst_x1, dst_y1 = qx1, qy2 - paste_h
            elif i == 2: src_x1, src_y1 = new_w - paste_w, 0; dst_x1, dst_y1 = qx2 - paste_w, qy1
            else: src_x1, src_y1 = 0, 0; dst_x1, dst_y1 = qx1, qy1
            src_x2 = src_x1 + paste_w; src_y2 = src_y1 + paste_h
            dst_x2 = dst_x1 + paste_w; dst_y2 = dst_y1 + paste_h
            mosaic_img[dst_y1:dst_y2, dst_x1:dst_x2] = img_resized[src_y1:src_y2, src_x1:src_x2]
            scale_x = new_w / w; scale_y = new_h / h
            padw = dst_x1 - src_x1; padh = dst_y1 - src_y1
            if len(labels) > 0:
                labels_scaled = labels.copy()
                labels_scaled[:, 2] = labels[:, 2] * scale_x + padw
                labels_scaled[:, 3] = labels[:, 3] * scale_y + padh
                labels_scaled[:, 4] = labels[:, 4] * scale_x
                labels_scaled[:, 5] = labels[:, 5] * scale_y
                for label in labels_scaled:
                    lxc, lyc, lbw, lbh = label[2], label[3], label[4], label[5]
                    lx1, ly1 = lxc - lbw/2, lyc - lbh/2
                    lx2, ly2 = lxc + lbw/2, lyc + lbh/2
                    lx1 = np.clip(lx1, 0, out_w); ly1 = np.clip(ly1, 0, out_h)
                    lx2 = np.clip(lx2, 0, out_w); ly2 = np.clip(ly2, 0, out_h)
                    if (lx2 - lx1) > _MIN_MOSAIC_BOX_PX and (ly2 - ly1) > _MIN_MOSAIC_BOX_PX:
                        label[2] = (lx1 + lx2) / 2
                        label[3] = (ly1 + ly2) / 2
                        label[4] = lx2 - lx1
                        label[5] = ly2 - ly1
                        mosaic_labels.append(label)
        if len(mosaic_labels) > 0: mosaic_labels = np.array(mosaic_labels)
        else: 
            # Use appropriate column count based on this dataset
            # Determine from self.labels if possible, else default to 6
            n_cols = self.labels[0].shape[1] if hasattr(self, 'labels') and len(self.labels) > 0 else 6
            mosaic_labels = np.zeros((0, n_cols))
        return mosaic_img, mosaic_labels
    
    def _mixup(self, img1, labels1, img2, labels2, alpha=0.5):
        ratio = np.random.beta(alpha, alpha)
        ratio = max(ratio, 1 - ratio)
        if img1.shape != img2.shape: img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        mixed_img = (img1 * ratio + img2 * (1 - ratio)).astype(np.uint8)
        if len(labels1) > 0 and len(labels2) > 0: mixed_labels = np.vstack([labels1, labels2])
        elif len(labels1) > 0: mixed_labels = labels1
        elif len(labels2) > 0: mixed_labels = labels2
        else:
             n_cols = labels1.shape[1] if len(labels1) > 0 else (labels2.shape[1] if len(labels2) > 0 else 6)
             mixed_labels = np.zeros((0, n_cols))
        return mixed_img, mixed_labels

    def __getitem__(self, idx):
        use_mosaic = self.augment and hasattr(self, 'mosaic_prob') and random.random() < self.mosaic_prob
        need_denormalize = self.labels_is_normalized[idx] if idx < len(self.labels_is_normalized) else False
        if use_mosaic:
            img, targets = self._mosaic4(idx)
            h, w = img.shape[:2]
            need_denormalize = False
            if hasattr(self, 'mixup_prob') and random.random() < self.mixup_prob:
                idx2 = random.randint(0, len(self) - 1)
                img2, targets2 = self._mosaic4(idx2)
                img, targets = self._mixup(img, targets, img2, targets2)
        else:
            img_path = self.img_files[idx]
            targets = self.labels[idx].copy()
            img = cv2.imread(img_path)
            if img is None:
                # Bounded retry to avoid RecursionError on consecutive corrupt images
                for retry_idx in range(1, _MAX_CORRUPT_RETRIES + 1):
                    next_idx = (idx + retry_idx) % len(self)
                    img = cv2.imread(self.img_files[next_idx])
                    if img is not None:
                        idx = next_idx
                        targets = self.labels[idx].copy()
                        # Refresh normalization flag for the replacement image
                        need_denormalize = self.labels_is_normalized[idx] if idx < len(self.labels_is_normalized) else False
                        break
                if img is None:
                    # All retries exhausted — return blank image
                    h, w = self.img_size[1], self.img_size[0]
                    n_cols = self.labels[0].shape[1] if len(self.labels) > 0 else 6
                    img = np.zeros((h, w, 3), dtype=np.uint8)
                    targets = np.zeros((0, n_cols))
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        if need_denormalize and len(targets) > 0:
            targets[:, 2] = targets[:, 2] * w; targets[:, 3] = targets[:, 3] * h
            targets[:, 4] = targets[:, 4] * w; targets[:, 5] = targets[:, 5] * h
            need_denormalize = False
        
        if self.augment:
            apply_aug = random.random() < self.aug_apply_prob
            if apply_aug and not use_mosaic:
                if random.random() < 0.5:
                    img, targets = self._random_scale_and_translate(img, targets, self.random_scale, self.random_translate)
                    h, w = img.shape[:2]
            if apply_aug:
                if random.random() < self.cj_prob:
                    img = self._np_color_jitter(img)
                if random.random() < self.horizontal_flip_prob:
                    img = np.fliplr(img).copy()
                    if len(targets) > 0: targets[:, 2] = w - targets[:, 2]
                if random.random() < self.cutout_prob: img = self._random_cutout(img)
                if random.random() < self.re_prob: img = self._np_random_erasing(img)
                if random.random() < self.blur_prob: img = self._gaussian_blur(img, self.blur_kernel_sizes)
                if random.random() < self.noise_prob: img = self._gaussian_noise(img, self.noise_std)
                if self.grayscale_prob > 0 and random.random() < self.grayscale_prob:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                if random.random() < self.channel_augment_prob: img = self._channel_augment(img)
                if random.random() < self.clahe_prob: img = self._clahe(img)
                if random.random() < self.color_transfer_prob: img = self._color_transfer(img)
                if self.surv_enabled:
                    if random.random() < self.jpeg_prob: img = self._jpeg_compression(img, self.jpeg_quality_range)
                    if random.random() < self.res_degrade_prob: img = self._resolution_degradation(img, self.res_scale_range)
                    if random.random() < self.gamma_prob: img = self._gamma_shift(img, self.gamma_range)
        
        img_resized, ratio, (pad_w, pad_h) = letterbox(img, self.img_size)
        # ── Re-normalize targets from pixel coords → [0, 1] after letterbox ──
        # Apply letterbox scale + padding offset, then divide by output size.
        if len(targets) > 0:
            targets[:, 2] *= ratio; targets[:, 3] *= ratio; targets[:, 4] *= ratio; targets[:, 5] *= ratio
            targets[:, 2] += pad_w; targets[:, 3] += pad_h
            targets[:, 2] /= self.img_size[0]; targets[:, 3] /= self.img_size[1]
            targets[:, 4] /= self.img_size[0]; targets[:, 5] /= self.img_size[1]
            valid = (targets[:, 4] > 0.01) & (targets[:, 5] > 0.01)
            targets = targets[valid]
        
        if len(targets) > 0:
             # Safety net: if coords are still in pixel range (>2.0) after the
             # normalization above, it means they were already normalized before
             # letterbox (double-normalized).  Divide again to fix.
             if targets[:, 2].max() > _DOUBLE_NORM_THRESHOLD or targets[:, 3].max() > _DOUBLE_NORM_THRESHOLD:
                 targets[:, 2] /= self.img_size[0]; targets[:, 3] /= self.img_size[1]
                 targets[:, 4] /= self.img_size[0]; targets[:, 5] /= self.img_size[1]
             np.clip(targets[:, 2], 0, 1, out=targets[:, 2])
             np.clip(targets[:, 3], 0, 1, out=targets[:, 3])
             np.clip(targets[:, 4], 0, 1, out=targets[:, 4])
             np.clip(targets[:, 5], 0, 1, out=targets[:, 5])
             min_dim = 2.0 / max(self.img_size)
             valid = (targets[:, 4] > min_dim) & (targets[:, 5] > min_dim)
             targets = targets[valid]

        img_tensor = T.ToTensor()(img_resized)
        # ImageNet normalization — aligns input distribution with timm pretrained weights
        img_tensor = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img_tensor)
        targets_tensor = torch.from_numpy(targets).float()
        
        # Camera ID (per-image integer)
        cam_id = self.camera_ids[idx] if idx < len(self.camera_ids) else -1
        
        return img_tensor, targets_tensor, cam_id


# ═══════════════════════════════════════════════════════════════════════════════
# Samplers — Batch Construction Strategies
# ═══════════════════════════════════════════════════════════════════════════════

class SequenceGroupedSampler(torch.utils.data.Sampler):
    """Samples batches where all frames come from the same sequence,
    using CONSECUTIVE frames to ensure identity overlap for triplet loss.
    
    Key insight: consecutive surveillance/MOT frames share 80-90% of person IDs
    because people persist across adjacent frames. Random shuffling within a 
    sequence destroys this temporal locality, causing 0 positive pairs in 
    triplet loss with small batch sizes.
    
    Strategy:
    - Sort frame indices within each sequence (temporal order)
    - Form batches from consecutive frames (guaranteed ID overlap)
    - Shuffle batch ORDER across sequences (epoch variety)
    - Random per-epoch offset per sequence (different frame groupings each epoch)
    
    NOTE: For better triplet learning, use IdentityAwareSampler instead.
    Consecutive frames share IDs but have near-identical poses (weak signal).
    IdentityAwareSampler uses temporal gaps + cross-camera pairing for diverse positives.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Build sequence -> frame_indices mapping
        self.seq_groups = {}
        for idx, img_path in enumerate(dataset.img_files):
            seq = self._get_sequence(img_path)
            if seq not in self.seq_groups:
                self.seq_groups[seq] = []
            self.seq_groups[seq].append(idx)

        # Sort indices within each sequence for temporal locality
        for seq in self.seq_groups:
            self.seq_groups[seq].sort()

        for seq, indices in self.seq_groups.items():
            print(f"  SequenceGroupedSampler: {seq} -> {len(indices)} frames")

    @staticmethod
    def _get_sequence(path):
        """Extract sequence identifier from an image path.
        
        Handles:
        - MOT17: .../MOT17/train/MOT17-02-SDP/img1/000001.jpg → MOT17-02-SDP
        - MOT20: .../MOT20/train/MOT20-01/img1/000001.jpg → MOT20-01
        - SOMPT22: .../sompt22/train/SOMPT22-02/img1/000001.jpg → SOMPT22-02
        - P-DESTRE: .../P-DESTRE/images/08-11-2019_1_1/file.jpg → P-DESTRE/08-11-2019_1_1
        - CrowdHuman: .../CrowdHuman_train/Images/file.jpg → CrowdHuman
        """
        parts = path.replace('\\', '/').split('/')
        
        # For MOT/SOMPT sequences: image is in .../SEQNAME/img1/frame.jpg
        # The sequence name is the parent of img1
        for i, p in enumerate(parts):
            if p == 'img1' and i > 0:
                return parts[i - 1]  # MOT17-02-SDP, MOT20-01, SOMPT22-02
        
        # For P-DESTRE: .../images/SEQNAME/frame.jpg or .../Images/SEQNAME/frame.jpg
        # For CrowdHuman: .../CrowdHuman_train/Images/frame.jpg (no subdirectory)
        for i, p in enumerate(parts):
            if p.lower() == 'images':
                if i + 2 < len(parts):
                    # Has subdirectory: .../images/SEQNAME/file.jpg
                    seq_dir = parts[i + 1]
                    if i > 0 and parts[i - 1] not in ('Dataset', 'train', 'test', 'val'):
                        return f"{parts[i - 1]}/{seq_dir}"
                    return seq_dir
                elif i > 0:
                    # No subdirectory: .../PARENT/Images/file.jpg → use PARENT as group
                    return parts[i - 1]
        
        return 'unknown'

    def set_skip_batches(self, n):
        """Skip first n batches in next __iter__ call (for mid-epoch resume)."""
        self._skip_batches = n

    def __iter__(self):
        all_batches = []
        for seq, indices in self.seq_groups.items():
            # Keep frame order (already sorted in __init__) for temporal locality
            # Use random offset each epoch so different frame groupings appear
            idx_sorted = indices.copy()
            if self.shuffle and len(idx_sorted) > self.batch_size:
                offset = random.randint(0, self.batch_size - 1)
                idx_sorted = idx_sorted[offset:]
            for i in range(0, len(idx_sorted), self.batch_size):
                batch = idx_sorted[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)
        if self.shuffle:
            random.shuffle(all_batches)  # Shuffle batch order, NOT frames within batches
        skip = getattr(self, '_skip_batches', 0)
        if skip > 0:
            all_batches = all_batches[skip:]
            self._skip_batches = 0  # Only skip once (first epoch after resume)
        for batch in all_batches:
            yield batch

    def __len__(self):
        return sum(len(v) // self.batch_size for v in self.seq_groups.values())


# ── IdentityAwareSampler ─────────────────────────────────────────────────────

class IdentityAwareSampler(torch.utils.data.Sampler):
    """Smart batch sampling for effective triplet/metric learning.
    
    Problem: consecutive frames have near-identical poses → weak triplet signal.
    Solution: pair frames of the same person with MAXIMUM visual diversity.
    
    Two strategies combined:
    
    1. Interleaved temporal segments: within each sequence, split frames into
       `batch_size` temporal segments and pair one frame from each segment.
       Gap ≈ seq_length / batch_size frames (typically many seconds).
       Same person, very different pose/scale/occlusion.
    
    2. Cross-camera identity matching: for datasets with cross-camera IDs
       (e.g., P-DESTRE), inject bonus batches from different cameras showing
       the same person. Hardest positives = best ReID training signal.
    
    Every image appears exactly once per epoch (interleaved batches).
    Cross-camera batches are ADDITIONAL bonus training signal.
    """
    
    def __init__(self, dataset, batch_size, cross_camera_ratio=0.2, shuffle=True):
        self.batch_size = batch_size
        self.cross_camera_ratio = cross_camera_ratio
        self.shuffle = shuffle
        
        # Build sequence → sorted frame indices
        self.seq_groups = {}
        for idx, img_path in enumerate(dataset.img_files):
            seq = SequenceGroupedSampler._get_sequence(img_path)
            if seq not in self.seq_groups:
                self.seq_groups[seq] = []
            self.seq_groups[seq].append(idx)
        
        for seq in self.seq_groups:
            self.seq_groups[seq].sort()  # Frame order
        
        for seq, indices in self.seq_groups.items():
            print(f"  IdentityAwareSampler: {seq} -> {len(indices)} frames")
        
        # Build identity → {seq: [img_indices]} for cross-camera sampling
        self.id_to_seq_images = defaultdict(lambda: defaultdict(list))
        idx_to_seq = {}
        for idx, img_path in enumerate(dataset.img_files):
            idx_to_seq[idx] = SequenceGroupedSampler._get_sequence(img_path)
        
        for idx, labels in enumerate(dataset.labels):
            if len(labels) > 0:
                seq = idx_to_seq[idx]
                for pid in set(labels[:, 1].astype(int)):
                    if pid >= 0:
                        self.id_to_seq_images[pid][seq].append(idx)
        
        # Cross-camera IDs: appear in 2+ sequences
        self.cross_camera_ids = [
            pid for pid, seq_dict in self.id_to_seq_images.items()
            if len(seq_dict) >= 2
        ]
        
        # Diagnostic: temporal gap stats per sequence
        total_interleaved = 0
        for seq, indices in self.seq_groups.items():
            n = len(indices)
            if n < batch_size:
                continue
            seg_len = n // batch_size
            gap_frames = seg_len  # gap between paired frames ≈ seg_len
            total_interleaved += seg_len
            print(f"    {seq}: temporal gap ≈ {gap_frames} frames between batch members")
        
        n_cross = int(total_interleaved * cross_camera_ratio) if self.cross_camera_ids else 0
        print(f"  IdentityAwareSampler: {total_interleaved} interleaved + "
              f"{n_cross} cross-camera = {total_interleaved + n_cross} total batches/epoch")
        print(f"  Cross-camera IDs available: {len(self.cross_camera_ids)}")
        
        # Diagnostic: estimate ID overlap in interleaved batches
        n_checked = 0
        n_with_overlap = 0
        for seq, indices in self.seq_groups.items():
            n = len(indices)
            if n < batch_size:
                continue
            seg_len = n // batch_size
            for i in range(min(seg_len, 50)):  # check up to 50 batches per seq
                batch_indices = [indices[s * seg_len + i] for s in range(batch_size)]
                all_ids = set()
                has_shared = False
                for bidx in batch_indices:
                    if bidx < len(dataset.labels) and len(dataset.labels[bidx]) > 0:
                        frame_ids = set(dataset.labels[bidx][:, 1].astype(int)) - {-1}
                        if all_ids & frame_ids:
                            has_shared = True
                        all_ids |= frame_ids
                n_checked += 1
                if has_shared:
                    n_with_overlap += 1
        if n_checked > 0:
            pct = 100 * n_with_overlap / n_checked
            print(f"  ID overlap estimate: {n_with_overlap}/{n_checked} sampled batches "
                  f"({pct:.1f}%) have shared IDs for triplet loss")
    
    def set_skip_batches(self, n):
        """Skip first n batches in next __iter__ call (for mid-epoch resume)."""
        self._skip_batches = n

    def __iter__(self):
        all_batches = []
        
        # === Strategy 1: Interleaved temporal segments ===
        # Split each sequence into batch_size segments, pair one frame from each.
        # E.g., batch_size=2, seq=[0..499]: seg_A=[0..249], seg_B=[250..499]
        #   batch 0: [A[0], B[0]], batch 1: [A[1], B[1]], ...
        # Gap between paired frames ≈ seq_length/batch_size (many seconds).
        # Shuffle WITHIN segments each epoch → different pairings.
        # Every image appears exactly once.
        for seq, indices in self.seq_groups.items():
            n = len(indices)
            if n < self.batch_size:
                continue
            
            seg_len = n // self.batch_size
            segments = []
            for s in range(self.batch_size):
                start = s * seg_len
                seg = indices[start:start + seg_len].copy() if isinstance(indices, list) else list(indices[start:start + seg_len])
                if self.shuffle:
                    random.shuffle(seg)
                segments.append(seg)
            
            for i in range(seg_len):
                batch = [segments[s][i] for s in range(self.batch_size)]
                all_batches.append(batch)
        
        # === Strategy 2: Cross-camera bonus batches ===
        # For datasets with shared IDs across cameras (P-DESTRE), inject extra
        # batches showing the same person from different viewpoints.
        # These are the HARDEST positive pairs and most valuable for ReID.
        if self.cross_camera_ids:
            n_cross = int(len(all_batches) * self.cross_camera_ratio)
            for _ in range(n_cross):
                pid = random.choice(self.cross_camera_ids)
                seq_dict = self.id_to_seq_images[pid]
                seq_keys = list(seq_dict.keys())
                
                batch = []
                if len(seq_keys) >= self.batch_size:
                    # Pick batch_size different cameras
                    selected = random.sample(seq_keys, self.batch_size)
                    for s in selected:
                        batch.append(random.choice(seq_dict[s]))
                else:
                    # Use all available cameras, fill rest from other cameras
                    for s in seq_keys:
                        batch.append(random.choice(seq_dict[s]))
                    while len(batch) < self.batch_size:
                        s = random.choice(seq_keys)
                        batch.append(random.choice(seq_dict[s]))
                
                all_batches.append(batch[:self.batch_size])
        
        if self.shuffle:
            random.shuffle(all_batches)
        
        skip = getattr(self, '_skip_batches', 0)
        if skip > 0:
            all_batches = all_batches[skip:]
            self._skip_batches = 0  # Only skip once (first epoch after resume)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        n = sum(len(v) // self.batch_size for v in self.seq_groups.values())
        if self.cross_camera_ids:
            n += int(n * self.cross_camera_ratio)
        return n


# ═══════════════════════════════════════════════════════════════════════════════
# Collation
# ═══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """Collate images, targets, and optional camera IDs into a batch.

    Supports two return formats depending on what __getitem__ yields:
      • 2-tuple (img, targets)          → returns (imgs, targets)
      • 3-tuple (img, targets, cam_id)  → returns (imgs, targets, cam_ids)

    Targets are concatenated with a leading batch-index column so downstream
    code can tell which targets belong to which image:
      [batch_idx, cls, id, cx, cy, w, h, conf, vis]   (9 columns)
    """
    if len(batch[0]) == 3:
        imgs, targets, cam_ids = zip(*batch)
        cam_ids = torch.tensor(cam_ids, dtype=torch.long)
    else:
        imgs, targets = zip(*batch)
        cam_ids = None
    imgs = torch.stack(imgs)
    new_targets = []
    for i, t in enumerate(targets):
        if t.shape[0] > 0:
            batch_idx = torch.full((t.shape[0], 1), i)
            new_t = torch.cat((batch_idx, t), dim=1)
            new_targets.append(new_t)
    if new_targets: targets = torch.cat(new_targets, dim=0)
    else: 
        # Default: batch + 8 columns [batch, cls, id, cx, cy, w, h, conf, vis]
        targets = torch.zeros((0, 9))
    if cam_ids is not None:
        return imgs, targets, cam_ids
    return imgs, targets
