"""
Cross-camera global ID association for multi-camera tracking.

The design here is intentionally small:
  - extract a compact gallery from each local track
  - compare local galleries against persistent global-ID galleries
  - apply per-camera-safe assignment with optional temporal and transition gating

Legacy research flags from older ablation code are still accepted at construction
time for compatibility, but they are ignored by the implementation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import lap
except ImportError:
    lap = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


TimestampInput = Optional[Union[float, int, Mapping[Any, Union[float, int]]]]


@dataclass
class GlobalIdentityState:
    """Persistent state for one global identity."""

    gallery: List[np.ndarray] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    last_camera: Any = None
    last_timestamp: Optional[float] = None
    seen_by_camera: Dict[Any, float] = field(default_factory=dict)
    # Part-aware ReID: per-camera best part features + attention weights
    part_feats: Optional[np.ndarray] = None    # [num_parts, C] EMA of part embeddings
    attn_weights: Optional[np.ndarray] = None  # [num_parts] EMA attention weights
    gallery_camera_ids: List[Any] = field(default_factory=list)  # camera source per gallery entry


class CrossCameraAssociator:
    """Resolve per-camera local IDs into shared global IDs."""

    LEGACY_FLAGS = (
        "use_optimal_assignment",
        "mutual_nn_verify",
        "quality_weighted",
        "gallery_decay",
        "normalize_camera_pairs",
        "corrective_cascade",
        "cascade_strict_thresh",
    )

    def __init__(
        self,
        match_thresh: float = 0.4,
        gallery_size: int = 10,
        temporal_window: Optional[float] = 300,
        transition_priors: Optional[Mapping[Any, Any]] = None,
        transition_weight: float = 0.15,
        max_global_ids: int = 2000,
        timestamp_mode: str = "frames",
        gallery_top_k: int = 3,
        part_aware_alpha: float = 0.4,
        distinctiveness_ratio: float = 0.85,
        merge_thresh: float = 0.0,
        **legacy_kwargs: Any,
    ) -> None:
        self.match_thresh = float(match_thresh)
        self.gallery_size = max(1, int(gallery_size))
        self.temporal_window = None if temporal_window in (None, 0) else float(temporal_window)
        self.transition_priors = transition_priors or {}
        self.transition_weight = float(transition_weight)
        self.max_global_ids = max(1, int(max_global_ids))
        self.timestamp_mode = str(timestamp_mode)
        self.gallery_top_k = max(1, int(gallery_top_k))
        self.part_aware_alpha = float(np.clip(part_aware_alpha, 0.0, 1.0))
        self.distinctiveness_ratio = float(np.clip(distinctiveness_ratio, 0.0, 1.0))
        self.merge_thresh = float(merge_thresh)

        self.local_to_global: Dict[Any, Dict[int, int]] = defaultdict(dict)
        self.global_states: Dict[int, GlobalIdentityState] = {}
        self.global_id_order: List[int] = []
        self.global_id_counter = 1

        self.ignored_legacy_kwargs = {
            key: value for key, value in legacy_kwargs.items() if key in self.LEGACY_FLAGS
        }

    @property
    def global_embeddings(self) -> Dict[int, np.ndarray]:
        """Compatibility view of global centroids."""
        return {
            gid: state.embedding.copy()
            for gid, state in self.global_states.items()
            if state.embedding is not None
        }

    @property
    def global_last_seen(self) -> Dict[int, Tuple[Any, Optional[float]]]:
        """Compatibility view of most recent camera and timestamp."""
        return {
            gid: (state.last_camera, state.last_timestamp)
            for gid, state in self.global_states.items()
        }

    def ablation_flags(self) -> Dict[str, Any]:
        """Return the active runtime configuration."""
        return {
            "match_thresh": self.match_thresh,
            "gallery_size": self.gallery_size,
            "temporal_window": self.temporal_window,
            "transition_weight": self.transition_weight,
            "max_global_ids": self.max_global_ids,
            "timestamp_mode": self.timestamp_mode,
            "gallery_top_k": self.gallery_top_k,
            "part_aware_alpha": self.part_aware_alpha,
            "ignored_legacy_flags": sorted(self.ignored_legacy_kwargs.keys()),
        }

    @staticmethod
    def _normalize(feat: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if feat is None:
            return None
        arr = np.asarray(feat, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return None
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            return None
        return arr / norm

    def _resolve_timestamp(self, camera_id: Any, timestamp: TimestampInput) -> Optional[float]:
        if self.timestamp_mode == "disabled" or timestamp is None:
            return None
        if isinstance(timestamp, Mapping):
            value = timestamp.get(camera_id)
            if value is None:
                return None
            return float(value)
        return float(timestamp)

    def _transition_prior(self, src_camera: Any, dst_camera: Any) -> float:
        if not self.transition_priors or src_camera is None or dst_camera is None:
            return 1.0
        if src_camera == dst_camera:
            return 1.0

        priors = self.transition_priors
        value = None

        if src_camera in priors and isinstance(priors[src_camera], Mapping):
            value = priors[src_camera].get(dst_camera)
            if value is None:
                value = priors[src_camera].get(str(dst_camera))
        if value is None and str(src_camera) in priors and isinstance(priors[str(src_camera)], Mapping):
            src_map = priors[str(src_camera)]
            value = src_map.get(dst_camera)
            if value is None:
                value = src_map.get(str(dst_camera))

        if value is None:
            return 1.0
        return float(np.clip(value, 0.0, 1.0))

    def _temporal_ok(
        self,
        state: GlobalIdentityState,
        dst_camera: Any,
        dst_timestamp: Optional[float],
    ) -> bool:
        if self.temporal_window is None or dst_timestamp is None or not state.seen_by_camera:
            return True
        for src_camera, src_timestamp in state.seen_by_camera.items():
            if src_camera == dst_camera:
                continue
            if abs(dst_timestamp - src_timestamp) <= self.temporal_window:
                return True
        return False

    def _camera_has_gid(
        self,
        camera_id: Any,
        gid: int,
        current_local_id: Optional[int] = None,
    ) -> bool:
        for local_id, mapped_gid in self.local_to_global.get(camera_id, {}).items():
            if mapped_gid == gid and local_id != current_local_id:
                return True
        return False

    def _track_gallery(self, track: Any) -> Optional[np.ndarray]:
        rows: List[np.ndarray] = []

        gallery = getattr(track, "gallery", None)
        if gallery is not None:
            gallery_matrix = None
            if hasattr(gallery, "get_all"):
                gallery_matrix = gallery.get_all()
            else:
                gallery_matrix = np.asarray(gallery, dtype=np.float64)
            if gallery_matrix is not None:
                gallery_matrix = np.asarray(gallery_matrix, dtype=np.float64)
                if gallery_matrix.ndim == 1:
                    gallery_matrix = gallery_matrix.reshape(1, -1)
                for row in gallery_matrix:
                    normalized = self._normalize(row)
                    if normalized is not None:
                        rows.append(normalized)

        for attr_name in ("smooth_feat", "curr_feat"):
            normalized = self._normalize(getattr(track, attr_name, None))
            if normalized is not None:
                rows.append(normalized)

        if not rows:
            return None

        unique_rows: List[np.ndarray] = []
        for row in rows:
            if not unique_rows:
                unique_rows.append(row)
                continue
            sims = np.asarray(unique_rows) @ row
            if np.max(sims) < 0.995:
                unique_rows.append(row)
            else:
                unique_rows[-1] = row

        if len(unique_rows) > self.gallery_size:
            unique_rows = unique_rows[-self.gallery_size :]

        return np.asarray(unique_rows, dtype=np.float64)

    def _track_parts(self, track: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract part features and attention weights from a track."""
        part_feats = getattr(track, "smooth_part_feats", None)
        attn_weights = getattr(track, "smooth_attn_weights", None)
        if part_feats is None or attn_weights is None:
            return None, None
        part_feats = np.asarray(part_feats, dtype=np.float64)
        attn_weights = np.asarray(attn_weights, dtype=np.float64)
        # Re-normalize parts
        norms = np.linalg.norm(part_feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        part_feats = part_feats / norms
        return part_feats, attn_weights

    def _gallery_distance(self, query_gallery: np.ndarray, global_gallery: np.ndarray) -> float:
        # Top-K pairwise matching: preserves discriminative viewpoint information
        # that centroid averaging would wash out.  Falls back to centroid-to-centroid
        # only when gallery_top_k <= 0 (or galleries are very small).
        top_k = self.gallery_top_k
        if top_k > 0 and query_gallery.shape[0] >= 1 and global_gallery.shape[0] >= 1:
            sim_matrix = query_gallery @ global_gallery.T  # [N_q, N_g]
            k = min(top_k, sim_matrix.size)
            top_sims = np.sort(sim_matrix.ravel())[-k:]
            avg_sim = float(np.clip(top_sims.mean(), -1.0, 1.0))
            return 1.0 - avg_sim

        # Centroid fallback
        q_centroid = query_gallery.mean(axis=0)
        q_norm = np.linalg.norm(q_centroid)
        if q_norm < 1e-12:
            return 1.0
        q_centroid = q_centroid / q_norm

        g_centroid = global_gallery.mean(axis=0)
        g_norm = np.linalg.norm(g_centroid)
        if g_norm < 1e-12:
            return 1.0
        g_centroid = g_centroid / g_norm

        sim = np.clip(float(q_centroid @ g_centroid), -1.0, 1.0)
        return 1.0 - sim

    def _part_distance(
        self,
        query_parts: Optional[np.ndarray],
        query_attn: Optional[np.ndarray],
        global_parts: Optional[np.ndarray],
        global_attn: Optional[np.ndarray],
    ) -> Optional[float]:
        """Mutual-visibility-weighted part distance (mirrors unified_tracker logic)."""
        if query_parts is None or global_parts is None:
            return None
        if query_attn is None or global_attn is None:
            return None
        # Per-part cosine similarity
        part_sims = np.sum(query_parts * global_parts, axis=1)
        # Mutual visibility: min of attention weights
        mutual_vis = np.minimum(query_attn, global_attn)
        vis_sum = mutual_vis.sum()
        if vis_sum > 1e-8:
            part_sim = np.sum(mutual_vis * part_sims) / vis_sum
        else:
            part_sim = np.mean(part_sims)
        return float(1.0 - part_sim)

    @staticmethod
    def _recompute_centroid(gallery: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        if not gallery:
            return None
        matrix = np.asarray(gallery, dtype=np.float64)
        centroid = matrix.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-12:
            return None
        return centroid / norm

    def _touch_global_state(
        self,
        gid: int,
        camera_id: Any,
        timestamp: Optional[float],
    ) -> None:
        state = self.global_states.get(gid)
        if state is None:
            return
        if timestamp is not None:
            state.seen_by_camera[camera_id] = timestamp
            if state.last_timestamp is None or timestamp >= state.last_timestamp:
                state.last_timestamp = timestamp
                state.last_camera = camera_id
        elif state.last_camera is None:
            state.last_camera = camera_id

    def _update_global_state(
        self,
        gid: int,
        gallery_matrix: Optional[np.ndarray],
        camera_id: Any,
        timestamp: Optional[float],
        part_feats: Optional[np.ndarray] = None,
        attn_weights: Optional[np.ndarray] = None,
    ) -> None:
        state = self.global_states[gid]
        if gallery_matrix is not None:
            for row in gallery_matrix:
                if not state.gallery:
                    state.gallery.append(row)
                    state.gallery_camera_ids.append(camera_id)
                    continue
                sims = np.asarray(state.gallery) @ row
                # Threshold calibrated for ArcFace embeddings: all pairwise
                # distances lie in [0.001, 0.019] (sims in [0.981, 0.999]).
                # Using 0.998 (dist≈0.002) as the near-duplicate threshold so
                # distinct viewpoints within the narrow range still accumulate.
                if np.max(sims) < 0.998:
                    state.gallery.append(row)
                    state.gallery_camera_ids.append(camera_id)
                else:
                    most_similar_idx = int(np.argmax(sims))
                    state.gallery[most_similar_idx] = row
                    state.gallery_camera_ids[most_similar_idx] = camera_id
            if len(state.gallery) > self.gallery_size:
                # Camera-aware trim: keep at least one entry per camera
                # to preserve cross-camera viewpoint diversity, then fill
                # the remainder with the most recent entries.
                cam_last: Dict[Any, int] = {}
                for idx, cid in enumerate(state.gallery_camera_ids):
                    cam_last[cid] = idx  # last occurrence per camera
                keep_set = set(cam_last.values())
                # Fill remainder from most recent entries not yet kept
                remaining = [
                    i for i in range(len(state.gallery) - 1, -1, -1)
                    if i not in keep_set
                ]
                for idx in remaining:
                    if len(keep_set) >= self.gallery_size:
                        break
                    keep_set.add(idx)
                keep_list = sorted(keep_set)[:self.gallery_size]
                state.gallery = [state.gallery[i] for i in keep_list]
                state.gallery_camera_ids = [state.gallery_camera_ids[i] for i in keep_list]
            state.embedding = self._recompute_centroid(state.gallery)
        # EMA update of part features on the global state
        if part_feats is not None and attn_weights is not None:
            alpha = 0.9
            if state.part_feats is None:
                state.part_feats = part_feats.copy()
                state.attn_weights = attn_weights.copy()
            else:
                state.part_feats = alpha * state.part_feats + (1 - alpha) * part_feats
                norms = np.linalg.norm(state.part_feats, axis=1, keepdims=True)
                state.part_feats = state.part_feats / np.maximum(norms, 1e-12)
                state.attn_weights = alpha * state.attn_weights + (1 - alpha) * attn_weights
        self._touch_global_state(gid, camera_id, timestamp)

    def _transition_penalty(self, state: GlobalIdentityState, dst_camera: Any) -> float:
        if self.transition_weight <= 0 or not state.seen_by_camera:
            return 0.0
        best_prior = 1.0
        if self.transition_priors:
            best_prior = max(
                self._transition_prior(src_camera, dst_camera)
                for src_camera in state.seen_by_camera.keys()
            )
        return self.transition_weight * (1.0 - best_prior)

    def _compute_cost(self, entry: Dict[str, Any], gid: int) -> float:
        state = self.global_states.get(gid)
        if state is None or not state.gallery:
            return np.inf
        if self._camera_has_gid(entry["camera_id"], gid, current_local_id=entry["local_id"]):
            return np.inf
        if not self._temporal_ok(state, entry["camera_id"], entry["timestamp"]):
            return np.inf

        global_gallery = np.asarray(state.gallery, dtype=np.float64)
        appearance_cost = self._gallery_distance(entry["gallery"], global_gallery)

        # Part-aware distance blending (mirrors unified_tracker logic)
        alpha = self.part_aware_alpha
        if alpha > 0:
            part_dist = self._part_distance(
                entry.get("part_feats"), entry.get("attn_weights"),
                state.part_feats, state.attn_weights,
            )
            if part_dist is not None:
                appearance_cost = (1 - alpha) * appearance_cost + alpha * part_dist

        total_cost = appearance_cost + self._transition_penalty(state, entry["camera_id"])
        return total_cost

    def _build_cost_matrix(
        self,
        entries: Sequence[Dict[str, Any]],
        gids: Sequence[int],
    ) -> np.ndarray:
        cost_matrix = np.full((len(entries), len(gids)), np.inf, dtype=np.float64)
        for row_idx, entry in enumerate(entries):
            for col_idx, gid in enumerate(gids):
                cost_matrix[row_idx, col_idx] = self._compute_cost(entry, gid)
        return cost_matrix

    def _solve_assignments(
        self,
        cost_matrix: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        n_rows, n_cols = cost_matrix.shape
        finite_mask = np.isfinite(cost_matrix)
        if not finite_mask.any():
            return [], list(range(n_rows)), list(range(n_cols))

        invalid_cost = self.match_thresh + self.transition_weight + 2.0
        dense_cost = np.where(finite_mask, cost_matrix, invalid_cost)

        if lap is not None:
            # Use a cost_limit proportional to the actual distance regime.
            # With ArcFace embeddings all pairwise distances fall in [0.001, 0.05];
            # using match_thresh (e.g. 0.50) as cost_limit makes "unmatched" so
            # expensive that the solver always prefers matching (even bad ones).
            # A tighter limit lets the solver leave ambiguous rows unmatched.
            finite_vals = dense_cost[finite_mask]
            if finite_vals.size > 0:
                adaptive_limit = float(np.median(finite_vals) + 2.0 * np.std(finite_vals))
                adaptive_limit = max(adaptive_limit, 0.02)  # floor
            else:
                adaptive_limit = 0.05
            _, row_assignments, col_assignments = lap.lapjv(
                dense_cost,
                extend_cost=True,
                cost_limit=adaptive_limit,
            )
            raw_matches = [
                (row_idx, col_idx)
                for row_idx, col_idx in enumerate(row_assignments)
                if col_idx >= 0 and np.isfinite(cost_matrix[row_idx, col_idx])
            ]
        elif linear_sum_assignment is not None:
            row_ids, col_ids = linear_sum_assignment(dense_cost)
            raw_matches = []
            for row_idx, col_idx in zip(row_ids.tolist(), col_ids.tolist()):
                if np.isfinite(cost_matrix[row_idx, col_idx]):
                    raw_matches.append((row_idx, col_idx))
        else:
            candidates = [
                (float(cost_matrix[row_idx, col_idx]), row_idx, col_idx)
                for row_idx in range(n_rows)
                for col_idx in range(n_cols)
                if np.isfinite(cost_matrix[row_idx, col_idx])
            ]
            candidates.sort(key=lambda item: item[0])
            used_rows: set = set()
            used_cols: set = set()
            raw_matches = []
            for _, row_idx, col_idx in candidates:
                if row_idx in used_rows or col_idx in used_cols:
                    continue
                used_rows.add(row_idx)
                used_cols.add(col_idx)
                raw_matches.append((row_idx, col_idx))

        # Relative-margin filter: works with collapsed embedding spaces where
        # all absolute costs lie in a narrow range (e.g. [0.001, 0.019]).
        # Strategy: row-normalize the cost values, then apply a relative threshold.
        # A match is accepted only if its cost is noticeably lower than alternatives
        # in the same row (i.e. it is clearly the best candidate).
        ratio_thresh = self.distinctiveness_ratio  # Lowe-style ratio test (if > 0)
        matches = []
        for row_idx, col_idx in raw_matches:
            best_cost = cost_matrix[row_idx, col_idx]
            row_finite = cost_matrix[row_idx][np.isfinite(cost_matrix[row_idx])]

            # Lowe-style ratio test (only when ratio_thresh > 0)
            if ratio_thresh > 0 and len(row_finite) >= 2:
                sorted_costs = np.sort(row_finite)
                second_best = sorted_costs[1]
                if second_best > 1e-12 and best_cost / second_best > ratio_thresh:
                    continue  # Not distinctive enough

            # Relative-margin gate: apply absolute match_thresh only if row range is
            # wide enough that the threshold is meaningful; otherwise use z-score to
            # detect outlier-low costs in the narrow-range collapsed embedding regime.
            if len(row_finite) >= 2:
                row_min = row_finite.min()
                row_max = row_finite.max()
                row_range = row_max - row_min
                if row_range > 1e-6:
                    # Normalised position in [0, 1]: 0 = best in row, 1 = worst
                    norm_cost = (best_cost - row_min) / row_range
                    if norm_cost > self.match_thresh:
                        continue  # Not clearly the best candidate in this row
                else:
                    # All costs are effectively equal — apply absolute threshold
                    if best_cost > self.match_thresh:
                        continue
            else:
                # Only one finite candidate: accept if within absolute threshold
                if best_cost > self.match_thresh:
                    continue

            matches.append((row_idx, col_idx))

        matched_rows = {row for row, _ in matches}
        matched_cols = {col for _, col in matches}
        unmatched_rows = [row for row in range(n_rows) if row not in matched_rows]
        unmatched_cols = [col for col in range(n_cols) if col not in matched_cols]
        return matches, unmatched_rows, unmatched_cols

    def _evict_one_global_id(self) -> None:
        if len(self.global_states) < self.max_global_ids:
            return

        referenced = {
            gid for camera_map in self.local_to_global.values() for gid in camera_map.values()
        }

        candidates = [gid for gid in self.global_id_order if gid not in referenced]
        if not candidates:
            candidates = list(self.global_id_order)

        def sort_key(gid: int) -> Tuple[bool, float]:
            state = self.global_states.get(gid)
            timestamp = -1.0 if state is None or state.last_timestamp is None else state.last_timestamp
            return state is None, timestamp

        gid_to_remove = min(candidates, key=sort_key)
        self.global_states.pop(gid_to_remove, None)
        if gid_to_remove in self.global_id_order:
            self.global_id_order.remove(gid_to_remove)
        for camera_id in list(self.local_to_global.keys()):
            stale_local_ids = [
                local_id
                for local_id, mapped_gid in self.local_to_global[camera_id].items()
                if mapped_gid == gid_to_remove
            ]
            for local_id in stale_local_ids:
                del self.local_to_global[camera_id][local_id]

    def _allocate_global_id(
        self,
        camera_id: Any,
        local_id: int,
        gallery_matrix: Optional[np.ndarray],
        timestamp: Optional[float],
        part_feats: Optional[np.ndarray] = None,
        attn_weights: Optional[np.ndarray] = None,
    ) -> int:
        self._evict_one_global_id()
        gid = self.global_id_counter
        self.global_id_counter += 1
        self.global_states[gid] = GlobalIdentityState()
        self.global_id_order.append(gid)
        self.local_to_global[camera_id][local_id] = gid
        self._update_global_state(gid, gallery_matrix, camera_id, timestamp,
                                  part_feats=part_feats, attn_weights=attn_weights)
        return gid

    # ------------------------------------------------------------------
    #  Global ID merge pass
    # ------------------------------------------------------------------

    def _merge_pass(self) -> None:
        """Merge cross-camera GIDs whose centroids are very similar.

        This fixes the cold-start problem: when multiple cameras start
        simultaneously, each camera's tracks create separate GIDs even
        for the same physical person.  The merge pass detects these
        duplicates and unifies them.
        """
        if self.merge_thresh <= 0:
            return

        # Collect centroids (only from mature galleries with >= 3 entries)
        centroids: Dict[int, np.ndarray] = {}
        for gid, state in self.global_states.items():
            if state.embedding is not None and len(state.gallery) >= 3:
                centroids[gid] = state.embedding
        if len(centroids) < 2:
            return

        # Group GIDs by last_camera for fast cross-camera iteration
        cam_to_gids: Dict[Any, List[int]] = defaultdict(list)
        for gid in centroids:
            cam_to_gids[self.global_states[gid].last_camera].append(gid)

        cameras = sorted(cam_to_gids.keys(), key=str)

        # Find merge candidates (cross-camera pairs above threshold)
        merge_pairs: List[Tuple[float, int, int]] = []
        for i in range(len(cameras)):
            for j in range(i + 1, len(cameras)):
                for gid_a in cam_to_gids[cameras[i]]:
                    for gid_b in cam_to_gids[cameras[j]]:
                        sim = float(centroids[gid_a] @ centroids[gid_b])
                        if sim >= self.merge_thresh:
                            merge_pairs.append((sim, gid_a, gid_b))

        if not merge_pairs:
            return

        # Process most confident merges first
        merge_pairs.sort(key=lambda x: -x[0])
        merged_away: set = set()

        for _, gid_a, gid_b in merge_pairs:
            if gid_a in merged_away or gid_b in merged_away:
                continue

            # Check camera conflict: no camera should have tracks in both GIDs
            cams_a: set = set()
            cams_b: set = set()
            for cam_id, local_map in self.local_to_global.items():
                for mapped_gid in local_map.values():
                    if mapped_gid == gid_a:
                        cams_a.add(cam_id)
                    elif mapped_gid == gid_b:
                        cams_b.add(cam_id)
            if cams_a & cams_b:
                continue  # Would create same-camera duplicate

            self._execute_merge(gid_a, gid_b)
            merged_away.add(gid_b)

    def _execute_merge(self, keep_gid: int, remove_gid: int) -> None:
        """Merge *remove_gid* into *keep_gid*."""
        keep = self.global_states[keep_gid]
        remove = self.global_states[remove_gid]

        # Merge galleries
        for emb, cam_id in zip(remove.gallery, remove.gallery_camera_ids):
            keep.gallery.append(emb)
            keep.gallery_camera_ids.append(cam_id)
        if len(keep.gallery) > self.gallery_size:
            # Camera-aware trim (same logic as _update_global_state)
            cam_last: Dict[Any, int] = {}
            for idx, cid in enumerate(keep.gallery_camera_ids):
                cam_last[cid] = idx
            keep_set = set(cam_last.values())
            remaining = [
                i for i in range(len(keep.gallery) - 1, -1, -1)
                if i not in keep_set
            ]
            for idx in remaining:
                if len(keep_set) >= self.gallery_size:
                    break
                keep_set.add(idx)
            keep_list = sorted(keep_set)[:self.gallery_size]
            keep.gallery = [keep.gallery[i] for i in keep_list]
            keep.gallery_camera_ids = [keep.gallery_camera_ids[i] for i in keep_list]
        keep.embedding = self._recompute_centroid(keep.gallery)

        # Merge camera histories
        for cam, ts in remove.seen_by_camera.items():
            if cam not in keep.seen_by_camera or ts > keep.seen_by_camera[cam]:
                keep.seen_by_camera[cam] = ts

        # Merge part features (average)
        if remove.part_feats is not None:
            if keep.part_feats is None:
                keep.part_feats = remove.part_feats.copy()
                keep.attn_weights = remove.attn_weights.copy() if remove.attn_weights is not None else None
            else:
                keep.part_feats = 0.5 * keep.part_feats + 0.5 * remove.part_feats
                norms = np.linalg.norm(keep.part_feats, axis=1, keepdims=True)
                keep.part_feats /= np.maximum(norms, 1e-12)
                if keep.attn_weights is not None and remove.attn_weights is not None:
                    keep.attn_weights = 0.5 * keep.attn_weights + 0.5 * remove.attn_weights

        if remove.last_timestamp is not None:
            if keep.last_timestamp is None or remove.last_timestamp > keep.last_timestamp:
                keep.last_timestamp = remove.last_timestamp

        # Redirect all local_to_global mappings
        for cam_id, local_map in self.local_to_global.items():
            for local_id in list(local_map.keys()):
                if local_map[local_id] == remove_gid:
                    local_map[local_id] = keep_gid

        # Delete the merged-away GID
        del self.global_states[remove_gid]
        if remove_gid in self.global_id_order:
            self.global_id_order.remove(remove_gid)

    def _prune_stale_local_ids(self, camera_id: Any, active_local_ids: Iterable[int]) -> None:
        active_set = {int(local_id) for local_id in active_local_ids}
        stale_local_ids = [
            local_id for local_id in self.local_to_global.get(camera_id, {}) if local_id not in active_set
        ]
        for local_id in stale_local_ids:
            del self.local_to_global[camera_id][local_id]

    def update(
        self,
        per_camera_tracks: Mapping[Any, Sequence[Any]],
        timestamp: TimestampInput = None,
    ) -> Dict[Any, Dict[int, int]]:
        """
        Update the global association state.

        Args:
            per_camera_tracks: camera_id -> sequence of track objects that expose
                `track_id` plus either `gallery.get_all()` and/or `smooth_feat`.
                Optionally `smooth_part_feats` and `smooth_attn_weights` for
                part-aware matching.
            timestamp: one shared scalar timestamp or a dict of camera_id -> timestamp.

        Returns:
            Mapping of camera_id -> {local_track_id: global_id} for the current update.
        """
        unmatched_by_camera: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

        for camera_id, tracks in per_camera_tracks.items():
            active_local_ids = []
            parsed_tracks: List[Tuple[Any, int, Optional[np.ndarray], Optional[float],
                                      Optional[np.ndarray], Optional[np.ndarray]]] = []
            camera_timestamp = self._resolve_timestamp(camera_id, timestamp)

            for track in tracks:
                local_id = getattr(track, "track_id", None)
                if local_id is None:
                    continue
                local_id = int(local_id)
                active_local_ids.append(local_id)
                gallery_matrix = self._track_gallery(track)
                part_feats, attn_weights = self._track_parts(track)
                parsed_tracks.append((track, local_id, gallery_matrix, camera_timestamp,
                                      part_feats, attn_weights))

            self._prune_stale_local_ids(camera_id, active_local_ids)

            for _, local_id, gallery_matrix, camera_timestamp, part_feats, attn_weights in parsed_tracks:
                existing_gid = self.local_to_global[camera_id].get(local_id)
                if existing_gid is not None and existing_gid in self.global_states:
                    self._update_global_state(existing_gid, gallery_matrix, camera_id, camera_timestamp,
                                              part_feats=part_feats, attn_weights=attn_weights)
                    continue

                if existing_gid is not None and existing_gid not in self.global_states:
                    del self.local_to_global[camera_id][local_id]

                if gallery_matrix is None:
                    continue

                unmatched_by_camera[camera_id].append(
                    {
                        "camera_id": camera_id,
                        "local_id": local_id,
                        "gallery": gallery_matrix,
                        "timestamp": camera_timestamp,
                        "part_feats": part_feats,
                        "attn_weights": attn_weights,
                    }
                )

        # Global assignment: build ONE cost matrix for ALL unmatched entries
        # across all cameras to eliminate camera-ordering bias.
        all_unmatched: List[Dict[str, Any]] = []
        for camera_id in sorted(unmatched_by_camera.keys(), key=str):
            all_unmatched.extend(unmatched_by_camera[camera_id])

        gids = list(self.global_states.keys())
        if gids and all_unmatched:
            cost_matrix = self._build_cost_matrix(all_unmatched, gids)
            matches, unmatched_rows, _ = self._solve_assignments(cost_matrix)
        else:
            matches = []
            unmatched_rows = list(range(len(all_unmatched)))

        for row_idx, col_idx in matches:
            entry = all_unmatched[row_idx]
            gid = gids[col_idx]
            self.local_to_global[entry["camera_id"]][entry["local_id"]] = gid
            self._update_global_state(
                gid,
                entry["gallery"],
                entry["camera_id"],
                entry["timestamp"],
                part_feats=entry.get("part_feats"),
                attn_weights=entry.get("attn_weights"),
            )

        # Second pass: remaining unmatched entries get new global IDs,
        # then attempt cross-matching among the newly-created set.
        still_unmatched = [all_unmatched[i] for i in unmatched_rows]
        new_gids: List[int] = []
        for entry in still_unmatched:
            new_gid = self._allocate_global_id(
                camera_id=entry["camera_id"],
                local_id=entry["local_id"],
                gallery_matrix=entry["gallery"],
                timestamp=entry["timestamp"],
                part_feats=entry.get("part_feats"),
                attn_weights=entry.get("attn_weights"),
            )
            new_gids.append(new_gid)

        # Merge pass: runs every frame (not just when new GIDs allocated)
        # to catch cold-start duplicates as galleries mature
        self._merge_pass()

        return {
            camera_id: dict(self.local_to_global.get(camera_id, {}))
            for camera_id in per_camera_tracks.keys()
        }
