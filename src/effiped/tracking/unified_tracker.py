"""
Unified multi-object tracker used by EffiPed Identity Review.

Combines proven techniques from:
  - ByteTrack: Two-threshold detection, low-confidence second association
  - FairMOT: Mahalanobis motion gating, proper Kalman filter, EMA features
  - BoT-SORT: Score-weighted cost fusion
  - OC-SORT: Observation-centric momentum during occlusions

Design goals:
  - Single bounded-memory tracker for local identity trajectories
  - Standard 8-state Kalman (x, y, a, h, vx, vy, va, vh) — same as FairMOT/ByteTrack
  - lap.lapjv for fast assignment (3-5x faster than scipy Hungarian)
  - Vectorized IoU via numpy broadcasting (no Python loops)
  - Configurable via dict, sensible defaults
  - No unbounded memory growth (bounded lost/removed buffers)
"""

from collections import deque

import lap
import numpy as np
from scipy.spatial.distance import cdist

# Chi-squared inverse at 95% confidence for different dimensions
_CHI2INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


# ============================================================================
#  Embedding Gallery (Diversity-Aware Multi-View Storage)
# ============================================================================

class EmbeddingGallery:
    """
    Stores up to max_size diverse embeddings per track.
    
    Diversity threshold: a new embedding is only added if its cosine distance
    to ALL existing entries exceeds min_diversity. This prevents storing
    near-duplicate views (e.g., consecutive frames from same angle) while
    preserving distinct viewpoints (front, side, back).
    
    Matching uses MINIMUM distance to any gallery entry (not average), so a
    front-view query matches the front-view gallery entry without interference
    from dissimilar back-view entries.
    """
    
    def __init__(self, max_size=10, min_diversity=0.005):
        self.max_size = max_size
        self.min_diversity = min_diversity
        self.embeddings = []  # List of L2-normalized numpy arrays
        self._cached_matrix = None
    
    def update(self, feat):
        """Add embedding if it's diverse enough from existing gallery entries."""
        if feat is None:
            return
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        
        if len(self.embeddings) == 0:
            self.embeddings.append(feat)
            self._cached_matrix = None
            return
        
        # Check diversity against all existing entries
        gallery = np.array(self.embeddings)
        cos_sims = gallery @ feat  # [N_gallery]
        cos_dists = 1.0 - cos_sims
        
        # Only add if cosine distance > threshold to ALL existing entries
        if np.all(cos_dists > self.min_diversity):
            if len(self.embeddings) >= self.max_size:
                # Replace the entry most similar to the new one (least diverse)
                most_similar_idx = np.argmin(cos_dists)
                self.embeddings[most_similar_idx] = feat
            else:
                self.embeddings.append(feat)
            self._cached_matrix = None
    
    def min_distance(self, query_feat):
        """Return minimum cosine distance between query and any gallery entry."""
        if len(self.embeddings) == 0:
            return 1.0  # Maximum distance
        query_feat = query_feat / (np.linalg.norm(query_feat) + 1e-12)
        gallery = self.get_all()
        cos_sims = gallery @ query_feat
        return float(1.0 - np.max(cos_sims))
    
    def get_all(self):
        """Return all gallery embeddings as numpy array."""
        if len(self.embeddings) == 0:
            return np.zeros((0, 0), dtype=np.float64)
        if self._cached_matrix is None:
            self._cached_matrix = np.asarray(self.embeddings, dtype=np.float64)
        return self._cached_matrix

    def clone(self):
        """Deep-copy gallery for archival / recovery usage."""
        cloned = EmbeddingGallery(max_size=self.max_size, min_diversity=self.min_diversity)
        cloned.embeddings = [e.copy() for e in self.embeddings]
        if self._cached_matrix is not None:
            cloned._cached_matrix = self._cached_matrix.copy()
        return cloned
    
    def __len__(self):
        return len(self.embeddings)


class GlobalMemoryBank:
    """
    Stores galleries from deleted tracks for long-term cross-gap re-identification.
    
    When a track is removed after being lost for max_time_lost frames, its gallery
    is archived here. New detections that don't match any active/lost track can
    query this bank to recover identities from long gaps.
    
    For MCMOT: each camera's tracker produces tracklets with galleries; cross-camera
    association module can compare gallery-to-gallery from this bank.
    """
    
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.bank = {}  # track_id -> EmbeddingGallery
        self._insertion_order = []  # For FIFO eviction
    
    def store(self, track_id, gallery):
        """Archive a track's gallery."""
        if len(gallery) == 0:
            return
        if track_id in self.bank:
            return  # Already stored
        # Evict oldest if at capacity
        while len(self.bank) >= self.capacity and self._insertion_order:
            oldest_id = self._insertion_order.pop(0)
            self.bank.pop(oldest_id, None)
        self.bank[track_id] = gallery.clone()
        self._insertion_order.append(track_id)
    
    def query(self, feat, threshold=0.3, return_gallery=False):
        """Find best matching track_id from memory bank.
        Returns (track_id, distance) or (None, 1.0) if no match.
        If return_gallery=True, also returns archived gallery clone."""
        if len(self.bank) == 0:
            if return_gallery:
                return None, 1.0, None
            return None, 1.0
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        best_id, best_dist = None, 1.0
        for tid, gallery in self.bank.items():
            dist = gallery.min_distance(feat)
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        if best_dist < threshold:
            if return_gallery:
                return best_id, best_dist, self.bank[best_id].clone()
            return best_id, best_dist
        if return_gallery:
            return None, 1.0, None
        return None, 1.0
    
    def remove(self, track_id):
        """Remove a track from the bank (e.g., after successful re-ID)."""
        self.bank.pop(track_id, None)
        if track_id in self._insertion_order:
            self._insertion_order.remove(track_id)
    
    def __len__(self):
        return len(self.bank)


# ============================================================================
#  Kalman Filter (8-state: x, y, a, h, vx, vy, va, vh)
# ============================================================================

class KalmanFilter:
    """
    Standard Kalman filter for bounding box tracking.
    State: [x, y, a, h, vx, vy, va, vh]
      - (x, y): bounding box center
      - a: aspect ratio (width / height)
      - h: height
      - (vx, vy, va, vh): velocities

    Measurement: [x, y, a, h]
    """

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create new track state from unassociated measurement [x, y, a, h]."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        """Run Kalman prediction step."""
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance, score=None):
        """Project state to measurement space. NSA-Kalman: inflate noise for low scores."""
        h = mean[3]
        std = np.array([
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ])
        if score is not None:
            # NSA-Kalman: high uncertainty for low-confidence detections
            std *= 1.0 + 3.0 * (1.0 - score)
        innovation_cov = np.diag(np.square(std))
        proj_mean = self._update_mat @ mean
        proj_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return proj_mean, proj_cov

    def update(self, mean, covariance, measurement, score=None):
        """Run Kalman correction step."""
        proj_mean, proj_cov = self.project(mean, covariance, score=score)
        try:
            chol = np.linalg.cholesky(proj_cov)
            K = np.linalg.lstsq(
                chol.T,
                np.linalg.lstsq(chol, (covariance @ self._update_mat.T).T, rcond=None)[0],
                rcond=None,
            )[0].T
        except np.linalg.LinAlgError:
            K = np.linalg.lstsq(proj_cov, (covariance @ self._update_mat.T).T, rcond=None)[0].T
        innovation = measurement - proj_mean
        new_mean = mean + innovation @ K.T
        new_cov = covariance - K @ proj_cov @ K.T
        return new_mean, new_cov

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute Mahalanobis distance for gating."""
        proj_mean, proj_cov = self.project(mean, covariance)
        if only_position:
            proj_mean = proj_mean[:2]
            proj_cov = proj_cov[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - proj_mean
        try:
            L = np.linalg.cholesky(proj_cov)
            z = np.linalg.lstsq(L, d.T, rcond=None)[0]
            return np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            return np.sum(d * d, axis=1)


# ============================================================================
#  Track State & STrack
# ============================================================================

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Single object track with Kalman state, EMA appearance features, diversity gallery, and OC-SORT momentum."""

    shared_kalman = KalmanFilter()
    _count = 0
    ema_alpha = 0.9  # Class-level default; overridden by UnifiedTracker from config

    def __init__(self, tlwh, score, feat, buffer_size=30):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # Appearance features (fused embedding)
        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=buffer_size)
        self.alpha = STrack.ema_alpha  # EMA decay for appearance (configurable via tracker config)
        
        # Part-based ReID features (6 part embeddings + attention weights)
        self.smooth_part_feats = None   # [6, C] EMA of part embeddings
        self.curr_part_feats = None     # [6, C] latest detection parts
        self.curr_attn_weights = None   # [6]    latest attention weights
        self.smooth_attn_weights = None # [6]    EMA attention weights
        
        # Diversity-aware gallery for multi-view matching
        self.gallery = EmbeddingGallery(max_size=10, min_diversity=0.005)
        
        if feat is not None:
            self.update_features(feat, score=score)

        # EMA box smoothing
        self.smooth_box = None

        # Track lifecycle
        self.state = TrackState.New
        self.frame_id = 0
        self.start_frame = 0
        self.track_id = 0

        # OC-SORT: Observation-centric momentum
        self.last_observation = None
        self.observations = deque([], maxlen=50)
        self.velocity = None

    # ---- Appearance ----

    def update_features(self, feat, part_feats=None, attn_weights=None, score=None):
        """EMA update of appearance features + gallery update.
        
        Args:
            feat: [C] fused embedding (L2-normalized)
            part_feats: [6, C] part embeddings (optional, for part-aware matching)
            attn_weights: [6] attention weights (optional)
            score: detection confidence — higher score → more weight to new feat
        """
        if feat is None:
            return
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        self.curr_feat = feat
        # Score-weighted EMA: high-confidence detections contribute more
        if score is not None and score > 0:
            alpha = self.alpha + (1 - self.alpha) * (1 - score)
        else:
            alpha = self.alpha
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
        self.features.append(feat)
        self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-12)
        # Update diversity gallery
        self.gallery.update(feat)
        
        # Part-based feature update (EMA per-part)
        if part_feats is not None:
            part_feats = part_feats / (np.linalg.norm(part_feats, axis=1, keepdims=True) + 1e-12)
            self.curr_part_feats = part_feats
            if self.smooth_part_feats is None:
                self.smooth_part_feats = part_feats.copy()
            else:
                self.smooth_part_feats = alpha * self.smooth_part_feats + (1 - alpha) * part_feats
                self.smooth_part_feats = self.smooth_part_feats / (np.linalg.norm(self.smooth_part_feats, axis=1, keepdims=True) + 1e-12)
        if attn_weights is not None:
            self.curr_attn_weights = attn_weights
            if self.smooth_attn_weights is None:
                self.smooth_attn_weights = attn_weights.copy()
            else:
                self.smooth_attn_weights = alpha * self.smooth_attn_weights + (1 - alpha) * attn_weights

    # ---- Kalman ----

    def predict(self):
        """Predict state using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0  # Zero height velocity when lost
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Predict all tracks (batch)."""
        for st in stracks:
            if st.state != TrackState.Tracked:
                # Damped velocity prevents runaway boxes without crippling camera motion tracking
                st.mean[4] *= 0.5
                st.mean[5] *= 0.5
                st.mean[6] *= 0.5
                st.mean[7] *= 0.5
            st.mean, st.covariance = STrack.shared_kalman.predict(st.mean, st.covariance)

    # ---- Lifecycle ----

    def activate(self, kalman_filter, frame_id):
        """Initialise a new tracklet."""
        self.kalman_filter = kalman_filter
        STrack._count += 1
        self.track_id = STrack._count
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        # OC-SORT
        self.last_observation = self._tlwh.copy()
        self.observations.append((frame_id, self._tlwh.copy()))

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a lost track with a new detection."""
        new_xyah = self.tlwh_to_xyah(new_track.tlwh)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            new_xyah,
            score=new_track.score,
        )
        # Prevent shape explosion under camera motion
        self.mean[2] = new_xyah[2]
        self.mean[3] = new_xyah[3]
        
        self.smooth_box = self.tlbr  # Reset box smoothing
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        # Feature updates should only happen on unoccluded, high-confidence detections
        if new_track.score >= 0.5:
            self.update_features(new_track.curr_feat, new_track.curr_part_feats, new_track.curr_attn_weights, score=new_track.score)
        if new_id:
            STrack._count += 1
            self.track_id = STrack._count
        self._update_observation(new_track.tlwh, frame_id)

    def update(self, new_track, frame_id):
        """Update a matched track with a new detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        new_xyah = self.tlwh_to_xyah(new_tlwh)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            new_xyah,
            score=new_track.score,
        )
        # Prevent shape explosion under camera motion
        self.mean[2] = new_xyah[2]
        self.mean[3] = new_xyah[3]
        
        # Disable EMA box smoothing for moving cameras (prevents lagging)
        self.smooth_box = self.tlbr
        self.state = TrackState.Tracked
        # Don't override is_activated here — let the tracker's update() handle
        # probation promotion via tracklet_len >= min_hits check
        if self.is_activated:
            pass  # Already confirmed — keep activated
        # If not activated, tracker's Step 4 will check tracklet_len for promotion
        self.score = new_track.score
        # Feature updates should only happen on unoccluded, high-confidence detections
        if new_track.score >= 0.5:
            self.update_features(new_track.curr_feat, new_track.curr_part_feats, new_track.curr_attn_weights, score=new_track.score)
        self._update_observation(new_tlwh, frame_id)

    # ---- OC-SORT ----

    def _update_observation(self, tlwh, frame_id):
        """Update observation history and compute velocity."""
        if self.last_observation is not None and len(self.observations) > 0:
            dt = frame_id - self.observations[-1][0]
            if dt > 0:
                vel = (tlwh[:2] - self.last_observation[:2]) / dt
                self.velocity = vel if self.velocity is None else 0.7 * self.velocity + 0.3 * vel
        self.last_observation = tlwh.copy()
        self.observations.append((frame_id, tlwh.copy()))

    def get_oc_sort_box(self, current_frame=None):
        """OC-SORT: Use last observation + velocity for prediction when occluded.
        
        Args:
            current_frame: The tracker's current frame_id. Required for correct
                          dt computation (track's self.frame_id may be stale).
        """
        if self.velocity is None or self.last_observation is None:
            return self.tlbr
        obs = self.last_observation.copy()
        # Use tracker's current_frame for correct dt (self.frame_id may equal
        # observations[-1][0] since both are set together in update())
        ref_frame = current_frame if current_frame is not None else self.frame_id
        dt = ref_frame - self.observations[-1][0] if self.observations else 0
        if dt > 0:
            obs[:2] += self.velocity * dt
        ret = obs.copy()
        ret[2:] += ret[:2]  # tlwh -> tlbr
        return ret

    # ---- Coordinate conversions ----

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # a*h = w
        ret[:2] -= ret[2:] / 2  # center -> top-left
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2  # top-left -> center
        ret[2] /= ret[3]        # w -> a (aspect ratio)
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @property
    def end_frame(self):
        return self.frame_id

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def __repr__(self):
        return f'T{self.track_id}({self.start_frame}-{self.end_frame})'


# ============================================================================
#  Utility Functions (Vectorized)
# ============================================================================

def _iou_batch(bboxes1, bboxes2):
    """Vectorized IoU between two sets of [x1,y1,x2,y2] boxes. Returns [N,M] IoU matrix."""
    bboxes1 = np.asarray(bboxes1)
    bboxes2 = np.asarray(bboxes2)
    b2 = np.expand_dims(bboxes2, 0)  # [1, M, 4]
    b1 = np.expand_dims(bboxes1, 1)  # [N, 1, 4]
    xx1 = np.maximum(b1[..., 0], b2[..., 0])
    yy1 = np.maximum(b1[..., 1], b2[..., 1])
    xx2 = np.minimum(b1[..., 2], b2[..., 2])
    yy2 = np.minimum(b1[..., 3], b2[..., 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter
    return np.where(union > 0, inter / union, 0.0)


def iou_distance(atracks, btracks, use_oc_sort=False, current_frame=None):
    """Compute IoU distance (1 - IoU) between tracks/detections.
    
    Args:
        current_frame: Tracker's current frame_id, passed to get_oc_sort_box()
                      for correct velocity extrapolation.
    """
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float64)
    if use_oc_sort and hasattr(atracks[0], 'get_oc_sort_box'):
        a_boxes = np.asarray([t.get_oc_sort_box(current_frame) for t in atracks])
    elif hasattr(atracks[0], 'tlbr'):
        a_boxes = np.asarray([t.tlbr for t in atracks])
    else:
        a_boxes = np.asarray(atracks)
    if hasattr(btracks[0], 'tlbr'):
        b_boxes = np.asarray([t.tlbr for t in btracks])
    else:
        b_boxes = np.asarray(btracks)
    return 1 - _iou_batch(a_boxes, b_boxes)


def embedding_distance(
    tracks,
    detections,
    metric='cosine',
    use_gallery=True,
    part_aware_alpha=0.4,
    gallery_cache=None,
):
    """Compute pairwise embedding distance with gallery-aware min-distance + part-aware matching.
    
    For each (track, detection) pair, computes:
      1) Fused distance: cosine(track.smooth_feat, det.curr_feat)
      2) Part-aware distance: mutual-visibility-weighted per-part cosine distance
      3) Gallery distance: min over gallery entries of cosine(entry, det.curr_feat)
    
    Part-aware matching (when available):
      For each of 6 body parts, computes cosine distance weighted by mutual visibility.
      mutual_vis[i] = min(track_attn[i], det_attn[i]) — ignores parts occluded in either.
      Combined: (1-α) * fused_distance + α * part_distance
    
    Final distance = min(combined_distance, gallery distance).
    Returns [N_tracks, M_dets].
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float64)
    det_features = np.asarray([d.curr_feat for d in detections], dtype=np.float64)
    track_features = np.asarray([t.smooth_feat for t in tracks], dtype=np.float64)
    
    # Standard EMA-based cosine distance (fused embeddings)
    ema_cost = np.maximum(0.0, cdist(track_features, det_features, metric))
    
    # Part-aware distance: per-part cosine weighted by mutual visibility
    has_parts = any(t.smooth_part_feats is not None for t in tracks) and \
                any(d.curr_part_feats is not None for d in detections)
    if has_parts and part_aware_alpha > 0:
        part_cost = np.full_like(ema_cost, 1.0)
        for i, track in enumerate(tracks):
            if track.smooth_part_feats is None or track.smooth_attn_weights is None:
                part_cost[i, :] = ema_cost[i, :]  # Fallback to fused
                continue
            for j, det in enumerate(detections):
                if det.curr_part_feats is None or det.curr_attn_weights is None:
                    part_cost[i, j] = ema_cost[i, j]  # Fallback to fused
                    continue
                # Per-part cosine similarity: dot product of L2-normalized part vectors
                part_sims = np.sum(track.smooth_part_feats * det.curr_part_feats, axis=1)  # [6]
                # Mutual visibility: min of attention weights (ignore parts occluded in either)
                mutual_vis = np.minimum(track.smooth_attn_weights, det.curr_attn_weights)  # [6]
                vis_sum = mutual_vis.sum()
                if vis_sum > 1e-8:
                    part_sim = np.sum(mutual_vis * part_sims) / vis_sum
                else:
                    part_sim = np.sum(part_sims) / 6.0  # Fallback: equal weights
                part_cost[i, j] = 1.0 - part_sim  # Convert similarity → distance
        # Combine fused and part-aware distances
        ema_cost = (1 - part_aware_alpha) * ema_cost + part_aware_alpha * part_cost
    
    if not use_gallery:
        return ema_cost

    # Gallery-based min-distance (vectorized across all track galleries)
    gallery_cost = np.full_like(ema_cost, 1.0)
    gallery_blocks = []
    owner_indices = []
    for i, track in enumerate(tracks):
        if gallery_cache is not None and id(track) in gallery_cache:
            gallery_mat = gallery_cache[id(track)]
        else:
            gallery_mat = track.gallery.get_all() if len(track.gallery) > 0 else None
        if gallery_mat is None or gallery_mat.ndim != 2 or gallery_mat.shape[0] == 0:
            continue
        gallery_blocks.append(gallery_mat)
        owner_indices.append(np.full(gallery_mat.shape[0], i, dtype=np.int32))

    if gallery_blocks:
        all_gallery = np.concatenate(gallery_blocks, axis=0)
        if metric == 'cosine':
            g_dists = np.maximum(0.0, 1.0 - all_gallery @ det_features.T)
        else:
            g_dists = np.maximum(0.0, cdist(all_gallery, det_features, metric))
        owners = np.concatenate(owner_indices, axis=0)
        for track_idx in np.unique(owners):
            gallery_cost[track_idx] = g_dists[owners == track_idx].min(axis=0)

    # Take element-wise minimum: best of EMA or gallery
    return np.minimum(ema_cost, gallery_cost)


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.95, reid_trust=0.2):
    """Fuse embedding cost with Mahalanobis motion distance.
    
    lambda_: embedding weight (1-lambda_ = motion weight). Higher = more ReID trust.
    reid_trust: cosine distance threshold — trust ReID even when motion gate says no.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = _CHI2INV95[gating_dim]
    measurements = np.asarray([d.to_xyah() for d in detections])
    for row, track in enumerate(tracks):
        g_dist = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        # Gate: only mark as inf if BOTH motion and appearance are bad
        # (trust strong ReID even when motion looks unlikely)
        gate_mask = (g_dist > gating_threshold) & (cost_matrix[row] > reid_trust)
        cost_matrix[row, gate_mask] = np.inf
        # Normalize Mahalanobis distance to [0, 1] using the 95% confidence threshold
        normalized_g_dist = np.clip(g_dist / gating_threshold, 0, 1.0)
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * normalized_g_dist
    return cost_matrix


def fuse_score(cost_matrix, detections, alpha=0.3):
    """Fuse detection score into IoU cost matrix.
    
    Uses BoT-SORT formula: fuse_cost = 1 - iou_sim * det_confs
    This multiplicatively penalizes low-confidence detections.
    The alpha parameter is ignored (kept for API compatibility).
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_confs = np.array([d.score for d in detections])
    det_confs = np.expand_dims(det_confs, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_confs
    return 1 - fuse_sim


def linear_assignment(cost_matrix, thresh):
    """Solve linear assignment using lap.lapjv (Jonker-Volgenant)."""
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
    unmatched_a = tuple(np.where(x < 0)[0])
    unmatched_b = tuple(np.where(y < 0)[0])
    return np.asarray(matches).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int), unmatched_a, unmatched_b


def joint_stracks(list_a, list_b):
    """Combine two STrack lists, no duplicates by track_id."""
    seen = {}
    res = []
    for t in list_a:
        seen[t.track_id] = 1
        res.append(t)
    for t in list_b:
        if t.track_id not in seen:
            seen[t.track_id] = 1
            res.append(t)
    return res


def sub_stracks(list_a, list_b):
    """Remove list_b track_ids from list_a."""
    ids_b = {t.track_id for t in list_b}
    return [t for t in list_a if t.track_id not in ids_b]


def remove_duplicate_stracks(stracks_a, stracks_b):
    """Remove duplicate tracks based on IoU distance."""
    pdist = iou_distance(stracks_a, stracks_b)
    pairs = np.where(pdist < 0.15)
    dup_a, dup_b = set(), set()
    for p, q in zip(*pairs):
        time_a = stracks_a[p].frame_id - stracks_a[p].start_frame
        time_b = stracks_b[q].frame_id - stracks_b[q].start_frame
        if time_a > time_b:
            dup_b.add(q)
        else:
            dup_a.add(p)
    res_a = [t for i, t in enumerate(stracks_a) if i not in dup_a]
    res_b = [t for i, t in enumerate(stracks_b) if i not in dup_b]
    return res_a, res_b


# ============================================================================
#  Unified Tracker
# ============================================================================

class UnifiedTracker:
    """
    Production MOT tracker combining ByteTrack + FairMOT + BoT-SORT + OC-SORT.

    Association pipeline:
      Step 1: High-conf dets ↔ tracked+lost tracks (embedding + motion + score fusion)
      Step 2: Remaining high-conf dets ↔ remaining tracks (IoU, OC-SORT enhanced)
      Step 3: Low-conf dets ↔ still-unmatched tracks (IoU only — ByteTrack)
      Step 4: Unconfirmed tracks ↔ remaining dets (IoU)
      Step 5: Init new tracks from leftover high-conf dets
      Step 6: Lifecycle management (lost → removed after max_time_lost)

    Config (all tunable via dict):
      det_thresh:       High-confidence detection threshold (default 0.4)
      low_thresh:       Low-confidence threshold for ByteTrack (default 0.1)
      match_thresh:     Max cost for embedding+motion match (default 0.5)
      track_buffer:     Frames to keep lost tracks (default 30)
      frame_rate:       Video frame rate (default 30)
      use_oc_sort:      Use observation-centric IoU (default True)
      use_botsort:      Use BoT-SORT score fusion (default True)
      botsort_alpha:    Weight for score in cost (default 0.3)
    """

    def __init__(self, config=None, frame_rate=30):
        cfg = config or {}

        self.det_thresh = cfg.get('det_thresh', 0.4)
        self.low_thresh = cfg.get('low_thresh', 0.1)
        self.match_thresh = cfg.get('match_thresh', 0.4)
        self.use_oc_sort = cfg.get('use_oc_sort', True)
        self.use_botsort = cfg.get('use_botsort', True)
        self.botsort_alpha = cfg.get('botsort_alpha', 0.3)

        self.min_hits = cfg.get('min_hits', 3)
        self.frame_rate = frame_rate

        track_buffer = cfg.get('track_buffer', 30)
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.buffer_size = self.max_time_lost

        self.kalman_filter = KalmanFilter()

        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0

        # Global memory bank for long-term re-identification
        memory_capacity = cfg.get('memory_bank_capacity', 500)
        self.memory_bank = GlobalMemoryBank(capacity=memory_capacity)
        self.use_gallery = cfg.get('use_gallery', True)
        self.reid_thresh = cfg.get('reid_thresh', 0.4)

        # Set EMA alpha for track appearance smoothing (class-level so all STracks use it)
        STrack.ema_alpha = cfg.get('embedding_fusion_alpha', 0.9)

        # Configurable fusion parameters
        self.motion_lambda = cfg.get('motion_lambda', cfg.get('embedding_weight', 0.85))    # fuse_motion: higher = more ReID
        self.reid_trust = cfg.get('reid_trust', 0.2)              # motion gate: trust ReID even when motion says no
        self.step2_reid_weight = cfg.get('step2_reid_weight', 0.5) # Step 2: ReID weight in IoU+ReID blend
        self.part_aware_alpha = cfg.get('part_aware_alpha', 0.4)
        self._part_aware_requested = 'part_aware_alpha' in cfg
        self._part_aware_missing_warned = False

        # New track threshold: only dets above this score create new tracks
        # None = use det_thresh (original behavior)
        self.new_track_thresh = cfg.get('new_track_thresh', None)

        # Appearance threshold: hard gate on raw embedding distance in Step 1
        # If raw cosine distance > this, the match is rejected even if motion/score say yes
        # None = disabled (original behavior)
        self.appearance_thresh = cfg.get('appearance_thresh', None)

        # BoT-SORT-style first association: min(IoU_cost, emb_cost/2) with proximity gating
        # proximity_thresh: IoU distance above which embedding is disabled (default 0.5)
        # fuse_first_associate: fuse detection score into IoU cost before min() (default False)
        self.proximity_thresh = cfg.get('proximity_thresh', 0.5)
        self.fuse_first_associate = cfg.get('fuse_first_associate', False)

        # Stage thresholds (configurable; defaults preserve current behavior)
        self.step2_match_thresh = cfg.get('step2_match_thresh', 0.5)
        self.step3_match_thresh = cfg.get('step3_match_thresh', 0.5)
        self.unconfirmed_match_thresh = cfg.get('unconfirmed_match_thresh', 0.7)
        self.overlap_mask_thresh = cfg.get('overlap_mask_thresh', 0.7)

        # Reset global track counter
        STrack._count = 0

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections: One of:
              - dict with 'boxes' [N,4], 'scores' [N], 'embeddings' [N,D]
                (direct output from head.decode_detections)
              - list of [x1, y1, x2, y2, score, class, embedding]

        Returns:
            List of [x1, y1, x2, y2, track_id, score] for active tracks
        """
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        # -------- Parse detections into STrack objects --------
        det_stracks = self._parse_detections(detections)
        if self._part_aware_requested and self.part_aware_alpha > 0 and not self._part_aware_missing_warned:
            has_part_features = any(
                d.curr_part_feats is not None and d.curr_attn_weights is not None
                for d in det_stracks
            )
            if not has_part_features:
                print(
                    "[UnifiedTracker] part_aware_alpha requested but detections have no part features. "
                    "Falling back to fused-embedding matching."
                )
                self._part_aware_missing_warned = True

        # ByteTrack: split by confidence
        dets_high = [d for d in det_stracks if d.score >= self.det_thresh]
        dets_low = [d for d in det_stracks if self.low_thresh <= d.score < self.det_thresh]

        # Split tracked into confirmed and unconfirmed
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        # ========================================================
        # Step 1: High-conf dets ↔ (tracked + lost) tracks
        #         BoT-SORT-style: min(IoU_cost, emb_cost/2) with proximity gating
        # ========================================================
        strack_pool = joint_stracks(tracked, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # IoU cost (primary spatial matching)
        ious_dists = iou_distance(strack_pool, dets_high)
        ious_dists_mask = ious_dists > self.proximity_thresh  # boxes too far apart

        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, dets_high)

        # Embedding cost (appearance matching)
        gallery_cache_step1 = self._build_gallery_cache(strack_pool) if self.use_gallery else None
        emb_dists = embedding_distance(
            strack_pool,
            dets_high,
            use_gallery=self.use_gallery,
            part_aware_alpha=self.part_aware_alpha,
            gallery_cache=gallery_cache_step1,
        )
        emb_dists = emb_dists / 2.0  # Scale to [0, 1] range (cosine dist is [0, 2])

        # Appearance hard gate: reject if embedding too dissimilar
        if self.appearance_thresh is not None:
            emb_dists[emb_dists > self.appearance_thresh] = 1.0

        # Proximity gate: disable embedding for spatially distant pairs
        emb_dists[ious_dists_mask] = 1.0

        # Final cost: take the minimum — if EITHER modality matches, the match succeeds
        dists = np.minimum(ious_dists, emb_dists)

        matches, u_track, u_det = linear_assignment(dists, thresh=self.match_thresh)
        for it, id_ in matches:
            track = strack_pool[it]
            det = dets_high[id_]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        # ========================================================
        # Step 2: Remaining high-conf dets ↔ remaining tracked (IoU only)
        #         Pure spatial matching for tracks that failed appearance match
        # ========================================================
        dets_remain = [dets_high[i] for i in u_det]
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists = iou_distance(r_tracked, dets_remain, use_oc_sort=self.use_oc_sort, current_frame=self.frame_id)
        
        matches, u_track2, u_det2 = linear_assignment(dists, thresh=self.step2_match_thresh)
        for it, id_ in matches:
            track = r_tracked[it]
            det = dets_remain[id_]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        # ========================================================
        # Step 3: ByteTrack — Low-conf dets ↔ still-unmatched tracked
        #         IoU only (recover occluded targets from partial dets)
        # ========================================================
        r_tracked2 = [r_tracked[i] for i in u_track2]
        dets_remain2 = [dets_remain[i] for i in u_det2]

        if r_tracked2 and dets_low:
            dists = iou_distance(r_tracked2, dets_low, use_oc_sort=self.use_oc_sort, current_frame=self.frame_id)
            
            # ReID Guard: prevent pure IoU from stealing visually orthogonal clutter boxes
            has_emb_low = any(t.smooth_feat is not None for t in r_tracked2) and any(d.curr_feat is not None for d in dets_low)
            if has_emb_low:
                emb_dists = embedding_distance(r_tracked2, dets_low, use_gallery=False, part_aware_alpha=0.0)
                dists[emb_dists > 0.6] = np.inf
                
            matches, u_track_low, _ = linear_assignment(dists, thresh=self.step3_match_thresh)
            for it, id_ in matches:
                track = r_tracked2[it]
                det = dets_low[id_]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind.append(track)
            for i in u_track_low:
                t = r_tracked2[i]
                if t.state != TrackState.Lost:
                    t.mark_lost()
                    lost.append(t)
        else:
            for t in r_tracked2:
                if t.state != TrackState.Lost:
                    t.mark_lost()
                    lost.append(t)

        # ========================================================
        # Step 4: Unconfirmed tracks ↔ remaining high-conf dets
        # ========================================================
        dists = iou_distance(unconfirmed, dets_remain2)
        matches, u_unconf, u_det_final = linear_assignment(dists, thresh=self.unconfirmed_match_thresh)
        for it, id_ in matches:
            unconfirmed[it].update(dets_remain2[id_], self.frame_id)
            # Check for promotion from probation
            if unconfirmed[it].tracklet_len >= self.min_hits:
                unconfirmed[it].is_activated = True
                activated.append(unconfirmed[it])
            else:
                # Still in probation — keep in tracked pool (as unconfirmed)
                # so it can be matched again next frame
                activated.append(unconfirmed[it])
                
        for i in u_unconf:
            unconfirmed[i].mark_removed()
            removed.append(unconfirmed[i])

        # ========================================================
        # Step 5: Memory bank recovery — query unmatched dets
        #         against archived galleries from long-lost tracks
        #         (runs BEFORE new track init to recover IDs first)
        # ========================================================
        if self.use_gallery and len(dets_remain2) > 0:
            still_unmatched = []
            for i in u_det_final:
                det = dets_remain2[i]
                if det.score >= self.det_thresh and det.curr_feat is not None:
                    recovered_id, dist, archived_gallery = self.memory_bank.query(
                        det.curr_feat,
                        threshold=self.reid_thresh,
                        return_gallery=True,
                    )
                    if recovered_id is not None:
                        # Recover identity from memory bank
                        self.memory_bank.remove(recovered_id)
                        det.activate(self.kalman_filter, self.frame_id)
                        det.track_id = recovered_id
                        if archived_gallery is not None and len(archived_gallery) > 0:
                            det.gallery = archived_gallery
                            if det.curr_feat is not None:
                                det.gallery.update(det.curr_feat)
                            gallery_mat = det.gallery.get_all()
                            if gallery_mat.ndim == 2 and gallery_mat.shape[0] > 0:
                                restored_feat = gallery_mat.mean(axis=0)
                                det.smooth_feat = restored_feat / (np.linalg.norm(restored_feat) + 1e-12)
                        det.is_activated = True
                        det.state = TrackState.Tracked
                        activated.append(det)
                    else:
                        still_unmatched.append(i)
                else:
                    still_unmatched.append(i)
            u_det_final = still_unmatched

        # ========================================================
        # Step 5b: Init new tracks from unmatched high-conf dets
        # ========================================================
        init_thresh = self.new_track_thresh if self.new_track_thresh is not None else self.det_thresh
        for i in u_det_final:
            det = dets_remain2[i]
            if det.score >= init_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                if self.min_hits <= 1:
                    # No probation — immediately confirmed
                    det.is_activated = True
                    det.state = TrackState.Tracked
                    activated.append(det)
                else:
                    # Probation: track must be matched min_hits times before
                    # being output. Add to tracked pool (as unconfirmed) so it
                    # survives to the next frame and can be matched in Step 4.
                    det.is_activated = False
                    det.state = TrackState.Tracked  # Must be Tracked to survive pool rebuild
                    activated.append(det)  # Add to pool — will appear as unconfirmed next frame


        # ========================================================
        # Step 6: Lifecycle — expire old lost tracks
        #         Archive their galleries to memory bank
        # ========================================================
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                # Archive gallery before removing
                if self.use_gallery and len(track.gallery) > 0:
                    self.memory_bank.store(track.track_id, track.gallery)
                track.mark_removed()
                removed.append(track)

        # Update track pools
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        # Cap removed buffer to prevent unbounded growth
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-500:]
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # ========================================================
        # Output: active tracks with smoothed boxes
        # ========================================================
        output = []
        for t in self.tracked_stracks:
            if t.is_activated:
                box = t.smooth_box if t.smooth_box is not None else t.tlbr
                output.append([box[0], box[1], box[2], box[3], t.track_id, t.score])
        return output

    # ---- Internal helpers ----

    @staticmethod
    def _build_gallery_cache(tracks):
        """Build per-track gallery matrix cache for one association cycle."""
        cache = {}
        for track in tracks:
            if len(track.gallery) == 0:
                continue
            gallery_mat = track.gallery.get_all()
            if gallery_mat.ndim == 2 and gallery_mat.shape[0] > 0:
                cache[id(track)] = gallery_mat
        return cache

    def _parse_detections(self, detections):
        """Convert various detection formats to list of STrack."""
        if detections is None or (isinstance(detections, (list, np.ndarray)) and len(detections) == 0):
            return []

        stracks = []

        if isinstance(detections, dict):
            # Dict format from decode_detections: {boxes, scores, embeddings, part_embeddings, attention_weights}
            boxes = detections['boxes']
            scores = detections['scores']
            embeddings = detections.get('embeddings', None)
            part_embeddings = detections.get('part_embeddings', None)
            attention_weights = detections.get('attention_weights', None)

            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                if embeddings is not None:
                    embeddings = embeddings.cpu().numpy()
                if part_embeddings is not None:
                    part_embeddings = part_embeddings.cpu().numpy()
                if attention_weights is not None:
                    attention_weights = attention_weights.cpu().numpy()

            for i in range(len(scores)):
                x1, y1, x2, y2 = boxes[i]
                emb = embeddings[i] if embeddings is not None else None
                parts = part_embeddings[i] if part_embeddings is not None else None
                attn = attention_weights[i] if attention_weights is not None else None
                tlwh = [x1, y1, x2 - x1, y2 - y1]
                st = STrack(tlwh, float(scores[i]), emb, self.buffer_size)
                # Attach part features directly (update_features in __init__ only has fused)
                if parts is not None:
                    parts = parts / (np.linalg.norm(parts, axis=1, keepdims=True) + 1e-12)
                    st.curr_part_feats = parts
                    st.smooth_part_feats = parts.copy()
                if attn is not None:
                    st.curr_attn_weights = attn
                    st.smooth_attn_weights = attn.copy()
                stracks.append(st)
        else:
            # List format: [[x1, y1, x2, y2, score, cls, embedding], ...]
            for det in detections:
                x1, y1, x2, y2, score = det[:5]
                emb = det[6] if len(det) > 6 else None
                if hasattr(emb, 'cpu'):
                    emb = emb.cpu().numpy()
                tlwh = [x1, y1, x2 - x1, y2 - y1]
                stracks.append(STrack(tlwh, float(score), emb, self.buffer_size))

        return stracks
