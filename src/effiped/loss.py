"""
effiped Loss Module — CenterNet Multi-Task Loss
═══════════════════════════════════════════════════════

Table of Contents (approximate line numbers):
─────────────────────────────────────────────
  L49    Helper: CosineClassifier — BOT/TransReID-style cosine CE (no margin)
  L79    Helper: ArcFaceLoss — Angular-margin classifier for ReID (sub-center K support)
  L149   Helper: CrossBatchMemory — XBM FIFO queue with camera-aware mining (circle/triplet)
  L324   Helper: BatchHardTripletLoss — Hard mining triplet for metric learning
  L379   Helper: CircleLoss — Self-paced pair similarity optimization (Sun et al., CVPR 2020)
  L473   Helper: RelationalDistillLoss — Pairwise similarity alignment for KD
  L567   Helper: FeatureDistillLoss — Cosine embedding alignment for KD
  L620   Function: uniformity_loss — BAU global hypersphere spreading (Cho et al., NeurIPS 2024)
  L640   Function: camera_uniformity_loss — BAU per-camera uniformity
  L675   Utility: bbox_giou — Generalized IoU between box pairs
  L701   Utility: gaussian_radius / draw_gaussian — Heatmap target rendering
  L788   Function: generate_centernet_targets — GT heatmap + multi-positive targets
  L958   Function: modified_focal_loss — Gaussian focal loss for heatmaps (per-pixel vis weight)
  L971   Function: repulsion_loss — Anti-overlap penalty (GT-GT IoU-gated)
  L1151  Class: CenterNetLoss
  L1169    __init__() — Loss config, BNNeck, ArcFace, XBM/Circle, BAU, uncertainty, distillation
  L1408    forward() — Main loss computation entry point
  L1429      Heatmap Loss — Focal loss with per-pixel visibility boost
  L1456      Regression Loss — GIoU (LTRB) + L1 offset at multi-positive pixels
  L1507      Repulsion Loss — Anti-overlap between neighbor boxes
  L1514      ReID / ID Loss — Part-based RoI extraction + ArcFace + Circle/Triplet
  L1624      BNNeck split → fused + per-part ArcFace/Circle (XBM-backed)
  L1756      BAU Uniformity — Global + per-camera hypersphere spreading
  L1772      Knowledge Distillation — Relational similarity alignment with OSNet teacher
  L1835      IoU Quality Loss — Gaussian-weighted BCE on predicted IoU
  L1872      Total Loss Assembly — 2-param uncertainty or fixed weighting

Key Invariants:
  • forward() returns 8-tuple — ALL callers unpack this exact shape
  • BNNeck: pre-BN → triplet (metric), post-BN → ArcFace (classification)
  • ID warm-up: id_weight ramps 0→full over id_warmup_epochs (set in train.py)
  • Defaults are "off" state: diversity_loss_weight=0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .models.common import sample_feature_map_bilinear


# ═══════════════════════════════════════════════════════════════
#  LOSS HELPERS — Classifiers & Triplet
# ═══════════════════════════════════════════════════════════════

class CosineClassifier(nn.Module):
    """
    Cosine classifier (BOT/TransReID-style) — standard ReID ID loss.

    L2-normalizes both embeddings and weights, computes scaled cosine logits.
    Used with F.cross_entropy(label_smoothing=ε). No angular margin.

    This is the universal choice in PCB, MGN, BOT, TransReID, ISP, AAformer.
    """
    def __init__(self, embedding_dim, num_classes, s=16.0):
        super().__init__()
        self.s = s
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward_logits(self, embeddings, labels=None):
        """Scaled cosine logits — same signature as ArcFaceLoss for drop-in use."""
        return self.class_logits(embeddings) * self.s

    def class_logits(self, embeddings):
        """Margin-free cosine logits."""
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-6)
        weight_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-6)
        return F.linear(embeddings, weight_norm).clamp(-1, 1)

    def forward(self, embeddings, labels):
        return F.cross_entropy(self.forward_logits(embeddings, labels), labels)


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) for discriminative ReID embeddings.
    Scale s and margin m are configurable; defaults calibrated for small-ID
    datasets (196 cross-camera IDs in P-DESTRE).
    
    Sub-center ArcFace (K>1): each class gets K sub-centers in the weight
    matrix. Cosine similarity is computed against all K sub-centers and the
    max is taken per class. This relaxes the intra-class compactness
    constraint, letting the model handle noisy labels and multi-modal
    appearance distributions (e.g., same person in different lighting/pose).
    """
    def __init__(self, embedding_dim, num_classes, s=16.0, m=0.25, easy_margin=False, subcenter_k=1):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.K = subcenter_k
        self.num_classes = num_classes

        # K sub-centers per class: weight is [num_classes * K, embedding_dim]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * subcenter_k, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward_logits(self, embeddings, labels):
        """Returns scaled logits for external loss computation (e.g., visibility weighting).
        
        Embeddings do NOT need to be pre-normalized — this method handles
        L2-normalization internally to compute proper cosine similarity.
        This allows callers to pass BNNeck output directly, preserving the
        BNNeck affine transform until the last possible moment.
        """
        cos_theta = self.class_logits(embeddings)
        sin_theta = torch.sqrt((1.0 - cos_theta.pow(2)).clamp(min=1e-7))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        return output * self.s

    def class_logits(self, embeddings):
        """Return margin-free class logits with sub-centers collapsed per class."""
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-6)
        weight_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-6)

        if self.K > 1:
            # Sub-center: compute cosine against all K*C sub-centers, then max-pool per class
            # cos_all: [batch, num_classes * K]
            cos_all = F.linear(embeddings, weight_norm).clamp(-1, 1)
            # Reshape to [batch, num_classes, K] and take max over sub-centers
            cos_theta = cos_all.view(-1, self.num_classes, self.K).max(dim=2).values
        else:
            cos_theta = F.linear(embeddings, weight_norm).clamp(-1, 1)
        return cos_theta

    def forward(self, embeddings, labels):
        return F.cross_entropy(self.forward_logits(embeddings, labels), labels)


class CrossBatchMemory(nn.Module):
    """
    Cross-Batch Memory (XBM) for triplet mining at small batch sizes.
    
    Maintains a FIFO queue of L2-normalized embeddings + labels from previous
    batches. Triplet mining runs against the full queue instead of the tiny
    current batch, dramatically improving hard-negative discovery at BS=2.
    
    Camera-aware mode: stores camera IDs alongside embeddings and weights
    cross-camera pairs higher during hard mining (same person from different
    cameras = most informative positive; different person same camera = hardest negative).
    
    Based on: Wang et al., "Cross-Batch Memory for Embedding Learning" (CVPR 2020)
    
    Usage:
        xbm = CrossBatchMemory(embedding_dim=256, memory_size=4096, margin=0.5)
        loss = xbm(current_embeddings, current_labels)  # auto-enqueues
    """
    def __init__(self, embedding_dim, memory_size=4096, margin=0.5, start_after=200,
                 camera_aware=False, cross_camera_weight=3.0,
                 use_circle_loss=False, circle_m=0.25, circle_gamma=128):
        super().__init__()
        self.memory_size = memory_size
        self.margin = margin
        self.start_after = start_after  # Skip first N enqueue ops (queue needs filling)
        self.camera_aware = camera_aware
        self.cross_camera_weight = cross_camera_weight
        self.use_circle_loss = use_circle_loss
        if use_circle_loss:
            self.circle_loss = CircleLoss(m=circle_m, gamma=circle_gamma)
        
        # FIFO queue (not a parameter — no gradients through stored embeddings)
        self.register_buffer('queue_embeddings', torch.zeros(memory_size, embedding_dim))
        self.register_buffer('queue_labels', torch.full((memory_size,), -1, dtype=torch.long))
        self.register_buffer('queue_cameras', torch.full((memory_size,), -1, dtype=torch.long))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('enqueue_count', torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def enqueue(self, embeddings, labels, cameras=None):
        """Add embeddings to the FIFO queue (detached, no gradient)."""
        batch_size = embeddings.shape[0]
        ptr = int(self.queue_ptr.item())
        
        if ptr + batch_size > self.memory_size:
            # Wrap around
            overflow = (ptr + batch_size) - self.memory_size
            self.queue_embeddings[ptr:] = embeddings[:batch_size - overflow].detach()
            self.queue_labels[ptr:] = labels[:batch_size - overflow].detach()
            self.queue_embeddings[:overflow] = embeddings[batch_size - overflow:].detach()
            self.queue_labels[:overflow] = labels[batch_size - overflow:].detach()
            if cameras is not None:
                self.queue_cameras[ptr:] = cameras[:batch_size - overflow].detach()
                self.queue_cameras[:overflow] = cameras[batch_size - overflow:].detach()
        else:
            self.queue_embeddings[ptr:ptr + batch_size] = embeddings.detach()
            self.queue_labels[ptr:ptr + batch_size] = labels.detach()
            if cameras is not None:
                self.queue_cameras[ptr:ptr + batch_size] = cameras.detach()
        
        self.queue_ptr[0] = (ptr + batch_size) % self.memory_size
        self.enqueue_count[0] += 1  # Count enqueue calls, not embeddings
    
    def forward(self, embeddings, labels, cameras=None):
        """
        Compute triplet loss using current batch + memory queue.
        
        Args:
            embeddings: [N, D] L2-normalized embeddings (current batch, HAS gradients)
            labels: [N] identity labels
            cameras: [N] camera IDs (optional, for camera-aware mining)
        Returns:
            triplet loss scalar
        """
        # Enqueue current batch AFTER mining (detached copy goes into history)
        # This ensures we only mine against previous batches, not the current one.
        # Mining first, then enqueue — prevents trivially easy self-matches.
        
        # Don't compute loss until queue has enough entries
        if self.enqueue_count.item() < self.start_after:
            self.enqueue(embeddings, labels, cameras)
            return torch.tensor(0.0, device=embeddings.device)
        
        # Get valid queue entries (labels != -1)
        valid_mask = self.queue_labels >= 0
        if valid_mask.sum() < 10:
            self.enqueue(embeddings, labels, cameras)
            return torch.tensor(0.0, device=embeddings.device)
        
        mem_embeddings = self.queue_embeddings[valid_mask]  # [M, D] — detached
        mem_labels = self.queue_labels[valid_mask]           # [M]
        mem_cameras = self.queue_cameras[valid_mask] if self.camera_aware else None  # [M]
        
        # Current batch embeddings are anchors (HAVE gradients)
        # Memory embeddings are candidates (NO gradients — stop-gradient by design)
        # Cosine similarity: anchor vs memory
        cos_sim = torch.mm(embeddings, mem_embeddings.t())  # [N, M]
        
        # For each anchor: identify positive and negative pairs in memory
        anchor_labels = labels.view(-1, 1)      # [N, 1]
        mem_labels_row = mem_labels.view(1, -1)  # [1, M]
        
        pos_mask_bool = (anchor_labels == mem_labels_row)  # [N, M]
        neg_mask_bool = (anchor_labels != mem_labels_row)  # [N, M]
        
        if self.use_circle_loss:
            # ---- Circle Loss path: self-paced weighting over ALL pairs in memory ----
            # Advantages over batch-hard triplet:
            #   - Uses ALL pairs (not just hardest) → richer gradient signal
            #   - Self-paced weights auto-amplify hard cases, suppress easy ones
            #   - No fixed margin threshold → smoother convergence
            # Camera-aware: boost cross-cam positives and same-cam negatives
            cross_cam_mask = None
            if self.camera_aware and cameras is not None and mem_cameras is not None:
                anchor_cams = cameras.view(-1, 1)       # [N, 1]
                mem_cams_row = mem_cameras.view(1, -1)   # [1, M]
                cross_cam_mask = (anchor_cams != mem_cams_row)  # [N, M]
            loss = self.circle_loss.compute_from_similarity(
                cos_sim, pos_mask_bool, neg_mask_bool,
                cross_cam_mask=cross_cam_mask,
                cross_cam_boost=self.cross_camera_weight
            )
            self.enqueue(embeddings, labels, cameras)
            return loss
        
        # ---- Triplet Loss path: batch-hard mining against memory ----
        dist_mat = 1 - cos_sim  # [N, M]
        pos_mask = pos_mask_bool.float()
        neg_mask = neg_mask_bool.float()
        
        # Camera-aware mining: bias selection toward cross-camera pairs
        if self.camera_aware and cameras is not None and mem_cameras is not None:
            anchor_cams = cameras.view(-1, 1)       # [N, 1]
            mem_cams_row = mem_cameras.view(1, -1)   # [1, M]
            cross_cam = (anchor_cams != mem_cams_row).float()  # [N, M]
            
            # Bias magnitude: how much to shift distances for camera-aware selection.
            # Typical cosine distances are [0, 0.5] (similar) to [1.0, 2.0] (dissimilar).
            # A bias of ~0.2 is enough to prefer cross-cam positives and same-cam negatives
            # without overwhelming the actual distance signal.
            cam_bias = (self.cross_camera_weight - 1.0) * 0.1  # default: (3-1)*0.1 = 0.2
            
            # For positives: boost cross-camera distance → selected as "hardest positive"
            # → forces embeddings to bridge the camera domain gap
            pos_dist = dist_mat * pos_mask - 1e9 * (1 - pos_mask)
            pos_dist = pos_dist + cam_bias * cross_cam * pos_mask
            
            # For negatives: reduce same-camera distance → selected as "hardest negative"
            # → forces embeddings to distinguish same-camera different-ID pairs
            neg_dist = dist_mat * neg_mask + 1e9 * (1 - neg_mask)
            neg_dist = neg_dist - cam_bias * (1 - cross_cam) * neg_mask
            # Clamp to prevent negative distances (which inflate effective margin)
            neg_dist = neg_dist.clamp(min=0.0)
        else:
            # Standard hard mining (no camera awareness)
            pos_dist = dist_mat * pos_mask - 1e9 * (1 - pos_mask)
            neg_dist = dist_mat * neg_mask + 1e9 * (1 - neg_mask)
        
        hardest_pos, _ = pos_dist.max(dim=1)  # [N]
        hardest_neg, _ = neg_dist.min(dim=1)  # [N]
        
        # Only compute loss for anchors that have at least one positive in memory
        has_positive = (pos_mask.sum(dim=1) > 0)
        if has_positive.sum() == 0:
            self.enqueue(embeddings, labels, cameras)
            return torch.tensor(0.0, device=embeddings.device)
        
        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        loss = F.relu(hardest_pos[has_positive] - hardest_neg[has_positive] + self.margin)
        
        # Enqueue AFTER mining to avoid self-matching
        self.enqueue(embeddings, labels, cameras)
        return loss.mean()


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss with online hard mining.
    For each anchor, selects hardest positive (farthest same-ID) 
    and hardest negative (closest different-ID).
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [N, D] L2-normalized embeddings
            labels: [N] identity labels
        Returns:
            triplet loss scalar
        """
        if len(embeddings) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Pairwise cosine similarity (since embeddings are L2-normalized, dot = cosine)
        # Convert to distance: dist = 1 - cos_sim (range 0-2)
        cos_sim = torch.mm(embeddings, embeddings.t())  # [N, N]
        dist_mat = 1 - cos_sim  # Cosine distance
        
        # Create masks for positive pairs (same ID) and negative pairs (different ID)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float()  # [N, N]
        neg_mask = (labels != labels.t()).float()  # [N, N]
        
        # Remove diagonal from positive mask
        pos_mask = pos_mask - torch.eye(len(labels), device=embeddings.device)
        
        # For each anchor, find hardest positive (max distance among positives)
        # Mask out non-positives with -inf
        pos_dist = dist_mat * pos_mask - 1e9 * (1 - pos_mask)
        hardest_pos_dist, _ = pos_dist.max(dim=1)  # [N]
        
        # For each anchor, find hardest negative (min distance among negatives)
        # Mask out non-negatives with +inf
        neg_dist = dist_mat * neg_mask + 1e9 * (1 - neg_mask)
        hardest_neg_dist, _ = neg_dist.min(dim=1)  # [N]
        
        # Only consider anchors that have at least one positive
        valid_mask = (pos_mask.sum(dim=1) > 0)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        triplet_loss = F.relu(hardest_pos_dist[valid_mask] - hardest_neg_dist[valid_mask] + self.margin)
        
        return triplet_loss.mean()


class CircleLoss(nn.Module):
    """
    Circle Loss (Sun et al., CVPR 2020) — unified pair similarity optimization.

    Self-paced weighting: each pair's gradient is scaled by how far it is from
    the decision boundary. Hard positives (low sim) and hard negatives (high sim)
    get amplified; easy/converged pairs contribute less. This emerges from the
    circular decision boundary geometry: L = log[1 + sum_n exp(gamma*an*(sn-dn))
    * sum_p exp(-gamma*ap*(sp-dp))].

    Fully vectorized via masked logsumexp — no Python loops over anchors.

    Args:
        m: Relaxation margin (default: 0.25, matching ArcFace margin for 196-ID P-DESTRE)
        gamma: Scale factor (default: 128, standard for ReID similarity learning)
    """
    def __init__(self, m=0.25, gamma=128):
        super().__init__()
        self.m = m
        self.gamma = gamma

    def compute_from_similarity(self, sim_mat, pos_mask, neg_mask,
                                 cross_cam_mask=None, cross_cam_boost=1.0):
        """
        Core vectorized circle loss from pre-computed similarity matrix.

        Args:
            sim_mat: [N, M] cosine similarities (anchors × candidates)
            pos_mask: [N, M] bool — True for same-identity pairs
            neg_mask: [N, M] bool — True for different-identity pairs
            cross_cam_mask: [N, M] bool — True where anchor and candidate cameras differ
            cross_cam_boost: float — multiplier for informative camera-aware pairs
        Returns:
            scalar loss (0.0 if no valid anchors)
        """
        delta_p = 1 - self.m  # optimal positive similarity target
        delta_n = self.m      # optimal negative similarity target

        # Self-paced weights: detach to prevent gradient through weighting
        # alpha_p increases as sp drops below (1+m) — hard positives get amplified
        # alpha_n increases as sn rises above (-m) — hard negatives get amplified
        alpha_p = torch.clamp_min(1 + self.m - sim_mat.detach(), min=0.)  # [N, M]
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.)      # [N, M]

        # Camera-aware boosting: amplify the most informative pairs
        # Cross-camera positives (same person, different camera) → hardest, most useful
        # Same-camera negatives (different person, same camera) → hardest confusers
        if cross_cam_mask is not None and cross_cam_boost > 1.0:
            alpha_p = alpha_p * torch.where(cross_cam_mask & pos_mask, cross_cam_boost, 1.0)
            alpha_n = alpha_n * torch.where((~cross_cam_mask) & neg_mask, cross_cam_boost, 1.0)

        # Scaled logits: push positives toward delta_p=0.75, negatives below delta_n=0.25
        logit_p = -self.gamma * alpha_p * (sim_mat - delta_p)  # [N, M]
        logit_n =  self.gamma * alpha_n * (sim_mat - delta_n)  # [N, M]

        # Mask invalid pairs with -inf (logsumexp ignores exp(-inf) = 0)
        logit_p = logit_p.masked_fill(~pos_mask, float('-inf'))  # [N, M]
        logit_n = logit_n.masked_fill(~neg_mask, float('-inf'))  # [N, M]

        # Per-anchor logsumexp aggregation (numerically stable)
        lse_p = torch.logsumexp(logit_p, dim=1)  # [N]
        lse_n = torch.logsumexp(logit_n, dim=1)  # [N]

        # Only average over anchors with both positive and negative pairs
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)  # [N]
        if not valid.any():
            return torch.tensor(0.0, device=sim_mat.device)

        loss = F.softplus(lse_p[valid] + lse_n[valid])
        return loss.mean()

    def forward(self, embeddings, labels):
        """
        Batch-level circle loss from L2-normalized embeddings.

        Args:
            embeddings: [N, D] L2-normalized embeddings
            labels: [N] identity labels
        Returns:
            scalar loss
        """
        if len(embeddings) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        sim_mat = torch.mm(embeddings, embeddings.t())  # [N, N]
        labels_col = labels.view(-1, 1)
        pos_mask = (labels_col == labels_col.t())
        neg_mask = ~pos_mask
        # Remove self-pairs from positive mask (diagonal similarity = 1.0, trivially easy)
        pos_mask = pos_mask & ~torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)

        return self.compute_from_similarity(sim_mat, pos_mask, neg_mask)


class RelationalDistillLoss(nn.Module):
    """
    Weighted Relational Knowledge Distillation for ReID.

    Aligns pairwise cosine similarity structure between teacher (OSNet, 512-d)
    and student (our model, 256-d) — dimension-agnostic via structural matching.

    Key improvements over naive MSE(S_s, S_t):
      1. Diagonal masked — self-similarity is trivially 1.0, wasted gradient
      2. Same-ID pairs upweighted — these are the hard positives we care about
      3. Cross-camera same-ID pairs get highest weight — the exact failure mode
      4. Hard negatives (top-k confusing different-ID) upweighted
      5. Instance-aligned: matched by GT instance index, not person ID
    """
    def __init__(self, temperature: float = 1.0, pos_weight: float = 5.0,
                 cross_cam_weight: float = 10.0, hard_neg_k: int = 3):
        super().__init__()
        self.temperature = temperature
        self.pos_weight = pos_weight           # Weight for same-ID pairs
        self.cross_cam_weight = cross_cam_weight  # Extra weight for cross-camera same-ID
        self.hard_neg_k = hard_neg_k           # Top-k hardest negatives per anchor to upweight

    def forward(self, student_embs: torch.Tensor, teacher_embs: torch.Tensor,
                labels: torch.Tensor = None, cameras: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            student_embs: [N, D_s] L2-normalized student embeddings
            teacher_embs: [N, D_t] L2-normalized teacher embeddings
            labels: [N] person IDs (for weighting same-ID pairs). Optional.
            cameras: [N] camera IDs (for cross-camera weighting). Optional.

        Returns:
            Scalar weighted relational distillation loss
        """
        N = len(student_embs)
        if N < 2:
            return torch.tensor(0.0, device=student_embs.device)

        device = student_embs.device

        # Pairwise cosine similarity (already L2-normalized)
        S_s = torch.mm(student_embs, student_embs.t())  # [N, N]
        S_t = torch.mm(teacher_embs, teacher_embs.t())  # [N, N]

        # Temperature scaling
        if self.temperature != 1.0:
            S_s = S_s / self.temperature
            S_t = S_t / self.temperature

        # Per-element squared error
        sq_err = (S_s - S_t) ** 2  # [N, N]

        # Build weight matrix
        W = torch.ones(N, N, device=device)

        # Mask diagonal (self-similarity is trivially 1.0)
        diag_mask = torch.eye(N, device=device, dtype=torch.bool)
        W[diag_mask] = 0.0

        if labels is not None and len(labels) == N:
            # Same-ID pairs: upweight
            same_id = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
            same_id[diag_mask] = False  # exclude self
            W[same_id] = self.pos_weight

            # Cross-camera same-ID: highest priority
            if cameras is not None and len(cameras) == N:
                diff_cam = cameras.unsqueeze(0) != cameras.unsqueeze(1)  # [N, N]
                cross_cam_pos = same_id & diff_cam
                W[cross_cam_pos] = self.cross_cam_weight

            # Hard negatives: for each anchor, upweight top-k most confusing different-ID pairs
            # (highest teacher similarity among different-ID = most confusable pairs)
            diff_id = ~(labels.unsqueeze(0) == labels.unsqueeze(1))
            diff_id[diag_mask] = False
            if self.hard_neg_k > 0 and diff_id.any():
                # Use teacher similarity to find hard negatives
                S_t_neg = S_t.clone()
                S_t_neg[~diff_id] = -2.0  # mask non-negatives
                # Top-k hardest negatives per row
                k = min(self.hard_neg_k, diff_id.sum(dim=1).min().item())
                if k > 0:
                    _, hard_idx = S_t_neg.topk(k, dim=1)
                    # Upweight these pairs (3× base weight)
                    for row in range(N):
                        W[row, hard_idx[row]] = 3.0

        # Weighted MSE (exclude diagonal via W=0)
        num_pairs = (W > 0).sum().clamp(min=1).float()
        loss = (W * sq_err).sum() / num_pairs

        return loss


class FeatureDistillLoss(nn.Module):
    """
    Feature-level Knowledge Distillation for ReID embeddings.

    Projects student embeddings (256-d) to teacher embedding space (512-d)
    via a learned linear projector, then minimizes cosine distance between
    projected student and teacher embeddings (both L2-normalized).

    Unlike RelationalDistillLoss (which saturates once pairwise structure aligns),
    this provides a persistent per-instance training signal that encourages the
    student to produce semantically similar features to the teacher.

    Key design choices:
      - Linear projector (no nonlinearity) to preserve embedding geometry
      - Cosine distance loss: scale-invariant, properly scaled for normalized vectors
        (smooth_l1/MSE on L2-normalized high-dim vectors is ~1/D, disappears)
      - Operates on pre-BN (metric space) embeddings for both student and teacher
    """
    def __init__(self, student_dim: int = 256, teacher_dim: int = 512):
        super().__init__()
        self.projector = nn.Linear(student_dim, teacher_dim, bias=False)
        # Xavier init for stable gradients at start
        nn.init.xavier_uniform_(self.projector.weight)

    def forward(self, student_embs: torch.Tensor, teacher_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_embs: [N, D_s] L2-normalized student embeddings (pre-BN)
            teacher_embs: [N, D_t] L2-normalized teacher embeddings

        Returns:
            Scalar feature-level distillation loss (cosine distance, range [0, 2])
        """
        if len(student_embs) < 1:
            return torch.tensor(0.0, device=student_embs.device)

        # Project student → teacher space
        projected = self.projector(student_embs)  # [N, D_t]
        # L2-normalize projected embeddings (teacher is already normalized)
        projected = F.normalize(projected, p=2, dim=1, eps=1e-6)

        # Cosine distance: 1 - cos_sim (range [0, 2], properly scaled)
        cos_sim = F.cosine_similarity(projected, teacher_embs.detach(), dim=1)  # [N]
        loss = (1.0 - cos_sim).mean()

        return loss


# ═══════════════════════════════════════════════════════════════
#  BAU UNIFORMITY LOSSES — Hypersphere Spreading Regularizers
#  From: Cho et al., "Balancing Alignment and Uniformity" (NeurIPS 2024)
# ═══════════════════════════════════════════════════════════════

def uniformity_loss(embeddings, t=2.0):
    """
    Uniformity loss: encourages embeddings to spread uniformly on the unit hypersphere.

    L_uniform = log( mean( exp(-t * ||x_i - x_j||^2) ) )   for all i≠j

    Minimizing this pushes all pairwise distances apart, preventing feature collapse.
    Returns ~0 when features are maximally spread, large negative when well-spread,
    approaches 0 from below when collapsed.

    Args:
        embeddings: [N, D] L2-normalized embeddings (N >= 2)
        t: temperature (default 2.0, from Wang & Isola "Understanding Contrastive Representation
           Learning through Alignment and Uniformity on the Hypersphere", ICML 2020)
    Returns:
        Scalar uniformity loss (lower = more uniform spread)
    """
    return torch.pdist(embeddings, p=2).pow(2).mul(-t).exp().mean().log()


def camera_uniformity_loss(embeddings, cam_ids, t=2.0, min_per_cam=2):
    """
    Per-camera uniformity loss: spreads features within each camera domain.

    Prevents camera-specific feature collapse where all embeddings from one
    camera collapse to a cluster, which is the core failure mode with only
    374 cross-camera training IDs.

    Args:
        embeddings: [N, D] L2-normalized embeddings
        cam_ids: [N] camera ID per embedding
        t: temperature (default 2.0)
        min_per_cam: minimum samples per camera to compute loss (default 2)
    Returns:
        Scalar per-camera uniformity loss (average over cameras with enough samples)
    """
    unique_cams = cam_ids.unique()
    total_loss = torch.tensor(0.0, device=embeddings.device)
    n_active = 0
    for cam in unique_cams:
        cam_mask = cam_ids == cam
        if cam_mask.sum() < min_per_cam:
            continue
        cam_embs = embeddings[cam_mask]
        total_loss = total_loss + uniformity_loss(cam_embs, t=t)
        n_active += 1
    if n_active > 0:
        return total_loss / n_active
    return total_loss


# ═══════════════════════════════════════════════════════════════
#  GEOMETRY UTILITIES — GIoU, Gaussian radius, Gaussian draw
# ═══════════════════════════════════════════════════════════════

def bbox_giou(box1, box2, eps=1e-7):
    """Calculate Generalized Intersection over Union (GIoU)."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    w1, h1 = (b1_x2 - b1_x1).clamp(0), (b1_y2 - b1_y1).clamp(0)
    w2, h2 = (b2_x2 - b2_x1).clamp(0), (b2_y2 - b2_y1).clamp(0)
    union_area = w1 * h1 + w2 * h2 - inter_area + eps

    iou = inter_area / union_area

    cw = (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)).clamp(min=eps)
    ch = (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)).clamp(min=eps)
    c_area = cw * ch + eps

    giou = iou - (c_area - union_area) / c_area
    return giou


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    if height <= 0 or width <= 0: return 0
    if min_overlap <= 0 or min_overlap >= 1: min_overlap = 0.7

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0, b1 ** 2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0, b2 ** 2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    if a3 > 0:
        sq3 = math.sqrt(max(0, b3 ** 2 - 4 * a3 * c3))
        r3 = (b3 + sq3) / (2 * a3)
    else:
        r3 = float('inf')
    return max(0, min(r1, r2, r3))


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = torch.zeros(diameter, diameter, device=heatmap.device, dtype=heatmap.dtype)
    sigma = diameter / 6.0
    x = torch.arange(0, diameter, device=heatmap.device, dtype=heatmap.dtype)
    y = torch.arange(0, diameter, device=heatmap.device, dtype=heatmap.dtype)
    y = y.unsqueeze(1)
    x0, y0 = diameter // 2, diameter // 2
    gaussian = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    height, width = heatmap.shape
    x, y = int(center[0]), int(center[1])
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    if left + right <= 0 or top + bottom <= 0: return

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if masked_heatmap.shape[0] > 0 and masked_heatmap.shape[1] > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


# ═══════════════════════════════════════════════════════════════
#  TARGET GENERATION — CenterNet heatmap, TRBL, offset, visibility
# ═══════════════════════════════════════════════════════════════

def generate_heatmap_only(targets, H, W, num_classes=1, device='cuda', batch_size=1):
    """
    Lightweight heatmap-only target generation for split-stride P2 heatmap.

    Only renders Gaussian peaks — no WH, offset, ID, visibility, or
    multi-positive loops. ~10× faster than generate_centernet_targets
    for stride-4 (152×272) since it skips all regression target allocation
    and the O(N²) visibility computation.
    """
    heatmap = torch.zeros(batch_size, num_classes, H, W, device=device)
    if targets.shape[0] == 0:
        return heatmap

    for i in range(targets.shape[0]):
        t = targets[i]
        b = int(t[0].item())
        cx = t[3].item() * W
        cy = t[4].item() * H
        w = t[5].item() * W
        h = t[6].item() * H
        if w <= 0 or h <= 0:
            continue
        cx_int, cy_int = int(cx), int(cy)
        if cx_int < 0 or cx_int >= W or cy_int < 0 or cy_int >= H:
            continue
        radius = max(2, int(gaussian_radius((h, w), min_overlap=0.7)))
        radius = min(radius, min(H, W) // 2)
        draw_gaussian(heatmap[b, 0], (cx, cy), radius)

    return heatmap

def generate_centernet_targets(targets, H, W, num_classes=1, device='cuda', batch_size=1,
                               mp_thresh=0.3):
    """
    Generate CenterNet targets with IoU-overlap-based visibility proxy.

    Visibility is computed from pairwise GT box IoU: for each GT box, visibility =
    1.0 - (fraction of box area covered by other GT boxes). This replaces the
    crude 1/object_count proxy with a geometrically accurate occlusion estimate.
    """
    heatmap = torch.zeros(batch_size, num_classes, H, W, device=device)
    wh_target = torch.zeros(batch_size, 4, H, W, device=device)  # TRBL format
    offset_target = torch.zeros(batch_size, 2, H, W, device=device)
    reg_mask = torch.zeros(batch_size, H, W, device=device)
    id_target = torch.full((batch_size, H, W), -1, device=device, dtype=torch.long)  # -1 = no ID (multi-positive safe)
    visibility_target = torch.ones(batch_size, 1, H, W, device=device)

    if targets.shape[0] == 0:
        return heatmap, wh_target, offset_target, reg_mask, id_target, visibility_target

    # Check if ANY target has explicit visibility annotation
    has_explicit_vis = targets.shape[1] > 7
    has_conf = targets.shape[1] > 8

    # ---- Pre-compute IoU-based visibility proxy per batch ----
    # Group targets by batch index for pairwise IoU computation
    if not has_explicit_vis:
        per_batch_boxes = {}
        per_batch_indices = {}
        for i in range(targets.shape[0]):
            b = int(targets[i, 0].item())
            cx = targets[i, 3].item() * W
            cy = targets[i, 4].item() * H
            w = targets[i, 5].item() * W
            h = targets[i, 6].item() * H
            if w <= 0 or h <= 0:
                continue
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            if b not in per_batch_boxes:
                per_batch_boxes[b] = []
                per_batch_indices[b] = []
            per_batch_boxes[b].append([x1, y1, x2, y2])
            per_batch_indices[b].append(i)

        # Compute visibility for each target via pairwise box overlap (vectorized)
        iou_vis = {}  # target_index -> visibility
        for b, boxes in per_batch_boxes.items():
            if len(boxes) <= 1:
                for idx in per_batch_indices[b]:
                    iou_vis[idx] = 1.0
                continue
            boxes_arr = torch.tensor(boxes, device=device, dtype=torch.float32)
            n = len(boxes_arr)
            areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])

            # Vectorized pairwise intersection
            inter_x1 = torch.max(boxes_arr[:, 0].unsqueeze(1), boxes_arr[:, 0].unsqueeze(0))
            inter_y1 = torch.max(boxes_arr[:, 1].unsqueeze(1), boxes_arr[:, 1].unsqueeze(0))
            inter_x2 = torch.min(boxes_arr[:, 2].unsqueeze(1), boxes_arr[:, 2].unsqueeze(0))
            inter_y2 = torch.min(boxes_arr[:, 3].unsqueeze(1), boxes_arr[:, 3].unsqueeze(0))
            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_areas = inter_w * inter_h  # [n, n]

            # Zero out self-intersection (diagonal)
            inter_areas.fill_diagonal_(0)

            # Sum of all pairwise intersections per target
            total_occluded = inter_areas.sum(dim=1)  # [n]
            occluded_fractions = (total_occluded / (areas + 1e-6)).clamp(max=0.9)

            for i in range(n):
                iou_vis[per_batch_indices[b][i]] = 1.0 - occluded_fractions[i].item()

    # ---- Generate targets ----
    for i in range(targets.shape[0]):
        t = targets[i]
        b = int(t[0].item())
        cls = 0
        obj_id = int(t[2].item())
        cx = t[3].item() * W
        cy = t[4].item() * H
        w = t[5].item() * W
        h = t[6].item() * H

        if w <= 0 or h <= 0: continue
        cx_int, cy_int = int(cx), int(cy)
        if cx_int < 0 or cx_int >= W or cy_int < 0 or cy_int >= H: continue

        radius = max(2, int(gaussian_radius((h, w), min_overlap=0.7)))
        radius = min(radius, min(H, W) // 2)
        draw_gaussian(heatmap[b, cls], (cx, cy), radius)
        
        # Conf/Vis parsing
        conf = 1.0
        vis = 1.0
        if has_conf:
             conf = t[7].item()
             vis = t[8].item()
        elif has_explicit_vis:
             vis = t[7].item()
        else:
             vis = iou_vis.get(i, 1.0)

        # ── Multi-Positive Assignment ──
        # Assign regression targets to ALL pixels within the Gaussian radius
        # where the Gaussian value > threshold. Each pixel predicts the SAME
        # box (LTRB = half-widths, center-relative) and offset points to the
        # true box center. This is CenterNet-compatible (not FCOS-style).
        # Weight contribution by Gaussian center-ness (center=1.0, edge=thresh).
        # mp_thresh: configurable via loss.mp_thresh in YAML (default 0.3).
        # Lower values (e.g. 0.15) assign more reg-positive pixels per object,
        # improving recall for crowded/occluded pedestrians at the cost of
        # slightly noisier regression from peripheral pixels.
        sigma = (2 * radius + 1) / 6.0

        y_lo = max(0, cy_int - radius)
        y_hi = min(H - 1, cy_int + radius)
        x_lo = max(0, cx_int - radius)
        x_hi = min(W - 1, cx_int + radius)

        # LTRB from center (same at all pixels — CenterNet paradigm)
        half_w = w / 2
        half_h = h / 2

        for py in range(y_lo, y_hi + 1):
            for px in range(x_lo, x_hi + 1):
                # Compute Gaussian center-ness at this pixel
                g_val = math.exp(-((px - cx) ** 2 + (py - cy) ** 2) / (2 * sigma ** 2))
                if g_val < mp_thresh:
                    continue

                # Only overwrite if this object has higher center-ness than existing
                existing_weight = reg_mask[b, py, px].item()
                new_weight = conf * g_val
                if new_weight > existing_weight:
                    wh_target[b, 0, py, px] = half_w  # left (from center)
                    wh_target[b, 1, py, px] = half_h  # top (from center)
                    wh_target[b, 2, py, px] = half_w  # right (from center)
                    wh_target[b, 3, py, px] = half_h  # bottom (from center)
                    reg_mask[b, py, px] = new_weight
                    # Offset: sub-pixel correction from THIS pixel to box center
                    offset_target[b, 0, py, px] = cx - px
                    offset_target[b, 1, py, px] = cy - py
                    # Visibility applies to all assigned pixels
                    if vis >= 0:
                        visibility_target[b, 0, py, px] = vis

        # ID target: center pixel only (ArcFace needs unique location per object)
        # Only overwrite if this object has higher center-ness than the existing
        # one at this pixel (prevents box/ID desync when two GT share a center)
        # NOTE: Must use conf * g_val (same scale as reg_mask) for fair comparison.
        # The integer center (cx_int, cy_int) has g_val < 1.0 when the float
        # center (cx, cy) has a fractional component — using bare `conf` would
        # make the condition always-true, defeating the protection.
        center_g = math.exp(-((cx_int - cx) ** 2 + (cy_int - cy) ** 2) / (2 * sigma ** 2))
        center_weight = conf * center_g
        existing_id_weight = reg_mask[b, cy_int, cx_int].item()
        if obj_id >= 0 and center_weight >= existing_id_weight:
            id_target[b, cy_int, cx_int] = obj_id

    return heatmap, wh_target, offset_target, reg_mask, id_target, visibility_target


# ═══════════════════════════════════════════════════════════════
#  DETECTION LOSSES — Focal Loss & Repulsion Loss
# ═══════════════════════════════════════════════════════════════

def modified_focal_loss(pred, target, alpha=2, beta=4, pos_weight=None):
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    pred = pred.clamp(min=1e-6, max=1 - 1e-6)
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    if pos_weight is not None:
        pos_loss = pos_loss * pos_weight
    neg_weight = torch.pow(1 - target, beta)
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg_mask
    num_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def repulsion_loss(wh_pred, wh_target, offset_pred, offset_target, reg_mask,
                   heatmap_target, targets=None, sigma=0.5):
    """
    Repulsion Loss: pushes predicted boxes away from neighboring GT boxes.

    For each GT center, finds the nearest other GT center and adds a smooth-L1
    penalty that discourages the predicted box from overlapping with neighbors.
    This prevents merged detections in dense crowds.

    Uses a zero-margin GT-GT IoU baseline:
      rep_target = max(0, gt_gt_iou)
      excess     = max(0, pred_iou - rep_target)
      loss       = smooth_l1(excess, 0)

    At P=GT, loss=0 and gradient=0 — predictions can perfectly match GT
    without penalty. Only penalizes overlap that EXCEEDS the natural GT-GT
    overlap level. Previous margin=0.1 caused ~1-3% box shrinkage in crowds.

    Only active when multiple GT objects exist in a batch element.

    Args:
        wh_pred: [B, 4, H, W] predicted box distances (TRBL distances)
        wh_target: [B, 4, H, W] target box distances (TRBL distances)
        offset_pred: [B, 2, H, W] predicted offset
        offset_target: [B, 2, H, W] target offset
        reg_mask: [B, H, W] mask of GT center locations
        heatmap_target: [B, C, H, W] Gaussian heatmap targets
        sigma: smooth L1 transition point

    Returns:
        Scalar repulsion loss
    """
    B, _, H, W = wh_pred.shape
    total_loss = torch.tensor(0.0, device=wh_pred.device)
    count = 0

    for b in range(B):
        # Find ALL GT objects for this batch element.
        # Using raw targets (when available) handles the edge case where two GT
        # objects share the same heatmap center pixel — heatmap peaks would only
        # find one, but raw targets list every object explicitly.
        if targets is not None:
            b_mask = targets[:, 0] == b
            b_targets = targets[b_mask]
            if len(b_targets) < 2:
                continue
            # Raw target format: [batch_idx, cls, id, cx, cy, w, h, ...]
            # cx, cy, w, h are in [0,1] — convert to feature-map coords
            gt_cx = b_targets[:, 3] * W
            gt_cy = b_targets[:, 4] * H
            gt_hw = b_targets[:, 5] * W / 2  # half-width
            gt_hh = b_targets[:, 6] * H / 2  # half-height
            gt_l = gt_hw
            gt_t = gt_hh
            gt_r = gt_hw
            gt_b = gt_hh
            # Get predicted values at the integer center locations
            ys = gt_cy.long().clamp(0, H - 1)
            xs = gt_cx.long().clamp(0, W - 1)
            n = len(b_targets)
        else:
            # Fallback: use heatmap peaks (may miss overlapping-center objects)
            hm_max = heatmap_target[b, 0]
            pos_indices = torch.nonzero(hm_max >= 0.9999, as_tuple=False)
            n = pos_indices.shape[0]
            if n < 2:
                continue
            ys = pos_indices[:, 0]
            xs = pos_indices[:, 1]
            has_reg = reg_mask[b][ys, xs] > 0
            if has_reg.sum() < 2:
                continue
            ys = ys[has_reg]
            xs = xs[has_reg]
            n = len(ys)
            gt_cx = xs.float() + offset_target[b, 0, ys, xs]
            gt_cy = ys.float() + offset_target[b, 1, ys, xs]
            gt_l = wh_target[b, 0, ys, xs]
            gt_t = wh_target[b, 1, ys, xs]
            gt_r = wh_target[b, 2, ys, xs]
            gt_b = wh_target[b, 3, ys, xs]

        # TRBL decoding for Prediction
        pred_l = wh_pred[b, 0, ys, xs]
        pred_t = wh_pred[b, 1, ys, xs]
        pred_r = wh_pred[b, 2, ys, xs]
        pred_b = wh_pred[b, 3, ys, xs]
        pred_cx = xs.float() + offset_pred[b, 0, ys, xs]
        pred_cy = ys.float() + offset_pred[b, 1, ys, xs]

        # Decode predicted boxes: [N, 4] as x1, y1, x2, y2
        pred_x1 = pred_cx - pred_l
        pred_y1 = pred_cy - pred_t
        pred_x2 = pred_cx + pred_r
        pred_y2 = pred_cy + pred_b

        # Decode GT neighbor boxes
        gt_x1 = gt_cx - gt_l
        gt_y1 = gt_cy - gt_t
        gt_x2 = gt_cx + gt_r
        gt_y2 = gt_cy + gt_b

        # For each GT, find the nearest OTHER GT box (by center distance)
        centers = torch.stack([gt_cx, gt_cy], dim=1)  # [N, 2]
        dists = torch.cdist(centers, centers)  # [N, N]
        dists.fill_diagonal_(float('inf'))
        nearest_idx = dists.argmin(dim=1)  # [N]

        # Compute IoU between each predicted box and its nearest GT neighbor.
        # This penalizes predicted boxes that encroach on neighboring GT boxes.
        neighbor_x1 = gt_x1[nearest_idx]
        neighbor_y1 = gt_y1[nearest_idx]
        neighbor_x2 = gt_x2[nearest_idx]
        neighbor_y2 = gt_y2[nearest_idx]

        # --- Pred ↔ Neighbor IoU ---
        inter_x1 = torch.max(pred_x1, neighbor_x1)
        inter_y1 = torch.max(pred_y1, neighbor_y1)
        inter_x2 = torch.min(pred_x2, neighbor_x2)
        inter_y2 = torch.min(pred_y2, neighbor_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        pred_area = (pred_x2 - pred_x1).clamp(min=1e-4) * (pred_y2 - pred_y1).clamp(min=1e-4)
        neighbor_area = (neighbor_x2 - neighbor_x1).clamp(min=1e-4) * (neighbor_y2 - neighbor_y1).clamp(min=1e-4)

        iou_with_neighbor = inter_area / (pred_area + neighbor_area - inter_area + 1e-6)

        # --- GT ↔ Neighbor GT IoU (natural baseline overlap) ---
        # When GT boxes physically overlap in crowds, the zero-target approach
        # (smooth_l1(pred_iou, 0)) produces a non-zero gradient even at the
        # perfect prediction (pred=GT), causing systematic box shrinkage.
        #
        # Fix: compute the GT-GT IoU baseline and use a margin-shifted
        # one-sided penalty. Only penalizes when predicted overlap EXCEEDS
        # a threshold set slightly below the natural GT overlap level.
        #
        # rep_target = max(0, gt_gt_iou)  [zero margin]
        # excess     = max(0, pred_iou - rep_target)
        # loss       = smooth_l1(excess, 0)
        #
        # Properties:
        #   - One-sided: never penalizes boxes for being more separated (✓)
        #   - Zero gradient at P=GT: predictions can match GT without penalty
        #   - Only penalizes overlap exceeding natural GT-GT level
        #   - At pred=GT, gt_gt_iou=0.3: excess=0, loss=0 (no box shrinkage)
        gt_inter_x1 = torch.max(gt_x1, neighbor_x1)
        gt_inter_y1 = torch.max(gt_y1, neighbor_y1)
        gt_inter_x2 = torch.min(gt_x2, neighbor_x2)
        gt_inter_y2 = torch.min(gt_y2, neighbor_y2)

        gt_inter_w = (gt_inter_x2 - gt_inter_x1).clamp(min=0)
        gt_inter_h = (gt_inter_y2 - gt_inter_y1).clamp(min=0)
        gt_inter_area = gt_inter_w * gt_inter_h

        gt_area = (gt_x2 - gt_x1).clamp(min=1e-4) * (gt_y2 - gt_y1).clamp(min=1e-4)
        gt_gt_iou = gt_inter_area / (gt_area + neighbor_area - gt_inter_area + 1e-6)

        # Zero-margin: allow predictions to match GT overlap without penalty.
        # Only penalizes predicted overlap EXCEEDING the natural GT-GT overlap.
        # (Previous margin=0.1 caused ~1-3% systematic box shrinkage in crowds)
        rep_target = gt_gt_iou.detach().clamp(min=0)
        excess_iou = (iou_with_neighbor - rep_target).clamp(min=0)
        rep_loss = F.smooth_l1_loss(excess_iou,
                                     torch.zeros_like(excess_iou),
                                     beta=sigma, reduction='sum')
        total_loss = total_loss + rep_loss
        count += n

    if count > 0:
        return total_loss / count
    return total_loss


# ═══════════════════════════════════════════════════════════════
#  MAIN LOSS CLASS — CenterNetLoss
# ═══════════════════════════════════════════════════════════════

class CenterNetLoss(nn.Module):
    """
    EffiPed CenterNet loss with learnable uncertainty weighting.

    Loss Components:
      - Heatmap: Modified Focal Loss
      - WH: GIoU regression (LTRB) at multi-positive pixels
      - Offset: L1 regression at GT centers
      - ReID: ArcFace (s=16, m=0.25) + XBM-Triplet (BatchHard, cosine)
      - Repulsion: Penalizes predicted boxes overlapping with neighbor GT
      - IoU: BCE on predicted IoU quality vs center-aligned GT IoU

    Uncertainty Weighting (Kendall et al., CVPR 2018):
      Each loss task has a learnable log-variance parameter s_i.
      Effective weight = exp(-s_i), with regularizer 0.5 * s_i.
      This auto-balances tasks: if one loss dominates, its effective
      weight decreases. Parameters clamped to [-4, 4] for stability.
    """
    def __init__(self, num_classes=1, embedding_dim=128, num_identities=500, hm_weight=1.0,
                 wh_weight=0.1, off_weight=1.0, id_weight=1.0,
                 rep_weight=0.2, iou_weight=1.0,
                 use_visibility_weighted_loss=True,
                 reid_extraction='center',
                 part_extractor=None,
                 bnneck=None,
                 part_bnneck=None,
                 num_parts_v=4, num_parts_h=1,
                 roi_output_size=(32, 8),
                 part_loss_weight=0.5,
                 use_uncertainty_weighting=True,
                 arcface_s=16.0, arcface_m=0.25,
                 arcface_subcenter_k=1,
                 triplet_weight=0.3,
                 triplet_margin=0.3,
                 reid_dropout=0.0,
                 diversity_loss_weight=0.0,
                 label_smoothing=0.0,
                 reid_stride_ratio=1,
                 det_stride_ratio=1,
                 loss_cfg=None):
        super().__init__()
        loss_cfg = loss_cfg or {}
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.mp_thresh = loss_cfg.get('mp_thresh', 0.3)  # Multi-positive Gaussian threshold
        self.reid_stride_ratio = reid_stride_ratio  # 1 = stride-4 (default), 2 = stride-8
        self.det_stride_ratio = det_stride_ratio    # 1 = stride-4 (default), 2 = stride-8
        # Ratio to map det coordinates → embedding map coordinates.
        # det_stride=8, reid_stride=8 → 1.0 (same grid).
        # det_stride=8, reid_stride=4 → 2.0 (emb grid has 2× more pixels).
        # det_stride=4, reid_stride=8 → 0.5 (emb grid has 2× fewer pixels).
        self._det_to_emb = det_stride_ratio / reid_stride_ratio
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.id_weight = id_weight
        self.rep_weight = rep_weight
        self.iou_weight = iou_weight
        self.use_visibility_weighted_loss = use_visibility_weighted_loss
        self.reid_extraction = reid_extraction
        self.part_loss_weight = part_loss_weight
        self.part_classifier_mode = loss_cfg.get('part_classifier_mode', 'per_part')
        self.current_epoch = 0
        self.triplet_start_epoch = int(loss_cfg.get('triplet_start_epoch', 0))
        self.triplet_warmup_epochs = int(loss_cfg.get('triplet_warmup_epochs', 0))
        self.part_loss_start_epoch = int(loss_cfg.get('part_loss_start_epoch', 0))
        self.part_loss_warmup_epochs = int(loss_cfg.get('part_loss_warmup_epochs', 0))
        self.part_triplet_start_epoch = int(loss_cfg.get('part_triplet_start_epoch', 0))
        self.part_triplet_warmup_epochs = int(loss_cfg.get('part_triplet_warmup_epochs', 0))
        self.diversity_start_epoch = int(loss_cfg.get('diversity_start_epoch', 0))
        self.diversity_warmup_epochs = int(loss_cfg.get('diversity_warmup_epochs', 0))
        self.bau_start_epoch = int(loss_cfg.get('bau_start_epoch', 0))
        self.bau_warmup_epochs = int(loss_cfg.get('bau_warmup_epochs', 0))
        part_triplet_weight_cfg = loss_cfg.get('part_triplet_weight', None)
        if part_triplet_weight_cfg is None:
            # Legacy G/N behavior: part triplet rode on the main triplet weight
            # scaled by part_loss_weight via
            #   triplet_weight * (triplet_fused + part_loss_weight * triplet_parts)
            # New shared-mode configs pin part_triplet_weight explicitly.
            self.part_triplet_weight = float(triplet_weight * part_loss_weight)
        else:
            self.part_triplet_weight = float(part_triplet_weight_cfg)
        self._last_tiny_roi_rate = 0.0
        self._last_part_attention_entropy_raw = 0.0
        self._last_part_collapse_rate_raw = 0.0
        self._last_part_attention_entropy_fused = 0.0
        self._last_part_collapse_rate_fused = 0.0
        self._last_part_attention_entropy = 0.0  # legacy alias: raw
        self._last_part_collapse_rate = 0.0      # legacy alias: raw
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # ReID embedding dropout (prevents ArcFace overfitting to camera-specific features)
        self.reid_dropout = nn.Dropout(p=reid_dropout) if reid_dropout > 0 else None
        if reid_dropout > 0:
            print(f"  [enabled] ReID embedding dropout: p={reid_dropout}")

        # ArcFace label smoothing: prevents over-confident class boundaries
        # Softens one-hot targets: (1-ε)·one_hot + ε/K → better generalization
        self.label_smoothing = label_smoothing
        if label_smoothing > 0:
            print(f"  [enabled] Label smoothing: eps={label_smoothing}")
            
        # Part Diversity: Gram orthogonality loss between part embeddings
        # Penalizes cosine similarity between parts → forces distinct representations
        self.diversity_loss_weight = diversity_loss_weight
        if self.diversity_loss_weight > 0:
            print(f"  [enabled] Part Diversity (Gram Orth): weight={self.diversity_loss_weight}")

        # BAU Uniformity: hypersphere spreading regularizers (Cho et al., NeurIPS 2024)
        # Prevents feature collapse — critical with small ID pools (374 cross-cam IDs).
        # Global uniformity: spreads ALL embeddings on unit hypersphere.
        # Per-camera uniformity: spreads embeddings WITHIN each camera domain,
        # preventing camera-specific clustering that kills cross-cam matching.
        self.bau_uniformity_weight = loss_cfg.get('bau_uniformity_weight', 0.0)
        self.bau_camera_uniformity_weight = loss_cfg.get('bau_camera_uniformity_weight', 0.0)
        self.bau_temperature = loss_cfg.get('bau_temperature', 2.0)
        if self.bau_uniformity_weight > 0:
            print(f"  [enabled] BAU Uniformity: weight={self.bau_uniformity_weight}, t={self.bau_temperature}")
        if self.bau_camera_uniformity_weight > 0:
            print(f"  [enabled] BAU Camera Uniformity: weight={self.bau_camera_uniformity_weight}, t={self.bau_temperature}")

        # Knowledge Distillation: align pairwise similarity structure with OSNet teacher
        # Uses relational KD (Park et al., CVPR 2019) — no dimension projection needed
        self.distillation_weight = loss_cfg.get('distillation_weight', 0.0)
        self.feature_kd_weight = loss_cfg.get('feature_kd_weight', 0.0)
        self.kd_warmup_batches = loss_cfg.get('kd_warmup_batches', 500)
        self._kd_step = 0
        self._last_loss_distill = 0.0  # Exposed for CSV logging without changing 8-tuple API
        self._last_loss_feat_distill = 0.0  # Feature-level KD loss for CSV logging
        self.distill_loss = None
        self.feature_distill_loss = None
        if self.distillation_weight > 0:
            distill_temp = loss_cfg.get('distillation_temperature', 1.0)
            self.distill_loss = RelationalDistillLoss(temperature=distill_temp)
            # Feature-level KD: direct embedding alignment via learned projector
            # Provides persistent per-instance signal (unlike relational which saturates)
            teacher_dim = loss_cfg.get('teacher_embedding_dim', 512)
            self.feature_distill_loss = FeatureDistillLoss(
                student_dim=embedding_dim, teacher_dim=teacher_dim
            )
            self.feature_kd_weight = loss_cfg.get('feature_kd_weight', 1.0)  # override default 0.0
            print(f"  [enabled] Knowledge Distillation: weight={self.distillation_weight}, temp={distill_temp}, warmup={self.kd_warmup_batches} batches")
            print(f"  [enabled] Feature-level KD: proj {embedding_dim}→{teacher_dim}, weight={self.feature_kd_weight}")

        # BNNeck: BatchNorm barrier between metric (triplet) and classification (ArcFace) spaces
        # Pre-BN embedding → triplet loss (preserves camera-specific info for metric learning)
        # Post-BN embedding → ArcFace (normalizes distribution, removes camera bias)
        # From: Luo et al., "Bag of Tricks for Deep Person ReID" (CVPRW 2019)
        self.use_bnneck = loss_cfg.get('use_bnneck', True)
        if self.use_bnneck:
            if bnneck is not None:
                object.__setattr__(self, 'bnneck', bnneck)
                print("  [enabled] BNNeck: model-owned pre-BN -> classifier")
            else:
                self.bnneck = nn.BatchNorm1d(embedding_dim, affine=True)
                nn.init.ones_(self.bnneck.weight)
                nn.init.zeros_(self.bnneck.bias)
                print("  [enabled] BNNeck: criterion-owned fallback")
        else:
            self.bnneck = None
            print("  [disabled] BNNeck (FairMOT-like baseline)")

        # Learnable uncertainty parameters (Kendall et al.)
        # FairMOT-style 2-param: s_det (all detection losses), s_id (ReID loss)
        # Init: s_det=-1.85, s_id=-1.05 (FairMOT's proven values: det_w≈6.4, id_w≈2.9)
        self.uncertainty_mode = 'none'
        if use_uncertainty_weighting:
            self.uncertainty_mode = '2param'
            self.s_det = nn.Parameter(torch.tensor([-1.85]))  # FairMOT init: w_det ≈ 6.4
            self.s_id_unc = nn.Parameter(torch.tensor([-1.05]))  # FairMOT init: w_id ≈ 2.9
            print("  [enabled] Uncertainty weighting: 2-param (FairMOT-style, s_det=-1.85, s_id=-1.05)")

        # ID Loss configuration — ArcFace or CosineClassifier (BOT/TransReID-style CE)
        # use_arcface=True (default): angular margin loss — tighter clusters
        # use_arcface=False: standard cosine classifier + label-smooth CE — proven in all top ReID models
        self.arcface_loss = None
        self.use_arcface = loss_cfg.get('use_arcface', True)

        if id_weight > 0 and num_identities > 1:
            if self.use_arcface:
                self.arcface_loss = ArcFaceLoss(
                    embedding_dim=embedding_dim,
                    num_classes=num_identities,
                    s=arcface_s,
                    m=arcface_m,
                    easy_margin=False,
                    subcenter_k=arcface_subcenter_k,
                )
                if arcface_subcenter_k > 1:
                    print(f"  [enabled] Sub-center ArcFace: K={arcface_subcenter_k} sub-centers per class")
                print(f"  ArcFace Loss (s={arcface_s}, m={arcface_m}) + Triplet Loss (margin={triplet_margin}, weight={triplet_weight})")
            else:
                self.arcface_loss = CosineClassifier(
                    embedding_dim=embedding_dim,
                    num_classes=num_identities,
                    s=arcface_s,  # reuse scale for cosine temperature
                )
                print(f"  Cosine Classifier (s={arcface_s}, no margin) + Label-Smooth CE (eps={label_smoothing}) + Triplet (margin={triplet_margin}, weight={triplet_weight})")
            # Add triplet/circle loss for metric learning (complementary to classification)
            # Circle Loss (Sun et al., CVPR 2020) replaces batch-hard triplet when enabled:
            #   - Uses ALL pairs with self-paced weighting (not just hardest pos/neg)
            #   - Better convergence on small datasets (196 IDs in P-DESTRE)
            #   - Integrates seamlessly with XBM for large-pool pair mining
            use_circle_loss = loss_cfg.get('use_circle_loss', False)
            circle_m = loss_cfg.get('circle_margin', 0.25)
            circle_gamma = loss_cfg.get('circle_gamma', 128)
            self.use_circle_loss = use_circle_loss
            
            if use_circle_loss:
                self.metric_loss = CircleLoss(m=circle_m, gamma=circle_gamma)
                print(f"  Circle Loss (m={circle_m}, gamma={circle_gamma}, weight={triplet_weight})")
            else:
                self.metric_loss = BatchHardTripletLoss(margin=triplet_margin)
            # Keep triplet_loss as alias for backward compatibility with per-part code
            self.triplet_loss = self.metric_loss
            self.triplet_weight = triplet_weight
            
            # Cross-Batch Memory: FIFO queue for pair mining at small batch sizes
            # At BS=2 with ~3773 IDs, batch-level mining is near-useless (93%
            # of batches have zero valid triplets/pairs). XBM stores embeddings from last
            # K batches so we mine from ~8192 candidates instead of ~24.
            # With Circle Loss: computes self-paced loss over all memory pairs.
            # With Triplet: batch-hard mining against memory entries.
            # Disabled when triplet_weight=0 or explicitly via config.
            xbm_memory_size = loss_cfg.get('xbm_memory_size', 4096)
            xbm_start_after = int(loss_cfg.get('xbm_start_after', 200))
            xbm_camera_aware = loss_cfg.get('xbm_camera_aware', False)
            xbm_cross_camera_weight = loss_cfg.get('xbm_cross_camera_weight', 3.0)
            self.xbm = CrossBatchMemory(
                embedding_dim=embedding_dim,
                memory_size=xbm_memory_size,
                margin=triplet_margin,
                start_after=xbm_start_after,
                camera_aware=xbm_camera_aware,
                cross_camera_weight=xbm_cross_camera_weight,
                use_circle_loss=use_circle_loss,
                circle_m=circle_m,
                circle_gamma=circle_gamma,
            ) if triplet_weight > 0 else None
            if self.xbm is not None:
                cam_str = f", camera_aware={xbm_camera_aware}" if xbm_camera_aware else ""
                loss_type = "circle" if use_circle_loss else "triplet"
                print(f"  [enabled] Cross-Batch Memory: size={xbm_memory_size}, loss={loss_type}, start_after={xbm_start_after}{cam_str}")

            # ---- Part-based ReID infrastructure ----
            self.num_parts = num_parts_v * num_parts_h
            if reid_extraction == 'part_based':
                # Use the model's PartBasedExtractor (shared instance — avoids dual instantiation)
                # The part_extractor lives in the model (head.part_extractor) and is passed here
                # so loss backprop updates the same weights used at inference.
                # Bypass nn.Module registration to avoid duplicating part_extractor
                # params in the optimizer — model.head already owns this submodule.
                # See train.py optimizer param groups for the ownership boundary.
                assert part_extractor is not None, (
                    "reid_extraction='part_based' requires part_extractor from model.head — "
                    "pass model.head.part_extractor to CenterNetLoss"
                )
                object.__setattr__(self, 'part_extractor', part_extractor)
                # Validate part count consistency between loss and extractor
                assert self.num_parts == part_extractor.num_parts, (
                    f"Part count mismatch: loss expects {self.num_parts} parts "
                    f"(num_parts_v={num_parts_v} × num_parts_h={num_parts_h}), "
                    f"but part_extractor has {part_extractor.num_parts} parts"
                )
                layout = f"{num_parts_v}×{num_parts_h}" if num_parts_h > 1 else f"{num_parts_v} strips"
                # Only instantiate per-part classifier heads when weight > 0
                # With weight=0, these are dead params + wasted forward passes/batch
                if part_loss_weight > 0:
                    if self.part_classifier_mode == 'shared':
                        if self.use_arcface:
                            self.part_arcface = ArcFaceLoss(
                                embedding_dim, num_identities, s=arcface_s, m=arcface_m,
                                easy_margin=False, subcenter_k=arcface_subcenter_k
                            )
                            cls_name = "shared part ArcFace"
                        else:
                            self.part_arcface = CosineClassifier(embedding_dim, num_identities, s=arcface_s)
                            cls_name = "shared part CosineClassifier"
                        print(f"  [enabled] Part-based ReID: {layout}, {cls_name} + BNNeck, part_loss_weight={part_loss_weight}, part_triplet_weight={self.part_triplet_weight}")
                    elif self.use_arcface:
                        self.part_arcface = nn.ModuleList([
                            ArcFaceLoss(embedding_dim, num_identities, s=arcface_s, m=arcface_m,
                                        easy_margin=False, subcenter_k=arcface_subcenter_k)
                            for _ in range(self.num_parts)
                        ])
                        print(f"  [enabled] Part-based ReID: {layout}, {self.num_parts} part ArcFace + BNNeck, part_loss_weight={part_loss_weight}, part_triplet_weight={self.part_triplet_weight}")
                    else:
                        self.part_arcface = nn.ModuleList([
                            CosineClassifier(embedding_dim, num_identities, s=arcface_s)
                            for _ in range(self.num_parts)
                        ])
                        print(f"  [enabled] Part-based ReID: {layout}, {self.num_parts} part CosineClassifier + BNNeck, part_loss_weight={part_loss_weight}, part_triplet_weight={self.part_triplet_weight}")
                else:
                    print(f"  [enabled] Part-based ReID: {layout}, part_loss_weight=0 (per-part classifier skipped, BNNeck kept for eval)")
                # Always create part BNNeck — needed at eval time for part embedding
                # normalization even when part_loss_weight=0 (training may have used
                # part_loss > 0 early and captured running stats we need at inference).
                if part_bnneck is not None:
                    object.__setattr__(self, 'part_bnneck', part_bnneck)
                else:
                    self.part_bnneck = nn.ModuleList([
                        nn.BatchNorm1d(embedding_dim, affine=True) for _ in range(self.num_parts)
                    ])
                    for bn in self.part_bnneck:
                        nn.init.ones_(bn.weight)
                        nn.init.zeros_(bn.bias)

    def migrate_shared_part_classifier_from_state(self, criterion_state: dict) -> int:
        """Initialize shared part classifier from old per-part classifier heads.

        Old checkpoints used keys like ``part_arcface.0.weight`` through
        ``part_arcface.3.weight``. New part-efficient configs use one shared
        classifier, so averaging compatible old heads gives a useful warm start.
        """
        if (
            not criterion_state
            or self.part_classifier_mode != 'shared'
            or not hasattr(self, 'part_arcface')
            or not hasattr(self.part_arcface, 'weight')
        ):
            return 0

        weights = []
        for idx in range(getattr(self, 'num_parts', 0)):
            key = f'part_arcface.{idx}.weight'
            v = criterion_state.get(key)
            if v is not None and v.shape == self.part_arcface.weight.shape:
                weights.append(v)

        if not weights:
            return 0

        with torch.no_grad():
            self.part_arcface.weight.copy_(torch.stack(weights, dim=0).mean(dim=0))
        return len(weights)

    def _scheduled_weight(self, base_weight, start_epoch: int, warmup_epochs: int) -> float:
        """Ramp optional ReID regularizers for clean from-scratch part training."""
        base = float(base_weight)
        if base <= 0:
            return 0.0

        epoch = float(getattr(self, 'current_epoch', 0))
        start = float(start_epoch)
        if epoch < start:
            return 0.0

        warmup = float(warmup_epochs)
        if warmup <= 0:
            return base

        progress = (epoch - start + 1.0) / warmup
        scale = max(0.0, min(1.0, progress))
        return base * scale

    def forward(self, outputs, targets, id_weight_scale=1.0, cam_ids=None, teacher_embeddings=None):
        """
        Args:
            outputs: dict with 'hm', 'wh', 'offset', 'embedding'
            targets: [N, 7+] target tensor
            id_weight_scale: multiplier for id_weight (0->1 during warm-up)
            cam_ids: [B] per-image camera IDs (optional, for camera-aware XBM mining)
            teacher_embeddings: tuple (teacher_embs [N_valid, D_t], valid_mask [N_targets])
                                from OSNet teacher for relational distillation (optional)

        Returns:
            (total_loss, loss_hm, loss_reg, loss_id, id_acc, loss_rep, loss_iou, loss_dcn_penalty)
        """
        device = outputs['hm'].device
        B, _, H, W = outputs['hm'].shape
        self._last_tiny_roi_rate = 0.0
        self._last_part_attention_entropy_raw = 0.0
        self._last_part_collapse_rate_raw = 0.0
        self._last_part_attention_entropy_fused = 0.0
        self._last_part_collapse_rate_fused = 0.0
        self._last_part_attention_entropy = 0.0
        self._last_part_collapse_rate = 0.0

        hm_target, wh_target, off_target, reg_mask, id_target, vis_target = generate_centernet_targets(
            targets, H, W, self.num_classes, device, batch_size=B,
            mp_thresh=self.mp_thresh
        )

        # ---- Heatmap Loss (Modified Focal) ----
        hm_pred = torch.sigmoid(outputs['hm'])
        if self.use_visibility_weighted_loss:
            # Per-pixel occlusion boost: higher weight for occluded objects (low vis)
            vis_boost = 1.0 + 0.5 * (1.0 - vis_target)  # [B, 1, H, W]
            loss_hm = modified_focal_loss(hm_pred, hm_target, pos_weight=vis_boost)
        else:
            loss_hm = modified_focal_loss(hm_pred, hm_target)

        # ---- P2 Stride-4 Heatmap Loss (Split-Stride) ----
        # When enabled, compute a second focal loss on the high-resolution
        # stride-4 heatmap. This provides dense supervision for peak separation
        # in crowds. Added to loss_hm with 1:1 weight (both go through
        # hm_weight + uncertainty weighting).
        # Uses generate_heatmap_only (no WH/offset/vis allocation → ~10× faster).
        hm_s4 = outputs.get('hm_s4')
        if hm_s4 is not None:
            H_s4, W_s4 = hm_s4.shape[2], hm_s4.shape[3]
            hm_s4_target = generate_heatmap_only(
                targets, H_s4, W_s4, self.num_classes, device, batch_size=B
            )
            hm_s4_pred = torch.sigmoid(hm_s4)
            loss_hm_s4 = modified_focal_loss(hm_s4_pred, hm_s4_target)
            loss_hm = loss_hm + loss_hm_s4

        num_pos = reg_mask.sum().clamp(min=1)

        # ---- Box Regression (GIoU for TRBL, L1 for offset) ----
        wh_pred = outputs['wh']
        off_pred = outputs.get('offset', torch.zeros_like(wh_pred[:, :2]))
        
        mask_bool = reg_mask > 0
        if mask_bool.sum() > 0:
            b_idx, y_idx, x_idx = torch.where(mask_bool)
            
            p_trbl = wh_pred[b_idx, :, y_idx, x_idx]
            p_off = off_pred[b_idx, :, y_idx, x_idx]
            
            t_trbl = wh_target[b_idx, :, y_idx, x_idx]
            t_off = off_target[b_idx, :, y_idx, x_idx]
            
            # Predict centers
            p_cx = x_idx.float() + p_off[:, 0]
            p_cy = y_idx.float() + p_off[:, 1]
            t_cx = x_idx.float() + t_off[:, 0]
            t_cy = y_idx.float() + t_off[:, 1]
            
            # Reconstruct predicted bounding boxes
            p_x1 = p_cx - p_trbl[:, 0]
            p_y1 = p_cy - p_trbl[:, 1]
            p_x2 = p_cx + p_trbl[:, 2]
            p_y2 = p_cy + p_trbl[:, 3]
            p_boxes = torch.stack([p_x1, p_y1, p_x2, p_y2], dim=1)
            
            # Reconstruct target bounding boxes
            t_x1 = t_cx - t_trbl[:, 0]
            t_y1 = t_cy - t_trbl[:, 1]
            t_x2 = t_cx + t_trbl[:, 2]
            t_y2 = t_cy + t_trbl[:, 3]
            t_boxes = torch.stack([t_x1, t_y1, t_x2, t_y2], dim=1)
            
            # GIoU calculation
            giou = bbox_giou(p_boxes, t_boxes)
            vis_w = reg_mask[b_idx, y_idx, x_idx]
            
            # Normalize GIoU loss by number of valid objects
            loss_wh = ((1.0 - giou) * vis_w).sum() / num_pos
            
            # Retain L1 loss for the precise center sub-pixel offset
            loss_off = F.l1_loss(p_off * vis_w.unsqueeze(1), t_off * vis_w.unsqueeze(1), reduction='sum') / num_pos
        else:
            loss_wh = torch.tensor(0.0, device=device)
            loss_off = torch.tensor(0.0, device=device)

        loss_id = torch.tensor(0.0, device=device)
        id_acc = torch.full((), float('nan'), device=device)
        loss_rep = torch.tensor(0.0, device=device)

        # ---- Repulsion Loss ----
        if self.rep_weight > 0 and num_pos > 1:
            loss_rep = repulsion_loss(
                wh_pred, wh_target, off_pred, off_target,
                reg_mask, hm_target, targets=targets
            )

        # ---- ID Loss (with warm-up scale, visibility weighting) ----
        effective_id_weight = self.id_weight * id_weight_scale

        if effective_id_weight > 0 and self.arcface_loss is not None:
            # Use id_target >= 0 mask (center pixels only) — avoids wasted
            # extraction at multi-positive non-center pixels (id_target = -1)
            mask = id_target >= 0
            if mask.sum() > 0:
                embed_map = outputs['embedding']  # [B, C, H_emb, W_emb]
                _, C_emb, H_emb, W_emb = embed_map.shape
                # Map detection grid coords → embedding grid coords
                det_to_emb = self._det_to_emb

                batch_indices, cy_coords, cx_coords = torch.where(mask)
                valid_roi_t = None  # Set by part_based branch; used by diversity loss filter

                if self.reid_extraction == 'part_based' and hasattr(self, 'part_extractor'):
                    # ---- Part-based extraction: RoI-Align 32×8 → 4×1 strips → attention fusion ----
                    roi_boxes = []
                    valid_roi = []  # True = box large enough for RoI-Align
                    for k in range(len(batch_indices)):
                        b_idx = batch_indices[k]
                        # Apply sub-pixel offset to get true GT center
                        # (matches inference path in head.py decode_detections)
                        cy_k = cy_coords[k].float() + off_target[b_idx, 1, cy_coords[k], cx_coords[k]]
                        cx_k = cx_coords[k].float() + off_target[b_idx, 0, cy_coords[k], cx_coords[k]]
                        # Decode TRBL for RoI
                        l_k = wh_target[b_idx, 0, cy_coords[k], cx_coords[k]]
                        t_k = wh_target[b_idx, 1, cy_coords[k], cx_coords[k]]
                        r_k = wh_target[b_idx, 2, cy_coords[k], cx_coords[k]]
                        b_k = wh_target[b_idx, 3, cy_coords[k], cx_coords[k]]
                        w_k = l_k + r_k
                        h_k = t_k + b_k
                        if w_k < 2.0 or h_k < 2.0:
                            valid_roi.append(False)
                            roi_boxes.append([b_idx.float(), cx_k - 0.5, cy_k - 0.5,
                                              cx_k + 0.5, cy_k + 0.5])
                        else:
                            valid_roi.append(True)
                            x1 = (cx_k - l_k).clamp(min=0)
                            y1 = (cy_k - t_k).clamp(min=0)
                            x2 = (cx_k + r_k).clamp(max=W - 1)
                            y2 = (cy_k + b_k).clamp(max=H - 1)
                            roi_boxes.append([b_idx.float(), x1, y1, x2, y2])

                    roi_boxes_t = torch.stack([torch.stack(b) for b in roi_boxes]).to(device)
                    # spatial_scale maps det-grid box coords to embed_map pixel coords.
                    # det=4 & reid=4 → 1.0; det=8 & reid=8 → 1.0; det=4 & reid=8 → 0.5.
                    part_out = self.part_extractor(
                        embed_map, roi_boxes_t, spatial_scale=det_to_emb,
                        return_attention_details=True,
                    )
                    if len(part_out) == 4:
                        fused_emb, part_embs, attn_weights, raw_attn_weights = part_out
                    else:
                        fused_emb, part_embs, attn_weights = part_out
                        raw_attn_weights = attn_weights
                    selected_embeddings = fused_emb  # [N_pos, C]

                    # For tiny boxes, override fused embedding with center-pixel extraction.
                    # Tiny samples keep contributing to fused ID loss, but are excluded
                    # from part-specific losses because their strip RoIs are not meaningful.
                    # IMPORTANT: use non-inplace ops — fused_emb/part_embs are in the autograd graph
                    valid_roi_t = torch.tensor(valid_roi, device=device)
                    if (~valid_roi_t).any():
                        tiny_mask = ~valid_roi_t
                        # Scale det-grid coords to embed map coords (integer floor for indexing)
                        cy_tiny = (cy_coords[tiny_mask].float() * det_to_emb).long().clamp(max=H_emb - 1)
                        cx_tiny = (cx_coords[tiny_mask].float() * det_to_emb).long().clamp(max=W_emb - 1)
                        center_embs = embed_map[batch_indices[tiny_mask], :,
                                                cy_tiny, cx_tiny]
                        center_embs = F.normalize(center_embs, p=2, dim=1, eps=1e-6)
                        # Non-inplace: scatter center_embs into a full-size zeros tensor,
                        # then use torch.where to select between center (tiny) and RoI (valid)
                        center_full = torch.zeros_like(selected_embeddings)
                        center_full[tiny_mask] = center_embs  # safe: center_full has no grad_fn
                        mask_2d = tiny_mask.unsqueeze(1).expand_as(selected_embeddings)
                        selected_embeddings = torch.where(mask_2d, center_full, selected_embeddings)
                    self._last_tiny_roi_rate = float((~valid_roi_t).float().mean().detach().cpu().item())
                    diag_mask = valid_roi_t if valid_roi_t is not None else torch.ones(
                        attn_weights.shape[0], dtype=torch.bool, device=device
                    )
                    if attn_weights is not None and attn_weights.numel() > 0 and diag_mask.any():
                        fused_diag = attn_weights[diag_mask]
                        fused_clamped = fused_diag.clamp_min(1e-8)
                        fused_entropy = -(fused_clamped * fused_clamped.log()).sum(dim=1)
                        self._last_part_attention_entropy_fused = float(fused_entropy.mean().detach().cpu().item())
                        self._last_part_collapse_rate_fused = float((fused_diag.max(dim=1).values > 0.85).float().mean().detach().cpu().item())
                    if raw_attn_weights is not None and raw_attn_weights.numel() > 0 and diag_mask.any():
                        raw_diag = raw_attn_weights[diag_mask]
                        raw_clamped = raw_diag.clamp_min(1e-8)
                        raw_entropy = -(raw_clamped * raw_clamped.log()).sum(dim=1)
                        raw_entropy_val = float(raw_entropy.mean().detach().cpu().item())
                        raw_collapse_val = float((raw_diag.max(dim=1).values > 0.85).float().mean().detach().cpu().item())
                        self._last_part_attention_entropy_raw = raw_entropy_val
                        self._last_part_collapse_rate_raw = raw_collapse_val
                        self._last_part_attention_entropy = raw_entropy_val
                        self._last_part_collapse_rate = raw_collapse_val
                else:
                    # ---- Center-pixel bilinear extraction (FairMOT-style) ----
                    cx_sub = cx_coords.float() + off_target[batch_indices, 0, cy_coords, cx_coords]
                    cy_sub = cy_coords.float() + off_target[batch_indices, 1, cy_coords, cx_coords]
                    # Scale det-grid sub-pixel coords to embed_map coords
                    selected_embeddings = sample_feature_map_bilinear(
                        embed_map,
                        cx_sub * det_to_emb,
                        cy_sub * det_to_emb,
                        batch_indices=batch_indices,
                    )
                    part_embs = None
                
                selected_ids = id_target[mask]
                
                # Get visibility weights for each sample (visibility-weighted ID loss)
                selected_vis = vis_target[:, 0, :, :][mask].clamp(min=0.1, max=1.0)  # Floor to avoid zero

                valid_mask = selected_ids >= 0
                if valid_mask.sum() > 0:
                    final_embeddings = selected_embeddings[valid_mask]
                    final_ids = selected_ids[valid_mask]
                    final_vis = selected_vis[valid_mask]  # Visibility weights for valid samples
                    
                    # Map per-detection camera IDs from batch-level cam_ids
                    final_cameras = None
                    if cam_ids is not None:
                        selected_cams = cam_ids[batch_indices]  # per-detection cam from per-image
                        final_cameras = selected_cams[valid_mask]

                    if self.arcface_loss is not None:
                        final_embeddings = F.normalize(final_embeddings, p=2, dim=1, eps=1e-6)
                        
                        # ---- BNNeck: split metric vs classification embedding spaces ----
                        # Pre-BN (L2-normalized) → triplet loss (metric space)
                        # Post-BN → dropout → ArcFace (classification space, camera bias removed)

                        # Triplet loss on pre-BN embeddings (metric space preserves geometry)
                        # Uses Cross-Batch Memory (XBM) when available — mines from
                        # ~4096 stored embeddings instead of the tiny current batch.
                        # Falls back to batch-level hard mining as secondary signal.
                        triplet_weight_eff = self._scheduled_weight(
                            self.triplet_weight,
                            self.triplet_start_epoch,
                            self.triplet_warmup_epochs,
                        )
                        triplet_loss_val = torch.tensor(0.0, device=device)
                        if triplet_weight_eff > 0 and hasattr(self, 'xbm') and self.xbm is not None:
                            triplet_loss_val = self.xbm(final_embeddings, final_ids, cameras=final_cameras)
                        elif triplet_weight_eff > 0 and hasattr(self, 'triplet_loss') and len(final_ids) >= 4:
                            triplet_loss_val = self.triplet_loss(final_embeddings, final_ids)
                        elif self.training and hasattr(self, 'xbm') and self.xbm is not None:
                            with torch.no_grad():
                                self.xbm.enqueue(final_embeddings.detach(), final_ids.detach(),
                                                 cameras=final_cameras.detach() if final_cameras is not None else None)

                        # BNNeck -> ArcFace (classification space)
                        if self.use_bnneck:
                            if self.training and len(final_embeddings) <= 2:
                                # Use functional batch_norm with running stats for tiny batches
                                # (avoids unstable variance AND mode-toggle mid-forward)
                                bn_embeddings = F.batch_norm(
                                    final_embeddings, self.bnneck.running_mean, self.bnneck.running_var,
                                    self.bnneck.weight, self.bnneck.bias, training=False,
                                    momentum=self.bnneck.momentum, eps=self.bnneck.eps
                                )
                            else:
                                bn_embeddings = self.bnneck(final_embeddings)
                        else:
                            bn_embeddings = final_embeddings
                        
                        # Apply dropout to post-BN embeddings for ArcFace regularization
                        arcface_emb = bn_embeddings
                        if self.reid_dropout is not None:
                            arcface_emb = self.reid_dropout(bn_embeddings)
                        # NOTE: No F.normalize here — ArcFace.forward_logits normalizes
                        # internally. This preserves BNNeck's directional shift through
                        # to the cosine computation. Previously a redundant external
                        # normalize was applied which worked but was architecturally
                        # confusing about where normalization responsibility lies.
                        
                        # ---- Fused ArcFace loss (visibility-weighted) ----
                        arcface_logits = self.arcface_loss.forward_logits(arcface_emb, final_ids)
                        per_sample_loss = F.cross_entropy(arcface_logits, final_ids, reduction='none',
                                                           label_smoothing=self.label_smoothing)

                        arcface_loss = (per_sample_loss * final_vis).sum() / final_vis.sum().clamp(min=0.1)

                        part_loss_weight_eff = self._scheduled_weight(
                            self.part_loss_weight,
                            self.part_loss_start_epoch,
                            self.part_loss_warmup_epochs,
                        )
                        part_triplet_weight_eff = self._scheduled_weight(
                            self.part_triplet_weight,
                            self.part_triplet_start_epoch,
                            self.part_triplet_warmup_epochs,
                        )
                        diversity_loss_weight_eff = self._scheduled_weight(
                            self.diversity_loss_weight,
                            self.diversity_start_epoch,
                            self.diversity_warmup_epochs,
                        )
                        bau_uniformity_weight_eff = self._scheduled_weight(
                            self.bau_uniformity_weight,
                            self.bau_start_epoch,
                            self.bau_warmup_epochs,
                        )
                        bau_camera_uniformity_weight_eff = self._scheduled_weight(
                            self.bau_camera_uniformity_weight,
                            self.bau_start_epoch,
                            self.bau_warmup_epochs,
                        )
                        
                        # ---- Part-level losses: ArcFace + Triplet per part ----
                        part_loss_total = torch.tensor(0.0, device=device)
                        part_triplet_total = torch.tensor(0.0, device=device)
                        if part_embs is not None and part_loss_weight_eff > 0 and hasattr(self, 'part_arcface'):
                            n_parts = self.num_parts
                            # part_embs: [N_pos, n_parts, C] — select valid samples
                            part_sample_mask = valid_mask
                            if valid_roi_t is not None:
                                part_sample_mask = valid_mask & valid_roi_t
                            valid_part_embs = part_embs[part_sample_mask]  # [N_part_valid, n_parts, C]
                            part_ids = selected_ids[part_sample_mask]
                            part_vis = selected_vis[part_sample_mask]
                            # Only compute part losses for boxes with non-zero part embeddings
                            part_norms = valid_part_embs.norm(dim=2)  # [N_valid, n_parts]
                            n_parts_with_triplet = 0
                            n_parts_active = 0
                            flat_part_embs = []
                            flat_part_ids = []
                            flat_part_vis = []
                            for p_idx in range(n_parts):
                                p_emb = valid_part_embs[:, p_idx, :]  # [N_valid, C]
                                # Skip parts that are all-zero (from tiny box fallback)
                                p_valid = part_norms[:, p_idx] > 0.1
                                if p_valid.sum() < 2:
                                    continue
                                n_parts_active += 1
                                p_emb_valid = F.normalize(p_emb[p_valid], p=2, dim=1, eps=1e-6)
                                p_ids = part_ids[p_valid]
                                p_vis = part_vis[p_valid]

                                # Per-part triplet loss on PRE-BN embeddings (metric space)
                                # Each part learns to form tight clusters independently —
                                # critical for cross-camera ReID where parts occlude differently
                                if part_triplet_weight_eff > 0 and hasattr(self, 'triplet_loss') and p_valid.sum() >= 4:
                                    part_triplet_total = part_triplet_total + self.triplet_loss(p_emb_valid, p_ids)
                                    n_parts_with_triplet += 1

                                # Apply part-level BNNeck before ArcFace (classification space)
                                if self.use_bnneck and hasattr(self, 'part_bnneck'):
                                    if self.training and len(p_emb_valid) <= 2:
                                        bn = self.part_bnneck[p_idx]
                                        p_emb_valid = F.batch_norm(
                                            p_emb_valid, bn.running_mean, bn.running_var,
                                            bn.weight, bn.bias, training=False,
                                            momentum=bn.momentum, eps=bn.eps
                                        )
                                    else:
                                        p_emb_valid = self.part_bnneck[p_idx](p_emb_valid)
                                # Apply dropout to part embeddings for consistent regularization
                                if self.reid_dropout is not None:
                                    p_emb_valid = self.reid_dropout(p_emb_valid)
                                if self.part_classifier_mode == 'shared' and not isinstance(self.part_arcface, nn.ModuleList):
                                    flat_part_embs.append(p_emb_valid)
                                    flat_part_ids.append(p_ids)
                                    flat_part_vis.append(p_vis)
                                    continue
                                # No F.normalize — ArcFace normalizes internally
                                part_classifier = self.part_arcface
                                if isinstance(self.part_arcface, nn.ModuleList):
                                    part_classifier = self.part_arcface[p_idx]
                                p_logits = part_classifier.forward_logits(p_emb_valid, p_ids)
                                p_loss = F.cross_entropy(p_logits, p_ids, reduction='none',
                                                          label_smoothing=self.label_smoothing)
                                p_loss = (p_loss * p_vis).sum() / p_vis.sum().clamp(min=0.1)
                                part_loss_total = part_loss_total + p_loss
                            if flat_part_embs:
                                flat_emb = torch.cat(flat_part_embs, dim=0)
                                flat_ids = torch.cat(flat_part_ids, dim=0)
                                flat_vis = torch.cat(flat_part_vis, dim=0)
                                p_logits = self.part_arcface.forward_logits(flat_emb, flat_ids)
                                p_loss = F.cross_entropy(p_logits, flat_ids, reduction='none',
                                                          label_smoothing=self.label_smoothing)
                                part_loss_total = (p_loss * flat_vis).sum() / flat_vis.sum().clamp(min=0.1)
                            else:
                                part_loss_total = part_loss_total / max(n_parts_active, 1)  # Average over ACTIVE parts
                            if n_parts_with_triplet > 0:
                                part_triplet_total = part_triplet_total / n_parts_with_triplet
                        
                        # ---- Part Diversity: Gram Orthogonality Loss ----
                        # Penalizes off-diagonal elements of P @ P^T where P is
                        # the [N, 4, C] L2-normalized part embedding matrix.
                        # Diagonal = 1 (self-sim), off-diagonal → 0 (orthogonality).
                        # IMPORTANT: Exclude tiny-box samples whose parts are copies
                        # of the center embedding — these always produce max penalty
                        # (Gram = all-ones) and would dominate the diversity gradient.
                        loss_diversity = torch.tensor(0.0, device=device)
                        div_eligible = valid_mask.clone()
                        if hasattr(self, 'part_extractor') and valid_roi_t is not None:
                            # Only RoI-extracted parts have meaningful diversity signal
                            div_eligible = valid_mask & valid_roi_t
                        if (diversity_loss_weight_eff > 0
                                and part_embs is not None
                                and div_eligible.sum() >= 2):
                            P = F.normalize(part_embs[div_eligible], p=2, dim=2, eps=1e-6)  # [N_eligible, 4, C]
                            G = torch.bmm(P, P.transpose(1, 2))  # [N_valid, 4, 4]
                            eye = torch.eye(G.shape[1], device=device).unsqueeze(0)
                            # MSE of off-diagonal elements (diagonal is always 1 after L2-norm)
                            off_diag = (G - eye) ** 2
                            num_off = G.shape[1] * (G.shape[1] - 1)  # 4*3 = 12
                            loss_diversity = off_diag.sum(dim=[1, 2]).mean() / max(num_off, 1)

                        # Combined loss:
                        #   ArcFace_fused
                        #   + part_weight * ArcFace_parts
                        #   + triplet_weight * (triplet_fused + part_weight * triplet_parts)
                        #   + diversity_weight * Gram_orthogonality
                        loss_id = (arcface_loss
                                   + part_loss_weight_eff * part_loss_total
                                   + triplet_weight_eff * triplet_loss_val
                                   + part_triplet_weight_eff * part_triplet_total
                                   + diversity_loss_weight_eff * loss_diversity)

                        # ---- BAU Uniformity: hypersphere spreading regularization ----
                        # Applied to pre-BN L2-normalized embeddings (metric space).
                        # final_embeddings are already L2-normalized at this point.
                        if bau_uniformity_weight_eff > 0 and len(final_embeddings) >= 2:
                            loss_uniform = uniformity_loss(final_embeddings, t=self.bau_temperature)
                            loss_id = loss_id + bau_uniformity_weight_eff * loss_uniform

                        if (bau_camera_uniformity_weight_eff > 0
                                and final_cameras is not None
                                and len(final_embeddings) >= 2):
                            loss_cam_uniform = camera_uniformity_loss(
                                final_embeddings, final_cameras,
                                t=self.bau_temperature
                            )
                            loss_id = loss_id + bau_camera_uniformity_weight_eff * loss_cam_uniform

                        # ---- Knowledge Distillation: weighted relational alignment ----
                        # Instance-aligned: teacher embeddings ordered by GT instance index,
                        # student embeddings at the same GT center positions.
                        # Uses labels and cameras for selective weighting:
                        #   - Same-ID pairs upweighted (pos_weight=5×)
                        #   - Cross-camera same-ID pairs highest weight (10×)
                        #   - Hard negatives (top-k by teacher similarity) upweighted (3×)
                        #   - Diagonal masked (self-similarity is trivially 1.0)
                        loss_distill = torch.tensor(0.0, device=device)
                        loss_feat_distill = torch.tensor(0.0, device=device)
                        if (self.distill_loss is not None
                                and self.distillation_weight > 0
                                and teacher_embeddings is not None):
                            t_embs, t_valid_mask, t_instance_ids = teacher_embeddings
                            if len(t_embs) >= 2 and len(final_embeddings) >= 2:
                                # Instance-aligned matching: both are ordered by GT annotation.
                                # Teacher: extracted from targets[pid>=0] in order.
                                # Student: extracted via id_target>=0 → heatmap raster order.
                                # Match using instance IDs (person IDs serve as instance keys
                                # since each annotation maps to exactly one center pixel).
                                s_idx_list, t_idx_list = [], []
                                t_iid_to_idx = {}
                                for idx, iid in enumerate(t_instance_ids.tolist()):
                                    if iid not in t_iid_to_idx:  # first occurrence
                                        t_iid_to_idx[iid] = idx
                                for s_idx, pid in enumerate(final_ids.tolist()):
                                    if pid in t_iid_to_idx:
                                        s_idx_list.append(s_idx)
                                        t_idx_list.append(t_iid_to_idx.pop(pid))
                                if len(s_idx_list) >= 2:
                                    s_matched = final_embeddings[s_idx_list]  # [M, D_s]
                                    t_matched = t_embs[t_idx_list]            # [M, D_t]
                                    matched_ids = final_ids[s_idx_list]       # [M]
                                    matched_cams = final_cameras[s_idx_list] if final_cameras is not None else None
                                    loss_distill = self.distill_loss(
                                        s_matched, t_matched,
                                        labels=matched_ids, cameras=matched_cams,
                                    )
                                    # Feature-level KD: direct embedding alignment
                                    if self.feature_distill_loss is not None:
                                        loss_feat_distill = self.feature_distill_loss(
                                            s_matched, t_matched
                                        )
                                    else:
                                        loss_feat_distill = torch.tensor(0.0, device=device)
                        
                        # KD warmup: linear ramp from 0→full over kd_warmup_batches
                        kd_factor = min(1.0, self._kd_step / max(1, self.kd_warmup_batches))
                        self._kd_step += 1
                        self._last_loss_distill = loss_distill.item() if loss_distill.item() > 0 else 0.0
                        self._last_loss_feat_distill = loss_feat_distill.item() if isinstance(loss_feat_distill, torch.Tensor) and loss_feat_distill.item() > 0 else 0.0
                        loss_id = loss_id + self.distillation_weight * kd_factor * loss_distill
                        loss_id = loss_id + self.feature_kd_weight * kd_factor * loss_feat_distill

                        # ID accuracy uses post-BN embeddings (classification space) for consistency
                        with torch.no_grad():
                            if self.use_bnneck:
                                class_logits = self.arcface_loss.class_logits(bn_embeddings)
                            else:
                                class_logits = self.arcface_loss.class_logits(final_embeddings)
                            pred_ids = torch.argmax(class_logits, dim=1)
                            id_acc = (pred_ids == final_ids).float().mean()

        # ---- IoU Loss (quality-aware scoring) ----
        loss_iou = torch.tensor(0.0, device=device)
        if outputs.get('iou') is not None and num_pos > 0:
            iou_pred = outputs['iou']  # [B, 1, H, W] raw logits
            # Compute center-aligned IoU targets at GT locations
            mask_pos = reg_mask > 0
            if mask_pos.sum() > 0:
                b_idx, cy_idx, cx_idx = torch.where(mask_pos)
                p_l = wh_pred[b_idx, 0, cy_idx, cx_idx]
                p_t = wh_pred[b_idx, 1, cy_idx, cx_idx]
                p_r = wh_pred[b_idx, 2, cy_idx, cx_idx]
                p_b = wh_pred[b_idx, 3, cy_idx, cx_idx]
                
                g_l = wh_target[b_idx, 0, cy_idx, cx_idx]
                g_t = wh_target[b_idx, 1, cy_idx, cx_idx]
                g_r = wh_target[b_idx, 2, cy_idx, cx_idx]
                g_b = wh_target[b_idx, 3, cy_idx, cx_idx]
                
                # Center-aligned IoU (boxes share center, compute overlap from sizes)
                inter_w = torch.min(p_l, g_l) + torch.min(p_r, g_r)
                inter_h = torch.min(p_t, g_t) + torch.min(p_b, g_b)
                inter_area = inter_w * inter_h
                pred_area = (p_l + p_r) * (p_t + p_b)
                gt_area = (g_l + g_r) * (g_t + g_b)
                iou_targets = inter_area / (pred_area + gt_area - inter_area + 1e-6)
                iou_targets = iou_targets.clamp(0, 1).detach()
                # BCE loss on predicted IoU logits, weighted by Gaussian center-ness
                # (matches the regression loss weighting convention)
                iou_logits = iou_pred[b_idx, 0, cy_idx, cx_idx]
                iou_w = reg_mask[b_idx, cy_idx, cx_idx]  # Gaussian center-ness
                loss_iou = (F.binary_cross_entropy_with_logits(
                    iou_logits, iou_targets, reduction='none'
                ) * iou_w).sum() / num_pos

        # DCN penalty slot: always zero (DCN bbox penalty removed — weight was 0 + doubly dead)
        loss_dcn_penalty = torch.tensor(0.0, device=device)

        # ---- Total Loss (with Uncertainty Weighting) ----
        # Combine all detection losses for FairMOT-style 2-param weighting
        loss_det_combined = (
            self.hm_weight * loss_hm +
            self.wh_weight * loss_wh +
            self.off_weight * loss_off +
            self.rep_weight * loss_rep +
            self.iou_weight * loss_iou
        )
        loss_id_combined = effective_id_weight * loss_id

        if self.uncertainty_mode == '2param':
            # FairMOT-style: L = 0.5 * [exp(-s_det)*L_det + exp(-s_id)*L_id + s_det + s_id]
            # Proven stable by FairMOT across millions of training steps.
            # s_det clamp [-4, 4], s_id clamp [-4, 2] (prevent ID starving det)
            s_det = self.s_det.clamp(-4, 4)
            s_id = self.s_id_unc.clamp(-4, 2)
            if id_weight_scale < 1.0:
                s_id = s_id.detach()  # Freeze during warmup
            total_loss = 0.5 * (
                torch.exp(-s_det) * loss_det_combined + s_det +
                torch.exp(-s_id) * loss_id_combined + s_id
            )
        else:
            total_loss = loss_det_combined + loss_id_combined

        return total_loss, loss_hm, loss_wh + loss_off, loss_id, id_acc, loss_rep, loss_iou, loss_dcn_penalty
