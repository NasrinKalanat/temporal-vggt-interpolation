"""Temporal conditioning modules for Temporal-VGGT v1.

Encodes (role, day-of-year, relative gap features) into FiLM gamma/beta vectors
that modulate patch token features after patch embedding.
"""
import math
from typing import Dict

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Encodes temporal context into FiLM scale/shift parameters.

    Three sources of information are summed:
      - role embedding: which temporal slot (t1=0, t3=1, t2_query=2)
      - date MLP: sinusoidal day-of-year encoding
      - relative gap MLP: tau, 1-tau, normalized left/right/total gaps

    The output projection is zero-initialized so conditioning is identity at
    the start of training (gamma=0 → scale=1, beta=0 → no shift with FiLM).
    """

    def __init__(self, dim: int = 1024, hidden_dim: int = 1024, rel_gap_dim: int = 5):
        super().__init__()

        self.role_embed = nn.Embedding(3, hidden_dim)

        self.date_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rel_gap_mlp = nn.Sequential(
            nn.Linear(rel_gap_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.to_gamma_beta = nn.Linear(hidden_dim, 2 * dim)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(
        self,
        role_id: torch.Tensor,      # [B] long, 0=t1, 1=t3, 2=t2_query
        day_of_year: torch.Tensor,  # [B] float
        rel_gap_feat: torch.Tensor, # [B, 5]
    ):
        """Returns gamma [B, dim] and beta [B, dim]."""
        day_norm = day_of_year.float() / 365.0
        date_feat = torch.stack(
            [
                torch.sin(2 * math.pi * day_norm),
                torch.cos(2 * math.pi * day_norm),
            ],
            dim=-1,
        )

        h = (
            self.role_embed(role_id)
            + self.date_mlp(date_feat)
            + self.rel_gap_mlp(rel_gap_feat.float())
        )

        gamma, beta = self.to_gamma_beta(h).chunk(2, dim=-1)
        return gamma, beta  # each [B, dim]


class ResidualAdaLN(nn.Module):
    """Residual adaptive layer norm: x' = x + alpha * (LN(x) * (1 + gamma) + beta).

    alpha is a learnable scalar initialized to zero so the module starts as
    identity, preserving pretrained VGGT behavior.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,       # [B, ..., dim]
        gamma: torch.Tensor,   # [B, dim]
        beta: torch.Tensor,    # [B, dim]
    ) -> torch.Tensor:
        # Broadcast gamma/beta over all middle dimensions
        for _ in range(x.dim() - gamma.dim()):
            gamma = gamma.unsqueeze(-2)
            beta = beta.unsqueeze(-2)
        return x + self.alpha * (self.norm(x) * (1 + gamma) + beta)


def apply_film(
    x: torch.Tensor,       # [B, ..., dim]
    gamma: torch.Tensor,   # [B, dim]
    beta: torch.Tensor,    # [B, dim]
) -> torch.Tensor:
    """FiLM without LayerNorm: x' = x * (1 + gamma) + beta.

    Zero-init gamma/beta means identity at start.
    gamma and beta are broadcast over all intermediate dimensions.
    """
    for _ in range(x.dim() - gamma.dim()):
        gamma = gamma.unsqueeze(-2)
        beta = beta.unsqueeze(-2)
    return x * (1 + gamma) + beta


class CameraEncoder(nn.Module):
    """Encodes per-view camera parameters into (gamma, beta) conditioning pair.

    Input features per view (16 dims):
        R:  flattened 3×3 rotation from transform_matrix        [9]
        t:  scene-normalized camera center: (center - avg_pos) * scale  [3]
        fl: focal lengths normalized by image dims               [2]
        pp: principal point normalized by image dims             [2]

    Output projection is zero-initialized → identity conditioning at training start.
    """

    INPUT_DIM = 16

    def __init__(self, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, camera: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            camera: dict with:
                transform_matrix: [B, V, 4, 4]
                fl_x, fl_y, cx, cy, img_w, img_h: [B, V]
                avg_pos: [B, 3]
                scale:   [B]
        Returns:
            gamma, beta: each [B, V, out_dim]
        """
        T = camera["transform_matrix"]                            # [B, V, 4, 4]
        R = T[..., :3, :3].reshape(*T.shape[:-2], 9)             # [B, V, 9]
        t_world = T[..., :3, 3]                                   # [B, V, 3]

        avg_pos = camera["avg_pos"]                               # [B, 3]
        scale   = camera["scale"]                                 # [B]
        t_norm  = (t_world - avg_pos.unsqueeze(1)) * scale.view(-1, 1, 1)  # [B, V, 3]

        fl_x  = camera["fl_x"].unsqueeze(-1)                     # [B, V, 1]
        fl_y  = camera["fl_y"].unsqueeze(-1)
        cx    = camera["cx"].unsqueeze(-1)
        cy    = camera["cy"].unsqueeze(-1)
        img_w = camera["img_w"].unsqueeze(-1)
        img_h = camera["img_h"].unsqueeze(-1)

        fl_norm = torch.cat([fl_x / img_w, fl_y / img_h], dim=-1)  # [B, V, 2]
        pp_norm = torch.cat([cx  / img_w,  cy  / img_h], dim=-1)   # [B, V, 2]

        feat = torch.cat([R, t_norm, fl_norm, pp_norm], dim=-1)     # [B, V, 16]
        gamma, beta = self.mlp(feat).chunk(2, dim=-1)               # each [B, V, out_dim]
        return gamma, beta


class EndpointTimeEncoder(nn.Module):
    """Endpoint-specific temporal context encoder (no role embedding).

    Encodes (endpoint_date, target_date, rel_gap_feat) into conditioning params.
    Feature vector: day-of-year sin/cos (2) + signed gap to target/365 (1) + rel_gap (5) = 8 dims.

    Both heads are zero-initialized so conditioning is identity at training start.

    Args:
        out_dim:    Output feature dimension (typically D = 2C = 2048 for VGGT-1B).
        hidden_dim: Hidden dim of the MLP backbone.
        gap_scale:  Denominator for signed gap normalization (default 365.0).
    """

    def __init__(self, out_dim: int, hidden_dim: int = 256, gap_scale: float = 365.0):
        super().__init__()
        self.gap_scale = gap_scale
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),  # 2 + 1 + 5
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.to_gamma_beta = nn.Linear(hidden_dim, 2 * out_dim)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.to_additive = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.to_additive.weight)
        nn.init.zeros_(self.to_additive.bias)

    def _encode(
        self,
        endpoint_date: torch.Tensor,   # [B] day-of-year float
        target_date: torch.Tensor,     # [B] day-of-year float (t2)
        rel_gap_feat: torch.Tensor,    # [B, 5]
    ) -> torch.Tensor:                 # [B, hidden_dim]
        day_norm = endpoint_date.float() / 365.0
        date_feat = torch.stack([
            torch.sin(2 * math.pi * day_norm),
            torch.cos(2 * math.pi * day_norm),
        ], dim=-1)  # [B, 2]
        signed_gap = (target_date - endpoint_date).float().unsqueeze(-1) / self.gap_scale  # [B, 1]
        feat = torch.cat([date_feat, signed_gap, rel_gap_feat.float()], dim=-1)  # [B, 8]
        return self.mlp(feat)

    def gamma_beta(self, endpoint_date, target_date, rel_gap_feat):
        """Returns (gamma, beta) each [B, out_dim] for film/residual_adaln."""
        return self.to_gamma_beta(self._encode(endpoint_date, target_date, rel_gap_feat)).chunk(2, dim=-1)

    def additive(self, endpoint_date, target_date, rel_gap_feat):
        """Returns [B, out_dim] additive offset."""
        return self.to_additive(self._encode(endpoint_date, target_date, rel_gap_feat))


class SharedTimeEncoder(nn.Module):
    """Shared temporal context encoder with separate source/target projection heads.

    Encodes (role, day-of-year, relative gap features) into a shared hidden H,
    then projects to (gamma, beta) pairs for two different output dimensions:
      source_head → [B, source_dim]  for t1/t3 features at dim=2C
      target_head → [B, target_dim]  for t2 query tokens at dim=C

    Both heads are zero-initialized so conditioning is identity at training start.
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 1024,
        rel_gap_dim: int = 5,
    ):
        super().__init__()
        self.role_embed = nn.Embedding(3, hidden_dim)
        self.date_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rel_gap_mlp = nn.Sequential(
            nn.Linear(rel_gap_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.source_head = nn.Linear(hidden_dim, 2 * source_dim)
        self.target_head = nn.Linear(hidden_dim, 2 * target_dim)
        nn.init.zeros_(self.source_head.weight)
        nn.init.zeros_(self.source_head.bias)
        nn.init.zeros_(self.target_head.weight)
        nn.init.zeros_(self.target_head.bias)

        self.source_additive_head = nn.Linear(hidden_dim, source_dim)
        self.target_additive_head = nn.Linear(hidden_dim, target_dim)
        nn.init.zeros_(self.source_additive_head.weight)
        nn.init.zeros_(self.source_additive_head.bias)
        nn.init.zeros_(self.target_additive_head.weight)
        nn.init.zeros_(self.target_additive_head.bias)

    def _encode(
        self,
        role_id: torch.Tensor,       # [B] long
        day_of_year: torch.Tensor,   # [B] float
        rel_gap_feat: torch.Tensor,  # [B, 5]
    ) -> torch.Tensor:               # [B, hidden_dim]
        day_norm = day_of_year.float() / 365.0
        date_feat = torch.stack(
            [
                torch.sin(2 * math.pi * day_norm),
                torch.cos(2 * math.pi * day_norm),
            ],
            dim=-1,
        )
        return (
            self.role_embed(role_id)
            + self.date_mlp(date_feat)
            + self.rel_gap_mlp(rel_gap_feat.float())
        )

    def source(self, role_id, day_of_year, rel_gap_feat):
        """Returns gamma, beta each [B, source_dim]."""
        h = self._encode(role_id, day_of_year, rel_gap_feat)
        return self.source_head(h).chunk(2, dim=-1)

    def target(self, role_id, day_of_year, rel_gap_feat):
        """Returns gamma, beta each [B, target_dim]."""
        h = self._encode(role_id, day_of_year, rel_gap_feat)
        return self.target_head(h).chunk(2, dim=-1)

    def both(self, role_id, day_of_year, rel_gap_feat):
        """Returns ((src_gamma, src_beta), (tgt_gamma, tgt_beta)) in one encode pass."""
        h = self._encode(role_id, day_of_year, rel_gap_feat)
        src = self.source_head(h).chunk(2, dim=-1)
        tgt = self.target_head(h).chunk(2, dim=-1)
        return src, tgt

    def source_additive(self, role_id, day_of_year, rel_gap_feat):
        """Returns [B, source_dim] additive offset for t1/t3 tokens."""
        h = self._encode(role_id, day_of_year, rel_gap_feat)
        return self.source_additive_head(h)

    def target_additive(self, role_id, day_of_year, rel_gap_feat):
        """Returns [B, target_dim] additive offset for t2 query tokens."""
        h = self._encode(role_id, day_of_year, rel_gap_feat)
        return self.target_additive_head(h)


class SharedCameraEncoder(nn.Module):
    """Shared camera parameter encoder with separate source/target projection heads.

    Encodes per-view 16-dim camera features [R(9), t_norm(3), fl_norm(2), pp_norm(2)]
    through a shared MLP backbone, then projects to (gamma, beta) pairs:
      source_head → [B, V, source_dim]  for t1/t3 features at dim=2C
      target_head → [B, V, target_dim]  for t2 query tokens at dim=C

    Both heads are zero-initialized so conditioning is identity at training start.
    The shared backbone is NOT zero-initialized; identity is preserved because the
    zero-initialized heads zero out the backbone activations.
    """

    INPUT_DIM = 16

    def __init__(self, source_dim: int, target_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.source_head = nn.Linear(hidden_dim, 2 * source_dim)
        self.target_head = nn.Linear(hidden_dim, 2 * target_dim)
        nn.init.zeros_(self.source_head.weight)
        nn.init.zeros_(self.source_head.bias)
        nn.init.zeros_(self.target_head.weight)
        nn.init.zeros_(self.target_head.bias)

    @staticmethod
    def _extract_feat(camera: dict) -> torch.Tensor:
        T = camera["transform_matrix"]                                    # [B, V, 4, 4]
        R = T[..., :3, :3].reshape(*T.shape[:-2], 9)                    # [B, V, 9]
        t_world = T[..., :3, 3]                                          # [B, V, 3]
        avg_pos = camera["avg_pos"]                                       # [B, 3]
        scale   = camera["scale"]                                         # [B]
        t_norm  = (t_world - avg_pos.unsqueeze(1)) * scale.view(-1, 1, 1)
        fl_x  = camera["fl_x"].unsqueeze(-1)
        fl_y  = camera["fl_y"].unsqueeze(-1)
        cx    = camera["cx"].unsqueeze(-1)
        cy    = camera["cy"].unsqueeze(-1)
        img_w = camera["img_w"].unsqueeze(-1)
        img_h = camera["img_h"].unsqueeze(-1)
        fl_norm = torch.cat([fl_x / img_w, fl_y / img_h], dim=-1)
        pp_norm = torch.cat([cx  / img_w,  cy  / img_h], dim=-1)
        return torch.cat([R, t_norm, fl_norm, pp_norm], dim=-1)          # [B, V, 16]

    def _encode(self, camera: dict) -> torch.Tensor:
        return self.shared_mlp(self._extract_feat(camera))               # [B, V, hidden_dim]

    def source(self, camera: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns gamma, beta each [B, V, source_dim]."""
        h = self._encode(camera)
        return self.source_head(h).chunk(2, dim=-1)

    def target(self, camera: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns gamma, beta each [B, V, target_dim]."""
        h = self._encode(camera)
        return self.target_head(h).chunk(2, dim=-1)


class AdditiveCameraConditioner(nn.Module):
    """Per-view additive camera conditioner with a shared backbone and split projections.

    backbone:     Linear(16, H) → GELU → Linear(H, H) → GELU
    source_proj:  Linear(H, 2C)  zero-init → [B, V, 2C] offset added to t1/t3 features
    target_proj:  Linear(H, C)   zero-init → [B, V, C]  offset added to t2 query tokens

    Applied as `feats += conditioner(cam)[:, :, None, :]` to broadcast over the patch dim.
    Zero-initialized projections preserve pretrained VGGT behavior at training start.
    """

    INPUT_DIM = 16

    def __init__(self, hidden_dim: int, C: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.source_proj = nn.Linear(hidden_dim, 2 * C)
        self.target_proj = nn.Linear(hidden_dim, C)
        nn.init.zeros_(self.source_proj.weight)
        nn.init.zeros_(self.source_proj.bias)
        nn.init.zeros_(self.target_proj.weight)
        nn.init.zeros_(self.target_proj.bias)

    @staticmethod
    def _extract_feat(camera: dict) -> torch.Tensor:
        T = camera["transform_matrix"]                                    # [B, V, 4, 4]
        R = T[..., :3, :3].reshape(*T.shape[:-2], 9)                    # [B, V, 9]
        t_world = T[..., :3, 3]                                          # [B, V, 3]
        avg_pos = camera["avg_pos"]                                       # [B, 3]
        scale   = camera["scale"]                                         # [B]
        t_norm  = (t_world - avg_pos.unsqueeze(1)) * scale.view(-1, 1, 1)
        fl_x  = camera["fl_x"].unsqueeze(-1)
        fl_y  = camera["fl_y"].unsqueeze(-1)
        cx    = camera["cx"].unsqueeze(-1)
        cy    = camera["cy"].unsqueeze(-1)
        img_w = camera["img_w"].unsqueeze(-1)
        img_h = camera["img_h"].unsqueeze(-1)
        fl_norm = torch.cat([fl_x / img_w, fl_y / img_h], dim=-1)
        pp_norm = torch.cat([cx  / img_w,  cy  / img_h], dim=-1)
        return torch.cat([R, t_norm, fl_norm, pp_norm], dim=-1)          # [B, V, 16]

    def source(self, camera: dict) -> torch.Tensor:
        """Returns [B, V, 2C] additive offset for t1/t3 VGGT features."""
        return self.source_proj(self.backbone(self._extract_feat(camera)))

    def target(self, camera: dict) -> torch.Tensor:
        """Returns [B, V, C] additive offset for t2 query tokens."""
        return self.target_proj(self.backbone(self._extract_feat(camera)))


class BlockTimeEncoder(nn.Module):
    """Maps rel_gap_feat → (gamma, beta) for block-level LoRA time conditioning.

    Uses only the triplet-level temporal geometry (tau, gap sizes) because at
    block level the token sequence mixes all three roles [t1 | t3 | t2], so
    per-role date conditioning is not applicable.

    Output projection is zero-initialized so conditioning starts as identity.
    """

    def __init__(self, gap_dim: int = 5, dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gap_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2 * dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, rel_gap_feat: torch.Tensor):
        """Args:
            rel_gap_feat: [B, gap_dim]
        Returns:
            gamma, beta: each [B, dim]
        """
        gamma, beta = self.mlp(rel_gap_feat.float()).chunk(2, dim=-1)
        return gamma, beta


def build_relative_gap_features(
    t1_day: torch.Tensor,
    t2_day: torch.Tensor,
    t3_day: torch.Tensor,
    gap_scale: float = 365.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build the 5-dim relative gap feature vector shared across all token groups.

    Args:
        t1_day, t2_day, t3_day: [B] ordinal day-of-year tensors.

    Returns:
        rel_gap_feat: [B, 5] with columns:
            [tau, 1-tau, left_gap_norm, right_gap_norm, total_gap_norm]
    """
    left_gap = (t2_day - t1_day).float()
    right_gap = (t3_day - t2_day).float()
    total_gap = (t3_day - t1_day).float()
    total_safe = total_gap.clamp_min(eps)

    return torch.stack(
        [
            left_gap / total_safe,
            right_gap / total_safe,
            left_gap / gap_scale,
            right_gap / gap_scale,
            total_gap / gap_scale,
        ],
        dim=-1,
    )


def extract_camera_feat(camera: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract 16-dim per-view camera feature: R(9) + t_norm(3) + fl_norm(2) + pp_norm(2).
    Returns [B, V, 16].
    """
    T = camera["transform_matrix"]                                    # [B, V, 4, 4]
    R = T[..., :3, :3].reshape(*T.shape[:-2], 9)                     # [B, V, 9]
    t_world = T[..., :3, 3]                                           # [B, V, 3]
    avg_pos = camera["avg_pos"]                                       # [B, 3]
    scale   = camera["scale"]                                         # [B]
    t_norm  = (t_world - avg_pos.unsqueeze(-2)) * scale.view(-1, 1, 1)  # [B, V, 3]
    fl_x = camera["fl_x"].unsqueeze(-1)                              # [B, V, 1]
    fl_y = camera["fl_y"].unsqueeze(-1)
    cx   = camera["cx"].unsqueeze(-1)
    cy   = camera["cy"].unsqueeze(-1)
    img_w = camera["img_w"].unsqueeze(-1)
    img_h = camera["img_h"].unsqueeze(-1)
    fl_norm = torch.cat([fl_x / img_w, fl_y / img_h], dim=-1)        # [B, V, 2]
    pp_norm = torch.cat([cx / img_w, cy / img_h], dim=-1)            # [B, V, 2]
    return torch.cat([R, t_norm, fl_norm, pp_norm], dim=-1)           # [B, V, 16]


def compute_relative_camera_feat(
    source_camera: Dict[str, torch.Tensor],
    target_camera: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute per-source-view relative camera features to mean target pose.

    For each source view, computes the relative rotation (9) and relative
    translation (3) to the mean t2 camera pose. Returns [B, V_src, 12].
    """
    T_src = source_camera["transform_matrix"]   # [B, V_src, 4, 4]
    T_tgt = target_camera["transform_matrix"]   # [B, V_tgt, 4, 4]

    # Mean target pose
    T_tgt_mean = T_tgt.mean(dim=1, keepdim=True)  # [B, 1, 4, 4]
    R_tgt = T_tgt_mean[..., :3, :3]               # [B, 1, 3, 3]
    t_tgt = T_tgt_mean[..., :3, 3]                # [B, 1, 3]

    R_src = T_src[..., :3, :3]                    # [B, V_src, 3, 3]
    t_src = T_src[..., :3, 3]                     # [B, V_src, 3]

    # Relative rotation: R_tgt^T @ R_src  (per source view)
    R_rel = torch.matmul(R_tgt.transpose(-1, -2), R_src)  # [B, V_src, 3, 3]
    R_rel_flat = R_rel.reshape(*R_rel.shape[:-2], 9)      # [B, V_src, 9]

    # Relative translation (normalized by scene scale)
    scale = source_camera["scale"].view(-1, 1, 1)          # [B, 1, 1]
    t_rel = (t_src - t_tgt) * scale                        # [B, V_src, 3]

    return torch.cat([R_rel_flat, t_rel], dim=-1)           # [B, V_src, 12]


class CameraMLP(nn.Module):
    """Per-view additive camera encoder. Maps 16-dim camera feat -> d_model offset.
    Zero-init output -> identity at training start.
    """
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, camera: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns [B, V, d_model] additive offset."""
        return self.mlp(extract_camera_feat(camera))


class RelativeCameraMLP(nn.Module):
    """Per-view relative camera encoder. Maps 12-dim relative feat -> d_model offset.
    Zero-init output -> identity at training start.
    """
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        source_camera: Dict[str, torch.Tensor],
        target_camera: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns [B, V_src, d_model] additive offset."""
        return self.mlp(compute_relative_camera_feat(source_camera, target_camera))

