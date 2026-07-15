"""Temporal-VGGT v5: Full VGGT aggregator with TimeCrossBlock at the final layer.

Architecture (see docs/architectures/model_v5.py for the full pseudo-code):
  1. Patch-embed t1/t3 images with frozen VGGT patch_embed.
  2. Apply camera+time conditioning to t1, t3, and t2 query (mask_patch_token) patch features.
  3. Concatenate all 2S+Q frames [t1 | t2q | t3] and run through the full VGGT aggregator
     (frozen frame_blocks + global_blocks).
  4. At time_block_layer (default 23): inject one TimeCrossBlock that updates only t2 query
     tokens using t1+t3 tokens as cross-attention memory.
  5. Cache 4 intermediate outputs at cache_layers [4, 11, 17, 23] as concat(frame,global)→2C.
  6. Select t2 query frames; decode with pretrained VGGT DPTHead.

Trainable: mask_patch_token, time_encoder, [cam_conditioner/cam_additive],
           source/target_time_adaln, time_cross_block, point_head (fine-tuned).
Frozen:    VGGT aggregator (patch_embed, frame_blocks, global_blocks, special tokens).
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.heads.dpt_head import DPTHead

from models.time_encoding import (
    SharedTimeEncoder,
    ResidualAdaLN,
    apply_film,
    build_relative_gap_features,
)
from models.camera_encoding import build_camera_features

logger = logging.getLogger(__name__)


# ── Camera conditioners (same as v4) ─────────────────────────────────────────

class CameraConditioner(nn.Module):
    """Maps 13-dim camera features to FiLM (gamma, beta) pairs.

    Input: rot6d(6) + t_norm(3) + fl_norm(2) + pp_norm(2) = 13 dims.
    Shared MLP backbone; two zero-init heads for source (t1/t3) and target (t2).
    """

    INPUT_DIM = 13

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

    def source(self, cam_feat: torch.Tensor):
        """cam_feat: [B, V, 13] → gamma, beta each [B, V, source_dim]"""
        h = self.shared_mlp(cam_feat)
        return self.source_head(h).chunk(2, dim=-1)

    def target(self, cam_feat: torch.Tensor):
        """cam_feat: [B, V, 13] → gamma, beta each [B, V, target_dim]"""
        h = self.shared_mlp(cam_feat)
        return self.target_head(h).chunk(2, dim=-1)


class AdditiveCameraConditioner(nn.Module):
    """Maps 13-dim camera features to per-view additive offsets.

    Produces direct additive offsets: feats += offset[:, :, None, :].
    Separate zero-init source/target projections → identity at training start.
    """

    INPUT_DIM = 13

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.source_proj = nn.Linear(hidden_dim, dim)
        self.target_proj = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.source_proj.weight)
        nn.init.zeros_(self.source_proj.bias)
        nn.init.zeros_(self.target_proj.weight)
        nn.init.zeros_(self.target_proj.bias)

    def source(self, cam_feat: torch.Tensor) -> torch.Tensor:
        """cam_feat: [B, V, 13] → [B, V, dim]"""
        return self.source_proj(self.backbone(cam_feat))

    def target(self, cam_feat: torch.Tensor) -> torch.Tensor:
        """cam_feat: [B, V, 13] → [B, V, dim]"""
        return self.target_proj(self.backbone(cam_feat))


# ── Time cross block ──────────────────────────────────────────────────────────

class TimeCrossBlock(nn.Module):
    """Pre-norm cross-attention block: updates t2 query tokens from t1+t3 context.

    Residual update applied internally:
        q_out = q + cross_attn(norm_q(q), norm_kv(kv))
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.attn_dropout = dropout

        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj   = nn.Linear(dim, dim)
        self.k_proj   = nn.Linear(dim, dim)
        self.v_proj   = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:  [B, Q*T, C]   t2 query tokens
            kv: [B, 2S*T, C]  t1+t3 memory tokens
        Returns:
            [B, Q*T, C]  residual-updated t2 query tokens
        """
        B, Sq, C = q.shape
        Skv = kv.shape[1]
        H, D = self.num_heads, self.head_dim

        q_n  = self.norm_q(q)
        kv_n = self.norm_kv(kv)

        q_h = self.q_proj(q_n).view(B, Sq,  H, D).transpose(1, 2)   # [B, H, Sq,  D]
        k   = self.k_proj(kv_n).view(B, Skv, H, D).transpose(1, 2)  # [B, H, Skv, D]
        v   = self.v_proj(kv_n).view(B, Skv, H, D).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q_h, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # [B, H, Sq, D]
        attn_out = self.out_proj(out.transpose(1, 2).reshape(B, Sq, C))
        return q + attn_out


# ── Main model ────────────────────────────────────────────────────────────────

_DEFAULT_CACHE_LAYERS = [4, 11, 17, 23]


class TemporalVGGTv5(nn.Module):
    """Full VGGT aggregator with single TimeCrossBlock at final layer.

    Args:
        vggt_model_id:            HuggingFace model id or local path for VGGT.
        embed_dim:                Token dimension C (1024 for VGGT-1B).
        num_query_views:          Stored for dataset config; model infers Q from batch.
        time_conditioning_mode:   "film" | "residual_adaln".
        camera_conditioning_mode: "none" | "film" | "residual_adaln" | "additive".
        time_hidden_dim:          Hidden dim for SharedTimeEncoder (default: embed_dim).
        camera_hidden_dim:        Hidden dim for camera conditioner backbone.
        time_block_layer:         Aggregator layer index where TimeCrossBlock is injected.
        time_block_heads:         Attention heads in TimeCrossBlock.
        time_block_dropout:       Dropout inside TimeCrossBlock attention.
        cache_layers:             4 aggregator layer indices to cache for DPTHead.
        init_point_head_from_vggt: If True, copy pretrained VGGT DPTHead weights.
    """

    def __init__(
        self,
        vggt_model_id: str = "facebook/VGGT-1B",
        embed_dim: int = 1024,
        num_query_views: int = 1,
        time_conditioning_mode: str = "residual_adaln",
        camera_conditioning_mode: str = "none",
        time_hidden_dim: Optional[int] = None,
        camera_hidden_dim: int = 256,
        time_block_layer: int = 23,
        time_block_heads: int = 8,
        time_block_dropout: float = 0.0,
        cache_layers: Optional[List[int]] = None,
        init_point_head_from_vggt: bool = True,
    ):
        super().__init__()
        assert time_conditioning_mode in ("film", "residual_adaln"), time_conditioning_mode
        assert camera_conditioning_mode in ("none", "film", "residual_adaln", "additive"), \
            camera_conditioning_mode

        if cache_layers is None:
            cache_layers = _DEFAULT_CACHE_LAYERS
        assert len(cache_layers) == 4, "DPTHead requires exactly 4 cache layers"
        assert time_block_layer in cache_layers, \
            f"time_block_layer={time_block_layer} must be in cache_layers={cache_layers}"

        self.embed_dim = embed_dim
        self.num_query_views = num_query_views
        self.time_conditioning_mode = time_conditioning_mode
        self.camera_conditioning_mode = camera_conditioning_mode
        self.time_block_layer = time_block_layer
        self.cache_layers = cache_layers
        self.cache_set = set(cache_layers)

        # --- Load VGGT, freeze aggregator ---
        logger.info(f"Loading VGGT from {vggt_model_id!r}")
        vggt = VGGT.from_pretrained(vggt_model_id)
        self.aggregator = vggt.aggregator
        self.patch_start_idx = self.aggregator.patch_start_idx  # 5 = 1 camera + 4 register
        for p in self.aggregator.parameters():
            p.requires_grad_(False)

        # --- Learnable mask patch token: represents "unknown t2 patch" ---
        self.mask_patch_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # --- Time encoder ---
        _time_h = time_hidden_dim if time_hidden_dim is not None else embed_dim
        self.time_encoder = SharedTimeEncoder(
            source_dim=embed_dim,
            target_dim=embed_dim,
            hidden_dim=_time_h,
        )
        if time_conditioning_mode == "residual_adaln":
            self.source_time_adaln = ResidualAdaLN(embed_dim)
            self.target_time_adaln = ResidualAdaLN(embed_dim)
        else:
            self.source_time_adaln = None
            self.target_time_adaln = None

        # --- Camera conditioner (optional) ---
        self.cam_conditioner  = None
        self.cam_additive     = None
        self.source_cam_adaln = None
        self.target_cam_adaln = None

        if camera_conditioning_mode in ("film", "residual_adaln"):
            self.cam_conditioner = CameraConditioner(
                source_dim=embed_dim,
                target_dim=embed_dim,
                hidden_dim=camera_hidden_dim,
            )
            if camera_conditioning_mode == "residual_adaln":
                self.source_cam_adaln = ResidualAdaLN(embed_dim)
                self.target_cam_adaln = ResidualAdaLN(embed_dim)
        elif camera_conditioning_mode == "additive":
            self.cam_additive = AdditiveCameraConditioner(
                dim=embed_dim,
                hidden_dim=camera_hidden_dim,
            )

        # --- Time cross block (injected at time_block_layer) ---
        self.time_cross_block = TimeCrossBlock(
            dim=embed_dim,
            num_heads=time_block_heads,
            dropout=time_block_dropout,
        )

        # --- Point head (DPTHead, optionally initialized from pretrained VGGT) ---
        if init_point_head_from_vggt and vggt.point_head is not None:
            self.point_head = copy.deepcopy(vggt.point_head)
            # DPTHead will receive a 4-element list; use sequential indices.
            self.point_head.intermediate_layer_idx = [0, 1, 2, 3]
        else:
            self.point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )

        del vggt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cond(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        adaln: Optional[ResidualAdaLN],
    ) -> torch.Tensor:
        """Apply FiLM or ResidualAdaLN conditioning; adaln=None → FiLM."""
        if adaln is not None:
            return adaln(x, gamma, beta)
        return apply_film(x, gamma, beta)

    @staticmethod
    def _to_device(d: dict, device) -> dict:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    def _patch_embed(
        self,
        images: torch.Tensor,  # [B*V, 3, H, W] (already normalised)
    ) -> torch.Tensor:
        """Run frozen VGGT patch_embed; handles both dict and tensor outputs."""
        with torch.no_grad():
            out = self.aggregator.patch_embed(images)
        if isinstance(out, dict):
            out = out["x_norm_patchtokens"]
        return out  # [B*V, P_patch, C]

    # ------------------------------------------------------------------
    # Aggregator loop
    # ------------------------------------------------------------------

    def _run_aggregator_v5(
        self,
        patch_tokens: torch.Tensor,  # [B*S_total, P_patch, C]; order: [t1 | t2q | t3]
        B: int,
        S: int,   # endpoint views per side (V1 == V3)
        Q: int,   # t2 query views
        H_img: int,
        W_img: int,
    ) -> List[torch.Tensor]:
        """Run VGGT aggregator block-by-block, injecting TimeCrossBlock at time_block_layer.

        Returns:
            List of len(cache_layers) tensors, each [B, Q, P_total, 2C].
            P_total = patch_start_idx + P_patch.  Order matches cache_layers (ascending).
        """
        S_total      = 2 * S + Q
        idx_t2_start = S
        idx_t2_end   = S + Q

        agg     = self.aggregator
        patch_h = H_img // agg.patch_size
        patch_w = W_img // agg.patch_size
        _, P_patch, C = patch_tokens.shape

        # Prepend frozen special tokens: [camera(1) | register(4) | patch(P_patch)]
        camera_tok   = slice_expand_and_flatten(agg.camera_token,   B, S_total)  # [B*S_total, 1, C]
        register_tok = slice_expand_and_flatten(agg.register_token, B, S_total)  # [B*S_total, 4, C]
        tokens = torch.cat([camera_tok, register_tok, patch_tokens], dim=1)      # [B*S_total, P_total, C]
        _, P_total, _ = tokens.shape

        # RoPE positions: zero for special tokens, 1-indexed for patch tokens
        pos = None
        if agg.rope is not None:
            pos = agg.position_getter(B * S_total, patch_h, patch_w, device=tokens.device)
            pos = pos + 1
            pos_special = torch.zeros(
                B * S_total, agg.patch_start_idx, 2,
                device=tokens.device, dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)  # [B*S_total, P_total, 2]

        frame_idx  = 0
        global_idx = 0
        cached_t2: List[torch.Tensor] = []

        for layer_count in range(agg.aa_block_num):
            tokens, frame_idx,  frame_ints  = agg._process_frame_attention(
                tokens, B, S_total, P_total, C, frame_idx,  pos=pos
            )
            tokens, global_idx, global_ints = agg._process_global_attention(
                tokens, B, S_total, P_total, C, global_idx, pos=pos
            )

            # aa_block_size=1 → exactly one intermediate per call
            fi = frame_ints[0]   # [B, S_total, P_total, C]
            gi = global_ints[0]  # [B, S_total, P_total, C]
            del frame_ints, global_ints

            # ── Inject TimeCrossBlock (updates t2 portion of global output only) ──
            if layer_count == self.time_block_layer:
                q_time = gi[:, idx_t2_start:idx_t2_end].reshape(B, Q * P_total, C)
                kv_time = torch.cat([
                    gi[:, :idx_t2_start].reshape(B, S * P_total, C),  # t1
                    gi[:, idx_t2_end:  ].reshape(B, S * P_total, C),  # t3
                ], dim=1)                                               # [B, 2S*P_total, C]

                q_updated = self.time_cross_block(q_time, kv_time)     # [B, Q*P_total, C]
                del q_time, kv_time

                gi = torch.cat([
                    gi[:, :idx_t2_start],
                    q_updated.reshape(B, Q, P_total, C),
                    gi[:, idx_t2_end:],
                ], dim=1)                                               # [B, S_total, P_total, C]
                tokens = gi.reshape(B * S_total, P_total, C)           # update running state

            # ── Cache t2 query frames at specified layers ──
            if layer_count in self.cache_set:
                fi_t2 = fi[:, idx_t2_start:idx_t2_end]  # [B, Q, P_total, C]
                gi_t2 = gi[:, idx_t2_start:idx_t2_end]  # [B, Q, P_total, C]
                cached_t2.append(torch.cat([fi_t2, gi_t2], dim=-1))  # [B, Q, P_total, 2C]
                del fi_t2, gi_t2

            # Free full [B, S_total, P_total, C] tensors; only the Q-slice is cached above.
            del fi, gi

        return cached_t2  # 4 × [B, Q, P_total, 2C]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch (dict):
                images_t1:       [B, V1, 3, H, W]
                images_t3:       [B, V3, 3, H, W]
                date_t1/t2/t3:   [B] day-of-year (float)
                t1_day/t2_day/t3_day: [B] ordinal day
                rel_gap_feat:    [B, 5]  (optional, computed on-the-fly otherwise)
                camera_t1/t3/t2_query: dicts  (required when camera_conditioning_mode != "none")
        Returns:
            pred_points: [B, Q, H_out, W_out, 3]
            pred_conf:   [B, Q, H_out, W_out]
        """
        device = next(self.parameters()).device

        images_t1 = batch["images_t1"].to(device)  # [B, V1, 3, H, W]
        images_t3 = batch["images_t3"].to(device)  # [B, V3, 3, H, W]
        B, V1, _, H, W = images_t1.shape
        V3 = images_t3.shape[1]
        assert V1 == V3, f"v5 requires symmetric endpoint views, got V1={V1} V3={V3}"
        S = V1
        Q = S  # query views = endpoint views (num_query_views in dataset config)

        # --- Temporal gap features and dates ---
        if "rel_gap_feat" in batch:
            rel_gap = batch["rel_gap_feat"].to(device)
        else:
            rel_gap = build_relative_gap_features(
                batch["t1_day"], batch["t2_day"], batch["t3_day"]
            ).to(device)
        date_t1 = batch["date_t1"].to(device)
        date_t2 = batch["date_t2"].to(device)
        date_t3 = batch["date_t3"].to(device)

        role_t1 = torch.zeros(B,      dtype=torch.long, device=device)
        role_t3 = torch.ones(B,       dtype=torch.long, device=device)
        role_t2 = torch.full((B,), 2, dtype=torch.long, device=device)

        # --- Camera features (built once if any camera conditioning) ---
        cam_feat_t1 = cam_feat_t3 = cam_feat_t2q = None
        if self.cam_conditioner is not None or self.cam_additive is not None:
            cam_feat_t1  = build_camera_features(**self._to_device(batch["camera_t1"],       device))  # [B, V1, 13]
            cam_feat_t3  = build_camera_features(**self._to_device(batch["camera_t3"],       device))  # [B, V3, 13]
            cam_feat_t2q = build_camera_features(**self._to_device(batch["camera_t2_query"], device))  # [B, Q,  13]

        # --- Patch embed t1/t3 (frozen; use precomputed if provided) ---
        agg = self.aggregator
        if "patch_t1" in batch and "patch_t3" in batch:
            patch_t1 = batch["patch_t1"].to(device)  # [B, V1, P_patch, C]
            patch_t3 = batch["patch_t3"].to(device)
        else:
            mean, std = agg._resnet_mean, agg._resnet_std
            raw_t1 = self._patch_embed(((images_t1 - mean) / std).view(B * V1, 3, H, W))
            raw_t3 = self._patch_embed(((images_t3 - mean) / std).view(B * V3, 3, H, W))
            patch_t1 = raw_t1.view(B, V1, raw_t1.shape[1], raw_t1.shape[2])
            patch_t3 = raw_t3.view(B, V3, raw_t3.shape[1], raw_t3.shape[2])

        del images_t1, images_t3
        P_patch = patch_t1.shape[2]
        C       = patch_t1.shape[3]

        # --- t2 query tokens: broadcast mask_patch_token ---
        patch_t2q = self.mask_patch_token.expand(B, Q, P_patch, C)  # [B, Q, P_patch, C]

        # --- Camera conditioning on patch tokens (applied before aggregator) ---
        if self.cam_conditioner is not None:
            g1, b1 = self.cam_conditioner.source(cam_feat_t1)   # [B, V1, C]
            g3, b3 = self.cam_conditioner.source(cam_feat_t3)   # [B, V3, C]
            gq, bq = self.cam_conditioner.target(cam_feat_t2q)  # [B, Q,  C]
            patch_t1  = self._cond(patch_t1,  g1, b1, self.source_cam_adaln)
            patch_t3  = self._cond(patch_t3,  g3, b3, self.source_cam_adaln)
            patch_t2q = self._cond(patch_t2q, gq, bq, self.target_cam_adaln)
        elif self.cam_additive is not None:
            patch_t1  = patch_t1  + self.cam_additive.source(cam_feat_t1).unsqueeze(2)
            patch_t3  = patch_t3  + self.cam_additive.source(cam_feat_t3).unsqueeze(2)
            patch_t2q = patch_t2q + self.cam_additive.target(cam_feat_t2q).unsqueeze(2)

        # --- Time conditioning on patch tokens (applied before aggregator) ---
        g_t1, b_t1 = self.time_encoder.source(role_t1, date_t1, rel_gap)  # [B, C]
        g_t3, b_t3 = self.time_encoder.source(role_t3, date_t3, rel_gap)
        g_t2, b_t2 = self.time_encoder.target(role_t2, date_t2, rel_gap)

        patch_t1  = self._cond(patch_t1,  g_t1, b_t1, self.source_time_adaln)
        patch_t3  = self._cond(patch_t3,  g_t3, b_t3, self.source_time_adaln)
        patch_t2q = self._cond(patch_t2q, g_t2, b_t2, self.target_time_adaln)

        # --- Concatenate: [t1 | t2q | t3] and flatten for aggregator ---
        # Order matches pseudo-code: t2q sits between t1 and t3 (index S..S+Q-1)
        all_patches = torch.cat([patch_t1, patch_t2q, patch_t3], dim=1)  # [B, 2S+Q, P, C]
        del patch_t1, patch_t2q, patch_t3
        S_total = 2 * S + Q
        all_patches_flat = all_patches.view(B * S_total, P_patch, C)
        del all_patches

        # --- Run aggregator with TimeCrossBlock injection at time_block_layer ---
        cached_t2 = self._run_aggregator_v5(all_patches_flat, B, S, Q, H, W)
        # cached_t2: 4 × [B, Q, P_total, 2C]

        # --- Decode with pretrained DPTHead ---
        mock_images = torch.zeros(B, Q, 3, H, W, device=device, dtype=all_patches_flat.dtype)
        del all_patches_flat
        pred_points, pred_conf = self.point_head(
            cached_t2,
            images=mock_images,
            patch_start_idx=self.patch_start_idx,
        )
        # pred_points: [B, Q, H_out, W_out, 3]
        # pred_conf:   [B, Q, H_out, W_out]

        return {"pred_points": pred_points, "pred_conf": pred_conf}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def trainable_parameter_groups(self) -> List[Dict]:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

