"""Temporal-VGGT v4: Frozen VGGT endpoint features + masked-query transformer decoder.

Architecture (see docs/architectures/model_v4.py for the full pseudo-code):
  1. Run frozen VGGT independently on t1 and t3; extract the final global-attention
     patch features → F1 [B, V1, P, C], F3 [B, V3, P, C].
  2. Optionally camera-condition then time-condition F1 / F3 → memory M.
  3. Single learnable mask_token expands to t2 query grid Q2 [B, Q, P, C].
  4. Optionally camera-condition then time-condition Q2.
  5. N decoder blocks: RoPE self-attn + cross-attn(Q→M) + MLP.
  6. PointHead (class configurable via point_head_class) decodes Z2 to point map + conf.

Trainable: mask_token, time_encoder, [cam_conditioner], decoder, point_head.
Frozen:    VGGT aggregator + patch_embed.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.layers.attention import Attention
from vggt.layers.mlp import Mlp
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.heads.head_act import activate_head

from models.time_encoding import (
    SharedTimeEncoder,
    ResidualAdaLN,
    apply_film,
    build_relative_gap_features,
)
from models.camera_encoding import build_camera_features
from models.point_heads import get_point_head_class

logger = logging.getLogger(__name__)


# ── Camera conditioner ────────────────────────────────────────────────────────

class CameraConditioner(nn.Module):
    """Maps pre-built 13-dim camera features to FiLM (gamma, beta) pairs.

    Input: rot6d(6) + t_norm(3) + fl_norm(2) + pp_norm(2) = 13 dims,
    as produced by models.camera_encoding.build_camera_features.

    Shared MLP backbone with two zero-initialized projection heads:
      source_head → gamma, beta each [B, V, source_dim]  for t1/t3 patch features
      target_head → gamma, beta each [B, V, target_dim]  for t2 query tokens
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


# ── Additive camera conditioner ──────────────────────────────────────────────

class AdditiveCameraConditioner(nn.Module):
    """Maps pre-built 13-dim camera features to per-view additive offsets.

    Unlike CameraConditioner (FiLM gamma/beta), this adds a direct offset:
        feats += offset[:, :, None, :]   # broadcast over the patch dim

    Separate zero-initialized source/target projections → identity at training start.
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


# ── Cross-attention (bare: no internal norm or residual) ─────────────────────

class BareCrossAttention(nn.Module):
    """Cross-attention with SDPA. No internal LayerNorm or residual connection.

    Pre-norm and residual are the caller's responsibility (see DecoderBlock).
    Uses F.scaled_dot_product_attention so memory stays O(n) for long sequences.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.attn_dropout = dropout

        self.q_proj = nn.Linear(q_dim, d_model)
        self.k_proj = nn.Linear(kv_dim, d_model)
        self.v_proj = nn.Linear(kv_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:  [B, Sq, q_dim]
            kv: [B, Skv, kv_dim]
        Returns:
            [B, Sq, d_model]
        """
        B, Sq, _ = q.shape
        Skv = kv.shape[1]
        H, D = self.num_heads, self.head_dim

        q_h = self.q_proj(q).view(B, Sq,  H, D).transpose(1, 2)   # [B, H, Sq,  D]
        k   = self.k_proj(kv).view(B, Skv, H, D).transpose(1, 2)  # [B, H, Skv, D]
        v   = self.v_proj(kv).view(B, Skv, H, D).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q_h, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # [B, H, Sq, D]
        return self.out_proj(out.transpose(1, 2).reshape(B, Sq, self.d_model))


# ── Decoder block ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block.

    z = z + self_attn(norm1(z), rope_pos)    # RoPE self-attention
    z = z + cross_attn(norm2(z), M)          # cross-attention to memory, no RoPE
    z = z + mlp(norm3(z))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rope: Optional[RotaryPositionEmbedding2D] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, attn_drop=dropout, rope=rope)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = BareCrossAttention(dim, dim, dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(
        self,
        z: torch.Tensor,                          # [B, Sq, dim]
        m: torch.Tensor,                          # [B, Sm, dim]
        rope_pos: Optional[torch.Tensor] = None,  # [B, Sq, 2]
    ) -> torch.Tensor:
        z = z + self.self_attn(self.norm1(z), pos=rope_pos)
        z = z + self.cross_attn(self.norm2(z), m)
        z = z + self.mlp(self.norm3(z))
        return z


# ── Main model ────────────────────────────────────────────────────────────────

class TemporalVGGTv4(nn.Module):
    """Masked target-query temporal decoder for t2 point-map prediction.

    Args:
        vggt_model_id:            HuggingFace model id or local path for VGGT.
        embed_dim:                Token dimension C (1024 for VGGT-1B).
        num_query_views:          Passed through to dataset config; ignored by model.
        num_decoder_blocks:       Number of decoder blocks N.
        decoder_heads:            Attention heads in each decoder block.
        decoder_mlp_ratio:        MLP expansion ratio in decoder blocks.
        decoder_dropout:          Dropout probability inside decoder attention.
        time_conditioning_mode:   "film" | "residual_adaln".
        camera_conditioning_mode: "none" | "film" | "residual_adaln".
        time_hidden_dim:          Hidden dim for SharedTimeEncoder (default: embed_dim).
        camera_hidden_dim:        Hidden dim for CameraConditioner backbone.
        point_head_class:         Name of PointHead class from models.point_heads.
        point_head_kwargs:        Extra kwargs forwarded to the point head constructor.
    """

    def __init__(
        self,
        vggt_model_id: str = "facebook/VGGT-1B",
        embed_dim: int = 1024,
        num_query_views: int = 1,      # dataset config only; not used in forward
        num_decoder_blocks: int = 4,
        decoder_heads: int = 8,
        decoder_mlp_ratio: float = 4.0,
        decoder_dropout: float = 0.0,
        time_conditioning_mode: str = "film",
        camera_conditioning_mode: str = "none",
        time_hidden_dim: Optional[int] = None,
        camera_hidden_dim: int = 256,
        point_head_class: str = "PointHeadSmall",
        point_head_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        assert time_conditioning_mode in ("film", "residual_adaln"), time_conditioning_mode
        assert camera_conditioning_mode in ("none", "film", "residual_adaln", "additive"), camera_conditioning_mode

        self.embed_dim = embed_dim
        self.time_conditioning_mode = time_conditioning_mode
        self.camera_conditioning_mode = camera_conditioning_mode

        # --- Frozen VGGT ---
        logger.info(f"Loading VGGT from {vggt_model_id!r}")
        vggt = VGGT.from_pretrained(vggt_model_id)
        self.aggregator = vggt.aggregator
        self.patch_start_idx = self.aggregator.patch_start_idx
        assert self.aggregator.aa_order[-1] == "global", (
            "v4 requires aggregator aa_order ending with 'global'"
        )
        for p in self.aggregator.parameters():
            p.requires_grad_(False)
        del vggt

        # --- Learnable mask token: represents "unknown t2 patch" ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # --- Time encoder: source_head (→C) for t1/t3; target_head (→C) for t2 ---
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
        self.cam_conditioner = None
        self.cam_additive    = None
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

        # --- RoPE shared across all decoder self-attention layers ---
        self.rope = RotaryPositionEmbedding2D(frequency=100.0)
        self.position_getter = PositionGetter()

        # --- Decoder ---
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim,
                num_heads=decoder_heads,
                mlp_ratio=decoder_mlp_ratio,
                dropout=decoder_dropout,
                rope=self.rope,
            )
            for _ in range(num_decoder_blocks)
        ])

        # --- Point head ---
        head_cls = get_point_head_class(point_head_class)
        self.point_head = head_cls(
            dim_in=embed_dim,
            patch_size=self.aggregator.patch_size,
            **(point_head_kwargs or {}),
        )

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

    def _extract_endpoint_features(
        self,
        patch_tokens: torch.Tensor,  # [B*S, P_patch, C]
        B: int,
        S: int,
        H: int,
        W: int,
    ) -> torch.Tensor:  # [B, S, P_patch, C]
        """Run frozen aggregator; return final global-attention patch-only features."""
        agg = self.aggregator
        patch_h = H // agg.patch_size
        patch_w = W // agg.patch_size
        _, P_patch, C = patch_tokens.shape

        camera_tok  = slice_expand_and_flatten(agg.camera_token,   B, S)
        register_tok = slice_expand_and_flatten(agg.register_token, B, S)
        tokens = torch.cat([camera_tok, register_tok, patch_tokens], dim=1)
        _, P_total, _ = tokens.shape

        pos = None
        if agg.rope is not None:
            pos = agg.position_getter(B * S, patch_h, patch_w, device=tokens.device)
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, agg.patch_start_idx, 2,
                device=tokens.device, dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)

        frame_idx = global_idx = 0
        global_intermediates = None

        for _ in range(agg.aa_block_num):
            for attn_type in agg.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, _ = agg._process_frame_attention(
                        tokens, B, S, P_total, C, frame_idx, pos=pos
                    )
                else:
                    tokens, global_idx, global_intermediates = agg._process_global_attention(
                        tokens, B, S, P_total, C, global_idx, pos=pos
                    )

        # global_intermediates[-1]: [B, S, P_total, C]; slice out patch tokens
        return global_intermediates[-1][:, :, self.patch_start_idx:, :]  # [B, S, P_patch, C]

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
                patch_t1:        [B, V1, P, C]  optional precomputed
                patch_t3:        [B, V3, P, C]  optional precomputed
                camera_t1/t3/t2_query: dicts   required when camera_conditioning_mode != "none"
        Returns:
            pred_points: [B, Q, H, W, 3]
            pred_conf:   [B, Q, H, W]
        """
        device = next(self.parameters()).device

        images_t1 = batch["images_t1"]
        images_t3 = batch["images_t3"]
        B, V1, _, H, W = images_t1.shape
        V3 = images_t3.shape[1]
        Q  = V1
        patch_h = H // self.aggregator.patch_size
        patch_w = W // self.aggregator.patch_size

        # --- 13-dim camera features (built once, before patch embed) ---
        cam_feat_t1 = cam_feat_t3 = cam_feat_t2q = None
        if self.cam_conditioner is not None or self.cam_additive is not None:
            cam_feat_t1  = build_camera_features(**self._to_device(batch["camera_t1"],       device))  # [B, V1, 13]
            cam_feat_t3  = build_camera_features(**self._to_device(batch["camera_t3"],       device))  # [B, V3, 13]
            cam_feat_t2q = build_camera_features(**self._to_device(batch["camera_t2_query"], device))  # [B, Q,  13]

        # --- Patch embed (frozen) ---
        agg = self.aggregator
        if "patch_t1" in batch and "patch_t3" in batch:
            patch_t1 = batch["patch_t1"].to(device)
            patch_t3 = batch["patch_t3"].to(device)
            del images_t1, images_t3
        else:
            images_t1 = images_t1.to(device)
            images_t3 = images_t3.to(device)
            mean, std = agg._resnet_mean, agg._resnet_std
            with torch.no_grad():
                raw_t1 = agg.patch_embed(((images_t1 - mean) / std).view(B * V1, 3, H, W))
                raw_t3 = agg.patch_embed(((images_t3 - mean) / std).view(B * V3, 3, H, W))
            if isinstance(raw_t1, dict): raw_t1 = raw_t1["x_norm_patchtokens"]
            if isinstance(raw_t3, dict): raw_t3 = raw_t3["x_norm_patchtokens"]
            patch_t1 = raw_t1.view(B, V1, raw_t1.shape[1], raw_t1.shape[2])
            patch_t3 = raw_t3.view(B, V3, raw_t3.shape[1], raw_t3.shape[2])
            del images_t1, images_t3

        P_patch = patch_t1.shape[2]
        C       = patch_t1.shape[3]

        # --- Run frozen aggregator: final global features ---
        with torch.no_grad():
            F1 = self._extract_endpoint_features(patch_t1.view(B * V1, P_patch, C), B, V1, H, W)  # [B, V1, P, C]
            F3 = self._extract_endpoint_features(patch_t3.view(B * V3, P_patch, C), B, V3, H, W)  # [B, V3, P, C]
        del patch_t1, patch_t3

        # --- Camera conditioning on memory ---
        if self.cam_conditioner is not None:
            g1, b1 = self.cam_conditioner.source(cam_feat_t1)  # [B, V1, C]
            g3, b3 = self.cam_conditioner.source(cam_feat_t3)  # [B, V3, C]
            F1 = self._cond(F1, g1, b1, self.source_cam_adaln)
            F3 = self._cond(F3, g3, b3, self.source_cam_adaln)
        elif self.cam_additive is not None:
            F1 = F1 + self.cam_additive.source(cam_feat_t1).unsqueeze(2)  # [B, V1, 1, C] → broadcast
            F3 = F3 + self.cam_additive.source(cam_feat_t3).unsqueeze(2)  # [B, V3, 1, C] → broadcast

        # --- Relative gap features ---
        if "rel_gap_feat" in batch:
            rel_gap = batch["rel_gap_feat"].to(device)
        else:
            rel_gap = build_relative_gap_features(
                batch["t1_day"], batch["t2_day"], batch["t3_day"]
            ).to(device)

        date_t1 = batch["date_t1"].to(device)
        date_t2 = batch["date_t2"].to(device)
        date_t3 = batch["date_t3"].to(device)
        role_t1 = torch.zeros(B,       dtype=torch.long, device=device)
        role_t3 = torch.ones(B,        dtype=torch.long, device=device)
        role_t2 = torch.full((B,), 2,  dtype=torch.long, device=device)

        # --- Time conditioning on memory ---
        g_t1, b_t1 = self.time_encoder.source(role_t1, date_t1, rel_gap)
        g_t3, b_t3 = self.time_encoder.source(role_t3, date_t3, rel_gap)
        F1 = self._cond(F1, g_t1, b_t1, self.source_time_adaln)
        F3 = self._cond(F3, g_t3, b_t3, self.source_time_adaln)

        # --- t2 query: mask token + conditioning ---
        Q2 = self.mask_token.expand(B, Q, P_patch, C)  # [B, Q, P, C]

        if self.cam_conditioner is not None:
            gq, bq = self.cam_conditioner.target(cam_feat_t2q)  # [B, Q, C]
            Q2 = self._cond(Q2, gq, bq, self.target_cam_adaln)
        elif self.cam_additive is not None:
            Q2 = Q2 + self.cam_additive.target(cam_feat_t2q).unsqueeze(2)  # [B, Q, 1, C] → broadcast

        g_t2, b_t2 = self.time_encoder.target(role_t2, date_t2, rel_gap)
        Q2 = self._cond(Q2, g_t2, b_t2, self.target_time_adaln)

        # --- Build memory M and query Z ---
        M = torch.cat([
            F1.reshape(B, V1 * P_patch, C),
            F3.reshape(B, V3 * P_patch, C),
        ], dim=1)                               # [B, (V1+V3)*P, C]
        Z = Q2.reshape(B, Q * P_patch, C)       # [B, Q*P, C]
        del F1, F3, Q2

        # --- RoPE positions for decoder self-attention ---
        # Same patch-grid positions tiled across Q views, consistent with
        # VGGT's global-attention position encoding (views share the same grid).
        rope_pos = self.position_getter(1, patch_h, patch_w, device=device)  # [1, P, 2]
        rope_pos = (rope_pos + 1).expand(B * Q, -1, -1).reshape(B, Q * P_patch, 2)

        # --- Transformer decoder ---
        for block in self.decoder:
            Z = block(Z, M, rope_pos)

        # --- Reshape Z to token grid and decode ---
        # Z: [B, Q*P, C] → [B*Q, C, Hp, Wp]
        Z2_grid = (
            Z.reshape(B * Q, P_patch, C)
             .permute(0, 2, 1)
             .reshape(B * Q, C, patch_h, patch_w)
        )
        raw = self.point_head(Z2_grid)  # [B*Q, 4, H, W]

        pts3d, conf = activate_head(raw, activation="inv_log", conf_activation="expp1")
        # pts3d: [B*Q, H, W, 3],  conf: [B*Q, H, W]

        return {
            "pred_points": pts3d.view(B, Q, *pts3d.shape[1:]),  # [B, Q, H, W, 3]
            "pred_conf":   conf.view(B, Q, *conf.shape[1:]),    # [B, Q, H, W]
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def trainable_parameter_groups(self) -> List[Dict]:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

