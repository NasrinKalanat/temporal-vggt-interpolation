"""Temporal-VGGT v6: Date-local global attention + gated time self-attention blocks.

Architecture (see docs/architectures/model_v6.py for the full pseudo-code):
  1. Patch-embed t1/t3 with frozen VGGT patch_embed; build t2q from mask_patch_token.
  2. Prepend VGGT special tokens; concatenate [t1 | t2q | t3] → [B, 2S+Q, T, C].
  3. Per aggregator layer 0..23:
       a. Optional conditioning (camera + time) if layer in conditioning_before_layers.
       b. Frame attention on ALL 2S+Q views (frozen, same as original VGGT).
       c. Date-local global attention: the same global block runs SEPARATELY on each
          date group [t1, t2q, t3] — groups don't interact here.
       d. Optional conditioning immediately before each time block.
       e. If layer in time_block_layers: gated self-attention + gated MLP (TimeBlock)
          over the full (2S+Q)*T token sequence — all groups interact here.
       f. Cache concat(frame_out, layer_out) at cache_layers.
  4. Select t2 query frames; decode with pretrained VGGT DPTHead.

Key difference from v5:
  - v5 global: all 2S+Q views interact every layer → standard VGGT behaviour.
  - v6 global: each date group attends only within itself; cross-date mixing
    is handled exclusively by the configurable TimeBlock(s).

Trainable: mask_patch_token, time_encoder, [cam_conditioner/cam_additive],
           source/target_time_adaln, time_blocks (gated), point_head (fine-tuned).
Frozen:    VGGT aggregator (patch_embed, frame_blocks, global_blocks, special tokens).
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.layers.attention import Attention
from vggt.layers.mlp import Mlp
from vggt.heads.dpt_head import DPTHead

from models.time_encoding import (
    SharedTimeEncoder,
    ResidualAdaLN,
    apply_film,
    build_relative_gap_features,
)
from models.camera_encoding import build_camera_features

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_LAYERS   = [4, 11, 17, 23]
_DEFAULT_TIME_BLOCK_LAYERS = [23]
_DEFAULT_COND_BEFORE_LAYERS = [0]


# ── Camera conditioners (same as v4/v5) ──────────────────────────────────────

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
        return self.source_proj(self.backbone(cam_feat))

    def target(self, cam_feat: torch.Tensor) -> torch.Tensor:
        return self.target_proj(self.backbone(cam_feat))


# ── Time block ────────────────────────────────────────────────────────────────

class TimeBlock(nn.Module):
    """Gated self-attention + gated MLP over the full (2S+Q)*T token sequence.

    Both residual branches are scaled by learnable scalar gates initialized to 0,
    so the block is identity at the start of training.

    Uses the same VGGT Attention with RoPE: per-view 2D spatial positions are
    passed in so each token knows its (row, col) location within its view.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rope=None,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn      = Attention(dim, num_heads=num_heads, attn_drop=dropout, rope=rope)
        self.gate_attn = nn.Parameter(torch.zeros(1))

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp      = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.gate_mlp = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        """x: [B, (2S+Q)*T, C], pos: [B, (2S+Q)*T, 2] → [B, (2S+Q)*T, C]"""
        x = x + self.gate_attn * self.attn(self.norm_attn(x), pos=pos)
        x = x + self.gate_mlp  * self.mlp(self.norm_mlp(x))
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class TemporalVGGTv6(nn.Module):
    """Date-local global attention + configurable gated time self-attention.

    Args:
        vggt_model_id:                 HuggingFace model id or local path for VGGT.
        embed_dim:                     Token dimension C (1024 for VGGT-1B).
        num_query_views:               Stored for dataset config; model reads Q from batch.
        time_conditioning_mode:        "film" | "residual_adaln" | "additive".
        camera_conditioning_mode:      "none" | "film" | "residual_adaln" | "additive".
        time_hidden_dim:               Hidden dim for SharedTimeEncoder (default: embed_dim).
        camera_hidden_dim:             Hidden dim for camera conditioner backbone.
        time_block_layers:             Aggregator layer indices where TimeBlock is applied.
        time_block_heads:              Attention heads in each TimeBlock.
        time_block_mlp_ratio:          MLP expansion ratio in TimeBlock.
        time_block_dropout:            Dropout inside TimeBlock attention.
        conditioning_before_layers:    Apply camera+time conditioning before these layers.
        conditioning_before_time_blocks: Also condition immediately before each time block.
        cache_layers:                  4 layer indices to cache for DPTHead.
        init_point_head_from_vggt:     If True, copy pretrained VGGT DPTHead weights.
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
        time_block_layers: Optional[List[int]] = None,
        time_block_heads: int = 8,
        time_block_mlp_ratio: float = 4.0,
        time_block_dropout: float = 0.0,
        conditioning_before_layers: Optional[List[int]] = None,
        conditioning_before_time_blocks: bool = True,
        cache_layers: Optional[List[int]] = None,
        init_point_head_from_vggt: bool = True,
    ):
        super().__init__()
        assert time_conditioning_mode in ("film", "residual_adaln", "additive"), time_conditioning_mode
        assert camera_conditioning_mode in ("none", "film", "residual_adaln", "additive"), \
            camera_conditioning_mode

        if time_block_layers is None:
            time_block_layers = _DEFAULT_TIME_BLOCK_LAYERS
        if conditioning_before_layers is None:
            conditioning_before_layers = _DEFAULT_COND_BEFORE_LAYERS
        if cache_layers is None:
            cache_layers = _DEFAULT_CACHE_LAYERS

        assert len(cache_layers) == 4, "DPTHead requires exactly 4 cache layers"

        self.embed_dim   = embed_dim
        self.num_query_views = num_query_views
        self.time_conditioning_mode  = time_conditioning_mode
        self.camera_conditioning_mode = camera_conditioning_mode
        self.time_block_layers  = time_block_layers
        self.time_block_set: Set[int] = set(time_block_layers)
        self.cache_layers = cache_layers
        self.cache_set: Set[int] = set(cache_layers)
        self.conditioning_before_layers_set: Set[int] = set(conditioning_before_layers)
        self.conditioning_before_time_blocks = conditioning_before_time_blocks

        # --- Load VGGT, freeze aggregator ---
        logger.info(f"Loading VGGT from {vggt_model_id!r}")
        vggt = VGGT.from_pretrained(vggt_model_id)
        self.aggregator = vggt.aggregator
        self.patch_start_idx = self.aggregator.patch_start_idx  # 5 = 1 camera + 4 register
        for p in self.aggregator.parameters():
            p.requires_grad_(False)

        # --- Learnable mask patch token + Phase-2 reinjection gate ---
        self.mask_patch_token   = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.mask_reinject_gate = nn.Parameter(torch.full((1,), 1e-3))

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

        # --- Time blocks (one per entry in time_block_layers) ---
        self.time_blocks = nn.ModuleDict({
            str(layer): TimeBlock(
                dim=embed_dim,
                num_heads=time_block_heads,
                mlp_ratio=time_block_mlp_ratio,
                dropout=time_block_dropout,
                rope=self.aggregator.rope,
            )
            for layer in time_block_layers
        })

        # --- Point head (DPTHead, optionally initialized from pretrained VGGT) ---
        if init_point_head_from_vggt and vggt.point_head is not None:
            self.point_head = copy.deepcopy(vggt.point_head)
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
        if adaln is not None:
            return adaln(x, gamma, beta)
        return apply_film(x, gamma, beta)

    @staticmethod
    def _to_device(d: dict, device) -> dict:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    def _patch_embed(self, images: torch.Tensor) -> torch.Tensor:
        """Run frozen VGGT patch_embed on [B*V, 3, H, W] images (already normalised)."""
        with torch.no_grad():
            out = self.aggregator.patch_embed(images)
        if isinstance(out, dict):
            out = out["x_norm_patchtokens"]
        return out  # [B*V, P_patch, C]

    def _apply_conditioning(
        self,
        x: torch.Tensor,              # [B, S_total, T, C]
        S: int,
        Q: int,
        cam_feat_t1:  Optional[torch.Tensor],  # [B, S, 13]
        cam_feat_t2q: Optional[torch.Tensor],  # [B, Q, 13]
        cam_feat_t3:  Optional[torch.Tensor],  # [B, S, 13]
        role_t1:  torch.Tensor,  # [B]
        role_t2:  torch.Tensor,  # [B]
        role_t3:  torch.Tensor,  # [B]
        date_t1:  torch.Tensor,  # [B]
        date_t2:  torch.Tensor,  # [B]
        date_t3:  torch.Tensor,  # [B]
        rel_gap:  torch.Tensor,  # [B, 5]
    ) -> torch.Tensor:           # [B, S_total, T, C]
        """Apply camera+time conditioning to each date group in x."""
        x_t1  = x[:, :S]       # [B, S, T, C]
        x_t2q = x[:, S:S+Q]    # [B, Q, T, C]
        x_t3  = x[:, S+Q:]     # [B, S, T, C]

        # Camera conditioning
        if self.cam_conditioner is not None:
            g1, b1 = self.cam_conditioner.source(cam_feat_t1)   # [B, S, C]
            g3, b3 = self.cam_conditioner.source(cam_feat_t3)
            gq, bq = self.cam_conditioner.target(cam_feat_t2q)  # [B, Q, C]
            x_t1  = self._cond(x_t1,  g1, b1, self.source_cam_adaln)
            x_t3  = self._cond(x_t3,  g3, b3, self.source_cam_adaln)
            x_t2q = self._cond(x_t2q, gq, bq, self.target_cam_adaln)
        elif self.cam_additive is not None:
            x_t1  = x_t1  + self.cam_additive.source(cam_feat_t1).unsqueeze(2)
            x_t3  = x_t3  + self.cam_additive.source(cam_feat_t3).unsqueeze(2)
            x_t2q = x_t2q + self.cam_additive.target(cam_feat_t2q).unsqueeze(2)

        # Time conditioning
        if self.time_conditioning_mode == "additive":
            off_t1 = self.time_encoder.source_additive(role_t1, date_t1, rel_gap)  # [B, C]
            off_t3 = self.time_encoder.source_additive(role_t3, date_t3, rel_gap)
            off_t2 = self.time_encoder.target_additive(role_t2, date_t2, rel_gap)
            x_t1  = x_t1  + off_t1[:, None, None, :]
            x_t3  = x_t3  + off_t3[:, None, None, :]
            x_t2q = x_t2q + off_t2[:, None, None, :]
        else:
            g_t1, b_t1 = self.time_encoder.source(role_t1, date_t1, rel_gap)  # [B, C]
            g_t3, b_t3 = self.time_encoder.source(role_t3, date_t3, rel_gap)
            g_t2, b_t2 = self.time_encoder.target(role_t2, date_t2, rel_gap)
            x_t1  = self._cond(x_t1,  g_t1, b_t1, self.source_time_adaln)
            x_t3  = self._cond(x_t3,  g_t3, b_t3, self.source_time_adaln)
            x_t2q = self._cond(x_t2q, g_t2, b_t2, self.target_time_adaln)

        return torch.cat([x_t1, x_t2q, x_t3], dim=1)  # [B, S_total, T, C]

    # ------------------------------------------------------------------
    # Aggregator loop
    # ------------------------------------------------------------------

    def _run_aggregator_v6(
        self,
        patch_tokens: torch.Tensor,   # [B*S_total, P_patch, C]; order: [t1 | t2q | t3]
        B: int,
        S: int,    # endpoint views per side (V1 == V3)
        Q: int,    # t2 query views
        H_img: int,
        W_img: int,
        # conditioning context (forwarded from batch)
        cam_feat_t1:  Optional[torch.Tensor],
        cam_feat_t2q: Optional[torch.Tensor],
        cam_feat_t3:  Optional[torch.Tensor],
        role_t1:  torch.Tensor,
        role_t2:  torch.Tensor,
        role_t3:  torch.Tensor,
        date_t1:  torch.Tensor,
        date_t2:  torch.Tensor,
        date_t3:  torch.Tensor,
        rel_gap:  torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run VGGT aggregator with date-local global and injected TimeBlocks.

        Phase 1 (no_grad): run all 24 VGGT layers; save fi_t2 / gi_full at key layers.
        Phase 2 (with grad): reinject mask_patch_token into t2q patch tokens, apply
          conditioning + TimeBlock.  Gradient paths: mask_patch_token, mask_reinject_gate,
          time_encoder, TimeBlock, DPTHead.

        Returns:
            List of len(cache_layers) tensors, each [B, Q, T, 2C],
            ordered by ascending cache_layer index.
        """
        S_total = 2 * S + Q

        agg     = self.aggregator
        patch_h = H_img // agg.patch_size
        patch_w = W_img // agg.patch_size
        _, P_patch, C = patch_tokens.shape

        # Prepend frozen special tokens [camera | register | patch]
        # camera_tok / register_tok are frozen; patch_tokens may carry grad (mask_patch_token).
        with torch.no_grad():
            camera_tok   = slice_expand_and_flatten(agg.camera_token,   B, S_total)
            register_tok = slice_expand_and_flatten(agg.register_token, B, S_total)
        tokens_flat = torch.cat([camera_tok, register_tok, patch_tokens], dim=1)
        _, T, _ = tokens_flat.shape  # T = R + P_patch

        # RoPE positions (same grid for all views; zero for special tokens)
        pos      = None
        pos_flat = None  # [B, S_total*T, 2] — for TimeBlock full-sequence RoPE
        if agg.rope is not None:
            pos = agg.position_getter(B * S_total, patch_h, patch_w, device=tokens_flat.device)
            pos = pos + 1
            pos_sp = torch.zeros(
                B * S_total, agg.patch_start_idx, 2,
                device=tokens_flat.device, dtype=pos.dtype,
            )
            pos    = torch.cat([pos_sp, pos], dim=1)   # [B*S_total, T, 2]
            pos_4d = pos.view(B, S_total, T, 2)

            # Pre-compute per-group positions (constant across layers)
            pos_t1  = pos_4d[:, :S       ].reshape(B, S * T, 2)
            pos_t2q = pos_4d[:, S:S+Q    ].reshape(B, Q * T, 2)
            pos_t3  = pos_4d[:, S+Q:     ].reshape(B, S * T, 2)

            # Full-sequence positions for TimeBlock: each token keeps its per-view 2D pos
            pos_flat = pos_4d.reshape(B, S_total * T, 2)
        else:
            pos_t1 = pos_t2q = pos_t3 = None

        cond_args = (S, Q,
                     cam_feat_t1, cam_feat_t2q, cam_feat_t3,
                     role_t1, role_t2, role_t3,
                     date_t1, date_t2, date_t3, rel_gap)

        def _global_block(block, fi):
            """Date-local global attention, batched across date groups."""
            fi_t1  = fi[:, :S    ].reshape(B, S * T, C)
            fi_t2q = fi[:, S:S+Q ].reshape(B, Q * T, C)
            fi_t3  = fi[:, S+Q:  ].reshape(B, S * T, C)
            if S == Q:
                groups = torch.cat([fi_t1, fi_t2q, fi_t3], dim=0)
                pos_g  = torch.cat([pos_t1, pos_t2q, pos_t3], dim=0) if pos_t1 is not None else None
                out    = block(groups, pos=pos_g)
                g_t1   = out[:B    ].view(B, S, T, C)
                g_t2q  = out[B:2*B ].view(B, Q, T, C)
                g_t3   = out[2*B:  ].view(B, S, T, C)
            else:
                t1_t3  = torch.cat([fi_t1, fi_t3], dim=0)
                pos_g  = torch.cat([pos_t1, pos_t3], dim=0) if pos_t1 is not None else None
                out    = block(t1_t3, pos=pos_g)
                g_t1   = out[:B].view(B, S, T, C)
                g_t3   = out[B:].view(B, S, T, C)
                g_t2q  = block(fi_t2q, pos=pos_t2q).view(B, Q, T, C)
            return torch.cat([g_t1, g_t2q, g_t3], dim=1)  # [B, S_total, T, C]

        # ── Phase 1: full aggregator under no_grad (no activation storage) ──────
        saved_fi_t2:   Dict[int, torch.Tensor] = {}
        saved_gi_full: Dict[int, torch.Tensor] = {}
        saved_gi_t2:   Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            x = tokens_flat.view(B, S_total, T, C)
            frame_idx  = 0
            global_idx = 0

            for layer_count in range(agg.aa_block_num):

                if layer_count in self.conditioning_before_layers_set:
                    x = self._apply_conditioning(x, *cond_args)

                _, frame_idx, frame_ints = agg._process_frame_attention(
                    x, B, S_total, T, C, frame_idx, pos=pos
                )
                fi = frame_ints[0]

                if layer_count in self.cache_set:
                    saved_fi_t2[layer_count] = fi[:, S:S+Q].clone()

                gi = _global_block(agg.global_blocks[global_idx], fi)
                global_idx += 1

                if layer_count in self.time_block_set:
                    saved_gi_full[layer_count] = gi.clone()
                    gi_cond = self._apply_conditioning(gi, *cond_args) if self.conditioning_before_time_blocks else gi
                    x_time  = self.time_blocks[str(layer_count)](gi_cond.reshape(B, S_total * T, C), pos=pos_flat)
                    layer_out = x_time.view(B, S_total, T, C)
                else:
                    layer_out = gi
                    if layer_count in self.cache_set:
                        saved_gi_t2[layer_count] = gi[:, S:S+Q].clone()

                x = layer_out

        # ── Phase 2: re-apply trainable ops with gradient tracking ───────────────
        # mask_patch_token gradient path: reinjection → TimeBlock → DPTHead → loss.
        cached_t2: List[torch.Tensor] = []
        for layer_count in sorted(self.cache_set):
            fi_t2 = saved_fi_t2[layer_count]
            if layer_count in self.time_block_set:
                gi_full = saved_gi_full[layer_count]  # [B, S_total, T, C], detached

                # Reinject mask_patch_token into t2q patch-token slice (not special tokens).
                # gradient path: mask_patch_token, mask_reinject_gate → TimeBlock → loss.
                patch_offset = self.mask_reinject_gate * self.mask_patch_token  # [1, 1, 1, C]
                gi_t2q_new = torch.cat([
                    gi_full[:, S:S+Q, :self.patch_start_idx],           # special tokens unchanged
                    gi_full[:, S:S+Q,  self.patch_start_idx:] + patch_offset,  # patch tokens
                ], dim=2)
                gi_full = torch.cat([gi_full[:, :S], gi_t2q_new, gi_full[:, S+Q:]], dim=1)

                if self.conditioning_before_time_blocks:
                    gi_full = self._apply_conditioning(gi_full, *cond_args)
                x_time = self.time_blocks[str(layer_count)](gi_full.reshape(B, S_total * T, C), pos=pos_flat)
                lo_t2  = x_time.view(B, S_total, T, C)[:, S:S+Q]
            else:
                lo_t2 = saved_gi_t2[layer_count]
            cached_t2.append(torch.cat([fi_t2, lo_t2], dim=-1))

        return cached_t2  # 4 × [B, Q, T, 2C]

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
                rel_gap_feat:    [B, 5]  optional
                camera_t1/t3/t2_query: dicts  required when camera_conditioning_mode != "none"
        Returns:
            pred_points: [B, Q, H_out, W_out, 3]
            pred_conf:   [B, Q, H_out, W_out]
        """
        device = next(self.parameters()).device

        images_t1 = batch["images_t1"].to(device)
        images_t3 = batch["images_t3"].to(device)
        B, V1, _, H, W = images_t1.shape
        V3 = images_t3.shape[1]
        assert V1 == V3, f"v6 requires symmetric endpoint views, got V1={V1} V3={V3}"
        S = V1
        Q = self.num_query_views

        # --- Temporal gap features ---
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

        # --- Camera features ---
        cam_feat_t1 = cam_feat_t3 = cam_feat_t2q = None
        if self.cam_conditioner is not None or self.cam_additive is not None:
            cam_feat_t1  = build_camera_features(**self._to_device(batch["camera_t1"],       device))
            cam_feat_t3  = build_camera_features(**self._to_device(batch["camera_t3"],       device))
            cam_feat_t2q = build_camera_features(**self._to_device(batch["camera_t2_query"], device))

        # --- Patch embed t1/t3 (frozen) ---
        agg = self.aggregator
        if "patch_t1" in batch and "patch_t3" in batch:
            patch_t1 = batch["patch_t1"].to(device)
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

        # --- t2 query tokens: expand mask_patch_token ---
        patch_t2q = self.mask_patch_token.expand(B, Q, P_patch, C)  # [B, Q, P_patch, C]

        # --- Concatenate [t1 | t2q | t3] and flatten ---
        S_total = 2 * S + Q
        all_patches = torch.cat([patch_t1, patch_t2q, patch_t3], dim=1)  # [B, S_total, P, C]
        del patch_t1, patch_t2q, patch_t3
        all_patches_flat = all_patches.view(B * S_total, P_patch, C)
        del all_patches

        # --- Run aggregator with date-local global + time blocks ---
        cached_t2 = self._run_aggregator_v6(
            all_patches_flat, B, S, Q, H, W,
            cam_feat_t1, cam_feat_t2q, cam_feat_t3,
            role_t1, role_t2, role_t3,
            date_t1, date_t2, date_t3, rel_gap,
        )
        del all_patches_flat
        # cached_t2: 4 × [B, Q, T, 2C]

        # --- Decode with pretrained DPTHead ---
        mock_images = cached_t2[0].new_zeros(B, Q, 3, H, W)
        pred_points, pred_conf = self.point_head(
            cached_t2,
            images=mock_images,
            patch_start_idx=self.patch_start_idx,
        )

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

