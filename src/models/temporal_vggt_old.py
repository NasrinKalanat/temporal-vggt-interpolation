"""TemporalVGGT: predicts dense 3D geometry at t2 from t1+t3 multi-view images.

Architecture:
  1. Frozen VGGT aggregator encodes endpoint images → 2048D tokens per position
  2. Linear projection (2048→1024) + role embeddings (3 roles, 1024D)
  3. K=32 learned target view-slot queries (Q=256 tokens each, 1024D)
  4. Temporal decoder: deep copy of 24 VGGT global_blocks (1024D self-attention)
       lora_layers (default 18-23): frozen base + LoRA on attn.qkv/proj, mlp.fc1/fc2
       film_layers (default 20-23): LoRA + role-specific FiLM (subset of lora_layers)
  5. Time-FiLM controller: 11D time vector → γ/β [B, L_film, R, 1024]
  6. DPT-style geometry head: multi-scale decoder features → point maps + depths
       Intermediate decoder outputs at layers [4, 11, 17, 23] feed the DPT fusion.

Training:
  Trainable: token_proj, role_embed, target_queries, LoRA params, FiLM controller, geometry head
  Frozen:    VGGT aggregator, decoder base block weights (norms, attention, mlp)
"""
from __future__ import annotations

import copy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from models.base import TemporalGeometryPredictor
from vggt.heads.dpt_head import _make_scratch, _make_fusion_block

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around a frozen nn.Linear.

    forward(x) = W x + (B A x) * (alpha / rank)
    lora_B is zero-initialized → identity at init.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.linear = linear
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale


def _inject_lora(block, rank: int, alpha: float, dropout: float) -> None:
    """Replace attn.qkv, attn.proj, mlp.fc1, mlp.fc2 with LoRA versions.

    The original linear parameters are frozen before wrapping.
    """
    targets = [
        (block.attn, "qkv"),
        (block.attn, "proj"),
        (block.mlp, "fc1"),
        (block.mlp, "fc2"),
    ]
    for parent, attr in targets:
        linear = getattr(parent, attr)
        for p in linear.parameters():
            p.requires_grad_(False)
        setattr(parent, attr, LoRALinear(linear, rank, alpha, dropout))
        logger.debug("LoRA injected into block.%s.%s", parent.__class__.__name__, attr)


# ---------------------------------------------------------------------------
# FiLM helpers
# ---------------------------------------------------------------------------

def _apply_role_film(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    role_ids: torch.Tensor,
) -> torch.Tensor:
    """Role-specific FiLM: (1 + γ[role]) * x + β[role].

    x:        [B, S, C]
    gamma:    [B, R, C]
    beta:     [B, R, C]
    role_ids: [S] long, values in {0, 1, 2}
    """
    g = gamma[:, role_ids, :]   # [B, S, C]
    b = beta[:, role_ids, :]    # [B, S, C]
    return (1.0 + g) * x + b


# ---------------------------------------------------------------------------
# Decoder block wrappers
# ---------------------------------------------------------------------------

class _PlainBlock(nn.Module):
    """Frozen VGGT Block wrapper with uniform call signature."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(
        self, x: torch.Tensor, film_attn=None, film_mlp=None, role_ids=None
    ) -> torch.Tensor:
        return self.block(x, pos=None)


class _FiLMBlock(nn.Module):
    """VGGT Block with LoRA already injected, optional role-specific FiLM.

    Reimplements Block.forward for the pos=None, drop_path=0 case so that
    FiLM can be inserted between norm1→attn and norm2→mlp.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(
        self,
        x: torch.Tensor,
        film_attn=None,
        film_mlp=None,
        role_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = self.block
        # Attention branch
        u = b.norm1(x)
        if film_attn is not None and role_ids is not None:
            u = _apply_role_film(u, film_attn[0], film_attn[1], role_ids)
        x = x + b.ls1(b.attn(u, pos=None))
        # MLP branch
        v = b.norm2(x)
        if film_mlp is not None and role_ids is not None:
            v = _apply_role_film(v, film_mlp[0], film_mlp[1], role_ids)
        x = x + b.ls2(b.mlp(v))
        return x


# ---------------------------------------------------------------------------
# Temporal decoder
# ---------------------------------------------------------------------------

class TemporalDecoder(nn.Module):
    """24-block global self-attention decoder (reuses VGGT global_blocks).

    Parameters:
        lora_layers: Block indices (0-based) to inject LoRA, e.g. [18,19,20,21,22,23].
        film_layers: Subset of lora_layers also receiving FiLM, e.g. [20,21,22,23].
        intermediate_layer_idx: Block indices whose outputs are saved and returned
            for the DPT geometry head (4 values, default [4, 11, 17, 23]).
    """

    def __init__(
        self,
        source_blocks: nn.ModuleList,
        lora_layers: list[int],
        film_layers: list[int],
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        intermediate_layer_idx: tuple[int, ...] = (4, 11, 17, 23),
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self._lora_set = set(lora_layers)
        self._film_set = set(film_layers)
        self._film_layer_to_idx = {layer: i for i, layer in enumerate(sorted(film_layers))}
        self._intermediate_set = set(intermediate_layer_idx)

        wrapped: list[nn.Module] = []
        for i, src_block in enumerate(source_blocks):
            blk = copy.deepcopy(src_block)
            blk.attn.rope = None  # disable RoPE — no spatial grid in our sequence
            for p in blk.parameters():
                p.requires_grad_(False)  # freeze base weights

            if i in self._lora_set:
                _inject_lora(blk, lora_rank, lora_alpha, lora_dropout)
                wrapped.append(_FiLMBlock(blk))
            else:
                wrapped.append(_PlainBlock(blk))

        self.blocks = nn.ModuleList(wrapped)
        self.gradient_checkpointing = gradient_checkpointing
        logger.info(
            "TemporalDecoder: %d blocks, LoRA in %s, FiLM in %s, grad_ckpt=%s",
            len(wrapped), sorted(lora_layers), sorted(film_layers), gradient_checkpointing,
        )

    def forward(
        self,
        x: torch.Tensor,
        role_ids: torch.Tensor,
        film_params: dict | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """
        x:          [B, total_tokens, 1024]
        role_ids:   [total_tokens] long, values in {0,1,2}
        film_params: dict with gamma_attn, beta_attn, gamma_mlp, beta_mlp,
                     each [B, L_film, R, 1024].  None → no FiLM.

        Returns:
            (final_output [B, T, C],
             saved_outputs dict mapping layer_idx → [B, T, C] for DPT head)
        """
        saved: dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.blocks):
            use_ckpt = self.gradient_checkpointing and self.training
            if i in self._film_set and film_params is not None:
                fi = self._film_layer_to_idx[i]
                ga = film_params["gamma_attn"][:, fi]   # [B, R, C]
                ba = film_params["beta_attn"][:, fi]
                gm = film_params["gamma_mlp"][:, fi]
                bm = film_params["beta_mlp"][:, fi]
                if use_ckpt:
                    # Default-arg capture avoids late-binding in loop closure.
                    def _fn(x, ga, ba, gm, bm, _b=blk, _r=role_ids):
                        return _b(x, film_attn=(ga, ba), film_mlp=(gm, bm), role_ids=_r)
                    x = _ckpt(_fn, x, ga, ba, gm, bm, use_reentrant=False)
                else:
                    x = blk(x, film_attn=(ga, ba), film_mlp=(gm, bm), role_ids=role_ids)
            else:
                if use_ckpt:
                    x = _ckpt(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
            if i in self._intermediate_set:
                saved[i] = x
        return x, saved


# ---------------------------------------------------------------------------
# Time features and FiLM controller
# ---------------------------------------------------------------------------

def make_time_film_features(
    doy_t1: torch.Tensor,
    doy_t2: torch.Tensor,
    doy_t3: torch.Tensor,
    day_index_t1: torch.Tensor,
    day_index_t2: torch.Tensor,
    day_index_t3: torch.Tensor,
    gap_scale: float = 365.0,
) -> torch.Tensor:
    """Build 11D time feature vector from batch temporal fields.

    Returns [B, 11]:
      [sin/cos doy(t1), sin/cos doy(t2), sin/cos doy(t3),
       tau, 1-tau, left_gap_norm, right_gap_norm, total_gap_norm]
    """
    k = 2.0 * math.pi / 365.0
    sin1, cos1 = torch.sin(doy_t1 * k), torch.cos(doy_t1 * k)
    sin2, cos2 = torch.sin(doy_t2 * k), torch.cos(doy_t2 * k)
    sin3, cos3 = torch.sin(doy_t3 * k), torch.cos(doy_t3 * k)

    left_gap  = (day_index_t2 - day_index_t1).float()
    right_gap = (day_index_t3 - day_index_t2).float()
    total_gap = (day_index_t3 - day_index_t1).float().clamp(min=1.0)

    tau           = left_gap  / total_gap
    one_minus_tau = right_gap / total_gap
    left_norm     = left_gap  / gap_scale
    right_norm    = right_gap / gap_scale
    total_norm    = total_gap / gap_scale

    return torch.stack(
        [sin1, cos1, sin2, cos2, sin3, cos3, tau, one_minus_tau, left_norm, right_norm, total_norm],
        dim=-1,
    )  # [B, 11]


class TimeFiLMController(nn.Module):
    """Maps 11D time feature vector → layer-wise role-specific FiLM params.

    Output: dict with gamma_attn, beta_attn, gamma_mlp, beta_mlp,
            each [B, L_film, R, C].

    Final linear initialized to zero so γ=0, β=0 at init (identity FiLM).
    """

    def __init__(
        self,
        time_feat_dim: int,
        hidden_dim: int,
        num_film_layers: int,
        num_roles: int,
        token_dim: int,
    ):
        super().__init__()
        self.L = num_film_layers
        self.R = num_roles
        self.C = token_dim
        out_dim = 4 * num_film_layers * num_roles * token_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, u: torch.Tensor) -> dict[str, torch.Tensor]:
        """u: [B, 11]  →  dict each [B, L, R, C]."""
        B = u.shape[0]
        out = self.mlp(u).view(B, 4, self.L, self.R, self.C)
        return {
            "gamma_attn": out[:, 0],
            "beta_attn":  out[:, 1],
            "gamma_mlp":  out[:, 2],
            "beta_mlp":   out[:, 3],
        }


# ---------------------------------------------------------------------------
# Target view-slot query builder
# ---------------------------------------------------------------------------

class TargetViewSlotQueryBuilder(nn.Module):
    """Learned fixed target view-slot queries (no camera input)."""

    def __init__(self, num_target_views: int, query_grid_hw: tuple[int, int], token_dim: int):
        super().__init__()
        Hq, Wq = query_grid_hw
        self.num_query_tokens = Hq * Wq
        self.target_queries = nn.Parameter(
            torch.randn(1, num_target_views, Hq * Wq, token_dim) * 0.02
        )

    def forward(self, batch_size: int, device=None) -> torch.Tensor:
        return self.target_queries.to(device).expand(batch_size, -1, -1, -1)


# ---------------------------------------------------------------------------
# DPT-style geometry head
# ---------------------------------------------------------------------------

class TemporalDPTHead(nn.Module):
    """DPT-style geometry head for decoded target token sequences.

    Adapted from vggt.heads.dpt_head.DPTHead:
      - Extracts target tokens from saved intermediate decoder outputs at 4 layers.
      - Reshapes [B, K*Q, C] → [B*K, C, Hq, Wq] spatial grids.
      - Multi-scale DPT fusion (projects, resize, scratch fusion blocks).
      - Shared DPT backbone with separate point (4D) and depth (2D) output heads.

    Output resolutions (for Hq=Wq=16):
      scale 0 (finest, layer 4):   16×16 → 64×64   (ConvTranspose2d ×4)
      scale 1 (layer 11):          16×16 → 32×32   (ConvTranspose2d ×2)
      scale 2 (layer 17):          16×16 → 16×16   (Identity)
      scale 3 (coarsest, layer 23): 16×16 → 8×8    (Conv stride-2)
      DPT fusion:  8→16→32→64→128 (final ×2 from refinenet1)

    Args:
        token_dim:             Decoder token dimension (1024).
        num_target_views:      K = 32.
        query_grid_hw:         (Hq, Wq) = (16, 16).
        intermediate_layer_idx: 4 decoder layer indices for multi-scale DPT.
        features:              DPT internal channel count (256).
        out_channels:          Per-scale projection output channels.
    """

    def __init__(
        self,
        token_dim: int,
        num_target_views: int,
        query_grid_hw: tuple[int, int],
        intermediate_layer_idx: tuple[int, ...] = (4, 11, 17, 23),
        features: int = 256,
        out_channels: tuple[int, ...] = (256, 512, 1024, 1024),
    ):
        super().__init__()
        self.Hq, self.Wq = query_grid_hw
        self.K = num_target_views
        self.intermediate_layer_idx = list(intermediate_layer_idx)

        self.norm = nn.LayerNorm(token_dim)

        # Per-scale 1×1 Conv projection: token_dim → out_channels[i]
        self.projects = nn.ModuleList([
            nn.Conv2d(token_dim, oc, kernel_size=1) for oc in out_channels
        ])

        # Resize layers: ×4, ×2, ×1, ×0.5 (matches DPTHead)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # DPT scratch fusion (reuses VGGT's _make_scratch / _make_fusion_block)
        self.scratch = _make_scratch(list(out_channels), features)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_ch = 32

        # Point head: xyz (3) + confidence (1)
        self.point_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, head_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, 4, kernel_size=1),
        )

        # Depth head: depth (1) + confidence (1)
        self.depth_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, head_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, 2, kernel_size=1),
        )

    def forward(
        self,
        saved_outputs: dict[int, torch.Tensor],
        target_start: int,
        K: int,
        Q: int,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            saved_outputs: Dict mapping decoder layer index → [B, T, C] tensor.
            target_start:  Index where target tokens begin in the sequence T.
            K:             Number of target views (32).
            Q:             Tokens per view (256 = 16×16).

        Returns:
            dict: point_maps [B,K,3,H,W], point_confidence [B,K,1,H,W],
                  depths [B,K,1,H,W], depth_confidence [B,K,1,H,W]
        """
        B = next(iter(saved_outputs.values())).shape[0]
        Hq, Wq = self.Hq, self.Wq

        multi_scale: list[torch.Tensor] = []
        for i, layer_idx in enumerate(self.intermediate_layer_idx):
            x = saved_outputs[layer_idx]                          # [B, T, C]
            x = x[:, target_start: target_start + K * Q, :]      # [B, K*Q, C]
            x = self.norm(x)
            # Reshape to spatial grid: [B*K, C, Hq, Wq]
            x = x.reshape(B * K, Hq, Wq, -1).permute(0, 3, 1, 2).contiguous()
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            multi_scale.append(x)

        # DPT scratch forward (identical to DPTHead.scratch_forward)
        l1, l2, l3, l4 = multi_scale
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        fused = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        fused = self.scratch.refinenet3(fused, l3_rn, size=l2_rn.shape[2:])
        fused = self.scratch.refinenet2(fused, l2_rn, size=l1_rn.shape[2:])
        fused = self.scratch.refinenet1(fused, l1_rn)  # default ×2 upsample

        # Point head — inv_log activation (matches VGGT point_head)
        pt = self.point_conv2(self.point_conv1(fused))   # [B*K, 4, H_out, W_out]
        H_out, W_out = pt.shape[-2:]
        xyz_raw = pt[:, :3]
        xyz     = torch.sign(xyz_raw) * torch.expm1(torch.abs(xyz_raw))  # inv_log

        out = {"point_maps": xyz.view(B, K, 3, H_out, W_out)}

        # Confidence and depth are not used in training loss — skip during training
        # to avoid computing/storing their forward graph.
        if not self.training:
            p_conf = 1.0 + torch.exp(pt[:, 3:4])                        # expp1
            dp     = self.depth_conv2(self.depth_conv1(fused))
            depth  = torch.exp(dp[:, :1])
            d_conf = 1.0 + torch.exp(dp[:, 1:2])
            out.update({
                "point_confidence": p_conf.view(B, K, 1, H_out, W_out),
                "depth_maps":       depth.view(B, K, 1, H_out, W_out),
                "depth_confidence": d_conf.view(B, K, 1, H_out, W_out),
            })

        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TemporalVGGT(TemporalGeometryPredictor):
    """Temporal 3D geometry prediction from t1+t3 multi-view images.

    Args:
        vggt_model_id: HuggingFace model ID for VGGT, e.g. "facebook/VGGT-1B".
            Pass None to skip pretrained weights (for testing only).
        num_target_views: K, number of fixed output view slots. Default 32.
        target_query_grid: (Hq, Wq) spatial grid for target queries. Default (16, 16).
        gap_scale: Day-count normalization constant for time features. Default 365.0.
        lora_layers: List of decoder block indices (0-based) to receive LoRA.
            Default [18, 19, 20, 21, 22, 23].
        film_layers: Subset of lora_layers that also receive FiLM conditioning.
            Default [20, 21, 22, 23]. Must satisfy film_layers ⊆ lora_layers.
        lora_rank: LoRA rank. Default 8.
        lora_alpha: LoRA scaling factor alpha. Default 16.0.
        lora_dropout: Dropout applied to LoRA input. Default 0.05.
        film_hidden_dim: Hidden dim of the Time-FiLM MLP. Default 256.
        max_encoder_views: If not None, subsample this many views evenly from
            each endpoint before feeding the decoder. None uses all views.
            Reduce to lower GPU memory usage (e.g. 8 for 32-view inputs).
        gradient_checkpointing: If True, recompute decoder block activations
            during backward instead of storing them. Reduces peak GPU memory
            at the cost of ~30% extra forward compute. Default False.
    """

    def __init__(
        self,
        vggt_model_id: str = "facebook/VGGT-1B",
        num_target_views: int = 32,
        target_query_grid: tuple[int, int] = (16, 16),
        gap_scale: float = 365.0,
        lora_layers: list[int] = (18, 19, 20, 21, 22, 23),
        film_layers: list[int] = (20, 21, 22, 23),
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        film_hidden_dim: int = 256,
        max_encoder_views: int | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        lora_set = set(lora_layers)
        film_set = set(film_layers)
        if not film_set.issubset(lora_set):
            raise ValueError(
                f"film_layers must be a subset of lora_layers. "
                f"FiLM-only layers: {sorted(film_set - lora_set)}"
            )

        self.gap_scale = gap_scale
        self.num_target_views = num_target_views
        self.target_query_grid = target_query_grid
        self.max_encoder_views = max_encoder_views

        # Load (or randomly init) VGGT and freeze all parameters
        from vggt.models.vggt import VGGT
        if vggt_model_id is not None:
            logger.info("Loading VGGT from %s", vggt_model_id)
            self.vggt = VGGT.from_pretrained(vggt_model_id)
        else:
            logger.warning("vggt_model_id=None — VGGT initialized with random weights")
            self.vggt = VGGT()
        for p in self.vggt.parameters():
            p.requires_grad_(False)

        # Infer dims from loaded model
        embed_dim   = self.vggt.aggregator.frame_blocks[0].norm1.normalized_shape[0]  # 1024
        enc_out_dim = 2 * embed_dim   # aggregator concatenates frame+global → 2048D

        # Project VGGT 2048D output → 1024D decoder tokens
        self.token_proj = nn.Linear(enc_out_dim, embed_dim, bias=False)

        # Role embeddings: 0=t1, 1=t3, 2=target t2
        self.role_embed = nn.Embedding(3, embed_dim)

        # Learned target view-slot queries
        self.target_query_builder = TargetViewSlotQueryBuilder(
            num_target_views, target_query_grid, embed_dim
        )

        # DPT intermediate layer indices (shared between decoder and head)
        _dpt_intermediate = (4, 11, 17, 23)

        # Temporal decoder: independent copy of VGGT global_blocks
        self.decoder = TemporalDecoder(
            source_blocks=self.vggt.aggregator.global_blocks,
            lora_layers=list(lora_layers),
            film_layers=list(film_layers),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            intermediate_layer_idx=_dpt_intermediate,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Time-FiLM controller
        self.time_film_controller = TimeFiLMController(
            time_feat_dim=11,
            hidden_dim=film_hidden_dim,
            num_film_layers=len(film_layers),
            num_roles=3,
            token_dim=embed_dim,
        )

        # DPT-style geometry head
        self.geometry_head = TemporalDPTHead(
            token_dim=embed_dim,
            num_target_views=num_target_views,
            query_grid_hw=target_query_grid,
            intermediate_layer_idx=_dpt_intermediate,
        )

    def train(self, mode: bool = True) -> "TemporalVGGT":
        super().train(mode)
        self.vggt.eval()  # frozen encoder always stays in eval (affects BN/dropout)
        return self

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images with frozen VGGT aggregator.

        images: [B, S, 3, H, W] in [0, 1]
        Returns [B, S, P, 2048] with gradients disabled.
        """
        with torch.no_grad():
            agg_tokens, _ = self.vggt.aggregator(images)
            tokens = agg_tokens[-1]  # [B, S, P, 2048]
        return tokens.detach()

    def _subsample_views(self, tokens: torch.Tensor) -> torch.Tensor:
        """Evenly subsample views when max_encoder_views is set.

        tokens: [B, S, P, C]  →  [B, S', P, C]
        """
        S = tokens.shape[1]
        n = self.max_encoder_views
        if n is None or S <= n:
            return tokens
        idx = torch.linspace(0, S - 1, n, dtype=torch.long, device=tokens.device)
        return tokens[:, idx]

    def forward(self, batch: dict) -> dict:
        images_t1 = batch["images_t1"]   # [B, M1, 3, H, W]
        images_t3 = batch["images_t3"]   # [B, M3, 3, H, W]
        B = images_t1.shape[0]
        device = images_t1.device

        # 1. Encode with frozen VGGT aggregator
        tokens_t1 = self._encode(images_t1)    # [B, M1, P, 2048]
        tokens_t3 = self._encode(images_t3)    # [B, M3, P, 2048]

        # Optionally subsample views before the (expensive) decoder
        tokens_t1 = self._subsample_views(tokens_t1)   # [B, S1, P, 2048]
        tokens_t3 = self._subsample_views(tokens_t3)   # [B, S3, P, 2048]

        # Flatten view × token dimensions
        S1, P = tokens_t1.shape[1], tokens_t1.shape[2]
        S3     = tokens_t3.shape[1]
        tokens_t1 = tokens_t1.view(B, S1 * P, -1)    # [B, N1, 2048]
        tokens_t3 = tokens_t3.view(B, S3 * P, -1)    # [B, N3, 2048]

        # 2. Project to decoder dim (1024) — gradients flow through here
        tokens_t1 = self.token_proj(tokens_t1)   # [B, N1, 1024]
        tokens_t3 = self.token_proj(tokens_t3)   # [B, N3, 1024]

        # 3. Add role embeddings
        tokens_t1 = tokens_t1 + self.role_embed.weight[0]   # [1024] broadcast
        tokens_t3 = tokens_t3 + self.role_embed.weight[1]

        # 4. Build target view-slot queries and add role embedding
        K = self.num_target_views
        Q = self.target_query_builder.num_query_tokens
        target_tokens = self.target_query_builder(B, device=device)  # [B, K, Q, 1024]
        target_tokens = target_tokens + self.role_embed.weight[2]
        target_flat   = target_tokens.reshape(B, K * Q, -1)          # [B, K*Q, 1024]

        # 5. Build role_ids for FiLM (non-trainable long tensor)
        N1, N3 = tokens_t1.shape[1], tokens_t3.shape[1]
        role_ids = torch.cat([
            tokens_t1.new_zeros(N1, dtype=torch.long),
            tokens_t1.new_ones(N3, dtype=torch.long),
            tokens_t1.new_full((K * Q,), 2, dtype=torch.long),
        ])  # [N1 + N3 + K*Q]

        # 6. Concatenate: t1 tokens | t3 tokens | target queries
        all_tokens   = torch.cat([tokens_t1, tokens_t3, target_flat], dim=1)  # [B, T, 1024]
        target_start = N1 + N3

        # 7. Time-FiLM features from batch temporal metadata
        time_feat   = make_time_film_features(
            batch["doy_t1"], batch["doy_t2"], batch["doy_t3"],
            batch["day_index_t1"], batch["day_index_t2"], batch["day_index_t3"],
            gap_scale=self.gap_scale,
        )                                        # [B, 11]
        film_params = self.time_film_controller(time_feat)

        # 8. Temporal decoder (LoRA + FiLM in upper layers); collect DPT intermediates
        _, saved_outputs = self.decoder(all_tokens, role_ids=role_ids, film_params=film_params)

        # 9. DPT geometry head: multi-scale target token features → dense geometry
        return self.geometry_head(saved_outputs, target_start, K, Q)

