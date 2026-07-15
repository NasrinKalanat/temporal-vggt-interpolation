"""Temporal Residual Endpoint Transformer for t2 prediction.

Architecture:
  1. Run frozen VGGT independently on t1 and t3 endpoints; cache 4 intermediate layers.
  2. For each cache layer independently:
     a. Condition f1 and f3 with endpoint-specific time features (no role embedding).
     b. Flatten: query = conditioned f1 [B, S*T, D], memory = conditioned f3 [B, S3*T, D].
     c. Run N temporal transformer blocks:
          - Self-attention over f1 tokens with optional 2D RoPE (S frames × T tokens).
          - Cross-attention from f1 to f3.
          - MLP.
     d. Gated residual: out = f1 + gate[layer] * (transformer_output - query).
  3. Feed the 4 updated cached features to VGGT DPTHead.

Key differences from temporal_vggt_temporal_transformer:
  - No learnable t2 query tokens.
  - No in/out projection adapters; transformer operates at VGGT dim D = 2C = 2048.
  - Output is t2-like geometry in the t1 view/canvas (not arbitrary t2 query cameras).
  - Time encoder has no role embedding; uses endpoint date + signed gap to t2.
  - Teacher targets must be matched to t1 camera views.

Trainable: endpoint_time_encoder, time_adaln (if residual_adaln mode),
           temporal_transformer_blocks, gates, point_head.
Frozen:    VGGT aggregator (patch_embed, frame_blocks, global_blocks, special tokens).
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.heads.dpt_head import DPTHead
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.mlp import Mlp

from models.time_encoding import (
    EndpointTimeEncoder,
    CameraMLP,
    RelativeCameraMLP,
    ResidualAdaLN,
    apply_film,
    build_relative_gap_features,
)
from models.feature_cache_utils import run_cached_endpoint
from models.lora import apply_lora_to_dpt_head

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_LAYERS = [4, 11, 17, 23]




# ── Building blocks ───────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """Pre-norm cross-attention: q attends to kv, with residual.

    q_out = q + Attn(norm(q), norm(kv))
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = dropout

        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj  = nn.Linear(dim, dim)
        self.k_proj  = nn.Linear(dim, dim)
        self.v_proj  = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nq, C = q.shape
        Nkv = kv.shape[1]
        H, D = self.num_heads, self.head_dim
        qh = self.q_proj(self.norm_q(q)).view(B, Nq,  H, D).transpose(1, 2)
        k  = self.k_proj(self.norm_kv(kv)).view(B, Nkv, H, D).transpose(1, 2)
        v  = self.v_proj(self.norm_kv(kv)).view(B, Nkv, H, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            qh, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        return q + self.out_proj(out.transpose(1, 2).reshape(B, Nq, C))


class TemporalTransformerBlock(nn.Module):
    """Self-attention (optional 2D RoPE) + cross-attention + MLP, all pre-norm + residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rope: Optional[RotaryPositionEmbedding2D] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = dropout
        self.rope = rope

        self.norm_sa = nn.LayerNorm(dim)
        self.qkv_sa  = nn.Linear(dim, 3 * dim)
        self.out_sa  = nn.Linear(dim, dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(
        self,
        z: torch.Tensor,      # [B, N, dim]   query tokens
        memory: torch.Tensor, # [B, M, dim]   key/value tokens
        pos: Optional[torch.Tensor] = None,  # [B, N, 2] 2D RoPE positions
    ) -> torch.Tensor:
        B, N, C = z.shape
        H, D = self.num_heads, self.head_dim
        qkv = self.qkv_sa(self.norm_sa(z)).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        sa_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        z = z + self.out_sa(sa_out.transpose(1, 2).reshape(B, N, C))
        z = self.cross_attn(z, memory)
        z = z + self.ffn(self.norm_ffn(z))
        return z


# ── Main model ────────────────────────────────────────────────────────────────

class TemporalResidualEndpoint(nn.Module):
    """VGGT endpoint features + gated residual temporal transformer for t2 prediction.

    f1 features serve as the query canvas; f3 features serve as key/value memory.
    The temporal transformer uses normal internal residuals, so its output is
    converted to an explicit delta by subtracting the input query. That delta is
    added to the original f1 feature through a small per-layer gate.

    Args:
        vggt_model_id:              HuggingFace model id or local path for VGGT.
        embed_dim:                  Token dimension C (1024 for VGGT-1B). D = 2C = 2048.
        num_transformer_layers:     Number of temporal transformer blocks.
        num_transformer_heads:      Attention heads (d_model must be divisible).
        mlp_ratio:                  FFN hidden dim multiplier.
        dropout:                    Dropout in attention and FFN layers.
        time_conditioning_mode:     "additive" | "film" | "residual_adaln".
        time_hidden_dim:            Hidden dim of EndpointTimeEncoder MLP.
        cache_layers:               4 aggregator layer indices to cache (for DPTHead).
        use_rope_in_self_attention: Apply 2D RoPE to t1 self-attention (recommended).
        init_point_head_from_vggt:  Copy pretrained VGGT DPTHead weights.
        use_gradient_checkpoint:    Gradient-checkpoint temporal transformer blocks.
        point_head_mode:            "full" (all head params trainable) or "lora"
                                    (freeze head, inject LoRA adapters).
        point_head_lora:            Dict with LoRA config when point_head_mode="lora":
                                    {rank, alpha, dropout, target_modules}.
        d_model_down_proj:          If set, project cached features from D to this dim
                                    before the transformer, then project back after.
                                    Per-layer projections (xavier down, zero-init up).
        use_camera_cond:            If True, add per-view camera conditioning after
                                    down-projection. Includes absolute camera MLP and
                                    relative camera MLP (source→t2).
        camera_hidden_dim:          Hidden dim for camera MLPs.
    """

    def __init__(
        self,
        vggt_model_id: str = "facebook/VGGT-1B",
        embed_dim: int = 1024,
        num_query_views: int = 1,  # used by dataset loader only; ignored here
        num_transformer_layers: int = 4,
        num_transformer_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_conditioning_mode: str = "additive",
        time_hidden_dim: int = 256,
        cache_layers: Optional[List[int]] = None,
        use_rope_in_self_attention: bool = True,
        init_point_head_from_vggt: bool = True,
        use_gradient_checkpoint: bool = False,
        skip_aggregator: bool = False,
        point_head_mode: str = "full",
        point_head_lora: Optional[Dict] = None,
        d_model_down_proj: Optional[int] = None,
        use_camera_cond: bool = False,
        camera_hidden_dim: int = 256,
    ):
        super().__init__()
        if cache_layers is None:
            cache_layers = _DEFAULT_CACHE_LAYERS
        assert len(cache_layers) == 4, "DPTHead requires exactly 4 cache layers"
        assert time_conditioning_mode in ("additive", "film", "residual_adaln"), time_conditioning_mode

        self.embed_dim = embed_dim
        self.D = 2 * embed_dim  # VGGT cached feature dim (2048 for VGGT-1B)
        self.d_model = d_model_down_proj if d_model_down_proj is not None else self.D
        self.use_down_proj = d_model_down_proj is not None
        self.cache_layers = sorted(cache_layers)
        self.cache_set = set(cache_layers)
        self.time_conditioning_mode = time_conditioning_mode
        self.use_gradient_checkpoint = use_gradient_checkpoint

        # --- Load VGGT, freeze aggregator ---
        if skip_aggregator:
            logger.info("skip_aggregator=True: aggregator not loaded (feature cache assumed full)")
            self.aggregator = None
            self.patch_start_idx = 5  # constant for VGGT-1B
            vggt = VGGT.from_pretrained(vggt_model_id) if init_point_head_from_vggt else None
        else:
            logger.info(f"Loading VGGT from {vggt_model_id!r}")
            vggt = VGGT.from_pretrained(vggt_model_id)
            self.aggregator = vggt.aggregator
            self.patch_start_idx = self.aggregator.patch_start_idx  # 5
            for p in self.aggregator.parameters():
                p.requires_grad_(False)

        # --- Per-layer down/up projections (if d_model_down_proj is set) ---
        if self.use_down_proj:
            self.down_norms = nn.ModuleList([
                nn.LayerNorm(self.D) for _ in cache_layers
            ])
            self.down_projs = nn.ModuleList([
                nn.Linear(self.D, self.d_model) for _ in cache_layers
            ])
            self.up_norms = nn.ModuleList([
                nn.LayerNorm(self.d_model) for _ in cache_layers
            ])
            self.up_projs = nn.ModuleList([
                nn.Linear(self.d_model, self.D) for _ in cache_layers
            ])
            for proj in self.down_projs:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
            for proj in self.up_projs:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        else:
            self.down_norms = None
            self.down_projs = None
            self.up_norms = None
            self.up_projs = None

        # --- Endpoint time encoder (shared for t1 and t3) ---
        # Features are already endpoint-specific (endpoint_date, signed gap to t2, rel_gap).
        self.endpoint_time_encoder = EndpointTimeEncoder(
            out_dim=self.d_model, hidden_dim=time_hidden_dim
        )

        # --- Camera conditioning (optional) ---
        self.use_camera_cond = use_camera_cond
        if use_camera_cond:
            self.camera_mlp = CameraMLP(self.d_model, camera_hidden_dim)
            self.relative_camera_mlp = RelativeCameraMLP(self.d_model, camera_hidden_dim)

        # ResidualAdaLN only needed for residual_adaln mode
        if time_conditioning_mode == "residual_adaln":
            self.time_adaln = ResidualAdaLN(self.d_model)
        else:
            self.time_adaln = None

        # --- 2D RoPE + position getter for self-attention ---
        rope = RotaryPositionEmbedding2D(frequency=100) if use_rope_in_self_attention else None
        self.position_getter = PositionGetter()

        # --- Temporal transformer blocks (shared across all 4 cache layers) ---
        self.temporal_transformer_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                dim=self.d_model,
                num_heads=num_transformer_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                rope=rope,
            )
            for _ in range(num_transformer_layers)
        ])

        # --- Per-layer gates (small scalar, near-identity start) ---
        self.gates = nn.ParameterList([
            nn.Parameter(torch.full((1,), 1e-3)) for _ in cache_layers
        ])

        # --- Point head ---
        if init_point_head_from_vggt and vggt is not None and vggt.point_head is not None:
            self.point_head = copy.deepcopy(vggt.point_head)
            self.point_head.intermediate_layer_idx = [0, 1, 2, 3]
        else:
            self.point_head = DPTHead(
                dim_in=self.D,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )

        if vggt is not None:
            del vggt

        # --- Point head mode: full training vs LoRA ---
        assert point_head_mode in ("full", "lora"), f"Invalid point_head_mode: {point_head_mode!r}"
        self.point_head_mode = point_head_mode
        if point_head_mode == "lora":
            lora_cfg = point_head_lora or {}
            apply_lora_to_dpt_head(
                self.point_head,
                rank=lora_cfg.get("rank", 8),
                alpha=lora_cfg.get("alpha", 16.0),
                dropout=lora_cfg.get("dropout", 0.0),
                target_modules=lora_cfg.get("target_modules", None),
                target_kernel_sizes=lora_cfg.get("target_kernel_sizes", None),
                target_layer_types=lora_cfg.get("target_layer_types", None),
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.aggregator is not None:
            self.aggregator.eval()
        return self

    def _run_vggt_endpoint(
        self, images: torch.Tensor, B: int, S: int
    ) -> List[torch.Tensor]:
        """Run frozen VGGT on one endpoint, collect 4 cache layers.

        Returns list of 4 tensors [B, S, T, D] at self.cache_layers.
        """
        agg = self.aggregator
        H_img, W_img = images.shape[-2:]
        patch_h = H_img // agg.patch_size
        patch_w = W_img // agg.patch_size

        with torch.no_grad():
            imgs_norm = (images - agg._resnet_mean) / agg._resnet_std
            patch_tokens = agg.patch_embed(imgs_norm.view(B * S, 3, H_img, W_img))
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            _, P_patch, C = patch_tokens.shape
            del imgs_norm

            camera_tok   = slice_expand_and_flatten(agg.camera_token,   B, S)
            register_tok = slice_expand_and_flatten(agg.register_token, B, S)
            tokens = torch.cat([camera_tok, register_tok, patch_tokens], dim=1)
            del camera_tok, register_tok, patch_tokens
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

            frame_idx  = 0
            global_idx = 0
            cached: List[torch.Tensor] = []

            for layer_count in range(agg.aa_block_num):
                tokens, frame_idx,  frame_ints  = agg._process_frame_attention(
                    tokens, B, S, P_total, C, frame_idx, pos=pos
                )
                tokens, global_idx, global_ints = agg._process_global_attention(
                    tokens, B, S, P_total, C, global_idx, pos=pos
                )
                fi = frame_ints[0]
                gi = global_ints[0]
                del frame_ints, global_ints

                if layer_count in self.cache_set:
                    cached.append(torch.cat([fi, gi], dim=-1))  # [B, S, T, D]

                del fi, gi

        return cached  # 4 × [B, S, T, D]

    def _run_vggt_endpoints(
        self,
        images_t1,
        images_t3,
        B: int,
        t1_key=None,
        t3_key=None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run frozen VGGT on t1 and t3 with one cache entry per sample."""
        return (
            run_cached_endpoint(self, images_t1, B, t1_key, "t1"),
            run_cached_endpoint(self, images_t3, B, t3_key, "t3"),
        )

    def _apply_time_cond(
        self,
        x: torch.Tensor,          # [B, S, T, d_model]
        endpoint_date: torch.Tensor,  # [B]
        target_date: torch.Tensor,    # [B]
        rel_gap: torch.Tensor,        # [B, 5]
    ) -> torch.Tensor:
        if self.time_conditioning_mode == "additive":
            off = self.endpoint_time_encoder.additive(endpoint_date, target_date, rel_gap)
            for _ in range(x.dim() - off.dim()):
                off = off.unsqueeze(-2)
            return x + off
        else:
            g, b = self.endpoint_time_encoder.gamma_beta(endpoint_date, target_date, rel_gap)
            if self.time_conditioning_mode == "film":
                return apply_film(x, g, b)
            else:  # residual_adaln
                return self.time_adaln(x, g, b)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch:
                images_t1:            [B, S, 3, H, W]
                images_t3:            [B, S3, 3, H, W]
                date_t1/t2/t3:        [B] day-of-year (float)
                t1_day/t2_day/t3_day: [B] ordinal day
                rel_gap_feat:         [B, 5] (optional, computed if absent)
        Returns:
            pred_points: [B, S, H_out, W_out, 3]   t2-like geometry in t1 canvas
            pred_conf:   [B, S, H_out, W_out]
        """
        device = next(self.parameters()).device

        images_t1 = batch["images_t1"]
        images_t3 = batch["images_t3"]
        B = batch["date_t1"].shape[0]
        H = W = 518  # VGGT always preprocesses to this size; used only for mock_images

        if "rel_gap_feat" in batch:
            rel_gap = batch["rel_gap_feat"].to(device)
        else:
            rel_gap = build_relative_gap_features(
                batch["t1_day"], batch["t2_day"], batch["t3_day"]
            ).to(device)
        date_t1 = batch["date_t1"].to(device)
        date_t2 = batch["date_t2"].to(device)
        date_t3 = batch["date_t3"].to(device)

        # --- Frozen VGGT endpoint pass (cache-aware) ---
        t1_key = batch.get("t1_cache_key")
        t3_key = batch.get("t3_cache_key")
        cached_t1, cached_t3 = self._run_vggt_endpoints(
            images_t1, images_t3, B,
            t1_key=t1_key, t3_key=t3_key,
        )

        # The pair cache may store t1+t2 views concatenated (e.g. 32 = 16+16).
        # Slice to the actual endpoint view count using camera metadata.
        n_t1 = batch["camera_t1"]["transform_matrix"].shape[-3]  # [B, V, 4, 4] → V
        n_t3 = batch["camera_t3"]["transform_matrix"].shape[-3]
        if cached_t1[0].shape[1] > n_t1:
            cached_t1 = [f[:, :n_t1] for f in cached_t1]
        if cached_t3[0].shape[1] > n_t3:
            cached_t3 = [f[:, :n_t3] for f in cached_t3]

        S  = cached_t1[0].shape[1]
        S3 = cached_t3[0].shape[1]
        del images_t1, images_t3

        T = cached_t1[0].shape[2]

        # --- Camera dicts (moved to device) ---
        if self.use_camera_cond:
            camera_t1 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["camera_t1"].items()}
            camera_t3 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["camera_t3"].items()}
            camera_t2 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["camera_t2_query"].items()}
        else:
            camera_t1 = camera_t3 = camera_t2 = None

        # --- 2D RoPE positions for self-attention over S×T tokens ---
        pos = self.position_getter(B, S, T, device)  # [B, S*T, 2]

        # --- Per-cache-layer temporal residual update ---
        updated_cached: List[torch.Tensor] = []

        for idx in range(len(self.cache_layers)):
            f1 = cached_t1.pop(0)  # [B, S, T, D]
            f3 = cached_t3.pop(0)  # [B, S3, T, D]

            # Down-project if configured
            if self.use_down_proj:
                f1_low = self.down_projs[idx](self.down_norms[idx](f1))  # [B, S, T, d_model]
                f3_low = self.down_projs[idx](self.down_norms[idx](f3))  # [B, S3, T, d_model]
            else:
                f1_low = f1
                f3_low = f3

            # Camera conditioning: absolute + relative (source → t2)
            if self.use_camera_cond:
                f1_low = f1_low + self.camera_mlp(camera_t1).unsqueeze(2)
                f3_low = f3_low + self.camera_mlp(camera_t3).unsqueeze(2)
                f1_low = f1_low + self.relative_camera_mlp(camera_t1, camera_t2).unsqueeze(2)
                f3_low = f3_low + self.relative_camera_mlp(camera_t3, camera_t2).unsqueeze(2)

            # Time conditioning: t1 endpoint knows it's left of t2, t3 knows it's right
            f1_cond = self._apply_time_cond(f1_low, date_t1, date_t2, rel_gap)
            f3_cond = self._apply_time_cond(f3_low, date_t3, date_t2, rel_gap)
            del f1_low, f3_low

            # Flatten for transformer
            q  = f1_cond.reshape(B, S  * T, self.d_model)   # [B, S*T, d_model]
            kv = f3_cond.reshape(B, S3 * T, self.d_model)   # [B, S3*T, d_model]
            del f1_cond, f3_cond

            # Temporal transformer blocks
            h = q
            for block in self.temporal_transformer_blocks:
                if self.use_gradient_checkpoint and self.training:
                    h = grad_checkpoint(block, h, kv, pos, use_reentrant=False)
                else:
                    h = block(h, kv, pos)
            del kv

            # Convert the transformer's updated hidden state into an explicit
            # residual branch, then add it to the original VGGT f1 feature.
            h = (h - q).reshape(B, S, T, self.d_model)
            if self.use_down_proj:
                h = self.up_projs[idx](self.up_norms[idx](h))  # [B, S, T, D]
            updated_cached.append(f1 + self.gates[idx] * h)
            del f1, h

        # --- VGGT DPTHead ---
        if getattr(self, "freeze_point_head", False):
            return {"pred_cached_layers": updated_cached}

        mock_images = torch.zeros(
            B, S, 3, H, W,
            device=device,
            dtype=updated_cached[0].dtype,
        )
        pred_points, pred_conf = self.point_head(
            updated_cached,
            images=mock_images,
            patch_start_idx=self.patch_start_idx,
        )
        return {"pred_points": pred_points, "pred_conf": pred_conf, "pred_cached_layers": updated_cached}

    def trainable_parameter_groups(self) -> List[Dict]:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
