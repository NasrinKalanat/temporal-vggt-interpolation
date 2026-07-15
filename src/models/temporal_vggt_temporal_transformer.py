"""Temporal-VGGT with temporal transformer for explicit t2 query view prediction.

Architecture:
  1. Run frozen VGGT on t1 and t3 endpoints separately; cache 4 intermediate layers.
  2. For each cache layer independently:
     a. Expand per-layer learnable t2 query tokens (shape [1,1,1,D] → [B,Q,T,D]).
     b. Apply time conditioning (role + date + rel_gap) to f1, f3, and t2q at dim D=2C.
     c. Flatten: z = t2q [B, Q*T, D]; memory = [f1, f3] [B, (S1+S3)*T, D].
     d. Run N temporal transformer blocks:
          - Self-attention among t2q tokens with 2D RoPE (Q frames × T tokens).
          - Cross-attention from t2q to f1+f3 memory.
          - MLP.
     e. Reshape output [B, Q, T, D] as this layer's t2 cached features.
  3. VGGT DPTHead decodes the 4-layer list to t2 pointmaps + confidence.

No projection adapters. No gated residual. VGGT never sees t2 images.
t2 features are built from a per-layer learnable query updated by the transformer.

Trainable: learnable_t2_query, time_encoder, time_adaln_source/target,
           temporal_transformer_blocks, point_head.
Frozen:    VGGT aggregator (patch_embed, frame_blocks, global_blocks, special tokens).
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.heads.dpt_head import DPTHead
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter

from models.temporal_vggt_residual_endpoint import CrossAttention, TemporalTransformerBlock
from models.time_encoding import (
    SharedTimeEncoder,
    ResidualAdaLN,
    apply_film,
    build_relative_gap_features,
)
from models.feature_cache_utils import run_cached_endpoint
from models.lora import apply_lora_to_dpt_head

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_LAYERS = [4, 11, 17, 23]


class TemporalVGGTTemporalTransformer(nn.Module):
    """Learnable t2 query + temporal transformer for t2 prediction at explicit query views.

    Args:
        vggt_model_id:              HuggingFace model id or local path for VGGT.
        embed_dim:                  Token dimension C (1024 for VGGT-1B). D = 2C = 2048.
        num_query_views:            Number of t2 query views Q (must match dataset).
        num_transformer_layers:     Number of temporal transformer blocks.
        num_transformer_heads:      Attention heads (D must be divisible).
        mlp_ratio:                  FFN hidden dim multiplier.
        dropout:                    Dropout in attention and FFN layers.
        time_conditioning_mode:     "additive" | "film" | "residual_adaln".
        time_hidden_dim:            Hidden dim of SharedTimeEncoder MLP.
        cache_layers:               4 aggregator layer indices to cache (for DPTHead).
        use_rope_in_self_attention: Apply 2D RoPE (Q×T grid) to t2q self-attention.
        init_point_head_from_vggt:  Copy pretrained VGGT DPTHead weights.
        use_gradient_checkpoint:    Gradient-checkpoint temporal transformer blocks.
        point_head_mode:            "full" (all head params trainable) or "lora"
                                    (freeze head, inject LoRA adapters).
        point_head_lora:            Dict with LoRA config when point_head_mode="lora":
                                    {rank, alpha, dropout, target_modules}.
        query_init_mode:            How the learnable t2 query carries positional identity:
                                    "single"     [1,1,1,D] one token over all (Q,T) [baseline]
                                    "per_patch"  [1,1,T,D] per spatial/patch position
                                    "full"       [1,Q,T,D] per (view, patch) position
                                    "factorized" [1,1,T,D] patch + [1,Q,1,D] view, summed
        num_query_tokens:           T = tokens per frame used to size per-patch query params.
                                    None -> 5 + (518//14)**2 = 1374 (VGGT-1B @ 518px).
    """

    def __init__(
        self,
        vggt_model_id: str = "facebook/VGGT-1B",
        embed_dim: int = 1024,
        num_query_views: int = 16,
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
        add_t1_to_transformer: bool = False,
        skip_aggregator: bool = False,
        point_head_mode: str = "full",
        point_head_lora: Optional[Dict] = None,
        query_init_mode: str = "single",
        num_query_tokens: Optional[int] = None,
    ):
        super().__init__()
        if cache_layers is None:
            cache_layers = _DEFAULT_CACHE_LAYERS
        assert len(cache_layers) == 4, "DPTHead requires exactly 4 cache layers"
        assert time_conditioning_mode in ("additive", "film", "residual_adaln"), time_conditioning_mode
        assert query_init_mode in ("single", "per_patch", "full", "factorized"), query_init_mode

        self.embed_dim = embed_dim
        self.D = 2 * embed_dim  # VGGT cached feature dim (2048 for VGGT-1B)
        self.num_query_views = num_query_views
        self.query_init_mode = query_init_mode
        # T = tokens per frame = patch_start_idx (5 special) + num_patches.
        # VGGT-1B at 518px / patch 14 -> 5 + 37*37 = 1374. Needed at __init__ to
        # size per-patch query parameters; validated against runtime T in forward.
        self.num_query_tokens = (
            num_query_tokens if num_query_tokens is not None else 5 + (518 // 14) ** 2
        )
        self.cache_layers = sorted(cache_layers)
        self.cache_set = set(cache_layers)
        self.time_conditioning_mode = time_conditioning_mode
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.add_t1_to_transformer = add_t1_to_transformer

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

        # --- Learnable t2 query (one set per cache layer), built to [1,Q,T,D] in forward ---
        # query_init_mode controls how much positional identity the query carries:
        #   "single":     [1,1,1,D] — one token broadcast over all (Q,T) (baseline)
        #   "per_patch":  [1,1,T,D] — distinct token per spatial/patch position, shared over views
        #   "full":       [1,Q,T,D] — distinct token per (view, patch) position
        #   "factorized": [1,1,T,D] (patch) + [1,Q,1,D] (view), summed -> [1,Q,T,D]
        # single uses zero init (baseline behavior); the multi-token modes use a small
        # truncated-normal init to break positional symmetry from the start.
        Q = num_query_views
        Tq = self.num_query_tokens
        D = self.D
        n_layers = len(cache_layers)

        def _zeros(*shape):
            return nn.Parameter(torch.zeros(*shape))

        def _normal(*shape):
            p = nn.Parameter(torch.empty(*shape))
            nn.init.trunc_normal_(p, std=0.02)
            return p

        if query_init_mode == "single":
            self.learnable_t2_query = nn.ParameterList(
                [_zeros(1, 1, 1, D) for _ in range(n_layers)]
            )
        elif query_init_mode == "per_patch":
            self.learnable_t2_query = nn.ParameterList(
                [_normal(1, 1, Tq, D) for _ in range(n_layers)]
            )
        elif query_init_mode == "full":
            self.learnable_t2_query = nn.ParameterList(
                [_normal(1, Q, Tq, D) for _ in range(n_layers)]
            )
        else:  # factorized
            self.learnable_t2_query_patch = nn.ParameterList(
                [_normal(1, 1, Tq, D) for _ in range(n_layers)]
            )
            self.learnable_t2_query_view = nn.ParameterList(
                [_normal(1, Q, 1, D) for _ in range(n_layers)]
            )

        # --- Time encoder with role embedding (role_id 0=t1, 1=t3, 2=t2_query) ---
        self.time_encoder = SharedTimeEncoder(
            source_dim=self.D,
            target_dim=self.D,
            hidden_dim=time_hidden_dim,
        )

        # ResidualAdaLN only needed for residual_adaln mode
        if time_conditioning_mode == "residual_adaln":
            self.time_adaln_source = ResidualAdaLN(self.D)  # for t1 and t3
            self.time_adaln_target = ResidualAdaLN(self.D)  # for t2 query
        else:
            self.time_adaln_source = None
            self.time_adaln_target = None

        # --- 2D RoPE + position getter for t2q self-attention (Q×T grid) ---
        rope = RotaryPositionEmbedding2D(frequency=100) if use_rope_in_self_attention else None
        self.position_getter = PositionGetter()

        # --- Temporal transformer blocks (shared across all 4 cache layers) ---
        self.temporal_transformer_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                dim=self.D,
                num_heads=num_transformer_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                rope=rope,
            )
            for _ in range(num_transformer_layers)
        ])

        # --- Per-layer gates for gated residual (only when add_t1_to_transformer=True) ---
        if add_t1_to_transformer:
            self.gates = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in cache_layers
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

    def _build_t2_query(self, idx: int, B: int, Q: int, T: int) -> torch.Tensor:
        """Build the t2 query for cache layer `idx`, expanded to [B, Q, T, D]."""
        if self.query_init_mode == "factorized":
            base = self.learnable_t2_query_patch[idx] + self.learnable_t2_query_view[idx]
        else:
            base = self.learnable_t2_query[idx]
        return base.expand(B, Q, T, self.D)

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

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch:
                images_t1:            [B, S1, 3, H, W]
                images_t3:            [B, S3, 3, H, W]
                date_t1/t2/t3:        [B] day-of-year (float)
                t1_day/t2_day/t3_day: [B] ordinal day
                rel_gap_feat:         [B, 5] (optional, computed if absent)
        Returns:
            pred_points: [B, Q, H_out, W_out, 3]
            pred_conf:   [B, Q, H_out, W_out]
        """
        device = next(self.parameters()).device

        images_t1 = batch["images_t1"]
        images_t3 = batch["images_t3"]
        B = batch["date_t1"].shape[0]
        H = W = 518  # VGGT always preprocesses to this size; used only for mock_images
        Q = self.num_query_views

        if "rel_gap_feat" in batch:
            rel_gap = batch["rel_gap_feat"].to(device)
        else:
            rel_gap = build_relative_gap_features(
                batch["t1_day"], batch["t2_day"], batch["t3_day"]
            ).to(device)
        date_t1 = batch["date_t1"].to(device)
        date_t2 = batch["date_t2"].to(device)
        date_t3 = batch["date_t3"].to(device)

        role_t1 = torch.zeros(B, dtype=torch.long, device=device)   # 0 = t1
        role_t3 = torch.ones(B,  dtype=torch.long, device=device)   # 1 = t3
        role_t2 = torch.full((B,), 2, dtype=torch.long, device=device)  # 2 = t2 query

        # --- Frozen VGGT endpoint pass (cache-aware) ---
        t1_key = batch.get("t1_cache_key")
        t3_key = batch.get("t3_cache_key")
        cached_t1, cached_t3 = self._run_vggt_endpoints(
            images_t1, images_t3, B,
            t1_key=t1_key, t3_key=t3_key,
        )
        # S1/S3 from features: placeholder images have S=1 but cached features have the real S
        S1 = cached_t1[0].shape[1]
        S3 = cached_t3[0].shape[1]
        if self.add_t1_to_transformer:
            assert S1 == Q, (
                f"add_t1_to_transformer requires S1==Q, got S1={S1} Q={Q}"
            )
        del images_t1, images_t3

        T = cached_t1[0].shape[2]
        if self.query_init_mode != "single":
            assert T == self.num_query_tokens, (
                f"query_init_mode={self.query_init_mode!r} sized for "
                f"num_query_tokens={self.num_query_tokens} but runtime T={T}. "
                f"Set model_kwargs.num_query_tokens to {T}."
            )

        # --- Time conditioning (computed once, shared across all 4 cache layers) ---
        if self.time_conditioning_mode in ("residual_adaln", "film"):
            g_t1, b_t1 = self.time_encoder.source(role_t1, date_t1, rel_gap)
            g_t3, b_t3 = self.time_encoder.source(role_t3, date_t3, rel_gap)
            g_t2, b_t2 = self.time_encoder.target(role_t2, date_t2, rel_gap)
        else:  # additive
            off_t1 = self.time_encoder.source_additive(role_t1, date_t1, rel_gap)
            off_t3 = self.time_encoder.source_additive(role_t3, date_t3, rel_gap)
            off_t2 = self.time_encoder.target_additive(role_t2, date_t2, rel_gap)

        # --- 2D RoPE positions for t2q self-attention (Q×T grid) ---
        pos = self.position_getter(B, Q, T, device)  # [B, Q*T, 2]

        # --- Per-cache-layer temporal decoding ---
        aggregated_tokens_list_t2: List[torch.Tensor] = []

        for idx in range(len(self.cache_layers)):
            f1 = cached_t1.pop(0)  # [B, S1, T, D]
            f3 = cached_t3.pop(0)  # [B, S3, T, D]

            # Learnable t2 query at D, expanded over (B, Q, T)
            t2q = self._build_t2_query(idx, B, Q, T)  # [B, Q, T, D]

            # Save original f1 before conditioning for gated residual
            f1_orig = f1 if self.add_t1_to_transformer else None

            # Time conditioning
            if self.time_conditioning_mode == "residual_adaln":
                f1  = self.time_adaln_source(f1,  g_t1, b_t1)
                f3  = self.time_adaln_source(f3,  g_t3, b_t3)
                t2q = self.time_adaln_target(t2q, g_t2, b_t2)
            elif self.time_conditioning_mode == "film":
                f1  = apply_film(f1,  g_t1, b_t1)
                f3  = apply_film(f3,  g_t3, b_t3)
                t2q = apply_film(t2q, g_t2, b_t2)
            else:  # additive
                def _add(x, off):
                    for _ in range(x.dim() - off.dim()):
                        off = off.unsqueeze(-2)
                    return x + off
                f1  = _add(f1,  off_t1)
                f3  = _add(f3,  off_t3)
                t2q = _add(t2q, off_t2)

            # Flatten for transformer
            z = t2q.reshape(B, Q * T, self.D)
            memory = torch.cat([f1, f3], dim=1).reshape(B, (S1 + S3) * T, self.D)
            del f1, f3, t2q

            # Temporal transformer blocks
            for block in self.temporal_transformer_blocks:
                if self.use_gradient_checkpoint and self.training:
                    z = grad_checkpoint(block, z, memory, pos, use_reentrant=False)
                else:
                    z = block(z, memory, pos)
            del memory

            z = z.reshape(B, Q, T, self.D)
            if self.add_t1_to_transformer:
                z = f1_orig + self.gates[idx] * z
            aggregated_tokens_list_t2.append(z)
            del z

        # --- VGGT DPTHead ---
        if getattr(self, "freeze_point_head", False):
            return {"pred_cached_layers": aggregated_tokens_list_t2}

        mock_images = torch.zeros(
            B, Q, 3, H, W,
            device=device,
            dtype=aggregated_tokens_list_t2[0].dtype,
        )
        pred_points, pred_conf = self.point_head(
            aggregated_tokens_list_t2,
            images=mock_images,
            patch_start_idx=self.patch_start_idx,
        )
        return {"pred_points": pred_points, "pred_conf": pred_conf, "pred_cached_layers": aggregated_tokens_list_t2}

    def trainable_parameter_groups(self) -> List[Dict]:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
