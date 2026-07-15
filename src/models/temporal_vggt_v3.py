"""Temporal-VGGT v3: Frozen VGGT + cross-attention fusion for t2 prediction.

Architecture overview
---------------------
Given multi-view images at t1 and t3, predict the 3D point map at t2 for a
query camera view.

Forward pass:
  1. Patch-embed t1 and t3 images via frozen VGGT patch_embed.
  2. Run frozen VGGT aggregator independently on t1 → select 4 intermediate outputs.
  3. Run frozen VGGT aggregator independently on t3 → select 4 intermediate outputs.
  4. Compute time embeddings via SharedTimeEncoder:
       t1/t3: source_head → [B, 2C] gamma/beta for feature conditioning
       t2:    target_head → [B, C]  gamma/beta for query grid conditioning
              source_head → [B, 2C] gamma_t2_src for camera head Q
  5. Optionally compute camera embeddings via SharedCameraEncoder:
       t1/t3: source_head → [B, V, 2C] for feature conditioning
       t2:    target_head → [B, Q, C]  for query conditioning (learnable/factorized only)
  6. Form the t2 query Q — two modes controlled by t2_query_mode:
       "broadcast":
           gamma_t2_src [B, 2C] from source_head, expanded to [B, Q*P_patch, 2C].
           Every patch position holds the same t2 time vector.
       "learnable_grid":
           Learnable query_grid [B, Q, P_patch, C] conditioned via camera then time
           (both using target-dim heads). Each patch has a distinct learned prior.
       "factorized_grid":
           Additive factorization: query_spatial_grid [1,1,P,C] + query_view_embed [1,Q,1,C],
           then conditioned same as learnable_grid.
  7. For each of the 4 selected VGGT layers:
       a. Camera-condition then time-condition t1/t3 features → K, V = [B, (V1+V3)*P_total, 2C].
       b. Cross-attention: Q → K, V → attended t2 features [B, Q*P_patch, 2C].
       c. Reshape + zero-pad special tokens → [B, Q, P_total, 2C].
  8. Pass 4 attended tensors to DPTHead → point map + confidence.

Trainable parameters
---------------------
  time_encoder        (SharedTimeEncoder: shared backbone → source_head [2C] for t1/t3
                       and target_head [C] for t2 query; broadcast uses source_head only)
  source_time_adaln   (ResidualAdaLN dim=2C; only if residual_adaln)
  target_time_adaln   (ResidualAdaLN dim=C; only if residual_adaln; learnable/factorized only)
  -- learnable_grid / factorized_grid only --
  query_grid          (learnable blank t2 query tokens)  or
  query_spatial_grid + query_view_embed  (factorized_grid)
  --
  cross_attn          (CrossAttentionLayer, shared across the 4 selected layers)
  point_head          (DPTHead, fresh or copied from VGGT)
  -- camera head only (use_camera_head=True) --
  camera_cross_attn   (CrossAttentionLayer for camera token prediction)
  camera_head         (CameraHead, copied or fresh init)
  -- camera conditioning (camera_conditioning_mode != "none") --
  cam_encoder         (SharedCameraEncoder: shared backbone → source_head [2C] for t1/t3
                       and target_head [C] for t2 query; broadcast mode skips target_head)
  source_cam_adaln    (ResidualAdaLN dim=2C; only if residual_adaln)
  target_cam_adaln    (ResidualAdaLN dim=C; only if residual_adaln; learnable/factorized only)

Frozen parameters
------------------
  aggregator.*        (all VGGT weights, no LoRA)
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead

from models.time_encoding import (
    SharedTimeEncoder,
    SharedCameraEncoder,
    AdditiveCameraConditioner,
    ResidualAdaLN,
    apply_film,
    build_relative_gap_features,
)

logger = logging.getLogger(__name__)


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer: Q from t2, K/V from t1+t3 features.

    Uses F.scaled_dot_product_attention (flash attention) directly so that
    cross-attention (Q != KV) also benefits from O(n) memory — nn.MultiheadAttention
    falls back to explicit softmax for cross-attention, materialising the full
    [B, H, Sq, Skv] matrix which would be ~30 GB for Sq=21904, Skv=43968.

    Args:
        q_dim:     Query input dimension.
        kv_dim:    K/V input dimension.
        d_model:   Output dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        dropout:   Attention dropout probability.
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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:  [B, Sq, q_dim]
            kv: [B, Skv, kv_dim]
        Returns:
            out: [B, Sq, d_model]
        """
        B, Sq, _ = q.shape
        Skv = kv.shape[1]
        H, D = self.num_heads, self.head_dim

        q_proj = self.q_proj(q)                                            # [B, Sq, d_model]
        k = self.k_proj(kv).view(B, Skv, H, D).transpose(1, 2)            # [B, H, Skv, D]
        v = self.v_proj(kv).view(B, Skv, H, D).transpose(1, 2)            # [B, H, Skv, D]
        q_h = q_proj.view(B, Sq, H, D).transpose(1, 2)                    # [B, H, Sq, D]

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_h, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )                                                                   # [B, H, Sq, D]
        attn_out = attn_out.transpose(1, 2).reshape(B, Sq, self.d_model)   # [B, Sq, d_model]

        out = self.out_proj(attn_out)
        return self.norm(q_proj + out)


class TemporalVGGTv3(nn.Module):
    """View-Conditioned Temporal VGGT V3.

    Fully freezes VGGT and runs it independently on t1 and t3. Fuses the
    time-conditioned outputs via a learnable cross-attention layer. The
    attended features are decoded by DPTHead.

    Args:
        vggt_model_id:          HuggingFace model id or local path for VGGT.
        embed_dim:              Token embedding dimension (1024 for VGGT-1B).
        num_query_views:        Number of t2 query views (Q). Default 1.
        query_patch_h:          Patch-grid height for the t2 query grid.
        query_patch_w:          Patch-grid width  for the t2 query grid.
        query_grid_std:         Std for normal init of the learnable query grids
                                (used when t2_query_mode is "learnable_grid" or
                                "factorized_grid").
        time_conditioning_mode: "film" or "residual_adaln". Controls FiLM vs
                                AdaLN for both t1/t3 post-features and the t2
                                query grid (learnable_grid / factorized_grid only).
        t2_query_mode:          "broadcast" — t2 time embedding expanded to all
                                patch positions (spatially uniform Q).
                                "learnable_grid" — learned per-position query
                                tokens [1, Q, P, C] conditioned on t2 time embedding.
                                "factorized_grid" — factorized form of learnable_grid:
                                query_spatial_grid [1, 1, P, C] + query_view_embed
                                [1, Q, 1, C]; fewer parameters, forces spatial and
                                view-specific patterns to be additive.
        intermediate_layer_idx: Exactly 4 indices into the 24 VGGT intermediate
                                outputs to use. Default [4, 11, 17, 23].
        attn_heads:             Number of cross-attention heads.
        attn_dropout:           Dropout inside the cross-attention layer.
        init_point_head_from_vggt: Copy VGGT's point_head as initialisation.
    """

    def __init__(
        self,
        vggt_model_id: str = "facebook/VGGT-1B",
        embed_dim: int = 1024,
        num_query_views: int = 1,
        query_patch_h: int = 37,
        query_patch_w: int = 37,
        query_grid_std: float = 0.02,
        time_conditioning_mode: str = "film",
        t2_query_mode: str = "learnable_grid",
        intermediate_layer_idx: Optional[List[int]] = None,
        attn_heads: int = 8,
        attn_dropout: float = 0.0,
        init_point_head_from_vggt: bool = False,
        use_camera_head: bool = False,
        init_camera_head_from_vggt: bool = False,
        freeze_point_head: bool = False,
        freeze_camera_head: bool = False,
        camera_conditioning_mode: str = "none",  # "none" | "film" | "residual_adaln" | "additive"
        time_hidden_dim: Optional[int] = None,   # hidden dim for SharedTimeEncoder
        camera_hidden_dim: Optional[int] = None, # hidden dim for camera conditioning
    ):
        super().__init__()

        if intermediate_layer_idx is None:
            intermediate_layer_idx = [4, 11, 17, 23]
        assert len(intermediate_layer_idx) == 4, (
            "intermediate_layer_idx must have exactly 4 elements (DPTHead uses 4 scales)"
        )
        assert t2_query_mode in ("broadcast", "learnable_grid", "factorized_grid"), (
            f"Unknown t2_query_mode: {t2_query_mode!r}"
        )
        assert camera_conditioning_mode in ("none", "film", "residual_adaln", "additive"), (
            f"Unknown camera_conditioning_mode: {camera_conditioning_mode!r}"
        )

        self.embed_dim = embed_dim
        self.feat_dim = 2 * embed_dim  # VGGT intermediate = concat(frame, global)
        self.num_query_views = num_query_views
        self.time_conditioning_mode = time_conditioning_mode
        self.t2_query_mode = t2_query_mode
        self.intermediate_layer_idx = intermediate_layer_idx
        self.camera_conditioning_mode = camera_conditioning_mode

        # --- Load and fully freeze VGGT ---
        logger.info(f"Loading VGGT from {vggt_model_id!r}")
        vggt = VGGT.from_pretrained(vggt_model_id)
        self.aggregator = vggt.aggregator
        self.patch_start_idx = self.aggregator.patch_start_idx  # = 5
        for param in self.aggregator.parameters():
            param.requires_grad_(False)

        # --- Shared time encoder: source_head (2C) for t1/t3; target_head (C) for t2 query ---
        _time_hidden = time_hidden_dim   if time_hidden_dim   is not None else embed_dim
        _cam_hidden  = camera_hidden_dim if camera_hidden_dim is not None else embed_dim

        self.time_encoder = SharedTimeEncoder(
            source_dim=self.feat_dim,
            target_dim=embed_dim,
            hidden_dim=_time_hidden,
        )
        if time_conditioning_mode == "residual_adaln":
            self.source_time_adaln = ResidualAdaLN(self.feat_dim)
            self.target_time_adaln = ResidualAdaLN(embed_dim)
        elif time_conditioning_mode != "film":
            raise ValueError(f"Unknown time_conditioning_mode: {time_conditioning_mode!r}")

        # --- t2 query setup (mode-specific) ---
        P_patch = query_patch_h * query_patch_w
        if t2_query_mode in ("learnable_grid", "factorized_grid"):
            if t2_query_mode == "learnable_grid":
                self.query_grid = nn.Parameter(
                    torch.randn(1, num_query_views, P_patch, embed_dim) * query_grid_std
                )
            else:  # factorized_grid
                self.query_spatial_grid = nn.Parameter(
                    torch.randn(1, 1, P_patch, embed_dim) * query_grid_std
                )
                self.query_view_embed = nn.Parameter(
                    torch.randn(1, num_query_views, 1, embed_dim) * query_grid_std
                )
            q_dim = embed_dim
        else:  # broadcast: gamma_t2_src from source_head → q_dim = feat_dim
            q_dim = self.feat_dim

        # --- Cross-attention (shared across all 4 selected layers) ---
        # Output dim = feat_dim = 2C to match DPTHead's expected dim_in.
        self.cross_attn = CrossAttentionLayer(
            q_dim=q_dim,
            kv_dim=self.feat_dim,
            d_model=self.feat_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
        )

        # --- DPTHead: receives 4 tensors indexed [0, 1, 2, 3] ---
        if init_point_head_from_vggt and vggt.point_head is not None:
            self.point_head = copy.deepcopy(vggt.point_head)
            # VGGT's DPTHead indexes into a 24-entry list; v3 passes exactly 4 tensors
            self.point_head.intermediate_layer_idx = [0, 1, 2, 3]
        else:
            self.point_head = DPTHead(
                dim_in=self.feat_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
                intermediate_layer_idx=[0, 1, 2, 3],
            )

        # --- Camera head (optional) ---
        # use_camera_head gates existence; init_camera_head_from_vggt controls weight source.
        if use_camera_head:
            if init_camera_head_from_vggt:
                if vggt.camera_head is not None:
                    self.camera_head = copy.deepcopy(vggt.camera_head)
                else:
                    logger.warning("init_camera_head_from_vggt=True but VGGT has no camera_head; using fresh init")
                    self.camera_head = CameraHead(dim_in=self.feat_dim)
            else:
                self.camera_head = CameraHead(dim_in=self.feat_dim)
            self.camera_cross_attn = CrossAttentionLayer(
                q_dim=self.feat_dim,
                kv_dim=self.feat_dim,
                d_model=self.feat_dim,
                num_heads=attn_heads,
                dropout=attn_dropout,
            )
        else:
            self.camera_head = None
            self.camera_cross_attn = None

        if freeze_point_head:
            for param in self.point_head.parameters():
                param.requires_grad_(False)
            logger.info("point_head frozen")

        if freeze_camera_head and self.camera_head is not None:
            for param in self.camera_head.parameters():
                param.requires_grad_(False)
            logger.info("camera_head frozen")

        # --- Camera conditioning (optional) ---
        # "film" / "residual_adaln": SharedCameraEncoder → (gamma, beta) applied via FiLM/AdaLN.
        # "additive":                AdditiveCameraConditioner → offset added directly to features.
        # In broadcast mode, target (t2q) conditioning is skipped (no learnable query tensor).
        if camera_conditioning_mode in ("film", "residual_adaln"):
            self.cam_encoder = SharedCameraEncoder(
                source_dim=self.feat_dim,
                target_dim=q_dim,
                hidden_dim=_cam_hidden,
            )
            if camera_conditioning_mode == "residual_adaln":
                self.source_cam_adaln = ResidualAdaLN(self.feat_dim)
                self.target_cam_adaln = ResidualAdaLN(q_dim)
            else:
                self.source_cam_adaln = None
                self.target_cam_adaln = None
        else:
            self.cam_encoder      = None
            self.source_cam_adaln = None
            self.target_cam_adaln = None

        if camera_conditioning_mode == "additive":
            self.camera_cond = AdditiveCameraConditioner(
                hidden_dim=_cam_hidden,
                C=embed_dim,
            )
        else:
            self.camera_cond = None

        del vggt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_source_time_conditioning(
        self,
        x: torch.Tensor,      # [B, ..., feat_dim=2C]
        gamma: torch.Tensor,  # [B, feat_dim]
        beta: torch.Tensor,   # [B, feat_dim]
    ) -> torch.Tensor:
        if self.time_conditioning_mode == "film":
            return apply_film(x, gamma, beta)
        return self.source_time_adaln(x, gamma, beta)

    def _apply_target_time_conditioning(
        self,
        x: torch.Tensor,      # [B, ..., embed_dim=C]
        gamma: torch.Tensor,  # [B, embed_dim]
        beta: torch.Tensor,   # [B, embed_dim]
    ) -> torch.Tensor:
        if self.time_conditioning_mode == "film":
            return apply_film(x, gamma, beta)
        return self.target_time_adaln(x, gamma, beta)

    @staticmethod
    def _cam_dict_to_device(cam: dict, device: torch.device) -> dict:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cam.items()}

    def _apply_source_cam_conditioning(
        self,
        x: torch.Tensor,      # [B, V, P, feat_dim=2C]
        gamma: torch.Tensor,  # [B, V, feat_dim]
        beta: torch.Tensor,   # [B, V, feat_dim]
    ) -> torch.Tensor:
        if self.camera_conditioning_mode == "film":
            return apply_film(x, gamma, beta)
        return self.source_cam_adaln(x, gamma, beta)

    def _apply_target_cam_conditioning(
        self,
        x: torch.Tensor,      # [B, V, P, q_dim=C]
        gamma: torch.Tensor,  # [B, V, q_dim]
        beta: torch.Tensor,   # [B, V, q_dim]
    ) -> torch.Tensor:
        if self.camera_conditioning_mode == "film":
            return apply_film(x, gamma, beta)
        return self.target_cam_adaln(x, gamma, beta)

    def _run_frozen_aggregator(
        self,
        patch_tokens: torch.Tensor,  # [B*S, P_patch, C]
        B: int,
        S: int,
        H_img: int,
        W_img: int,
    ) -> Dict[int, torch.Tensor]:
        """Run aggregator on one timestep's frames; store only the 4 selected layers.

        Returns:
            dict mapping layer_idx → [B, S, P_total, 2C]
        """
        agg = self.aggregator
        patch_h = H_img // agg.patch_size
        patch_w = W_img // agg.patch_size
        _, P_patch, C = patch_tokens.shape

        camera_tok = slice_expand_and_flatten(agg.camera_token, B, S)
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

        frame_idx = 0
        global_idx = 0
        layer_count = 0
        select_set = set(self.intermediate_layer_idx)
        selected: Dict[int, torch.Tensor] = {}

        for _ in range(agg.aa_block_num):
            for attn_type in agg.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_ints = agg._process_frame_attention(
                        tokens, B, S, P_total, C, frame_idx, pos=pos
                    )
                else:
                    tokens, global_idx, global_ints = agg._process_global_attention(
                        tokens, B, S, P_total, C, global_idx, pos=pos
                    )
            for fi, gi in zip(frame_ints, global_ints):
                if layer_count in select_set:
                    selected[layer_count] = torch.cat([fi, gi], dim=-1)  # [B, S, P_total, 2C]
                layer_count += 1

        return selected

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch (dict):
                images_t1:       [B, V1, 3, H, W]
                images_t3:       [B, V3, 3, H, W]
                date_t1:         [B] day-of-year (float)
                date_t2:         [B] day-of-year (float)
                date_t3:         [B] day-of-year (float)
                t1_day:          [B] ordinal day
                t2_day:          [B] ordinal day
                t3_day:          [B] ordinal day
                patch_t1:        [B, V1, P, C]  (optional, precomputed)
                patch_t3:        [B, V3, P, C]  (optional, precomputed)

        Returns:
            dict:
                pred_points: [B, Q, H, W, 3]
                pred_conf:   [B, Q, H, W]
        """
        device = next(self.parameters()).device

        # Extract shape from CPU tensors; avoid moving images to GPU until needed
        images_t1 = batch["images_t1"]
        images_t3 = batch["images_t3"]
        B, V1, _, H, W = images_t1.shape
        V3 = images_t3.shape[1]
        Q = V1  # predict same number of views as context

        # --- Relative gap features ---
        if "rel_gap_feat" in batch:
            rel_gap_feat = batch["rel_gap_feat"].to(device)
        else:
            rel_gap_feat = build_relative_gap_features(
                batch["t1_day"], batch["t2_day"], batch["t3_day"]
            ).to(device)

        date_t1 = batch["date_t1"].to(device)
        date_t2 = batch["date_t2"].to(device)
        date_t3 = batch["date_t3"].to(device)

        # --- Camera conditioning (computed once before the layer loop) ---
        # film/residual_adaln: produces (gamma, beta) pairs for FiLM/AdaLN application.
        # additive:            produces per-view offset vectors broadcast-added to features.
        cam_gamma_t1 = cam_beta_t1 = None
        cam_gamma_t3 = cam_beta_t3 = None
        cam_gamma_t2q = cam_beta_t2q = None
        if self.cam_encoder is not None:
            cam_t1 = self._cam_dict_to_device(batch["camera_t1"], device)
            cam_t3 = self._cam_dict_to_device(batch["camera_t3"], device)
            cam_gamma_t1, cam_beta_t1 = self.cam_encoder.source(cam_t1)  # [B, V1, feat_dim]
            cam_gamma_t3, cam_beta_t3 = self.cam_encoder.source(cam_t3)  # [B, V3, feat_dim]
            if self.t2_query_mode in ("learnable_grid", "factorized_grid"):
                cam_t2q = self._cam_dict_to_device(batch["camera_t2_query"], device)
                cam_gamma_t2q, cam_beta_t2q = self.cam_encoder.target(cam_t2q)  # [B, Q, q_dim]

        cam_add_t1 = cam_add_t3 = cam_add_t2q = None
        if self.camera_cond is not None:
            cam_t1 = self._cam_dict_to_device(batch["camera_t1"], device)
            cam_t3 = self._cam_dict_to_device(batch["camera_t3"], device)
            cam_add_t1 = self.camera_cond.source(cam_t1)  # [B, V1, 2C]
            cam_add_t3 = self.camera_cond.source(cam_t3)  # [B, V3, 2C]
            if self.t2_query_mode in ("learnable_grid", "factorized_grid"):
                cam_t2q = self._cam_dict_to_device(batch["camera_t2_query"], device)
                cam_add_t2q = self.camera_cond.target(cam_t2q)  # [B, Q, C]

        # --- Patch embed (frozen) ---
        agg = self.aggregator
        if "patch_t1" in batch and "patch_t3" in batch:
            patch_t1 = batch["patch_t1"].to(device)
            patch_t3 = batch["patch_t3"].to(device)
            del images_t1, images_t3  # not needed in precomputed mode
        else:
            images_t1 = images_t1.to(device)
            images_t3 = images_t3.to(device)
            mean = agg._resnet_mean
            std  = agg._resnet_std
            imgs_t1_flat = ((images_t1 - mean) / std).view(B * V1, 3, H, W)
            imgs_t3_flat = ((images_t3 - mean) / std).view(B * V3, 3, H, W)
            del images_t1, images_t3
            with torch.no_grad():
                raw_t1 = agg.patch_embed(imgs_t1_flat)
                raw_t3 = agg.patch_embed(imgs_t3_flat)
            if isinstance(raw_t1, dict):
                raw_t1 = raw_t1["x_norm_patchtokens"]
            if isinstance(raw_t3, dict):
                raw_t3 = raw_t3["x_norm_patchtokens"]
            patch_t1 = raw_t1.view(B, V1, raw_t1.shape[1], raw_t1.shape[2])
            patch_t3 = raw_t3.view(B, V3, raw_t1.shape[1], raw_t1.shape[2])

        P_patch = patch_t1.shape[2]
        C       = patch_t1.shape[3]

        # --- Run frozen aggregator independently on t1 and t3 ---
        with torch.no_grad():
            out_t1 = self._run_frozen_aggregator(
                patch_t1.view(B * V1, P_patch, C), B, V1, H, W
            )  # dict: layer_idx → [B, V1, P_total, 2C]
            out_t3 = self._run_frozen_aggregator(
                patch_t3.view(B * V3, P_patch, C), B, V3, H, W
            )  # dict: layer_idx → [B, V3, P_total, 2C]
        del patch_t1, patch_t3

        P_total = next(iter(out_t1.values())).shape[2]

        # --- Time embeddings via SharedTimeEncoder ---
        role_t1 = torch.zeros(B, dtype=torch.long, device=device)
        role_t3 = torch.ones(B, dtype=torch.long, device=device)
        role_t2 = torch.full((B,), 2, dtype=torch.long, device=device)

        # source_head → 2C gamma/beta for t1/t3 feature conditioning
        gamma_t1, beta_t1 = self.time_encoder.source(role_t1, date_t1, rel_gap_feat)
        gamma_t3, beta_t3 = self.time_encoder.source(role_t3, date_t3, rel_gap_feat)

        # --- Build Q depending on t2_query_mode ---
        if self.t2_query_mode in ("learnable_grid", "factorized_grid"):
            # both() encodes once: source_head → 2C (for camera head Q), target_head → C (for query grid)
            (gamma_t2_src, _), (gamma_t2, beta_t2) = self.time_encoder.both(
                role_t2, date_t2, rel_gap_feat
            )
            if self.t2_query_mode == "learnable_grid":
                patch_t2q = self.query_grid.expand(B, Q, P_patch, C)
            else:  # factorized_grid
                patch_t2q = (
                    self.query_spatial_grid + self.query_view_embed[:, :Q]
                ).expand(B, Q, P_patch, C)
            if cam_gamma_t2q is not None:
                patch_t2q = self._apply_target_cam_conditioning(
                    patch_t2q, cam_gamma_t2q, cam_beta_t2q,
                )
            if cam_add_t2q is not None:
                patch_t2q = patch_t2q + cam_add_t2q[:, :, None, :]  # [B,Q,1,C] → [B,Q,P,C]
            patch_t2q = self._apply_target_time_conditioning(patch_t2q, gamma_t2, beta_t2)
            q = patch_t2q.reshape(B, Q * P_patch, C)                  # [B, Q*P_patch, C]
        else:  # broadcast: source_head gives 2C, used directly as q
            gamma_t2_src, _ = self.time_encoder.source(role_t2, date_t2, rel_gap_feat)
            q = gamma_t2_src.unsqueeze(1).expand(B, Q * P_patch, self.feat_dim)  # [B, Q*P_patch, 2C]

        # --- Cross-attention per selected layer → build DPTHead input list ---
        # One batched MHA call per layer: Q×P_patch queries together.
        # This is 16× cheaper in backward memory than a per-view loop because
        # the k/v projections of kv [43968, 2C] are saved once instead of Q times.
        # Flash attention (SDPA) keeps peak memory O(n), not O(n²).
        last_li = self.intermediate_layer_idx[-1]
        raw_cam_t1: Optional[torch.Tensor] = None
        raw_cam_t3: Optional[torch.Tensor] = None

        dpt_input: List[torch.Tensor] = []
        for layer_i in self.intermediate_layer_idx:
            layer_feats_t1 = out_t1.pop(layer_i)
            layer_feats_t3 = out_t3.pop(layer_i)

            # Save raw camera tokens (position 0) from the last selected layer for
            # the camera head; capture before time-conditioning to avoid a second pass.
            if self.camera_head is not None and layer_i == last_li:
                raw_cam_t1 = layer_feats_t1[:, :, 0, :].clone()  # [B, V1, 2C]
                raw_cam_t3 = layer_feats_t3[:, :, 0, :].clone()  # [B, V3, 2C]

            if cam_gamma_t1 is not None:
                layer_feats_t1 = self._apply_source_cam_conditioning(
                    layer_feats_t1, cam_gamma_t1, cam_beta_t1,
                )
                layer_feats_t3 = self._apply_source_cam_conditioning(
                    layer_feats_t3, cam_gamma_t3, cam_beta_t3,
                )
            if cam_add_t1 is not None:
                layer_feats_t1 = layer_feats_t1 + cam_add_t1[:, :, None, :]  # [B,V1,1,2C] → [B,V1,P,2C]
                layer_feats_t3 = layer_feats_t3 + cam_add_t3[:, :, None, :]  # [B,V3,1,2C] → [B,V3,P,2C]
            feat_t1 = self._apply_source_time_conditioning(
                layer_feats_t1, gamma_t1, beta_t1
            )
            feat_t3 = self._apply_source_time_conditioning(
                layer_feats_t3, gamma_t3, beta_t3
            )
            kv = torch.cat([
                feat_t1.reshape(B, V1 * P_total, self.feat_dim),
                feat_t3.reshape(B, V3 * P_total, self.feat_dim),
            ], dim=1)
            del feat_t1, feat_t3
            # q: [B, Q*P_patch, q_dim] → [B, Q*P_patch, feat_dim]
            attn_out = self.cross_attn(q, kv).view(B, Q, P_patch, self.feat_dim)
            del kv
            pad = attn_out.new_zeros(B, Q, self.patch_start_idx, self.feat_dim)
            dpt_input.append(torch.cat([pad, attn_out], dim=2))  # [B, Q, P_total, 2C]
            del attn_out, pad

        # --- Decode with DPTHead ---
        mock_images = torch.zeros(B, Q, 3, H, W, device=device)
        pred_points, pred_conf = self.point_head(
            dpt_input, mock_images, self.patch_start_idx, frames_chunk_size=1,
        )
        # DPTHead already returns [B, Q, H, W, 3] and [B, Q, H, W]

        result: Dict[str, torch.Tensor] = {
            "pred_points": pred_points,
            "pred_conf": pred_conf,
        }

        # --- Camera head (optional) ---
        # Cross-attend from t2 time embedding to t1+t3 camera tokens to produce
        # a predicted t2 camera token, then decode with CameraHead.
        if self.camera_head is not None and raw_cam_t1 is not None:
            # Time-condition the saved camera tokens using the same conditioning as patches.
            cam_t1_feat = self._apply_source_time_conditioning(
                raw_cam_t1.unsqueeze(2), gamma_t1, beta_t1,
            ).squeeze(2)  # [B, V1, 2C]
            cam_t3_feat = self._apply_source_time_conditioning(
                raw_cam_t3.unsqueeze(2), gamma_t3, beta_t3,
            ).squeeze(2)  # [B, V3, 2C]
            kv_cam = torch.cat([cam_t1_feat, cam_t3_feat], dim=1)  # [B, V1+V3, 2C]

            # Q: t2 source time embedding (2C), one per output query view.
            q_cam = gamma_t2_src.unsqueeze(1).expand(B, Q, self.feat_dim)  # [B, Q, 2C]
            pred_cam_tokens = self.camera_cross_attn(q_cam, kv_cam)         # [B, Q, 2C]

            # CameraHead expects [..., P_total, 2C] with camera token at position 0.
            cam_holder = pred_cam_tokens.new_zeros(B, Q, P_total, self.feat_dim)
            cam_holder[:, :, 0, :] = pred_cam_tokens
            pred_pose_enc_list = self.camera_head([cam_holder])
            result["pred_extrinsic"] = pred_pose_enc_list[-1]          # [B, Q, 9]  absT_quaR_FoV
            result["pred_pose_enc_list_t2q"] = pred_pose_enc_list       # list of [B, Q, 9] for camera loss

        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def trainable_parameter_groups(self) -> List[Dict]:
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params}]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

