"""Temporal-VGGT v2: v1 + time conditioning at each LoRA-adapted aggregator block.

Architecture overview
---------------------
Identical to v1 except:
  - A BlockTimeEncoder maps rel_gap_feat → (gamma_block, beta_block) once per
    forward pass.
  - Before each LoRA-adapted frame/global block in the aggregator, the token
    sequence [B*S, P, C] is modulated by (gamma_block, beta_block) using the
    same time_conditioning_mode as the patch-level conditioning (FiLM or
    residual AdaLN).
  - All other components (patch-embed, patch-level conditioning, query grid,
    DPT head, camera head) are unchanged from v1.

Trainable parameters (additions over v1)
-----------------------------------------
  block_time_encoder          (rel_gap_feat → gamma_block / beta_block)
  block_adaln_frame           (ResidualAdaLN per LoRA frame block; residual_adaln only)
  block_adaln_global          (ResidualAdaLN per LoRA global block; residual_adaln only)
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.heads.dpt_head import DPTHead
from vggt.heads.camera_head import CameraHead

from models.lora import LoRALinear
from models.time_encoding import (
    TimeEncoder,
    BlockTimeEncoder,
    ResidualAdaLN,
    apply_film,
    build_relative_gap_features,
)
from models.camera_encoding import CameraEmbedding, build_camera_features

logger = logging.getLogger(__name__)


class TemporalVGGTv2(nn.Module):
    """View-Conditioned Temporal VGGT V2.

    Same as V1 with additional block-level time conditioning injected before
    each LoRA-adapted frame and global attention block in the aggregator.

    Args:
        vggt_model_id: HuggingFace model id or local path for VGGT checkpoint.
        embed_dim: Token embedding dimension (1024 for VGGT-1B).
        num_query_views: Number of t2 query views (Q). Default 1.
        query_patch_h: Patch-grid height for the t2 query grid.
        query_patch_w: Patch-grid width  for the t2 query grid.
        query_grid_std: Std for normal init of the learned query grid.
        time_conditioning_mode: "film" or "residual_adaln". Controls both
            patch-level and block-level time conditioning.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha (scaling = alpha / rank).
        lora_dropout: Dropout applied inside LoRA path.
        lora_frame_layers: Indices of frame_blocks to equip with LoRA.
        lora_global_layers: Indices of global_blocks to equip with LoRA.
        init_point_head_from_vggt: If True, copy VGGT's point_head weights.
        use_camera_conditioning: Add camera embedding to input tokens.
        use_camera_head: Train a camera pose prediction head.
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
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.05,
        lora_frame_layers: Optional[List[int]] = None,
        lora_global_layers: Optional[List[int]] = None,
        init_point_head_from_vggt: bool = False,
        use_camera_conditioning: bool = True,
        use_camera_head: bool = False,
    ):
        super().__init__()

        if lora_frame_layers is None:
            lora_frame_layers = [20, 21, 22, 23]
        if lora_global_layers is None:
            lora_global_layers = [16, 17, 18, 19, 20, 21, 22, 23]

        self.embed_dim = embed_dim
        self.num_query_views = num_query_views
        self.time_conditioning_mode = time_conditioning_mode
        self.use_camera_conditioning = use_camera_conditioning
        self.use_camera_head = use_camera_head
        # Store for use in _run_aggregator_blocks
        self.lora_frame_layers = list(lora_frame_layers)
        self.lora_global_layers = list(lora_global_layers)

        # --- Load VGGT and extract aggregator ---
        logger.info(f"Loading VGGT from {vggt_model_id!r}")
        vggt = VGGT.from_pretrained(vggt_model_id)

        self.aggregator = vggt.aggregator
        self.patch_start_idx = self.aggregator.patch_start_idx  # = 5

        for param in self.aggregator.parameters():
            param.requires_grad_(False)

        # --- LoRA adaptation on selected attention blocks ---
        self._apply_lora(lora_frame_layers, lora_global_layers, lora_rank, lora_alpha, lora_dropout)

        # --- Patch-level temporal conditioning (same as v1) ---
        self.time_encoder = TimeEncoder(dim=embed_dim, hidden_dim=embed_dim)
        if time_conditioning_mode == "residual_adaln":
            self.time_adaln = ResidualAdaLN(embed_dim)
        elif time_conditioning_mode != "film":
            raise ValueError(f"Unknown time_conditioning_mode: {time_conditioning_mode!r}")

        # --- Block-level temporal conditioning (new in v2) ---
        self.block_time_encoder = BlockTimeEncoder(gap_dim=5, dim=embed_dim)
        if time_conditioning_mode == "residual_adaln":
            self.block_adaln_frame = nn.ModuleList(
                [ResidualAdaLN(embed_dim) for _ in lora_frame_layers]
            )
            self.block_adaln_global = nn.ModuleList(
                [ResidualAdaLN(embed_dim) for _ in lora_global_layers]
            )

        # --- Camera embedding ---
        self.camera_embedding = CameraEmbedding(cam_dim=13, token_dim=embed_dim) if use_camera_conditioning else None

        # --- Learnable t2 query token grid ---
        P_patch = query_patch_h * query_patch_w
        self.query_grid = nn.Parameter(
            torch.randn(1, num_query_views, P_patch, embed_dim) * query_grid_std
        )

        # --- Point-map prediction head ---
        if init_point_head_from_vggt and vggt.point_head is not None:
            self.point_head = copy.deepcopy(vggt.point_head)
        else:
            self.point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )

        # --- Camera head ---
        if use_camera_head and vggt.camera_head is not None:
            self.camera_head = copy.deepcopy(vggt.camera_head)
            self.aggregator.camera_token.requires_grad_(True)
        else:
            self.camera_head = None

        del vggt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cam_to_device(cam_dict: Dict, device: torch.device) -> Dict:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cam_dict.items()}

    def _apply_lora(
        self,
        frame_layers: List[int],
        global_layers: List[int],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        for idx in frame_layers:
            block = self.aggregator.frame_blocks[idx]
            block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha, dropout)
            block.attn.proj = LoRALinear(block.attn.proj, rank, alpha, dropout)

        for idx in global_layers:
            block = self.aggregator.global_blocks[idx]
            block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha, dropout)
            block.attn.proj = LoRALinear(block.attn.proj, rank, alpha, dropout)

    def _apply_time_conditioning(
        self,
        x: torch.Tensor,       # [B, V, P, C]
        gamma: torch.Tensor,   # [B, C]
        beta: torch.Tensor,    # [B, C]
    ) -> torch.Tensor:
        if self.time_conditioning_mode == "film":
            return apply_film(x, gamma, beta)
        else:
            return self.time_adaln(x, gamma, beta)

    def _apply_block_conditioning(
        self,
        tokens: torch.Tensor,          # [B*S, P_total, C]
        B: int,
        S: int,
        gamma: torch.Tensor,           # [B, C]
        beta: torch.Tensor,            # [B, C]
        adaln: Optional[ResidualAdaLN],
    ) -> torch.Tensor:
        """Apply time conditioning to the mixed token sequence before a LoRA block."""
        P, C = tokens.shape[1], tokens.shape[2]
        x = tokens.view(B, S, P, C)   # [B, S, P, C]
        if adaln is not None:          # residual_adaln
            x = adaln(x, gamma, beta)
        else:                          # film
            x = apply_film(x, gamma, beta)
        return x.view(B * S, P, C)

    def _run_aggregator_blocks(
        self,
        patch_tokens: torch.Tensor,  # [B*S_total, P_patch, C]
        B: int,
        S_total: int,
        H_img: int,
        W_img: int,
        gamma_block: torch.Tensor,   # [B, C]
        beta_block: torch.Tensor,    # [B, C]
    ) -> List[torch.Tensor]:
        """Run aggregator frame/global blocks, injecting block-level time conditioning
        before each LoRA-adapted block.

        Returns:
            output_list: 24 tensors, each [B, S_total, P_total, 2*C]
        """
        agg = self.aggregator
        patch_size = agg.patch_size
        patch_h = H_img // patch_size
        patch_w = W_img // patch_size

        _, P_patch, C = patch_tokens.shape

        # Lookup: block_index → index into block_adaln_* ModuleList
        frame_lora_idx = {layer: i for i, layer in enumerate(self.lora_frame_layers)}
        global_lora_idx = {layer: i for i, layer in enumerate(self.lora_global_layers)}

        # Prepend frozen special tokens
        camera_tok = slice_expand_and_flatten(agg.camera_token, B, S_total)
        register_tok = slice_expand_and_flatten(agg.register_token, B, S_total)
        tokens = torch.cat([camera_tok, register_tok, patch_tokens], dim=1)

        _, P_total, _ = tokens.shape

        # RoPE positions
        pos = None
        if agg.rope is not None:
            pos = agg.position_getter(B * S_total, patch_h, patch_w, device=tokens.device)
            pos = pos + 1
            pos_special = torch.zeros(
                B * S_total, agg.patch_start_idx, 2,
                device=tokens.device, dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)

        frame_idx = 0
        global_idx = 0
        output_list: List[torch.Tensor] = []

        for _ in range(agg.aa_block_num):
            for attn_type in agg.aa_order:
                if attn_type == "frame":
                    # Inject time conditioning before this block if it has LoRA
                    if frame_idx in frame_lora_idx:
                        adaln = (
                            self.block_adaln_frame[frame_lora_idx[frame_idx]]
                            if self.time_conditioning_mode == "residual_adaln"
                            else None
                        )
                        tokens = self._apply_block_conditioning(
                            tokens, B, S_total, gamma_block, beta_block, adaln
                        )
                    tokens, frame_idx, frame_ints = agg._process_frame_attention(
                        tokens, B, S_total, P_total, C, frame_idx, pos=pos
                    )
                else:  # "global"
                    if global_idx in global_lora_idx:
                        adaln = (
                            self.block_adaln_global[global_lora_idx[global_idx]]
                            if self.time_conditioning_mode == "residual_adaln"
                            else None
                        )
                        tokens = self._apply_block_conditioning(
                            tokens, B, S_total, gamma_block, beta_block, adaln
                        )
                    tokens, global_idx, global_ints = agg._process_global_attention(
                        tokens, B, S_total, P_total, C, global_idx, pos=pos
                    )

            for fi, gi in zip(frame_ints, global_ints):
                output_list.append(torch.cat([fi, gi], dim=-1))  # [B, S_total, P_total, 2C]

        return output_list

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
                camera_t1:       dict [B, V1, ...]
                camera_t3:       dict [B, V3, ...]
                camera_t2_query: dict [B, Q, ...]
        Returns:
            dict: pred_points [B, Q, H_out, W_out, 3], pred_conf [B, Q, H_out, W_out]
        """
        images_t1 = batch["images_t1"]
        images_t3 = batch["images_t3"]

        B, V1, _, H, W = images_t1.shape
        V3 = images_t3.shape[1]
        Q = self.num_query_views
        S_total = V1 + V3 + Q
        device = images_t1.device

        # --- Relative gap features ---
        if "rel_gap_feat" in batch:
            rel_gap_feat = batch["rel_gap_feat"]
        else:
            rel_gap_feat = build_relative_gap_features(
                batch["t1_day"], batch["t2_day"], batch["t3_day"]
            ).to(device)

        date_t1 = batch["date_t1"].to(device)
        date_t2 = batch["date_t2"].to(device)
        date_t3 = batch["date_t3"].to(device)
        rel_gap_feat = rel_gap_feat.to(device)

        # --- Patch embed t1 and t3 ---
        agg = self.aggregator
        if "patch_t1" in batch and "patch_t3" in batch:
            patch_t1 = batch["patch_t1"].to(device)
            patch_t3 = batch["patch_t3"].to(device)
        else:
            mean = agg._resnet_mean
            std  = agg._resnet_std
            imgs_t1_flat = ((images_t1 - mean) / std).view(B * V1, 3, H, W)
            imgs_t3_flat = ((images_t3 - mean) / std).view(B * V3, 3, H, W)
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

        # --- Patch-level time conditioning (same as v1) ---
        role_t1 = torch.zeros(B, dtype=torch.long, device=device)
        role_t3 = torch.ones(B, dtype=torch.long, device=device)
        role_t2 = torch.full((B,), 2, dtype=torch.long, device=device)

        gamma_t1, beta_t1 = self.time_encoder(role_t1, date_t1, rel_gap_feat)
        gamma_t3, beta_t3 = self.time_encoder(role_t3, date_t3, rel_gap_feat)
        gamma_t2, beta_t2 = self.time_encoder(role_t2, date_t2, rel_gap_feat)

        patch_t1 = self._apply_time_conditioning(patch_t1, gamma_t1, beta_t1)
        patch_t3 = self._apply_time_conditioning(patch_t3, gamma_t3, beta_t3)

        # --- Camera embedding ---
        if self.use_camera_conditioning:
            cam_feat_t1 = build_camera_features(**self._cam_to_device(batch["camera_t1"], device))
            cam_feat_t3 = build_camera_features(**self._cam_to_device(batch["camera_t3"], device))
            patch_t1 = patch_t1 + self.camera_embedding(cam_feat_t1).unsqueeze(2)
            patch_t3 = patch_t3 + self.camera_embedding(cam_feat_t3).unsqueeze(2)

        # --- t2 query tokens ---
        patch_t2q = self.query_grid.expand(B, Q, P_patch, C)
        patch_t2q = self._apply_time_conditioning(patch_t2q, gamma_t2, beta_t2)

        if self.use_camera_conditioning:
            cam_feat_t2q = build_camera_features(**self._cam_to_device(batch["camera_t2_query"], device))
            patch_t2q = patch_t2q + self.camera_embedding(cam_feat_t2q).unsqueeze(2)

        # --- Block-level time conditioning signal (new in v2) ---
        gamma_block, beta_block = self.block_time_encoder(rel_gap_feat)

        # --- Concatenate and run through aggregator blocks ---
        all_patches = torch.cat([patch_t1, patch_t3, patch_t2q], dim=1)  # [B, S_total, P, C]
        all_patches_flat = all_patches.view(B * S_total, P_patch, C)

        output_list = self._run_aggregator_blocks(
            all_patches_flat, B, S_total, H, W,
            gamma_block, beta_block,
        )

        # --- Select t2 query token outputs ---
        t2q_start = V1 + V3
        t2q_output_list = [out[:, t2q_start:, :, :] for out in output_list]

        # --- Decode with point-map head ---
        mock_images = images_t1.new_zeros(B, Q, 3, H, W)
        pred_points, pred_conf = self.point_head(
            t2q_output_list,
            images=mock_images,
            patch_start_idx=self.patch_start_idx,
        )

        out = {"pred_points": pred_points, "pred_conf": pred_conf}

        if self.camera_head is not None:
            pose_enc_list = self.camera_head(output_list)
            out["pred_pose_enc_list_t2q"] = [p[:, V1 + V3:, :] for p in pose_enc_list]
            out["pred_pose_enc_t2q"] = pose_enc_list[-1][:, V1 + V3:, :]

        return out

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

