"""Temporal-VGGT v1: View-Conditioned Temporal VGGT for point-map prediction.

Architecture overview
---------------------
Given multi-view images at endpoint dates t1 and t3, predict the 3D point map
at intermediate date t2 for a specified query camera view.

Forward pass:
  1. Patch-embed t1 and t3 images via frozen VGGT patch_embed.
  2. Apply time conditioning (FiLM or residual AdaLN) to t1 and t3 patch tokens.
  3. Add camera embedding (additive) to t1 and t3 tokens.
  4. Construct learnable t2 query tokens; apply time + camera conditioning.
  5. Concatenate [t1 | t3 | t2_query] tokens.
  6. Prepend frozen VGGT camera/register special tokens.
  7. Run through VGGT aggregator frame_blocks + global_blocks (selected layers
     have LoRA adapters; all other weights are frozen).
  8. Collect the 24 intermediate outputs (same format as original VGGT).
  9. Slice the t2_query frame outputs from each intermediate.
  10. Decode t2_query tokens through a trainable DPT point-map head.
  11. Return predicted point map and confidence.

Trainable parameters
---------------------
  time_encoder, time_adaln (if residual_adaln mode)
  camera_embedding
  query_grid (learned blank t2 query tokens)
  LoRA adapters in selected frame_blocks and global_blocks
  point_head (fresh DPTHead)

Frozen parameters
------------------
  aggregator.patch_embed
  aggregator.camera_token (frozen unless use_camera_head=True)
  aggregator.register_token
  aggregator.frame_blocks and global_blocks (base weights; LoRA delta is trainable)
  aggregator.rope, aggregator.position_getter
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
    ResidualAdaLN,
    apply_film,
    build_relative_gap_features,
)
from models.camera_encoding import CameraEmbedding, build_camera_features

logger = logging.getLogger(__name__)


class TemporalVGGTv1(nn.Module):
    """View-Conditioned Temporal VGGT V1.

    Args:
        vggt_model_id: HuggingFace model id or local path for VGGT checkpoint.
        embed_dim: Token embedding dimension (1024 for VGGT-1B).
        num_query_views: Number of t2 query views (Q). Default 1.
        query_patch_h: Patch-grid height for the t2 query grid (H_img // patch_size).
        query_patch_w: Patch-grid width  for the t2 query grid (W_img // patch_size).
        query_grid_std: Std for normal init of the learned query grid.
        time_conditioning_mode: "film" (default) or "residual_adaln".
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha (scaling = alpha / rank).
        lora_dropout: Dropout applied inside LoRA path.
        lora_frame_layers: Indices of frame_blocks to equip with LoRA.
        lora_global_layers: Indices of global_blocks to equip with LoRA.
        init_point_head_from_vggt: If True, copy VGGT's point_head weights as
            initialization for the new point head.
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

        # --- Load VGGT and extract aggregator ---
        logger.info(f"Loading VGGT from {vggt_model_id!r}")
        vggt = VGGT.from_pretrained(vggt_model_id)

        self.aggregator = vggt.aggregator
        self.patch_start_idx = self.aggregator.patch_start_idx  # = 5 (1+4 register)

        # Freeze the entire aggregator first; LoRA will selectively unfreeze.
        for param in self.aggregator.parameters():
            param.requires_grad_(False)

        # --- LoRA adaptation on selected attention blocks ---
        self._apply_lora(lora_frame_layers, lora_global_layers, lora_rank, lora_alpha, lora_dropout)

        # --- Temporal conditioning ---
        self.time_encoder = TimeEncoder(dim=embed_dim, hidden_dim=embed_dim)
        if time_conditioning_mode == "residual_adaln":
            self.time_adaln = ResidualAdaLN(embed_dim)
        elif time_conditioning_mode != "film":
            raise ValueError(f"Unknown time_conditioning_mode: {time_conditioning_mode!r}")

        # --- Camera embedding ---
        self.camera_embedding = CameraEmbedding(cam_dim=13, token_dim=embed_dim) if use_camera_conditioning else None

        # --- Learnable t2 query token grid ---
        # Shape [1, Q, P_patch, C]; each view gets the same base grid which is
        # then conditioned per-view by time + camera embeddings.
        P_patch = query_patch_h * query_patch_w
        self.query_grid = nn.Parameter(
            torch.randn(1, num_query_views, P_patch, embed_dim) * query_grid_std
        )

        # --- Point-map prediction head (trainable from scratch or VGGT init) ---
        if init_point_head_from_vggt and vggt.point_head is not None:
            self.point_head = copy.deepcopy(vggt.point_head)
        else:
            self.point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )

        # --- Camera head (trainable when use_camera_head=True) ---
        if use_camera_head and vggt.camera_head is not None:
            self.camera_head = copy.deepcopy(vggt.camera_head)
            # Unfreeze the camera token in the aggregator — it is the input
            # to the camera head and should be trained alongside it.
            self.aggregator.camera_token.requires_grad_(True)
        else:
            self.camera_head = None

        del vggt  # free memory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cam_to_device(cam_dict: Dict, device: torch.device) -> Dict:
        """Move all tensor values in a camera dict to device."""
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cam_dict.items()}

    def _apply_lora(
        self,
        frame_layers: List[int],
        global_layers: List[int],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        """Replace qkv and proj in selected blocks with LoRALinear wrappers."""
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

    def _run_aggregator_blocks(
        self,
        patch_tokens: torch.Tensor,  # [B*S_total, P_patch, C]
        B: int,
        S_total: int,
        H_img: int,
        W_img: int,
    ) -> List[torch.Tensor]:
        """Run aggregator frame/global blocks, collecting 24 intermediate outputs.

        Returns:
            output_list: 24 tensors, each [B, S_total, P_total, 2*C]
                where P_total = patch_start_idx + P_patch.
        """
        agg = self.aggregator
        patch_size = agg.patch_size
        patch_h = H_img // patch_size
        patch_w = W_img // patch_size

        _, P_patch, C = patch_tokens.shape
        assert C == self.embed_dim

        # Prepend frozen special tokens (camera + register).
        camera_tok = slice_expand_and_flatten(agg.camera_token, B, S_total)   # [B*S, 1, C]
        register_tok = slice_expand_and_flatten(agg.register_token, B, S_total)  # [B*S, 4, C]
        tokens = torch.cat([camera_tok, register_tok, patch_tokens], dim=1)   # [B*S, P_total, C]

        _, P_total, _ = tokens.shape

        # Build RoPE positions (special tokens get zero position).
        pos = None
        if agg.rope is not None:
            pos = agg.position_getter(B * S_total, patch_h, patch_w, device=tokens.device)
            pos = pos + 1  # shift by 1 so patch positions are 1-indexed
            pos_special = torch.zeros(
                B * S_total, agg.patch_start_idx, 2,
                device=tokens.device, dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)

        frame_idx = 0
        global_idx = 0
        output_list: List[torch.Tensor] = []

        for _ in range(agg.aa_block_num):  # 24 iterations
            for attn_type in agg.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_ints = agg._process_frame_attention(
                        tokens, B, S_total, P_total, C, frame_idx, pos=pos
                    )
                else:  # "global"
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
                images_t1:       [B, V1, 3, H, W]  endpoint images at t1
                images_t3:       [B, V3, 3, H, W]  endpoint images at t3
                date_t1:         [B] day-of-year (float) for t1
                date_t2:         [B] day-of-year (float) for t2 target
                date_t3:         [B] day-of-year (float) for t3
                t1_day:          [B] ordinal day for t1 (for gap features)
                t2_day:          [B] ordinal day for t2
                t3_day:          [B] ordinal day for t3
                camera_t1:       dict with keys transform_matrix, fl_x, fl_y, cx, cy,
                                     img_w, img_h, avg_pos, scale  (shapes [B, V1, ...])
                camera_t3:       dict (same layout, [B, V3, ...])
                camera_t2_query: dict (same layout, [B, Q, ...])

                # Teacher supervision (kept on CPU until needed):
                # teacher_points_t2:   [B, Q, H, W, 3]
                # teacher_conf_t2:     [B, Q, H, W]
                # teacher_valid_mask_t2: [B, Q, H, W]

        Returns:
            dict:
                pred_points: [B, Q, H_out, W_out, 3]
                pred_conf:   [B, Q, H_out, W_out]
        """
        images_t1 = batch["images_t1"]  # [B, V1, 3, H, W]
        images_t3 = batch["images_t3"]  # [B, V3, 3, H, W]

        B, V1, _, H, W = images_t1.shape
        V3 = images_t3.shape[1]
        Q = self.num_query_views
        S_total = V1 + V3 + Q
        device = images_t1.device

        # --- Relative gap features (same for all token groups in this triplet) ---
        if "rel_gap_feat" in batch:
            rel_gap_feat = batch["rel_gap_feat"]  # [B, 5]
        else:
            rel_gap_feat = build_relative_gap_features(
                batch["t1_day"], batch["t2_day"], batch["t3_day"]
            ).to(device)

        date_t1 = batch["date_t1"].to(device)
        date_t2 = batch["date_t2"].to(device)
        date_t3 = batch["date_t3"].to(device)
        rel_gap_feat = rel_gap_feat.to(device)

        # --- Patch embed t1 and t3 ---
        # Use precomputed embeddings from cache when available (skips frozen DINOv2).
        agg = self.aggregator
        if "patch_t1" in batch and "patch_t3" in batch:
            patch_t1 = batch["patch_t1"].to(device)   # [B, V1, P, C]
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

        # --- Time conditioning ---
        role_t1 = torch.zeros(B, dtype=torch.long, device=device)
        role_t3 = torch.ones(B, dtype=torch.long, device=device)
        role_t2 = torch.full((B,), 2, dtype=torch.long, device=device)

        gamma_t1, beta_t1 = self.time_encoder(role_t1, date_t1, rel_gap_feat)
        gamma_t3, beta_t3 = self.time_encoder(role_t3, date_t3, rel_gap_feat)
        gamma_t2, beta_t2 = self.time_encoder(role_t2, date_t2, rel_gap_feat)

        patch_t1 = self._apply_time_conditioning(patch_t1, gamma_t1, beta_t1)
        patch_t3 = self._apply_time_conditioning(patch_t3, gamma_t3, beta_t3)

        # --- Camera embedding for t1 and t3 ---
        if self.use_camera_conditioning:
            cam_feat_t1 = build_camera_features(**self._cam_to_device(batch["camera_t1"], device))
            cam_feat_t3 = build_camera_features(**self._cam_to_device(batch["camera_t3"], device))
            patch_t1 = patch_t1 + self.camera_embedding(cam_feat_t1).unsqueeze(2)
            patch_t3 = patch_t3 + self.camera_embedding(cam_feat_t3).unsqueeze(2)

        # --- t2 query tokens (learnable grid + time + camera conditioning) ---
        patch_t2q = self.query_grid.expand(B, Q, P_patch, C)  # [B, Q, P, C]
        patch_t2q = self._apply_time_conditioning(patch_t2q, gamma_t2, beta_t2)

        if self.use_camera_conditioning:
            cam_feat_t2q = build_camera_features(**self._cam_to_device(batch["camera_t2_query"], device))
            patch_t2q = patch_t2q + self.camera_embedding(cam_feat_t2q).unsqueeze(2)

        # --- Concatenate and run through aggregator blocks ---
        all_patches = torch.cat([patch_t1, patch_t3, patch_t2q], dim=1)  # [B, S_total, P, C]
        all_patches_flat = all_patches.view(B * S_total, P_patch, C)

        output_list = self._run_aggregator_blocks(all_patches_flat, B, S_total, H, W)
        # output_list: 24 × [B, S_total, P_total, 2C]

        # --- Select t2 query token outputs ---
        t2q_start = V1 + V3
        t2q_output_list = [out[:, t2q_start:, :, :] for out in output_list]
        # each: [B, Q, P_total, 2C]

        # --- Decode with point-map head ---
        # DPTHead uses images only for spatial dimensions (H, W).
        mock_images = images_t1.new_zeros(B, Q, 3, H, W)
        pred_points, pred_conf = self.point_head(
            t2q_output_list,
            images=mock_images,
            patch_start_idx=self.patch_start_idx,
        )
        # pred_points: [B, Q, H_out, W_out, 3]
        # pred_conf:   [B, Q, H_out, W_out]

        out = {"pred_points": pred_points, "pred_conf": pred_conf}

        if self.camera_head is not None:
            pose_enc_list = self.camera_head(output_list)
            # Slice all iterations to t2 query frames: list of [B, Q, 9]
            out["pred_pose_enc_list_t2q"] = [p[:, V1 + V3:, :] for p in pose_enc_list]
            out["pred_pose_enc_t2q"] = pose_enc_list[-1][:, V1 + V3:, :]  # [B, Q, 9]

        return out

    # ------------------------------------------------------------------
    # Convenience: parameter groups for the optimizer
    # ------------------------------------------------------------------

    def trainable_parameter_groups(self) -> List[Dict]:
        """Returns parameter groups for AdamW (all trainable params, one group)."""
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params}]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

