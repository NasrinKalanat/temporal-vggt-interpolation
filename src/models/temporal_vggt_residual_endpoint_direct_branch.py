"""Residual endpoint ablation with a direct gated transformer branch.

This variant keeps most of the TemporalResidualEndpoint architecture, but uses
one temporal transformer stack per VGGT cache layer and changes the final
cached-feature update rule from:

    updated = f1 + gate * up_proj(norm(h - q))

to:

    updated = f1 + gate * up_proj(norm(h))

The branch is still normalized before projection back to the VGGT feature
dimension when ``d_model_down_proj`` is enabled. For this direct branch,
``up_proj`` starts near zero and gates start at 1.0, so the model begins close
to the identity ``updated ~= f1`` while gradients can flow through the branch.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from models.temporal_vggt_residual_endpoint import TemporalResidualEndpoint
from models.time_encoding import build_relative_gap_features


class TemporalResidualEndpointDirectBranch(TemporalResidualEndpoint):
    """Direct-branch variant with one transformer stack per cache layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.temporal_transformer_blocks_per_layer = torch.nn.ModuleList([
            copy.deepcopy(self.temporal_transformer_blocks)
            for _ in self.cache_layers
        ])
        self.temporal_transformer_blocks = torch.nn.ModuleList()

        if self.use_down_proj:
            for proj in self.up_projs:
                torch.nn.init.normal_(proj.weight, std=1.0e-4)
                torch.nn.init.zeros_(proj.bias)

        with torch.no_grad():
            for gate in self.gates:
                gate.fill_(1.0)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Same input/output contract as TemporalResidualEndpoint.

        The only intentional behavioral difference is the final update:
        this class adds the direct normalized transformer hidden branch instead
        of subtracting the initial query ``q`` first.
        """
        device = next(self.parameters()).device

        images_t1 = batch["images_t1"]
        images_t3 = batch["images_t3"]
        B = batch["date_t1"].shape[0]
        H = W = 518

        if "rel_gap_feat" in batch:
            rel_gap = batch["rel_gap_feat"].to(device)
        else:
            rel_gap = build_relative_gap_features(
                batch["t1_day"], batch["t2_day"], batch["t3_day"]
            ).to(device)
        date_t1 = batch["date_t1"].to(device)
        date_t2 = batch["date_t2"].to(device)
        date_t3 = batch["date_t3"].to(device)

        cached_t1, cached_t3 = self._run_vggt_endpoints(
            images_t1,
            images_t3,
            B,
            t1_key=batch.get("t1_cache_key"),
            t3_key=batch.get("t3_cache_key"),
        )

        n_t1 = batch["camera_t1"]["transform_matrix"].shape[-3]
        n_t3 = batch["camera_t3"]["transform_matrix"].shape[-3]
        if cached_t1[0].shape[1] > n_t1:
            cached_t1 = [f[:, :n_t1] for f in cached_t1]
        if cached_t3[0].shape[1] > n_t3:
            cached_t3 = [f[:, :n_t3] for f in cached_t3]

        S = cached_t1[0].shape[1]
        S3 = cached_t3[0].shape[1]
        del images_t1, images_t3

        T = cached_t1[0].shape[2]

        if self.use_camera_cond:
            camera_t1 = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch["camera_t1"].items()
            }
            camera_t3 = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch["camera_t3"].items()
            }
            camera_t2 = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch["camera_t2_query"].items()
            }
        else:
            camera_t1 = camera_t3 = camera_t2 = None

        pos = self.position_getter(B, S, T, device)
        updated_cached: List[torch.Tensor] = []

        for idx in range(len(self.cache_layers)):
            f1 = cached_t1.pop(0)
            f3 = cached_t3.pop(0)

            if self.use_down_proj:
                f1_low = self.down_projs[idx](self.down_norms[idx](f1))
                f3_low = self.down_projs[idx](self.down_norms[idx](f3))
            else:
                f1_low = f1
                f3_low = f3

            if self.use_camera_cond:
                f1_low = f1_low + self.camera_mlp(camera_t1).unsqueeze(2)
                f3_low = f3_low + self.camera_mlp(camera_t3).unsqueeze(2)
                f1_low = f1_low + self.relative_camera_mlp(camera_t1, camera_t2).unsqueeze(2)
                f3_low = f3_low + self.relative_camera_mlp(camera_t3, camera_t2).unsqueeze(2)

            f1_cond = self._apply_time_cond(f1_low, date_t1, date_t2, rel_gap)
            f3_cond = self._apply_time_cond(f3_low, date_t3, date_t2, rel_gap)
            del f1_low, f3_low

            q = f1_cond.reshape(B, S * T, self.d_model)
            kv = f3_cond.reshape(B, S3 * T, self.d_model)
            del f1_cond, f3_cond

            h = q
            for block in self.temporal_transformer_blocks_per_layer[idx]:
                if self.use_gradient_checkpoint and self.training:
                    h = grad_checkpoint(block, h, kv, pos, use_reentrant=False)
                else:
                    h = block(h, kv, pos)
            del kv

            h = h.reshape(B, S, T, self.d_model)
            if self.use_down_proj:
                h = self.up_projs[idx](self.up_norms[idx](h))

            updated_cached.append(f1 + self.gates[idx] * h)
            del f1, h

        if getattr(self, "freeze_point_head", False):
            return {"pred_cached_layers": updated_cached}

        mock_images = torch.zeros(
            B,
            S,
            3,
            H,
            W,
            device=device,
            dtype=updated_cached[0].dtype,
        )
        pred_points, pred_conf = self.point_head(
            updated_cached,
            images=mock_images,
            patch_start_idx=self.patch_start_idx,
        )
        return {
            "pred_points": pred_points,
            "pred_conf": pred_conf,
            "pred_cached_layers": updated_cached,
        }
