"""TRPM-Small-Cam-Depth: Camera-Aware TRPM with Depth Prediction.

Extends TRPMSmallCam to additionally predict a depth map in the target t2
camera frame. The depth can be unprojected using t2 intrinsics to get a
cloud in the same space as the depth-based ground truth from build_geometry_assets.

Prediction:
    P2_cam_hat = P1_cam + tau * G * delta_P       (point map in camera frame)
    D2_hat     = depth head output                 (depth per pixel in camera frame)

At evaluation time, D2_hat can be unprojected via t2 intrinsics to produce
a cloud directly comparable to depth-based ground truth.

Inputs (same as TRPMSmallCam):
    P1_world [B, 3, H, W]      t1 point map in GPS world space
    C1       [B, 1, H, W]      t1 confidence
    P3_world [B, V3, 3, H, W]  all t3 views in GPS world space
    C3       [B, V3, 1, H, W]  all t3 confidences
    T2_c2w   [B, 4, 4]         target t2 camera-to-world pose (GPS)
    T1_c2w   [B, 4, 4]         t1 camera-to-world pose (GPS)
    T3_c2w   [B, V3, 4, 4]     t3 camera-to-world poses (GPS)
    K2       [B, 3, 3]         t2 intrinsics scaled to point-map resolution
    K3       [B, V3, 3, 3]     t3 intrinsics
    tau      [B, 1]

Outputs:
    P2_cam_hat  [B, 3, H, W]  predicted t2 point map in target camera frame
    D2_hat      [B, 1, H, W]  predicted t2 depth in target camera frame
    delta_P     [B, 3, H, W]
    G           [B, 1, H, W]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trpm.trpm_small_cam import (
    TRPMSmallCam,
    _make_grid,
    world_to_cam,
    compute_ray_map,
)


class TRPMSmallCamDepth(TRPMSmallCam):
    """Camera-Aware TRPM-Small with Depth Head.

    Inherits all of TRPMSmallCam and adds a depth prediction head.
    The depth head predicts per-pixel depth in the target camera frame,
    which can be unprojected using K2 for evaluation.
    """

    def __init__(
        self,
        conf_threshold: float = 0.02,
        num_t3_points:  int   = 10240,
        cond_dim:       int   = 192,
    ):
        super().__init__(
            conf_threshold=conf_threshold,
            num_t3_points=num_t3_points,
            cond_dim=cond_dim,
        )

        # Depth prediction head (predicts positive depth via softplus)
        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 1, 1),
            nn.Softplus(),  # depth > 0
        )

    def forward(
        self,
        P1_world: torch.Tensor,
        C1:       torch.Tensor,
        P3_world: torch.Tensor,
        C3:       torch.Tensor,
        T2_c2w:   torch.Tensor,
        T1_c2w:   torch.Tensor,
        T3_c2w:   torch.Tensor,
        K2:       torch.Tensor,
        K3:       torch.Tensor,
        tau:      torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        B, _, H, W = P1_world.shape
        device = P1_world.device

        # ── Same as TRPMSmallCam up to decoder output ─────────────────────────
        P1_cam = world_to_cam(P1_world, T2_c2w)
        ray2 = compute_ray_map(K2, H, W)

        M1      = (C1 > self.conf_threshold).float()
        xy      = _make_grid(H, W, str(device)).expand(B, -1, -1, -1)
        tau_map = tau.view(B, 1, 1, 1).expand(B, 1, H, W)

        X1 = torch.cat([P1_cam, C1, M1, xy, ray2, tau_map], dim=1)

        e0 = self.e0(X1)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        Q3 = self._sample_t3_points(P3_world, C3, T2_c2w, T3_c2w, K3)
        z3 = self.pointnet(Q3)

        z_pose = self.pose_enc(T1_c2w, T2_c2w, T3_c2w)
        ztau   = self.time_enc(tau)

        z_cond = self.fusion_mlp(torch.cat([z3, z_pose, ztau], dim=-1))

        d2 = self.d2(e3, e2, z_cond)
        d1 = self.d1(d2, e1, z_cond)
        d0 = self.d0(d1, e0, z_cond)

        # ── Prediction heads ──────────────────────────────────────────────────
        delta_P = self.residual_head(d0)
        G       = self.gate_head(d0)
        P2_cam_hat = P1_cam + tau_map * G * delta_P

        # Depth head
        D2_hat = self.depth_head(d0)  # [B, 1, H, W]

        return {
            "P2_cam_hat": P2_cam_hat,
            "D2_hat":     D2_hat,
            "delta_P":    delta_P,
            "G":          G,
        }

