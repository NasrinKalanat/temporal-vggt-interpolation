"""TRPM-Small-Cam-Color: Camera-Aware TRPM with RGB Color Prediction.

Extends TRPMSmallCam to additionally predict the target t2 RGB image as a
residual over the t1 image:
    RGB2_hat = I1 + tau * delta_RGB

Inputs (same as TRPMSmallCam plus):
    I1       [B, 3, H, W]       t1 RGB image (aligned with point map)
    I3       [B, V3, 3, H, W]   t3 RGB images

Outputs:
    P2_cam_hat  [B, 3, H, W]   predicted t2 point map in target camera frame
    delta_P     [B, 3, H, W]
    G           [B, 1, H, W]
    RGB2_hat    [B, 3, H, W]   predicted t2 RGB image
    delta_RGB   [B, 3, H, W]   residual color prediction
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trpm.trpm_small_cam import (
    _make_grid,
    world_to_cam,
    compute_ray_map,
    ConvBlock,
    DownBlock,
    FiLM,
    UpBlock,
    TimeEncoder,
    RelativePoseEncoder,
)


class PointNetEncoderColor(nn.Module):
    """Shared MLP + global max-pool over K sampled t3 points.

    Input feature dim = 15:
        xyz (3) + confidence (1) + RGB3 (3) + source uv (2)
        + source ray in target frame (3) + source camera center in target frame (3)
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(15, 64),  nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.mlp(q).max(dim=1).values  # [B, out_dim]


class TRPMSmallCamColor(nn.Module):
    """Camera-Aware TRPM-Small with Color Prediction.

    Args:
        conf_threshold: threshold for valid-point masks.
        num_t3_points:  total K points sampled from all t3 views combined.
        cond_dim:       FiLM conditioning vector dimension.
    """

    def __init__(
        self,
        conf_threshold: float = 0.02,
        num_t3_points:  int   = 10240,
        cond_dim:       int   = 192,
    ):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.num_t3_points  = num_t3_points

        # t1 U-Net encoder — 14 input channels
        # P1_cam(3) + C1(1) + M1(1) + I1(3) + xy(2) + ray2(3) + tau_map(1) = 14
        self.e0 = ConvBlock(14, 32)
        self.e1 = DownBlock(32,  64)
        self.e2 = DownBlock(64,  128)
        self.e3 = DownBlock(128, 192)

        # t3 context: shared PointNet over 15-dim features → 128
        self.pointnet = PointNetEncoderColor(out_dim=128)

        # Pose and time encoders → 128 each
        self.pose_enc = RelativePoseEncoder(out_dim=128)
        self.time_enc = TimeEncoder(out_dim=128)

        # Fusion: concat(z3=128, z_pose=128, ztau=128) = 384 → cond_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(384, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU(),
        )

        # Decoder
        self.d2 = UpBlock(192, 128, 128, cond_dim)
        self.d1 = UpBlock(128,  64,  64, cond_dim)
        self.d0 = UpBlock( 64,  32,  32, cond_dim)

        # Geometry prediction heads
        self.residual_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32,  3, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32,  1, 1),
            nn.Sigmoid(),
        )

        # Color prediction head (residual)
        self.color_delta_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),  # delta_RGB in [-1, 1]
        )

    def _sample_t3_points(
        self,
        P3_world: torch.Tensor,   # [B, V3, 3, H, W]
        C3:       torch.Tensor,   # [B, V3, 1, H, W]
        I3:       torch.Tensor,   # [B, V3, 3, H, W]
        T2_c2w:   torch.Tensor,   # [B, 4, 4]
        T3_c2w:   torch.Tensor,   # [B, V3, 4, 4]
        K3:       torch.Tensor,   # [B, V3, 3, 3]
    ) -> torch.Tensor:
        """Sample num_t3_points total from all V3 t3 views.

        Each point feature (15-dim):
            xyz in target t2 cam frame   (3)
            confidence                   (1)
            RGB3                         (3)
            source pixel uv (normalized) (2)
            source ray in target frame   (3)
            source cam center in target  (3)

        Returns: [B, K_total, 15]
        """
        B, V3, _, H, W = P3_world.shape
        K_per_view = max(1, self.num_t3_points // V3)
        N = H * W
        device = P3_world.device

        R2 = T2_c2w[:, :3, :3]
        c2 = T2_c2w[:, :3,  3]
        R3 = T3_c2w[:, :, :3, :3]
        c3 = T3_c2w[:, :, :3,  3]

        # Convert all t3 points to target camera frame
        BV3 = B * V3
        T2_exp = T2_c2w.unsqueeze(1).expand(B, V3, 4, 4).reshape(BV3, 4, 4)
        P3_cam = world_to_cam(P3_world.reshape(BV3, 3, H, W), T2_exp)
        P3_cam = P3_cam.view(B, V3, 3, H, W)

        # Source camera centers in target frame
        R2_exp = R2.unsqueeze(1).expand(B, V3, 3, 3).reshape(BV3, 3, 3)
        c2_exp = c2.unsqueeze(1).expand(B, V3, 3).reshape(BV3, 3)
        c3_flat = c3.reshape(BV3, 3)
        c3_in_tgt = torch.bmm(
            R2_exp.transpose(1, 2), (c3_flat - c2_exp).unsqueeze(-1)
        ).squeeze(-1).view(B, V3, 3)

        # Source rays rotated into target frame
        uv_grid = _make_grid(H, W, str(device))
        K3_flat = K3.reshape(BV3, 3, 3)
        ray_src = compute_ray_map(K3_flat, H, W)
        R3_flat = R3.reshape(BV3, 3, 3)
        R_rel = torch.bmm(R2_exp.transpose(1, 2), R3_flat)
        ray_tgt = torch.bmm(R_rel, ray_src.view(BV3, 3, N)).view(B, V3, 3, H, W)

        # Flatten spatial and sample K_per_view per view
        conf_flat = C3.view(B, V3, N)
        mask_flat = conf_flat > self.conf_threshold
        P3_flat = P3_cam.view(B, V3, 3, N)
        I3_flat = I3.view(B, V3, 3, N)
        ray_flat = ray_tgt.view(B, V3, 3, N)
        uv_flat = uv_grid.view(1, 1, 2, N).expand(B, V3, -1, -1)

        valid_min = int(mask_flat.sum(dim=2).min().item())
        if valid_min <= 0:
            raise RuntimeError("At least one t3 view has no valid points above conf_threshold.")

        K_per_view = min(K_per_view, valid_min)

        noise = -torch.empty_like(conf_flat).exponential_().log()
        scores = torch.where(
            mask_flat,
            conf_flat + noise,
            torch.full_like(conf_flat, -1e9),
        )
        idx = scores.topk(K_per_view, dim=2).indices

        idx3 = idx.unsqueeze(2).expand(B, V3, 3, K_per_view)
        idx2 = idx.unsqueeze(2).expand(B, V3, 2, K_per_view)
        idx1 = idx.unsqueeze(2)

        xyz  = P3_flat.gather(3, idx3).permute(0, 1, 3, 2)    # [B, V3, K, 3]
        conf = conf_flat.unsqueeze(2).gather(3, idx1).permute(0, 1, 3, 2)  # [B, V3, K, 1]
        rgb3 = I3_flat.gather(3, idx3).permute(0, 1, 3, 2)    # [B, V3, K, 3]
        uv   = uv_flat.gather(3, idx2).permute(0, 1, 3, 2)    # [B, V3, K, 2]
        ray  = ray_flat.gather(3, idx3).permute(0, 1, 3, 2)   # [B, V3, K, 3]
        c3e  = c3_in_tgt.unsqueeze(2).expand(B, V3, K_per_view, 3)

        feats = torch.cat([xyz, conf, rgb3, uv, ray, c3e], dim=-1)  # [B, V3, K, 15]
        return feats.reshape(B, V3 * K_per_view, 15)

    def forward(
        self,
        P1_world: torch.Tensor,   # [B, 3, H, W]
        C1:       torch.Tensor,   # [B, 1, H, W]
        I1:       torch.Tensor,   # [B, 3, H, W]
        P3_world: torch.Tensor,   # [B, V3, 3, H, W]
        C3:       torch.Tensor,   # [B, V3, 1, H, W]
        I3:       torch.Tensor,   # [B, V3, 3, H, W]
        T2_c2w:   torch.Tensor,   # [B, 4, 4]
        T1_c2w:   torch.Tensor,   # [B, 4, 4]
        T3_c2w:   torch.Tensor,   # [B, V3, 4, 4]
        K2:       torch.Tensor,   # [B, 3, 3]
        K3:       torch.Tensor,   # [B, V3, 3, 3]
        tau:      torch.Tensor,   # [B, 1]
    ) -> dict[str, torch.Tensor]:

        B, _, H, W = P1_world.shape
        device = P1_world.device

        # Convert P1 to target t2 camera frame
        P1_cam = world_to_cam(P1_world, T2_c2w)

        # Target ray map from t2 intrinsics
        ray2 = compute_ray_map(K2, H, W)

        # Build t1 encoder input (14 channels)
        M1      = (C1 > self.conf_threshold).float()
        xy      = _make_grid(H, W, str(device)).expand(B, -1, -1, -1)
        tau_map = tau.view(B, 1, 1, 1).expand(B, 1, H, W)

        P1_cam = P1_cam * M1
        I1_masked = I1 * M1

        X1 = torch.cat([P1_cam, C1, M1, I1_masked, xy, ray2, tau_map], dim=1)  # [B, 14, H, W]

        # U-Net encoder
        e0 = self.e0(X1)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        # t3 context with RGB
        Q3 = self._sample_t3_points(P3_world, C3, I3, T2_c2w, T3_c2w, K3)
        z3 = self.pointnet(Q3)

        # Pose + time conditioning
        z_pose = self.pose_enc(T1_c2w, T2_c2w, T3_c2w)
        ztau   = self.time_enc(tau)

        z_cond = self.fusion_mlp(torch.cat([z3, z_pose, ztau], dim=-1))

        # Decoder
        d2 = self.d2(e3, e2, z_cond)
        d1 = self.d1(d2, e1, z_cond)
        d0 = self.d0(d1, e0, z_cond)

        # Geometry heads
        delta_P = self.residual_head(d0)
        G       = self.gate_head(d0)
        P2_cam_hat = P1_cam + tau_map * G * delta_P

        # Color head
        delta_RGB = self.color_delta_head(d0)
        RGB2_hat = (I1 + tau_map * delta_RGB).clamp(0.0, 1.0)

        return {
            "P2_cam_hat": P2_cam_hat,
            "delta_P":    delta_P,
            "G":          G,
            "RGB2_hat":   RGB2_hat,
            "delta_RGB":  delta_RGB,
        }

