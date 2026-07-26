"""TRDM: Temporal Residual Depth-Map Network."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trpm.trdm_geometry import (
    build_ray_map,
    build_xy_grid,
    sample_t3_depth_context,
    warp_depth_to_target,
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(x)))
        hidden = F.silu(self.norm2(self.conv2(hidden)))
        return hidden + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)
        self.film = FiLM(cond_dim, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.film(self.block(x), cond)


class PointNetContextEncoder(nn.Module):
    def __init__(self, input_dim: int = 13, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
            nn.SiLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.mlp(points).max(dim=1).values


class PoseEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(15, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        T1_c2w: torch.Tensor,
        T2_c2w: torch.Tensor,
        T3_c2w: torch.Tensor,
    ) -> torch.Tensor:
        R2 = T2_c2w[:, :3, :3]
        c2 = T2_c2w[:, :3, 3]
        R1 = T1_c2w[:, :3, :3]
        c1 = T1_c2w[:, :3, 3]
        R3 = T3_c2w[:, :, :3, :3]
        c3 = T3_c2w[:, :, :3, 3]

        c1_rel = torch.bmm(R2.transpose(1, 2), (c1 - c2).unsqueeze(-1)).squeeze(-1)
        f1_world = R1[:, :, 2]
        f1_rel = torch.bmm(R2.transpose(1, 2), f1_world.unsqueeze(-1)).squeeze(-1)

        batch_size, num_t3 = c3.shape[:2]
        R2_exp = R2[:, None].expand(batch_size, num_t3, 3, 3).reshape(batch_size * num_t3, 3, 3)
        c2_exp = c2[:, None].expand(batch_size, num_t3, 3).reshape(batch_size * num_t3, 3)
        c3_rel = torch.bmm(
            R2_exp.transpose(1, 2),
            (c3.reshape(batch_size * num_t3, 3) - c2_exp).unsqueeze(-1),
        ).squeeze(-1).view(batch_size, num_t3, 3)

        f3_world = R3[:, :, :, 2].reshape(batch_size * num_t3, 3)
        f3_rel = torch.bmm(R2_exp.transpose(1, 2), f3_world.unsqueeze(-1)).squeeze(-1)
        f3_rel = f3_rel.view(batch_size, num_t3, 3)

        pose_feat = torch.cat(
            [
                c1_rel,
                F.normalize(f1_rel, dim=-1),
                c3_rel.mean(dim=1),
                c3_rel.std(dim=1, unbiased=False),
                F.normalize(f3_rel.mean(dim=1), dim=-1),
            ],
            dim=-1,
        )
        return self.mlp(pose_feat)


class TimeEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
            nn.SiLU(),
        )

    def forward(self, time_feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(time_feat)


class TRDM(nn.Module):
    """Camera-aware temporal residual depth-map model."""

    def __init__(
        self,
        input_channels: int = 9,
        base_channels: int = 32,
        context_dim: int = 128,
        pose_dim: int = 128,
        time_dim: int = 128,
        cond_dim: int = 256,
        t3_context_samples: int = 4096,
        conf_threshold: float = 0.02,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.t3_context_samples = t3_context_samples
        self.conf_threshold = conf_threshold
        self.eps = eps
        self.learned_log_depth_prior = nn.Parameter(torch.tensor(0.0))

        self.e0 = ConvBlock(input_channels, base_channels)
        self.e1 = DownBlock(base_channels, base_channels * 2)
        self.e2 = DownBlock(base_channels * 2, base_channels * 4)
        self.e3 = DownBlock(base_channels * 4, base_channels * 6)

        self.pointnet = PointNetContextEncoder(input_dim=13, out_dim=context_dim)
        self.pose_encoder = PoseEncoder(out_dim=pose_dim)
        self.time_encoder = TimeEncoder(out_dim=time_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(context_dim + pose_dim + time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

        self.d2 = UpBlock(base_channels * 6, base_channels * 4, base_channels * 4, cond_dim)
        self.d1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, cond_dim)
        self.d0 = UpBlock(base_channels * 2, base_channels, base_channels, cond_dim)

        self.delta_logD_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.delta_logD_head[-1].weight)
        nn.init.zeros_(self.delta_logD_head[-1].bias)
        nn.init.constant_(self.gate_head[-2].bias, -2.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        D1 = batch["D1"]
        C1 = batch["C1"]
        D3 = batch["D3"]
        C3 = batch["C3"]
        K1 = batch["K1"]
        K2 = batch["K2"]
        K3 = batch["K3"]
        T1_c2w = batch["T1_c2w"]
        T2_c2w = batch["T2_c2w"]
        T3_c2w = batch["T3_c2w"]
        tau = batch["tau"]

        batch_size, _, height, width = D1.shape
        device = D1.device
        time_feat = batch.get("time_feat")
        if time_feat is None:
            time_feat = torch.cat(
                [
                    tau,
                    1.0 - tau,
                    torch.zeros_like(tau),
                    torch.zeros_like(tau),
                    torch.zeros_like(tau),
                ],
                dim=1,
            )

        D1_to_t2, C1_to_t2, M1_to_t2 = warp_depth_to_target(
            D1,
            C1,
            K1,
            T1_c2w,
            K2,
            T2_c2w,
            conf_threshold=self.conf_threshold,
            eps=self.eps,
        )
        ray2 = build_ray_map(K2, height, width)
        xy = build_xy_grid(batch_size, height, width, device).to(D1.dtype)
        tau_map = tau[:, :, None, None].expand(batch_size, 1, height, width)
        log_D1 = torch.where(
            M1_to_t2 > 0,
            torch.log(D1_to_t2.clamp_min(self.eps)),
            torch.zeros_like(D1_to_t2),
        )
        x_in = torch.cat([log_D1, C1_to_t2, M1_to_t2, xy, ray2, tau_map], dim=1)

        Q3 = sample_t3_depth_context(
            D3,
            C3,
            K3,
            T3_c2w,
            T2_c2w,
            num_samples=self.t3_context_samples,
            conf_threshold=self.conf_threshold,
            eps=self.eps,
        )
        z3 = self.pointnet(Q3)
        z_pose = self.pose_encoder(T1_c2w, T2_c2w, T3_c2w)
        z_time = self.time_encoder(time_feat.to(D1))
        z_cond = self.fusion_mlp(torch.cat([z3, z_pose, z_time], dim=1))

        e0 = self.e0(x_in)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(e3, e2, z_cond)
        d1 = self.d1(d2, e1, z_cond)
        d0 = self.d0(d1, e0, z_cond)

        delta_logD = self.delta_logD_head(d0)
        gate = self.gate_head(d0)
        base_logD = torch.where(
            M1_to_t2 > 0,
            torch.log(D1_to_t2.clamp_min(self.eps)),
            self.learned_log_depth_prior.expand_as(D1_to_t2),
        )
        pred_logD = base_logD + tau_map * gate * delta_logD
        D2_hat = torch.exp(pred_logD).clamp_min(self.eps)

        return {
            "D2_hat": D2_hat,
            "pred_logD": pred_logD,
            "delta_logD": delta_logD,
            "gate": gate,
            "G": gate,
            "D1_to_t2": D1_to_t2,
            "C1_to_t2": C1_to_t2,
            "M1_to_t2": M1_to_t2,
            "Q3": Q3,
        }
