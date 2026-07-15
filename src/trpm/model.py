"""TRPM-Small: Temporal Residual Point-Map Network.

Predicts the t2 point map as a gated residual over t1:
    P2_hat = P1 + tau * G * ΔP

Inputs:  P1, C1 (t1 point map + confidence), P3, C3 (t3), tau (scalar)
Outputs: P2_hat [B, 3, H, W], ΔP [B, 3, H, W], G [B, 1, H, W]
"""
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


@functools.lru_cache(maxsize=4)
def _make_grid(H: int, W: int, device: str) -> torch.Tensor:
    """Cached normalized [-1,1] coordinate grid [1, 2, H, W]."""
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two conv-GN-SiLU layers with optional residual."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool  = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class FiLM(nn.Module):
    """Feature-wise linear modulation: F' = gamma(z) * F + beta(z)."""

    def __init__(self, cond_dim: int, feat_ch: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_ch)
        self.beta  = nn.Linear(cond_dim, feat_ch)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], z: [B, cond_dim]
        g = self.gamma(z).view(z.shape[0], -1, 1, 1)
        b = self.beta(z).view(z.shape[0], -1, 1, 1)
        return x * (1 + g) + b


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.block = ConvBlock(in_ch + skip_ch, out_ch)
        self.film  = FiLM(cond_dim, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.film(self.block(x), z)


# ── t3 PointNet encoder ───────────────────────────────────────────────────────

class PointNetEncoder(nn.Module):
    """MLP + global max-pool over K sampled t3 points."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 64),  nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, q3: torch.Tensor) -> torch.Tensor:
        # q3: [B, K, 6]
        return self.mlp(q3).max(dim=1).values  # [B, out_dim]


# ── Time encoder ──────────────────────────────────────────────────────────────

class TimeEncoder(nn.Module):
    """Fourier features + MLP for scalar tau."""

    def __init__(self, num_freqs: int = 16, out_dim: int = 256):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs).float())
        in_dim = 2 * num_freqs + 1  # sin + cos + raw
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(),
            nn.Linear(128, out_dim), nn.SiLU(),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: [B, 1]
        t = tau * self.freqs.unsqueeze(0)          # [B, num_freqs]
        feat = torch.cat([tau, torch.sin(t), torch.cos(t)], dim=-1)
        return self.mlp(feat)                       # [B, out_dim]


# ── Main model ────────────────────────────────────────────────────────────────

class TRPMSmall(nn.Module):
    """
    Args:
        conf_threshold: threshold for valid-point masks from confidence maps.
        num_t3_points:  K points sampled from t3 for PointNet encoder.
        cond_dim:       conditioning vector dimension fed to FiLM layers.
    """

    def __init__(
        self,
        conf_threshold: float = 1.5,
        num_t3_points: int = 1024,
        cond_dim: int = 192,
    ):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.num_t3_points  = num_t3_points

        # t1 U-Net encoder
        self.e0 = ConvBlock(8, 32)
        self.e1 = DownBlock(32, 64)
        self.e2 = DownBlock(64, 128)
        self.e3 = DownBlock(128, 192)

        # t3 context + time encoders
        self.pointnet  = PointNetEncoder(out_dim=256)
        self.time_enc  = TimeEncoder(out_dim=256)

        # Fusion → conditioning vector
        self.fusion_mlp = nn.Sequential(
            nn.Linear(512, cond_dim), nn.SiLU(),
        )

        # Decoder
        self.d2 = UpBlock(192, 128, 128, cond_dim)
        self.d1 = UpBlock(128,  64,  64, cond_dim)
        self.d0 = UpBlock( 64,  32,  32, cond_dim)

        # Prediction heads
        self.residual_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32,  3, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32,  1, 1),
            nn.Sigmoid(),
        )

    def _make_xy_grid(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        return _make_grid(H, W, str(device)).expand(B, -1, -1, -1)

    def _sample_t3_points(
        self,
        P3: torch.Tensor,   # [B, 3, H, W]
        C3: torch.Tensor,   # [B, 1, H, W]
        M3: torch.Tensor,   # [B, 1, H, W] bool
    ) -> torch.Tensor:
        """Sample K t3 points with features [x,y,z,conf,u,v]. Returns [B, K, 6].

        Fully batched on GPU: add Gumbel noise to confidence scores and take
        top-K, so valid points are preferred but the op is branchless.
        """
        B, _, H, W = P3.shape
        K = self.num_t3_points
        device = P3.device

        # uv grid: [1, 2, H*W] for easy gather — cached
        uv = _make_grid(H, W, str(device)).view(1, 2, H * W).expand(B, -1, -1)

        # Flatten spatial dims
        conf_flat = C3.view(B, H * W)                    # [B, N]
        mask_flat = M3.view(B, H * W).float()            # [B, N]  1=valid, 0=invalid
        P3_flat   = P3.view(B, 3, H * W)                # [B, 3, N]

        # Gumbel-top-K: score = mask * conf + gumbel_noise → top-K indices
        valid_min = int(mask_flat.sum(dim=1).min().item())
        if valid_min <= 0:
            raise RuntimeError("At least one sample has no valid t3 points above conf_threshold.")

        K = min(K, valid_min)

        noise = -torch.empty_like(conf_flat).exponential_().log()
        scores = torch.where(
            mask_flat > 0,
            conf_flat + noise,
            torch.full_like(conf_flat, -1e9),
        )
        idx = scores.topk(K, dim=1).indices


        xyz  = P3_flat.gather(2, idx.unsqueeze(1).expand(B, 3, K)).permute(0, 2, 1)  # [B, K, 3]
        conf = conf_flat.gather(1, idx).unsqueeze(-1)                                 # [B, K, 1]
        u    = uv[:, 0].gather(1, idx).unsqueeze(-1)                                  # [B, K, 1]
        v    = uv[:, 1].gather(1, idx).unsqueeze(-1)                                  # [B, K, 1]

        return torch.cat([xyz, conf, u, v], dim=-1)      # [B, K, 6]

    def forward(
        self,
        P1: torch.Tensor,   # [B, 3, H, W]
        C1: torch.Tensor,   # [B, 1, H, W]
        P3: torch.Tensor,   # [B, 3, H, W]
        C3: torch.Tensor,   # [B, 1, H, W]
        tau: torch.Tensor,  # [B, 1]
    ) -> dict:
        B, _, H, W = P1.shape
        device = P1.device

        M1 = (C1 > self.conf_threshold).float()
        M3 = (C3 > self.conf_threshold).bool()

        xy      = self._make_xy_grid(B, H, W, device)
        tau_map = tau.view(B, 1, 1, 1).expand(B, 1, H, W)

        # t1 encoder
        X1 = torch.cat([P1, C1, M1, xy, tau_map], dim=1)  # [B, 8, H, W]
        e0 = self.e0(X1)   # [B, 32,  H,   W]
        e1 = self.e1(e0)   # [B, 64,  H/2, W/2]
        e2 = self.e2(e1)   # [B, 128, H/4, W/4]
        e3 = self.e3(e2)   # [B, 192, H/8, W/8]

        # t3 context
        Q3 = self._sample_t3_points(P3, C3, M3)   # [B, K, 6]
        z3   = self.pointnet(Q3)                   # [B, 256]
        ztau = self.time_enc(tau)                  # [B, 256]

        z_cond = self.fusion_mlp(torch.cat([z3, ztau], dim=-1))  # [B, cond_dim]

        # Decoder
        d2 = self.d2(e3, e2, z_cond)  # [B, 128, H/4, W/4]
        d1 = self.d1(d2, e1, z_cond)  # [B, 64,  H/2, W/2]
        d0 = self.d0(d1, e0, z_cond)  # [B, 32,  H,   W]

        delta_P = self.residual_head(d0)  # [B, 3, H, W]
        G       = self.gate_head(d0)      # [B, 1, H, W]

        P2_hat = P1 + tau_map * G * delta_P

        return {"P2_hat": P2_hat, "delta_P": delta_P, "G": G}

