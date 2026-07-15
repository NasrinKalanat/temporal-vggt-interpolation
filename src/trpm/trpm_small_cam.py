"""TRPM-Small-Cam: Camera-Aware Temporal Residual Point-Map Network.

All geometry is expressed in the target t2 camera frame (one per view v).
The model receives GPS world-space point maps + camera poses and performs
the coordinate conversion internally.

Inputs (per target view v):
    P1_world [B, 3, H, W]   t1 point map in GPS world space
    C1       [B, 1, H, W]   t1 confidence
    P3_world [B, V3, 3, H, W]  all t3 views in GPS world space
    C3       [B, V3, 1, H, W]  all t3 confidences
    T2_c2w   [B, 4, 4]      target t2 camera-to-world pose (GPS)
    T1_c2w   [B, 4, 4]      t1 camera-to-world pose (GPS)
    T3_c2w   [B, V3, 4, 4]  t3 camera-to-world poses (GPS)
    K2       [B, 3, 3]      t2 intrinsics scaled to point-map resolution
    tau      [B, 1]

Outputs:
    P2_cam_hat  [B, 3, H, W]  predicted t2 point map in target camera frame
    delta_P     [B, 3, H, W]
    G           [B, 1, H, W]

Loss target: P2_world converted to target camera frame using T2_c2w.
"""
from __future__ import annotations

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Cached grid ───────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=4)
def _make_grid(H: int, W: int, device: str) -> torch.Tensor:
    """Normalized [-1, 1] coordinate grid [1, 2, H, W]."""
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]


# ── Coordinate helpers ────────────────────────────────────────────────────────

def world_to_cam(P_world: torch.Tensor, T_c2w: torch.Tensor) -> torch.Tensor:
    """Transform world-space point map into camera frame.

    Args:
        P_world: [B, 3, H, W]  points in GPS world space
        T_c2w:   [B, 4, 4]     camera-to-world pose

    Returns:
        P_cam:   [B, 3, H, W]  points in camera frame
    """
    B, _, H, W = P_world.shape
    R = T_c2w[:, :3, :3]   # [B, 3, 3]  c2w rotation
    c = T_c2w[:, :3,  3]   # [B, 3]     camera center in world

    pts = P_world.view(B, 3, -1)                          # [B, 3, N]
    pts_cam = torch.bmm(R.transpose(1, 2), pts - c.unsqueeze(-1))  # [B, 3, N]
    return pts_cam.view(B, 3, H, W)


def compute_ray_map(K: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Compute normalized ray directions for every pixel using intrinsics K.

    Args:
        K:  [B, 3, 3]  intrinsic matrix (scaled to H×W)
        H, W: output spatial size

    Returns:
        rays: [B, 3, H, W]  unit-length ray directions in camera frame
    """
    B = K.shape[0]
    device = K.device

    # Pixel grid (u, v) at integer centers
    vs = torch.arange(H, dtype=torch.float32, device=device)  # [H]
    us = torch.arange(W, dtype=torch.float32, device=device)  # [W]
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")    # [H, W]

    # [B, H, W]
    u = grid_u.unsqueeze(0).expand(B, -1, -1)
    v = grid_v.unsqueeze(0).expand(B, -1, -1)

    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)

    x = (u - cx) / fx   # [B, H, W]
    y = (v - cy) / fy
    z = torch.ones_like(x)

    rays = torch.stack([x, y, z], dim=1)                      # [B, 3, H, W]
    rays = F.normalize(rays, dim=1)
    return rays


def rotate_rays(rays: torch.Tensor, R_src_c2w: torch.Tensor, R_tgt_c2w: torch.Tensor) -> torch.Tensor:
    """Rotate source camera rays into target camera frame.

    ray_in_target = R_tgt^T * R_src * ray_src_cam

    Args:
        rays:        [B, 3, H, W]  rays in source camera frame
        R_src_c2w:   [B, 3, 3]    source camera-to-world rotation
        R_tgt_c2w:   [B, 3, 3]    target camera-to-world rotation

    Returns:
        [B, 3, H, W]  rays expressed in target camera frame
    """
    B, _, H, W = rays.shape
    R_rel = torch.bmm(R_tgt_c2w.transpose(1, 2), R_src_c2w)  # [B, 3, 3]
    r = rays.view(B, 3, -1)
    r_rot = torch.bmm(R_rel, r)
    return r_rot.view(B, 3, H, W)


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
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
    def __init__(self, cond_dim: int, feat_ch: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_ch)
        self.beta  = nn.Linear(cond_dim, feat_ch)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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
    """Shared MLP + global max-pool over K sampled t3 points.

    Input feature dim = 12:
        xyz (3) + confidence (1) + source uv (2) + source ray in target frame (3)
        + source camera center in target frame (3)
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(12, 64),  nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: [B, K, 12]
        return self.mlp(q).max(dim=1).values  # [B, out_dim]


# ── Time encoder ──────────────────────────────────────────────────────────────

class TimeEncoder(nn.Module):
    def __init__(self, num_freqs: int = 16, out_dim: int = 128):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs).float())
        in_dim = 2 * num_freqs + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(),
            nn.Linear(128, out_dim), nn.SiLU(),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        t = tau * self.freqs.unsqueeze(0)
        feat = torch.cat([tau, torch.sin(t), torch.cos(t)], dim=-1)
        return self.mlp(feat)  # [B, out_dim]


# ── Relative pose encoder ─────────────────────────────────────────────────────

class RelativePoseEncoder(nn.Module):
    """Encode relative poses of t1 and t3 cameras w.r.t. target t2 camera.

    Features (per section 10 of TRPM_fix.md):
        c1_rel          [3]   t1 camera center in target frame
        R1_rel[:, :2]   [6]   first two columns of relative rotation (t1→t2)
        mean(c3_rel)    [3]   mean t3 camera center in target frame
        std(c3_rel)     [3]   std  t3 camera center in target frame
        mean(R3_rel[:,:2]) [6] mean first-two-cols of t3 relative rotations
    Total input: 3 + 6 + 3 + 3 + 6 = 21
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(21, 64),  nn.SiLU(),
            nn.Linear(64, out_dim), nn.SiLU(),
        )

    def forward(
        self,
        T1_c2w:  torch.Tensor,   # [B, 4, 4]
        T2_c2w:  torch.Tensor,   # [B, 4, 4]
        T3_c2w:  torch.Tensor,   # [B, V3, 4, 4]
    ) -> torch.Tensor:
        B  = T2_c2w.shape[0]
        V3 = T3_c2w.shape[1]

        R2 = T2_c2w[:, :3, :3]   # [B, 3, 3]
        c2 = T2_c2w[:, :3,  3]   # [B, 3]
        R1 = T1_c2w[:, :3, :3]
        c1 = T1_c2w[:, :3,  3]

        # t1 relative pose
        R1_rel  = torch.bmm(R2.transpose(1, 2), R1)          # [B, 3, 3]
        c1_rel  = torch.bmm(R2.transpose(1, 2),
                            (c1 - c2).unsqueeze(-1)).squeeze(-1)  # [B, 3]
        R1_cols = R1_rel[:, :, :2].reshape(B, 6)              # [B, 6]

        # t3 relative poses
        R3 = T3_c2w[:, :, :3, :3]   # [B, V3, 3, 3]
        c3 = T3_c2w[:, :, :3,  3]   # [B, V3, 3]

        R2_exp = R2.unsqueeze(1).expand(B, V3, 3, 3)
        c2_exp = c2.unsqueeze(1).expand(B, V3, 3)

        # c3_rel: [B, V3, 3]
        c3_rel = torch.bmm(
            R2_exp.reshape(B * V3, 3, 3).transpose(1, 2),
            (c3 - c2_exp).reshape(B * V3, 3).unsqueeze(-1)
        ).squeeze(-1).view(B, V3, 3)

        # R3_rel first two cols: [B, V3, 3, 2]
        R3_rel = torch.bmm(
            R2_exp.reshape(B * V3, 3, 3).transpose(1, 2),
            R3.reshape(B * V3, 3, 3)
        ).view(B, V3, 3, 3)
        R3_cols = R3_rel[:, :, :, :2].reshape(B, V3, 6)  # [B, V3, 6]

        mean_c3  = c3_rel.mean(dim=1)          # [B, 3]
        std_c3   = c3_rel.std(dim=1)           # [B, 3]
        mean_R3  = R3_cols.mean(dim=1)         # [B, 6]

        feat = torch.cat([c1_rel, R1_cols, mean_c3, std_c3, mean_R3], dim=-1)  # [B, 21]
        return self.mlp(feat)  # [B, out_dim]


# ── Main model ────────────────────────────────────────────────────────────────

class TRPMSmallCam(nn.Module):
    """Camera-Aware TRPM-Small.

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

        # t1 U-Net encoder — 11 input channels (sec. 7)
        # P1_cam(3) + C1(1) + M1(1) + xy(2) + ray2(3) + tau_map(1) = 11
        self.e0 = ConvBlock(11, 32)
        self.e1 = DownBlock(32,  64)
        self.e2 = DownBlock(64,  128)
        self.e3 = DownBlock(128, 192)

        # t3 context: shared PointNet over 12-dim features → 128
        self.pointnet = PointNetEncoder(out_dim=128)

        # Pose and time encoders → 128 each
        self.pose_enc = RelativePoseEncoder(out_dim=128)
        self.time_enc = TimeEncoder(out_dim=128)

        # Fusion: concat(z3=128, z_pose=128, ztau=128) = 384 → cond_dim (sec. 12)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(384, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU(),
        )

        # Decoder (sec. 14)
        self.d2 = UpBlock(192, 128, 128, cond_dim)
        self.d1 = UpBlock(128,  64,  64, cond_dim)
        self.d0 = UpBlock( 64,  32,  32, cond_dim)

        # Prediction heads (sec. 15)
        self.residual_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32,  3, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32,  1, 1),
            nn.Sigmoid(),
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _sample_t3_points(
        self,
        P3_world: torch.Tensor,   # [B, V3, 3, H, W]  GPS world space
        C3:       torch.Tensor,   # [B, V3, 1, H, W]
        T2_c2w:   torch.Tensor,   # [B, 4, 4]
        T3_c2w:   torch.Tensor,   # [B, V3, 4, 4]
        K3:       torch.Tensor,   # [B, V3, 3, 3]
    ) -> torch.Tensor:
        """Sample num_t3_points total from all V3 t3 views, equally per view.

        Each point feature (12-dim, sec. 9):
            xyz in target t2 cam frame   (3)
            confidence                   (1)
            source pixel uv (normalized) (2)
            source ray in target frame   (3)
            source cam center in target  (3)

        Returns: [B, K_total, 12]
        """
        B, V3, _, H, W = P3_world.shape
        K_per_view = max(1, self.num_t3_points // V3)
        N = H * W
        device = P3_world.device

        R2 = T2_c2w[:, :3, :3]              # [B, 3, 3]
        c2 = T2_c2w[:, :3,  3]              # [B, 3]
        R3 = T3_c2w[:, :, :3, :3]           # [B, V3, 3, 3]
        c3 = T3_c2w[:, :, :3,  3]           # [B, V3, 3]

        # ── Convert all t3 points to target camera frame (vectorized) ─────────
        # P3_world: [B, V3, 3, H, W] → [B*V3, 3, H, W]
        BV3 = B * V3
        T2_exp = T2_c2w.unsqueeze(1).expand(B, V3, 4, 4).reshape(BV3, 4, 4)
        P3_cam = world_to_cam(P3_world.reshape(BV3, 3, H, W), T2_exp)  # [B*V3, 3, H, W]
        P3_cam = P3_cam.view(B, V3, 3, H, W)

        # ── Source camera centers in target frame (vectorized) ────────────────
        R2_exp = R2.unsqueeze(1).expand(B, V3, 3, 3).reshape(BV3, 3, 3)
        c2_exp = c2.unsqueeze(1).expand(B, V3, 3).reshape(BV3, 3)
        c3_flat = c3.reshape(BV3, 3)
        c3_in_tgt = torch.bmm(
            R2_exp.transpose(1, 2), (c3_flat - c2_exp).unsqueeze(-1)
        ).squeeze(-1).view(B, V3, 3)         # [B, V3, 3]

        # ── Source rays rotated into target frame (vectorized) ────────────────
        uv_grid = _make_grid(H, W, str(device))  # [1, 2, H, W]

        # Correct source rays from t3 intrinsics
        K3_flat = K3.reshape(BV3, 3, 3)          # [B*V3, 3, 3]
        ray_src = compute_ray_map(K3_flat, H, W) # [B*V3, 3, H, W]

        R3_flat = R3.reshape(BV3, 3, 3)
        R_rel = torch.bmm(R2_exp.transpose(1, 2), R3_flat)
        ray_tgt = torch.bmm(R_rel, ray_src.view(BV3, 3, N)).view(B, V3, 3, H, W)

        # ── Flatten spatial and sample K_per_view per view ────────────────────
        conf_flat = C3.view(B, V3, N)  # [B, V3, N]
        mask_flat = conf_flat > self.conf_threshold
        P3_flat = P3_cam.view(B, V3, 3, N)
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

        idx3 = idx.unsqueeze(2).expand(B, V3, 3, K_per_view)  # [B, V3, 3, K]
        idx2 = idx.unsqueeze(2).expand(B, V3, 2, K_per_view)
        idx1 = idx.unsqueeze(2)                                # [B, V3, 1, K]

        xyz  = P3_flat.gather(3, idx3).permute(0, 1, 3, 2)    # [B, V3, K, 3]
        conf = conf_flat.unsqueeze(2).gather(3, idx1).permute(0, 1, 3, 2)  # [B, V3, K, 1]
        uv   = uv_flat.gather(3, idx2).permute(0, 1, 3, 2)    # [B, V3, K, 2]
        ray  = ray_flat.gather(3, idx3).permute(0, 1, 3, 2)   # [B, V3, K, 3]
        c3e  = c3_in_tgt.unsqueeze(2).expand(B, V3, K_per_view, 3)  # [B, V3, K, 3]

        feats = torch.cat([xyz, conf, uv, ray, c3e], dim=-1)   # [B, V3, K, 12]
        return feats.reshape(B, V3 * K_per_view, 12)           # [B, K_total, 12]

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        P1_world: torch.Tensor,   # [B, 3, H, W]  t1 in GPS world space
        C1:       torch.Tensor,   # [B, 1, H, W]
        P3_world: torch.Tensor,   # [B, V3, 3, H, W]  all t3 views in GPS world space
        C3:       torch.Tensor,   # [B, V3, 1, H, W]
        T2_c2w:   torch.Tensor,   # [B, 4, 4]  target t2 camera-to-world
        T1_c2w:   torch.Tensor,   # [B, 4, 4]  t1 camera-to-world
        T3_c2w:   torch.Tensor,   # [B, V3, 4, 4]  t3 cameras-to-world
        K2:       torch.Tensor,   # [B, 3, 3]  t2 intrinsics at point-map resolution
        K3:       torch.Tensor,   # [B, 3, 3]
        tau:      torch.Tensor,   # [B, 1]
    ) -> dict[str, torch.Tensor]:

        B, _, H, W = P1_world.shape
        device = P1_world.device

        # ── Step 1: convert P1 to target t2 camera frame (sec. 3) ────────────
        P1_cam = world_to_cam(P1_world, T2_c2w)   # [B, 3, H, W]

        # ── Step 2: compute target ray map from t2 intrinsics (sec. 8) ───────
        ray2 = compute_ray_map(K2, H, W)           # [B, 3, H, W]

        # ── Step 3: build t1 encoder input (sec. 7) ──────────────────────────
        M1      = (C1 > self.conf_threshold).float()
        xy      = _make_grid(H, W, str(device)).expand(B, -1, -1, -1)  # [B, 2, H, W]
        tau_map = tau.view(B, 1, 1, 1).expand(B, 1, H, W)

        # X1: P1_cam(3) + C1(1) + M1(1) + xy(2) + ray2(3) + tau_map(1) = 11
        X1 = torch.cat([P1_cam, C1, M1, xy, ray2, tau_map], dim=1)  # [B, 11, H, W]

        # ── Step 4: U-Net encoder ─────────────────────────────────────────────
        e0 = self.e0(X1)   # [B, 32,  H,   W]
        e1 = self.e1(e0)   # [B, 64,  H/2, W/2]
        e2 = self.e2(e1)   # [B, 128, H/4, W/4]
        e3 = self.e3(e2)   # [B, 192, H/8, W/8]

        # ── Step 5: t3 context (sec. 9) ───────────────────────────────────────
        Q3 = self._sample_t3_points(P3_world, C3, T2_c2w, T3_c2w, K3)  # [B, K_total, 12]
        z3 = self.pointnet(Q3)                                       # [B, 128]

        # ── Step 6: pose + time conditioning (sec. 10, 11, 12) ───────────────
        z_pose = self.pose_enc(T1_c2w, T2_c2w, T3_c2w)  # [B, 128]
        ztau   = self.time_enc(tau)                       # [B, 128]

        z_cond = self.fusion_mlp(torch.cat([z3, z_pose, ztau], dim=-1))  # [B, cond_dim]

        # ── Step 7: decoder (sec. 14) ─────────────────────────────────────────
        d2 = self.d2(e3, e2, z_cond)   # [B, 128, H/4, W/4]
        d1 = self.d1(d2, e1, z_cond)   # [B, 64,  H/2, W/2]
        d0 = self.d0(d1, e0, z_cond)   # [B, 32,  H,   W]

        # ── Step 8: prediction heads (sec. 15) ────────────────────────────────
        delta_P = self.residual_head(d0)   # [B, 3, H, W]
        G       = self.gate_head(d0)       # [B, 1, H, W]

        P2_cam_hat = P1_cam + tau_map * G * delta_P   # in target camera frame

        return {
            "P2_cam_hat": P2_cam_hat,
            "delta_P":    delta_P,
            "G":          G,
        }

