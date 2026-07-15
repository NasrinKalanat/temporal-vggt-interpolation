"""Loss functions for TRPM-Small.

L = L_point + λ_chamfer * L_chamfer + λ_res * L_res + λ_gate * L_gate
"""
import torch
import torch.nn.functional as F


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batched symmetric Chamfer distance.

    Args:
        pred:   [B, N, 3]
        target: [B, N, 3]
    Returns:
        scalar
    """
    diff = pred.unsqueeze(2) - target.unsqueeze(1)   # [B, N, N, 3]
    dist2 = (diff ** 2).sum(-1)                       # [B, N, N]
    return dist2.min(dim=2).values.mean() + dist2.min(dim=1).values.mean()


def trpm_loss(
    P2_hat: torch.Tensor,   # [B, 3, H, W]
    delta_P: torch.Tensor,  # [B, 3, H, W]
    G: torch.Tensor,        # [B, 1, H, W]
    P2: torch.Tensor,       # [B, 3, H, W]  target
    C2: torch.Tensor,       # [B, 1, H, W]  target confidence
    RGB2_hat: torch.Tensor | None = None,  # [B, 3, H, W]
    RGB2: torch.Tensor | None = None,      # [B, 3, H, W]  target RGB
    D2_hat: torch.Tensor | None = None,    # [B, 1, H, W]  predicted depth
    D2: torch.Tensor | None = None,        # [B, 1, H, W]  target depth
    K2: torch.Tensor | None = None,        # [B, 3, 3]     t2 intrinsics (for depth Chamfer)
    conf_threshold: float = 1.5,
    lambda_chamfer: float = 0.05,
    lambda_res: float = 0.01,
    lambda_gate: float = 0.001,
    lambda_rgb: float = 0.05,
    lambda_depth: float = 0.1,
    use_chamfer: bool = True,
    chamfer_num_points: int = 2048,
) -> dict:
    M2 = (C2 > conf_threshold).float()  # [B, 1, H, W]

    # Point-map loss (masked SmoothL1)
    raw_point = F.smooth_l1_loss(P2_hat, P2, reduction="none")  # [B, 3, H, W]
    M2_xyz = M2.expand_as(raw_point)
    l_point = (raw_point * M2_xyz).sum() / M2_xyz.sum().clamp_min(1.0)

    # Residual regularization
    l_res = (G * delta_P).abs().mean()

    # Gate regularization
    l_gate = G.mean()

    total = l_point + lambda_res * l_res + lambda_gate * l_gate

    l_chamfer = torch.zeros(1, device=P2.device)[0]
    if use_chamfer:
        B, _, H, W = P2.shape
        N = H * W
        K = min(chamfer_num_points, N)

        # Gumbel-top-K on mask scores → fixed [B, K] indices, fully batched
        mask_flat = M2.view(B, N) > 0

        K_eff = min(K, int(mask_flat.sum(dim=1).min().item()))

        if K_eff > 0:
            scores = torch.where(
                mask_flat,
                torch.rand(B, N, device=P2.device),
                torch.full((B, N), -1e9, device=P2.device),
            )
            idx = scores.topk(K_eff, dim=1).indices  # [B, K_eff]

            pred_pts = P2_hat.view(B, 3, N).permute(0, 2, 1)
            target_pts = P2.view(B, 3, N).permute(0, 2, 1)

            idx_exp = idx.unsqueeze(-1).expand(B, K_eff, 3)
            pred_pts = pred_pts.gather(1, idx_exp)
            target_pts = target_pts.gather(1, idx_exp)

            l_chamfer = chamfer_distance(pred_pts, target_pts)
            total = total + lambda_chamfer * l_chamfer

    l_rgb = torch.zeros(1, device=P2.device)[0]
    if RGB2_hat is not None and RGB2 is not None:
        raw_rgb = F.l1_loss(RGB2_hat, RGB2, reduction="none")
        M2_rgb = M2.expand_as(raw_rgb)
        l_rgb = (raw_rgb * M2_rgb).sum() / M2_rgb.sum().clamp_min(1.0)
        total = total + lambda_rgb * l_rgb

    l_depth = torch.zeros(1, device=P2.device)[0]
    if D2_hat is not None and D2 is not None:
        raw_depth = F.smooth_l1_loss(D2_hat, D2, reduction="none")
        # Mask: only supervise where target depth is valid (>0) and confidence is high
        D2_valid = (D2 > 0).float() * M2
        l_depth = (raw_depth * D2_valid).sum() / D2_valid.sum().clamp_min(1.0)
        total = total + lambda_depth * l_depth

        # Depth-based Chamfer: unproject both D2_hat and D2 using K2, compare clouds
        if use_chamfer and K2 is not None:
            B, _, H, W = D2.shape
            N = H * W
            fx = K2[:, 0, 0].view(B, 1)  # [B, 1]
            fy = K2[:, 1, 1].view(B, 1)
            cx = K2[:, 0, 2].view(B, 1)
            cy = K2[:, 1, 2].view(B, 1)
            vs = torch.arange(H, device=D2.device, dtype=torch.float32)
            us = torch.arange(W, device=D2.device, dtype=torch.float32)
            grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
            u_flat = grid_u.reshape(1, N).expand(B, -1)  # [B, N]
            v_flat = grid_v.reshape(1, N).expand(B, -1)

            d_pred = D2_hat.view(B, N)
            d_tgt  = D2.view(B, N)
            mask_flat = (D2_valid.view(B, N) > 0)

            K_eff = min(chamfer_num_points, int(mask_flat.sum(dim=1).min().item()))
            if K_eff > 0:
                scores = torch.where(
                    mask_flat,
                    torch.rand(B, N, device=D2.device),
                    torch.full((B, N), -1e9, device=D2.device),
                )
                idx = scores.topk(K_eff, dim=1).indices  # [B, K]

                u_s = u_flat.gather(1, idx)   # [B, K]
                v_s = v_flat.gather(1, idx)
                dp  = d_pred.gather(1, idx)
                dt  = d_tgt.gather(1, idx)

                x_pred = (u_s - cx) / fx * dp
                y_pred = (v_s - cy) / fy * dp
                pts_pred = torch.stack([x_pred, y_pred, dp], dim=-1)  # [B, K, 3]

                x_tgt = (u_s - cx) / fx * dt
                y_tgt = (v_s - cy) / fy * dt
                pts_tgt = torch.stack([x_tgt, y_tgt, dt], dim=-1)    # [B, K, 3]

                l_chamfer = l_chamfer + chamfer_distance(pts_pred, pts_tgt)
                total = total + lambda_chamfer * chamfer_distance(pts_pred, pts_tgt)

    return {
        "loss": total,
        "loss_point": l_point,
        "loss_chamfer": l_chamfer,
        "loss_res": l_res,
        "loss_gate": l_gate,
        "loss_rgb": l_rgb,
        "loss_depth": l_depth,
    }

