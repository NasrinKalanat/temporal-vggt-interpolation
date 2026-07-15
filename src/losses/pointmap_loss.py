"""VGGT paper-style point-map loss for Temporal-VGGT v1.

Loss formula:
    L = mean_M( || C_pred[...,None] ⊙ (P_pred - P_teacher) || )
      + lambda_grad * mean_Mgrad( || C_grad[...,None] ⊙ (∇P_pred - ∇P_teacher) || )
      - alpha * mean_M( log C_pred )

Teacher confidence:
  - Always used to mask low-confidence teacher points (mandatory).
  - Optionally used to softly weight the point regression term only.
  - Never used as a supervision target for C_pred.
"""
import torch
from typing import Dict


def build_teacher_conf_mask(
    teacher_conf: torch.Tensor,   # [B, Q, H, W]
    base_mask: torch.Tensor,      # [B, Q, H, W] bool
    threshold: float,
    threshold_type: str = "quantile",
) -> torch.Tensor:
    """Returns base_mask & (teacher_conf > threshold)."""
    if threshold_type == "absolute":
        conf_mask = teacher_conf > threshold

    elif threshold_type == "quantile":
        valid_conf = teacher_conf[base_mask]
        if valid_conf.numel() == 0:
            return torch.zeros_like(base_mask)
        thr = torch.quantile(valid_conf.detach().float(), threshold)
        conf_mask = teacher_conf > thr

    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type!r}")

    return base_mask & conf_mask


def build_teacher_conf_weight(
    teacher_conf: torch.Tensor,   # [B, Q, H, W]
    mask: torch.Tensor,           # [B, Q, H, W] bool
    enabled: bool = False,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
) -> torch.Tensor:
    """Optional soft weight for the point regression term.

    Returns ones if disabled. When enabled, normalizes teacher confidence by
    the mean over valid pixels and clips to [clip_min, clip_max].
    """
    if not enabled:
        return torch.ones_like(teacher_conf)

    valid_conf = teacher_conf[mask]
    if valid_conf.numel() == 0:
        return torch.ones_like(teacher_conf)

    w = teacher_conf / valid_conf.mean().clamp_min(1e-6)
    return w.clamp(clip_min, clip_max)


def pointmap_gradient_loss(
    pred_points: torch.Tensor,    # [B, Q, H, W, 3]
    teacher_points: torch.Tensor, # [B, Q, H, W, 3]
    pred_conf: torch.Tensor,      # [B, Q, H, W]  (already clamped)
    mask: torch.Tensor,           # [B, Q, H, W] bool
    use_pred_conf: bool = True,
) -> torch.Tensor:
    """Point-map gradient loss using finite differences.

    Confidence for each edge is the average of the two adjacent pixel confidences,
    channel-broadcast over xyz before the norm (matches paper formulation).
    """
    # Horizontal gradients (width axis)
    pred_dx = pred_points[:, :, :, 1:, :] - pred_points[:, :, :, :-1, :]
    gt_dx = teacher_points[:, :, :, 1:, :] - teacher_points[:, :, :, :-1, :]
    mask_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]

    # Vertical gradients (height axis)
    pred_dy = pred_points[:, :, 1:, :, :] - pred_points[:, :, :-1, :, :]
    gt_dy = teacher_points[:, :, 1:, :, :] - teacher_points[:, :, :-1, :, :]
    mask_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]

    res_dx = pred_dx - gt_dx  # [B, Q, H, W-1, 3]
    res_dy = pred_dy - gt_dy  # [B, Q, H-1, W, 3]

    if use_pred_conf:
        conf_dx = 0.5 * (pred_conf[:, :, :, 1:] + pred_conf[:, :, :, :-1])
        conf_dy = 0.5 * (pred_conf[:, :, 1:, :] + pred_conf[:, :, :-1, :])
        res_dx = conf_dx[..., None] * res_dx
        res_dy = conf_dy[..., None] * res_dy

    err_dx = torch.norm(res_dx, dim=-1)  # [B, Q, H, W-1]
    err_dy = torch.norm(res_dy, dim=-1)  # [B, Q, H-1, W]

    n_dx = mask_dx.sum()
    n_dy = mask_dy.sum()

    if n_dx == 0 or n_dy == 0:
        return pred_points.sum() * 0.0

    return 0.5 * (err_dx[mask_dx].mean() + err_dy[mask_dy].mean())


def temporal_vggt_pointmap_loss(
    pred_points: torch.Tensor,    # [B, Q, H, W, 3]
    pred_conf: torch.Tensor,      # [B, Q, H, W]
    teacher_points: torch.Tensor, # [B, Q, H, W, 3]
    teacher_conf: torch.Tensor,   # [B, Q, H, W]
    base_mask: torch.Tensor,      # [B, Q, H, W] bool

    teacher_conf_mask_threshold: float = 0.2,
    teacher_conf_threshold_type: str = "quantile",

    use_teacher_conf_weighted_reg: bool = False,
    teacher_conf_weight_clip_min: float = 0.25,
    teacher_conf_weight_clip_max: float = 4.0,

    alpha: float = 0.2,
    lambda_grad: float = 1.0,
    use_gradient_loss: bool = True,

    pred_conf_clamp_min: float = 1e-6,
    pred_conf_clamp_max: float = 100.0,
) -> Dict[str, torch.Tensor]:
    """Full Temporal-VGGT v1 point-map loss.

    Returns a dict with:
        loss_pointmap:      total loss (scalar)
        loss_point_reg:     weighted point regression term
        loss_point_grad:    gradient consistency term
        loss_point_conf_reg: -alpha * log(C_pred) regularizer
        valid_ratio:        fraction of pixels in final mask
    """
    # 1. Mask low-confidence teacher points (always applied).
    mask = build_teacher_conf_mask(
        teacher_conf=teacher_conf,
        base_mask=base_mask,
        threshold=teacher_conf_mask_threshold,
        threshold_type=teacher_conf_threshold_type,
    )

    if mask.sum() == 0:
        zero = pred_points.sum() * 0.0
        return {
            "loss_pointmap": zero,
            "loss_point_reg": zero,
            "loss_point_grad": zero,
            "loss_point_conf_reg": zero,
            "valid_ratio": mask.float().mean(),
        }

    # 2. Clamp predicted confidence for numerical stability.
    pred_conf_safe = pred_conf.clamp(pred_conf_clamp_min, pred_conf_clamp_max)

    # 3. Optional teacher confidence weights (point regression term only).
    w_teacher = build_teacher_conf_weight(
        teacher_conf=teacher_conf,
        mask=mask,
        enabled=use_teacher_conf_weighted_reg,
        clip_min=teacher_conf_weight_clip_min,
        clip_max=teacher_conf_weight_clip_max,
    )

    # 4. Paper-style channel-broadcast point regression term.
    point_residual = pred_points - teacher_points                         # [B,Q,H,W,3]
    weighted_residual = pred_conf_safe[..., None] * point_residual        # [B,Q,H,W,3]
    point_error = torch.norm(weighted_residual, dim=-1)                   # [B,Q,H,W]

    loss_point_reg = (w_teacher[mask] * point_error[mask]).mean()

    # 5. Predicted-confidence regularizer (-alpha * log C_pred).
    #    Teacher confidence does NOT weight this term.
    loss_point_conf_reg = (-alpha * torch.log(pred_conf_safe[mask])).mean()

    # 6. Optional point-map gradient consistency term.
    if use_gradient_loss:
        loss_grad = pointmap_gradient_loss(
            pred_points=pred_points,
            teacher_points=teacher_points,
            pred_conf=pred_conf_safe,
            mask=mask,
            use_pred_conf=True,
        )
    else:
        loss_grad = pred_points.sum() * 0.0

    loss_total = loss_point_reg + loss_point_conf_reg + lambda_grad * loss_grad

    return {
        "loss_pointmap": loss_total,
        "loss_point_reg": loss_point_reg,
        "loss_point_grad": loss_grad,
        "loss_point_conf_reg": loss_point_conf_reg,
        "valid_ratio": mask.float().mean(),
    }

