"""Depth-native losses for TRDM."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from trpm.trdm_geometry import unproject_depth


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if valid.sum() == 0:
        return _zero_like_loss(pred)
    return F.smooth_l1_loss(pred[valid], target[valid], beta=beta)


def log_depth_loss(
    D2_hat: torch.Tensor,
    D2: torch.Tensor,
    C2: torch.Tensor,
    conf_threshold: float,
    beta: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (
        torch.isfinite(D2_hat)
        & torch.isfinite(D2)
        & (D2_hat > eps)
        & (D2 > eps)
        & (C2 > conf_threshold)
    )
    pred_log = torch.log(D2_hat.clamp_min(eps))
    target_log = torch.log(D2.clamp_min(eps))
    return _masked_smooth_l1(pred_log, target_log, valid, beta), pred_log, target_log, valid


def depth_gradient_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    valid: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    valid_x = valid[..., :, 1:] & valid[..., :, :-1]
    valid_y = valid[..., 1:, :] & valid[..., :-1, :]
    pred_gx = pred_log[..., :, 1:] - pred_log[..., :, :-1]
    target_gx = target_log[..., :, 1:] - target_log[..., :, :-1]
    pred_gy = pred_log[..., 1:, :] - pred_log[..., :-1, :]
    target_gy = target_log[..., 1:, :] - target_log[..., :-1, :]
    loss_x = _masked_smooth_l1(pred_gx, target_gx, valid_x, beta)
    loss_y = _masked_smooth_l1(pred_gy, target_gy, valid_y, beta)
    return loss_x + loss_y


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pred, target)
    return dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()


def depth_chamfer_loss(
    D2_hat: torch.Tensor,
    D2: torch.Tensor,
    K2: torch.Tensor,
    valid: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    batch_size, _, height, width = D2.shape
    num_pixels = height * width
    points_pred = unproject_depth(D2_hat, K2).view(batch_size, num_pixels, 3)
    points_tgt = unproject_depth(D2, K2).view(batch_size, num_pixels, 3)
    valid_flat = valid.view(batch_size, num_pixels)
    k_eff = min(num_points, int(valid_flat.sum(dim=1).min().item()))
    if k_eff <= 0:
        return _zero_like_loss(D2_hat)
    scores = torch.where(
        valid_flat,
        torch.rand(batch_size, num_pixels, device=D2.device),
        torch.full((batch_size, num_pixels), -1e9, device=D2.device),
    )
    idx = scores.topk(k_eff, dim=1).indices
    idx_exp = idx.unsqueeze(-1).expand(batch_size, k_eff, 3)
    return chamfer_distance(points_pred.gather(1, idx_exp), points_tgt.gather(1, idx_exp))


def trdm_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict,
) -> dict[str, torch.Tensor]:
    D2_hat = outputs["D2_hat"]
    delta_logD = outputs["delta_logD"]
    gate = outputs["gate"]
    D2 = batch["D2"]
    C2 = batch["C2"]

    eps = cfg.get("eps", 1e-4)
    conf_threshold = cfg.get("conf_threshold", 0.02)
    beta = cfg.get("depth_smooth_l1_beta", 0.05)
    log_weight = cfg.get("log_depth_weight", 1.0)
    grad_weight = cfg.get("depth_gradient_weight", 0.2)
    chamfer_weight = cfg.get("depth_chamfer_weight", 0.01)
    residual_weight = cfg.get("residual_weight", 0.01)
    gate_weight = cfg.get("gate_weight", 0.001)
    chamfer_num_points = cfg.get("chamfer_num_points", 4096)

    loss_log, pred_log, target_log, valid = log_depth_loss(
        D2_hat,
        D2,
        C2,
        conf_threshold,
        beta,
        eps,
    )
    loss_grad = depth_gradient_loss(pred_log, target_log, valid, beta)
    loss_chamfer = _zero_like_loss(D2_hat)
    if chamfer_weight > 0:
        loss_chamfer = depth_chamfer_loss(D2_hat, D2, batch["K2"], valid, chamfer_num_points)
    loss_residual = torch.mean(torch.abs(gate * delta_logD))
    loss_gate = torch.mean(gate)

    total = (
        log_weight * loss_log
        + grad_weight * loss_grad
        + chamfer_weight * loss_chamfer
        + residual_weight * loss_residual
        + gate_weight * loss_gate
    )
    return {
        "loss": total,
        "loss_log_depth": loss_log.detach(),
        "loss_depth_gradient": loss_grad.detach(),
        "loss_depth_chamfer": loss_chamfer.detach(),
        "loss_residual": loss_residual.detach(),
        "loss_gate": loss_gate.detach(),
        "valid_ratio": valid.float().mean().detach(),
        "warp_valid_ratio": outputs["M1_to_t2"].float().mean().detach(),
        "mean_gate": gate.mean().detach(),
        "mean_abs_delta_logD": delta_logD.abs().mean().detach(),
        "mean_D2_hat": D2_hat.mean().detach(),
        "mean_D2": D2.mean().detach(),
    }
