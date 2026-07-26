"""VGGT cached-feature supervision loss.

Compares predicted t2 cached layers against frozen VGGT teacher cached layers.
Per-layer: LayerNorm(channel) → SmoothL1 + cosine loss, patch tokens only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


def cached_feature_loss(
    pred_layers: List[torch.Tensor],       # 4 × [B, S, T, D]
    teacher_layers: List[torch.Tensor],    # 4 × [B, S, T, D]
    patch_start_idx: int = 5,
    smooth_l1_beta: float = 0.1,
    cos_weight: float = 0.1,
    patch_only: bool = True,
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute cached feature loss across all 4 layers.

    Returns dict with loss_cache (total), loss_cache_l1, loss_cache_cos.
    """
    n_layers = len(pred_layers)
    if n_layers == 0:
        raise ValueError("cached_feature_loss requires at least one cached layer.")
    if n_layers != len(teacher_layers):
        raise ValueError(
            f"Expected the same number of predicted and teacher layers, got "
            f"{n_layers} and {len(teacher_layers)}."
        )

    for pred, teacher in zip(pred_layers, teacher_layers):
        assert pred.shape == teacher.shape, (
            f"Shape mismatch: pred {pred.shape} vs teacher {teacher.shape}"
        )

    same_shape = all(pred.shape == pred_layers[0].shape for pred in pred_layers)
    if same_shape:
        pred = torch.stack(pred_layers, dim=0)
        teacher = torch.stack(teacher_layers, dim=0).to(device=pred.device, dtype=pred.dtype)
        if patch_only:
            pred = pred[:, :, :, patch_start_idx:]
            teacher = teacher[:, :, :, patch_start_idx:]

        if normalize:
            pred = F.layer_norm(pred, [pred.shape[-1]])
            teacher = F.layer_norm(teacher, [teacher.shape[-1]])

        loss_l1 = F.smooth_l1_loss(pred, teacher, beta=smooth_l1_beta)
        if cos_weight > 0:
            loss_cos = (1.0 - F.cosine_similarity(pred, teacher, dim=-1)).mean()
        else:
            loss_cos = pred.new_tensor(0.0)
    else:
        total_l1 = pred_layers[0].new_tensor(0.0)
        total_cos = pred_layers[0].new_tensor(0.0)

        for pred, teacher in zip(pred_layers, teacher_layers):
            teacher = teacher.to(device=pred.device, dtype=pred.dtype)
            # pred, teacher: [B, S, T, D]
            if patch_only:
                pred = pred[:, :, patch_start_idx:]
                teacher = teacher[:, :, patch_start_idx:]

            if normalize:
                pred = F.layer_norm(pred, [pred.shape[-1]])
                teacher = F.layer_norm(teacher, [teacher.shape[-1]])

            total_l1 = total_l1 + F.smooth_l1_loss(pred, teacher, beta=smooth_l1_beta)

            if cos_weight > 0:
                # Cosine loss: 1 - cos_sim, averaged
                cos_sim = F.cosine_similarity(pred, teacher, dim=-1)
                total_cos = total_cos + (1.0 - cos_sim).mean()

        loss_l1 = total_l1 / n_layers
        loss_cos = total_cos / n_layers
    loss_cache = loss_l1 + cos_weight * loss_cos

    return {
        "loss_cache": loss_cache,
        "loss_cache_l1": loss_l1,
        "loss_cache_cos": loss_cos,
    }
