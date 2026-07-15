"""Point cloud geometry metrics and training losses.

Training loss:
    asymmetric_chamfer(pred, target, alpha, beta)
        = alpha * accuracy + beta * completeness
    where accuracy  = mean NN dist from pred to target
          completeness = mean NN dist from target to pred
    Uses PyTorch3D's chamfer_distance for GPU-efficient NN computation.

Evaluation metrics:
    compute_metrics(pred_points, gt_points, threshold, voxel_size, alpha, beta)
    Returns dict with: accuracy, completeness,
                       asymmetric_chamfer (= alpha*acc + beta*comp),
                       precision, recall, f1, normal_consistency,
                       height_max_error, height_median_error, voxel_iou,
                       pointmap_l1, pointmap_l2 (NaN unless dense aligned maps available)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PyTorch training loss (differentiable)
# ---------------------------------------------------------------------------

_EPS = 1e-8


def asymmetric_chamfer(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> torch.Tensor:
    """Asymmetric Chamfer loss: alpha * accuracy + beta * completeness.

    Uses Euclidean (not squared) NN distances, matching the evaluation metric.

    Args:
        pred:   (N, 3)
        target: (M, 3)
    Returns:
        scalar loss
    """
    from pytorch3d.ops import knn_points as _knn_points
    x = pred.unsqueeze(0)    # (1, N, 3)
    y = target.unsqueeze(0)  # (1, M, 3)
    # knn_points returns squared distances; sqrt gives Euclidean to match eval
    accuracy = torch.sqrt(_knn_points(x, y, K=1).dists[0, :, 0] + _EPS).mean()
    completeness = torch.sqrt(_knn_points(y, x, K=1).dists[0, :, 0] + _EPS).mean()
    return alpha * accuracy + beta * completeness


def point_map_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked L1 loss on dense point maps.

    Args:
        pred:   [B, K, 3, H, W]
        target: [B, K, 3, H, W]
        mask:   [B, K, 1, H, W] float in {0, 1}, or None (all valid)
    Returns:
        scalar loss
    """
    diff = (pred - target).abs()  # [B, K, 3, H, W]
    if mask is None:
        return diff.mean()
    # interpolate mask to pred resolution if needed
    if mask.shape[-2:] != pred.shape[-2:]:
        import torch.nn.functional as F
        B, K = mask.shape[:2]
        mask = F.interpolate(
            mask.view(B * K, 1, *mask.shape[-2:]),
            size=pred.shape[-2:],
            mode="nearest",
        ).view(B, K, 1, *pred.shape[-2:])
    masked = diff * mask  # broadcast over 3 channels
    denom = (mask.sum() * 3).clamp(min=1.0)
    return masked.sum() / denom


# ---------------------------------------------------------------------------
# NumPy evaluation metrics
# ---------------------------------------------------------------------------


def _pca_height_axis(points: np.ndarray) -> np.ndarray:
    """Return the field-normal axis (smallest PCA direction) for height metrics."""
    pts = points.astype(np.float64)
    mu = pts.mean(0)
    pts_c = pts - mu
    _, evecs = np.linalg.eigh((pts_c.T @ pts_c) / max(len(pts_c), 1))
    height_axis = evecs[:, 0].astype(np.float32)
    # Orient so canopy (dense surface) → large h values.
    h = pts.astype(np.float32) @ height_axis
    hist, bin_edges = np.histogram(h, bins=100)
    peak_h = float(0.5 * (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]))
    if peak_h < float((h.min() + h.max()) / 2):
        height_axis = -height_axis
    return height_axis


def _estimate_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    """Estimate per-point normals via PCA on k nearest neighbours."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    normals = np.zeros_like(points)
    for i, neighbours in enumerate(idx):
        patch = points[neighbours] - points[neighbours].mean(0)
        _, evecs = np.linalg.eigh(patch.T @ patch)
        normals[i] = evecs[:, 0]
    return normals


def compute_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.05,
    voxel_size: float = 0.05,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> dict[str, float]:
    """Compute all evaluation metrics between predicted and ground-truth point clouds.

    Both arrays should be (N, 3) float32 in the same coordinate system.

    Returns dict with keys:
        accuracy, completeness, asymmetric_chamfer (= alpha*acc + beta*comp),
        precision, recall, f1, normal_consistency,
        height_max_error, height_median_error, voxel_iou,
        pointmap_l1, pointmap_l2 (NaN — requires dense aligned maps not available from baselines)
    """
    from scipy.spatial import cKDTree

    nan = float("nan")
    if len(pred_points) == 0 or len(gt_points) == 0:
        return dict(accuracy=nan, completeness=nan, asymmetric_chamfer=nan,
                    precision=0.0, recall=0.0, f1=0.0,
                    normal_consistency=nan,
                    height_max_error=nan, height_median_error=nan, voxel_iou=0.0,
                    pointmap_l1=nan, pointmap_l2=nan)

    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)

    pred_to_gt_dist, pred_to_gt_idx = gt_tree.query(pred_points, k=1)
    gt_to_pred_dist, _ = pred_tree.query(gt_points, k=1)

    accuracy = float(np.mean(pred_to_gt_dist))
    completeness = float(np.mean(gt_to_pred_dist))
    asymmetric_chamfer_val = alpha * accuracy + beta * completeness

    precision = float(np.mean(pred_to_gt_dist <= threshold))
    recall = float(np.mean(gt_to_pred_dist <= threshold))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Normal consistency: mean |n_pred · n_gt_matched|
    if len(pred_points) >= 10 and len(gt_points) >= 10:
        pred_normals = _estimate_normals(pred_points)
        gt_normals = _estimate_normals(gt_points)
        matched_gt_normals = gt_normals[pred_to_gt_idx]
        dots = np.abs((pred_normals * matched_gt_normals).sum(axis=1))
        normal_consistency = float(np.mean(dots))
    else:
        normal_consistency = float("nan")

    joint = np.concatenate([pred_points, gt_points], axis=0)
    height_axis = _pca_height_axis(joint)
    pred_h = pred_points @ height_axis
    gt_h = gt_points @ height_axis
    height_max_error = float(abs(np.max(pred_h) - np.max(gt_h)))
    height_median_error = float(abs(np.median(pred_h) - np.median(gt_h)))

    pred_voxels = set(map(tuple, np.unique(np.floor(pred_points / voxel_size).astype(np.int64), axis=0).tolist()))
    gt_voxels = set(map(tuple, np.unique(np.floor(gt_points / voxel_size).astype(np.int64), axis=0).tolist()))
    intersection = len(pred_voxels & gt_voxels)
    union = len(pred_voxels | gt_voxels)
    voxel_iou = float(intersection / union) if union > 0 else 0.0

    return dict(
        accuracy=accuracy,
        completeness=completeness,
        asymmetric_chamfer=asymmetric_chamfer_val,
        precision=precision,
        recall=recall,
        f1=f1,
        normal_consistency=normal_consistency,
        height_max_error=height_max_error,
        height_median_error=height_median_error,
        voxel_iou=voxel_iou,
        pointmap_l1=nan,
        pointmap_l2=nan,
    )

