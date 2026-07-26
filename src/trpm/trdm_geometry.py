"""Geometry utilities for TRDM depth warping and unprojection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def vggt_extrinsic_to_c2w_np(extrinsic: np.ndarray) -> np.ndarray:
    """Invert VGGT world-to-camera extrinsic [3, 4] to camera-to-world [4, 4]."""
    rotation = extrinsic[:, :3].astype(np.float64)
    translation = extrinsic[:, 3].astype(np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ translation
    return c2w.astype(np.float32)


def vggt_extrinsic_to_c2w_torch(extrinsic: torch.Tensor) -> torch.Tensor:
    """Invert VGGT world-to-camera extrinsic [..., 3, 4] to c2w [..., 4, 4]."""
    rotation = extrinsic[..., :3]
    translation = extrinsic[..., 3]
    rotation_t = rotation.transpose(-1, -2)
    center = -torch.matmul(rotation_t, translation.unsqueeze(-1)).squeeze(-1)
    c2w = torch.zeros(*extrinsic.shape[:-2], 4, 4, device=extrinsic.device, dtype=extrinsic.dtype)
    c2w[..., :3, :3] = rotation_t
    c2w[..., :3, 3] = center
    c2w[..., 3, 3] = 1.0
    return c2w


def scale_intrinsics_for_pad_mode(
    w_orig: float,
    h_orig: float,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    scale = min(out_w / w_orig, out_h / h_orig)
    pad_left = (out_w - w_orig * scale) * 0.5
    pad_top = (out_h - h_orig * scale) * 0.5
    return np.array(
        [
            [fl_x * scale, 0.0, cx * scale + pad_left],
            [0.0, fl_y * scale, cy * scale + pad_top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_dataset_intrinsics(date_dir: Path, height: int, width: int, preprocess_mode: str = "pad") -> np.ndarray:
    """Fallback intrinsics from dataset_cameras.json, scaled to depth-map resolution."""
    cameras = json.loads((date_dir / "dataset_cameras.json").read_text())
    intr = cameras["intrinsics"]
    fl_x = float(intr["fl_x"])
    fl_y = float(intr["fl_y"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    w_orig = float(intr["w"])
    h_orig = float(intr["h"])
    if preprocess_mode == "pad":
        return scale_intrinsics_for_pad_mode(w_orig, h_orig, fl_x, fl_y, cx, cy, width, height)
    return np.array(
        [
            [fl_x * (width / w_orig), 0.0, cx * (width / w_orig)],
            [0.0, fl_y * (height / h_orig), cy * (height / h_orig)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_intrinsics(date_dir: Path, height: int, width: int, preprocess_mode: str = "pad") -> np.ndarray:
    """Load VGGT intrinsics if present, otherwise dataset intrinsics scaled to output size."""
    path = date_dir / "predictions" / "intrinsic.npy"
    if path.exists():
        intr = np.load(path).astype(np.float32)
        if intr.ndim == 2:
            intr = intr[None]
        return intr
    K = load_dataset_intrinsics(date_dir, height, width, preprocess_mode)
    depth = np.load(date_dir / "predictions" / "depth_map.npy", mmap_mode="r")
    return np.stack([K] * depth.shape[0]).astype(np.float32)


def build_xy_grid(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)


def build_ray_map(K: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Build pinhole +Z camera-frame rays [B, 3, H, W]."""
    batch_size = K.shape[0]
    device = K.device
    dtype = K.dtype
    vs = torch.arange(height, device=device, dtype=dtype)
    us = torch.arange(width, device=device, dtype=dtype)
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
    u = grid_u.unsqueeze(0).expand(batch_size, -1, -1)
    v = grid_v.unsqueeze(0).expand(batch_size, -1, -1)
    fx = K[:, 0, 0].view(batch_size, 1, 1)
    fy = K[:, 1, 1].view(batch_size, 1, 1)
    cx = K[:, 0, 2].view(batch_size, 1, 1)
    cy = K[:, 1, 2].view(batch_size, 1, 1)
    x = (u - cx) / fx.clamp_min(1e-6)
    y = (v - cy) / fy.clamp_min(1e-6)
    z = torch.ones_like(x)
    return F.normalize(torch.stack([x, y, z], dim=1), dim=1)


def unproject_depth(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Unproject depth [B,1,H,W] into camera-frame points [B,H,W,3]."""
    batch_size, _, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype
    vs = torch.arange(height, device=device, dtype=dtype)
    us = torch.arange(width, device=device, dtype=dtype)
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
    u = grid_u.view(1, height, width)
    v = grid_v.view(1, height, width)
    d = depth[:, 0]
    fx = K[:, 0, 0].view(batch_size, 1, 1).clamp_min(1e-6)
    fy = K[:, 1, 1].view(batch_size, 1, 1).clamp_min(1e-6)
    cx = K[:, 0, 2].view(batch_size, 1, 1)
    cy = K[:, 1, 2].view(batch_size, 1, 1)
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    return torch.stack([x, y, d], dim=-1)


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Transform point grids [..., 3] using matching/broadcastable [..., 4, 4]."""
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    return torch.matmul(transform, points_h.unsqueeze(-1)).squeeze(-1)[..., :3]


def warp_depth_to_target(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    K_src: torch.Tensor,
    T_src_c2w: torch.Tensor,
    K_tgt: torch.Tensor,
    T_tgt_c2w: torch.Tensor,
    conf_threshold: float = 0.02,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest-neighbor z-buffer warp of source depth into target camera view."""
    batch_size, _, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype
    src_cam = unproject_depth(depth, K_src)
    src_world = transform_points(src_cam, T_src_c2w[:, None, None])
    T_tgt_w2c = torch.linalg.inv(T_tgt_c2w)
    src_tgt = transform_points(src_world, T_tgt_w2c[:, None, None])

    X = src_tgt[..., 0]
    Y = src_tgt[..., 1]
    Z = src_tgt[..., 2]
    fx = K_tgt[:, 0, 0].view(batch_size, 1, 1)
    fy = K_tgt[:, 1, 1].view(batch_size, 1, 1)
    cx = K_tgt[:, 0, 2].view(batch_size, 1, 1)
    cy = K_tgt[:, 1, 2].view(batch_size, 1, 1)
    u = torch.round(fx * X / Z.clamp_min(eps) + cx).long()
    v = torch.round(fy * Y / Z.clamp_min(eps) + cy).long()

    out_depth = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    out_conf = torch.zeros_like(out_depth)
    out_mask = torch.zeros_like(out_depth)
    flat_size = height * width
    inf = torch.tensor(float("inf"), device=device, dtype=dtype)

    for batch_idx in range(batch_size):
        valid = (
            torch.isfinite(Z[batch_idx])
            & (Z[batch_idx] > eps)
            & (u[batch_idx] >= 0)
            & (u[batch_idx] < width)
            & (v[batch_idx] >= 0)
            & (v[batch_idx] < height)
            & (confidence[batch_idx, 0] > conf_threshold)
            & torch.isfinite(depth[batch_idx, 0])
        )
        if not valid.any():
            continue
        target_idx = (v[batch_idx][valid] * width + u[batch_idx][valid]).reshape(-1)
        z_values = Z[batch_idx][valid].reshape(-1)
        conf_values = confidence[batch_idx, 0][valid].reshape(-1)
        z_buffer = torch.full((flat_size,), inf, device=device, dtype=dtype)
        z_buffer.scatter_reduce_(0, target_idx, z_values, reduce="amin", include_self=True)
        selected = z_values <= z_buffer[target_idx] + eps
        if not selected.any():
            continue
        sel_idx = target_idx[selected]
        out_depth[batch_idx, 0].view(-1)[sel_idx] = z_values[selected]
        out_conf[batch_idx, 0].view(-1)[sel_idx] = conf_values[selected]
        out_mask[batch_idx, 0].view(-1)[sel_idx] = 1.0

    return out_depth, out_conf, out_mask


def sample_t3_depth_context(
    D3: torch.Tensor,
    C3: torch.Tensor,
    K3: torch.Tensor,
    T3_c2w: torch.Tensor,
    T2_c2w: torch.Tensor,
    num_samples: int = 4096,
    conf_threshold: float = 0.02,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Sample t3 depth context features [B, K, 13] in target t2 camera frame."""
    batch_size, num_views, _, height, width = D3.shape
    device = D3.device
    dtype = D3.dtype
    T2_w2c = torch.linalg.inv(T2_c2w)
    all_features: list[torch.Tensor] = []

    uv = build_xy_grid(1, height, width, device).view(2, -1).T.to(dtype)
    for batch_idx in range(batch_size):
        features_per_batch = []
        R2 = T2_c2w[batch_idx, :3, :3]
        c2 = T2_c2w[batch_idx, :3, 3]
        for view_idx in range(num_views):
            depth_view = D3[batch_idx : batch_idx + 1, view_idx]
            conf_view = C3[batch_idx : batch_idx + 1, view_idx]
            K_view = K3[batch_idx : batch_idx + 1, view_idx]
            T3_view = T3_c2w[batch_idx : batch_idx + 1, view_idx]
            pts_cam3 = unproject_depth(depth_view, K_view)
            pts_world = transform_points(pts_cam3, T3_view[:, None, None])
            pts_cam2 = transform_points(pts_world, T2_w2c[batch_idx : batch_idx + 1, None, None])

            rays_src = build_ray_map(K_view, height, width).view(1, 3, -1)
            R3 = T3_view[0, :3, :3]
            R_rel = R2.T @ R3
            rays_tgt = (R_rel @ rays_src[0]).T
            rays_tgt = F.normalize(rays_tgt, dim=-1)

            c3 = T3_view[0, :3, 3]
            c3_tgt = (R2.T @ (c3 - c2)).view(1, 3).expand(height * width, 3)
            xyz = pts_cam2.view(-1, 3)
            depth_flat = depth_view.view(-1, 1)
            conf_flat = conf_view.view(-1, 1)
            valid = (
                torch.isfinite(xyz).all(dim=1)
                & torch.isfinite(depth_flat[:, 0])
                & (depth_flat[:, 0] > eps)
                & torch.isfinite(conf_flat[:, 0])
                & (conf_flat[:, 0] > conf_threshold)
                & (xyz[:, 2] > eps)
            )
            if valid.any():
                features = torch.cat(
                    [
                        xyz[valid],
                        depth_flat[valid],
                        conf_flat[valid],
                        uv[valid],
                        rays_tgt[valid],
                        c3_tgt[valid],
                    ],
                    dim=-1,
                )
                features_per_batch.append(features)

        if features_per_batch:
            features_all = torch.cat(features_per_batch, dim=0)
            num_valid = features_all.shape[0]
            if num_valid >= num_samples:
                idx = torch.randperm(num_valid, device=device)[:num_samples]
            else:
                idx = torch.randint(num_valid, (num_samples,), device=device)
            all_features.append(features_all[idx])
        else:
            all_features.append(torch.zeros(num_samples, 13, device=device, dtype=dtype))

    return torch.stack(all_features, dim=0)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    var_src = float(np.mean(np.sum(src_c ** 2, axis=1)))
    cov = (dst_c.T @ src_c) / n
    U, S_vals, Vt = np.linalg.svd(cov)
    sign = float(np.sign(np.linalg.det(U @ Vt)))
    D = np.diag([1.0, 1.0, sign])
    R = (U @ D @ Vt).astype(np.float64)
    scale = float(np.sum(S_vals * np.diag(D)) / var_src) if var_src > 0 else 1.0
    t = (mu_d - scale * R @ mu_s).astype(np.float64)
    return scale, R, t


def gps_alignment(date_dir: Path) -> tuple[float, np.ndarray, np.ndarray] | None:
    ext_path = date_dir / "predictions" / "extrinsic.npy"
    cam_path = date_dir / "dataset_cameras.json"
    if not ext_path.exists() or not cam_path.exists():
        return None
    extrinsics = np.load(ext_path).astype(np.float64)
    frames = json.loads(cam_path.read_text())["frames"]
    n_frames = min(len(frames), extrinsics.shape[0])
    if n_frames < 4:
        return None
    vggt_centers = np.array(
        [-extrinsics[i, :, :3].T @ extrinsics[i, :, 3] for i in range(n_frames)],
        dtype=np.float64,
    )
    dataset_centers = np.array(
        [np.array(frames[i]["transform_matrix"], dtype=np.float64)[:3, 3] for i in range(n_frames)],
        dtype=np.float64,
    )
    return umeyama_similarity(vggt_centers, dataset_centers)


def apply_similarity(points: np.ndarray, alignment: tuple[float, np.ndarray, np.ndarray] | None) -> np.ndarray:
    if alignment is None:
        return points.astype(np.float32)
    scale, rotation, translation = alignment
    return (scale * rotation @ points.astype(np.float64).T + translation[:, None]).T.astype(np.float32)


def unproject_depth_numpy(depth: np.ndarray, K: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    fu, fv = K[0, 0], K[1, 1]
    cu, cv = K[0, 2], K[1, 2]
    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    d = depth.astype(np.float64)
    x = (uu - cu) / max(float(fu), 1e-6) * d
    y = (vv - cv) / max(float(fv), 1e-6) * d
    pts_cam = np.stack([x.ravel(), y.ravel(), d.ravel(), np.ones(height * width)], axis=1)
    return (T_c2w.astype(np.float64) @ pts_cam.T).T[:, :3].astype(np.float32)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
